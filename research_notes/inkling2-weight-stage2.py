#!/usr/bin/env python3
"""Stage 2 on-box: per-expert scale-energy statistics + fast zero-code sampling,
resumable per stage. Imports the verified dequant helpers from
/root/inkling2-weight-read.py so all bit-level logic stays single-sourced.

Usage: inkling2-weight-stage2.py <pack.gguf> <outdir>
Writes expert-blocks.npz incrementally; skips stages whose outputs exist.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/root")
import importlib.util as _ilu                                    # noqa: E402
_spec = _ilu.spec_from_file_location("inkling2_weight_read", "/root/inkling2-weight-read.py")
WR = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(WR)                                       # noqa: E402


def main():
    pack, outdir = sys.argv[1], sys.argv[2]
    ver, kv, tensors, data_start = WR.parse_gguf(pack)
    mm = np.memmap(pack, dtype=np.uint8, mode="r")

    def by_suffix(suffix):
        out = {}
        for t in tensors:
            parts = t["name"].split(".")
            if len(parts) >= 3 and ".".join(parts[2:]) == suffix:
                out[int(parts[1])] = t
        return dict(sorted(out.items()))

    gate_up = by_suffix("ffn_gate_up_exps.weight")
    downs = by_suffix("ffn_down_exps.weight")
    rows_per_expert = gate_up[min(gate_up)]["dims"][1]
    n_exp = gate_up[min(gate_up)]["dims"][2]

    DN_BL = {11: 110, 12: 144, 13: 176}
    CHUNK_ROWS = 8192
    total_rows = rows_per_expert * n_exp

    def geom(tinfo, blen):
        nb_row = tinfo["dims"][0] // 256
        rowbytes = nb_row * blen
        return nb_row, rowbytes

    # ---- stage A: energies --------------------------------------------------
    gu_done = os.path.exists(os.path.join(outdir, "energy_gateup.npy"))
    dn_done = os.path.exists(os.path.join(outdir, "energy_down.npy"))

    if not gu_done:
        e_list, lay_idx = [], []
        for lay in sorted(gate_up):
            tinfo = gate_up[lay]
            nb_row, rowbytes = geom(tinfo, 110)
            base = data_start + tinfo["offset"]
            acc = np.zeros(n_exp, dtype=np.float64)
            done = 0
            while done < total_rows:
                take = min(CHUNK_ROWS, total_rows - done)
                rows = np.asarray(mm[base + done * rowbytes : base + (done + take) * rowbytes],
                                  dtype=np.uint8).reshape(take, rowbytes)
                eb = WR.q3_energy_per_block_public(rows) if hasattr(WR, "q3_energy_per_block_public") \
                    else energy_q3(rows)
                eb = eb.reshape(take, nb_row).sum(axis=1)
                exp_idx = (done + np.arange(take)) // rows_per_expert
                np.add.at(acc, exp_idx, eb)
                done += take
            e_list.append(acc)
            lay_idx.append(int(lay))
            print(f"energy GU L{lay}", flush=True)
        EG_GU = np.stack(e_list)
        np.save(os.path.join(outdir, "energy_gateup.npy"), EG_GU)
        json.dump({"gateup_layers": lay_idx}, open(os.path.join(outdir, "energy-index.json"), "w"))
    else:
        EG_GU = np.load(os.path.join(outdir, "energy_gateup.npy"))
        meta = json.load(open(os.path.join(outdir, "energy-index.json")))
        lay_idx = meta["gateup_layers"]
        print("energy gate/up: loaded", flush=True)

    if not dn_done:
        e_list, dmeta = [], {}
        for lay in sorted(downs):
            tinfo = downs[lay]
            tt = tinfo["type"]
            blen = DN_BL.get(tt)
            if blen is None:
                raise ValueError(f"down type {tt}")
            nb_row, rowbytes = geom(tinfo, blen)
            base = data_start + tinfo["offset"]
            acc = np.zeros(n_exp, dtype=np.float64)
            done = 0
            while done < total_rows:
                take = min(CHUNK_ROWS, total_rows - done)
                rows = np.asarray(mm[base + done * rowbytes : base + (done + take) * rowbytes],
                                  dtype=np.uint8).reshape(take, rowbytes)
                eb = WR.energy_q45(rows, blen) if hasattr(WR, "energy_q45") else energy_q45_loc(rows, blen)
                eb = eb.reshape(take, nb_row).sum(axis=1)
                exp_idx = (done + np.arange(take)) // rows_per_expert
                np.add.at(acc, exp_idx, eb)
                done += take
            e_list.append(acc)
            dmeta[str(lay)] = {"type": int(tt), "blocks_per_row": int(nb_row)}
            print(f"energy DN L{lay}", flush=True)
        EG_DN = np.stack(e_list)
        np.save(os.path.join(outdir, "energy_down.npy"), EG_DN)
        json.dump(dmeta, open(os.path.join(outdir, "down-meta.json"), "w"))
    else:
        EG_DN = np.load(os.path.join(outdir, "energy_down.npy"))
        dmeta = json.load(open(os.path.join(outdir, "down-meta.json")))
        lay_dn = sorted(int(k) for k in dmeta.keys())
        print("energy down: loaded", flush=True)

    # ---- stage B: zero-code fraction, vectorized gather ----------------------
    zf_path = os.path.join(outdir, "zero-frac-progress.json")
    zl = {}
    if os.path.exists(zf_path):
        zl = json.load(open(zf_path))
        print(f"zero-frac resume with {len(zl)} layers", flush=True)
    # L0-2 measured at stride 16; the rest at 64 (sampling error ~0.05pp on the
    # fraction; recorded per layer in the npz meta). Resume keys keep their values.
    stride = 16 if len(zl) < 3 else 64
    for lay in sorted(gate_up):
        key = str(lay)
        if key in zl:
            continue
        tinfo = gate_up[lay]
        nb_row, rowbytes = geom(tinfo, 110)
        base = data_start + tinfo["offset"]
        rows_sel = np.asarray(mm[base : base + total_rows * rowbytes],
                              dtype=np.uint8).reshape(total_rows, rowbytes)[::stride]
        tot = zer = 0.0
        CH = 65536 // nb_row or 1
        buf = np.asarray(rows_sel, dtype=np.uint8)
        nsel = buf.shape[0]
        for s in range(0, nsel, CH):
            blks = buf[s : s + CH].reshape(-1, 110)
            deq = WR.deq_q3_K_vec(blks)
            zer += float((deq == 0).sum())
            tot += deq.size
        zl[key] = zer / max(tot, 1)
        json.dump(zl, open(zf_path, "w"))
        print(f"zero-frac L{lay}: {zl[key]:.4f}  ({len(zl)}/{len(gate_up)})", flush=True)

    zf_arr = np.array([zl[str(l)] for l in sorted(gate_up)])
    np.savez_compressed(
        os.path.join(outdir, "expert-blocks.npz"),
        energy_gateup=EG_GU,
        energy_down=EG_DN,
        gateup_layers=np.array(sorted(gate_up)),
        down_layers=np.array(sorted(int(k) for k in dmeta.keys())),
        zero_code_frac_gateup_sampled=zf_arr,
        sample_stride=np.int64(stride),
        rows_per_expert=np.int64(rows_per_expert),
        experts=np.int64(n_exp),
        down_meta=json.dumps(dmeta),
    )
    print("DONE", flush=True)


def energy_q3(rows_u8):
    """Same as main-file q3_energy_per_block (kept here as fallback copy)."""
    blks = rows_u8.reshape(-1, 110)
    aux = blks[:, 96:108].copy().view("<u4").astype(np.uint64)
    d_all = blks[:, 108:110].copy().view(np.float16).astype(np.float64)[..., 0]
    scb = WR.q3k_shuffle(aux).astype(np.float64)
    return (d_all ** 2) * np.square(scb - 32.0).sum(axis=1)


def energy_q45_loc(rows_u8, blen):
    blks = rows_u8.reshape(-1, blen)
    d = blks[:, 0:2].copy().view(np.float16).astype(np.float64)[..., 0]
    mnv = blks[:, 2:4].copy().view(np.float16).astype(np.float64)[..., 0]
    sb = blks[:, 4:16].astype(np.int64)
    n = blks.shape[0]
    sc = np.empty((n, 8), dtype=np.int64)
    mi = np.empty((n, 8), dtype=np.int64)
    for j in range(8):
        if j < 4:
            sc[:, j] = sb[:, j] & 63; mi[:, j] = sb[:, j + 4] & 63
        else:
            sc[:, j] = (sb[:, j + 4] & 0xF) | ((sb[:, j - 4] >> 6) << 4)
            mi[:, j] = (sb[:, j + 4] >> 4) | ((sb[:, j] >> 6) << 4)
    return (d ** 2) * np.square(sc.astype(np.float64)).sum(axis=1) \
         + (mnv ** 2) * np.square(mi.astype(np.float64)).sum(axis=1)


if __name__ == "__main__":
    main()
