#!/usr/bin/env python3
"""Stage 3 on-box: regenerate ONLY the down-tensor fingerprints inside an
existing fingerprints.npz (the first pass ran with a shifted type table).
Usage: inkling2-weight-stage3.py <pack.gguf> <outdir>
"""
import json
import os
import sys

import numpy as np

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

    downs = by_suffix("ffn_down_exps.weight")
    rows_per_expert = downs[min(downs)]["dims"][1]
    n_exp = downs[min(downs)]["dims"][2]
    DN_BL = {11: 110, 12: 144, 13: 176}

    fp_path = os.path.join(outdir, "fingerprints.npz")
    old = np.load(fp_path, allow_pickle=False)
    rng = np.random.default_rng(7)

    dn_layers_info = []
    FPDN_list = []
    proj_store = {}
    for lay in sorted(downs):
        tinfo = downs[lay]
        tt = tinfo["type"]
        blen = DN_BL.get(tt)
        if blen is None:
            raise ValueError(f"down type {tt}")
        nb_row = tinfo["dims"][0] // 256
        rowbytes = nb_row * blen
        ncols = tinfo["dims"][0]
        sample_rows = [0, 1151, 2303, 3455]
        prj = (rng.standard_normal((ncols, 64)) / np.sqrt(64.0)).astype(np.float32)
        proj_store[str(lay)] = prj
        per_layer = np.empty((n_exp, len(sample_rows), 64), dtype=np.float32)
        base = data_start + tinfo["offset"]
        vecf = WR.deq_q3_K_vec if tt == 11 else (lambda b: WR.deq_q45_K_vec(b, tt == 13))
        for e in range(n_exp):
            rs = [(e * rows_per_expert + rr) for rr in sample_rows]
            lo, hi = min(rs), max(rs) + 1
            sub = np.asarray(mm[base + lo * rowbytes : base + hi * rowbytes],
                             dtype=np.uint8).reshape(hi - lo, rowbytes)[[r - lo for r in rs]]
            blocks = sub.reshape(len(rs) * nb_row, blen)
            vals = vecf(blocks)
            rowmats = vals.reshape(len(rs), nb_row * 256)[:, :ncols]
            per_layer[e] = rowmats @ prj
        FPDN_list.append(per_layer.astype(np.float32))
        dn_layers_info.append({"layer": int(lay),
                               "type": {11: "Q3_K", 12: "Q4_K", 13: "Q5_K"}[tt],
                               "sample_rows": sample_rows})
        print(f"dn fp L{lay}", flush=True)
    FPDN = np.stack(FPDN_list)

    meta = json.loads(str(old["meta"]))
    meta["dn_layers"] = dn_layers_info
    meta["dn_regenerated_after_type_fix"] = True
    np.savez_compressed(
        fp_path,
        gu=old["gu"], dn=FPDN.astype(np.float32),
        proj_gu=old["proj_gu"],
        proj_dn=np.concatenate([v.flatten() for _, v in sorted(proj_store.items())]).astype(np.float32),
        meta=json.dumps(meta | {"dn_proj_shapes": [[k, list(v.shape)] for k, v in sorted(proj_store.items())]}),
    )
    n_nan = int(np.isnan(FPDN).sum())
    print(f"fingerprints.npz rewritten; dn shape {FPDN.shape} nan_count {n_nan}", flush=True)


if __name__ == "__main__":
    main()
