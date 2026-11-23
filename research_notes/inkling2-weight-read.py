#!/usr/bin/env python3
"""Stream-read Inkling-2-Small-abliterated-e4853745-Q3_K_M-imatrix.gguf (CPU only).

Outputs into <outdir>:
  gguf-metadata.json      full KV metadata dump
  tensor-inventory.json   every tensor: name, dims, type, offset
  routers.npz             float32 [L, 384, 4608] router gates read exactly
  router-stats.json       per-layer router structure statistics
  fingerprints.npz        exact-dequant sampled expert rows -> fixed projections
  golden-selftest.npz     raw blocks + both dequant outputs (vectorized vs oracle)
  expert-blocks.npz       per layer/expert scale-level block statistics
  run-log.json            timings and self-test results

Dequantizers are numpy transcriptions of ggml-quants.c reference loops
(dequantize_row_q3_K/q4_K/q5_K, get_scale_min_k4) and must pass a bitwise
self-test against scalar oracle ports before any number is trusted.
"""
import json
import os
import struct
import sys
import time

import numpy as np

KMASK1 = int(0x03030303)
KMASK2 = int(0x0F0F0F0F)

# ggml.h canonical ids (verified against include/ggml.h): Q3_K=11, Q4_K=12,
# Q5_K=13 — NOT the GGUF-metadata numbering.
TYPE_NAMES = {0: "F32", 1: "F16", 8: "Q8_0", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K",
              13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 30: "BF16", 39: "MXFP4"}


def fp16(b):
    return b.copy().view(np.float16).astype(np.float32)[..., 0]


# ---------------------------------------------------------------------------
# GGUF parsing
# ---------------------------------------------------------------------------

# GGUF metadata value types (ggml/include/gguf.h): STRING=8, ARRAY=9
KV_SCALAR = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2),
             4: ("<I", 4), 5: ("<i", 4), 6: ("<f", 4), 7: ("<?", 1),
             10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8)}
T_STRING = 8
T_ARRAY = 9


class _NeedMore(Exception):
    pass


def parse_gguf(path):
    size = os.path.getsize(path)

    def attempt(window, type_width):
        """Walk header with the given kv value-type width. Raises _NeedMore on
        truncation, LayoutError on a decode that cannot be trusted."""
        with open(path, "rb") as f:
            buf = f.read(min(window, size))
        if buf[:4] != b"GGUF":
            raise ValueError("bad magic")
        ver = struct.unpack_from("<I", buf, 4)[0]
        n_tensors = struct.unpack_from("<Q", buf, 8)[0]
        n_kv = struct.unpack_from("<Q", buf, 16)[0]

        class LayoutError(Exception):
            pass

        def need(pos, nb):
            if pos + nb > len(buf):
                raise _NeedMore()

        def rd_str(pos):
            slen = struct.unpack_from("<Q", buf, pos)[0]; pos += 8
            need(pos - 8, 8 + min(slen, 4096))
            raw = bytes(buf[pos : pos + slen])
            try:
                s = raw.decode("utf-8")
            except UnicodeDecodeError:
                raise LayoutError(f"string at {pos} not utf-8")
            return s, pos + slen

        tw = type_width          # bytes per value-type field (1 standard, 4 this pack)
        fmt_t = "<I" if tw == 4 else "<B"

        kv = {}
        pos = 24
        for _ in range(n_kv):
            klen = struct.unpack_from("<Q", buf, pos)[0]; pos += 8
            need(pos, min(klen, 4096))
            try:
                key = bytes(buf[pos : pos + klen]).decode("utf-8")
            except UnicodeDecodeError:
                raise LayoutError(f"key at {pos} not utf-8")
            pos += klen
            t = struct.unpack_from(fmt_t, buf, pos)[0]
            if tw == 1 and t > 12:
                raise LayoutError(f"type {t} out of range at {pos}")
            pos += tw
            if t in KV_SCALAR:
                fmt, sz = KV_SCALAR[t]
                need(pos, sz)
                v = struct.unpack_from(fmt, buf, pos)[0]
                if t == 7:
                    val = ("bool", bool(v))
                elif t in (6, 12):
                    val = ("float", float(v))
                else:
                    val = ("int", int(v) if not hasattr(v, "item") else int(v.item()))
                pos += sz
            elif t == T_STRING:
                s, p2 = rd_str(pos)
                val, pos = ("str", s), p2
            elif t == T_ARRAY:
                etw = tw if tw > 1 else 1
                et_fmt = "<I" if etw == 4 else "<B"
                et = struct.unpack_from(et_fmt, buf, pos)[0]
                if etw == 1 and et > 12:
                    raise LayoutError(f"elem type {et} out of range")
                pos += etw
                cnt = struct.unpack_from("<Q", buf, pos)[0]; pos += 8
                need(pos, 16)
                if et == T_STRING:
                    vals, p2 = [], pos
                    for i in range(cnt):
                        s, p2 = rd_str(p2)
                        if i < 128:
                            vals.append(s)
                    val, pos = ("arr", {"strings": vals, "n": int(cnt)}), p2
                elif et in KV_SCALAR:
                    _, sz = KV_SCALAR[et]
                    payload = cnt * sz
                    if payload > len(buf) - pos:
                        val = ("arr", {"n": int(cnt), "skipped_bytes": int(payload)})
                        # cannot skip blind without knowing size; grow window instead
                        raise _NeedMore()
                    dt = np.dtype({1: "<u1", 2: "<u2", 4: "<u4", 8: "<u8"}[sz])
                    arr = np.frombuffer(bytes(buf[pos : pos + payload]), dtype=dt)
                    val, pos = ("arr", {"n": int(cnt), "head": arr[:160].tolist()}), pos + payload
                else:
                    raise LayoutError(f"array elem type {et} at key {key}")
            else:
                raise LayoutError(f"kv value type {t} at key {key}")
            kv[key] = {"kind": val[0], "value": val[1]}

        tensors = []
        tpos = pos
        for _i in range(n_tensors):
            need(tpos, 24)
            klen = struct.unpack_from("<Q", buf, tpos)[0]; tpos += 8
            need(tpos, min(klen, 4096))
            name = bytes(buf[tpos : tpos + klen]).decode("utf-8"); tpos += klen
            nd = struct.unpack_from("<I", buf, tpos)[0]; tpos += 4
            dims_desc = list(struct.unpack_from("<" + "Q" * nd, buf, tpos)); tpos += 8 * nd
            ttype = struct.unpack_from("<I", buf, tpos)[0]; tpos += 4
            off = struct.unpack_from("<Q", buf, tpos)[0]; tpos += 8
            # ne[] is stored lowest index first; ne[0] is the fastest dim.
            tensors.append({"name": name, "dims": list(dims_desc),
                            "type": int(ttype), "offset": int(off)})
        align = int(kv.get("general.alignment", {}).get("value", 32))
        data_start = (tpos + align - 1) // align * align
        return ver, kv, tensors, data_start

    for type_width in (4, 1):
        window = 64 * 1024 * 1024
        while True:
            try:
                out = attempt(window, type_width)
                # remember which layout matched for the run log
                out[1].setdefault("_layout_note", {})["kv_type_width"] = type_width
                return out
            except _NeedMore:
                window *= 4
                if window > size:
                    raise
            except Exception as e:
                if type_width == 1:
                    raise
                break   # fall through to standard-width attempt
    raise ValueError("gguf header did not parse under any known layout")

# ---------------------------------------------------------------------------
# Scalar oracle dequantizers — literal transcription of ggml-quants.c.
# ---------------------------------------------------------------------------

def get_scale_min_k4(j, q):   # q: 12-byte scale field as list/array
    if j < 4:
        return q[j] & 63, q[j + 4] & 63
    d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4)
    m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4)
    return d, m


def q3k_shuffle(aux3):
    """aux3: (..., 3) uint64 of the scale field's three u32 words.
    Returns (..., 16) int64 sub-block scales, per dequantize_row_q3_K."""
    a0w, a1w, tmp = aux3[..., 0], aux3[..., 1], aux3[..., 2]
    a2 = ((a0w >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4)
    a3 = ((a1w >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4)
    a0 = (a0w & KMASK2) | (((tmp >> 0) & KMASK1) << 4)
    a1 = (a1w & KMASK2) | (((tmp >> 2) & KMASK1) << 4)
    w = np.stack([a0, a1, a2, a3], axis=-1).astype(np.uint32)
    return np.ascontiguousarray(w).view(np.int8)[..., :16].astype(np.int64)


def deq_q3_K_ref(block):
    hm = block[0:32]
    qs = np.frombuffer(bytes(block[32:96]), dtype=np.uint8)
    words = np.array(struct.unpack("<3I", bytes(block[96:108])), dtype=np.uint64)[None, :]
    sc = q3k_shuffle(words)[0]                 # 16 signed byte scales
    d_all = struct.unpack("<e", bytes(block[108:110]))[0]
    out, isx, mbits = [], 0, 1
    for h in range(2):
        shift = 0
        qoff = h * 32
        for j in range(4):
            dl = d_all * (sc[isx] - 32); isx += 1
            for l in range(16):
                code = int(qs[qoff + l]) >> shift & 3
                if not (hm[l] & mbits):
                    code -= 4
                out.append(dl * (code - 256 if code > 127 else code))
            dl = d_all * (sc[isx] - 32); isx += 1
            for l in range(16):
                code = int(qs[qoff + 16 + l]) >> shift & 3
                if not (hm[16 + l] & mbits):
                    code -= 4
                out.append(dl * (code - 256 if code > 127 else code))
            shift += 2
            mbits <<= 1
    return np.array(out, dtype=np.float64)


def deq_q4_K_ref(block):
    d = struct.unpack("<e", bytes(block[0:2]))[0]
    mn = struct.unpack("<e", bytes(block[2:4]))[0]
    sbytes = list(block[4:16])
    qs = block[16:144]
    out, isx, qoff = [], 0, 0
    for _w in range(4):
        sc1, m1v = get_scale_min_k4(isx, sbytes)
        d1, mm1 = d * sc1, mn * m1v
        sc2, m2v = get_scale_min_k4(isx + 1, sbytes)
        d2, mm2 = d * sc2, mn * m2v
        for l in range(32):
            out.append(d1 * (qs[qoff + l] & 0xF) - mm1)
        for l in range(32):
            out.append(d2 * (qs[qoff + l] >> 4) - mm2)
        qoff += 32; isx += 2
    return np.array(out, dtype=np.float64)


def deq_q5_K_ref(block):
    d = struct.unpack("<e", bytes(block[0:2]))[0]
    mn = struct.unpack("<e", bytes(block[2:4]))[0]
    sbytes = list(block[4:16])
    ql = block[16:144]
    qh = block[144:176]
    out, isx, qoff, u1, u2 = [], 0, 0, 1, 2
    for _w in range(4):
        sc1, m1v = get_scale_min_k4(isx, sbytes)
        d1, mm1 = d * sc1, mn * m1v
        sc2, m2v = get_scale_min_k4(isx + 1, sbytes)
        d2, mm2 = d * sc2, mn * m2v
        for l in range(32):
            lo = ql[qoff + l] & 0xF
            hi = 16 if (qh[l] & u1) else 0
            out.append(d1 * (lo + hi) - mm1)
        for l in range(32):
            lo = ql[qoff + l] >> 4
            hi = 16 if (qh[l] & u2) else 0
            out.append(d2 * (lo + hi) - mm2)
        qoff += 32; isx += 2; u1 <<= 2; u2 <<= 2
    return np.array(out, dtype=np.float64)


# ---------------------------------------------------------------------------
# Vectorized batch dequantizers (blocks: (n, blen) uint8).
# ---------------------------------------------------------------------------

def deq_q3_K_vec(blocks):
    n = blocks.shape[0]
    aux = blocks[:, 96:108].copy().view("<u4").astype(np.uint64)
    d_all = fp16(blocks[:, 108:110])                     # (n,)
    tmp = aux[:, 2]
    a0 = (aux[:, 0] & KMASK2) | (((tmp >> 0) & KMASK1) << 4)
    a1 = (aux[:, 1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4)
    a2 = ((aux[:, 0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4)
    a3 = ((aux[:, 1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4)
    w = np.stack([a0, a1, a2, a3], axis=1).astype(np.uint32)
    scb = np.ascontiguousarray(w).view(np.int8).reshape(n, 16).astype(np.int64)
    smul = (scb.astype(np.float32) - 32.0)               # (n, 16)

    hm = blocks[:, 0:32].astype(np.int64)                # (n, 32)
    qs = blocks[:, 32:96].astype(np.int64)               # (n, 64)
    out = np.empty((n, 256), dtype=np.float32)
    isx = 0
    bitpos = 0
    for h in range(2):
        qbase = h * 32
        vbase = h * 128
        for j in range(4):
            shift = np.int64(2 * j)
            bit = np.int64(1 << bitpos); bitpos += 1
            s_lo = smul[:, isx][:, None]; s_hi = smul[:, isx + 1][:, None]
            isx += 2
            blo = ((qs[:, qbase + 0 : qbase + 16] >> shift) & 3).astype(np.float32)
            bhi = ((qs[:, qbase + 16 : qbase + 32] >> shift) & 3).astype(np.float32)
            hlo = (hm[:, 0:16] & bit) != 0
            hri = (hm[:, 16:32] & bit) != 0
            dv = d_all[:, None]
            out[:, vbase + j * 32 : vbase + j * 32 + 16] = dv * s_lo * np.where(hlo, blo, blo - 4.0)
            out[:, vbase + j * 32 + 16 : vbase + j * 32 + 32] = dv * s_hi * np.where(hri, bhi, bhi - 4.0)
    return out


def deq_q45_K_vec(blocks, with_high):
    n = blocks.shape[0]
    d = fp16(blocks[:, 0:2])
    mnv = fp16(blocks[:, 2:4])
    sb = blocks[:, 4:16].astype(np.int64)
    sc = np.empty((n, 8), dtype=np.int64)
    mi = np.empty((n, 8), dtype=np.int64)
    for j in range(8):
        if j < 4:
            sc[:, j] = sb[:, j] & 63
            mi[:, j] = sb[:, j + 4] & 63
        else:
            sc[:, j] = (sb[:, j + 4] & 0xF) | ((sb[:, j - 4] >> 6) << 4)
            mi[:, j] = (sb[:, j + 4] >> 4) | ((sb[:, j] >> 6) << 4)

    qs = blocks[:, 16:144].astype(np.int64)
    qh = blocks[:, 144:176].astype(np.int64) if with_high else None

    out = np.empty((n, 256), dtype=np.float64)
    vbase = 0
    for w in range(4):                     # four 64-value windows
        d1 = (d * sc[:, 2 * w])[:, None]
        mm1 = (mnv * mi[:, 2 * w])[:, None]
        d2 = (d * sc[:, 2 * w + 1])[:, None]
        mm2 = (mnv * mi[:, 2 * w + 1])[:, None]
        blo = (qs[:, w * 32 : w * 32 + 32] & 0xF).astype(np.float64)
        bhi = (qs[:, w * 32 : w * 32 + 32] >> 4).astype(np.float64)
        if with_high:
            u1 = np.int64(1 << (2 * w))
            u2 = np.int64(2 << (2 * w))
            hlo = ((qh & u1) != 0).astype(np.float64) * 16.0     # qh is (n,32), reused each window
            hri = ((qh & u2) != 0).astype(np.float64) * 16.0
            out[:, vbase:vbase + 32] = d1 * (blo + hlo) - mm1
            out[:, vbase + 32:vbase + 64] = d2 * (bhi + hri) - mm2
        else:
            out[:, vbase:vbase + 32] = d1 * blo - mm1
            out[:, vbase + 32:vbase + 64] = d2 * bhi - mm2
        vbase += 64
    return out


def self_test(vecf, oracf, blocks, name):
    a = vecf(blocks).astype(np.float64)
    b = np.stack([oracf(x) for x in blocks])
    eq = bool(a.shape == b.shape and np.array_equal(a, b))
    md = float(np.abs(a - b).max()) if a.shape == b.shape else None
    print(f"self-test {name}: equal={eq} max_abs_diff={md}", flush=True)
    return {"equal": eq, "max_abs_diff": md}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def by_suffix(tensors, suffix):
    out = {}
    for t in tensors:
        parts = t["name"].split(".")
        if len(parts) >= 3 and ".".join(parts[2:]) == suffix:
            out[int(parts[1])] = t
    return dict(sorted(out.items()))


def geom(tinfo, blen):
    """(blocks per row, bytes per row, total rows) with dims [cols, rows/e, experts]."""
    nb_row = tinfo["dims"][0] // 256
    rowbytes = nb_row * blen
    rows_per_expert = tinfo["dims"][1]
    n_exp = tinfo["dims"][2] if len(tinfo["dims"]) > 2 else 1
    return nb_row, rowbytes, rows_per_expert, n_exp


def stage_done(outdir, name):
    return os.path.exists(os.path.join(outdir, name))


def main():
    pack, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()
    log = {"resumed": {}}

    if not (stage_done(outdir, "gguf-metadata.json") and
            stage_done(outdir, "tensor-inventory.json")):
        ver, kv, tensors, data_start = parse_gguf(pack)
        print(f"gguf v{ver}: {len(tensors)} tensors, data at {data_start}", flush=True)
        json.dump({"version": ver, "kv": kv},
                  open(os.path.join(outdir, "gguf-metadata.json"), "w"), indent=1, default=str)
        json.dump(tensors, open(os.path.join(outdir, "tensor-inventory.json"), "w"), indent=1)
    ver, kv, tensors, data_start = parse_gguf(pack)
    print(f"gguf v{ver} resumed: {len(tensors)} tensors, data at {data_start}", flush=True)
    json.dump({"version": ver, "kv": kv},
              open(os.path.join(outdir, "gguf-metadata.json"), "w"), indent=1, default=str)
    json.dump(tensors, open(os.path.join(outdir, "tensor-inventory.json"), "w"), indent=1)

    mm = np.memmap(pack, dtype=np.uint8, mode="r")
    routers = by_suffix(tensors, "ffn_gate_inp.weight")
    gate_up = by_suffix(tensors, "ffn_gate_up_exps.weight")
    downs = by_suffix(tensors, "ffn_down_exps.weight")
    print(f"routers {len(routers)}, gate/up {len(gate_up)}, down {len(downs)}", flush=True)

    types_table = {
        "gate_up": {str(l): TYPE_NAMES.get(t["type"], str(t["type"])) for l, t in gate_up.items()},
        "down": {str(l): TYPE_NAMES.get(t["type"], str(t["type"])) for l, t in downs.items()},
        "router": sorted({TYPE_NAMES.get(t["type"], "?") for t in routers.values()}),
        "attn_families": {},
    }
    attn_types = {}
    for t in tensors:
        parts = t["name"].split(".")
        if len(parts) >= 4 and parts[2].startswith("attn_"):
            fam = ".".join(parts[3:-1])
            if fam:
                attn_types.setdefault(fam, set()).add(TYPE_NAMES.get(t["type"], str(t["type"])))
    types_table["attn_families"] = {k: sorted(v) for k, v in sorted(attn_types.items())}

    # ---- self-test on real file blocks -------------------------------------
    cols_gu = gate_up[min(gate_up)]["dims"][0]
    nb_row_gu, rowbytes_gu, rows_per_expert, n_exp = geom(gate_up[min(gate_up)], 110)
    assert n_exp == 384 and cols_gu == 4608

    def read_rows(tinfo, blen, lo, hi):
        _nb, rb, _rpe, _ne = geom(tinfo, blen)
        base = data_start + tinfo["offset"]
        view = mm[base + lo * rb : base + hi * rb]
        return np.asarray(view, dtype=np.uint8).reshape(hi - lo, rb).copy()

    two_rows = read_rows(gate_up[min(gate_up)], 110, 0, 2)   # first rows of expert 0
    st_q3 = two_rows.reshape(-1, 110)[:32].copy()
    dn_l0 = downs[min(downs)]
    DN_BL = {11: 110, 12: 144, 13: 176}
    dn_blen = DN_BL[dn_l0["type"]]
    st_dn = read_rows(dn_l0, dn_blen, 0, 2).reshape(-1, dn_blen)[:16].copy()
    tests = {"gate_up_Q3_K": self_test(deq_q3_K_vec, deq_q3_K_ref, st_q3, "q3_K")}
    tests["down_%s" % TYPE_NAMES.get(dn_l0["type"], "?")] = self_test(
        lambda b: deq_q45_K_vec(b, dn_l0["type"] == 13),
        deq_q5_K_ref if dn_l0["type"] == 13 else deq_q4_K_ref,
        st_dn, TYPE_NAMES.get(dn_l0["type"], str(dn_l0["type"])))
    q5_layer = next((l for l, t in downs.items() if t["type"] == 13), None)
    q4_layer = next((l for l, t in downs.items() if t["type"] == 12), None)
    if q5_layer is not None and (q5_layer != min(downs)):
        s2 = read_rows(downs[q5_layer], 176, 0, 2).reshape(-1, 176)[:32].copy()
        tests["q5_K"] = self_test(lambda b: deq_q45_K_vec(b, True), deq_q5_K_ref, s2, "q5_K")
    if q4_layer is not None and (q4_layer != min(downs)):
        s3 = read_rows(downs[q4_layer], 144, 0, 2).reshape(-1, 144)[:32].copy()
        tests["q4_K"] = self_test(lambda b: deq_q45_K_vec(b, False), deq_q4_K_ref, s3, "q4_K")

    np.savez_compressed(
        os.path.join(outdir, "golden-selftest.npz"),
        q3_blocks=st_q3,
        dn_blocks=st_dn,
        q3_vec=deq_q3_K_vec(st_q3),
        q3_oracle=np.stack([deq_q3_K_ref(x) for x in st_q3]).astype(np.float32),
    )
    log["self_test"] = {k: v for k, v in tests.items()}
    if not stage_done(outdir, "routers.npz"):
      pass
    else:
      print("routers.npz exists — skipping router stage", flush=True)

    # ---- routers (skipped when already written) -----------------------------
    rng = np.random.default_rng(7)
    ROUTER_SKIP = stage_done(outdir, "routers.npz")
    if ROUTER_SKIP:
        pass  # placeholder; real skip via early branch below
    stack = []
    router_do = not ROUTER_SKIP
    if router_do:
      proj4608 = (rng.standard_normal((4608, 128)) / np.sqrt(128.0)).astype(np.float32)
    proj4608 = (rng.standard_normal((4608, 128)) / np.sqrt(128.0)).astype(np.float32)

    stack = []
    for lay in sorted(routers):
        tinfo = routers[lay]
        nb = int(np.prod(tinfo["dims"]))
        base = data_start + tinfo["offset"]
        arr = mm[base : base + nb * 4].view("<f4").reshape(tinfo["dims"][1], tinfo["dims"][0])
        stack.append(arr.copy())
    R = np.stack(stack).astype(np.float32)            # (L, 384, 4608)
    norms = np.linalg.norm(R, axis=2)                 # (L,384)
    rn = R / (norms[:, :, None] + 1e-30)
    L = R.shape[0]
    per_layer = []
    for i, lay in enumerate(sorted(routers)):
        g = np.einsum("ij,kj->ik", rn[i], rn[i]).astype(np.float64)
        off = g[np.triu_indices(384, 1)]
        wv = np.linalg.eigvalsh(g)[::-1]
        pr = float(wv.sum() ** 2 / np.square(wv).sum()) if wv.sum() > 0 else 0.0
        per_layer.append({
            "layer": int(lay),
            "norm_mean": float(norms[i].mean()), "norm_std": float(norms[i].std()),
            "norm_max_over_min": float(norms[i].max() / max(float(norms[i].min()), 1e-30)),
            "cos_offdiag_mean": float(off.mean()),
            "cos_abs_max": float(np.abs(off).max()),
            "pairs_cos_above_095": int((off > 0.95).sum()),
            "participation_ratio": pr,
            "top1_eigenvalue_share": float(wv[0] / wv.sum()),
        })
    json.dump({"per_layer": per_layer, "router_types": types_table["router"]},
              open(os.path.join(outdir, "router-stats.json"), "w"), indent=1)
    np.savez(os.path.join(outdir, "routers.npz"),
             weights=R, layers=np.array(sorted(routers), dtype=np.int64))
    print("router pass done", flush=True)

    # ---- sampled expert-row fingerprints (exact dequant) -------------------
    gu_layers = sorted(gate_up)
    dn_layers = sorted(downs)
    sample_rows_gu = [0, 1023, 2304, 3327]           # within one expert's 4608 rows
    FPGU = np.empty((len(gu_layers), n_exp, len(sample_rows_gu), 128), dtype=np.float32)
    golden_e0_blocks = None

    for li, lay in enumerate(gu_layers):
        tinfo = gate_up[lay]
        nb_row, rowbytes, rpe, ne = geom(tinfo, 110)
        assert nb_row == 18 and rpe == rows_per_expert and ne == n_exp
        base = data_start + tinfo["offset"]
        for e in range(n_exp):
            rs = [(e * rpe + rr) for rr in sample_rows_gu]
            lo, hi = min(rs), max(rs) + 1
            sub = np.asarray(mm[base + lo * rowbytes : base + hi * rowbytes],
                             dtype=np.uint8).reshape(hi - lo, rowbytes)[[r - lo for r in rs]]
            blocks = sub.reshape(len(rs) * nb_row, 110)
            vals = deq_q3_K_vec(blocks)              # (rows*nb_row,256)
            rowmats = vals.reshape(len(rs), nb_row * 256)[:, :cols_gu]
            FPGU[li, e] = rowmats @ proj4608
            if li == 0 and e == 0:
                golden_e0_blocks = blocks[:64].copy()
        print(f"gu fingerprint L{lay}", flush=True)

    dn_layers_info = []
    FPDN_list = []
    proj_dn_store = {}
    DN_DEQ = {
        11: (110, lambda b: deq_q3_K_vec(b)),
        12: (144, lambda b: deq_q45_K_vec(b, False)),   # Q4_K
        13: (176, lambda b: deq_q45_K_vec(b, True)),    # Q5_K
    }
    for lay in dn_layers:
        tinfo = downs[lay]
        tt = tinfo["type"]
        if tt not in DN_DEQ:
            raise ValueError(f"unsupported down type {tt} at layer {lay}")
        blen, vecf = DN_DEQ[tt]
        nb_row, rowbytes, rpe_d, ne_d = geom(tinfo, blen)
        assert rpe_d == rows_per_expert and ne_d == n_exp
        ncols = tinfo["dims"][0]                      # 2304
        sample_rows_dn = [0, 1151, 2303, 3455]
        prj = (rng.standard_normal((ncols, 64)) / np.sqrt(64.0)).astype(np.float32)
        proj_dn_store[str(lay)] = prj
        per_layer = np.empty((n_exp, len(sample_rows_dn), 64), dtype=np.float32)
        base = data_start + tinfo["offset"]
        for e in range(n_exp):
            rs = [(e * rpe_d + rr) for rr in sample_rows_dn]
            lo, hi = min(rs), max(rs) + 1
            sub = np.asarray(mm[base + lo * rowbytes : base + hi * rowbytes],
                             dtype=np.uint8).reshape(hi - lo, rowbytes)[[r - lo for r in rs]]
            blocks = sub.reshape(len(rs) * nb_row, blen)
            vals = vecf(blocks)
            rowmats = vals.reshape(len(rs), nb_row * 256)[:, :ncols]
            per_layer[e] = rowmats @ prj
        FPDN_list.append(per_layer)
        dn_layers_info.append({"layer": int(lay), "type": TYPE_NAMES.get(tinfo["type"]),
                               "sample_rows": sample_rows_dn})
    FPDN = np.stack(FPDN_list)

    if golden_e0_blocks is not None:
        np.savez_compressed(
            os.path.join(outdir, "golden-gu-e0.npz"), blocks=golden_e0_blocks)
    print(f"fingerprints done {FPGU.shape} {FPDN.shape}", flush=True)

    np.savez_compressed(
        os.path.join(outdir, "fingerprints.npz"),
        gu=FPGU.astype(np.float32), dn=FPDN.astype(np.float32),
        proj_gu=proj4608.astype(np.float32),
        proj_dn=np.concatenate([v.flatten() for _, v in sorted(proj_dn_store.items())]).astype(np.float32)
        if proj_dn_store else np.zeros(1, dtype=np.float32),
        meta=json.dumps({
            "gu_sample_rows_in_expert": [0, 1023, 2304, 3327],
            "gu_projection_seed": 7, "gu_dims": list(FPGU.shape),
            "dn_proj_shapes": [[k, v.shape] for k, v in proj_dn_store.items()],
            "dn_layers": dn_layers_info,
        }),
    )
    del FPGU, FPDN

    # ---- full-coverage scale statistics ------------------------------------
    def q3_energy_per_block(rows_u8):
        blks = rows_u8.reshape(-1, 110)
        aux = blks[:, 96:108].copy().view("<u4").astype(np.uint64)
        d_all = fp16(blks[:, 108:110])
        scb = q3k_shuffle(aux).astype(np.float64)              # (nblk, 16)
        ssum = np.square(scb - 32.0).sum(axis=1)
        return (d_all.astype(np.float64) ** 2) * ssum          # E factor per block

    def q45_energy_per_block(rows_u8, blen):
        blks = rows_u8.reshape(-1, blen)
        d = fp16(blks[:, 0:2]).astype(np.float64)
        mnv = fp16(blks[:, 2:4]).astype(np.float64)
        sb = blks[:, 4:16].astype(np.int64)
        sc = np.empty((blks.shape[0], 8), dtype=np.int64)
        mi = np.empty((blks.shape[0], 8), dtype=np.int64)
        for j in range(8):
            if j < 4:
                sc[:, j] = sb[:, j] & 63; mi[:, j] = sb[:, j + 4] & 63
            else:
                sc[:, j] = (sb[:, j + 4] & 0xF) | ((sb[:, j - 4] >> 6) << 4)
                mi[:, j] = (sb[:, j + 4] >> 4) | ((sb[:, j] >> 6) << 4)
        ssum = np.square(sc.astype(np.float64)).sum(axis=1)
        msum = np.square(mi.astype(np.float64)).sum(axis=1)
        return (d ** 2) * ssum + (mnv ** 2) * msum             # E factor per block

    CHUNK_ROWS = 4096
    energy_gu, energy_dn = [], []
    gu_layers_idx = []
    dn_layers_idx = []

    for lay in sorted(gate_up):
        tinfo = gate_up[lay]
        nb_row, rowbytes, _rpe, ne = geom(tinfo, 110)
        base = data_start + tinfo["offset"]
        total_rows = rows_per_expert * n_exp
        e_acc = np.zeros(n_exp, dtype=np.float64)
        done = 0
        while done < total_rows:
            take = min(CHUNK_ROWS, total_rows - done)
            rows = np.asarray(mm[base + done * rowbytes : base + (done + take) * rowbytes],
                              dtype=np.uint8).reshape(take, rowbytes)
            eb = q3_energy_per_block(rows).reshape(take, nb_row).sum(axis=1)
            exp_idx = (done + np.arange(take)) // rows_per_expert
            np.add.at(e_acc, exp_idx, eb)
            done += take
        energy_gu.append(e_acc)
        gu_layers_idx.append(int(lay))

    down_meta = {}
    for lay in sorted(downs):
        tinfo = downs[lay]
        tt = tinfo["type"]
        if tt not in DN_BL:
            raise ValueError(f"unsupported down type {tt} at layer {lay}")
        blen = DN_BL[tt]
        nb_row, rowbytes, rpe_d, ne_d = geom(tinfo, blen)
        assert rpe_d == rows_per_expert and ne_d == n_exp
        base = data_start + tinfo["offset"]
        total_rows = rows_per_expert * n_exp
        e_acc = np.zeros(n_exp, dtype=np.float64)
        done = 0
        while done < total_rows:
            take = min(CHUNK_ROWS, total_rows - done)
            rows = np.asarray(mm[base + done * rowbytes : base + (done + take) * rowbytes],
                              dtype=np.uint8).reshape(take, rowbytes)
            if tt == 11:
                eb = q3_energy_per_block(rows)
            else:
                eb = q45_energy_per_block(rows, blen)
            eb = eb.reshape(take, nb_row).sum(axis=1)
            exp_idx = (done + np.arange(take)) // rows_per_expert
            np.add.at(e_acc, exp_idx, eb)
            done += take
        energy_dn.append(e_acc)
        dn_layers_idx.append(int(lay))
        down_meta[str(lay)] = {"type": TYPE_NAMES.get(tinfo["type"]), "blocks_per_row": int(nb_row)}
    EG_GU = np.stack(energy_gu)
    EG_DN = np.stack(energy_dn)

    # exact zero-code fraction on a uniform 1/16 sample of gate/up blocks per layer
    zf = []
    for lay in sorted(gate_up):
        tinfo = gate_up[lay]
        _nb, rowbytes, _rpe, _ne = geom(tinfo, 110)
        base = data_start + tinfo["offset"]
        stride = 16
        tot, zer = 0, 0.0
        for start in range(0, rows_per_expert * n_exp, stride * 64):
            end = min(start + stride * 64, rows_per_expert * n_exp)
            rows = np.asarray(mm[base + start * rowbytes : base + end * rowbytes],
                              dtype=np.uint8).reshape(end - start, rowbytes)[::stride]
            deq = deq_q3_K_vec(rows.reshape(-1, 110))
            zer += float((deq == 0).sum())
            tot += deq.size
        zf.append(zer / max(tot, 1))
        print(f"zero-frac L{lay}: {zf[-1]:.4f}", flush=True)

    np.savez_compressed(
        os.path.join(outdir, "expert-blocks.npz"),
        energy_gateup=EG_GU,
        energy_down=EG_DN,
        gateup_layers=np.array(gu_layers_idx),
        down_layers=np.array(dn_layers_idx),
        zero_code_frac_gateup_sampled=np.array(zf),
        sample_stride=np.int64(16),
        rows_per_expert=np.int64(rows_per_expert),
        experts=np.int64(n_exp),
        down_meta=json.dumps(down_meta),
    )
    log["elapsed_s"] = time.time() - t0
    json.dump(log, open(os.path.join(outdir, "run-log.json"), "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
