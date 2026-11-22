#!/usr/bin/env python3
"""One-shot zero-code fraction for ONE gate/up layer (by sorted index position).
Updates zero-frac-progress.json atomically; exits after one layer so the caller's
per-layer ssh invocations survive host-side job reaping.
Usage: zf_one.py <pack.gguf> <outdir> <index_into_sorted_layers>
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
    pack, outdir, idx = sys.argv[1], sys.argv[2], int(sys.argv[3])
    ver, kv, tensors, data_start = WR.parse_gguf(pack)

    def by_suffix(suffix):
        out = {}
        for t in tensors:
            parts = t["name"].split(".")
            if len(parts) >= 3 and ".".join(parts[2:]) == suffix:
                out[int(parts[1])] = t
        return dict(sorted(out.items()))

    gate_up = by_suffix("ffn_gate_up_exps.weight")
    layers = sorted(gate_up)
    lay = layers[idx]
    zpath = os.path.join(outdir, "zero-frac-progress.json")
    zl = json.load(open(zpath)) if os.path.exists(zpath) else {}

    key = str(lay)
    if key in zl:
        print(f"zero-frac L{lay}: {zl[key]:.4f} (cached)")
        return

    tinfo = gate_up[lay]
    nb_row = tinfo["dims"][0] // 256
    rowbytes = nb_row * 110
    base = data_start + tinfo["offset"]
    total_rows = tinfo["dims"][1] * tinfo["dims"][2]
    stride = 16 if idx < 6 else 64     # first six measured at finer stride
    rows_sel = np.asarray(mm_view(pack)[base : base + total_rows * rowbytes],
                          dtype=np.uint8).reshape(total_rows, rowbytes)[::stride]
    tot = zer = 0.0
    CH_ROWS = max(3640, 1)
    for s in range(0, rows_sel.shape[0], CH_ROWS):
        blks = rows_sel[s : s + CH_ROWS].reshape(-1, 110)
        deq = WR.deq_q3_K_vec(blks)
        zer += float((deq == 0).sum())
        tot += deq.size
    zl[key] = zer / max(tot, 1)
    tmp = zpath + ".tmp"
    json.dump(zl, open(tmp, "w"))
    os.replace(tmp, zpath)
    print(f"zero-frac L{lay}: {zl[key]:.4f}  ({len(zl)}/48)")


_mm_cache = {}


def mm_view(pack):
    if pack not in _mm_cache:
        _mm_cache[pack] = np.memmap(pack, dtype=np.uint8, mode="r")
    return _mm_cache[pack]


if __name__ == "__main__":
    main()
