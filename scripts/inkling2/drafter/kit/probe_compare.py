"""How far recordings of one plan stand from a reference recording of it, layer by layer.

The reference is the plan read by one sequence alone. Every other run recorded the same prompt rows, so a row of a run
that is far from the reference's row was computed from something else than the reference had: this prints how many
rows are far at each recorded layer, and where in a part of a batch they sit (a part is one sequence's rows in one
decode call; its trace gives them).

Usage: probe_compare.py --ref DIR --run DIR [--run DIR ...] [--far 0.02]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_record import Recording  # noqa: E402
from diag_restore import read_steps, read_trace, row_facts  # noqa: E402

EDGES = [0, 1, 2, 3, 7, 15, 31, 47, 63, 127]


def gaps(a: Recording, pa: str, ia: list, b: Recording, pb: str, ib: list, rows: int) -> np.ndarray:
    """[rows, layers]: the largest difference in a row over the largest value of the reference's row."""
    out = np.empty((rows, len(ia)), dtype=np.float64)
    step = max(1, (64 << 20) // max(a.row_bytes, b.row_bytes))
    for r0 in range(0, rows, step):
        n = min(step, rows - r0)
        x, y = a.state_rows(pa, r0, n)[:, ia], b.state_rows(pb, r0, n)[:, ib]
        scale = np.maximum(np.abs(x).max(axis=2), 1e-30)
        out[r0:r0 + n] = np.abs(x - y).max(axis=2) / scale
    return out


def buckets(keys: np.ndarray) -> list:
    """(name, mask) for each range of EDGES that holds a row."""
    out, lo = [], None
    for hi in EDGES + [None]:
        m = np.ones(keys.size, dtype=bool)
        if lo is not None:
            m &= keys > lo
        if hi is not None:
            m &= keys <= hi
        if m.any():
            a = 0 if lo is None else lo + 1
            out.append(("%d" % a if hi == a else "%d..%s" % (a, "" if hi is None else hi), m))
        lo = hi
    return out


def by_place(title: str, keys: np.ndarray, far: np.ndarray, layers: list) -> None:
    print("  " + title)
    print("    %-9s %8s  %s" % ("place", "rows", "  ".join("layer %-2d" % t for t in layers)))
    for name, m in buckets(keys):
        print("    %-9s %8d  %s" % (name, int(m.sum()), "  ".join("%7.2f%%" % (100 * float(far[m, k].mean()))
                                                                for k in range(len(layers)))))


def verify_lines(path: str) -> None:
    name = os.path.join(path, "verify.txt")
    if not os.path.isfile(name):
        return
    moved, trips, total = [], [], None
    with open(name, encoding="utf-8") as f:
        for line in f:
            w = line.split()
            if w and w[0] == "M":
                moved.append(w)
            elif w and w[0] == "T":
                trips.append(w)
            elif w and w[0] == "S":
                total = [int(x) for x in w[1:]]
    if total:
        print("  states looked at: %d sequences a decode did not hold or that waited between two, %d changed; %d loaded "
              "states read back, %d not as loaded" % tuple(total))
    kinds = {}
    for w in moved:
        kinds.setdefault(w[1], []).append(w)
    for when, ws in sorted(kinds.items()):
        sizes = sorted({(int(w[6]), int(w[7])) for w in ws})
        first = sorted(int(w[8]) for w in ws)
        last = sorted(int(w[9]) for w in ws)
        count = sorted(int(w[10]) for w in ws)
        print("    changed %s: %d times; sizes before/after %s; first byte apart %d..%d, last %d..%d, bytes apart %d..%d "
              "(median %d)" % (when, len(ws), sizes[:4], first[0], first[-1], last[0], last[-1], count[0], count[-1],
                               count[len(count) // 2]))
        for w in ws[:6]:
            print("      step P%s G%s sequence %s (state %s): %s -> %s bytes, apart %s..%s, %s bytes; positions %s..%s" % (
                w[2], w[3], w[4], w[5], w[6], w[7], w[8], w[9], w[10], w[11], w[12]))
    kinds = {}
    for w in trips:
        kinds.setdefault(w[1], []).append(w)
    for what, ws in sorted(kinds.items()):
        print("    loaded at a %s and not read back the same: %d times" % (what, len(ws)))
        for w in ws[:6]:
            print("      step P%s G%s sequence %s: %s -> %s bytes, apart %s..%s, %s bytes" % (
                w[2], w[3], w[4], w[5], w[6], w[7], w[8], w[9]))


def compare(ref: Recording, run: Recording, far_at: float) -> None:
    layers = [t for t in ref.taps if t in run.taps]
    ia, ib = [ref.taps.index(t) for t in layers], [run.taps.index(t) for t in layers]
    t_ref, t_run = read_trace(ref.path), read_trace(run.path)
    steps, _ = read_steps(run.path, [])
    parts_in = {step: len(parts) for step, parts in steps.items()}
    cols = {}
    missing = 0
    for rid, e in ref.index.items():
        o = run.index.get(rid)
        if o is None or o["states_rows"] != e["states_rows"]:
            missing += 1
            continue
        rows, shared, name = e["states_rows"], e["shared_len"], e["file"]
        g = gaps(ref, ref.req(e, ".states"), ia, run, run.req(o, ".states"), ib, rows)
        fr, fo = row_facts(t_ref.get(name, []), shared, rows), row_facts(t_run.get(name, []), shared, rows)
        add = {"gap": g, "off": fo["off"], "ref_off": fr["off"], "n": fo["n"], "used": fo["used"],
               "parts": np.array([parts_in[s] for s in fo["step"]], dtype=np.int64), "row": np.arange(rows)}
        for k, v in add.items():
            cols.setdefault(k, []).append(v)
    print("== %s against %s: layers %s%s" % (os.path.basename(run.path), os.path.basename(ref.path), layers,
                                           ", %d requests not in both" % missing if missing else ""))
    if not cols:
        print("  nothing to compare")
        return
    c = {k: np.concatenate(v) for k, v in cols.items()}
    gap, far = c["gap"], c["gap"] > far_at
    for k, t in enumerate(layers):
        print("  layer %d: %d of %d rows far (%.2f%%), median %.2e, 99%% %.2e, 99.9%% %.2e, most %.2e" % (
            t, int(far[:, k].sum()), far.shape[0], 100 * float(far[:, k].mean()), float(np.median(gap[:, k])),
            float(np.quantile(gap[:, k], 0.99)), float(np.quantile(gap[:, k], 0.999)), float(gap[:, k].max())))
    by_place("far rows by the row's place in its part of the batch, in this run", c["off"], far, layers)
    by_place("far rows by the row's place in the reference's part", c["ref_off"], far, layers)
    by_place("far rows by the row's place in its request", c["row"], far, layers)
    print("  far rows by the parts in the row's batch, last layer")
    for name, m in (("1", c["parts"] == 1), ("2..4", (c["parts"] >= 2) & (c["parts"] <= 4)), ("5..", c["parts"] >= 5)):
        if m.any():
            print("    %-6s %8d rows, %6.2f%% far" % (name, int(m.sum()), 100 * float(far[m, -1].mean())))
    # the first layer a far row is far at
    last = far[:, -1]
    if last.any():
        first = np.argmax(far[last], axis=1)
        print("  rows far at the last layer, by the first layer they are far at: %s" % ", ".join(
            "layer %d: %d" % (layers[k], int((first == k).sum())) for k in range(len(layers))))
        # the median gap by place in part, for the starts that are far: how the difference decays along a part
        starts = np.flatnonzero((c["off"] == 0) & last)
        print("  parts whose first row is far at the last layer: %d" % starts.size)
    verify_lines(run.path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ref", required=True)
    p.add_argument("--run", action="append", default=[])
    p.add_argument("--far", type=float, default=0.02)
    args = p.parse_args()
    ref = Recording(args.ref)
    for path in args.run:
        if not os.path.isfile(os.path.join(path, "index.jsonl")) or not os.path.isfile(os.path.join(path, "meta.json")):
            print("== %s: no recording" % os.path.basename(path))
            continue
        try:
            compare(ref, Recording(path), args.far)
        except SystemExit as ex:
            print("== %s: %s" % (os.path.basename(path), ex))
        sys.stdout.flush()


if __name__ == "__main__":
    main()
