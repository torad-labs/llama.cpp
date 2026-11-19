"""Where the rows of a recording with continuations stand apart from a read without them.

Three recordings of one plan, each made with --trace: the run with continuations, the read in whole shares, and the read
ended at the same cuts. For every row it knows what batch read it in each run (how many tokens the batch held, how long the
row's own part was, where in it the row stood, whether the part ended at a cut), and prints how often a row is far from
the other read for each of those.

Usage: diag_restore.py --rec DIR --nocuts DIR --stops DIR [--far 0.02]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_record import Recording  # noqa: E402


def read_trace(path: str) -> dict:
    """file -> list of (step, g_steps, used, idx0, p0, n, to_cut)"""
    out = {}
    with open(os.path.join(path, "trace.txt"), encoding="utf-8") as f:
        for line in f:
            w = line.split()
            if len(w) != 10 or w[0] != "P" or w[4] == "-":
                continue
            out.setdefault(w[4], []).append((int(w[1]), int(w[2]), int(w[3]), int(w[6]), int(w[7]), int(w[8]), int(w[9])))
    return out


def read_steps(path: str, keeps: list) -> dict:
    """step -> list of parts (file or '-', seq, idx0, p0, n, to_cut), in batch order; and the generating steps before it"""
    steps, g_before = {}, {}
    with open(os.path.join(path, "trace.txt"), encoding="utf-8") as f:
        for line in f:
            w = line.split()
            if len(w) != 10 or w[0] != "P":
                continue
            step = int(w[1])
            g_before[step] = int(w[2])
            steps.setdefault(step, []).append((w[4], int(w[5]), int(w[6]), int(w[7]), int(w[8]), int(w[9])))
    return steps, g_before


def part_starts(name: str, rec: Recording, keeps: list, who: np.ndarray, rows_of: dict) -> list:
    """Every start of a part of a request in one run, with what came before it, and whether the run stands alone there."""
    steps, g_before = read_steps(rec.path, keeps)
    by_file = {e["file"]: e for e in rec.index.values()}
    last_step_of_seq, seen_file, out = {}, {}, []
    order = sorted(steps)
    for k, step in enumerate(order):
        parts = steps[step]
        prev = steps[order[k - 1]] if k > 0 else []
        prev_seqs = [q[1] for q in prev]
        seqs = [q[1] for q in parts]
        # what the tool did between the two batches
        keep_taken = [q[1] for q in prev if q[0] == "-" and (q[3] + q[4]) in keeps]
        starts = [q[1] for q in parts if q[0] != "-" and q[0] not in seen_file]
        cuts_back = [q[1] for q in parts if q[0] != "-" and q[0] in seen_file and seen_file[q[0]][1]]
        for rank, (file, seq, idx0, p0, n, to_cut) in enumerate(parts):
            if file != "-":
                e = by_file[file]
                r0 = p0 - e["shared_len"]
                base = rows_of[file]
                lone = who[base + r0:base + r0 + 2]
                first = file not in seen_file
                after_cut = (not first) and seen_file[file][1]
                out.append({
                    "run": name, "file": file, "row": r0, "seq": seq, "step": step, "n": n, "rank": rank, "n_parts": len(parts),
                    "alone": bool((lone == name).any()), "unclear": bool((lone == "?").any()),
                    "kind": "request start" if first else ("after a cut" if after_cut else "plain"),
                    "absent": seq not in prev_seqs, "same_seqs": seqs == prev_seqs, "same_set": sorted(seqs) == sorted(prev_seqs),
                    "gap_steps": step - last_step_of_seq.get(seq, step), "g_between": g_before[step] - g_before.get(order[k - 1], 0) if k else 0,
                    "others_start": len([q for q in starts if q != seq]), "others_cut_back": len([q for q in cuts_back if q != seq]),
                    "keeps_taken": len([q for q in keep_taken if q != seq]), "warm_in_batch": any(q[0] == "-" for q in parts),
                    "warm_before": any(q[0] == "-" for q in prev), "idx0": idx0,
                })
        for file, seq, idx0, p0, n, to_cut in parts:
            last_step_of_seq[seq] = step
            if file != "-":
                seen_file[file] = (step, to_cut)
    return out


def rate(title: str, starts: list, key) -> None:
    groups = {}
    for st in starts:
        groups.setdefault(key(st), []).append(st)
    print(title)
    for k in sorted(groups, key=str):
        g = groups[k]
        print("  %-22s %4d starts, %3d stand alone (%5.1f%%), %d unclear" % (
            k, len(g), sum(x["alone"] for x in g), 100 * sum(x["alone"] for x in g) / len(g), sum(x["unclear"] for x in g)))


def row_facts(chunks: list, shared: int, rows: int) -> dict:
    """Per row of a request: the batch that read it."""
    f = {k: np.full(rows, -1, dtype=np.int64) for k in ("step", "g", "used", "idx", "n", "off", "left", "cut")}
    for step, g, used, idx0, p0, n, to_cut in chunks:
        r0 = p0 - shared
        if r0 < 0 or r0 + n > rows:
            raise SystemExit("a traced part [%d, %d) is outside the request's %d rows" % (r0, r0 + n, rows))
        sl = slice(r0, r0 + n)
        f["step"][sl], f["g"][sl], f["used"][sl], f["n"][sl], f["cut"][sl] = step, g, used, n, to_cut
        f["idx"][sl] = idx0 + np.arange(n)
        f["off"][sl] = np.arange(n)
        f["left"][sl] = n - 1 - np.arange(n)
    if (f["step"] < 0).any():
        raise SystemExit("the trace does not cover every row")
    return f


def gaps_by_tap(a: Recording, pa: str, b: Recording, pb: str, rows: int) -> np.ndarray:
    out = np.empty((rows, len(a.taps)), dtype=np.float64)
    step = max(1, (64 << 20) // a.row_bytes)
    for r0 in range(0, rows, step):
        n = min(step, rows - r0)
        x, y = a.state_rows(pa, r0, n), b.state_rows(pb, r0, n)
        scale = np.maximum(np.abs(x).max(axis=2), 1e-30)
        out[r0:r0 + n] = np.abs(x - y).max(axis=2) / scale
    return out


def table(title: str, keys: np.ndarray, edges: list, far: np.ndarray, gap: np.ndarray) -> None:
    print(title)
    lo = None
    for hi in edges + [None]:
        m = np.ones(keys.size, dtype=bool)
        if lo is not None:
            m &= keys > lo
        if hi is not None:
            m &= keys <= hi
        name = "%s..%s" % ("" if lo is None else lo + 1, "" if hi is None else hi)
        if m.any():
            print("  %-14s %7d rows, %6.2f%% far, median %.2e, 99%% %.2e, most %.2e" % (
                name, int(m.sum()), 100 * float(far[m].mean()), float(np.median(gap[m])), float(np.quantile(gap[m], 0.99)),
                float(gap[m].max())))
        lo = hi


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rec", required=True)
    p.add_argument("--nocuts", required=True)
    p.add_argument("--stops", required=True)
    p.add_argument("--far", type=float, default=0.02)
    args = p.parse_args()
    rec, plain, stops = Recording(args.rec), Recording(args.nocuts), Recording(args.stops)
    t_rec, t_plain, t_stops = read_trace(args.rec), read_trace(args.nocuts), read_trace(args.stops)

    cols = {}
    listing = []
    for rid, e in rec.index.items():
        pe, se = plain.index[rid], stops.index[rid]
        rows, shared, name = e["states_rows"], e["shared_len"], e["file"]
        test = gaps_by_tap(rec, rec.req(e, ".states"), stops, stops.req(se, ".states"), rows)
        noise = gaps_by_tap(plain, plain.req(pe, ".states"), stops, stops.req(se, ".states"), rows)
        both = gaps_by_tap(rec, rec.req(e, ".states"), plain, plain.req(pe, ".states"), rows)
        fr, fs, fp = (row_facts(t.get(name, []), shared, rows) for t in (t_rec, t_stops, t_plain))
        back = sorted(s["cut"] - shared for s in e["segments"] if s["n_states"] > 0 and s["cut"] < e["n_tokens"])
        since = np.full(rows, -1, dtype=np.int64)  # rows since the last cut gone back to; -1 before any
        for c in back:
            since[c:] = np.arange(rows - c)
        add = {"test": test.max(axis=1), "noise": noise.max(axis=1), "both": both.max(axis=1), "since": since,
               "tap0": test[:, 0], "row": np.arange(rows)}
        for tag, f in (("r", fr), ("s", fs), ("p", fp)):
            for k, v in f.items():
                add[tag + "_" + k] = v
        for k, v in add.items():
            cols.setdefault(k, []).append(v)
        far = add["test"] > args.far
        if far.any():
            idx = np.flatnonzero(far)
            runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
            for run in runs:
                a, b = int(run[0]), int(run[-1])
                listing.append("  %s rows %d..%d of %d (first cut gone back to: %s): most %.3f | with continuations: batch of %d, "
                               "part of %d, row %d of it, batch row %d, ends at a cut %d, %d generating steps before | read ended "
                               "at cuts: batch of %d, part of %d, row %d of it | whole shares: %.3f apart from it" % (
                                   name[:28], a, b, rows, back[0] if back else "none", float(add["test"][a:b + 1].max()),
                                   fr["used"][a], fr["n"][a], fr["off"][a], fr["idx"][a], fr["cut"][a], fr["g"][a],
                                   fs["used"][a], fs["n"][a], fs["off"][a], float(add["both"][a:b + 1].max())))
    c = {k: np.concatenate(v) for k, v in cols.items()}
    far = c["test"] > args.far

    # which run stands alone on a row: far from both others while they agree
    rs, ns, rn = c["test"] > args.far, c["noise"] > args.far, c["both"] > args.far
    who = np.full(far.size, "", dtype=object)
    who[rs & rn & ~ns] = "R"
    who[rs & ns & ~rn] = "S"
    who[ns & rn & ~rs] = "N"
    who[rs & ns & rn] = "?"
    print("rows where one run stands alone: with continuations %d, read ended at cuts %d, read in whole shares %d; all three "
          "apart %d" % (int((who == "R").sum()), int((who == "S").sum()), int((who == "N").sum()), int((who == "?").sum())))
    rows_of, base = {}, 0
    for rid, e in rec.index.items():
        rows_of[e["file"]] = base
        base += e["states_rows"]
    keeps = sorted({rec.meta["k_tools"], rec.meta["k_sys"]})
    starts = []
    for name, r in (("R", rec), ("S", stops), ("N", plain)):
        starts += part_starts(name, r, keeps, who, rows_of)
    rate("starts of a part where the run stands alone, by run", starts, lambda x: x["run"])
    rate("by run and what came before", starts, lambda x: x["run"] + " " + x["kind"])
    rate("by whether the sequence was missing from the batch before", starts, lambda x: x["absent"])
    rate("by batches since the sequence was last read", starts, lambda x: min(x["gap_steps"], 3))
    rate("by whether the batch has the sequences of the batch before, in order", starts, lambda x: x["same_seqs"])
    rate("by whether it has the same set of sequences", starts, lambda x: x["same_set"])
    rate("by generating steps between the two batches", starts, lambda x: min(x["g_between"], 2))
    rate("by other sequences starting a request in this batch", starts, lambda x: min(x["others_start"], 2))
    rate("by other sequences back from a cut in this batch", starts, lambda x: min(x["others_cut_back"], 2))
    rate("by keep points other sequences took in the batch before", starts, lambda x: min(x["keeps_taken"], 2))
    rate("by shared-prefix parts in this batch", starts, lambda x: x["warm_in_batch"])
    rate("by shared-prefix parts in the batch before", starts, lambda x: x["warm_before"])
    rate("by the part's place in the batch", starts, lambda x: min(x["rank"], 3))
    rate("by the parts in the batch", starts, lambda x: "1" if x["n_parts"] == 1 else ("2..4" if x["n_parts"] <= 4 else "5.."))
    rate("by the length of the part", starts, lambda x: "..2" if x["n"] <= 2 else ("3..16" if x["n"] <= 16 else "17.."))
    print("the starts that stand alone:")
    for x in starts:
        if x["alone"]:
            print("  %s %s row %d: %s, step %d, part %d of %d, %d rows, sequence %d, missing before %s, %d batches since, same "
                  "sequences %s, %d generating steps between, others starting %d, others back from a cut %d, keeps taken %d" % (
                      x["run"], x["file"][:10], x["row"], x["kind"], x["step"], x["rank"], x["n_parts"], x["n"], x["seq"],
                      x["absent"], x["gap_steps"], x["same_seqs"], x["g_between"], x["others_start"], x["others_cut_back"],
                      x["keeps_taken"]))
    print("%d rows; %d (%.2f%%) are more than %.2f apart between the run with continuations and the read ended at the same "
          "cuts; %d between the two reads without continuations; %d between the run with continuations and the read in whole "
          "shares" % (far.size, int(far.sum()), 100 * float(far.mean()), args.far, int((c["noise"] > args.far).sum()),
                      int((c["both"] > args.far).sum())))
    print("the first recorded layer differs on %d rows (it is the token's own row of the table, so it should on none)" % int(
        (c["tap0"] > 0).sum()))
    table("by the tokens in the batch, in the run with continuations", c["r_used"], [23, 64, 256, 1024], far, c["test"])
    table("by the tokens in the batch, in the read ended at cuts", c["s_used"], [23, 64, 256, 1024], far, c["test"])
    table("by the length of the row's own part of the batch, with continuations", c["r_n"], [1, 4, 16, 64, 256], far, c["test"])
    table("by the length of the row's own part of the batch, read ended at cuts", c["s_n"], [1, 4, 16, 64, 256], far, c["test"])
    table("by the row's place in its part, with continuations", c["r_off"], [0, 3, 15, 63, 255], far, c["test"])
    table("by the rows after it in its part, with continuations", c["r_left"], [0, 3, 15, 63, 255], far, c["test"])
    table("by the row of the batch, with continuations", c["r_idx"], [22, 255, 511, 1023], far, c["test"])
    table("by whether its part ended at a cut, with continuations (0 no, 1 yes)", c["r_cut"], [0], far, c["test"])
    table("by generating steps before its batch, with continuations", c["r_g"], [0, 10, 100, 500], far, c["test"])
    table("by rows since the last cut gone back to (-1: before any)", c["since"], [-1, 15, 63, 255, 1023], far, c["test"])
    table("the two reads without continuations, by the tokens in the batch of the read in whole shares", c["p_used"],
          [23, 64, 256, 1024], c["noise"] > args.far, c["noise"])
    print("runs of far rows (%d), the first 12:" % len(listing))
    for line in listing[:12]:
        print(line)


if __name__ == "__main__":
    main()
