"""Where a server iteration's time goes, from the marks the engine prints when LLAMA_DECODE_TRACE=1.

  trace_split.py SERVER_LOG OUT_JSON

An iteration that only writes (no prompt rows, one target evaluation) is cut at the engine's marks into
  before   the iteration's start to the target evaluation's start: each sequence's draft state is copied and this
           round's drafts are made here, and work the draft's card still owed from the round before is waited for
  target   the target evaluation, to the end of its work on the cards
  feed     the target evaluation's end to the first sequence's sampling: the target's rows enter the draft's cache here
  sample   the sequences' rows sampled one after another on the host, and the drafts kept or set aside
  between  the iteration's end to the next one's start
and the iterations are grouped by the sequences in their batch. A line gives each part's median in ms.

The engine marks a sequence when its rows are sampled, so the time from one sequence's mark to the next one's is that
sequence's sampling: it gives the time a sampled row costs, and with it the first sequence's sampling is taken out of
"feed" and counted in "sample".

At log level 4 the draft code also prints, at each reply's end, the calls it has made to draft and the time they took
in all. Two such lines apart give the time of one call to draft between them; it is part of "before".
"""
from __future__ import annotations

import json
import os
import re
import statistics
import sys

MARK = re.compile(r"GLM53_TRACE event=(begin|end|mark) \S+ host_ns=(\d+) \S+ \S+ (stage=\S+.*)$")
STATS = re.compile(r"statistics\s+(\S+): #calls\(b,g,a\) =\s*(\d+)\s+(\d+)\s+(\d+),.*dur\(b,g,a\) = ([0-9.]+), ([0-9.]+), ([0-9.]+) ms")
PARTS = ("before", "target", "feed", "sample", "between")
ROW_EDGES = (1, 4, 8, 16, 32, 64, 128, 256)


def fields(label: str) -> dict:
    return dict(piece.split("=", 1) for piece in label.split() if "=" in piece)


def read(path: str):
    """The iterations with a target evaluation, in order, and the draft code's totals with the iteration each followed."""
    done, totals, cur = [], [], None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = STATS.search(line)
            if m:
                totals.append((len(done), m.group(1), int(m.group(3)), float(m.group(6))))
                continue
            m = MARK.search(line.rstrip("\n"))
            if not m:
                continue
            event, ns, got = m.group(1), int(m.group(2)), fields(m.group(3))
            stage = got.get("stage")
            if stage == "iteration":
                if event == "begin":
                    if done and done[-1]["between"] is None:
                        done[-1]["between"] = (ns - done[-1]["end"]) / 1e6
                    cur = {"start": ns, "evals": [], "marks": [], "seqs": 0, "prompt_rows": 0, "kept": 0, "offered": 0}
                elif cur is not None:
                    cur["end"], cur["between"] = ns, None
                    if cur["evals"]:
                        done.append(cur)
                    cur = None
            elif cur is None:
                continue
            elif stage == "target_eval":
                if event == "begin":
                    cur["evals"].append([ns, None, int(got.get("rows", 0))])
                elif cur["evals"]:
                    cur["evals"][-1][1] = ns
            elif stage == "batch":
                cur["seqs"] += 1
                cur["prompt_rows"] += int(got.get("prompt_rows", 0))
            elif stage == "accept_candidate":
                # the sequence's rows are sampled by now: one more than it kept, the last one being the target's own token
                cur["marks"].append([ns, None, int(got.get("accepted", 0)) + 1])
            elif stage == "accept":
                if cur["marks"]:
                    cur["marks"][-1][1] = ns
                cur["kept"] += int(got.get("accepted", 0))
                cur["offered"] += int(got.get("drafted", 0))
    return done, totals


def middle(values) -> float:
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else 0.0


def row_ms(done: list) -> float:
    """What sampling one row costs: the time from a sequence's last mark to the next sequence's first, a row sampled."""
    per = []
    for it in done:
        for (_, end, _), (begin, _, rows) in zip(it["marks"], it["marks"][1:]):
            if end is not None:
                per.append((begin - end) / 1e6 / rows)
    return middle(per)


def cut(it: dict, per_row: float):
    """A writing iteration's parts in ms, or None when it read a prompt, evaluated twice or lacks a mark."""
    if len(it["evals"]) != 1 or it["prompt_rows"] or not it["marks"] or it["evals"][0][1] is None:
        return None
    b, e, rows = it["evals"][0]
    first = min(per_row * it["marks"][0][2], (it["marks"][0][0] - e) / 1e6)
    return {"seqs": it["seqs"], "rows": rows, "kept": it["kept"], "offered": it["offered"],
            "sampled": sum(m[2] for m in it["marks"]),
            "before": (b - it["start"]) / 1e6, "target": (e - b) / 1e6, "feed": (it["marks"][0][0] - e) / 1e6 - first,
            "sample": (it["end"] - it["marks"][0][0]) / 1e6 + first,
            "between": it["between"] if it["between"] is not None and it["between"] < 1000 else None}


def line(label: str, rows: list) -> dict:
    out = {"iterations": len(rows), "rows": middle(r["rows"] for r in rows), "seqs": middle(r["seqs"] for r in rows),
           "kept": sum(r["kept"] for r in rows), "offered": sum(r["offered"] for r in rows),
           "written": sum(r["seqs"] + r["kept"] for r in rows), "sampled": middle(r["sampled"] for r in rows)}
    for part in PARTS:
        out[part] = middle(r[part] for r in rows)
    out["all"] = sum(out[part] for part in PARTS)
    a_seq = out["written"] / max(1, sum(r["seqs"] for r in rows))
    print("%-14s %5d iterations  %4.0f rows  before %6.1f  target %6.1f  feed %5.1f  sample %5.1f (%3.0f rows)  between %4.1f"
          "  in all %6.1f ms  %.2f tokens a sequence an iteration, %5.1f tok/s a sequence"
          % (label, out["iterations"], out["rows"], out["before"], out["target"], out["feed"], out["sample"], out["sampled"],
             out["between"], out["all"], a_seq, 1000.0 * a_seq / out["all"] if out["all"] else 0.0), flush=True)
    return out


def main() -> None:
    log, out_path = sys.argv[1], sys.argv[2]
    done, totals = read(log)
    per_row = row_ms(done)
    cuts = [cut(it, per_row) for it in done]
    writing = [c for c in cuts if c]
    print("trace: %d iterations with a target evaluation, %d of them only writing; a sampled row costs %.2f ms"
          % (len(done), len(writing), per_row), flush=True)
    result = {"iterations": len(done), "writing": len(writing), "sampled_row_ms": per_row, "by_sequences": {}, "by_rows": {},
              "draft_calls": {}}
    if not writing:
        raise SystemExit("trace: no iteration that only writes was found in %s" % log)

    print("by the sequences in the batch:")
    for n in sorted({c["seqs"] for c in writing}, reverse=True):
        rows = [c for c in writing if c["seqs"] == n]
        if len(rows) >= 8:
            result["by_sequences"][str(n)] = line("%d sequences" % n, rows)

    print("the target evaluation by the rows in the batch:")
    low = 0
    for edge in ROW_EDGES + (10 ** 9,):
        rows = [c for c in writing if low < c["rows"] <= edge]
        if rows:
            label = "%d to %d rows" % (low + 1, edge) if edge < 10 ** 9 else "over %d rows" % low
            entry = {"iterations": len(rows), "rows": middle(c["rows"] for c in rows), "seqs": middle(c["seqs"] for c in rows),
                     "target": middle(c["target"] for c in rows)}
            result["by_rows"][label] = entry
            print("%-22s %5d iterations  %4.0f rows  %4.0f sequences  target %6.1f ms"
                  % (label, entry["iterations"], entry["rows"], entry["seqs"], entry["target"]), flush=True)
        low = edge

    # one call to draft, from two of the draft code's totals: the iterations between them say how many sequences wrote
    spans = {}
    for (i0, name0, calls0, ms0), (i1, name1, calls1, ms1) in zip(totals, totals[1:]):
        between = [c for c in cuts[i0:i1] if c]
        if name0 != name1 or calls1 <= calls0 or not between:
            continue
        n = int(middle(c["seqs"] for c in between))
        calls, ms = spans.get(n, (0, 0.0))
        spans[n] = (calls + calls1 - calls0, ms + ms1 - ms0)
    if spans:
        print("one call to draft, by the sequences writing (the middle one of the iterations between two totals):")
        for n in sorted(spans, reverse=True):
            calls, ms = spans[n]
            if calls >= 8:
                result["draft_calls"][str(n)] = {"calls": calls, "ms": ms / calls}
                print("%-22s %5d calls  %6.1f ms a call" % ("%d sequences" % n, calls, ms / calls), flush=True)
    else:
        print("trace: no totals from the draft code (they are printed at log level 4)")

    with open(out_path + ".new", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    os.replace(out_path + ".new", out_path)


if __name__ == "__main__":
    main()
