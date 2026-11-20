#!/usr/bin/env python3
"""Where a captured window's time goes on each card: busy and idle time, the gaps, and the kernels, copies and waits."""
import bisect
import collections
import re
import sqlite3
import sys

db = sqlite3.connect(sys.argv[1])
tables = {r[0] for r in db.execute("select name from sqlite_master where type='table'")}
K, M, S, R, N = ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION",
                 "CUPTI_ACTIVITY_KIND_RUNTIME", "NVTX_EVENTS")


def cols(table):
    return {r[1] for r in db.execute(f"pragma table_info({table})")}


def section(title, work):
    print("\n== " + title)
    try:
        work()
    except Exception as error:  # a schema this script does not know still leaves the other sections
        print("  failed:", type(error).__name__, error)


def cards():
    t0, t1, n = db.execute(f"select min(start), max(end), count(*) from {K}").fetchone()
    print(f"  window {(t1 - t0) / 1e6:.0f} ms, {n} kernels")
    # A launched graph is traced whole, as one span of a table of its own: card time beside the kernels and copies
    work = [t for t in (K, M, "CUPTI_ACTIVITY_KIND_GRAPH_TRACE") if t in tables and {"start", "end", "deviceId"} <= cols(t)]
    print("  card time from:", ", ".join(f"{t[20:].lower()} {db.execute(f'select count(*) from {t}').fetchone()[0]}" for t in work))
    for (dev,) in db.execute(f"select distinct deviceId from {K} order by 1").fetchall():
        spans = db.execute(" union all ".join(f"select start, end from {t} where deviceId=?" for t in work) + " order by 1",
                           (dev,) * len(work)).fetchall()
        busy, gaps, edge = 0, [], None
        for start, end in spans:
            if edge is None:
                edge, begin = end, start
            elif start > edge:
                busy += edge - begin
                gaps.append(start - edge)
                begin, edge = start, end
            else:
                edge = max(edge, end)
        busy += edge - begin
        total = (t1 - t0)
        bands = [(0, 2e4, "<20us"), (2e4, 1e5, "20-100us"), (1e5, 1e6, "0.1-1ms"), (1e6, 1e7, "1-10ms"), (1e7, 1e18, ">10ms")]
        text = ", ".join(f"{name} {sum(1 for g in gaps if lo <= g < hi)}x {sum(g for g in gaps if lo <= g < hi) / 1e6:.0f}ms"
                         for lo, hi, name in bands)
        print(f"  card {dev}: {len(spans)} kernels, copies and graphs, busy {busy / 1e6:.0f} ms ({100 * busy / total:.1f}%), "
              f"idle {(total - busy) / 1e6:.0f} ms; gaps {text}")


def kernels(column, limit, width):
    def work():
        t0, t1 = db.execute(f"select min(start), max(end) from {K}").fetchone()
        rows = db.execute(f"select s.value, count(*), sum(k.end - k.start), max(k.end - k.start), count(distinct k.deviceId) "
                          f"from {K} k join StringIds s on s.id = k.{column} group by k.{column} order by 3 desc limit {limit}")
        print("  total ms |  share of window x cards | count | mean us | max us | name")
        ncards = db.execute(f"select count(distinct deviceId) from {K}").fetchone()[0]
        for name, n, total, longest, _ in rows:
            print(f"  {total / 1e6:8.1f} | {100 * total / ((t1 - t0) * ncards):5.1f}% | {n:7d} | {total / n / 1e3:8.1f} | "
                  f"{longest / 1e3:8.1f} | {name[:width]}")
    return work


def copies():
    have = cols(M)
    kind = "copyKind" if "copyKind" in have else "0"
    for row in db.execute(f"select deviceId, {kind}, count(*), sum(bytes), sum(end - start), max(end - start) from {M} "
                          "group by 1, 2 order by 5 desc limit 24"):
        print(f"  card {row[0]} kind {row[1]}: {row[2]} copies, {row[3] / 1e6:.1f} MB, {row[4] / 1e6:.1f} ms, longest {row[5] / 1e3:.0f} us")
    if "ENUM_CUDA_MEMCPY_OPER" in tables:
        print("  kinds:", dict(db.execute("select id, name from ENUM_CUDA_MEMCPY_OPER").fetchall()))


def waits():
    have = cols(S)
    kind = "syncType" if "syncType" in have else "0"
    for row in db.execute(f"select deviceId, {kind}, count(*), sum(end - start), max(end - start) from {S} group by 1, 2 "
                          "order by 4 desc limit 16"):
        print(f"  card {row[0]} type {row[1]}: {row[2]} waits, {row[3] / 1e6:.1f} ms, longest {row[4] / 1e3:.0f} us")
    if "ENUM_CUPTI_SYNC_TYPE" in tables:
        print("  types:", dict(db.execute("select id, name from ENUM_CUPTI_SYNC_TYPE").fetchall()))


def calls():
    for name, n, total, longest in db.execute(f"select s.value, count(*), sum(r.end - r.start), max(r.end - r.start) from {R} r "
                                              "join StringIds s on s.id = r.nameId group by r.nameId order by 3 desc limit 22"):
        print(f"  {total / 1e6:9.1f} ms | {n:8d} | mean {total / n / 1e3:8.1f} us | max {longest / 1e3:9.1f} us | {name[:60]}")


def kind(text):
    """A node's name with its layer and sizes taken out, so the layers of a model fall into one row."""
    return re.sub(r"\d+", "#", re.sub(r" [rw]=\d+", "", str(text).replace(" w0=", " weights=")))


def ranges():
    have = cols(N)
    name = "coalesce(n.text, s.value)" if "textId" in have else "n.text"
    join = "left join StringIds s on s.id = n.textId" if "textId" in have else ""
    totals = collections.defaultdict(lambda: [0, 0])
    for text, n, total in db.execute(f"select {name}, count(*), sum(n.end - n.start) from {N} n {join} where n.end is not null "
                                     "group by 1"):
        totals[kind(text)][0] += total
        totals[kind(text)][1] += n
    print("  host ms inside the range | count | mean us | range")
    for text, (total, n) in sorted(totals.items(), key=lambda item: -item[1][0])[:60]:
        print(f"  {total / 1e6:9.1f} | {n:8d} | {total / n / 1e3:8.1f} | {text[:140]}")
    marks = db.execute(f"select count(*) from {N} where end is null").fetchone()[0]
    print(f"  {marks} marks")


def nodes():
    """Card time by the graph node that launched it: a kernel belongs to the innermost range its launch call began in."""
    have = cols(N)
    name = "coalesce(n.text, s.value)" if "textId" in have else "n.text"
    join = "left join StringIds s on s.id = n.textId" if "textId" in have else ""
    spans = collections.defaultdict(list)
    for tid, start, end, text in db.execute(f"select n.globalTid, n.start, n.end, {name} from {N} n {join} "
                                            "where n.end is not null order by n.start, n.end desc"):
        spans[tid].append((start, end, str(text)))
    begins, parents = {}, {}
    for tid, rows in spans.items():
        begins[tid] = [row[0] for row in rows]
        parent, stack = [], []
        for i, (start, end, _) in enumerate(rows):
            while stack and rows[stack[-1]][1] < start:
                stack.pop()
            parent.append(stack[-1] if stack else -1)
            stack.append(i)
        parents[tid] = parent
    calls = {corr: (start, tid) for corr, start, tid in db.execute(f"select correlationId, start, globalTid from {R}")}
    totals = collections.defaultdict(lambda: [0, 0, 0, 0, set()])
    size = re.compile(r" ([rw])=(\d+)")
    seen = set()
    for corr, start, end in db.execute(f"select correlationId, start, end from {K}"):
        text = "(no launch call in the window)"
        call = calls.get(corr)
        if call and call[1] in spans:
            rows, i = spans[call[1]], bisect.bisect_right(begins[call[1]], call[0]) - 1
            while i >= 0 and rows[i][1] < call[0]:
                i = parents[call[1]][i]
            text = rows[i][2] if i >= 0 else "(outside every range)"
            key = (call[1], i)
        else:
            key = None
        sizes = dict(size.findall(text))
        label = kind(text)
        row = totals[label]
        row[0] += end - start
        row[1] += 1
        if key is not None and key not in seen:
            seen.add(key)
            row[2] += int(sizes.get("r", 0))
            row[3] += int(sizes.get("w", 0))
            row[4].add(key)
    whole = sum(row[0] for row in totals.values())
    print(f"  {len(totals)} kinds of node, {whole / 1e6:.0f} ms of card time in all")
    print("  card ms | share | kernels | launches | read MB | written MB | node")
    for label, row in sorted(totals.items(), key=lambda item: -item[1][0])[:90]:
        print(f"  {row[0] / 1e6:8.1f} | {100 * row[0] / whole:5.1f}% | {row[1]:8d} | {len(row[4]):7d} | {row[2] / 1e6:9.1f} | "
              f"{row[3] / 1e6:9.1f} | {label[:150]}")


print("tables:", " ".join(sorted(t for t in tables if t.startswith(("CUPTI", "NVTX")))))
section("cards: busy, idle and the gaps between kernels", cards)
section("kernels by total time, short names", kernels("shortName", 45, 90))
section("kernels by total time, full names", kernels("demangledName", 45, 230))
if M in tables:
    section("copies", copies)
if S in tables:
    section("waits on the cards", waits)
if R in tables:
    section("host calls into CUDA", calls)
if N in tables:
    section("named ranges", ranges)
if N in tables and R in tables:
    section("card time by graph node", nodes)

