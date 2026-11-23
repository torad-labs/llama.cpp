#!/usr/bin/env python3
# step-passes.py LOG [ROWS] [FIRST-LAST]...: the ROWS-row steps (iterations with one target eval of ROWS reply rows and
# no prompt row) pass by pass: how many, the iteration's median and mean, the target eval's median and mean, ms; then
# the same over each range of passes given (1-5) and over all. A pass is passbudget.py's: it starts at an iteration
# with prompt rows after 20 or more without. The population is printed with every line, because a log that holds
# other traffic after its rate passes (agents.py on the same server) has 69-row steps of another kind.
import re, sys, statistics as st
log = sys.argv[1]; want = int(sys.argv[2]) if len(sys.argv) > 2 else 69
spans = [tuple(map(int, x.split('-'))) for x in sys.argv[3:]]
ev = re.compile(r'GLM53_TRACE event=(begin|end|mark) .*?host_ns=(\d+) .*?stage=(\w+)(.*)')
its = []; cur = None
with open(log, errors='replace') as f:
    for line in f:
        m = ev.search(line)
        if not m: continue
        kind, ns, stage, rest = m[1], int(m[2]), m[3], m[4]
        if stage == 'iteration':
            if kind == 'begin': cur = {'b': ns, 'rows': 0, 'prows': 0, 'te': []}
            elif cur is not None: cur['e'] = ns; its.append(cur); cur = None
        elif cur is None: continue
        elif kind == 'mark' and stage == 'batch':
            r = re.search(r'rows=(\d+) prompt_rows=(\d+)', rest)
            if r: cur['rows'] += int(r[1]); cur['prows'] += int(r[2])
        elif stage == 'target_eval' and kind == 'begin':
            r = re.search(r'rows=(\d+)', rest); cur['te'].append([ns, None, int(r[1]) if r else 0])
        elif stage == 'target_eval' and kind == 'end' and cur['te'] and cur['te'][-1][1] is None:
            cur['te'][-1][1] = ns
its = [it for it in its if it['rows']]
passes = []; quiet = 99
for it in its:
    pm = it['prows'] > 0
    if pm and quiet >= 20: passes.append([])
    quiet = 0 if pm else quiet + 1
    if passes: passes[-1].append(it)
def line(name, sel):
    s = [it for it in sel if not it['prows'] and len(it['te']) == 1 and it['te'][0][1] and it['te'][0][2] == want]
    if not s: print(f"  {name}: no {want}-row steps"); return
    i = [(it['e'] - it['b'])/1e6 for it in s]; t = [(it['te'][0][1] - it['te'][0][0])/1e6 for it in s]
    print(f"  {name}: {len(s)} steps of {want} rows; iteration median {st.median(i):.2f} mean {st.mean(i):.2f}; "
          f"target eval median {st.median(t):.2f} mean {st.mean(t):.2f}")
print(f"{log}: {len(its)} iterations with rows, {len(passes)} passes")
for n, p in enumerate(passes, 1): line(f"pass {n}", p)
for a, b in spans: line(f"passes {a} to {b}", [it for p in passes[a-1:b] for it in p])
line("every pass", [it for p in passes for it in p])
