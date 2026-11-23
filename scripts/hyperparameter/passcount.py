# passcount.py LOG [FROM_LINE]: each target eval that carries prompt rows: its passes (graph builds and reuses of the
# target context between the eval's begin and end), by rows and sequences, and the eval's time.
import re, sys, collections, statistics as st
log = sys.argv[1]; frm = int(sys.argv[2]) if len(sys.argv) > 2 else 0
ev = re.compile(r'GLM53_TRACE event=(begin|end|mark) .*?host_ns=(\d+) .*?stage=(\w+)(.*)')
gt = re.compile(r'graph_trace layer=llama event=(build|reuse) .*?phase=(\w+) rows=(\d+) seqs=(\d+)')
evals = []; cur = None
with open(log, errors='replace') as f:
    for n, line in enumerate(f, 1):
        if n < frm: continue
        m = ev.search(line)
        if m:
            kind, ns, stage, rest = m[1], int(m[2]), m[3], m[4]
            if stage == 'target_eval':
                if kind == 'begin': cur = {'b': ns, 'rows': int(re.search(r'rows=(\d+)', rest)[1]), 'p': [], 'prows': 0, 'lens': []}
                elif cur is not None: cur['e'] = ns; evals.append(cur); cur = None
            elif stage == 'batch' and kind == 'mark' and cur is not None:
                r = re.search(r'rows=(\d+) prompt_rows=(\d+)', rest)
                cur['prows'] += int(r[2]); cur['lens'].append(int(r[1]))
            continue
        if cur is None: continue
        g = gt.search(line)
        if g and g[2] == 'target': cur['p'].append((g[1], int(g[3]), int(g[4])))
mixed = [e for e in evals if e['prows'] > 0]
print(f"{log}: {len(evals)} target evals, {len(mixed)} with prompt rows")
tot_t = sum(e['e'] - e['b'] for e in mixed)/1e9
np_ = [len(e['p']) for e in mixed]
single = sum(1 for e in mixed for p in e['p'] if p[2] == 1); allp = sum(np_)
print(f"  their time {tot_t:.1f} s; passes {allp} (mean {st.mean(np_):.1f} an eval, max {max(np_)}); passes of one sequence {single} ({100*single/allp:.0f} %)")
dl = collections.Counter(len(set(e['lens'])) for e in mixed)
print("  distinct row counts among an eval's sequences -> evals:", dict(sorted(dl.items())))
# what the same rows would take as passes if sequences of one length went together: the distinct lengths left after the shortest
ideal = sum(len(set(e['lens'])) for e in mixed)
print(f"  one pass a distinct length would be {ideal} passes ({ideal/len(mixed):.1f} an eval)")
by = collections.defaultdict(lambda: [0, 0.0])
for e in mixed:
    k = min(len(e['p']), 15); by[k][0] += 1; by[k][1] += (e['e'] - e['b'])/1e6
print("  passes an eval -> evals, mean ms:", ', '.join(f"{k}{'+' if k == 15 else ''}: {v[0]} x {v[1]/v[0]:.0f}" for k, v in sorted(by.items())))
