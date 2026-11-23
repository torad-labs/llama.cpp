#!/usr/bin/env python3
# evalwhy.py LOG [FROM_LINE [TO_LINE]]: every target eval of a traced log (serve.py --trace-decode) by its rows and by
# what the engine did for it: the target context's graph reused or built (kept in a slot, or on the main scheduler),
# and on the cards how many subgraph launches replayed a captured CUDA graph, ran node by node (direct) or were
# captured in it. Medians of the eval's time; the launches are means an eval, summed over the cards.
import re, sys, statistics as st, collections
log = sys.argv[1]; lo = int(sys.argv[2]) if len(sys.argv) > 2 else 0; hi = int(sys.argv[3]) if len(sys.argv) > 3 else 1 << 60
te = re.compile(r'GLM53_TRACE event=(begin|end) .*?host_ns=(\d+) .*?stage=target_eval .*?rows=(\d+)')
gl = re.compile(r'graph_trace layer=llama event=(build|reuse) .*?phase=(\w+) rows=(\d+) seqs=(\d+) .*?slots_allowed=(\d+)')
gc = re.compile(r'graph_trace layer=cuda event=(direct|replay|capture|instantiate)')
gd = re.compile(r'graph_trace layer=cuda event=direct .*?ggml_nodes=(\d+) reason=(\w+)')
cur = None; out = collections.defaultdict(list); why = collections.Counter()
for n, line in enumerate(open(log, errors='replace'), 1):
    if n < lo: continue
    if n > hi: break
    m = te.search(line)
    if m:
        if m[1] == 'begin': cur = {'t0': int(m[2]), 'rows': int(m[3]), 'graphs': [], 'direct': 0, 'replay': 0, 'capture': 0, 'instantiate': 0, 'dnodes': 0}
        elif cur:
            ms = (int(m[2]) - cur['t0'])/1e6; g = cur['graphs']
            kind = 'no graph line' if not g else 'reused' if all(e == 'reuse' for e, _ in g) else 'built, main sched' if any(e == 'build' and a == 0 for e, a in g) else 'built, kept'
            r = cur['rows']; band = '1-23' if r <= 23 else '24-69' if r <= 69 else '70-255' if r < 256 else '256-1023' if r < 1024 else '1024+'
            out[(band, kind, len(g))].append((ms, r, cur['direct'], cur['replay'], cur['capture'], cur['dnodes'])); cur = None
        continue
    if cur is None: continue
    m = gl.search(line)
    if m and m[2] == 'target': cur['graphs'].append((m[1], int(m[5]))); continue
    m = gc.search(line)
    if m:
        cur[m[1]] += 1
        d = gd.search(line)
        if d: cur['dnodes'] += int(d[1]); why[d[2]] += 1
tot = sum(x[0] for v in out.values() for x in v)
print('%s: %d target evals, %.1f s' % (log, sum(len(v) for v in out.values()), tot/1e3))
print('  rows      what                 passes   n   median ms  mean ms  rows  direct  (nodes)  replay capture  %% of eval time')
order = ['1-23', '24-69', '70-255', '256-1023', '1024+']
for k in sorted(out, key=lambda k: (order.index(k[0]), k[1], k[2])):
    v = out[k]; f = lambda i: st.mean(x[i] for x in v)
    print('  %-9s %-20s %4d %5d %9.1f %8.1f %5.0f %7.0f %8.0f %7.0f %7.0f %8.1f' % (k[0], k[1], k[2], len(v), st.median(x[0] for x in v), f(0), f(1), f(2), f(5), f(3), f(4), 100*sum(x[0] for x in v)/tot))
print('  direct launches by reason:', dict(why.most_common(6)))
