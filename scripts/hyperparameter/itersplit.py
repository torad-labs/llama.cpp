#!/usr/bin/env python3
# itersplit.py LOG [FROM_LINE [TO_LINE]]: where the server loop's time is over a stretch of a traced log, in seconds:
# the target evals, the head's ranges (proposal, proposal_sample, catchup), what is left before the first target eval
# of an iteration and after its last, and the time between iterations. Then the "after" part's distribution and the
# stage marks seen in the 40 longest, to name what a long one holds.
import re, sys, statistics as st, collections
log = sys.argv[1]; lo = int(sys.argv[2]) if len(sys.argv) > 2 else 0; hi = int(sys.argv[3]) if len(sys.argv) > 3 else 1 << 60
ev = re.compile(r'GLM53_TRACE event=(begin|end|mark) .*?host_ns=(\d+) .*?stage=(\w+)')
tot = collections.Counter(); it = None; last_end = None; afters = []; open_at = {}
for n, line in enumerate(open(log, errors='replace'), 1):
    if n < lo: continue
    if n > hi: break
    m = ev.search(line)
    if not m:
        if it is not None and it.get('te_end') and ('checkpoint' in line or 'prompt cache' in line or 'save' in line.lower()[:200] and 'slot' in line):
            it['notes'].append(line[16:150].strip())
        continue
    e, t, s = m[1], int(m[2]), m[3]
    if s == 'iteration':
        if e == 'begin':
            if last_end is not None: tot['between iterations'] += t - last_end
            it = {'t0': t, 'te0': None, 'te_end': None, 'in': collections.Counter(), 'marks': collections.Counter(), 'notes': []}
        elif it is not None:
            d = t - it['t0']; tot['iteration'] += d
            for k, v in it['in'].items(): tot[k] += v
            ranges = sum(it['in'].values())
            if it['te0'] is None: tot['no target eval: outside ranges'] += d - ranges
            else:
                before = it['te0'] - it['t0'] - sum(v for k, v in it['in'].items() if k in ('proposal', 'proposal_sample') ) - it.get('catch_before', 0)
                after = t - it['te_end'] - it.get('catch_after', 0)
                tot['before the first target eval, outside ranges'] += max(before, 0); tot['after the last target eval, outside ranges'] += after
                afters.append((after/1e6, n, dict(it['marks']), it['notes'][:3]))
            last_end = t; it = None
        continue
    if it is None: continue
    if e == 'mark': 
        if it['te_end']: it['marks'][s] += 1
        continue
    if e == 'begin': open_at[s] = t
    elif s in open_at:
        d = t - open_at.pop(s); it['in'][s] += d
        if s == 'target_eval':
            if it['te0'] is None: it['te0'] = t - d
            it['te_end'] = t
        elif s == 'catchup': it['catch_after' if it['te_end'] else 'catch_before'] = it.get('catch_after' if it['te_end'] else 'catch_before', 0) + d
whole = tot['iteration'] + tot['between iterations']
print('%s: %.1f s of server loop' % (log, whole/1e9))
for k, v in tot.most_common():
    if k != 'iteration': print('  %-48s %7.1f s  %5.1f %%' % (k, v/1e9, 100*v/whole))
a = sorted(x[0] for x in afters)
if a:
    q = lambda p: a[min(len(a)-1, int(p*len(a)))]
    print('  after the last target eval: n %d  median %.2f  p90 %.2f  p99 %.1f  max %.1f ms; over 20 ms: %d iterations holding %.1f s of its %.1f s'
          % (len(a), q(.5), q(.9), q(.99), a[-1], sum(1 for x in a if x > 20), sum(x for x in a if x > 20)/1e3, sum(a)/1e3))
    for ms, n, marks, notes in sorted(afters, reverse=True)[:6]:
        print('    %.0f ms at line %d  marks %s  %s' % (ms, n, marks, ' | '.join(notes)[:200]))
