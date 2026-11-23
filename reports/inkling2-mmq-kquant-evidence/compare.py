#!/usr/bin/env python3
"""Compare two mmq-microperf result files: per-case us_mean ratio and the sums per group."""
import re
import sys
from collections import OrderedDict

def load(path):
    cases = OrderedDict()
    for line in open(path):
        m = re.match(r"RESULT type=(\S+) m=(\d+) k=(\d+) tokens=(\d+) bcast=(\d+) J=(\d+) us_mean=([\d.]+) us_min=(\d+)", line)
        if m:
            key = (m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5)), int(m.group(6)))
            cases[key] = (float(m.group(7)), int(m.group(8)))
        r = re.match(r"RESULT type=(\S+) m=(\d+) k=(\d+) tokens=(\d+) bcast=(\d+) J=(\d+) compare=(\S+) nmse=(\S+)", line)
        if r:
            key = (r.group(1), int(r.group(2)), int(r.group(3)), int(r.group(4)), int(r.group(5)), int(r.group(6)))
            cases[key] = (r.group(7), float(r.group(8)))
    return cases

base = load(sys.argv[1])
cand = load(sys.argv[2])
print(f"{'case':52s} {'base':>10s} {'cand':>10s} {'cand/base':>9s}")
tot_b = tot_c = 0.0
for key, bv in base.items():
    if key not in cand:
        continue
    cv = cand[key]
    name = f"{key[0]} m={key[1]} k={key[2]} tok={key[3]} b={key[4]} J={key[5]}"
    if isinstance(bv[0], float) and isinstance(cv[0], float):
        ratio = cv[0] / bv[0] if bv[0] and cv[0] else float('nan')
        print(f"{name:52s} {bv[0]:10.2f} {cv[0]:10.2f} {ratio:9.3f}")
        tot_b += bv[0]
        tot_c += cv[0]
    else:
        print(f"{name:52s} {str(bv[0]):>10s} {str(cv[0]):>10s} nmse {cv[1]:.2e}")
if tot_b:
    print(f"{'TOTAL (perf rows)':52s} {tot_b:10.2f} {tot_c:10.2f} {tot_c/tot_b:9.3f}")
