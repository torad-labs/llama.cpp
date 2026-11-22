#!/usr/bin/env python3
"""q8dense-census.py TRACE.sqlite [--rows 69] [--stage target_eval] [--skip 5] [--outlier 1.5] [--header H.gguf]...
[--json OUT.json]: the dense Q8_0 products of a decode stage on one rank, from an Nsight capture's SQLite export.

What a product is. A MUL_MAT node whose launch ran mul_mat_q<Q8_0, J> (the routed experts are MUL_MAT_ID and are
not here). The capture must be made with GGML_CUDA_NVTX=1 and started before the graphs were captured: a replayed
kernel carries its graph node, the node chains through its clones to the node the capture recorded, and that node's
creation lies inside the host range of the op that launched it (census.py's rule). An op's range is labelled
"MUL_MAT name r=R w=W w0=weights" (nvtx.cuh): R the bytes of its sources, W the bytes of its result. Nodes a launch
consumed beside its own are marked "fused ..." inside the range, with their own R and W.

What is derived, and from what.
- M, K and n of a product on this rank, from R and W alone: Q8_0 weights [K, M] are 34 bytes a 32-element block, the
  input [K, n] and the result [M, n] are F32, so W = 4 n M and 32 R = K (34 M + 128 n). n is tried at the stage's
  row count first; where that leaves no whole M and K, every n to 4,096 is tried and the solutions printed.
- With --header (a GGUF shard's header, parsed here; tensor data is not read) the weights' whole shape and type by
  name, and so which side the split cut. A name the headers given do not hold is marked "-".
- J, the columns of a tile: the second template argument of the mul_mat_q that ran. The grid and block are the
  launch's as CUPTI recorded them. The fixup's last template argument and its grid likewise.
- Times are kernel durations on the device (first device of the capture), means over the instances of a class; "ms
  a decode" sums every kernel of the class's ops over the samples and divides by the samples. They overlap across
  streams and are not wall time.
The other devices are compared with the first, class by class (sites, shapes, J, grids, launch counts); their times
are not tabled.
"""
import argparse, bisect, collections, json, re, sqlite3, statistics, struct

ap = argparse.ArgumentParser()
ap.add_argument('db')
ap.add_argument('--rows', type=int)
ap.add_argument('--stage', default='target_eval')
ap.add_argument('--skip', type=int, default=5)
ap.add_argument('--outlier', type=float, default=1.5)
ap.add_argument('--header', action='append', default=[])
ap.add_argument('--json')
a = ap.parse_args()

GGML_TYPES = {0: 'F32', 1: 'F16', 2: 'Q4_0', 8: 'Q8_0', 18: 'IQ3_XXS', 30: 'BF16'}


def gguf_tensors(path):
    out = {}
    with open(path, 'rb') as f:
        def rd(fmt):
            return struct.unpack('<' + fmt, f.read(struct.calcsize('<' + fmt)))

        def rstr():
            (n,) = rd('Q')
            return f.read(n).decode('utf-8', 'replace')
        if f.read(4) != b'GGUF':
            raise SystemExit(path + ': not a GGUF header')
        rd('I')
        nt, nkv = rd('QQ')
        sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

        def skip(t):
            if t == 8:
                rstr()
            elif t == 9:
                (et,) = rd('I')
                (cnt,) = rd('Q')
                if et in sizes:
                    f.seek(sizes[et] * cnt, 1)
                else:
                    for _ in range(cnt):
                        skip(et)
            else:
                f.seek(sizes[t], 1)
        for _ in range(nkv):
            rstr()
            (t,) = rd('I')
            skip(t)
        for _ in range(nt):
            name = rstr()
            (nd,) = rd('I')
            dims = rd('Q' * nd)
            (t,) = rd('I')
            rd('Q')
            out[name] = (list(dims), GGML_TYPES.get(t, 'type%d' % t))
    return out


header = {}
for h in a.header:
    header.update(gguf_tensors(h))

con = sqlite3.connect('file:' + a.db + '?mode=ro', uri=True)
ranges = []
ops = collections.defaultdict(list)
marks = collections.defaultdict(list)
for start, end, tid, label in con.execute(
        "SELECT n.start,n.end,n.globalTid,coalesce(n.text,s.value) FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id "
        "ORDER BY n.start"):
    if not label:
        continue
    if label.startswith('fused '):
        marks[tid].append((start, label[6:]))
        continue
    if end is None or end <= start:
        continue
    f = dict(re.findall(r'(\w+)=([^ ]+)', label))
    if f.get('stage') in ('target_eval', 'proposal', 'catchup'):
        ranges.append(dict(start=start, end=end, pid=tid & -16777216, f=f, k=[]))
    elif label != 'graph' and not label.startswith('glm53.'):
        ops[tid].append((start, end, label))
starts = [r['start'] for r in ranges]
op_starts = {tid: [o[0] for o in v] for tid, v in ops.items()}
mark_starts = {tid: [m[0] for m in v] for tid, v in marks.items()}

sel = [r for r in ranges if r['f'].get('stage') == a.stage]
if a.rows is None:
    a.rows = collections.Counter(int(r['f'].get('rows', 0)) for r in sel).most_common(1)[0][0]
want = {id(r) for r in sel if int(r['f'].get('rows', 0)) == a.rows}

for start, end, dev, pid, launch, name, full, node, gx, gy, gz, bx, by, bz in con.execute(
        "SELECT k.start,k.end,k.deviceId,k.globalPid,r.start,s.value,d.value,k.graphNodeId,k.gridX,k.gridY,k.gridZ,"
        "k.blockX,k.blockY,k.blockZ FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "LEFT JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId=k.correlationId AND (r.globalTid & -16777216)=k.globalPid "
        "LEFT JOIN StringIds s ON s.id=k.shortName LEFT JOIN StringIds d ON d.id=k.demangledName ORDER BY k.start"):
    if launch is None:
        continue
    i = bisect.bisect_right(starts, launch) - 1
    if i < 0:
        continue
    r = ranges[i]
    if id(r) not in want or pid != r['pid'] or launch >= r['end']:
        continue
    r['k'].append((start, end, dev, name or '?', full or '', node, (gx, gy, gz), (bx, by, bz)))

sel = [r for r in sel if id(r) in want and r['k']][a.skip:]
if not sel:
    raise SystemExit('no samples')
devs = sorted({k[2] for r in sel for k in r['k']})
dev0 = devs[0]
n_all = len(sel)
if a.outlier > 0:
    ksum = lambda r: sum(k[1] - k[0] for k in r['k'] if k[2] == dev0)
    bar = a.outlier * statistics.median([ksum(r) for r in sel])
    sel = [r for r in sel if ksum(r) <= bar]
print(f"{a.db}: stage {a.stage}, {a.rows} rows: {len(sel)} samples ({n_all} after the first {a.skip}, "
      f"{n_all - len(sel)} dropped as outliers); devices {devs}")

made = {}
parent = {}
for t, tid, node, orig in con.execute("SELECT start,globalTid,graphNodeId,originalGraphNodeId FROM CUDA_GRAPH_NODE_EVENTS ORDER BY start"):
    if orig is None or orig == node:
        made.setdefault(node, (t, tid))
    else:
        parent.setdefault(node, orig)
cache = {}


def inst_of(node):
    """the op instance (thread, index of its range) whose launch made this kernel's node, or None"""
    if node is None:
        return None
    if node in cache:
        return cache[node]
    root = node
    hops = 0
    while root in parent and hops < 64:
        root = parent[root]
        hops += 1
    out = None
    if root in made:
        t, tid = made[root]
        if tid in ops:
            i = bisect.bisect_right(op_starts[tid], t) - 1
            if i >= 0 and t <= ops[tid][i][1]:
                out = (tid, i)
    cache[node] = out
    return out


def parse(label):
    f = label.split(' ')
    d = dict(op=f[0], r=None, w=None, w0='')
    name = []
    for x in f[1:]:
        if x.startswith('r=') and x[2:].isdigit():
            d['r'] = int(x[2:])
        elif x.startswith('w=') and x[2:].isdigit():
            d['w'] = int(x[2:])
        elif x.startswith('w0='):
            d['w0'] = x[3:]
        else:
            name.append(x)
    d['name'] = ' '.join(name)
    return d


def fold(s):
    return re.sub(r'-\d+', '-N', re.sub(r'(_l|node_|blk\.)\d+', r'\1N', s))


def layer(s):
    m = re.search(r'blk\.(\d+)\.', s)
    return int(m.group(1)) if m else None


def solve(r, w, n0):
    """(M, K, n) of a Q8_0 product from its label's bytes: n0 first, else every n to 4,096"""
    out = []
    for n in [n0] + [x for x in range(1, 4097) if x != n0]:
        if w % (4 * n):
            continue
        m = w // (4 * n)
        den = 34 * m + 128 * n
        if (r * 32) % den:
            continue
        k = r * 32 // den
        if k > 0 and k % 32 == 0:
            out.append((m, k, n))
            if n == n0:
                break
    return out


def shape_text(d, n0):
    """one product's shape on the rank, the whole weights from the header, and the side the split cut"""
    s = solve(d['r'], d['w'], n0)
    whole = header.get(d['w0'])
    if whole is not None and len(s) > 1:
        kf, mf = whole[0][0], whole[0][1]
        fit = [x for x in s if mf % x[0] == 0 and kf % x[1] == 0]
        s = fit or s
    if not s:
        return None, f"? (r={d['r']} w={d['w']})", '-'
    m, k, n = s[0]
    txt = f"{m} x {k}, n {n}" + (f" (+{len(s) - 1} other solutions)" if len(s) > 1 else '')
    if whole is None:
        return (m, k, n), txt, '-'
    kf, mf = whole[0][0], whole[0][1]
    cut = 'none' if (m, k) == (mf, kf) else f"M/{mf // m}" if k == kf and mf % m == 0 else f"K/{kf // k}" if m == mf and kf % k == 0 else 'M and K'
    return (m, k, n), txt, f"{mf} x {kf} {whole[1]}, cut {cut}"


def tmpl(full):
    m = re.search(r'<([^>]*)>', full)
    return [x.split(')')[-1].strip() for x in m.group(1).split(',')] if m else []


def census(dev):
    """classes of the device's dense Q8 products and the other MUL_MAT launches"""
    cls = {}
    other = collections.defaultdict(lambda: [0, 0, collections.Counter()])
    for r in sel:
        inst = collections.defaultdict(list)
        for k in r['k']:
            if k[2] == dev:
                inst[inst_of(k[5])].append(k)
        for key, ks in inst.items():
            if key is None:
                continue
            tid, i = key
            o_start, o_end, label = ops[tid][i]
            d = parse(label)
            if d['op'] != 'MUL_MAT':
                continue
            mmq = [k for k in ks if k[3] == 'mul_mat_q']
            q8 = [k for k in mmq if '(ggml_type)8,' in k[4]]
            site = fold(d['name']) + ' <- ' + fold(d['w0'])
            if not q8:
                o = other[site + '  [' + header.get(d['w0'], ([], '-'))[1] + ']']
                o[0] += 1
                o[1] += sum(k[1] - k[0] for k in ks)
                for k in ks:
                    o[2][k[3]] += 1
                continue
            fused = []
            if tid in marks:
                lo = bisect.bisect_left(mark_starts[tid], o_start)
                hi = bisect.bisect_right(mark_starts[tid], o_end)
                fused = [parse(m[1]) for m in marks[tid][lo:hi]]
            fmm = [f for f in fused if f['op'] == 'MUL_MAT' and f['w0']]
            fix = [k for k in ks if k[3] == 'mul_mat_q_stream_k_fixup']
            qnt = [k for k in ks if k[3].startswith('quantize_mmq')]
            rest = [k for k in ks if k not in mmq and k not in fix and k not in qnt]
            shape, stxt, whole = shape_text(d, a.rows)
            fshapes = [shape_text(f, a.rows) for f in fmm]
            ck = (site, tuple(fold(f['w0']) for f in fmm), stxt, tuple(x[1] for x in fshapes),
                  tuple((tmpl(k[4])[1], k[6], k[7]) for k in mmq),
                  tuple((tmpl(k[4])[3], k[6]) for k in fix),
                  tuple(('/'.join(tmpl(k[4])), k[6]) for k in qnt),
                  tuple(sorted(fold(f['op'] + ' ' + f['name']) for f in fused if f not in fmm)))
            c = cls.get(ck)
            if c is None:
                c = cls[ck] = dict(site=site, fused_weights=list(ck[1]), shape=shape, shape_text=stxt, whole=whole,
                                   fused_shapes=[dict(shape=x[0], text=x[1], whole=x[2]) for x in fshapes],
                                   mmq=[dict(J=int(j), grid=g, block=b) for j, g, b in ck[4]],
                                   fixup=[dict(variant=v, grid=g) for v, g in ck[5]],
                                   quantize=[dict(variant=v, grid=g) for v, g in ck[6]],
                                   fused_other=list(ck[7]), n=0, layers=set(), t_mmq=0, t_fix=0, t_qnt=0, t_rest=0,
                                   rest=collections.Counter())
            c['n'] += 1
            if layer(d['w0']) is not None:
                c['layers'].add(layer(d['w0']))
            c['t_mmq'] += sum(k[1] - k[0] for k in mmq)
            c['t_fix'] += sum(k[1] - k[0] for k in fix)
            c['t_qnt'] += sum(k[1] - k[0] for k in qnt)
            c['t_rest'] += sum(k[1] - k[0] for k in rest)
            for k in rest:
                c['rest'][k[3]] += 1
    return cls, other


def spans(layers):
    out = []
    for x in sorted(layers):
        if out and out[-1][1] == x - 1:
            out[-1][1] = x
        else:
            out.append([x, x])
    return ','.join(str(lo) if lo == hi else f"{lo}-{hi}" for lo, hi in out) or '-'


cls, other = census(dev0)
ns = len(sel)
order = sorted(cls.values(), key=lambda c: -(c['t_mmq'] + c['t_fix'] + c['t_qnt'] + c['t_rest']))
tot = sum(c['t_mmq'] + c['t_fix'] + c['t_qnt'] + c['t_rest'] for c in order) / ns / 1e6
print(f"device {dev0}: {len(order)} classes of dense Q8_0 product, {sum(c['n'] for c in order) / ns:.1f} launches and "
      f"{tot:.3f} ms of kernels a decode (mul_mat_q {sum(c['t_mmq'] for c in order) / ns / 1e6:.3f}, fixup "
      f"{sum(c['t_fix'] for c in order) / ns / 1e6:.3f}, quantize {sum(c['t_qnt'] for c in order) / ns / 1e6:.3f}, "
      f"other {sum(c['t_rest'] for c in order) / ns / 1e6:.3f})")
acc = 0
for c in order:
    ms = (c['t_mmq'] + c['t_fix'] + c['t_qnt'] + c['t_rest']) / ns / 1e6
    acc += ms
    us = lambda t: t / c['n'] / 1e3
    print(f"\n  {c['site']}")
    print(f"    {c['n'] / ns:5.1f} a decode over layers {spans(c['layers'])}; {ms:.3f} ms a decode (cum {acc:.3f}); "
          f"us a launch: quantize {us(c['t_qnt']):.1f}, mul_mat_q {us(c['t_mmq']):.1f}, fixup {us(c['t_fix']):.1f}, other {us(c['t_rest']):.1f}")
    print(f"    on the rank: M x K = {c['shape_text']}; whole: {c['whole']}")
    for w0, f in zip(c['fused_weights'], c['fused_shapes']):
        print(f"    fused in the launch: {w0}: M x K = {f['text']}; whole: {f['whole']}")
    for m in c['mmq']:
        print(f"    mul_mat_q J {m['J']}, grid {m['grid']}, block {m['block']}")
    for f in c['fixup']:
        print(f"    fixup variant {f['variant']}, grid {f['grid']}")
    if not c['fixup']:
        print("    no fixup kernel")
    for q in c['quantize']:
        print(f"    quantize <{q['variant']}>, grid {q['grid']}")
    if c['fused_other'] or c['rest']:
        print(f"    also fused: {', '.join(c['fused_other']) or '-'}; other kernels: "
              + (', '.join(f'{n} x{v / c["n"]:.1f}' for n, v in c['rest'].most_common()) or '-'))

print(f"\ndevice {dev0}: by shape on the rank (single products and groups), for the backend harness:")
by = collections.defaultdict(lambda: [0, 0, set(), set()])
for c in order:
    ms_ = [c['shape']] + [f['shape'] for f in c['fused_shapes']]
    if any(s is None for s in ms_):
        key = ('?', c['shape_text'])
    else:
        key = (tuple(s[0] for s in ms_), ms_[0][2], ms_[0][1], tuple(m['J'] for m in c['mmq']), tuple(f['variant'] for f in c['fixup']))
    b = by[key]
    b[0] += c['n']
    b[1] += c['t_mmq'] + c['t_fix'] + c['t_qnt'] + c['t_rest']
    b[2].add(c['site'].split(' <- ')[1].replace('blk.N.', '') + ''.join('+' + w.replace('blk.N.', '') for w in c['fused_weights']))
    b[3] |= c['layers']
for key, b in sorted(by.items(), key=lambda kv: -kv[1][1]):
    if key[0] == '?':
        print(f"    unsolved {key[1]}: {b[0] / ns:.1f} a decode, {b[1] / ns / 1e6:.3f} ms: {', '.join(sorted(b[2]))}")
        continue
    ms_, n, k, js, fx = key
    case = (f"test_mul_mat(GGML_TYPE_Q8_0, GGML_TYPE_F32, {ms_[0]}, {n}, {k}, {{1, 1}}, {{1, 1}})" if len(ms_) == 1 else
            f"test_mul_mat_group({{{', '.join(map(str, ms_))}}}, {n}, {k}, GGML_TYPE_Q8_0)")
    print(f"    {case}")
    print(f"        J {'/'.join(map(str, js))}, fixup {'/'.join(fx) or 'none'}; {b[0] / ns:.1f} a decode, {b[1] / ns / 1e6:.3f} ms a decode, "
          f"{b[1] / b[0] / 1e3:.1f} us each; {', '.join(sorted(b[2]))}; layers {spans(b[3])}")

print(f"\ndevice {dev0}: MUL_MAT launches of the stage that ran no mul_mat_q<Q8_0> (header type in brackets):")
for site, (n, t, kn) in sorted(other.items(), key=lambda kv: -kv[1][1]):
    print(f"    {site[:72]:72s} {n / ns:6.1f} a decode {t / ns / 1e6:7.3f} ms  {', '.join(x for x, _ in kn.most_common(3))}")

strip = lambda cl: {k: (c['n'], [m['J'] for m in c['mmq']]) for k, c in cl.items()}
base = strip(cls)
for dev in devs[1:]:
    o = strip(census(dev)[0])
    if o == base:
        print(f"device {dev}: the same classes as device {dev0} (sites, shapes, J, grids, fixups, launch counts)")
    else:
        diff = [k for k in set(o) | set(base) if o.get(k) != base.get(k)]
        print(f"device {dev}: {len(diff)} classes differ from device {dev0}:")
        for k in diff[:12]:
            print(f"    {k[0]} {k[2]}: {base.get(k)} against {o.get(k)}")

if a.json:
    for c in order:
        c['layers'] = sorted(c['layers'])
        c['rest'] = dict(c['rest'])
        c['per_decode'] = c['n'] / ns
        c['ms_per_decode'] = (c['t_mmq'] + c['t_fix'] + c['t_qnt'] + c['t_rest']) / ns / 1e6
        c['us_per_launch'] = dict(quantize=c['t_qnt'] / c['n'] / 1e3, mul_mat_q=c['t_mmq'] / c['n'] / 1e3,
                                  fixup=c['t_fix'] / c['n'] / 1e3, other=c['t_rest'] / c['n'] / 1e3)
        for k in ('t_mmq', 't_fix', 't_qnt', 't_rest', 'n'):
            del c[k]
    json.dump(dict(capture=a.db, stage=a.stage, rows=a.rows, samples=ns, device=dev0, classes=order), open(a.json, 'w'), indent=1)
    print(f"wrote {a.json}")
