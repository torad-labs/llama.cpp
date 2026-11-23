# idle-census.py TRACE.sqlite [ROWS]: inside the steady target evals of ROWS rows (NVTX "stage=target_eval ... rows=ROWS",
# durations within 10 % of their median), per device, a partition of the eval's host range: the time before its first
# kernel, the time with a kernel on the card (the union of the kernels' intervals, not their sum: kernels on different
# streams overlap), the time with none, split by whether a copy to or from the card overlaps the gap and by size, and
# the time after its last kernel. The parts add up to the range. A kernel is the eval's when it starts inside the
# range; what the kernels run past the range's end is printed apart and is not in the partition. A gap is labelled
# by the copies that overlap it, whatever issued them: nothing here says a copy belongs to a reduction.
# Means a target eval, ms. (idle-census-sums.py is the reader before Nov 11 5:30 PM CT: it added kernel durations
# and the gaps between kernels adjacent by start, which is not a partition where kernels overlap.)
import sqlite3, sys, bisect, statistics as st, collections
db = sys.argv[1]; want = sys.argv[2] if len(sys.argv) > 2 else '69'
c = sqlite3.connect('file:%s?mode=ro' % db, uri=True).cursor()
names = dict(c.execute('select id, value from StringIds'))
ev = []
for s, e, text, tid in c.execute('select start, end, text, textId from NVTX_EVENTS where end is not null'):
    t = text or names.get(tid, '')
    if 'stage=target_eval' in t and ('rows=%s ' % want in t + ' '): ev.append((s, e))
d = [e - s for s, e in ev]; med = st.median(d)
win = [(s, e) for s, e in ev if abs((e - s) - med) < 0.1*med]
print(f"{db}: {len(ev)} target evals of {want} rows, median {med/1e6:.2f} ms; {len(win)} within 10 % of it (mean {st.mean(e - s for s, e in win)/1e6:.2f} ms)")
devs = [r[0] for r in c.execute('select distinct deviceId from CUPTI_ACTIVITY_KIND_KERNEL order by 1')]
KEYS = ('before the first kernel', 'with a kernel (union)', 'gap with copies >= 64 KB', 'gap with small copies', 'bare gap < 2 us',
        'bare gap 2-5 us', 'bare gap 5-20 us', 'bare gap 20-100 us', 'bare gap > 100 us', 'after the last kernel')
for dev in devs:
    ks = c.execute('select start, end, shortName from CUPTI_ACTIVITY_KIND_KERNEL where deviceId=? order by start', (dev,)).fetchall()
    kstart = [k[0] for k in ks]
    ms = c.execute('select start, end, bytes from CUPTI_ACTIVITY_KIND_MEMCPY where deviceId=? or dstDeviceId=? order by start', (dev, dev)).fetchall()
    mstart = [m[0] for m in ms]
    tot = collections.Counter(); cnt = collections.Counter(); nk = 0; fam = collections.Counter(); ksum = 0; past = 0; n = 0
    for s, e in win:
        i0 = bisect.bisect_left(kstart, s); i1 = bisect.bisect_left(kstart, e)
        if i1 - i0 < 2: continue
        n += 1; nk += i1 - i0
        tot['before the first kernel'] += ks[i0][0] - s
        right = ks[i0][0]
        for i in range(i0, i1):
            ks_, ke, kn = ks[i]
            ksum += ke - ks_
            if ks_ > right:
                g0, g1 = right, ks_; g = g1 - g0
                j = bisect.bisect_left(mstart, g0 - 200000)
                has = False; cb = 0
                while j < len(ms) and ms[j][0] < g1:
                    if ms[j][1] > g0: has = True; cb += ms[j][2]
                    j += 1
                if has:
                    k = 'gap with copies >= 64 KB' if cb >= 65536 else 'gap with small copies'
                else:
                    k = 'bare gap < 2 us' if g < 2000 else 'bare gap 2-5 us' if g < 5000 else 'bare gap 5-20 us' if g < 20000 else 'bare gap 20-100 us' if g < 100000 else 'bare gap > 100 us'
                    if g >= 5000: fam[names.get(kn, '?')[:40]] += g
                tot[k] += g; cnt[k] += 1
                right = ks_
            if ke > right:
                tot['with a kernel (union)'] += min(ke, e) - min(right, e)
                right = ke
        tot['after the last kernel'] += max(0, e - right)
        past += max(0, right - e)
    print(f"device {dev}: {nk/n:.0f} kernels a target eval over {n} evals; their durations sum to {ksum/n/1e6:.2f} ms; they run {past/n/1e6:.2f} ms past the range's end")
    for k in KEYS:
        print(f"   {k:36s} {tot[k]/n/1e6:7.2f} ms" + (f"  ({cnt[k]/n:6.1f} a target eval, {tot[k]/max(1,cnt[k])/1e3:6.1f} us each)" if cnt[k] else ''))
    print(f"   {'the parts together':36s} {sum(tot[k] for k in KEYS)/n/1e6:7.2f} ms")
    if dev == devs[0]:
        print("   bare gaps of 5 us or more, by the kernel that follows (ms a target eval):", ', '.join(f"{k} {v/n/1e6:.2f}" for k, v in fam.most_common(8)))
