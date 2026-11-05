// Elapsed-time partitions, not sums of overlapping kernel durations.
const transferFamilies = new Set(['Expert / sampling gather', 'Expert reduction', 'Host KV staging']);

function merge(intervals, end) {
  const sorted = intervals.map(([a,b]) => [Math.max(0,a), Math.min(end,b)])
    .filter(([a,b]) => b > a).sort((a,b) => a[0]-b[0]);
  const result = [];
  for (const interval of sorted) {
    const last = result.at(-1);
    if (last && interval[0] <= last[1]) last[1] = Math.max(last[1], interval[1]);
    else result.push(interval);
  }
  return result;
}
function totals(intervals, span, count) {
  const bins = Array(count).fill(0);
  let duration = 0;
  if (!span) return {bins, duration};
  const width = span/count;
  for (const [a,b] of intervals) {
    duration += b-a;
    const first = Math.floor(a/width), last = Math.min(count-1, Math.ceil(b/width)-1);
    for (let i=first; i<=last; i++) bins[i] += Math.max(0, Math.min(b,(i+1)*width)-Math.max(a,i*width))/width;
  }
  return {bins, duration};
}
export function profileActivity(profile, binCount=96) {
  if (!Number.isInteger(binCount) || binCount < 1) throw new Error('Positive bin count required');
  const span = profile.summary.span_us*1000;
  const ranks = new Map(profile.devices.map(d => [d.device, {all:[], transfer:[]}]));
  function add(event, transfer) {
    const rank = ranks.get(event[2]);
    if (!rank) return;
    const range = [event[0],event[0]+event[1]];
    rank.all.push(range);
    if (transfer) rank.transfer.push(range);
  }
  for (const event of profile.kernel_events) add(event, transferFamilies.has(profile.kernels[event[3]].family));
  // CUPTI kind 8 is device-to-device. It is local activity, not a peer wire measurement.
  for (const event of profile.copy_events) add(event, profile.copies[event[4]].copy_kind !== 8);
  const devices = [...ranks].map(([device, rank]) => {
    const all = totals(merge(rank.all,span),span,binCount);
    const transfer = totals(merge(rank.transfer,span),span,binCount);
    return {device, known:true, total_us:span/1000,
      local_us:Math.max(0,all.duration-transfer.duration)/1000,
      transfer_us:transfer.duration/1000, gap_us:Math.max(0,span-all.duration)/1000,
      bins:all.bins.map((value,i) => [Math.max(0,value-transfer.bins[i]), transfer.bins[i], Math.max(0,1-value)])};
  });
  const aggregate = {local_us:0, transfer_us:0, gap_us:0, total_us:0};
  for (const rank of devices) for (const key of Object.keys(aggregate)) aggregate[key] += rank[key];
  return {devices, aggregate, measured_pcie_bytes:null};
}
export function idealActivity(ideal) {
  const r = ideal.result, max = Math.max(...r.owners);
  const aggregate = {local_us:r.dram_ms*1000, transfer_us:r.exposed_pcie_ms*1000, gap_us:0, total_us:r.cycle_ms*1000};
  return {aggregate, balanced:r.owners.every(value=>value===max),
    devices:r.owners.map((owners,device) => ({device, owners, known:owners===max,
      ...(owners===max ? aggregate : {local_us:null, transfer_us:null, gap_us:null, total_us:null})}))};
}
export function waffleCounts(values, cells=100) {
  const total = values.reduce((sum,value)=>sum+value,0);
  if (!total) return values.map((_,i)=>i===values.length-1?cells:0);
  const exact = values.map(value=>value/total*cells);
  const counts = exact.map(Math.floor);
  const remaining = cells-counts.reduce((sum,value)=>sum+value,0);
  const order = exact.map((value,index)=>({index,fraction:value-counts[index]}))
    .sort((a,b)=>b.fraction-a.fraction || a.index-b.index);
  for (let i=0;i<remaining;i++) counts[order[i].index]++;
  return counts;
}
