// Goal: show GPU-local work, transfer/wait and gaps without inventing a trace.
import test from 'node:test';
import assert from 'node:assert/strict';
import { profileActivity, idealActivity, waffleCounts } from '../../docs/torad/decode-lab/visual-core.mjs';

const profile = {
  summary: { span_us: 1000 },
  devices: [{device:0},{device:1}],
  kernels: [{family:'Projection matmul'}, {family:'Expert reduction'}],
  copies: [{copy_kind:1}],
  kernel_events: [[0,100000,0,0,-1],[50000,100000,0,1,-1],[0,50000,1,0,-1]],
  copy_events: [[200000,20000,0,1000,0]]
};

test('overlapping transfer and compute do not create extra elapsed time', () => {
  const result = profileActivity(profile, 10);
  assert.equal(result.devices[0].local_us, 50);
  assert.equal(result.devices[0].transfer_us, 120);
  assert.equal(result.devices[0].gap_us, 830);
  assert.equal(result.devices[1].local_us, 50);
  assert.equal(result.devices[1].gap_us, 950);
  for (const gpu of result.devices) {
    assert.equal(gpu.local_us + gpu.transfer_us + gpu.gap_us, 1000);
    for (const bin of gpu.bins) assert.ok(Math.abs(bin.reduce((a,b)=>a+b,0)-1)<1e-10);
  }
});
test('activity partitions match a simple occupancy model over generated overlaps', () => {
  let seed = 41;
  const next = n => { seed = seed * 16807 % 2147483647; return seed % n; };
  for (let run = 0; run < 50; run++) {
    const p = structuredClone(profile);
    p.summary.span_us = 100;
    p.copy_events = [];
    p.kernel_events = Array.from({length:30}, () => [next(100)*1000, next(30)*1000, next(2), next(2), -1]);
    const actual = profileActivity(p, 10);
    for (const gpu of actual.devices) {
      const expected = [0, 0, 0];
      for (let tick = 0; tick < 100; tick++) {
        const live = p.kernel_events.filter(e => e[2] === gpu.device && e[0] <= tick*1000 && e[0]+e[1] > tick*1000);
        expected[live.some(e => e[3] === 1) ? 1 : live.length ? 0 : 2]++;
      }
      assert.deepEqual([gpu.local_us, gpu.transfer_us, gpu.gap_us], expected);
    }
  }
});
test('waffle areas conserve the whole and stay within one-cell rounding', () => {
  let seed=53;
  for(let i=0;i<150;i++){
    const values=Array.from({length:3},()=>{seed=(seed*16807)%2147483647;return seed%1000;});
    const counts=waffleCounts(values);
    assert.equal(counts.reduce((a,b)=>a+b,0),100);
    const total=values.reduce((a,b)=>a+b,0);
    counts.forEach((count,j)=>assert.ok(Math.abs(count-values[j]/total*100)<1));
  }
});
test('the ideal never fabricates per-rank timing for unequal owners', () => {
  const result=idealActivity({result:{owners:[1,0,0,0],dram_ms:9,exposed_pcie_ms:1,cycle_ms:10}});
  assert.equal(result.devices[0].local_us,9000);
  assert.equal(result.devices[1].known,false);
  assert.equal(result.devices[1].local_us,null);
  assert.equal(result.aggregate.local_us,9000);
  assert.equal(result.aggregate.total_us,10000);
});
test('balanced ideal chips use the stated model, not captured measurements', () => {
  const result=idealActivity({result:{owners:[10,10,10,10],dram_ms:28,exposed_pcie_ms:2,cycle_ms:30}});
  assert.equal(result.devices.length,4);
  assert.ok(result.devices.every(d=>d.known&&d.gap_us===0));
  assert.equal(result.aggregate.transfer_us,2000);
});
test('collective and host staging kernels count as transfer/wait, not measured bandwidth', () => {
  const p=structuredClone(profile);
  p.kernels[0].family='Host KV staging';
  const result=profileActivity(p,10);
  assert.equal(result.devices[1].transfer_us,50);
  assert.equal(result.devices[1].local_us,0);
  assert.equal(result.measured_pcie_bytes,null);
});
