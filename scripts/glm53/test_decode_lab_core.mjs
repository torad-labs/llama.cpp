import test from 'node:test';
import assert from 'node:assert/strict';
import { compareStatus, filterRecords, joinWeight, copyFloor, selectEvents, parseProfile } from '../../docs/torad/decode-lab/lab-core.mjs';

const profile = {
  schema: 'glm53-profile-v1',
  metadata: { active: 1, context: 4096, depth_policy: 'adaptive', kv: 'host' },
  counts: { kernels: 2, copies: 1, runtime: 1 },
  coverage: { attributed_kernels: 2, unattributed_kernels: 0 },
  summary: { span_us: 20, kernel_sum_us: 10, kernel_union_us: 8 },
  source: { file: 'local.sqlite', sha256: 'test' }, hardware: [], caveats: [], devices: [],
  kernels: [
    { index: 0, device: 0, name: 'matmul', family: 'Projection', calls: 1, sum_us: 4, mean_us: 4, p50_us: 4, p95_us: 4, max_us: 4, measured_bytes: null, gbps: null },
    { index: 1, device: 1, name: '<script>bad()</script>', family: 'Other', calls: 1, sum_us: 6, mean_us: 6, p50_us: 6, p95_us: 6, max_us: 6, measured_bytes: null, gbps: null }
  ],
  operations: [
    { index: 0, device: 0, name: 'mm', weight: 'blk.3.weight', layer: 3, instances: 2, kernel_calls: 2, sum_us: 6, mean_us: 3, p50_us: 3, p95_us: 4, max_us: 4, logical_read_bytes: 100, logical_write_bytes: 40 },
    { index: 1, device: 0, name: 'mm other shape', weight: 'blk.3.weight', layer: 3, instances: 1, kernel_calls: 1, sum_us: 9, mean_us: 9, p50_us: 9, p95_us: 9, max_us: 9, logical_read_bytes: 300, logical_write_bytes: 40 }
  ],
  copies: [{ index: 0, device: 0, direction: 'Host to GPU', copy_kind: 1, calls: 1, bytes: 54000, sum_us: 1, mean_us: 1, p50_us: 1, p95_us: 1, max_us: 1, gbps: 54 }],
  apis: [{ index: 0, name: 'cudaLaunch', calls: 1, sum_us: 2, mean_us: 2, p50_us: 2, p95_us: 2, max_us: 2 }],
  kernel_events: [[0,4000,0,0,0],[2000,6000,1,1,1]],
  copy_events: [[6000,1000,0,54000,0]], api_events: [[-1000,2000,0]]
};

test('workload mismatch never becomes a speedup claim', () => {
  const status = compareStatus({ sessions: 40, context: 400000 }, profile.metadata);
  assert.equal(status.comparable, false);
  assert.ok(status.reasons.some(x => x.includes('400,000')));
  assert.ok(status.reasons.some(x => x.includes('40')));
});
test('matching batch settings are distinguished from cycle alignment', () => {
  const status = compareStatus({ sessions: 40, context: 400000 }, { active: 40, context: 400000, depth_policy: 'fixed-2' });
  assert.equal(status.workloadMatches, true);
  assert.equal(status.comparable, false);
});
test('device and text filters retain the actual records', () => {
  assert.equal(filterRecords(profile.kernels, '0', 'mat').length, 1);
  assert.equal(filterRecords(profile.kernels, '1', 'mat').length, 0);
});
test('weight comparisons pool instance means, not percentile averages', () => {
  const observed = joinWeight({ weight: 'blk.3.weight' }, profile, '0');
  assert.equal(observed.mean_us, 5);
  assert.equal(observed.instances, 3);
  assert.equal(observed.p95_us, null);
  assert.equal(joinWeight({ weight: 'missing' }, profile, '0'), null);
});
test('copy reference uses bytes per call and the selected directional rate', () => {
  assert.equal(copyFloor(profile.copies[0], { host_gbps: 54 }), 1);
});
test('all captured events remain accessible with device filtering', () => {
  assert.deepEqual(selectEvents(profile, 'kernels', 1, 'all', 0, 20), [profile.kernel_events[1]]);
  assert.equal(selectEvents(profile, 'kernels', 1, '0', 0, 20).length, 0);
});
test('import validation rejects non-finite timings and broken event references', () => {
  assert.equal(parseProfile(JSON.stringify(profile)).schema, 'glm53-profile-v1');
  const corrupt = structuredClone(profile);
  corrupt.kernel_events[0][3] = 500;
  assert.throws(() => parseProfile(JSON.stringify(corrupt)), /event|reference/i);
  corrupt.kernel_events[0] = [0,-1,0,0,0];
  assert.throws(() => parseProfile(JSON.stringify(corrupt)), /duration|event/i);
  assert.throws(() => parseProfile('{}'), /profile|schema/i);
});
