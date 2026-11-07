export const number = (value, digits = 2) => Number.isFinite(value)
  ? value.toLocaleString('en-US', { maximumFractionDigits: digits }) : 'Not measured';
export const bytes = value => {
  if (!Number.isFinite(value)) return 'Not measured';
  const units = ['B', 'kB', 'MB', 'GB'];
  let i = 0;
  while (value >= 1000 && i < 3) { value /= 1000; i++; }
  return number(value, 2) + ' ' + units[i];
};

export function compareStatus(ideal, metadata) {
  const reasons = [];
  if (ideal.sessions !== metadata.active) reasons.push(number(ideal.sessions) + ' ideal sessions vs ' + number(metadata.active) + ' recorded');
  if (ideal.context !== metadata.context) reasons.push(number(ideal.context) + ' ideal context vs ' + number(metadata.context) + ' recorded');
  if (metadata.depth_policy !== 'fixed-2') reasons.push('fixed MTP2 ideal vs ' + (metadata.depth_policy || 'unknown') + ' captured depth');
  const workloadMatches = reasons.length === 0;
  reasons.push('one ideal cycle vs a captured request; no direct speedup ratio');
  return { comparable: false, workloadMatches, reasons };
}

export function filterRecords(rows, device, search = '') {
  const query = search.toLowerCase().trim();
  return rows.filter(row => (device === 'all' || row.device === Number(device) || row.device === undefined)
    && (!query || [row.name, row.family, row.weight, row.direction, row.kind].filter(Boolean).join(' ').toLowerCase().includes(query)));
}

export function joinWeight(row, profile, device) {
  if (!row.weight) return null;
  const ops = filterRecords(profile.operations, device).filter(op => op.weight === row.weight);
  if (!ops.length) return null;
  const instances = ops.reduce((sum, op) => sum + op.instances, 0);
  return { mean_us: ops.reduce((sum, op) => sum + op.sum_us, 0) / instances,
    p95_us: ops.length === 1 ? ops[0].p95_us : null, instances, variants: ops.length, operations: ops };
}

export function copyFloor(row, parameters) {
  if (!row.calls) return null;
  const payload = row.bytes / row.calls;
  const rate = row.copy_kind === 1 || row.copy_kind === 2 ? parameters.host_gbps
    : row.copy_kind === 10 ? parameters.peer_gbps : null;
  return rate > 0 ? payload / (rate * 1000) : null;
}

export function selectEvents(profile, kind, index, device, offset, count) {
  const source = kind === 'copies' ? profile.copy_events : kind === 'apis' ? profile.api_events : profile.kernel_events;
  const column = kind === 'copies' || kind === 'operations' ? 4 : kind === 'apis' ? 2 : 3;
  const selected = source.filter(event => event[column] === index
    && (kind === 'apis' || device === 'all' || event[2] === Number(device)));
  return selected.slice(offset, offset + count);
}

export function parseProfile(text) {
  if (text.length > 80 * 1024 * 1024) throw new Error('Profile exceeds the 80 MB expanded limit.');
  const p = JSON.parse(text);
  if (p?.schema !== 'glm53-profile-v1') throw new Error('Use a glm53-profile-v1 JSON export from decode_lab.py.');
  for (const field of ['metadata', 'counts', 'coverage', 'summary', 'source']) {
    if (!p[field] || typeof p[field] !== 'object' || Array.isArray(p[field])) throw new Error('Invalid profile ' + field);
  }
  for (const field of ['kernels', 'operations', 'copies', 'apis', 'devices', 'hardware', 'caveats',
    'kernel_events', 'copy_events', 'api_events']) {
    if (!Array.isArray(p[field]) || p[field].length > 2000000) throw new Error('Invalid profile array: ' + field);
  }
  for (const key of ['active', 'context']) {
    if (p.metadata[key] != null && (!Number.isInteger(p.metadata[key]) || p.metadata[key] < 1))
      throw new Error('Invalid workload ' + key);
  }
  for (const field of ['kernels', 'operations', 'copies', 'apis']) {
    p[field].forEach((row, i) => {
      if (!row || row.index !== i) throw new Error('Invalid record index in ' + field);
      for (const key of ['calls', 'instances', 'sum_us', 'mean_us', 'p50_us', 'p95_us', 'max_us',
        'bytes', 'gbps', 'logical_read_bytes', 'logical_write_bytes']) {
        if (row[key] != null && (!Number.isFinite(row[key]) || row[key] < 0))
          throw new Error('Invalid measurement ' + key);
      }
      for (const key of ['name', 'direction', 'family', 'weight']) {
        if (row[key] != null && (typeof row[key] !== 'string' || row[key].length > 20000))
          throw new Error('Invalid record label');
      }
    });
  }
  const indexValid = (index, rows) => Number.isInteger(index) && index >= 0 && index < rows.length;
  for (const [field, length] of [['kernel_events', 5], ['copy_events', 5], ['api_events', 3]]) {
    for (const e of p[field]) {
      if (!Array.isArray(e) || e.length !== length || !e.every(Number.isFinite) || e[1] < 0)
        throw new Error('Invalid event duration or fields in ' + field);
      if (field === 'kernel_events' && (!indexValid(e[3], p.kernels) || (e[4] !== -1 && !indexValid(e[4], p.operations))))
        throw new Error('Invalid kernel event reference');
      if (field === 'copy_events' && (e[3] < 0 || !indexValid(e[4], p.copies))) throw new Error('Invalid copy event reference');
      if (field === 'api_events' && !indexValid(e[2], p.apis)) throw new Error('Invalid API event reference');
    }
  }
  if (p.counts.kernels !== p.kernel_events.length || p.counts.copies !== p.copy_events.length
      || p.counts.runtime !== p.api_events.length) throw new Error('Event counts do not match the exported records.');
  for (const row of p.devices) {
    if (!Number.isInteger(row.device) || !Array.isArray(row.bins)
      || row.bins.some(value => !Number.isFinite(value) || value < 0 || value > 1))
      throw new Error('Invalid device activity bins');
  }
  return p;
}
