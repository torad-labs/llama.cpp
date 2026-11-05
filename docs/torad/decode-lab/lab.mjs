import { number, bytes, compareStatus, filterRecords, joinWeight, copyFloor, selectEvents, parseProfile } from './lab-core.mjs';
import { createVisual } from './visual.mjs';

const $ = selector => document.querySelector(selector);
const el = (tag, text, cls) => {
  const node = document.createElement(tag);
  if (text !== undefined && text !== null) node.textContent = text;
  if (cls) node.className = cls;
  return node;
};
const us = value => number(value, 3) + ' us';
const phaseNames = { target: 'Target verification', 'proposal-1': 'Draft proposal 1', 'proposal-2': 'Draft proposal 2', catchup: 'Cache-only catch-up', all: 'Complete cycle' };
const state = { ideal: null, profile: null, view: 'layers', phase: 'target', layer: null, device: '0',
  search: '', page: 0, selected: null, eventPage: 0, pending: 0, captureTicket: 0, modelTicket: 0 };
const PAGE = 24;
const visual = createVisual({tooltip, openEvidence});
function busy(change, message) {
  state.pending += change;
  $('#app').setAttribute('aria-busy', String(state.pending > 0));
  if (message) $('#load-state').textContent = message;
}
function error(message) {
  $('#error-text').textContent = message;
  $('#load-error').hidden = false;
}
function clearError() { $('#load-error').hidden = true; }
function tooltip(node, message) {
  node.setAttribute('aria-label', message);
  node.title = message;
  const show = event => {
    const t = $('#tooltip');
    t.textContent = message;
    t.hidden = false;
    const box = node.getBoundingClientRect();
    const x = event.clientX || box.left;
    const y = event.clientY || box.top;
    t.style.left = Math.min(innerWidth - 336, Math.max(8, x + 12)) + 'px';
    t.style.top = Math.min(innerHeight - 80, Math.max(8, y + 24)) + 'px';
  };
  node.addEventListener('pointermove', show);
  node.addEventListener('focus', show);
  node.addEventListener('pointerleave', () => { $('#tooltip').hidden = true; });
  node.addEventListener('blur', () => { $('#tooltip').hidden = true; });
}
function missingMetric(label = 'Not attributed') {
  const box = el('span', null, 'unknown-meter');
  const track = el('span', null, 'unknown-track');
  track.setAttribute('aria-hidden', 'true');
  const icon = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  icon.setAttribute('viewBox', '0 0 20 20');
  const circle = document.createElementNS(icon.namespaceURI, 'circle');
  circle.setAttribute('cx', '10'); circle.setAttribute('cy', '10'); circle.setAttribute('r', '8');
  const path = document.createElementNS(icon.namespaceURI, 'path');
  path.setAttribute('d', 'M7.5 7a2.5 2.5 0 0 1 5 0c0 2-2.5 2-2.5 4M10 14v.1');
  icon.append(circle, path); track.append(icon);
  box.append(el('span', label), track);
  return box;
}
function measure(value, max, observed = false, suffix = '') {
  if (!Number.isFinite(value)) return missingMetric();
  const box = el('span', null, 'bar-number' + (observed ? ' observed' : ''));
  box.append(el('span', number(value, 3) + suffix));
  const bar = el('i');
  bar.style.width = Math.max(0.2, value / (max || 1) * 100) + '%';
  box.append(bar);
  return box;
}
function fieldList(parent, values) {
  const dl = el('dl');
  for (const [name, value] of values) dl.append(el('dt', name), el('dd', value));
  parent.append(dl);
}
function subTitle(row) {
  if (row.weight) return row.weight.replace(/^blk\.\d+\./, '').replace(/\.weight$/, '').replaceAll('_', ' ');
  return row.name;
}
function humanApi(name) {
  return name.replace(/^cuda/, '').replace(/_v\d+$/, '').replace(/([a-z])([A-Z])/g, '$1 $2');
}
function displayKernel(row) {
  const match = row.name.match(/(?:void\s+)?([a-zA-Z_]\w*)[<(]/);
  return match ? match[1] : row.name;
}
function groups() {
  const map = new Map();
  for (const row of state.ideal.rows) {
    if (state.phase !== 'all' && row.stage !== state.phase) continue;
    const key = row.stage + ':' + row.layer;
    if (!map.has(key)) map.set(key, { key, stage: row.stage, layer: row.layer, name: '', rows: [], time_us: 0, memory: 0 });
    const group = map.get(key);
    group.rows.push(row); group.time_us += row.time_us;
    group.memory += row.read_bytes + row.write_bytes;
  }
  for (const group of map.values()) {
    group.name = group.layer === null ? 'Output & host boundaries' : 'Layer ' + group.layer;
    group.description = group.layer === null ? phaseNames[group.stage]
      : group.layer === 45 ? phaseNames[group.stage]
      : (group.layer % 4 === 3 ? 'Sparse attention' : 'KDA recurrent state') + (group.layer < 3 ? ' / dense FFN' : ' / routed + shared experts');
    const ops = state.profile.operations.filter(op => op.layer === group.layer && group.layer !== null
      && (state.device === 'all' || op.device === Number(state.device)));
    group.observed_us = ops.length ? ops.reduce((sum, op) => sum + op.gpu_sum_us, 0) : null;
    group.observed_calls = ops.reduce((sum, op) => sum + op.kernel_calls, 0);
  }
  return [...map.values()];
}
function renderHead() {
  const r = state.ideal.result, p = state.profile, m = p.metadata;
  $('#ideal-workload').textContent = number(r.sessions) + ' active sessions / 400,000 context / fixed MTP2';
  $('#ideal-cycle').textContent = number(r.cycle_ms, 3);
  $('#ideal-rate').textContent = number(r.tps_per_session, 2) + ' tokens/s/session';
  $('#profile-workload').textContent = number(m.active) + ' active session' + (m.active === 1 ? '' : 's')
    + ' / ' + number(m.context) + ' context / ' + (m.depth_policy || 'unknown') + ' drafts';
  $('#profile-span').textContent = number(p.summary.span_us / 1000, 2);
  const rate = m.timings?.predicted_per_second;
  $('#profile-rate').textContent = Number.isFinite(rate) ? number(rate, 2) + ' tokens/s (profiled request)' : 'Request throughput not reported';
  $('#profile-count').textContent = number(p.counts.kernels, 0) + ' kernels / ' + number(p.counts.copies, 0) + ' explicit copies';
  const share = p.coverage.attributed_kernels / p.counts.kernels;
  $('#profile-attribution').textContent = number(share * 100, 1) + '% assigned to operations';
  const status = compareStatus(r, m);
  $('#comparison-status').textContent = status.workloadMatches ? 'Batch settings match. Cycle timing is not aligned.' : 'Different workloads. No direct speedup ratio.';
  $('#mismatch-detail').textContent = number(r.sessions) + ' vs ' + number(m.active) + ' sessions / 400,000 vs '
    + number(m.context) + ' tokens / ' + (m.depth_policy || 'unknown') + ' capture depth';
  $('#mismatch').title = status.reasons.join('\n');
  $('#provenance').textContent = 'Captured runtime ' + (m.runtime || 'unknown') + ' / source ' + (p.source.file || 'local import')
    + ' / no 40-session 400k qualification';
}
function rowsForView() {
  if (state.view === 'layers') return groups().filter(row => (row.name + ' ' + row.description).toLowerCase().includes(state.search.toLowerCase()));
  if (state.view === 'substeps') return state.ideal.rows.filter(row => row.stage === state.phase && row.layer === state.layer)
    .filter(row => (row.name + ' ' + (row.weight || '') + ' ' + row.kind).toLowerCase().includes(state.search.toLowerCase()))
    .sort((a,b) => b.time_us-a.time_us);
  return filterRecords(state.profile[state.view], state.device, state.search).sort((a,b) => b.sum_us-a.sum_us);
}
const specs = {
  layers: ['Step', 'Ideal memory / GPU', 'Ideal us / cycle', 'Assigned GPU ms / capture', 'Assigned kernel calls'],
  substeps: ['Substep', 'Budget bytes / GPU', 'Path', 'Ideal us / cycle', 'Recorded us / invocation', 'Recorded invocations'],
  operations: ['Operation', 'Logical R + W / invocation', 'Logical-byte budget / us', 'Recorded mean / us', 'P95 / us', 'Invocations'],
  kernels: ['Kernel', 'GPU', 'Measured traffic', 'Mean / us', 'P95 / us', 'GPU total / ms'],
  copies: ['Explicit copy', 'GPU', 'Payload / call', 'Reference / us', 'Measured / us', 'Payload GB/s', 'Calls'],
  apis: ['CUDA API', 'Calls', 'Mean / us', 'P95 / us', 'Host total / ms'],
};
const scopes = {
  layers: 'Ideal: one cycle. Recorded: assigned GPU work across the capture. Separate scales; no speedup ratio.',
  substeps: 'Ideal: busiest-GPU byte and dependency budget. Recorded: same named weight, captured invocation mean across shapes. Not a matched-workload speedup.',
  operations: 'NVTX logical bytes are not memory counters. GPU invocation spans include kernel gaps; graph clones are attributed through their creation records.',
  kernels: 'Every recorded kernel, grouped by exact signature and device. Kernel-driven PCIe bytes are not measured by this trace.',
  copies: 'Reference uses the same recorded bytes per call and your directional link rate. Measured GB/s is payload divided by summed copy duration, not bus utilization.',
  apis: 'Host API durations can overlap GPU work and other threads. They cannot be added to GPU duration or treated as cycle-wall-time shares.',
};
function renderTable() {
  const rows = rowsForView();
  state.page = Math.min(state.page, Math.max(0, Math.ceil(rows.length/PAGE)-1));
  const view = state.view;
  $('#back-layers').hidden = view !== 'substeps';
  $('#ledger-title').textContent = view === 'layers' ? phaseNames[state.phase] + ' budgets'
    : view === 'substeps' ? (state.layer === null ? 'Output & boundaries' : 'Layer ' + state.layer) + ' / substeps'
    : ({ operations: 'Named operations', kernels: 'Captured kernels', copies: 'Explicit memory copies', apis: 'CUDA API calls' })[view];
  $('#table-scope').textContent = scopes[view];
  document.querySelectorAll('[data-view]').forEach(b => b.setAttribute('aria-selected', String(b.dataset.view === (view === 'substeps' ? 'layers' : view))));
  const thead = $('#ledger thead'); const header = el('tr');
  specs[view].forEach((title, index) => {
    const th = el('th', title); th.scope = 'col';
    if ((view === 'layers' && index === 2) || (view === 'substeps' && index === 3)) {
      const key = el('span', null, 'key ideal-key'); th.prepend(key);
    }
    if ((view === 'layers' && index === 3) || (view === 'substeps' && index === 4)) {
      const key = el('span', null, 'key profile-key'); th.prepend(key);
    }
    header.append(th);
  }); thead.replaceChildren(header);
  const tbody = $('#ledger tbody'); tbody.replaceChildren();
  const maxIdeal = Math.max(...rows.map(row => row.time_us || 0), 1);
  const maxObserved = Math.max(...rows.map(row => row.observed_us || row.mean_us || 0), 1);
  const params = state.ideal.result.parameters;
  for (const row of rows.slice(state.page*PAGE, (state.page+1)*PAGE)) {
    const tr = el('tr');
    const button = el('button', view === 'substeps' ? subTitle(row) : view === 'kernels' ? displayKernel(row)
      : view === 'apis' ? humanApi(row.name) : row.name || row.direction, 'row-select');
    button.type = 'button';
    const secondary = view === 'layers' ? row.description : view === 'substeps' ? row.kind.replaceAll('-',' ')
      : view === 'kernels' ? row.family : view === 'operations' && row.weight ? row.weight
      : view === 'copies' ? 'CUDA copy events' : null;
    if (secondary) button.append(el('small', secondary));
    const first = el('td'); first.append(button); tr.append(first);
    const add = value => { const td = el('td', null, 'numeric'); value instanceof Node ? td.append(value) : td.textContent = String(value); tr.append(td); };
    if (view === 'layers') {
      add(bytes(row.memory)); add(measure(row.time_us, maxIdeal));
      add(measure(row.observed_us === null ? null : row.observed_us/1000, maxObserved/1000, true));
      add(number(row.observed_calls,0));
    } else if (view === 'substeps') {
      const observed = joinWeight(row,state.profile,state.device);
      const payload = row.sent_bytes ? Math.max(...row.sent_bytes, ...row.received_bytes) : row.read_bytes+row.write_bytes;
      add(bytes(payload)); add(row.route || (row.kind === 'latency' ? 'Assumed latency' : 'GDDR'));
      add(measure(row.time_us,maxIdeal));
      add(observed ? us(observed.mean_us) : missingMetric());
      add(observed ? number(observed.instances,0) : '--');
    } else if (view === 'operations') {
      const amount = row.logical_read_bytes + row.logical_write_bytes;
      add(bytes(amount)); add(us(amount / (1597 * params.dram_efficiency * 1000)));
      add(measure(row.mean_us,maxObserved,true)); add(us(row.p95_us)); add(number(row.instances,0));
    } else if (view === 'kernels') {
      add(String(row.device)); add(missingMetric('Not measured'));
      add(measure(row.mean_us,maxObserved,true)); add(us(row.p95_us)); add(number(row.sum_us/1000,3));
    } else if (view === 'copies') {
      add(String(row.device)); add(bytes(row.bytes/row.calls));
      const floor = copyFloor(row,params);
      add(floor === null ? missingMetric('Not modeled') : us(floor));
      add(measure(row.mean_us,maxObserved,true)); add(number(row.gbps,2)); add(number(row.calls,0));
    } else {
      add(number(row.calls,0)); add(measure(row.mean_us,maxObserved,true)); add(us(row.p95_us)); add(number(row.sum_us/1000,3));
    }
    button.addEventListener('click', () => {
      state.selected = { view, row }; state.eventPage = 0;
      tbody.querySelectorAll('tr').forEach(node => node.removeAttribute('data-selected'));
      tr.dataset.selected = 'true'; renderInspector();
    });
    tbody.append(tr);
  }
  if (!rows.length) {
    const tr = el('tr'), td = el('td','No matching records. Change the search or device filter.');
    td.colSpan = specs[view].length; tr.append(td); tbody.append(tr);
  }
  $('#row-count').textContent = rows.length ? number(state.page*PAGE+1,0)+'-'+number(Math.min((state.page+1)*PAGE,rows.length),0)+' of '+number(rows.length,0)+' rows' : '0 rows';
  $('#previous').disabled = state.page === 0; $('#next').disabled = (state.page+1)*PAGE >= rows.length;
}
function renderInspector() {
  const target = $('#inspector'); target.replaceChildren();
  if (!state.selected) { target.append(el('h2','Inspect a step'),el('p','Select a row for its bytes, timing basis and captured events.')); return; }
  const { view, row } = state.selected;
  target.append(el('h2',view === 'layers' ? row.name : view === 'substeps' ? subTitle(row)
    : view === 'kernels' ? displayKernel(row) : view === 'apis' ? humanApi(row.name) : row.name || row.direction, 'detail-title'));
  target.append(el('p', view === 'layers' ? row.description : view === 'substeps' ? phaseNames[row.stage]
    : 'Captured ' + (row.device !== undefined ? 'GPU '+row.device+' / ' : '') + view, 'detail-type'));
  if (view === 'layers') {
    fieldList(target,[['Ideal budget',us(row.time_us)],['Substeps',number(row.rows.length,0)],
      ['Memory budget',bytes(row.memory)],['Assigned GPU work',row.observed_us === null ? 'Not attributed' : number(row.observed_us/1000,3)+' ms / capture']]);
    const open = el('button','Inspect '+number(row.rows.length,0)+' substeps','primary');
    open.type='button'; open.addEventListener('click',()=>{
      state.view='substeps';state.phase=row.stage;state.layer=row.layer;state.page=0;state.search='';$('#search').value='';
      $('#phase').value=row.stage;renderTable();$('#ledger-title').focus();
    }); target.append(open);
    target.append(el('h3','Largest ideal costs'));
    const top = [...row.rows].sort((a,b)=>b.time_us-a.time_us).slice(0,6);
    const max = top[0]?.time_us || 1;
    for (const child of top) {
      const label = el('p',subTitle(child));
      const value = measure(child.time_us,max); target.append(label,value);
    }
    target.append(el('h3','What is comparable?'),el('p','The ideal is one complete decode cycle. Assigned GPU work is the captured total for named operations in this layer, not a measured layer-cycle latency.'));
  } else if (view === 'substeps') {
    target.append(el('h3','Ideal budget'));
    fieldList(target,[['Time',us(row.time_us)],['Read',bytes(row.read_bytes)],['Write',bytes(row.write_bytes)],
      ['Basis',row.route || (row.kind==='latency' ? 'Latency assumption' : '1,597 GB/s x efficiency')]]);
    target.append(el('p',row.basis));
    const formula = el('div',null,'formula');
    formula.append(el('code',row.kind==='latency' ? 'Time = explicit latency budget' : row.route
      ? 'Peer: max(Tx / link, Rx / link, total Tx / fabric)\nHost: dependent publish + consume legs'
      : 'Time = (read bytes + write bytes) / effective GDDR bandwidth'));
    target.append(formula);
    if (row.sent_bytes) {
      target.append(el('h3','Payload by GPU'));
      const paths=el('div',null,'rank-path'), max=Math.max(...row.sent_bytes,...row.received_bytes,1);
      row.sent_bytes.forEach((sent,i)=>{
        const line=el('div'), bar=el('i');bar.style.width=Math.max(sent,row.received_bytes[i])/max*100+'%';
        line.append(el('span','GPU '+i),bar,el('span',bytes(sent)+' Tx / '+bytes(row.received_bytes[i])+' Rx'));paths.append(line);
      });target.append(paths);
    }
    if (row.dependency) target.append(el('h3','Dependency'),el('p',row.dependency));
    const observed=joinWeight(row,state.profile,state.device);
    target.append(el('h3','Recorded evidence'));
    if (observed) {
      fieldList(target,[['Invocation mean',us(observed.mean_us)],['Invocations',number(observed.instances,0)],
        ['Shape variants',number(observed.variants,0)],['P95',observed.p95_us===null ? 'Not pooled across shapes' : us(observed.p95_us)]]);
      target.append(el('p','Same named weight; different workload and possibly fused kernels. No speedup ratio is computed.'));
      const inspect=el('button','Open recorded operation');inspect.type='button';
      inspect.addEventListener('click',()=>{state.selected={view:'operations',row:observed.operations[0]};state.eventPage=0;renderInspector();});
      target.append(inspect);
    } else {
      target.append(el('div','Not attributed to this substep. Inspect kernel families or explicit copies instead.','empty-measure'));
    }
    if (row.weight) target.append(el('h3','Tensor evidence'),el('code',row.weight),
      el('p',(row.quant||'')+' / stored tensor '+bytes(row.tensor_bytes)+'. Routed weight reads use the expected selected-expert union, divided over four GPUs.'));
    target.append(el('h3','Source'),el('code',row.route ? 'src/models/glm5next.cpp:612-655\nggml/src/ggml-cuda/allreduce.cu\nscripts/glm53/estimate_decode_pcie.py' : 'Pinned GGUF tensor manifest\nscripts/glm53/decode_lab.py'));
  } else {
    const count = row.instances ?? row.calls;
    fieldList(target,[['Calls / invocations',number(count,0)],['Mean',us(row.mean_us)],['P50',us(row.p50_us)],
      ['P95',us(row.p95_us)],['Maximum',us(row.max_us)],['Sum',number(row.sum_us/1000,3)+' ms']]);
    if (view==='operations') {
      fieldList(target,[['Logical reads / call',bytes(row.logical_read_bytes)],['Logical writes / call',bytes(row.logical_write_bytes)],
        ['Kernels',number(row.kernel_calls,0)]]);
      target.append(el('p','Logical bytes come from NVTX labels, not DRAM counters. Invocation time spans its attributed kernels, including gaps; fused child work is not timed separately.'));
    } else if (view==='copies') {
      fieldList(target,[['Recorded payload',bytes(row.bytes)],['Payload / call',bytes(row.bytes/row.calls)],['Payload rate',number(row.gbps,2)+' GB/s']]);
      target.append(el('p','Payload rate = recorded bytes / summed copy duration. Queue delay and shared-fabric contention are not separate hardware counters here.'));
    } else if (view==='kernels') {
      target.append(el('div','PCIe and DRAM bytes: not measured for this kernel. A duration alone cannot establish bandwidth.','empty-measure'));
    } else {
      target.append(el('p','Host API time can overlap GPU execution and other threads. Do not add this sum to GPU duration.'));
    }
    const symbol=el('details');symbol.append(el('summary','Exact source label'),el('code',row.name||row.direction));target.append(symbol);
    eventDetails(target,view,row);
  }
  const evidence=el('details');evidence.append(el('summary','Evidence boundaries'));
  const list=el('ul',null,'detail-list');state.profile.caveats.forEach(text=>list.append(el('li',text)));
  evidence.append(list);target.append(evidence);
}
function eventDetails(target,view,row) {
  target.append(el('h3','Captured events'));
  const events=selectEvents(state.profile,view,row.index,state.device,state.eventPage*12,13);
  const shown=events.slice(0,12);
  const wrap=el('div',null,'table-wrap');const table=el('table',null,'event-table'), head=el('thead'), tr=el('tr');
  ['Start / us','Duration / us',view==='copies'?'Payload':'GPU'].forEach(s=>tr.append(el('th',s)));head.append(tr);table.append(head);
  const body=el('tbody');
  for (const event of shown) {
    const r=el('tr');r.append(el('td',number(event[0]/1000,3)),el('td',number(event[1]/1000,3)),
      el('td',view==='apis'?'Host':view==='copies'?bytes(event[3]):String(event[2])));body.append(r);
  }
  table.append(body);wrap.append(table);target.append(wrap,el('p','Relative to first captured GPU activity. Recorded nanosecond units; no synthetic events.'));
  const controls=el('div',null,'pagination'), prev=el('button','Previous events'),next=el('button','Next events');
  prev.type=next.type='button';prev.disabled=state.eventPage===0;next.disabled=events.length<=12;
  prev.addEventListener('click',()=>{state.eventPage--;renderInspector();});
  next.addEventListener('click',()=>{state.eventPage++;renderInspector();});
  controls.append(prev,next);target.append(controls);
}
function renderAll() {
  if (!state.ideal || !state.profile) return;
  renderHead();visual.render(state.ideal,state.profile);renderTable();
  if (!state.selected) {
    const row=groups().find(g=>g.layer===3)||groups()[0];
    state.selected=row?{view:'layers',row}:null;
  }
  renderInspector();$('#app').dataset.ready='true';
  $('#raw-toggle').disabled=false;$('#configure').disabled=false;
}
function openEvidence({view,phase,layer,search='',device=state.device}={}) {
  $('#raw-data').hidden=false;$('#raw-toggle').setAttribute('aria-expanded','true');
  $('#raw-toggle').textContent='Close raw evidence';
  if(view)state.view=view;
  if(phase){state.phase=phase;$('#phase').value=phase;}
  if(layer!==undefined)state.layer=layer;
  state.search=search;$('#search').value=search;state.device=device;$('#device').value=device;
  state.page=0;state.selected=null;state.eventPage=0;renderTable();renderInspector();
  $('#ledger-title').focus();$('#raw-data').scrollIntoView({block:'start'});
}
$('#raw-toggle').addEventListener('click',()=>{
  if($('#raw-data').hidden)openEvidence();
  else {
    $('#raw-data').hidden=true;$('#raw-toggle').setAttribute('aria-expanded','false');
    $('#raw-toggle').textContent='Open raw evidence';
  }
});
$('#import-trigger').addEventListener('click',()=>$('#import-file').click());
async function readStream(stream) {
  const reader=stream.getReader(), decoder=new TextDecoder();
  let result='', size=0;
  while (true) {
    const {value,done}=await reader.read();if(done)break;
    size+=value.byteLength;
    if(size>80*1024*1024){await reader.cancel();throw new Error('Expanded profile exceeds 80 MB. Export a shorter capture.');}
    result+=decoder.decode(value,{stream:true});
  }
  return result+decoder.decode();
}
async function loadCapture(name) {
  if(name==='imported')return;
  const ticket=++state.captureTicket;busy(1,'Reading captured events...');clearError();
  try {
    const response=await fetch('data/'+name+'.json.gz');
    if(!response.ok)throw new Error('Cannot read this local capture. Reload or import its exported JSON.');
    const text=await readStream(response.body.pipeThrough(new DecompressionStream('gzip')));
    const profile=parseProfile(text);
    if(ticket!==state.captureTicket)return;
    state.profile=profile;state.selected=null;state.page=0;state.eventPage=0;renderAll();
    $('#load-state').textContent='Capture loaded / '+number(profile.counts.kernels,0)+' kernels preserved';
  } catch(e) {if(ticket===state.captureTicket)error(e.message);}
  finally {busy(-1);}
}
async function updateIdeal() {
  const form=$('#assumptions');if(!form.reportValidity())return;
  const ticket=++state.modelTicket;busy(1,'Recalculating ideal byte budget...');clearError();
  const query=new URLSearchParams(new FormData(form));
  try {
    const response=await fetch('/api/ideal?'+query), data=await response.json();
    if(!response.ok)throw new Error(data.error||'Model calculation failed.');
    if(ticket!==state.modelTicket)return;
    state.ideal=data;state.selected=null;state.page=0;renderAll();$('#load-state').textContent='Ideal model updated / recorded evidence unchanged';
  } catch(e){if(ticket===state.modelTicket)error(e.message);}
  finally{busy(-1);}
}
document.querySelectorAll('[data-view]').forEach(button=>button.addEventListener('click',()=>{
  state.view=button.dataset.view;state.page=0;state.selected=null;state.eventPage=0;renderTable();renderInspector();
}));
$('#device').addEventListener('change',event=>{state.device=event.target.value;state.page=0;state.selected=null;renderTable();renderInspector();});
$('#search').addEventListener('input',event=>{state.search=event.target.value;state.page=0;renderTable();});
$('#phase').addEventListener('change',event=>{state.phase=event.target.value;state.view='layers';state.page=0;state.selected=null;renderAll();});
$('#assumptions').addEventListener('submit',event=>event.preventDefault());
$('#assumptions').addEventListener('change',event=>{if(event.target.id!=='phase')updateIdeal();});
$('#capture').addEventListener('change',event=>loadCapture(event.target.value));
$('#retry').addEventListener('click',()=>loadCapture($('#capture').value==='imported'?'timeline-4k':$('#capture').value));
$('#back-layers').addEventListener('click',()=>{state.view='layers';state.page=0;state.search='';$('#search').value='';renderTable();$('#ledger-title').focus();});
$('#previous').addEventListener('click',()=>{state.page--;renderTable();});
$('#next').addEventListener('click',()=>{state.page++;renderTable();});
$('#import-file').addEventListener('change',async event=>{
  const file=event.target.files[0];if(!file)return;
  ++state.captureTicket;busy(1,'Reading local import...');clearError();
  try {
    if(file.size>30*1024*1024)throw new Error('File exceeds 30 MB. Use the compressed JSON export.');
    const stream=file.name.endsWith('.gz')?file.stream().pipeThrough(new DecompressionStream('gzip')):file.stream();
    const profile=parseProfile(await readStream(stream));
    state.profile=profile;state.selected=null;state.page=0;state.eventPage=0;
    let option=$('#capture option[value="imported"]');
    if(!option){option=el('option');option.value='imported';$('#capture').append(option);}
    option.textContent='Imported: '+file.name;$('#capture').value='imported';
    renderAll();$('#load-state').textContent='Local import loaded / nothing uploaded';
  } catch(e){error('Import failed: '+e.message+' Existing evidence was kept.');}
  finally{busy(-1);event.target.value='';}
});
function setTheme(theme) {
  document.documentElement.dataset.theme=theme;$('#theme').textContent=theme==='dark'?'Light theme':'Dark theme';
  try{localStorage.setItem('decode-lab-theme',theme);}catch{}
}
$('#theme').addEventListener('click',()=>{
  const dark=document.documentElement.dataset.theme==='dark'||(!document.documentElement.dataset.theme&&matchMedia('(prefers-color-scheme: dark)').matches);
  setTheme(dark?'light':'dark');
});
try{const theme=localStorage.getItem('decode-lab-theme');if(theme==='dark'||theme==='light')setTheme(theme);}catch{}
$('#raw-toggle').disabled=true;$('#configure').disabled=true;
try {
  const response=await fetch('data/ideal.json');
  if(!response.ok)throw new Error('Ideal data is missing. Rebuild the evidence pack using README.md.');
  const pack=await response.json();state.ideal=pack.detail;
  await loadCapture('timeline-4k');
} catch(e){error(e.message);}
finally {busy(0);$('#app').setAttribute('aria-busy','false');}
