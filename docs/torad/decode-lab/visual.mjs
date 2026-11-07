import {number} from './lab-core.mjs';
import {profileActivity, idealActivity, waffleCounts} from './visual-core.mjs';

const $ = selector => document.querySelector(selector);
const names = ['GPU-local work', 'Transfer / wait', 'Unrecorded'];
const classes = ['local', 'transfer', 'gap'];
const fields = ['local_us', 'transfer_us', 'gap_us'];
const stageNames = {'proposal-1':'Draft 1', 'proposal-2':'Draft 2', target:'Verify', catchup:'Catch up'};
const e = (tag, text, cls) => {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (cls) node.className = cls;
  return node;
};
const button = (label, cls, action) => {
  const b = e('button', label, cls); b.type='button';
  if (action) b.addEventListener('click',action);
  return b;
};
const pct = value => number(value*100,1)+'%';
const ms = value => number(value/1000,3)+' ms';
const fractions = rank => fields.map(field => rank.total_us ? rank[field]/rank.total_us : 0);
const isTransfer = row => row.route || row.kind === 'latency';
const cache = new WeakMap();
const futureActivity = owners => ({aggregate:{pending:true}, devices:owners.map((count,device)=>({device,owners:count,known:false,pending:true}))});

export function createVisual({tooltip, openEvidence}) {
  let ideal, profile, modeled, measured, opener;
  function close(id, restore=true) {
    $(id).hidden=true;
    if ($('#app').dataset.panel === (id==='#settings'?'settings':'inspect')) delete $('#app').dataset.panel;
    if (id==='#settings') $('#configure').setAttribute('aria-expanded','false');
    $('#tooltip').hidden=true;
    if (restore && opener?.isConnected) opener.focus();
  }
  function inspect(title, draw) {
    opener=document.activeElement;
    close('#settings',false);
    $('#visual-inspector-title').textContent=title;
    const body=$('#visual-inspector-body');body.replaceChildren();draw(body);
    $('#visual-inspector').hidden=false;$('#app').dataset.panel='inspect';
    $('#tooltip').hidden=true;
    $('#visual-inspector-close').focus();
  }
  function raw(options) { close('#visual-inspector',false);openEvidence(options); }
  function definition(parent, title, text) {
    const dl=e('dl',undefined,'data-definition');dl.append(e('dt',title),e('dd',text));parent.append(dl);
  }
  function sideValues(parent, first, second, firstLabel='Ideal / one cycle', secondLabel='Recorded / average GPU') {
    const wrap=e('div',undefined,'detail-comparison');
    for (const [name,value,note] of [[firstLabel,first[0],first[1]],[secondLabel,second[0],second[1]]]) {
      const column=e('div');column.append(e('h3',name),e('strong',value,'detail-value'),e('small',note));wrap.append(column);
    }
    parent.append(wrap);
  }
  function bars(parent, rows, unit='ms') {
    const max=Math.max(...rows.map(row=>row.value),1);
    const list=e('div',undefined,'detail-bars');
    for (const row of rows) {
      const item=e('div'), header=e('header'), bar=e('i');
      header.append(e('span',row.label),e('span',number(row.value,3)+' '+unit));
      bar.className=row.kind||'neutral';bar.style.width=row.value/max*100+'%';item.append(header,bar);list.append(item);
    }
    parent.append(list);
  }
  function pending(title, value=null) {
    inspect(title+' / future capture',body=>{
      sideValues(body,[value===null?'Not modeled':ms(value),'Ideal budget for this slot'],
        ['Awaiting capture','No measured time, traffic or speedup yet'],'Ideal','Future capture');
      body.append(e('div','Not measured','pending-detail'),
        e('p','This slot is reserved for the same GPU, phase or layer. Historical raw evidence is kept separate.'));
    });
  }
  function category(index) {
    inspect(names[index], body=>{
      const a=modeled.aggregate;
      sideValues(body, [ms(a[fields[index]]),pct(fractions(a)[index])+' of the modeled cycle'],
        ['Awaiting capture','No measurement in this slot'],'Ideal / one cycle','Future capture');
      if (index===1) {
        definition(body,'PCIe bandwidth: Not measured','Only explicit copies record payload bytes.');
        body.append(e('h3','Assumed link rates'));
        bars(body, [
          {label:'Peer link / GPU',value:ideal.result.parameters.peer_gbps,kind:'transfer'},
          {label:'Aggregate peer fabric',value:ideal.result.parameters.fabric_gbps,kind:'transfer'},
          {label:'Host link / GPU',value:ideal.result.parameters.host_gbps,kind:'transfer'},
        ],'GB/s');
        body.append(e('p','Model inputs, not measurements.'),
          button('Inspect explicit copies',undefined,()=>raw({view:'copies'})),
          button('Inspect collective kernels',undefined,()=>raw({view:'kernels',search:'ggml_cuda_ar'})));
        const details=e('details');
        details.append(e('summary','What counts as transfer / wait?'),
          e('p','Gather, reduction and host-KV kernels, plus copies except known local D2D. Unknown-route copies are not assumed to use PCIe. Duration includes work and waiting, not just wire time. Diagram lines show logical dependencies, not physical wiring or bandwidth.'));
        body.append(details);
      } else if (index===0) {
        definition(body,'Memory budget vs captured local activity',
          'Ideal blue is a GDDR read/write budget. Recorded blue is non-communication kernels and local device copies. It is not a measured DRAM utilization counter.');
        const groups=new Map();
        for(const row of ideal.rows) if(!isTransfer(row)) groups.set(row.kind,(groups.get(row.kind)||0)+row.time_us/1000);
        body.append(e('h3','Inside the ideal memory budget'));
        bars(body,[...groups].sort((a,b)=>b[1]-a[1]).map(([label,value])=>({label:label.replaceAll('-',' '),value,kind:'local'})));
        body.append(button('Inspect named operations',undefined,()=>raw({view:'operations'})));
      } else {
        definition(body,'Unrecorded does not mean idle',
          'Gray is time without an exported CUDA kernel or explicit-copy interval on that GPU. CPU work, other profiler activity and missing coverage may occupy it. The ideal has no modeled software gaps; that is an omission, not a prediction of zero overhead.');
        body.append(button('Inspect host API calls',undefined,()=>raw({view:'apis'})));
      }
      const timing=e('details');timing.append(e('summary','Timing definitions'),e('p','Each side uses its own interval. Overlapping recorded activity is counted once, with transfer/wait taking priority. These different workloads do not establish a speedup.'));body.append(timing);
    });
  }
  function gpu(device) {
    inspect('GPU '+device, body=>{
      const a=modeled.devices.find(d=>d.device===device);
      const b=null;
      sideValues(body, [a?.known?ms(a.total_us):'Not modeled',a?.known?'Busiest-owner model / not a measurement':'Unequal ownership; per-rank timing is unavailable'],
        ['Awaiting capture','No recorded interval on this GPU'],
        'Ideal / GPU '+device,'Future / GPU '+device);
      for(const [label,rank] of [['Ideal',a],['Future capture',b]]) {
        body.append(e('h3',label));
        if (rank?.known) bars(body, fields.map((field,i)=>({label:names[i],value:rank[field]/1000,kind:classes[i]})));
        else body.append(e('p','No per-rank timing available.'));
      }
      definition(body,'How to read the chip','100 squares show time shares, rounded to the nearest 1%. Squares are not CUDA cores or memory capacity. GPU time is not added across devices to claim a cycle latency.');
      body.append(button('Inspect GPU '+device+' kernels',undefined,()=>raw({view:'kernels',device:String(device)})));
    });
  }
  function group(stage,layer) {
    const rows=ideal.rows.filter(row=>row.stage===stage && (layer===undefined || row.layer===layer));
    const title=layer===undefined?stageNames[stage]:'Target layer '+layer;
    inspect(title,body=>{
      const sum=rows.reduce((value,row)=>value+row.time_us,0);
      sideValues(body,[ms(sum),'Ideal byte + dependency budget'],
        ['Awaiting capture','Same logical step / no measurement yet'],
        'Ideal / one cycle','Future / one cycle');
      const groups=new Map();
      for (const row of rows) {
        const label=row.kind.replaceAll('-',' ');
        groups.set(label,(groups.get(label)||0)+row.time_us/1000);
      }
      bars(body,[...groups].sort((a,b)=>b[1]-a[1]).map(([label,value])=>({label,value})));
      body.append(e('p',number(rows.length,0)+' underlying substeps. Bar lengths here are ideal budget costs, not measured kernel timing.'),
        button('Inspect every substep',undefined,()=>raw({view:layer===undefined?'layers':'substeps',phase:stage,layer})));
    });
  }
  function composition(target, rank) {
    if(rank.pending) {
      const track=button(undefined,'pending-track',()=>pending('Cycle'));
      tooltip(track,'Awaiting matching capture. No time shares have been measured.');
      const labels=e('div',undefined,'composition-labels');labels.append(e('span','Awaiting measurements'),e('span','--'));
      target.replaceChildren(track,labels);return;
    }
    const track=e('div',undefined,'composition-track'), labels=e('div',undefined,'composition-labels');
    fractions(rank).forEach((share,i)=>{
      if (share>0) {
        const b=button(undefined,classes[i],()=>category(i));b.style.flex=String(share);
        tooltip(b,names[i]+': '+pct(share));track.append(b);
      }
      const label=e('span'), swatch=e('i',undefined,'swatch '+classes[i]);
      swatch.setAttribute('aria-hidden','true');
      label.append(swatch,e('strong',pct(share)),document.createTextNode(' '+['local','transfer','unrecorded'][i]));
      labels.append(label);
    });
    target.replaceChildren(track,labels);
  }
  function chips(target, activity, side) {
    target.replaceChildren();
    for(const rank of activity.devices) {
      const b=button(undefined,'gpu-chip'+(rank.pending?' pending-chip':rank.known?'':' unknown-chip'),
        ()=>rank.pending?pending('GPU '+rank.device,modeled.devices[rank.device]?.total_us??null):gpu(rank.device));
      if(rank.pending)b.dataset.state='pending';
      const head=e('span',undefined,'chip-head');head.append(e('span','GPU '+rank.device),
        e('span',side==='ideal'?rank.owners+' owners':'expected','chip-owner'));
      const grid=e('span',undefined,'waffle');grid.setAttribute('aria-hidden','true');
      let foot='Not modeled', description='Per-rank timing not modeled for unequal ownership';
      if(rank.known) {
        const f=fractions(rank), counts=waffleCounts(fields.map(field=>rank[field]));
        counts.forEach((count,i)=>{for(let j=0;j<count;j++) grid.append(e('i',undefined,classes[i]));});
        const dominant=f.indexOf(Math.max(...f));
        foot=number(f[dominant]*100,0)+'% '+['local','transfer','unrecorded'][dominant];
        description=names.map((name,i)=>name+' '+pct(f[i])).join(', ');
      } else if(rank.pending) {
        for(let i=0;i<100;i++)grid.append(e('i',undefined,'pending-cell'));
        foot='Not recorded';description='Awaiting capture; empty time slots';
      } else grid.textContent='?';
      b.append(head,grid,e('span',foot,'chip-foot'));
      tooltip(b,(side==='ideal'?'Ideal':'Future capture')+' GPU '+rank.device+'. '+description+'. Select to inspect.');
      target.append(b);
    }
  }
  function connections(target, side) {
    const NS='http://www.w3.org/2000/svg';
    const svg=document.createElementNS(NS,'svg');
    svg.setAttribute('viewBox','0 0 1000 152');svg.setAttribute('preserveAspectRatio','none');
    svg.setAttribute('aria-hidden','true');
    const defs=document.createElementNS(NS,'defs'),marker=document.createElementNS(NS,'marker');
    marker.id=side+'-arrow';marker.setAttribute('viewBox','0 0 6 6');marker.setAttribute('refX','5');
    marker.setAttribute('refY','3');marker.setAttribute('markerWidth','5');marker.setAttribute('markerHeight','5');
    marker.setAttribute('orient','auto-start-reverse');
    const arrow=document.createElementNS(NS,'path');arrow.setAttribute('d','M1 1L5 3L1 5');
    marker.append(arrow);defs.append(marker);svg.append(defs);
    for(const x of [125,375,625,875]) {
      const path=document.createElementNS(NS,'path');path.classList.add('gpu-spoke');path.setAttribute('d','M'+x+' 2V44');
      path.setAttribute('marker-start','url(#'+side+'-arrow)');svg.append(path);
    }
    const bus=document.createElementNS(NS,'path');bus.classList.add('gpu-bus');bus.setAttribute('d','M125 44H875');svg.append(bus);
    const trunk=document.createElementNS(NS,'path');trunk.classList.add('mobile-trunk');trunk.setAttribute('d','M500 0V44');svg.append(trunk);
    const host=document.createElementNS(NS,'path');host.setAttribute('d','M500 44V105');
    host.classList.add('host-route');host.setAttribute('marker-end','url(#'+side+'-arrow)');svg.append(host);
    const kv=ideal.result.kv;
    target.dataset.state=side==='future'?'pending':'modeled';
    const note=kv==='resident'?'KV stays on GPU':kv==='unknown'||!kv?'KV location unknown':'Selected KV rows + controls';
    const action=()=>side==='future'?pending('PCIe / host'):category(1);
    const peer=button('PCIe / gather + reduce','bus-label',action);
    const cpu=button('Host','host-node',action);cpu.append(e('small',(side==='future'?'Expected: ':'')+note));
    target.replaceChildren(svg,peer,e('span','hidden state + controls','connection-caption'),cpu);
    target.setAttribute('aria-label','Logical paths: GPU gather and reduce over PCIe, host hidden-state and control handoffs. Not physical topology.');
  }
  function cycle() {
    const path=$('#cycle-path');path.replaceChildren();
    for(const stage of ['proposal-1','proposal-2','target','catchup']) {
      const rows=ideal.rows.filter(row=>row.stage===stage);
      const total=rows.reduce((sum,row)=>sum+row.time_us,0);
      const transfer=rows.filter(isTransfer).reduce((sum,row)=>sum+row.time_us,0);
      const b=button(undefined,'phase-step',()=>group(stage));
      b.append(e('b',stageNames[stage]),e('span',ms(total)));
      const strip=e('span',undefined,'mini-budget');
      for(const [cls,value] of [['local',total-transfer],['transfer',transfer]]) {
        const mark=e('i',undefined,cls);mark.style.flex=String(Math.max(0,value));strip.append(mark);
      }
      b.append(strip);tooltip(b,stageNames[stage]+': '+ms(total)+' ideal budget; select for substeps');path.append(b);
    }
    const map=$('#layer-map');map.replaceChildren();
    const layers=Array.from({length:45},(_,layer)=>{
      const rows=ideal.rows.filter(row=>row.stage==='target'&&row.layer===layer);
      return {layer,total:rows.reduce((s,r)=>s+r.time_us,0),transfer:rows.filter(isTransfer).reduce((s,r)=>s+r.time_us,0)};
    });
    const max=Math.max(...layers.map(row=>row.total),1);
    for(const row of layers) {
      const b=button(undefined,'layer-column',()=>group('target',row.layer));
      const stack=e('span',undefined,'layer-stack');stack.style.height=row.total/max*100+'%';
      for(const [cls,value] of [['transfer',row.transfer],['local',row.total-row.transfer]]) {
        if(value>0){const mark=e('i',undefined,cls);mark.style.flex=String(value);stack.append(mark);}
      }
      b.append(stack);tooltip(b,'Layer '+row.layer+': '+ms(row.total)+' ideal budget; '+ms(row.transfer)+' transfer/wait');map.append(b);
    }
  }
  function futureCycle() {
    const path=$('#future-cycle-path');path.replaceChildren();
    for(const stage of ['proposal-1','proposal-2','target','catchup']) {
      const total=ideal.rows.filter(row=>row.stage===stage).reduce((sum,row)=>sum+row.time_us,0);
      const b=button(undefined,'phase-step pending-phase',()=>pending(stageNames[stage],total));
      b.dataset.state='pending';b.append(e('b',stageNames[stage]),e('span','-- ms'),e('span',undefined,'pending-rule'));
      tooltip(b,stageNames[stage]+': awaiting capture');path.append(b);
    }
    const map=$('#future-layer-map');map.replaceChildren();
    for(let layer=0;layer<45;layer++) {
      const total=ideal.rows.filter(row=>row.stage==='target'&&row.layer===layer).reduce((sum,row)=>sum+row.time_us,0);
      const b=button(undefined,'layer-column pending-layer',()=>pending('Target layer '+layer,total));
      b.dataset.state='pending';b.append(e('span',undefined,'pending-layer-track'));
      tooltip(b,'Layer '+layer+': no measurement yet');map.append(b);
    }
  }
  function timeline() {
    const target=$('#visual-timeline');target.replaceChildren();
    for(const rank of measured.devices) {
      const lane=e('div',undefined,'timeline-lane'), bins=e('div',undefined,'timeline-bins');
      lane.append(e('span','GPU '+rank.device));
      rank.bins.forEach((shares,i)=>{
        const start=profile.summary.span_us*i/rank.bins.length, end=profile.summary.span_us*(i+1)/rank.bins.length;
        const message='GPU '+rank.device+' / '+ms(start)+' to '+ms(end)+'. '+names.map((name,j)=>name+' '+pct(shares[j])).join(', ');
        const b=button(undefined,'timeline-bin',()=>inspect('GPU '+rank.device+' / captured window',body=>{
          body.append(e('p',ms(start)+' to '+ms(end)+' after first captured GPU activity.'));
          bars(body,names.map((label,j)=>({label,value:shares[j]*100,kind:classes[j]})),'%');
          body.append(e('p','Elapsed intervals are unioned, not summed. Gray is missing exported activity, not proven idle.'),
            button('Inspect GPU kernels',undefined,()=>raw({view:'kernels',device:String(rank.device)})));
        }));
        b.tabIndex=i===0?0:-1;
        shares.forEach((share,j)=>{const mark=e('i',undefined,classes[j]);mark.style.flex=String(share);b.append(mark);});
        tooltip(b,message);bins.append(b);
      });
      bins.addEventListener('keydown',event=>{
        if(!['ArrowLeft','ArrowRight','Home','End'].includes(event.key)) return;
        event.preventDefault();
        const children=[...bins.children], current=children.indexOf(document.activeElement);
        const next=event.key==='Home'?0:event.key==='End'?children.length-1:Math.max(0,Math.min(children.length-1,current+(event.key==='ArrowRight'?1:-1)));
        children.forEach((b,i)=>b.tabIndex=i===next?0:-1);children[next].focus();
      });
      lane.append(bins);target.append(lane);
    }
    $('#timeline-mid').textContent=number(profile.summary.span_us/2000,0)+' ms';
    $('#timeline-end').textContent=number(profile.summary.span_us/1000,0)+' ms';
  }
  function valueTable() {
    const table=e('table'),head=e('thead'),tr=e('tr'),body=e('tbody');
    ['Basis','GPU','GPU-local / ms','Transfer + wait / ms','Unrecorded / ms'].forEach(s=>tr.append(e('th',s)));
    head.append(tr);table.append(head);
    for(const [name,activity] of [['Ideal cycle',modeled],['Recorded capture',measured]]) for(const rank of activity.devices) {
      const row=e('tr');row.append(e('td',name),e('td',String(rank.device)));
      fields.forEach(field=>row.append(e('td',rank.known?number(rank[field]/1000,3):'Not modeled')));body.append(row);
    }
    table.append(body);
    const windows=e('details'), windowTable=e('table'), windowHead=e('thead'), windowRow=e('tr'), windowBody=e('tbody');
    windowTable.id='activity-windows';
    windowRow.append(e('th','Window / ms'));
    for(const rank of measured.devices) windowRow.append(e('th','GPU '+rank.device+' / local, transfer, unrecorded %'));
    windowHead.append(windowRow);windowTable.append(windowHead);
    const count=measured.devices[0]?.bins.length || 0;
    for(let i=0;i<count;i++) {
      const row=e('tr');
      row.append(e('td',number(profile.summary.span_us*i/count/1000,3)+' - '+number(profile.summary.span_us*(i+1)/count/1000,3)));
      for(const rank of measured.devices) row.append(e('td',rank.bins[i].map(pct).join(' / ')));
      windowBody.append(row);
    }
    windowTable.append(windowBody);windows.append(e('summary','All 96 capture windows'),windowTable);
    $('#activity-table').replaceChildren(table,windows);
  }
  $('#configure').addEventListener('click',()=>{
    opener=document.activeElement;close('#visual-inspector',false);$('#settings').hidden=false;
    $('#app').dataset.panel='settings';$('#configure').setAttribute('aria-expanded','true');$('#settings-close').focus();
  });
  $('#settings-close').addEventListener('click',()=>close('#settings'));
  $('#visual-inspector-close').addEventListener('click',()=>close('#visual-inspector'));
  document.querySelectorAll('[data-inspect]').forEach(b=>b.addEventListener('click',()=>{
    if(ideal&&profile) category(classes.indexOf(b.dataset.inspect));
  }));
  document.addEventListener('keydown',event=>{
    if(event.key!=='Escape')return;
    if(!$('#visual-inspector').hidden) close('#visual-inspector');
    else if(!$('#settings').hidden) close('#settings');
  });
  return {render(nextIdeal,nextProfile) {
    ideal=nextIdeal;profile=nextProfile;modeled=idealActivity(ideal);
    if(!profile.pending&&!cache.has(profile))cache.set(profile,profileActivity(profile));
    measured=profile.pending?null:cache.get(profile);
    const future=futureActivity(ideal.result.owners);
    close('#visual-inspector',false);
    composition($('#ideal-composition'),modeled.aggregate);composition($('#profile-composition'),future.aggregate);
    composition($('#mobile-ideal'),modeled.aggregate);composition($('#mobile-profile'),future.aggregate);
    chips($('#ideal-gpus'),modeled,'ideal');chips($('#profile-gpus'),future,'future');
    connections($('#ideal-connections'),'ideal');connections($('#profile-connections'),'future');
    cycle();futureCycle();
    $('#history-activity').hidden=profile.pending;
    if(!profile.pending){timeline();valueTable();}
    else {
      $('#visual-timeline').replaceChildren();
      $('#activity-table').textContent='No raw capture loaded. Choose historical evidence above or import a profile.';
    }
    $('#ideal-note').textContent=modeled.balanced
      ?'Balanced ownership assumed. Each square is 1% of modeled time.'
      :'Unequal ownership. Only busiest-owner timing is modeled; other GPU times remain unknown.';
  }};
}
