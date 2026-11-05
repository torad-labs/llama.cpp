"""Offline evidence for a granular ideal-versus-profile comparison page."""
import argparse
from bisect import bisect_right
from collections import defaultdict
import gzip
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import re
import sqlite3
from urllib.parse import parse_qs, urlsplit

import estimate_decode_pcie as roofline

ROOT = Path(__file__).resolve().parents[2]
CATCH = {
    'nextn.eh_proj.weight', 'nextn.enorm.weight', 'nextn.hnorm.weight', 'attn_norm.weight',
    'attn_kv_a_mqa.weight', 'attn_kv_a_norm.weight', 'indexer.attn_k.weight',
    'indexer.k_norm.weight', 'indexer.k_norm.bias', 'indexer_compressor_gate.weight',
    'indexer_compressor_ape.weight',
}


def merged(intervals):
    result = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1][1] = max(end, result[-1][1])
        else:
            result.append([start, end])
    return result


def union_ns(intervals):
    return sum(end - start for start, end in merged(intervals))


def stats(durations):
    values = sorted(durations)
    if not values:
        return dict(calls=0, sum_us=0, mean_us=0, p50_us=0, p95_us=0, max_us=0)
    return dict(calls=len(values), sum_us=sum(values) / 1000,
                mean_us=sum(values) / len(values) / 1000,
                p50_us=values[math.ceil(len(values) * .5) - 1] / 1000,
                p95_us=values[math.ceil(len(values) * .95) - 1] / 1000,
                max_us=values[-1] / 1000)


def operation(label):
    match = re.search(r' r=(\d+) w=(\d+)(?: w0=(\S+))?', label or '')
    if not match or label.startswith('fused '):
        return None
    name = label[:match.start()]
    weight = match[3]
    layer = re.search(r'blk\.(\d+)\.', weight or '') or re.search(r'-(\d+)(?:\s|$)', name)
    return dict(name=name, weight=weight, layer=int(layer[1]) if layer else None,
                logical_read_bytes=int(match[1]), logical_write_bytes=int(match[2]))


def family(name):
    if 'ggml_cuda_ar_' in name:
        return 'Expert / sampling gather' if ('unsigned int' in name or 'uint32' in name) else 'Expert reduction'
    for fragment, label in (
        ('fattn_stage_host_rows', 'Host KV staging'), ('flash_attn', 'Attention'),
        ('fattn_', 'Attention'), ('kpool', 'Indexer / selection'),
        ('gated_delta', 'KDA recurrent state'), ('gdn', 'KDA recurrent state'),
        ('mmvq_moe', 'Routed expert matmul'), ('moe', 'Expert routing / matmul'),
        ('mul_mat', 'Projection matmul'), ('quantize', 'Quantization'),
        ('set_rows', 'Cache / row writes'), ('hc_', 'Hyper-connection'),
    ):
        if fragment in name:
            return label
    return 'Other GPU work'


class Ranges:
    def __init__(self, records):
        self.rows = defaultdict(list)
        for index, row in enumerate(records):
            if row['end'] is not None and row['end'] > row['start'] and operation(row['label']):
                self.rows[row['globalTid']].append((row['start'], row['end'], index))
        self.starts = {}
        self.ends = {}
        for tid, items in self.rows.items():
            items.sort()
            self.starts[tid] = [x[0] for x in items]
            running = 0
            self.ends[tid] = []
            for _, end, _ in items:
                running = max(running, end)
                self.ends[tid].append(running)

    def at(self, tid, time):
        items = self.rows.get(tid, [])
        i = bisect_right(self.starts.get(tid, []), time) - 1
        best = None
        span = math.inf
        while i >= 0 and self.ends[tid][i] > time:
            start, end, index = items[i]
            if start <= time < end and end - start < span:
                best, span = index, end - start
            i -= 1
        return best


def export_trace(path, metadata):
    """Read only whitelisted Nsight tables; never export environment or prompt text."""
    path = Path(path)
    db = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA query_only=ON')
    tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    required = {'StringIds', 'CUPTI_ACTIVITY_KIND_KERNEL'}
    if not required <= tables:
        db.close()
        raise ValueError('Not a supported Nsight SQLite export: kernel/StringIds tables missing')
    optional = ['NVTX_EVENTS', 'CUDA_GRAPH_NODE_EVENTS', 'CUPTI_ACTIVITY_KIND_RUNTIME',
                'CUPTI_ACTIVITY_KIND_MEMCPY', 'TARGET_INFO_GPU']
    strings = {row['id']: row['value'] for row in db.execute('SELECT id,value FROM StringIds')}
    nvtx = []
    if 'NVTX_EVENTS' in tables:
        for row in db.execute('SELECT start,end,globalTid,text,textId FROM NVTX_EVENTS'):
            nvtx.append(dict(row, label=row['text'] or strings.get(row['textId'], '')))
    ranges = Ranges(nvtx)
    nodes = {}
    origins = {}
    pid_of = lambda tid: (int(tid) >> 24) << 24
    if 'CUDA_GRAPH_NODE_EVENTS' in tables:
        for row in db.execute('SELECT start,globalTid,graphNodeId,originalGraphNodeId FROM CUDA_GRAPH_NODE_EVENTS ORDER BY start'):
            if row['globalTid'] is None:
                continue
            key = (pid_of(row['globalTid']), row['graphNodeId'])
            original = row['originalGraphNodeId']
            if original and original != row['graphNodeId']:
                origins[key] = (key[0], original)
            found = ranges.at(row['globalTid'], row['start'])
            if found is not None:
                nodes[key] = found

    def graph_range(key):
        seen = set()
        while key not in seen:
            if key in nodes:
                return nodes[key]
            seen.add(key)
            if key not in origins:
                break
            key = origins[key]
        return None

    runtime = []
    launch = {}
    if 'CUPTI_ACTIVITY_KIND_RUNTIME' in tables:
        runtime = [dict(row) for row in db.execute(
            'SELECT start,end,globalTid,correlationId,nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME')]
        for row in runtime:
            launch[(pid_of(row['globalTid']), row['correlationId'])] = row
    raw = [dict(row) for row in db.execute(
        'SELECT start,end,deviceId,globalPid,correlationId,demangledName,shortName,graphNodeId '
        'FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start')]
    copies = [dict(row) for row in db.execute(
        'SELECT start,end,deviceId,bytes,copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY ORDER BY start'
    )] if 'CUPTI_ACTIVITY_KIND_MEMCPY' in tables else []
    if not raw:
        db.close()
        raise ValueError('Capture has no GPU kernels')
    origin = min(row['start'] for row in raw + copies)
    end = max(row['end'] for row in raw + copies)
    kernel_groups, op_groups, api_groups, copy_groups = {}, {}, {}, {}
    kernel_events, copy_events, api_events = [], [], []
    graph_attributed = direct_attributed = 0
    for row in raw:
        name = strings.get(row['demangledName'], strings.get(row['shortName'], 'Unknown kernel'))
        key = (name, row['deviceId'])
        group = kernel_groups.setdefault(key, dict(index=len(kernel_groups), name=name,
                    device=row['deviceId'], durations=[], intervals=[], family=family(name)))
        group['durations'].append(row['end'] - row['start'])
        group['intervals'].append((row['start'], row['end']))
        ni = graph_range((row['globalPid'], row['graphNodeId'])) if row['graphNodeId'] else None
        from_graph = ni is not None
        if ni is None:
            api = launch.get((row['globalPid'], row['correlationId']))
            ni = ranges.at(api['globalTid'], api['start']) if api else None
        oi = -1
        if ni is not None:
            label = nvtx[ni]['label']
            op = operation(label)
            if op:
                og = op_groups.setdefault((label, row['deviceId']), dict(index=len(op_groups),
                    **op, device=row['deviceId'], instances_raw=defaultdict(list), kernel_calls=0))
                instance = (ni, row['correlationId'] if from_graph else 'direct')
                og['instances_raw'][instance].append((row['start'], row['end']))
                og['kernel_calls'] += 1
                oi = og['index']
                graph_attributed += int(from_graph)
                direct_attributed += int(not from_graph)
        kernel_events.append([row['start'] - origin, row['end'] - row['start'],
                              row['deviceId'], group['index'], oi])

    kernels = []
    for group in kernel_groups.values():
        kernels.append(dict(index=group['index'], name=group['name'], device=group['device'],
                            family=group['family'], **stats(group['durations']),
                            union_us=union_ns(group['intervals']) / 1000,
                            measured_bytes=None, gbps=None))
    operations = []
    for group in op_groups.values():
        instances = list(group.pop('instances_raw').values())
        spans = [max(x[1] for x in events) - min(x[0] for x in events) for events in instances]
        operations.append(dict(**group, **stats(spans), instances=len(instances),
                               gpu_sum_us=sum(b-a for evs in instances for a,b in evs) / 1000,
                               gpu_union_us=sum(union_ns(evs) for evs in instances) / 1000,
                               byte_basis='NVTX logical estimate, not measured DRAM or PCIe traffic'))
    directions = {1: 'Host to GPU', 2: 'GPU to host', 8: 'Device to device', 10: 'Peer to peer'}
    for row in copies:
        key = (row['copyKind'], row['deviceId'])
        group = copy_groups.setdefault(key, dict(index=len(copy_groups),
                   direction=directions.get(row['copyKind'], 'Other copy'),
                   copy_kind=row['copyKind'], device=row['deviceId'], bytes=0, durations=[]))
        group['bytes'] += row['bytes']
        group['durations'].append(row['end'] - row['start'])
        copy_events.append([row['start'] - origin, row['end'] - row['start'],
                            row['deviceId'], row['bytes'], group['index']])
    copy_output = []
    for group in copy_groups.values():
        s = stats(group.pop('durations'))
        copy_output.append(dict(**group, **s, gbps=group['bytes'] / (s['sum_us'] * 1000) if s['sum_us'] else None))
    for row in runtime:
        name = strings.get(row['nameId'], 'Unknown CUDA API')
        group = api_groups.setdefault(name, dict(index=len(api_groups), name=name, durations=[]))
        group['durations'].append(row['end'] - row['start'])
        api_events.append([row['start'] - origin, row['end'] - row['start'], group['index']])
    apis = [dict(index=g['index'], name=g['name'], **stats(g['durations'])) for g in api_groups.values()]
    devices = []
    for device in sorted({row['deviceId'] for row in raw}):
        intervals = [(row['start'], row['end']) for row in raw if row['deviceId'] == device]
        active = merged(intervals)
        bins = []
        for i in range(96):
            a, b = origin + (end-origin)*i/96, origin + (end-origin)*(i+1)/96
            occupied = sum(max(0, min(y,b) - max(x,a)) for x,y in active)
            bins.append(round(occupied / (b-a) if b>a else 0, 5))
        devices.append(dict(device=device, kernel_union_us=union_ns(intervals)/1000,
                            kernel_sum_us=sum(b-a for a,b in intervals)/1000, bins=bins))
    gpu_names = [row[0] for row in db.execute('SELECT name FROM TARGET_INFO_GPU')] if 'TARGET_INFO_GPU' in tables else []
    db.close()
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(chunk)
    digest = checksum.hexdigest()
    attributed = graph_attributed + direct_attributed
    return dict(schema='glm53-profile-v1', metadata=metadata, source=dict(file=path.name, sha256=digest),
                hardware=gpu_names, counts=dict(kernels=len(raw), copies=len(copies), runtime=len(runtime), nvtx=len(nvtx)),
                coverage=dict(attributed_kernels=attributed, unattributed_kernels=len(raw)-attributed,
                              graph_attributed=graph_attributed, direct_attributed=direct_attributed),
                missing_tables=[name for name in optional if name not in tables],
                summary=dict(span_us=(end-origin)/1000,
                             kernel_sum_us=sum(r['end']-r['start'] for r in raw)/1000,
                             kernel_union_us=union_ns([(r['start'],r['end']) for r in raw])/1000),
                kernels=kernels, operations=operations, copies=copy_output, apis=apis, devices=devices,
                kernel_events=kernel_events, copy_events=copy_events, api_events=api_events,
                event_columns=dict(kernel=['start_ns','duration_ns','device','kernel_index','operation_index'],
                                   copy=['start_ns','duration_ns','device','bytes','copy_index'],
                                   api=['start_ns','duration_ns','api_index']),
                caveats=[
                    'Recorded request is not a cycle-aligned 40-session/400k benchmark.',
                    'NVTX bytes are source-derived logical estimates, not measured memory or PCIe counters.',
                    'Kernel-driven peer and mapped-host traffic does not appear as explicit CUDA memcpy bytes.',
                    'Operation attribution follows direct launch ranges and graph-node creation/clone ancestry; unknown stays unknown.',
                    'Operation latency spans kernels in one invocation; GPU sum and interval union are separate.',
                    'GPU work summed across cards and CPU API sums are not critical-path wall time.',
                    'Nearest-rank percentiles; all captured kernel/copy/API events retained at nanosecond resolution.',
                ])


def build_ideal(tensors, inputs, **scenario):
    result = roofline.estimate(inputs, **scenario)
    if result['depth'] != 2 or result['catchup'] != 'cache-only' or result['overlap_shared']:
        raise ValueError('Detail ledger currently covers MTP2/cache-only catch-up without overlap credit')
    rows = []
    n = result['sessions']
    owners = max(result['owners'])
    bw = inputs['gpu_bandwidth_bytes_per_second'] * result['parameters']['dram_efficiency']
    p = lambda count: 1 - (1 - 8 / 288) ** count
    expert_rows = n * (3 if result['expert_span'] == 'independent' else 1)

    def add(stage, layer, kind, name, read=0, write=0, time=None, **extra):
        rows.append(dict(id=str(len(rows)), stage=stage, layer=layer, kind=kind, name=name,
                         read_bytes=read, write_bytes=write,
                         time_us=(read+write)/bw*1e6 if time is None else time,
                         basis=('Explicit latency assumption, not a measurement' if kind == 'latency' else
                                'PCIe payload budget at endpoint/shared-fabric limits' if extra.get('route') else
                                'Ideal GDDR read/write budget on busiest GPU; not kernel execution time'),
                         **extra))

    for tensor in tensors:
        name, size = tensor['name'], tensor['bytes']
        m = re.match(r'blk\.(\d+)\.(.+)', name)
        if m:
            layer, suffix = int(m[1]), m[2]
            expert = '_exps.' in name
            if layer < 45:
                read = size*p(expert_rows)/4 if expert else size
                add('target', layer, 'expert-weight' if expert else 'weight', suffix,
                    read=read, weight=name, quant=tensor['type'], tensor_bytes=size)
            elif layer == 45:
                for stage in ('proposal-1', 'proposal-2'):
                    read = size*p(n)/4 if expert else size
                    add(stage, 45, 'expert-weight' if expert else 'weight', suffix,
                        read=read, weight=name, quant=tensor['type'], tensor_bytes=size)
                if suffix in CATCH:
                    add('catchup', 45, 'weight', suffix, read=size, weight=name,
                        quant=tensor['type'], tensor_bytes=size)
        elif name in ('output.weight', 'output_norm.weight'):
            add('target', None, 'weight', name, read=size, weight=name, quant=tensor['type'], tensor_bytes=size)
    for stage in ('proposal-1', 'proposal-2'):
        add(stage, 45, 'weight', 'Draft shortlist projection', read=roofline.SHORTLIST,
            weight='derived.draft_shortlist', quant='Q8_0', tensor_bytes=roofline.SHORTLIST)
    for layer in range(45):
        if layer in range(3, 44, 4):
            add('target', layer, 'index-scan', 'Pooled index key scan', read=owners*inputs['index_key_bytes_per_layer_scan'])
            add('target', layer, 'latent-read', 'Selected latent KV read', read=owners*inputs['selected_latent_bytes_per_layer'])
        else:
            for label, size in (('KDA matrix', 1114112), ('Convolution history', 294912)):
                add('target', layer, 'state-read', label+' read', read=owners*size)
                for plane in range(3):
                    add('target', layer, 'state-write', label+' output '+str(plane+1), write=owners*size)
    for stage in ('proposal-1', 'proposal-2'):
        add(stage, 45, 'index-scan', 'Pooled index key scan', read=owners*inputs['index_key_bytes_per_layer_scan'])
        add(stage, 45, 'latent-read', 'Selected latent KV read', read=owners*inputs['selected_latent_bytes_per_layer'])
    add('catchup', 45, 'cache-write', 'New latent rows', write=owners*3*544)
    add('catchup', 45, 'cache-write', 'New indexer rows', write=owners*3*408)

    for event in result['events']:
        if event['stage'] == 'target':
            layers = list(range(3,45)) if event['kind'] in ('gather','reduce') else list(range(3,44,4))
            places = [('target', layers[i]) for i in range(event['repeat'])] if event['repeat']>1 else [('target',None)]
        elif event['stage'] == 'proposal':
            places = [('proposal-'+str(i+1),45) for i in range(event['repeat'])]
        elif event['stage'] == 'each-graph':
            places = [(stage,None) for stage in ('proposal-1','proposal-2','target','catchup')]
        else:
            places = [('catchup',45)] * event['repeat']
        for stage, layer in places:
            detail = dict(route=event['route'], sent_bytes=event['sent_bytes'],
                          received_bytes=event['received_bytes'], dependency=event['dependency'],
                          host_endpoint_bytes=event['host_endpoint_bytes'])
            add(stage, layer, event['kind'], event['kind'].replace('-',' ').capitalize()+' payload',
                time=event['payload_ms']*1000, **detail)
            if event['latency_ms']:
                add(stage, layer, 'latency', event['kind'].replace('-',' ').capitalize()+' latency budget',
                    time=event['latency_ms']*1000, dependency=event['dependency'])
            if event['kind'] == 'kv-stage':
                staged = max(event['received_bytes'])
                add(stage,layer,'staging-scratch','Staging scratch writes and extra attention reads',
                    write=staged, read=staged-owners*roofline.SELECTED)
    total = sum(row['time_us'] for row in rows)
    if not math.isclose(total, result['cycle_ms']*1000, rel_tol=1e-9, abs_tol=1e-6):
        raise ValueError('Detail ledger does not reconcile with parent estimate: '+str(total))
    return dict(schema='glm53-ideal-v1', result={k:v for k,v in result.items() if k!='events'}, rows=rows,
                omissions=['Quantization instructions, arithmetic throughput and intermediate activation traffic are not timed.',
                           'Tensor read budgets are not one-to-one CUDA kernel predictions; fused operations cannot be split by assertion.',
                           'Layers retain their dependency boundaries; ordering of weight entries inside a layer is a budget, not a traced schedule.'])


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, separators=(',',':'), allow_nan=False).encode()
    if path.suffix == '.gz':
        data = gzip.compress(data, mtime=0)
    path.write_bytes(data)


def make_server(directory, port):
    directory = Path(directory)
    pack = json.loads((directory/'data/ideal.json').read_text())
    converters = dict(sessions=int, protocol=str, kv=str, expert_span=str,
                      peer_gbps=float, fabric_gbps=float, host_gbps=float, host_fabric_gbps=float,
                      collective_us=float, host_stage_us=float, graph_boundary_us=float,
                      dram_efficiency=float, accepted=int, control_bytes=int)

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(directory), **kwargs)

        def end_headers(self):
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.send_header('Content-Security-Policy',
                             "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
                             "font-src 'self' https://fonts.gstatic.com; connect-src 'self'; object-src 'none'; base-uri 'none'")
            super().end_headers()

        def do_GET(self):
            url = urlsplit(self.path)
            if url.path != '/api/ideal':
                return super().do_GET()
            try:
                query = parse_qs(url.query, keep_blank_values=True)
                if any(key not in converters or len(value)!=1 for key,value in query.items()):
                    raise ValueError('Unknown or repeated model parameter')
                scenario = {key:converters[key](value[0]) for key,value in query.items()}
                result = build_ideal(pack['tensors'], pack['inputs'], **scenario)
                body = json.dumps(result, separators=(',',':'), allow_nan=False).encode()
                self.send_response(200)
            except (ValueError, TypeError) as error:
                body = json.dumps(dict(error=str(error))).encode()
                self.send_response(400)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return ThreadingHTTPServer(('127.0.0.1', port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    profile = sub.add_parser('profile')
    profile.add_argument('--sqlite', type=Path, required=True)
    profile.add_argument('--output', type=Path, required=True)
    profile.add_argument('--label', required=True)
    profile.add_argument('--active', type=int, required=True)
    profile.add_argument('--context', type=int, required=True)
    profile.add_argument('--runtime', default='unknown')
    profile.add_argument('--slots', type=int)
    profile.add_argument('--depth-max', type=int)
    profile.add_argument('--depth-policy', choices=('unknown', 'adaptive', 'fixed-2'), default='unknown')
    profile.add_argument('--kv', choices=('unknown', 'host', 'resident'), default='unknown')
    profile.add_argument('--scope', choices=('unknown', 'request', 'cycle'), default='unknown')
    profile.add_argument('--measurements', type=Path)
    ideal = sub.add_parser('ideal')
    ideal.add_argument('--manifest', type=Path, required=True)
    ideal.add_argument('--output', type=Path, required=True)
    serve = sub.add_parser('serve')
    serve.add_argument('--port', type=int, default=8768)
    serve.add_argument('--directory', type=Path, default=ROOT/'docs/torad/decode-lab')
    args = parser.parse_args()
    if args.command == 'serve':
        server = make_server(args.directory, args.port)
        print('Decode comparison: http://127.0.0.1:'+str(server.server_port), flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
        return
    if args.command == 'profile':
        metadata = dict(label=args.label, active=args.active, context=args.context,
                        runtime=args.runtime, configured_slots=args.slots, depth_max=args.depth_max,
                        depth_policy=args.depth_policy, kv=args.kv,
                        scope={'unknown': 'Unsegmented capture; phase boundaries not established',
                               'request': 'Captured request, including prompt tail; not isolated cycles',
                               'cycle': 'User-declared cycle; operation rows still have per-invocation units'}[args.scope])
        if args.measurements:
            records = json.loads(args.measurements.read_text())
            record = next(r for r in records if r.get('label') == args.label)
            metadata['timings'] = record['timings']
            metadata['request_wall_seconds'] = record['wall_seconds']
        value = export_trace(args.sqlite, metadata)
    else:
        manifest = json.loads(args.manifest.read_text())
        tensors = [t for shard in manifest['shards'] for t in shard['tensors']]
        inputs = json.loads((ROOT/'docs/torad/glm53-decode-roofline.json').read_text())['inputs']
        value = dict(schema='glm53-ideal-pack-v1', inputs=inputs,
                     tensors=[{k:t[k] for k in ('name','bytes','type')} for t in tensors],
                     detail=build_ideal(tensors,inputs))
    write_json(args.output,value)
    print(json.dumps(dict(output=str(args.output),schema=value['schema'],
                          counts=value.get('counts'),rows=len(value.get('rows',[])))))


if __name__ == '__main__':
    main()
