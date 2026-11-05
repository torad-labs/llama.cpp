"""Offline DP4 estimate with per-layer PCIe dependencies, not measured throughput."""
import argparse
import json
import math
from pathlib import Path

GPUS = 4
ROW = 4096 * 4
SELECTED = 2051 * 544
SHARED_WEIGHTS = 3 * 2048 * 4096 * 34 // 32
SHORTLIST = 98304 * 4096 * 34 // 32
INPUT = Path(__file__).resolve().parents[2] / 'docs/torad/glm53-decode-roofline.json'


def collective(sessions, rows_per_sequence, kind, protocol):
    """Bytes cross each GPU endpoint; peer fabric bytes count each send once."""
    if sessions < 1 or rows_per_sequence < 1 or kind not in ('gather', 'reduce'):
        raise ValueError('Positive rows and gather/reduce are required')
    counts = [len(range(rank, sessions, GPUS)) * rows_per_sequence for rank in range(GPUS)]
    rows = sum(counts)
    payload = rows * ROW
    if protocol == 'optimized':
        width = ROW if kind == 'gather' else ROW // 2
        sent = [3 * n * width if kind == 'gather' else (rows - n) * width for n in counts]
        received = [(rows - n) * width if kind == 'gather' else 3 * n * width for n in counts]
        route = 'owner-slice-peer' if kind == 'gather' else 'destination-reduce-peer'
    elif protocol == 'current':
        wire = payload if kind == 'gather' else payload // 2
        if kind == 'reduce' and wire >= 1024 * 1024:
            route = 'peer-copy-engine'
        elif wire <= 128 * 1024:
            route = 'peer-ll'
        else:
            route = 'peer-chunked' if kind == 'gather' else 'host-chunked'
        if route == 'host-chunked':
            sent, received = [wire] * GPUS, [3 * wire] * GPUS
        else:
            flags = 2 if route == 'peer-ll' else 1
            sent = [3 * wire * flags] * GPUS
            received = sent.copy()
    else:
        raise ValueError('Only current or optimized DP4 transport is modeled')
    return {
        'kind': kind, 'route': route, 'sent_bytes': sent, 'received_bytes': received,
        'host_endpoint_bytes': sum(sent) + sum(received) if route == 'host-chunked' else 0,
    }


def transfer_ms(event, *, peer_gbps=54, fabric_gbps=216, host_gbps=40, host_fabric_gbps=128):
    """Peer full duplex uses max; host publish then consume uses dependent legs."""
    sent, received = event['sent_bytes'], event['received_bytes']
    if event['host_endpoint_bytes']:
        upload = max(max(received) / host_gbps, sum(received) / host_fabric_gbps)
        download = max(max(sent) / host_gbps, sum(sent) / host_fabric_gbps)
        return (upload + download) / 1e6
    return max(max(sent) / peer_gbps, max(received) / peer_gbps,
               sum(sent) / fabric_gbps) / 1e6


def estimate(inputs, *, sessions=40, depth=2, accepted=None, expert_span='independent',
             protocol='optimized', kv='resident', catchup='cache-only', overlap_shared=False,
             peer_gbps=54, fabric_gbps=216, host_gbps=40, host_fabric_gbps=128,
             collective_us=5, host_stage_us=5, graph_boundary_us=20,
             control_bytes=65536, dram_efficiency=1.0, raw_logits=False):
    """A serial layer chain, with optional shared-expert work under its own transfers."""
    if type(sessions) is not int or not 1 <= sessions <= 40:
        raise ValueError('This estimate covers 1..40 consecutive active sequence IDs')
    if type(depth) is not int or depth not in (0, 2):
        raise ValueError('Only target-only or MTP2 is modeled')
    accepted = depth if accepted is None else accepted
    if type(accepted) is not int or not 0 <= accepted <= depth:
        raise ValueError('Accepted drafts must be in 0..depth')
    if protocol not in ('current', 'optimized') or catchup not in ('cache-only', 'full'):
        raise ValueError('Unknown DP transport or catch-up schedule')
    if kv not in ('resident', 'host-per-query', 'host-reuse') or (kv != 'resident' and depth != 2):
        raise ValueError('Host KV requires the MTP2 profile')
    if expert_span not in ('independent', 'same-within-sequence'):
        raise ValueError('Unknown expert reuse scenario')
    if overlap_shared and protocol != 'optimized':
        raise ValueError('Shared overlap is a proposed schedule, not the current transport')
    for value in (peer_gbps, fabric_gbps, host_gbps, host_fabric_gbps):
        if not math.isfinite(value) or value <= 0:
            raise ValueError('Bandwidth assumptions must be finite and positive')
    for value in (collective_us, host_stage_us, graph_boundary_us, control_bytes):
        if not math.isfinite(value) or value < 0:
            raise ValueError('Latency and control budgets cannot be negative')
    if not 0 < dram_efficiency <= 1:
        raise ValueError('DRAM efficiency must be in (0, 1]')
    if inputs['context_tokens_per_session'] != 400000:
        raise ValueError('This evidence pack and sparse-KV model are for 400k context')

    counts = [len(range(rank, sessions, GPUS)) for rank in range(GPUS)]
    owners = max(counts)
    queries = depth + 1
    touched = lambda rows: 1 - (1 - 8 / 288) ** rows
    expert_rows = sessions * (queries if expert_span == 'independent' else 1)
    index = inputs['index_key_bytes_per_layer_scan']
    latent = inputs['selected_latent_bytes_per_layer']
    state = inputs['current_recurrent_and_conv_bytes_per_sequence']
    bw = inputs['gpu_bandwidth_bytes_per_second'] * dram_efficiency
    target = (inputs['target_dense_read_bytes']
              + inputs['target_all_expert_bytes'] * touched(expert_rows) / GPUS
              + owners * (11 * (index + latent) + (depth + 2) * state))
    proposal = depth * (inputs['draft_dense_and_shortlist_read_bytes']
                       + inputs['draft_all_expert_bytes'] * touched(sessions) / GPUS
                       + owners * (index + latent))
    catch = inputs['draft_cache_only_read_bytes'] + owners * queries * 952 if depth else 0
    if depth and catchup == 'full':
        catch = (inputs['draft_dense_and_shortlist_read_bytes'] - SHORTLIST
                 + inputs['draft_all_expert_bytes'] * touched(expert_rows) / GPUS
                 + owners * (index + latent + queries * 952))

    events = []
    overlaps = []
    links = dict(peer_gbps=peer_gbps, fabric_gbps=fabric_gbps,
                 host_gbps=host_gbps, host_fabric_gbps=host_fabric_gbps)

    def add(event, stage, repeat, latency_us, dependency):
        event.update(stage=stage, repeat=repeat, dependency=dependency,
                     payload_ms=transfer_ms(event, **links), latency_ms=latency_us / 1000)
        event['total_ms'] = repeat * (event['payload_ms'] + event['latency_ms'])
        events.append(event)
        return event

    def host(kind, stage, sent, received, repeat, latency, dependency):
        return add(dict(kind=kind, route='host', sent_bytes=sent, received_bytes=received,
                        host_endpoint_bytes=sum(sent) + sum(received)),
                   stage, repeat, latency, dependency)

    def moe(stage, q, layers):
        g = add(collective(sessions, q, 'gather', protocol), stage, layers, collective_us,
                'owner activation -> gather -> routed experts')
        r = add(collective(sessions, q, 'reduce', protocol), stage, layers, collective_us,
                'routed experts -> reduction -> local shared join -> next layer')
        if overlap_shared:
            per_layer = g['payload_ms'] + r['payload_ms'] + 2 * collective_us / 1000
            overlaps.append(layers * min(SHARED_WEIGHTS / bw * 1000, per_layer))

    # Repetition compresses identical layers, not a single global communication phase.
    moe('target', queries, 42)
    if depth:
        moe('proposal', 1, depth)
        if catchup == 'full':
            moe('catchup', queries, 1)

    extra_dram = 0
    if kv != 'resident':
        passes = [('target', queries, 11), ('proposal', 1, depth)]
        if catchup == 'full':
            passes.append(('catchup', queries, 1))
        for stage, q, repeat in passes:
            staged_q = q if kv == 'host-per-query' else 1
            received = [n * staged_q * SELECTED for n in counts]
            host('kv-stage', stage, [0] * GPUS, received, repeat, host_stage_us,
                 'GPU index selection -> host latent rows -> GPU scratch -> attention')
            # Scratch writes plus attention rereads beyond the resident ideal-reuse term.
            extra_dram += repeat * (2 * max(received) - owners * SELECTED)
        for stage, q, repeat in [('target', queries, 11), ('proposal', 1, depth),
                                  ('catchup', queries, 1)]:
            host('kv-append', stage, [n * q * 544 for n in counts], [0] * GPUS,
                 repeat, host_stage_us, 'new latent projection -> host cache write')

    graphs = depth + 2 if depth else 1
    host('control-upload', 'each-graph', [0] * GPUS, [control_bytes] * GPUS,
         graphs, graph_boundary_us / 2, 'CPU tokens/positions/metadata -> graph')
    host('control-readback', 'each-graph', [control_bytes] * GPUS, [0] * GPUS,
         graphs, graph_boundary_us / 2, 'GPU sampling/control -> CPU scheduling')
    if depth:
        host('hidden-upload', 'proposal', [0] * GPUS, [n * ROW for n in counts],
             depth, host_stage_us, 'CPU pending/proposal hidden row -> draft input')
        host('hidden-readback', 'proposal', [n * ROW for n in counts], [0] * GPUS,
             depth - 1, host_stage_us, 'nonterminal draft hidden row -> CPU -> next proposal')
        device_h = sessions * queries >= 64
        carry_q = 1 if device_h else queries
        host('hidden-upload', 'catchup', [0] * GPUS, [n * carry_q * ROW for n in counts],
             1, host_stage_us, 'carry or shifted CPU hidden rows -> draft catch-up')
        host('hidden-readback', 'target', [n * queries * ROW for n in counts], [0] * GPUS,
             1, host_stage_us, 'verification hidden rows -> CPU retained state/acceptance')
    if raw_logits:
        host('raw-logits', 'target', [n * queries * 154880 * 4 for n in counts], [0] * GPUS,
             1, host_stage_us, 'full target logits -> CPU sampler/grammar fallback')

    dram_ms = (target + proposal + catch + extra_dram) / bw * 1000
    pcie_ms = sum(e['total_ms'] for e in events)
    overlap_ms = sum(overlaps)
    cycle_ms = dram_ms + pcie_ms - overlap_ms
    tps = (accepted + 1) * 1000 / cycle_ms
    return {
        'sessions': sessions, 'context': 400000, 'depth': depth, 'accepted': accepted,
        'expert_span': expert_span, 'protocol': protocol, 'kv': kv, 'catchup': catchup,
        'overlap_shared': overlap_shared, 'owners': counts,
        'parameters': dict(**links, collective_us=collective_us, host_stage_us=host_stage_us,
                           graph_boundary_us=graph_boundary_us, control_bytes=control_bytes,
                           dram_efficiency=dram_efficiency, raw_logits=raw_logits),
        'collective_boundaries': sum(e['repeat'] for e in events if e['kind'] in ('gather', 'reduce')),
        'dram_ms': dram_ms, 'pcie_ms': pcie_ms, 'overlap_ms': overlap_ms,
        'exposed_pcie_ms': pcie_ms - overlap_ms, 'cycle_ms': cycle_ms,
        'tps_per_session': tps, 'aggregate_tps': sessions * tps, 'events': events,
        'hardware_validated': False, 'capacity_qualified': False,
        'assumptions': [
            'DP4 only, consecutive sequence IDs, equal 400k histories, no prefill or admission.',
            'Exact tensor-derived weight inputs; ideal per-batch DRAM reuse and balanced routed experts.',
            'Bandwidths and phase latencies are scenario inputs, not concurrent-fabric measurements.',
            'Peer full duplex uses the slowest endpoint or shared-fabric limit, not send plus receive.',
            'Host publish/consume legs are serialized; no overlap with other attention or MoE layers.',
            'Current protocol assumes peer probes succeed; fallback failures and graph recapture are excluded.',
            'Optimized protocol, cache-only catch-up, host reuse and shared overlap are unimplemented hypotheses.',
            'Shared overlap hides only its own layer transfers, using at most that layer shared-weight time.',
            'Host-per-query is a no-reuse valid-row envelope; host-reuse assumes identical selections within a sequence.',
            'Small control traffic is an explicit per-GPU per-direction per-graph budget, not a source-derived byte count.',
            'Current hidden-state CPU handoffs retained; raw-logit fallback optional, CPU sampler work excluded.',
            'Temporary traffic, quantization/compute time, stragglers, replay and thermal limits are not measured.',
            'No universal maximum, numerical-preservation proof, capacity qualification or serving guarantee.',
        ],
    }


def sweep(inputs):
    """Preset sensitivity cases; none is a measurement of the four-card fabric."""
    cases = {
        'link-limit': dict(peer_gbps=63, fabric_gbps=252, host_gbps=63, host_fabric_gbps=252,
                           collective_us=0, host_stage_us=0, graph_boundary_us=0),
        'reference': {},
        'contended': dict(peer_gbps=27, fabric_gbps=54, host_gbps=20, host_fabric_gbps=64,
                          collective_us=20, host_stage_us=10, graph_boundary_us=50),
        'host-reuse': dict(kv='host-reuse'),
        'host-per-query': dict(kv='host-per-query'),
        'current-transport': dict(protocol='current'),
        'current-full-resident': dict(protocol='current', catchup='full'),
        'current-full-host': dict(protocol='current', catchup='full', kv='host-per-query'),
        'shared-overlap': dict(overlap_shared=True),
        'one-draft': dict(accepted=1),
        '70pct-one-draft': dict(accepted=1, dram_efficiency=0.7),
    }
    results = []
    counts = (1, 4, 8, 12, 24, 36, 40)
    for name, options in cases.items():
        for n in counts if name == 'reference' else (40,):
            for span in ('independent', 'same-within-sequence'):
                results.append(dict(scenario=name, **estimate(inputs, sessions=n, expert_span=span, **options)))
    for n in counts:
        results.append(dict(scenario='target-only', **estimate(inputs, sessions=n, depth=0)))
    return dict(layout='dp4', model_revision=inputs['model_revision'],
                hardware_validated=False, results=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, default=INPUT)
    parser.add_argument('--sweep', action='store_true', help='Run the documented presets; ignores scenario flags')
    parser.add_argument('--sessions', type=int, default=40)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--accepted', type=int)
    parser.add_argument('--expert-span', choices=('independent', 'same-within-sequence'), default='independent')
    parser.add_argument('--protocol', choices=('optimized', 'current'), default='optimized')
    parser.add_argument('--kv', choices=('resident', 'host-per-query', 'host-reuse'), default='resident')
    parser.add_argument('--catchup', choices=('cache-only', 'full'), default='cache-only')
    parser.add_argument('--overlap-shared', action='store_true')
    parser.add_argument('--raw-logits', action='store_true')
    for name, default in (('peer-gbps', 54), ('fabric-gbps', 216), ('host-gbps', 40),
                          ('host-fabric-gbps', 128), ('collective-us', 5), ('host-stage-us', 5),
                          ('graph-boundary-us', 20), ('dram-efficiency', 1)):
        parser.add_argument('--' + name, type=float, default=default)
    parser.add_argument('--control-bytes', type=int, default=65536)
    args = vars(parser.parse_args())
    inputs = json.loads(args.pop('inputs').read_text())['inputs']
    try:
        result = sweep(inputs) if args.pop('sweep') else estimate(inputs, **args)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
