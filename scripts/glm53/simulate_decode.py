#!/usr/bin/env python3
"""CPU-only structural simulation of the documented GLM four-card recipe."""
import argparse
import json
import math

from capacity import INDEXER_ROW_BYTES, LATENT_ROW_BYTES, MLA_LAYERS, SLOT_CELLS

DEVICES = 4
TARGET_LAYERS = 45
TARGET_MLA_LAYERS = MLA_LAYERS - 1
KDA_LAYERS = TARGET_LAYERS - TARGET_MLA_LAYERS
DENSE_LAYERS = 3
MOE_LAYERS = TARGET_LAYERS - DENSE_LAYERS
EMBEDDING = 4096
EXPERTS = 288
EXPERTS_USED = 8
INDEX_HEAD_BYTES = 128 // 32 * 34


def collective_routes(rows, bf16=True):
    payload = rows * EMBEDDING * 4
    wire = payload // 2 if bf16 else payload
    reduction = ('peer-copy-engine' if wire >= 1024 * 1024 else
                 'peer-ll' if wire <= 128 * 1024 else 'host-chunked')
    gather = 'peer-ll' if payload <= 128 * 1024 else 'peer-chunked'
    return {
        'scope': 'One routed layer; four ranks, successful peer probes/inbox allocation, default thresholds.',
        'gather': {'route': gather, 'full_buffer_bytes': payload,
                   'logical_peer_payload_bytes': 12 * payload,
                   'packet_payload_bytes': 12 * payload * (2 if gather == 'peer-ll' else 1),
                   'capturable': True},
        'reduction': {'route': reduction, 'wire_type': 'bf16' if bf16 else 'f32',
                      'wire_bytes_per_rank': wire,
                      'logical_peer_payload_bytes': 0 if reduction == 'host-chunked' else 12 * wire,
                      'logical_host_endpoint_bytes': 16 * wire if reduction == 'host-chunked' else 0,
                      'capturable': reduction != 'peer-copy-engine'},
    }


def simulate(*, slots=36, sequence_ids=(0,), context=400000, depth=2, accepted=1,
             host_kv=False, prefill_sequences=0, prefill_rows=512):
    sequence_ids = tuple(sequence_ids)
    integers = (slots, context, depth, accepted, prefill_sequences, prefill_rows, *sequence_ids)
    if any(type(n) is not int for n in integers):
        raise ValueError("Counts and sequence IDs must be integers")
    if not 1 <= slots <= (256 if host_kv else 40):
        raise ValueError("Slots exceed the launcher's resident or host-KV bounds")
    if not sequence_ids or len(set(sequence_ids)) != len(sequence_ids):
        raise ValueError("Active sequence IDs must be nonempty and unique")
    if any(seq < 0 or seq >= slots for seq in sequence_ids):
        raise ValueError("Active sequence IDs must refer to configured slots")
    if depth not in (0, 1, 2) or not 0 <= accepted <= depth:
        raise ValueError("Accepted drafts must be between zero and the configured depth")
    if context < 1 or context + depth + 1 > SLOT_CELLS:
        raise ValueError("The history and verification rows must fit in the private slot")
    if host_kv and depth != 2:
        raise ValueError("The documented host-KV recipe requires MTP depth two")
    if not 0 <= prefill_sequences <= slots or not 1 <= prefill_rows <= 512:
        raise ValueError("Prefill uses 0..slots sequences and 1..512 rows in this recipe")

    active = len(sequence_ids)
    if active * (depth + 1) > 512 or prefill_sequences > prefill_rows:
        raise ValueError("This model covers one equal-sequence microbatch; the requested rows require splitting")
    owner_slots = [len(range(device, slots, DEVICES)) for device in range(DEVICES)]
    owner_active = [sum(seq % DEVICES == device for seq in sequence_ids) for device in range(DEVICES)]
    target_rows = active * (depth + 1)
    proposal_rows = active * depth
    catchup_rows = target_rows if depth else 0
    state_bytes = 64 * 128 * 128 // 32 * 34
    convolution_bytes = 3 * (4 - 1) * 64 * 128 * 4
    recurrent_per_slot = KDA_LAYERS * (1 + depth) * (state_bytes + convolution_bytes)
    device_h = bool(depth) and target_rows >= 64
    layers = TARGET_MLA_LAYERS + bool(depth)
    latent_per_slot = SLOT_CELLS * layers * LATENT_ROW_BYTES
    indexer_per_slot = SLOT_CELLS * layers * INDEXER_ROW_BYTES
    gpu_per_slot = indexer_per_slot + (0 if host_kv else latent_per_slot)
    prefill_tokens = prefill_rows // prefill_sequences if prefill_sequences else 0
    prefill_path = ('none' if not prefill_sequences else 'chunked' if prefill_tokens >= 128
                    else 'decay-precompute' if prefill_tokens >= 32 else 'recurrent')
    row_bytes = EMBEDDING * 4
    no_expert_on_rank = math.comb(EXPERTS - EXPERTS // DEVICES, EXPERTS_USED) / math.comb(EXPERTS, EXPERTS_USED)

    return {
        'schema_version': 1,
        'scenario': {'slots': slots, 'sequence_ids': list(sequence_ids), 'context': context,
                     'depth': depth, 'accepted': accepted, 'host_kv': host_kv},
        'ownership': {
            'slots': owner_slots, 'active': owner_active,
            'target_rows': [n * (depth + 1) for n in owner_active],
            'dense_utilization_ceiling': active / (DEVICES * max(owner_active)),
        },
        'memory': {
            'reserved_latent_bytes': slots * latent_per_slot,
            'reserved_indexer_bytes': slots * indexer_per_slot,
            'active_history_bytes': active * context * layers * (LATENT_ROW_BYTES + INDEXER_ROW_BYTES),
            'gpu_history_bytes': slots * gpu_per_slot,
            'gpu_history_bytes_per_card': [n * gpu_per_slot for n in owner_slots],
            'host_latent_bytes': slots * latent_per_slot if host_kv else 0,
            'recurrent_bytes_per_slot': recurrent_per_slot,
            'recurrent_bytes_per_card': [n * recurrent_per_slot for n in owner_slots],
            'recurrent_scope': '34 KDA layers, Q8 state, F32 convolution, current plus depth rollback planes.',
            'not_in_history_total': ['weights and derived weight layouts', 'recurrent state and rollback',
                                     'convolution history', 'resident pool metadata', 'graph and allocator scratch',
                                     'host staging', 'prompt snapshots', 'projector', 'driver allocations'],
        },
        'cycle': {
            'context_before': context, 'context_after': context + accepted + 1,
            'target_rows': target_rows, 'proposal_rows': proposal_rows,
            'proposal_calls': depth, 'catchup_rows': catchup_rows, 'catchup_calls': int(bool(depth)),
            'delivered_tokens': active * (accepted + 1), 'rejected_tokens': active * (depth - accepted),
            'target_kda_row_updates': target_rows * KDA_LAYERS,
            'target_sparse_attention_queries': target_rows * TARGET_MLA_LAYERS,
            'target_shared_expert_rows': target_rows * MOE_LAYERS,
            'draft_attention_queries': proposal_rows,
        },
        'work': {
            'target_moe_input_payload_bytes': MOE_LAYERS * target_rows * row_bytes,
            'target_moe_collective_calls': 2 * MOE_LAYERS,
            'proposal_moe_collective_calls': 2 * depth,
            'catchup_moe_collective_calls': 2 * bool(depth),
            'pooled_key_bytes_per_layer_scan': context // 4 * INDEX_HEAD_BYTES,
            'target_pooled_key_logical_query_bytes': context // 4 * INDEX_HEAD_BYTES * target_rows * TARGET_MLA_LAYERS,
            'pool_scan_scope': 'Completed pools at cycle start, no padding; logical key/query bytes, not measured DRAM traffic.',
            'verified_hidden_d2h_bytes': target_rows * row_bytes if depth else 0,
            'catchup_hidden_logical_h2d_bytes': (active if device_h else target_rows) * row_bytes if depth else 0,
            'target_routed_expert_assignments': MOE_LAYERS * target_rows * EXPERTS_USED,
            'uniform_routing_expected_ranks_per_token': DEVICES * (1 - no_expert_on_rank),
            'uniform_routing_expected_remote_ranks': (DEVICES - 1) * (1 - no_expert_on_rank),
            'selected_host_bytes_per_cycle_upper': (
                (target_rows * TARGET_MLA_LAYERS + proposal_rows + catchup_rows) * 2080 * LATENT_ROW_BYTES
                if host_kv and context >= 16640 else None if host_kv else 0),
            'selected_host_bytes_scope': 'Long sparse target/proposal/catch-up envelope; excludes padding writes and cache writes. Null below all direct-index cutoffs.',
        },
        'prefill': {
            'scope': 'Separate equal-sequence batching scenario, not a prediction of the live server scheduler.',
            'sequences': prefill_sequences, 'row_budget': prefill_rows,
            'tokens_per_sequence': prefill_tokens, 'kda_path': prefill_path,
        },
        'transport': collective_routes(target_rows),
        'f32_wire_control': collective_routes(target_rows, bf16=False),
        'hypotheses': {
            'implemented': False,
            'compact_pool_savings_bytes_per_card': [
                n * layers * (SLOT_CELLS - (SLOT_CELLS + 3) // 4) * INDEX_HEAD_BYTES for n in owner_slots],
            'compact_pool_scope': 'Keep all raw keys/gates; store one pooled key per completed pool. Excludes mapping/allocator overhead.',
            'native_gather_peer_payload_bytes_per_layer': 3 * target_rows * row_bytes,
        },
        'layers': [
            {'index': layer, 'attention': 'MLA/DSA' if layer in range(3, 44, 4) else 'KDA',
             'feed_forward': 'dense' if layer < DENSE_LAYERS else 'routed plus shared',
             'owner_rows': [n * (depth + 1) for n in owner_active],
             'shared_expert_rows_per_owner': ([n * (depth + 1) for n in owner_active]
                                               if layer >= DENSE_LAYERS else [0] * DEVICES),
             'routed_expert_partitions': ([[d * 72, (d + 1) * 72] for d in range(DEVICES)]
                                          if layer >= DENSE_LAYERS else []),
             'order': ['mHC pre', 'normalize', 'attention', 'mHC post', 'mHC pre', 'normalize',
                       'feed forward', 'mHC post']}
            for layer in range(TARGET_LAYERS)
        ],
        'graphs': {
            'target_extra_slot_eligible': target_rows <= 32,
            'device_hidden_eligible': device_h,
            'recapture_is_inevitable': False,
            'extra_slots': 3, 'cuda_graph_count_cap': 'unlimited', 'cuda_graph_node_budget': 32768,
            'scope': 'Extra target-shape eligibility only; same-shape main graph reuse can still work above 32 rows.',
            'draft_scope': 'MTP generation/catch-up shapes have separate registration and allocation guards.',
        },
        'prediction': {'tokens_per_second': None, 'capacity_qualified': False, 'quality_qualified': False,
                       'reason': 'Byte and operation accounting is not a timing or numerical simulator.'},
        'assumptions': [
            'Independent equal-length active histories, one owner per sequence, four equal expert partitions.',
            'Inactive slots retain their full reservation; their unknown occupied history is not in the active-payload subtotal.',
            'One full proposal/verification cycle with fixed depth and acceptance; no EOS, grammar or draft replay.',
            'Q8 latent and indexer geometry from the launcher capacity model; architecture dimensions from retained metadata.',
            'Routing probabilities assume eight distinct uniformly chosen experts; actual routing can be skewed.',
            'The selected-row host envelope applies to the long-context sparse path, not the early dense attention path.',
            'Recurrent geometry assumes the retained metadata; buffer alignment and allocator overhead are excluded.',
            'Weights, graph residency, staging and driver buffers still prevent a total-VRAM or throughput claim.',
            'Full-buffer collective counts assume peer probing succeeded; host fallback and packet flags are separate costs.',
        ],
        'sources': [
            'scripts/glm53/serve.py:48', 'scripts/glm53/capacity.py:8',
            'docs/torad/glm53-dp.md:60', 'docs/torad/glm53-dp.md:263',
            'src/models/glm5next.cpp:375', 'ggml/src/ggml-cuda/gated_delta_net.cu',
            'common/speculative.cpp:1783', 'src/llama-memory-recurrent.cpp:114',
            'ggml/src/ggml-cuda/allreduce.cu:608', 'ggml/src/ggml-cuda/allreduce.cu:1626',
            'ggml/src/ggml-cuda/fattn.cu:1150',
            'docs/torad/glm53-four-gpu-simulation.md',
        ],
    }


def simulate_cycles(*, context=400000, acceptances=(1,), **scenario):
    acceptances = tuple(acceptances)
    if not 1 <= len(acceptances) <= 10000:
        raise ValueError("Supply 1..10000 acceptance counts")
    cycles = []
    for accepted in acceptances:
        result = simulate(context=context, accepted=accepted, **scenario)
        cycle = result['cycle']
        cycles.append(cycle)
        context = cycle['context_after']
    return {
        'scope': 'Fresh proposal/verification cycles with caller-supplied acceptance, not predicted model tokens.',
        'cycles': cycles, 'context_after': context,
        **{key: sum(cycle[key] for cycle in cycles) for key in
           ('delivered_tokens', 'target_rows', 'proposal_rows', 'rejected_tokens')},
        'quality_qualified': False, 'tokens_per_second': None,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--slots', type=int, default=36)
    p.add_argument('--active', type=int, default=4, help='Consecutive IDs starting at zero')
    p.add_argument('--sequence-ids', help='Explicit comma-separated slot IDs; replaces --active')
    p.add_argument('--context', type=int, default=400000)
    p.add_argument('--depth', type=int, choices=(0, 1, 2), default=2)
    p.add_argument('--accepted', type=int, help='Accepted proposals per sequence; default min(1, depth)')
    p.add_argument('--host-kv', action='store_true')
    p.add_argument('--acceptance-pattern', help='Comma-separated accepted counts for successive fresh cycles')
    p.add_argument('--prefill-sequences', type=int, default=0)
    p.add_argument('--prefill-rows', type=int, default=512)
    a = p.parse_args()
    try:
        if not 1 <= a.active <= 256:
            raise ValueError('Active sessions must be 1..256')
        ids = tuple(int(seq) for seq in a.sequence_ids.split(',')) if a.sequence_ids else tuple(range(a.active))
        result = simulate(slots=a.slots, sequence_ids=ids, context=a.context,
                          depth=a.depth, accepted=min(1, a.depth) if a.accepted is None else a.accepted,
                          host_kv=a.host_kv, prefill_sequences=a.prefill_sequences, prefill_rows=a.prefill_rows)
        if a.acceptance_pattern:
            result['run'] = simulate_cycles(slots=a.slots, sequence_ids=ids, context=a.context, depth=a.depth,
                                            host_kv=a.host_kv,
                                            acceptances=tuple(int(n) for n in a.acceptance_pattern.split(',')))
    except ValueError as error:
        p.error(str(error))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
