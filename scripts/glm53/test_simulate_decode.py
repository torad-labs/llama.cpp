"""The offline model must account for private histories, owners and useful output."""
import json
from pathlib import Path
import random
import subprocess
import sys
import unittest
from typing import Any

import simulate_decode
from simulate_decode import simulate


class DecodeSimulationTest(unittest.TestCase):
    def test_private_slots_reserve_history_even_when_only_one_is_active(self):
        result = simulate(slots=36, sequence_ids=(0,))
        self.assertEqual(result['ownership']['slots'], [9, 9, 9, 9])
        self.assertEqual(result['ownership']['active'], [1, 0, 0, 0])
        self.assertEqual(result['memory']['reserved_latent_bytes'], 36 * 442368 * 12 * 544)
        self.assertEqual(result['memory']['reserved_indexer_bytes'], 36 * 442368 * 12 * 408)
        self.assertEqual(result['memory']['active_history_bytes'], 400000 * 12 * 952)
        self.assertIsNone(result['prediction']['tokens_per_second'])

    def test_host_kv_moves_only_latent_history_and_does_not_claim_a_fit(self):
        resident = simulate(slots=40, sequence_ids=(0, 1, 2, 3))
        host = simulate(slots=40, sequence_ids=(0, 1, 2, 3), host_kv=True)
        self.assertEqual(host['memory']['host_latent_bytes'], 40 * 442368 * 12 * 544)
        self.assertEqual(resident['memory']['gpu_history_bytes'] - host['memory']['gpu_history_bytes'],
                         host['memory']['host_latent_bytes'])
        self.assertEqual(host['memory']['gpu_history_bytes'], 40 * 442368 * 12 * 408)
        self.assertFalse(host['prediction']['capacity_qualified'])
        self.assertGreater(host['work']['selected_host_bytes_per_cycle_upper'], 0)
        self.assertEqual(resident['work']['selected_host_bytes_per_cycle_upper'], 0)

    def test_owners_follow_sequence_ids_not_the_order_of_the_batch(self):
        result = simulate(slots=13, sequence_ids=(12, 8, 4, 0))
        self.assertEqual(result['ownership']['slots'], [4, 3, 3, 3])
        self.assertEqual(result['ownership']['active'], [4, 0, 0, 0])
        self.assertEqual(result['ownership']['target_rows'], [12, 0, 0, 0])
        self.assertEqual(result['ownership']['dense_utilization_ceiling'], 0.25)

    def test_rejected_drafts_consume_work_but_not_delivered_tokens(self):
        for accepted in (0, 1, 2):
            with self.subTest(accepted=accepted):
                result = simulate(sequence_ids=(0, 1, 2, 3), depth=2, accepted=accepted)
                self.assertEqual(result['cycle']['target_rows'], 12)
                self.assertEqual(result['cycle']['proposal_rows'], 8)
                self.assertEqual(result['cycle']['delivered_tokens'], 4 * (accepted + 1))
                self.assertEqual(result['cycle']['rejected_tokens'], 4 * (2 - accepted))
                self.assertEqual(result['cycle']['context_after'], 400000 + accepted + 1)

    def test_target_only_has_no_draft_cache_or_draft_work(self):
        result = simulate(depth=0, accepted=0)
        self.assertEqual(result['memory']['reserved_latent_bytes'], 36 * 442368 * 11 * 544)
        self.assertEqual(result['cycle']['proposal_rows'], 0)
        self.assertEqual(result['cycle']['delivered_tokens'], 1)
        self.assertEqual(result['cycle']['target_rows'], 1)

    def test_more_concurrency_changes_work_not_reserved_capacity(self):
        one = simulate(sequence_ids=(0,))
        four = simulate(sequence_ids=(0, 1, 2, 3))
        self.assertEqual(one['memory']['reserved_indexer_bytes'], four['memory']['reserved_indexer_bytes'])
        self.assertEqual(four['work']['target_moe_input_payload_bytes'],
                         4 * one['work']['target_moe_input_payload_bytes'])
        self.assertEqual(one['work']['target_moe_input_payload_bytes'], 42 * 3 * 4096 * 4)
        self.assertEqual(four['work']['target_routed_expert_assignments'], 42 * 12 * 8)

    def test_fair_prefill_can_disable_both_kda_fast_paths(self):
        many = simulate(prefill_sequences=36, prefill_rows=512)
        few = simulate(prefill_sequences=4, prefill_rows=512)
        self.assertEqual(many['prefill']['tokens_per_sequence'], 14)
        self.assertEqual(many['prefill']['kda_path'], 'recurrent')
        self.assertEqual(few['prefill']['tokens_per_sequence'], 128)
        self.assertEqual(few['prefill']['kda_path'], 'chunked')

    def test_target_shape_retention_cliff_depends_on_proposals_not_slot_count(self):
        ten = simulate(sequence_ids=tuple(range(10)), depth=2)
        eleven = simulate(sequence_ids=tuple(range(11)), depth=2)
        target_only = simulate(sequence_ids=tuple(range(32)), depth=0, accepted=0)
        self.assertTrue(ten['graphs']['target_extra_slot_eligible'])
        self.assertFalse(eleven['graphs']['target_extra_slot_eligible'])
        self.assertTrue(target_only['graphs']['target_extra_slot_eligible'])
        self.assertFalse(eleven['graphs']['recapture_is_inevitable'])

    def test_every_target_layer_and_shared_expert_has_an_owner(self):
        result = simulate(sequence_ids=(0, 1, 2, 3))
        layers = result['layers']
        self.assertEqual(len(layers), 45)
        self.assertEqual([layer['index'] for layer in layers if layer['attention'] == 'MLA/DSA'],
                         list(range(3, 44, 4)))
        self.assertEqual(sum(layer['attention'] == 'KDA' for layer in layers), 34)
        self.assertEqual(sum(layer['feed_forward'] == 'dense' for layer in layers), 3)
        for layer in layers[3:]:
            self.assertEqual(layer['shared_expert_rows_per_owner'], [3, 3, 3, 3])
            self.assertEqual(layer['routed_expert_partitions'], [[0, 72], [72, 144], [144, 216], [216, 288]])
        self.assertEqual(result['graphs']['cuda_graph_count_cap'], 'unlimited')
        self.assertEqual(result['graphs']['cuda_graph_node_budget'], 32768)

    def test_recurrent_rollback_allocation_does_not_scale_with_context(self):
        short = simulate(context=512)
        long = simulate(context=400000)
        self.assertEqual(short['memory']['recurrent_bytes_per_slot'], long['memory']['recurrent_bytes_per_slot'])
        self.assertEqual(long['memory']['recurrent_bytes_per_slot'], 34 * 3 * (1114112 + 294912))
        self.assertEqual(long['memory']['recurrent_bytes_per_card'], [9 * 34 * 3 * (1114112 + 294912)] * 4)

    def test_pool_scoring_reads_one_head_not_all_three_stored_heads(self):
        result = simulate(context=400000, sequence_ids=(0,))
        self.assertEqual(result['work']['pooled_key_bytes_per_layer_scan'], 100000 * 136)
        self.assertEqual(result['work']['target_pooled_key_logical_query_bytes'], 33 * 100000 * 136)
        self.assertEqual(result['hypotheses']['compact_pool_savings_bytes_per_card'],
                         [9 * 442368 * 12 * 102] * 4)

    def test_mtp_catchup_includes_rejected_rows_before_acceptance(self):
        result = simulate(sequence_ids=(0, 1, 2, 3), accepted=0, host_kv=True)
        self.assertEqual(result['cycle']['catchup_rows'], 12)
        self.assertEqual(result['cycle']['catchup_calls'], 1)
        self.assertEqual(result['cycle']['delivered_tokens'], 4)
        self.assertEqual(result['work']['selected_host_bytes_per_cycle_upper'], (12 * 11 + 8 + 12) * 2080 * 544)
        self.assertEqual(result['work']['verified_hidden_d2h_bytes'], 12 * 4096 * 4)

    def test_mid_sized_reductions_use_host_staging_despite_peer_access(self):
        for active, host, expected in ((5, False, 'peer-ll'), (6, False, 'host-chunked'),
                                       (40, True, 'host-chunked'), (43, True, 'peer-copy-engine')):
            result = simulate(slots=max(36, active), sequence_ids=tuple(range(active)), host_kv=host)
            self.assertEqual(result['transport']['reduction']['route'], expected)
            self.assertEqual(result['transport']['reduction']['capturable'], expected != 'peer-copy-engine')

    def test_native_gather_hypothesis_removes_zero_padded_payload_not_experts(self):
        result = simulate(sequence_ids=(0, 1, 2, 3))
        self.assertEqual(result['transport']['gather']['logical_peer_payload_bytes'], 12 * 12 * 4096 * 4)
        self.assertEqual(result['hypotheses']['native_gather_peer_payload_bytes_per_layer'], 3 * 12 * 4096 * 4)

    def test_middle_concurrency_band_loses_extra_graphs_before_device_hidden_gate(self):
        for active, extra_graph, device_h in ((10, True, False), (11, False, False),
                                              (21, False, False), (22, False, True)):
            result = simulate(sequence_ids=tuple(range(active)))
            self.assertEqual(result['graphs']['target_extra_slot_eligible'], extra_graph)
            self.assertEqual(result['graphs']['device_hidden_eligible'], device_h)

    def test_short_host_attention_is_not_reported_as_long_context_sparse_staging(self):
        result = simulate(context=512, host_kv=True)
        self.assertIsNone(result['work']['selected_host_bytes_per_cycle_upper'])

    def test_generated_owner_and_memory_conservation(self):
        rng = random.Random(53)
        for _ in range(150):
            slots = rng.randint(1, 40)
            ids = tuple(rng.sample(range(slots), rng.randint(1, slots)))
            context = rng.randint(1, 409600)
            depth = rng.randint(0, 2)
            accepted = rng.randint(0, depth)
            result = simulate(slots=slots, sequence_ids=ids, context=context, depth=depth, accepted=accepted)
            self.assertEqual(sum(result['ownership']['slots']), slots)
            self.assertEqual(sum(result['ownership']['active']), len(ids))
            self.assertEqual(sum(result['ownership']['target_rows']), len(ids) * (depth + 1))
            self.assertEqual(sum(result['memory']['gpu_history_bytes_per_card']), result['memory']['gpu_history_bytes'])
            self.assertLessEqual(result['memory']['active_history_bytes'],
                                 result['memory']['reserved_latent_bytes'] + result['memory']['reserved_indexer_bytes'])
            self.assertEqual(result['cycle']['delivered_tokens'] + result['cycle']['rejected_tokens'],
                             len(ids) * (depth + 1))

    def test_impossible_inputs_are_rejected(self):
        cases: tuple[dict[str, Any], ...] = ({'slots': 0}, {'slots': 41}, {'sequence_ids': ()}, {'sequence_ids': (0, 0)},
                       {'slots': 4, 'sequence_ids': (4,)}, {'context': -1}, {'context': 442368},
                       {'depth': 0, 'accepted': 1}, {'host_kv': True, 'depth': 1},
                       {'prefill_sequences': -1}, {'prefill_rows': 0})
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                simulate(**kwargs)

    def test_command_supports_owner_skew_and_target_only_without_extra_flags(self):
        script = str(Path(__file__).with_name('simulate_decode.py'))
        for flags, owners, delivered in ((['--sequence-ids', '0,4,8,12'], [4, 0, 0, 0], 8),
                                         (['--depth', '0'], [1, 1, 1, 1], 4)):
            completed = subprocess.run([sys.executable, script, *flags], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(completed.stdout)
            self.assertEqual(result['ownership']['active'], owners)
            self.assertEqual(result['cycle']['delivered_tokens'], delivered)

    def test_model_refuses_work_that_requires_unmodeled_microbatch_splitting(self):
        cases: tuple[dict[str, Any], ...] = (
            {'slots': 256, 'host_kv': True, 'sequence_ids': tuple(range(171))},
            {'prefill_sequences': 36, 'prefill_rows': 16})
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                simulate(**kwargs)

    def test_cycle_sequence_keeps_only_accepted_output_in_each_history(self):
        run = simulate_decode.simulate_cycles(slots=4, sequence_ids=(0, 1, 2, 3), context=400000,
                                              depth=2, acceptances=(0, 2, 1))
        self.assertEqual([cycle['context_before'] for cycle in run['cycles']], [400000, 400001, 400004])
        self.assertEqual(run['context_after'], 400006)
        self.assertEqual(run['delivered_tokens'], 24)
        self.assertEqual(run['target_rows'], 36)
        self.assertEqual(run['proposal_rows'], 24)
        self.assertEqual(run['rejected_tokens'], 12)
        self.assertFalse(run['quality_qualified'])

    def test_generated_cycle_sequences_conserve_context_and_private_capacity(self):
        rng = random.Random(400000)
        for _ in range(30):
            accepted = tuple(rng.randint(0, 2) for _ in range(rng.randint(1, 12)))
            run = simulate_decode.simulate_cycles(slots=4, sequence_ids=(0, 1), context=409600,
                                                  depth=2, acceptances=accepted)
            added = sum(1 + count for count in accepted)
            self.assertEqual(run['context_after'], 409600 + added)
            self.assertEqual(run['delivered_tokens'], 2 * added)
        with self.assertRaises(ValueError):
            simulate_decode.simulate_cycles(context=442364, depth=2, acceptances=(2, 2))

    def test_command_emits_a_complete_offline_scenario(self):
        completed = subprocess.run([sys.executable, str(Path(__file__).with_name('simulate_decode.py')),
                                    '--slots', '40', '--active', '8', '--context', '409600', '--host-kv'],
                                   capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout)
        self.assertEqual(result['ownership']['active'], [2, 2, 2, 2])
        self.assertEqual(result['cycle']['context_before'], 409600)
        self.assertIn('assumptions', result)
        self.assertIn('sources', result)
        self.assertFalse(result['prediction']['quality_qualified'])


if __name__ == '__main__':
    unittest.main()
