"""Goal: estimate DP4 decode with transfers at their actual dependency boundaries.

The offline caller must see payload, synchronization and host-KV costs separately;
no third scatter, impossible owner balance, or global overlap may inflate the rate.
"""
import json
from pathlib import Path
import random
import subprocess
import sys
import unittest

import estimate_decode_pcie as model

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / 'docs/torad/glm53-decode-roofline.json'


class PcieEstimateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = json.loads(INPUT.read_text())['inputs']

    def test_native_transfers_send_only_owner_slices_and_partial_results(self):
        gather = model.collective(40, 3, 'gather', 'optimized')
        reduce = model.collective(40, 3, 'reduce', 'optimized')
        self.assertEqual(gather['sent_bytes'], [3 * 30 * 16384] * 4)
        self.assertEqual(reduce['sent_bytes'], [90 * 8192] * 4)
        self.assertEqual(sum(gather['sent_bytes']), 3 * 120 * 16384)
        self.assertEqual(sum(reduce['sent_bytes']), 3 * 120 * 8192)

    def test_one_session_cannot_pool_four_pcie_links_for_its_activation(self):
        gather = model.collective(1, 3, 'gather', 'optimized')
        self.assertEqual(gather['sent_bytes'], [3 * 3 * 16384, 0, 0, 0])
        self.assertEqual(gather['received_bytes'], [0, 3 * 16384, 3 * 16384, 3 * 16384])

    def test_current_reduction_has_the_source_defined_host_staging_band(self):
        for sessions, route in ((5, 'peer-ll'), (6, 'host-chunked'),
                                (40, 'host-chunked'), (43, 'peer-copy-engine')):
            with self.subTest(sessions=sessions):
                result = model.collective(sessions, 3, 'reduce', 'current')
                self.assertEqual(result['route'], route)
        result = model.collective(40, 3, 'reduce', 'current')
        self.assertEqual(result['sent_bytes'], [120 * 8192] * 4)
        self.assertEqual(result['received_bytes'], [3 * 120 * 8192] * 4)
        self.assertEqual(result['host_endpoint_bytes'], 16 * 120 * 8192)

    def test_ll_flags_double_the_small_current_gather_payload(self):
        result = model.collective(1, 3, 'gather', 'current')
        self.assertEqual(result['route'], 'peer-ll')
        self.assertEqual(sum(result['sent_bytes']), 2 * 12 * 3 * 16384)

    def test_full_duplex_and_shared_fabric_are_not_added_or_confused(self):
        traffic = {'sent_bytes': [1000000] * 4, 'received_bytes': [1000000] * 4,
                   'host_endpoint_bytes': 0, 'route': 'peer'}
        self.assertAlmostEqual(model.transfer_ms(traffic, peer_gbps=1, fabric_gbps=4), 1)
        self.assertAlmostEqual(model.transfer_ms(traffic, peer_gbps=1, fabric_gbps=1), 4)
        host = dict(traffic, host_endpoint_bytes=8000000, route='host-chunked')
        self.assertAlmostEqual(model.transfer_ms(host, host_gbps=1, host_fabric_gbps=8), 2)

    def test_cycle_has_two_boundaries_per_moe_and_no_scatter_transfer(self):
        for depth, catchup, expected in ((0, 'cache-only', 84), (2, 'cache-only', 88),
                                         (2, 'full', 90)):
            result = model.estimate(self.inputs, depth=depth, catchup=catchup)
            self.assertEqual(result['collective_boundaries'], expected)
            self.assertFalse(any(e['kind'] == 'scatter' for e in result['events']))
        result = model.estimate(self.inputs)
        peer = sum(sum(e['sent_bytes']) * e['repeat'] for e in result['events']
                   if e['kind'] in ('gather', 'reduce'))
        self.assertEqual(peer, 4 * 94371840)

    def test_latency_is_paid_at_each_dependency_not_once_per_cycle(self):
        zero = model.estimate(self.inputs, collective_us=0)
        slow = model.estimate(self.inputs, collective_us=25)
        self.assertAlmostEqual(slow['cycle_ms'] - zero['cycle_ms'], 88 * 0.025)
        self.assertLess(slow['tps_per_session'], zero['tps_per_session'])

    def test_resident_kv_has_no_host_staging_and_reuse_is_explicit(self):
        resident = model.estimate(self.inputs)
        host = model.estimate(self.inputs, kv='host-per-query')
        reuse = model.estimate(self.inputs, kv='host-reuse')
        host_events = lambda result: [e for e in result['events'] if e['kind'] == 'kv-stage']
        self.assertEqual(host_events(resident), [])
        expected = (11 * 120 + 2 * 40) * 2051 * 544
        self.assertEqual(sum(sum(e['received_bytes']) * e['repeat'] for e in host_events(host)), expected)
        self.assertEqual(sum(sum(e['received_bytes']) * e['repeat'] for e in host_events(reuse)),
                         (11 * 40 + 2 * 40) * 2051 * 544)
        self.assertGreater(host['cycle_ms'], reuse['cycle_ms'])
        self.assertGreater(reuse['cycle_ms'], resident['cycle_ms'])

    def test_full_catchup_adds_attention_and_moe_before_acceptance(self):
        pruned = model.estimate(self.inputs, kv='host-per-query')
        full = model.estimate(self.inputs, kv='host-per-query', catchup='full')
        self.assertGreater(full['dram_ms'], pruned['dram_ms'])
        self.assertGreater(full['cycle_ms'], pruned['cycle_ms'])
        stages = [e for e in full['events'] if e['kind'] == 'kv-stage' and e['stage'] == 'catchup']
        self.assertEqual(sum(stages[0]['received_bytes']), 120 * 2051 * 544)

    def test_shared_overlap_is_bounded_by_independent_work_in_each_layer(self):
        serial = model.estimate(self.inputs)
        overlap = model.estimate(self.inputs, overlap_shared=True)
        cap_ms = 44 * 3 * 8912896 / (1597e9) * 1000
        self.assertGreater(overlap['overlap_ms'], 0)
        self.assertLessEqual(overlap['overlap_ms'], cap_ms)
        self.assertAlmostEqual(serial['cycle_ms'] - overlap['cycle_ms'], overlap['overlap_ms'])
        self.assertGreaterEqual(overlap['cycle_ms'], overlap['dram_ms'])

    def test_no_pcie_cost_recovers_the_existing_dp_dram_model(self):
        baseline = json.loads(INPUT.read_text())['results']
        for n in (1, 4, 12, 40):
            for span in ('independent', 'same-within-sequence'):
                result = model.estimate(self.inputs, sessions=n, expert_span=span)
                old = next(x for x in baseline if x['sessions'] == n and x['layout'] == 'dp'
                           and x['depth'] == 2 and x['expert_span'] == span)
                self.assertAlmostEqual(result['dram_ms'], old['ideal_cycle_ms'])
                self.assertGreater(result['cycle_ms'], result['dram_ms'])

    def test_generated_protocols_conserve_bytes_and_slower_links_cannot_help(self):
        rng = random.Random(6000)
        for _ in range(60):
            n = rng.randint(1, 40)
            rows = rng.randint(1, 3)
            for kind in ('gather', 'reduce'):
                event = model.collective(n, rows, kind, 'optimized')
                self.assertEqual(sum(event['sent_bytes']), sum(event['received_bytes']))
            fast = model.estimate(self.inputs, sessions=n)
            slow = model.estimate(self.inputs, sessions=n, peer_gbps=27, fabric_gbps=54)
            self.assertGreaterEqual(slow['cycle_ms'], fast['cycle_ms'])
            self.assertAlmostEqual(fast['aggregate_tps'], n * fast['tps_per_session'])
            self.assertAlmostEqual(fast['tps_per_session'] * fast['cycle_ms'], 3000)

    def test_acceptance_changes_useful_output_not_evaluated_work(self):
        all_accepted = model.estimate(self.inputs)
        one = model.estimate(self.inputs, accepted=1)
        self.assertEqual(one['cycle_ms'], all_accepted['cycle_ms'])
        self.assertAlmostEqual(one['tps_per_session'], all_accepted['tps_per_session'] * 2 / 3)

    def test_invalid_or_unmodeled_scenarios_are_rejected(self):
        for options in ({'sessions': 0}, {'sessions': 41}, {'peer_gbps': 0},
                        {'fabric_gbps': -1}, {'collective_us': -1}, {'depth': 3},
                        {'depth': 0, 'accepted': 1}, {'protocol': 'tensor'},
                        {'kv': 'host-per-query', 'depth': 0}, {'control_bytes': -1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                model.estimate(self.inputs, **options)

    def test_sweep_covers_every_named_assumption_without_pooled_layout_claims(self):
        result = subprocess.run([sys.executable, str(Path(__file__).with_name('estimate_decode_pcie.py')),
                                 '--sweep'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(len(data['results']), 41)
        self.assertEqual(len({(x['scenario'], x['sessions'], x['expert_span'])
                              for x in data['results']}), 41)
        self.assertEqual(data['model_revision'], self.inputs['model_revision'])
        self.assertEqual(data['layout'], 'dp4')
        self.assertFalse(data['hardware_validated'])

    def test_cli_returns_the_full_offline_ledger_and_assumptions(self):
        result = subprocess.run([sys.executable, str(Path(__file__).with_name('estimate_decode_pcie.py')),
                                 '--sessions', '40', '--kv', 'host-per-query'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data['sessions'], 40)
        self.assertGreater(data['pcie_ms'], 0)
        self.assertFalse(data['hardware_validated'])
        self.assertIn('assumptions', data)
        self.assertIn('events', data)


if __name__ == '__main__':
    unittest.main()
