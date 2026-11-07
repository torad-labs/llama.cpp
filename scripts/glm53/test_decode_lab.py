"""Goal: expose the finest supported profile evidence without inventing bandwidth.

Real SQLite records must remain attributable across graph clones, preserve device
and timing scope, and reconcile the ideal detail ledger with its parent estimate.
"""
import gzip
import json
from pathlib import Path
import random
import sqlite3
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

import decode_lab
import estimate_decode_pcie

ROOT = Path(__file__).resolve().parents[2]


class ProfileEvidenceTest(unittest.TestCase):
    def fixture(self, root):
        path = Path(root) / 'capture.sqlite'
        db = sqlite3.connect(path)
        db.executescript("""
            CREATE TABLE StringIds(id INTEGER,value TEXT);
            CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,globalTid INTEGER,text TEXT,textId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,globalTid INTEGER,correlationId INTEGER,nameId INTEGER);
            CREATE TABLE CUDA_GRAPH_NODE_EVENTS(start INTEGER,end INTEGER,globalTid INTEGER,graphNodeId INTEGER,originalGraphNodeId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,deviceId INTEGER,globalPid INTEGER,
                correlationId INTEGER,demangledName INTEGER,shortName INTEGER,graphNodeId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start INTEGER,end INTEGER,deviceId INTEGER,bytes INTEGER,copyKind INTEGER);
            CREATE TABLE TARGET_INFO_GPU(name TEXT);
            INSERT INTO StringIds VALUES (1,'mul_mat_vec_q'),(2,'cudaGraphLaunch'),(3,'ggml_cuda_ar_ll_kernel_n<float, __nv_bfloat16>');
            INSERT INTO NVTX_EVENTS VALUES (0,100,16777217,'MUL_MAT projection-3 r=2000 w=100 w0=blk.3.attn_q.weight',NULL);
            INSERT INTO CUDA_GRAPH_NODE_EVENTS VALUES (20,20,16777217,10,NULL),(200,200,16777217,11,10);
            INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (300,320,16777217,7,2);
            INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES
                (400,500,0,16777216,7,1,1,11),
                (450,550,1,16777216,7,1,1,11),
                (600,700,0,16777216,8,3,3,0);
            INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (500,1500,0,54000,1);
            INSERT INTO TARGET_INFO_GPU VALUES ('NVIDIA RTX PRO 6000 Blackwell Server Edition');
        """)
        db.commit()
        db.close()
        return path

    def test_clone_replay_keeps_its_original_operation_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile = decode_lab.export_trace(self.fixture(tmp), {'label': 'fixture', 'active': 1})
        self.assertEqual(profile['counts']['kernels'], 3)
        self.assertEqual(profile['coverage']['attributed_kernels'], 2)
        self.assertEqual(len(profile['operations']), 2)
        self.assertEqual({o['device'] for o in profile['operations']}, {0, 1})
        for op in profile['operations']:
            self.assertEqual(op['weight'], 'blk.3.attn_q.weight')
            self.assertEqual(op['layer'], 3)
            self.assertEqual(op['logical_read_bytes'], 2000)
            self.assertEqual(op['instances'], 1)
        self.assertEqual(len(profile['kernel_events']), 3)

    def test_gpu_work_is_not_reported_as_wall_time_or_pcie_bandwidth(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile = decode_lab.export_trace(self.fixture(tmp), {})
        self.assertAlmostEqual(profile['summary']['kernel_sum_us'], 0.3)
        self.assertAlmostEqual(profile['summary']['kernel_union_us'], 0.25)
        collective = next(k for k in profile['kernels'] if 'ar_ll' in k['name'])
        self.assertIsNone(collective['measured_bytes'])
        self.assertIsNone(collective['gbps'])
        self.assertEqual(collective['family'], 'Expert reduction')

    def test_explicit_copy_rate_uses_recorded_payload_and_duration_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile = decode_lab.export_trace(self.fixture(tmp), {})
        copy = profile['copies'][0]
        self.assertEqual(copy['bytes'], 54000)
        self.assertAlmostEqual(copy['sum_us'], 1)
        self.assertAlmostEqual(copy['gbps'], 54)
        self.assertEqual(copy['direction'], 'Host to GPU')

    def test_fused_children_do_not_duplicate_logical_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.fixture(tmp)
            db = sqlite3.connect(path)
            db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (505,520,0,16777216,7,1,1,11)')
            db.commit()
            db.close()
            profile = decode_lab.export_trace(path, {})
        operation = next(o for o in profile['operations'] if o['device'] == 0)
        self.assertEqual(operation['instances'], 1)
        self.assertEqual(operation['kernel_calls'], 2)
        self.assertEqual(operation['logical_read_bytes'], 2000)

    def test_missing_optional_tables_and_unattributed_nodes_are_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.fixture(tmp)
            db = sqlite3.connect(path)
            db.execute('DROP TABLE CUDA_GRAPH_NODE_EVENTS')
            db.execute('DROP TABLE NVTX_EVENTS')
            db.commit()
            db.close()
            profile = decode_lab.export_trace(path, {})
        self.assertEqual(profile['coverage']['attributed_kernels'], 0)
        self.assertEqual(profile['coverage']['unattributed_kernels'], 3)
        self.assertIn('NVTX_EVENTS', profile['missing_tables'])

    def test_generated_interval_unions_match_an_independent_discrete_model(self):
        rng = random.Random(6000)
        for _ in range(150):
            pairs = [(rng.randrange(40), rng.randrange(1, 15)) for _ in range(20)]
            intervals = [(start, start + length) for start, length in pairs]
            expected = len({t for start, end in intervals for t in range(start, end)})
            self.assertEqual(decode_lab.union_ns(intervals), expected)

    def test_cli_exports_gzip_without_publishing_private_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.fixture(tmp)
            out = Path(tmp) / 'capture.json.gz'
            result = subprocess.run([sys.executable, str(Path(__file__).with_name('decode_lab.py')), 'profile',
                                     '--sqlite', str(path), '--output', str(out), '--label', 'fixture',
                                     '--active', '1', '--context', '4096'],
                                    text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            with gzip.open(out, 'rt') as f:
                data = json.load(f)
        self.assertEqual(data['schema'], 'glm53-profile-v1')
        self.assertEqual(data['metadata']['active'], 1)
        self.assertEqual(data['metadata']['context'], 4096)
        self.assertEqual(data['metadata']['kv'], 'unknown')
        self.assertEqual(data['metadata']['depth_policy'], 'unknown')
        self.assertIsNone(data['metadata']['configured_slots'])
        self.assertNotIn(str(path), json.dumps(data))


class LocalComparisonServerTest(unittest.TestCase):
    def test_page_and_ideal_controls_work_through_the_real_local_http_api(self):
        server = decode_lab.make_server(ROOT / 'docs/torad/decode-lab', 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base = 'http://127.0.0.1:' + str(server.server_port)
            with urllib.request.urlopen(base + '/api/ideal?sessions=4&peer_gbps=27') as response:
                result = json.load(response)
            self.assertEqual(response.headers.get('Cache-Control'), 'no-store')
            self.assertEqual(result['result']['sessions'], 4)
            self.assertEqual(result['result']['parameters']['peer_gbps'], 27)
            self.assertAlmostEqual(sum(r['time_us'] for r in result['rows']),
                                   result['result']['cycle_ms'] * 1000)
            for suffix in ('sessions=0', 'manifest=/etc/passwd', 'peer_gbps=nan'):
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    urllib.request.urlopen(base + '/api/ideal?' + suffix)
                self.assertEqual(caught.exception.code, 400)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


class IdealDetailTest(unittest.TestCase):
    def test_exact_tensor_and_transfer_ledger_reconciles_to_parent_estimate(self):
        # A small manifest keeps the test independent of ignored captured headers.
        tensors = [
            {'name': 'blk.0.attn_norm.weight', 'bytes': 1000, 'type': 'F32'},
            {'name': 'blk.3.ffn_up_exps.weight', 'bytes': 288000, 'type': 'IQ3_XXS'},
            {'name': 'output.weight', 'bytes': 2000, 'type': 'Q8_0'},
            {'name': 'output_norm.weight', 'bytes': 100, 'type': 'F32'},
            {'name': 'blk.45.nextn.eh_proj.weight', 'bytes': 500, 'type': 'Q8_0'},
        ]
        inputs = json.loads((ROOT / 'docs/torad/glm53-decode-roofline.json').read_text())['inputs']
        inputs.update(target_dense_read_bytes=3100, target_all_expert_bytes=288000,
                      draft_dense_and_shortlist_read_bytes=500 + 427819008,
                      draft_all_expert_bytes=0, draft_cache_only_read_bytes=500)
        detail = decode_lab.build_ideal(tensors, inputs, sessions=40)
        expected = estimate_decode_pcie.estimate(inputs, sessions=40)
        self.assertAlmostEqual(sum(row['time_us'] for row in detail['rows']), expected['cycle_ms'] * 1000)
        self.assertEqual(sum(r['kind'] == 'gather' for r in detail['rows']), 44)
        self.assertEqual(sum(r['kind'] == 'reduce' for r in detail['rows']), 44)
        self.assertFalse(any(r['kind'] == 'scatter' for r in detail['rows']))
        routed = next(r for r in detail['rows'] if r.get('weight') == 'blk.3.ffn_up_exps.weight')
        self.assertLess(routed['read_bytes'], 288000 / 4)
        self.assertTrue(all(r['basis'] for r in detail['rows']))
        self.assertTrue(all('PCIe' in r['basis'] for r in detail['rows'] if r['kind'] == 'gather'))
        self.assertTrue(all('assumption' in r['basis'] for r in detail['rows'] if r['kind'] == 'latency'))


if __name__ == '__main__':
    unittest.main()
