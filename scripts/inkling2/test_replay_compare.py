import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import json

sys.path.insert(0, str(Path(__file__).parent))
from replay_compare import check_selected, compare
from replay_prepare import replay_manifest, token_digest
from replay_run import VALUE_FLAGS, engine_args, query_first_exercised, smoke_input

class ReplayComparison(unittest.TestCase):
    def test_same_prefix_full_logit_kl(self):
        with tempfile.TemporaryDirectory() as d:
            inputs = make_fixture(Path(d))
            out = compare(inputs)
            rows = out['rows']
            self.assertEqual(len(rows), 1)
            r = rows[0]
            self.assertAlmostEqual(r['stock_a/stock_b']['kl_ab'], 0.0, places=7)
            self.assertAlmostEqual(r['attention/stock_a']['kl_ab'], 0.2 * math.log(1.5), places=7)
            self.assertAlmostEqual(r['attention/stock_a']['kl_ba'], 0.2 * math.log(1.5), places=7)
            self.assertEqual(r['attention/stock_a']['top_a'], 1)
            self.assertEqual(r['attention/stock_a']['top_b'], 0)
            self.assertAlmostEqual(r['greedy/stock_b']['kl_ab'], 0.0, places=7)
            inputs["stack"] = inputs["greedy"]
            self.assertAlmostEqual(compare(inputs)["rows"][0]["stack/stock_a"]["kl_ba"], 0.0, places=7)

    def test_refuses_to_compare_different_conditional_rows(self):
        with tempfile.TemporaryDirectory() as d:
            inputs = make_fixture(Path(d))
            path = inputs['greedy'] / 'rows.json'
            meta = json.loads(path.read_text())
            meta['rows'][0]['condition']['forced_tokens'] = [19]
            path.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, 'conditional input'):
                compare(inputs)

    def test_selected_token_is_the_argmax_of_its_own_scores(self):
        with tempfile.TemporaryDirectory() as d:
            inputs = make_fixture(Path(d))
            self.assertEqual(check_selected(inputs['attention']), (1, 1, []))
            self.assertEqual(check_selected(inputs['stock_a']), (1, 1, []))
            path = inputs['stock_a'] / 'rows.json'
            meta = json.loads(path.read_text())
            meta['rows'][0]['selected_id'] = 1
            path.write_text(json.dumps(meta))
            self.assertEqual(check_selected(inputs['stock_a']), (0, 1, [(8, 2)]))

    def test_refuses_mixed_model_builds(self):
        with tempfile.TemporaryDirectory() as d:
            inputs = make_fixture(Path(d))
            path = inputs['attention'] / 'rows.json'
            meta = json.loads(path.read_text())
            meta['build_sha256'] = 'other'
            path.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, 'same build'):
                compare(inputs)


class ReplayPreparation(unittest.TestCase):
    def test_smoke_input_is_a_shorter_prompt_with_matching_digests(self):
        spec = {"common": list(range(10)), "private": [[1, 2, 3, 4], [5, 6, 7, 8]], "forced": [[9], [9]],
                "common_sha256": "old", "private_sha256": ["old", "old"]}
        short = smoke_input(spec, n_common=4, n_private=2)
        self.assertEqual((short["common"], short["private"]), ([0, 1, 2, 3], [[1, 2], [5, 6]]))
        self.assertEqual(short["common_sha256"], token_digest([0, 1, 2, 3]))
        self.assertEqual(short["private_sha256"], [token_digest([1, 2]), token_digest([5, 6])])
        self.assertEqual(short["forced"], spec["forced"])

    def test_counts_rows_that_can_take_the_query_first_launch(self):
        good = {"queries": 23, "query_heads": 48, "kv_cells": 300000, "kv_heads": 8, "low_mean": True}
        bad = dict(good, low_mean=False)
        per_rank = dict(good, query_heads=12, kv_heads=2)
        short = dict(good, kv_cells=100000)
        with tempfile.TemporaryDirectory() as d:
            rows = [{"attention_inputs": rows} for rows in ([good], [bad], [bad, per_rank], [short], [])]
            (Path(d) / "rows.json").write_text(json.dumps({"rows": rows}))
            self.assertEqual(query_first_exercised(Path(d), 4), (2, 5))

    def test_use_engine_flags_without_server_only_scheduling(self):
        expected = [item for i, flag in enumerate(VALUE_FLAGS) for item in (flag, str(i))]
        plan = {"argv": ["llama-server", "--prompt-step", "1024", *expected, "--kv-unified"]}
        self.assertEqual(engine_args(plan), expected + ["--attn-mask-bits"])

    def test_requires_exact_reference_prompt(self):
        prefix, tail = [1, 2], [3, 4]
        reference = {
            "status": "full_reuse", "common": 2, "private": 2, "common_sha256": token_digest(prefix),
            "clients": [{"client": 0, "private_sha256": token_digest(tail), "token_ids": [5, 6, 7, 8, 9]}],
        }
        manifest = replay_manifest(reference, [(json.dumps({"prompt": prefix + tail}), {"client": 0})],
                                   "build", "model")
        self.assertEqual(manifest["forced"], [[5, 6, 7, 8]])
        self.assertEqual(manifest["common"], prefix)
        self.assertEqual(manifest["private"], [tail])
        with self.assertRaisesRegex(ValueError, "does not match"):
            replay_manifest(reference, [(json.dumps({"prompt": [1, 2, 3, 9]}), {"client": 0})], "build", "model")


def make_fixture(base):
    inputs = {}
    for name in ('stock_a', 'stock_b', 'attention', 'greedy'):
        path = inputs[name] = base / name
        path.mkdir()
        scores = (math.log(.4), math.log(.6), 10) if name == 'attention' else (math.log(.6), math.log(.4), 10)
        (path/'logits.f32le').write_bytes(struct.pack('<3f', *scores))
        (path/'rows.json').write_text(json.dumps({'build_sha256':'build', 'model_sha256':'model', 'n_vocab':3, 'rows':[{
            'client': 8, 'token_position': 2,
            'condition': {'prompt_sha256':'test', 'forced_tokens':[17], 'query_position':262144,
                          'batch_sha256':'same order', 'kv_mask_sha256':'same mask', 'suppressed':[2]},
            'selected_id':1 if name=='attention' else 0
        }]}))
    return inputs

if __name__ == '__main__':
    unittest.main()
