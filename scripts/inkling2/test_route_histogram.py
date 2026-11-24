#!/usr/bin/env python3
"""Checks for route_histogram.py: parse/dedupe/uniform-expectation math and the
--check gate's teeth, on synthetic stores. Run with python3 -I."""
import io
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import route_histogram as rh


def site_lines(layers):
    out = []
    site = 0
    for layer in layers:
        for kind in ("ffn_gate_up_exps", "ffn_down_exps"):
            out.append('MMQ_ROUTE_SITE {"ctx":0,"device":0,"site":%d,"name":"blk.%d.%s.weight",'
                       '"type":"q3_K","experts":384,"experts_total":0,"paired":false}' % (site, layer, kind))
            site += 1
    return out


class TestUniformExpectation(unittest.TestCase):
    def test_one_token_selects_exactly_top_k_distinct(self):
        self.assertAlmostEqual(rh.uniform_expected_distinct(384, 8, 1), 8.0, places=9)

    def test_monotone_in_tokens_and_below_capacity(self):
        prev = -1.0
        for b in range(1, 400):
            e = rh.uniform_expected_distinct(384, 8, b)
            self.assertGreater(e, prev)
            self.assertLessEqual(e, 384)
            prev = e

    def test_known_value_at_twelve_tokens(self):
        # hand-computed via ln(47/48): 384 * (1 - exp(12*ln(47/48))) = 85.729247...
        self.assertAlmostEqual(rh.uniform_expected_distinct(384, 8, 12), 85.729247, places=5)


class TestParseStore(unittest.TestCase):
    def build_store(self, tmp, lines):
        path = os.path.join(tmp, "expert-routes.jsonl")
        with open(path, "w") as fh:
            fh.write("".join(line + "\n" for line in lines))
        return path

    def test_dedupe_products_and_devices(self):
        # two layers; each routing appears once per product x device = 4 records,
        # of which exactly one launch event must survive.
        lines = site_lines([0])
        for ctx in range(2):   # two devices
            for eval_ in (391, 395):
                for site in (0, 1):  # gate_up then down of layer 0
                    lines.append('MMQ_ROUTE {"ctx":%d,"seq":%d,"eval":%d,"site":%d,"cols":96,'
                                 '"used":62,"tiles8":62,"tiles":62,"j":8}' % (ctx, site, eval_, site))
        with tempfile.TemporaryDirectory() as tmp:
            launches, counts = rh.parse_store(self.build_store(tmp, lines))
        self.assertEqual(counts.raw_route_records, 8)
        self.assertEqual(len(launches), 2)          # two evals x one layer
        self.assertEqual(counts.product_repeats, 6)  # (product-1)+(device-1) repeats x2 evals
        self.assertEqual(counts.launches, len(launches))

    def test_phase_boundary_and_bad_cols(self):
        lines = site_lines([0]) + [
            'MMQ_ROUTE {"ctx":0,"seq":0,"eval":1,"site":0,"cols":256,"used":100,"tiles8":100,"tiles":100,"j":8}',
            'MMQ_ROUTE {"ctx":0,"seq":0,"eval":2,"site":0,"cols":264,"used":150,"tiles8":150,"tiles":150,"j":8}',
            'MMQ_ROUTE {"ctx":0,"seq":0,"eval":3,"site":0,"cols":97,"used":50,"tiles8":50,"tiles":50,"j":8}',
        ]
        with tempfile.TemporaryDirectory() as tmp:
            launches, counts = rh.parse_store(self.build_store(tmp, lines))
        phases = {phase for (_l, phase, _c, _u) in launches}
        cols_seen = sorted(c for (_l, _p, c, _u) in launches)
        self.assertIn("decode", phases)
        self.assertIn("prefill", phases)
        self.assertNotIn(97, cols_seen)              # not divisible by top_k -> rejected
        self.assertEqual(counts.bad_cols, 1)

    def test_unknown_site_counted(self):
        lines = site_lines([0]) + [
            'MMQ_ROUTE {"ctx":0,"seq":0,"eval":7,"site":99,"cols":96,"used":10,"tiles8":10,"tiles":10,"j":8}',
        ]
        with tempfile.TemporaryDirectory() as tmp:
            launches, counts = rh.parse_store(self.build_store(tmp, lines))
        self.assertEqual(counts.unknown_sites, 1)
        self.assertEqual(len(launches), 0)


class TestBuildAndCheck(unittest.TestCase):
    ARTIFACT_LINES = None

    def make_artifact(self, tmp, extra_records=None):
        layers = [0, 1]
        lines = site_lines(layers)
        seq = 0
        for eval_ in range(1, 4):
            for layer in layers:
                site = layer * 2
                cols = 96 if (layer + eval_) % 2 else 184   # mix both decode batch shapes
                used = 40 + 3 * layer
                for ctx in range(2):                        # two device copies
                    lines.append('MMQ_ROUTE {"ctx":%d,"seq":%d,"eval":%d,"site":%d,"cols":%d,'
                                 '"used":%d,"tiles8":%d,"tiles":%d,"j":8}'
                                 % (ctx, seq, eval_, site, cols, used, used, used))
                    seq += 1
        if extra_records:
            lines.extend(extra_records)
        store = os.path.join(tmp, "expert-routes.jsonl")
        with open(store, "w") as fh:
            fh.write("".join(line + "\n" for line in lines))
        out = os.path.join(tmp, "artifact.json")
        artifact = rh.build([store], out)
        return artifact, out

    def test_build_totals_and_verdict_branch_c(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact, out = self.make_artifact(tmp)
        totals = artifact["totals"]
        # 6 launch events: 3 evals x 2 layers; each seen once per device copy,
        # one product site each in this fixture -> 6 first-hits + 6 repeats
        self.assertEqual(totals["launches"], 6)
        self.assertEqual(totals["raw_route_records"], 12)
        self.assertEqual(totals["accounting_sum"], totals["raw_route_records"])
        self.assertIsNone(artifact["verdict"]["measured_top1_share"])
        self.assertEqual(artifact["verdict"]["branch"], "c")
        self.assertFalse(totals["expert_hit_histograms_present"])
        # slot draws recomputable from the histogram the artifact carries
        hist = artifact["cols_histogram_by_phase"]
        s = sum(int(c) * n for phase in ("decode", "prefill") for c, n in hist[phase].items())
        self.assertEqual(s, totals["slot_draws"])

    def test_check_gate_passes_fresh_artifact_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, good = self.make_artifact(tmp)
            self.assertEqual(rh.check(good), 0)

            # a claimed top-1 share without hit histograms must fail loudly
            with open(good) as fh:
                d = json.load(fh)
            d["verdict"]["measured_top1_share"] = 0.05   # above the 3.125% bar too
            bad = os.path.join(tmp, "bad.json")
            with open(bad, "w") as fh:
                json.dump(d, fh)
            self.assertEqual(rh.check(bad), 1)

            # branch (a)/(b)-style claims need evidence that cannot exist here yet;
            # verdict.branch forced to something else must also fail
            with open(good) as fh:
                d2 = json.load(fh)
            d2["verdict"]["branch"] = "a"
            bad2 = os.path.join(tmp, "bad2.json")
            with open(bad2, "w") as fh:
                json.dump(d2, fh)
            self.assertEqual(rh.check(bad2), 1)

    def test_check_detects_broken_accounting(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact, out = self.make_artifact(tmp)
            d = json.load(open(out))
            d["totals"]["slot_draws"] += 8     # one phantom launch of 96 cols... say 8 slots
            bad = os.path.join(tmp, "bad.json")
            with open(bad, "w") as fh:
                json.dump(d, fh)
            self.assertEqual(rh.check(bad), 1)


class TestHitHistograms(unittest.TestCase):
    def store(self, tmp, hits_by_layer, ctxs=(0, 1)):
        lines = []
        for ctx in ctxs:   # tensor split: each context sees every expert and the same routing
            lines += [l.replace('"ctx":0', '"ctx":%d' % ctx) for l in site_lines(sorted(hits_by_layer))]
            for i, layer in enumerate(sorted(hits_by_layer)):
                for site in (2 * i, 2 * i + 1):   # gate_up and down carry one routing
                    lines.append('MMQ_ROUTE_HIST {"ctx":%d,"device":0,"site":%d,"launches":5,"bins":384,"hits":%s}'
                                 % (ctx, site, json.dumps(hits_by_layer[layer])))
            lines.append('MMQ_ROUTE {"ctx":%d,"seq":0,"eval":1,"site":0,"cols":96,"used":60,"tiles8":60,"tiles":60,"j":8}' % ctx)
        path = os.path.join(tmp, "routes.jsonl")
        with open(path, "w") as fh:
            fh.write("".join(l + "\n" for l in lines))
        return path

    def test_uniform_routing_is_branch_b(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "a.json")
            a = rh.build([self.store(tmp, {0: [8] * 384, 1: [8] * 384})], out)
            self.assertEqual(rh.check(out), 0)
        self.assertTrue(a["totals"]["expert_hit_histograms_present"])
        self.assertEqual(a["verdict"]["branch"], "b")
        self.assertAlmostEqual(a["verdict"]["measured_top1_share"], 8 / 384.0 * 8 / 8.0 * 1.0, places=6)
        self.assertEqual(a["totals"]["hist_dropped"]["site_repeats"], 4)   # down sites, two contexts
        self.assertEqual(a["totals"]["hist_dropped"]["device_repeats"], 2)  # the second context's copies
        self.assertEqual(a["per_layer"][0]["expert_hits"]["slot_draws"], 8 * 384)
        self.assertTrue(a["verdict"]["sample_noise_reaches_bar"])   # 8 hits an expert: noise alone reaches 1.5x

    def test_hot_expert_is_branch_a(self):
        hot = [8] * 384
        hot[17] = 400
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "a.json")
            a = rh.build([self.store(tmp, {0: [8] * 384, 3: hot})], out)
            self.assertEqual(rh.check(out), 0)
            self.assertEqual(a["verdict"]["branch"], "a")
            self.assertEqual(a["verdict"]["hot_layers"], [3])
            d = json.load(open(out))
            d["per_layer"][3]["expert_hits"]["hits"][17] = 8   # the hot leader erased, shares left as they were
            with open(out, "w") as fh:
                json.dump(d, fh)
            self.assertEqual(rh.check(out), 1)

    def test_evidence_flag_is_the_artifact_path(self):
        import subprocess
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "a.json")
            rh.build([self.store(tmp, {0: [8] * 384})], out)
            r = subprocess.run([sys.executable, "-I", os.path.join(HERE, "route_histogram.py"),
                                "--check", "--evidence", out], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
