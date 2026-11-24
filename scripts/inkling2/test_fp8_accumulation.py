#!/usr/bin/env python3
"""Checks for fp8_accumulation.py: the parsers, the verdict and the --check gate's teeth on synthetic evidence."""
import copy
import json
import os
import sys
import tempfile
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fp8_accumulation as fa

PAIR_LINES = [
    "encode: 4096 values, mean hi err 5.600e-01, mean two-term err 1.200e-02",
    "fold32: 32/32 single-lane exact, x32 worst rel 1.200e-03, mixed worst rel 5.500e-04",
    "fold_slots(256): worst rel 3.100e-04",
    "sum_terms two-term (4 values,  8 terms): worst rel 2.000e-03",
    "merge pattern (256 parts, mode 0): worst rel 4.000e-04",
    "test-inkling-fp8-pair: all checks passed",
]


def model_lines(nmse, agree, rows=96, kernel=True):
    out = ["inkling2 kv1b-e4m3/served: 0 of them off the device"]
    if kernel:
        out.append("inkling2 kv1b-e4m3/served: ggml_cuda_fattn: kernel REL_DECODE on K f8_e4m3")
    out += ["inkling2 kv1b-e4m3/served: %d logits, NMSE %g (at most 0.0001)" % (rows * 128, nmse),
            "inkling2 kv1b-e4m3/served: argmax agrees on %d of %d rows" % (agree, rows),
            "OK inkling2 kv1b-e4m3/served"]
    return out


def evidence():
    run = lambda lines, env=None, rc=0: {"cmd": [], "env": env or {}, "rc": rc, "seconds": 1.0, "lines": lines}
    runs = {"pair": run(PAIR_LINES), "tile_ops": run(["  62/62 tests passed"]), "archs_inkling2": run([])}
    for c in fa.MODEL_CASES:
        runs["model_on_" + c] = run(model_lines(3.1e-5, 95), {"GGML_CUDA_FATTN_FP8_ACCUM": "1"})
        runs["model_off_" + c] = run(model_lines(3.0e-5, 95), {})
    ev = {"row": fa.ROW, "device": {"name": "NVIDIA GeForce RTX 5090", "compute_cap": "12.0"},
          "source_sha256": {s: fa.sha256_file(fa.ROOT / s) for s in fa.SOURCES}, "runs": runs}
    ev.update(fa.judge(ev))
    return ev


class TestParsers(unittest.TestCase):
    def test_pair_figures(self):
        p = fa.parse_pair(PAIR_LINES)
        self.assertTrue(p["passed"])
        self.assertAlmostEqual(p["pair_gain"], 0.56 / 0.012)
        self.assertEqual(p["fold32_exact_lanes"], 32)
        self.assertEqual(p["merge_worst_rel"][0]["parts"], 256)

    def test_ops_counts(self):
        self.assertEqual(fa.parse_ops(["  7/9 tests passed"]), {"passed": 7, "total": 9, "fail_lines": 0})

    def test_model_figures(self):
        m = fa.parse_model(model_lines(2e-5, 90))
        self.assertEqual((m["nmse"], m["argmax_agree"], m["rows"], m["ok"]), (2e-5, 90, 96, True))
        self.assertEqual(m["kernels"], ["REL_DECODE f8_e4m3"])


class TestJudge(unittest.TestCase):
    def test_clean_evidence_passes(self):
        self.assertTrue(evidence()["verdict"]["pass"])

    def test_identical_logits_mean_the_switch_did_not_engage(self):
        ev = evidence()
        ev["runs"]["model_on_served"]["lines"] = model_lines(3.0e-5, 95)
        self.assertIn("model_served_engaged", fa.judge(ev)["verdict"]["failed"])

    def test_greedy_pick_change_fails(self):
        ev = evidence()
        ev["runs"]["model_on_restore"]["lines"] = model_lines(3.1e-5, 94)
        self.assertIn("model_restore_greedy_parity", fa.judge(ev)["verdict"]["failed"])

    def test_nmse_above_bar_fails(self):
        ev = evidence()
        ev["runs"]["model_on_served"]["lines"] = model_lines(2e-4, 95)
        self.assertIn("model_served_ok", fa.judge(ev)["verdict"]["failed"])

    def test_wrong_kernel_fails(self):
        ev = evidence()
        ev["runs"]["model_on_served"]["lines"] = model_lines(3.1e-5, 95, kernel=False)
        self.assertIn("model_served_engaged", fa.judge(ev)["verdict"]["failed"])

    def test_weak_pair_gain_fails(self):
        ev = evidence()
        ev["runs"]["pair"]["lines"] = [l.replace("1.200e-02", "2.000e-01") for l in PAIR_LINES]
        self.assertIn("pair_gain_at_least_bar", fa.judge(ev)["verdict"]["failed"])

    def test_not_blackwell_fails(self):
        ev = evidence()
        ev["device"]["compute_cap"] = "8.9"
        self.assertIn("device_sm_120", fa.judge(ev)["verdict"]["failed"])


class TestCheck(unittest.TestCase):
    def check(self, ev):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ev.json")
            with open(path, "w") as fh:
                json.dump(ev, fh)
            try:
                return fa.cmd_check(types.SimpleNamespace(evidence=path))
            except SystemExit:
                return 2

    def test_check_passes_fresh_evidence(self):
        self.assertEqual(self.check(evidence()), 0)

    def test_check_rejects_a_forged_verdict(self):
        ev = evidence()
        ev["runs"]["tile_ops"]["lines"] = ["  60/62 tests passed"]
        self.assertEqual(self.check(ev), 2)  # recorded checks no longer follow from the lines

    def test_check_rejects_stale_sources(self):
        ev = evidence()
        ev["source_sha256"][fa.SOURCES[0]] = "0" * 64
        self.assertEqual(self.check(ev), 2)

    def test_check_rejects_reference_with_switch_on(self):
        ev = copy.deepcopy(evidence())
        ev["runs"]["model_off_served"]["env"] = {"GGML_CUDA_FATTN_FP8_ACCUM": "1"}
        self.assertEqual(self.check(ev), 2)

    def test_negative_verdict_exits_one(self):
        ev = evidence()
        ev["runs"]["archs_inkling2"]["rc"] = 1
        ev.update(fa.judge(ev))
        self.assertEqual(self.check(ev), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
