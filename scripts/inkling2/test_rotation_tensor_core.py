#!/usr/bin/env python3
"""Checks for rotation_tensor_core.py: the statistics, the verdict rules and the evidence gates, on synthetic
measurements. The GPU half of --check (build, replay, fresh run) runs only on the box."""
import os
import random
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rotation_tensor_core as rtc


def synthetic_measure(tc_us=3.0, gen_us=2.8, fwht_us=4.0, q8_ratio=0.9, tc_rel=1e-3, fp16_ratio=0.3,
                      growth=0.04, growth16=0.005, stream_us=0.5, copy_us=0.9, cublas_us=None, shared_us=None):
    rng = random.Random(7)
    cublas_us = 10.0 if cublas_us is None else cublas_us
    shared_us = tc_us if shared_us is None else shared_us
    arms = {"fwht_f32": fwht_us, "fwht_q8": fwht_us + 0.4, "copy_f32": copy_us, "basis_stream": stream_us * 8,
            "basis_stream_shared": stream_us, "cublas_gemm": cublas_us, "cublas_full": cublas_us + 1.0}
    for cfg in rtc.CONFIGS:
        arms["tc_basis_f32/" + cfg] = tc_us + 0.5
        arms["tc_shared_f32/" + cfg] = shared_us
        arms["tc_gen_f32/" + cfg] = gen_us
        arms["tc_basis_q8/" + cfg] = tc_us + 0.8
        arms["tc_shared_q8/" + cfg] = shared_us + 0.3
        arms["tc_gen_q8/" + cfg] = gen_us + 0.3
    shapes = []
    for rows in rtc.ROWS:
        scale = 1.0 + rows / 512.0
        arms_d = {}
        for name, us in arms.items():
            s = [us * scale * (1 + rng.uniform(-0.01, 0.01)) for _ in range(rtc.ROUNDS)]
            arms_d[name] = dict(samples_us=s, **rtc.stats_of(s))
        shapes.append({"rows": rows, "arms": arms_d, "errors": {
            "q8": {"ratio": q8_ratio, "ref_rel": 0.005, "tc_vs_f32": 0.005 * (1 + growth)},
            "q8_fp16": {"ratio": fp16_ratio, "tc_vs_f32": 0.005 * (1 + growth16)},
            "tc_vs_f32": {"rel_l2": tc_rel}}})
    return {"shapes": shapes}


class TestStats(unittest.TestCase):
    def test_median_even_and_odd(self):
        self.assertEqual(rtc.stats_of([3.0, 1.0, 2.0])["median_us"], 2.0)
        self.assertEqual(rtc.stats_of([4.0, 1.0, 2.0, 3.0])["median_us"], 2.5)

    def test_percentiles_order(self):
        st = rtc.stats_of([float(i) for i in range(1, 41)])
        self.assertLessEqual(st["min_us"], st["p10_us"])
        self.assertLessEqual(st["p10_us"], st["median_us"])
        self.assertLessEqual(st["median_us"], st["p90_us"])

    def test_basis_read_bytes_counts_row_tiles(self):
        one = rtc.NBLK * rtc.BLOCK * rtc.BLOCK
        self.assertEqual(rtc.basis_read_bytes(23, "bm16_bn128_w8", shared=False), 2 * one)
        self.assertEqual(rtc.basis_read_bytes(23, "bm32_bn128_w8", shared=False), one)
        self.assertEqual(rtc.basis_read_bytes(23, "bm32_bn128_w8"), rtc.BLOCK * rtc.BLOCK + rtc.NBLK * rtc.BLOCK)


class TestJudge(unittest.TestCase):
    def test_pass_when_faster_and_error_below_bar(self):
        v = rtc.judge(synthetic_measure())["verdict"]
        self.assertTrue(v["pass"])
        self.assertEqual(v["decided_by"], "none")

    def test_bin_flip_ratio_alone_does_not_decide(self):
        self.assertTrue(rtc.judge(synthetic_measure(q8_ratio=0.95, growth=0.03))["verdict"]["pass"])

    def test_growth_bar_matches_half_error_rule(self):
        self.assertAlmostEqual(rtc.BAR_Q8_TOTAL_GROWTH, 0.118034, places=5)

    def test_bf16_rounding_falsifier_fires_first(self):
        j = rtc.judge(synthetic_measure(growth=0.2, tc_us=9.0))
        self.assertFalse(j["verdict"]["pass"])
        self.assertEqual(j["verdict"]["decided_by"], "bf16_input_rounding")
        self.assertTrue(j["falsifiers"]["bf16_input_rounding"]["fired"])

    def test_rotation_error_bar_alone_fails(self):
        v = rtc.judge(synthetic_measure(tc_rel=0.05))["verdict"]
        self.assertEqual(v["decided_by"], "bf16_input_rounding")

    def test_tuned_gemm_losing_decides_matmul_cost(self):
        j = rtc.judge(synthetic_measure(tc_us=18.0, gen_us=13.0, cublas_us=6.0))
        self.assertFalse(j["verdict"]["pass"])
        self.assertEqual(j["verdict"]["decided_by"], "tensor_core_matmul_cost")
        self.assertFalse(j["falsifiers"]["basis_read_cost"]["fired"])

    def test_prototype_loses_when_tuned_gemm_wins(self):
        v = rtc.judge(synthetic_measure(tc_us=18.0, gen_us=13.0, cublas_us=2.0))["verdict"]
        self.assertEqual(v["decided_by"], "prototype_kernel_cost")

    def test_shared_basis_floor_alone_decides_basis_cost(self):
        j = rtc.judge(synthetic_measure(tc_us=18.0, gen_us=13.0, stream_us=4.2))
        self.assertEqual(j["verdict"]["decided_by"], "basis_read_cost")
        self.assertTrue(j["falsifiers"]["basis_read_cost"]["fired"])

    def test_per_block_basis_does_not_decide(self):
        # nine signed copies streaming slower than the butterfly is not the true basis read
        m = synthetic_measure(tc_us=18.0, gen_us=13.0, stream_us=0.6)
        self.assertGreater(m["shapes"][0]["arms"]["basis_stream"]["median_us"], 4.0)
        self.assertEqual(rtc.judge(m)["verdict"]["decided_by"], "tensor_core_matmul_cost")

    def test_headroom_is_butterfly_over_larger_floor(self):
        f = rtc.judge(synthetic_measure(stream_us=1.1, copy_us=0.9))["falsifiers"]["basis_read_cost"]
        self.assertAlmostEqual(f["headroom_us"], f["decode_fwht_us"] - f["decode_shared_basis_stream_floor_us"])

    def test_decode_shape_required(self):
        m = synthetic_measure()
        m["shapes"] = [s for s in m["shapes"] if s["rows"] != 23]
        with self.assertRaises(SystemExit):
            rtc.judge(m)


class TestEvidenceGates(unittest.TestCase):
    def evidence(self):
        m = synthetic_measure()
        j = rtc.judge(m)
        derived = {"shapes": {str(k): v for k, v in j["derived"]["shapes"].items()}}
        return {"measure": m, "derived": derived,
                "gpu_state": {"ready": [{"name": "x", "util_pct": 0, "mem_used_mib": 300}]}}

    def test_fresh_evidence_passes_offline_gates(self):
        ev = self.evidence()
        rtc.check_timings(ev)
        rtc.check_derived(ev)
        rtc.check_quiet(ev)

    def test_tampered_median_is_caught(self):
        ev = self.evidence()
        ev["measure"]["shapes"][0]["arms"]["fwht_f32"]["median_us"] *= 2
        with self.assertRaises(SystemExit):
            rtc.check_timings(ev)

    def test_tampered_derived_is_caught(self):
        ev = self.evidence()
        ev["derived"]["shapes"]["23"]["fwht_f32_us"] += 1.0
        with self.assertRaises(SystemExit):
            rtc.check_derived(ev)

    def test_busy_card_is_caught(self):
        ev = self.evidence()
        ev["gpu_state"]["ready"][0]["util_pct"] = 90
        with self.assertRaises(SystemExit):
            rtc.check_quiet(ev)


if __name__ == "__main__":
    unittest.main(verbosity=2)
