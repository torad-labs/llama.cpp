"""search_evidence on synthetic ledger summaries only: no server, no model, no GPU."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import search_evidence as se


def slot(n_per_step=0.0, sum_per_step=0.0, mean=0.0, p50=0, p99=0, top=0):
    """A summary() slot as scripts/ledger.py writes one."""
    return {"n_per_step": n_per_step, "sum_per_step": sum_per_step, "mean": mean, "p50": p50, "p99": p99, "max": top}


def trial(rate, slots, seconds=1.0, steps=1000, key="trial", settings=None):
    return {"key": key, "tokens_per_second": rate, "settings": settings or {},
            "ledger": {"seconds": seconds, "steps": steps, "unit": "decode", "slots": slots}}


ZERO = {"device_busy_share": 0.0, "captures_per_step": 0.0, "evictions_per_step": 0.0, "rejects_per_step": 0.0,
        "meta_reduce_gather_us_per_step": 0.0, "decode_graph_compute_us_per_step": 0.0, "rows_per_step": 0.0}


class FeaturesTest(unittest.TestCase):
    def test_a_fast_trial_reads_as_card_time_every_decode(self):
        found = se.features(trial(101.5, {
            "cuda.span_us[0]": slot(n_per_step=0.125, sum_per_step=250.0, mean=250.0),
            "cuda.span_us[1]": slot(n_per_step=0.125, sum_per_step=250.0, mean=250.0),
            "cuda.span_us[2]": slot(n_per_step=0.125, sum_per_step=250.0, mean=250.0),
            "cuda.span_us[3]": slot(n_per_step=0.125, sum_per_step=250.0, mean=250.0),
            # the capture count is the slot's additions a step; the time it spent is not the count
            "cuda.eval.capture_us[0]": slot(n_per_step=0.001, sum_per_step=12.0, mean=12000.0),
            "cuda.eval.capture_us[1]": slot(n_per_step=0.001, sum_per_step=12.0, mean=12000.0),
            "meta.reduce_us": slot(n_per_step=6.0, sum_per_step=12.5, mean=2.1),
            "meta.gather_us": slot(n_per_step=6.0, sum_per_step=4.5, mean=0.75),
            "decode.graph_compute": slot(n_per_step=1.0, sum_per_step=1500.0, mean=1500.0),
            "decode.rows": slot(n_per_step=1.0, sum_per_step=240.0, mean=240.0),
        }))
        # wall is 1.0 s / 1000 steps = 1000 us a decode; four cards at 250 us each are a card's worth of work
        self.assertEqual(found, {"device_busy_share": 1.0, "captures_per_step": 0.002, "evictions_per_step": 0.0,
                                 "rejects_per_step": 0.0, "meta_reduce_gather_us_per_step": 17.0,
                                 "decode_graph_compute_us_per_step": 1500.0, "rows_per_step": 240.0})

    def test_a_slow_trial_reads_its_evictions_and_refusals(self):
        found = se.features(trial(40.0, {
            "cuda.span_us[0]": slot(mean=500.0),
            "cuda.span_us[1]": slot(mean=500.0),
            "cuda.span_us[2]": slot(mean=400.0),
            "cuda.span_us[3]": slot(mean=400.0),
            "cuda.eval.capture_us[0]": slot(n_per_step=0.2),
            "cuda.eval.capture_us[1]": slot(n_per_step=0.2),
            "cuda.graph.evict.idle[0]": slot(n_per_step=0.05, sum_per_step=40.0),
            "cuda.graph.evict.count_cap[0]": slot(n_per_step=0.01, sum_per_step=8.0),
            # each refusal adds in rows it turned down; the count is the additions
            "graph.kept.reject.ubatch": slot(n_per_step=0.3, sum_per_step=900.0),
            "graph.kept.reject.kpool.pool": slot(n_per_step=0.2, sum_per_step=64.0),
            "decode.graph_compute": slot(n_per_step=1.0, sum_per_step=3200.0),
            "decode.rows": slot(n_per_step=1.0, sum_per_step=128.0),
        }, seconds=4.0, steps=400))
        # wall is 4.0 s / 400 steps = 10000 us; 1800 us of card time is 18% of it
        self.assertEqual(found, {"device_busy_share": 0.18, "captures_per_step": 0.4, "evictions_per_step": 0.06,
                                 "rejects_per_step": 0.5, "meta_reduce_gather_us_per_step": 0.0,
                                 "decode_graph_compute_us_per_step": 3200.0, "rows_per_step": 128.0})

    def test_a_trial_without_a_ledger_is_all_zeros(self):
        for empty in ({"tokens_per_second": 1.0}, {"tokens_per_second": 1.0, "ledger": None},
                      {"tokens_per_second": 1.0, "ledger": {"seconds": None, "steps": 0, "unit": "decode",
                                                           "slots": {}}},
                      {"tokens_per_second": 1.0, "ledger": {"seconds": 1.0, "steps": 7, "slots": None}}):
            self.assertEqual(se.features(empty), ZERO)

    def test_a_wall_time_that_is_missing_or_zero_does_not_divide(self):
        spans = {"cuda.span_us[0]": slot(mean=500.0)}
        for seconds, steps in ((0.0, 1000), (1.0, 0), (None, 1000)):
            found = se.features(trial(1.0, spans, seconds=seconds, steps=steps))
            self.assertEqual(found["device_busy_share"], 0.0)
            self.assertEqual(found["rows_per_step"], 0.0)

    def test_gauges_and_unknown_slots_are_ignored(self):
        found = se.features(trial(1.0, {"cuda.mem.used_mib[0]": {"value": 900, "max": 1000},
                                        "decode.graph_compute": {"value": 5, "max": 5},
                                        "decode.total": slot(p50=2000, p99=4000)}))
        self.assertEqual(found, ZERO)


class UnstableTest(unittest.TestCase):
    def test_a_steady_window_has_no_flags(self):
        found = se.unstable_flags(trial(100.0, {
            "decode.total": slot(n_per_step=1.0, sum_per_step=2000.0, mean=2000.0, p50=2048, p99=4096),
            "cuda.eval.capture_us[0]": slot(n_per_step=0.5),       # exactly the constant is still a steady window
            "graph.kept.reject.ubatch": slot(n_per_step=1.0),      # one a step is the ordinary varying-shape cost
        }))
        self.assertEqual(found, [])

    def test_the_signs_a_window_was_moving_are_named_in_a_fixed_order(self):
        found = se.unstable_flags(trial(100.0, {
            "decode.total": slot(n_per_step=1.0, sum_per_step=2000.0, mean=2000.0, p50=1000, p99=8000),
            "cuda.eval.capture_us[0]": slot(n_per_step=0.51),
            "graph.kept.reject.ubatch": slot(n_per_step=0.6),
            "graph.kept.reject.kpool.pool": slot(n_per_step=0.5),
        }))
        self.assertEqual(found, ["decode_total_p99_over_3x_p50", "captures_per_step_high", "rejects_per_step_high"])

    def test_a_run_with_no_ledger_has_no_flags(self):
        self.assertEqual(se.unstable_flags({"tokens_per_second": 1.0}), [])
        self.assertEqual(se.unstable_flags(trial(1.0, {"decode.total": slot(p50=1000)})), [])


class ExplainTest(unittest.TestCase):
    def shaped(self, key, rate, span_mean, rows):
        return trial(rate, {"cuda.span_us[0]": slot(mean=span_mean), "decode.rows": slot(sum_per_step=rows)},
                     key=key, settings={"--draft-depth": key.lstrip("t")})

    def field(self):
        # wall is 1000 us a decode: the share is the span's mean in thousands, rows its sum a step
        return [self.shaped("t1", 100.0, 800.0, 256.0), self.shaped("t2", 95.0, 600.0, 192.0),
                self.shaped("t3", 40.0, 300.0, 128.0), self.shaped("t4", 30.0, 200.0, 64.0)]

    def test_the_fastest_trials_against_the_rest(self):
        found = se.explain([*self.field()[2:3], self.field()[0], *self.field()[3:], self.field()[1]], top=2)
        table = found["features"]
        self.assertEqual(list(table)[:2], ["rows_per_step", "device_busy_share"])  # by |delta|, the rest alphabetical
        self.assertEqual(len(table), 7)
        self.assertEqual(table["rows_per_step"], {"top_mean": 224.0, "rest_mean": 96.0, "delta": 128.0})
        self.assertEqual(table["device_busy_share"], {"top_mean": 0.7, "rest_mean": 0.25, "delta": 0.45})
        self.assertEqual(table["captures_per_step"], {"top_mean": 0.0, "rest_mean": 0.0, "delta": 0.0})
        self.assertEqual([entry["key"] for entry in found["trials"]], ["t1", "t2"])
        self.assertEqual(found["trials"][0]["why"], "rows_per_step +160, device_busy_share +0.55")
        self.assertEqual(found["trials"][1]["why"], "rows_per_step +96, device_busy_share +0.35")

    def test_one_top_trial_against_every_other(self):
        found = se.explain(self.field(), top=1)
        table = found["features"]
        self.assertEqual(table["rows_per_step"], {"top_mean": 256.0, "rest_mean": 128.0, "delta": 128.0})
        self.assertEqual(table["device_busy_share"]["top_mean"], 0.8)
        self.assertEqual(table["device_busy_share"]["rest_mean"], 0.366667)
        self.assertEqual(found["trials"][0]["why"], "rows_per_step +128, device_busy_share +0.433333")

    def test_a_lone_trial_has_no_rest_to_compare_to(self):
        found = se.explain(self.field()[:1])
        self.assertEqual(found["features"]["rows_per_step"], {"top_mean": 256.0, "rest_mean": 0.0, "delta": 256.0})
        self.assertEqual(found["trials"][0]["why"], "rows_per_step +256, device_busy_share +0.8")

    def test_trials_without_a_rate_do_not_rank(self):
        found = se.explain([{"key": "dead", "tokens_per_second": None, "error": "the server exited",
                             "ledger": None}, *self.field()], top=2)
        self.assertEqual([entry["key"] for entry in found["trials"]], ["t1", "t2"])
        self.assertEqual(found["features"]["rows_per_step"]["rest_mean"], 96.0)

    def test_no_rated_trial_gives_an_empty_table(self):
        self.assertEqual(se.explain([]), {"features": {}, "trials": []})
        self.assertEqual(se.explain([{"key": "dead", "tokens_per_second": None}]), {"features": {}, "trials": []})


if __name__ == "__main__":
    unittest.main()
