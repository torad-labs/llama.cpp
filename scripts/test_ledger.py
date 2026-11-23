#!/usr/bin/env python3
"""The ledger's reader: what two snapshots differ by, a step."""
import io
import unittest

import ledger


def snapshot(t_us, steps, timed, slots):
    return {"version": 1, "on": 1, "t_us": t_us, "steps": steps, "timed_steps": timed, "period": 8, "slots": slots}


BEFORE = snapshot(1_000_000, 100, 13, {
    "decode.total": {"n": 100, "sum": 500_000, "max": 9000, "h": {"13": 100}},
    "stage.iteration": {"n": 50, "sum": 600_000, "max": 20000, "h": {"14": 50}},
    "cuda.span_us[0]": {"n": 13, "sum": 39_000, "max": 4000, "h": {"12": 13}},
    "cuda.mem.used_mib[0]": {"g": 90000, "max": 90000},
})
AFTER = snapshot(3_000_000, 300, 38, {
    "decode.total": {"n": 300, "sum": 1_500_000, "max": 120_000, "h": {"13": 298, "17": 2}},
    "stage.iteration": {"n": 150, "sum": 1_800_000, "max": 20000, "h": {"14": 150}},
    "cuda.span_us[0]": {"n": 38, "sum": 139_000, "max": 4100, "h": {"12": 38}},
    "cuda.mem.used_mib[0]": {"g": 97107, "max": 97107},
    "graph.kept.reject.ubatch_shape_or_sequences": {"n": 4, "sum": 276, "max": 69, "h": {"7": 4}},
})


class LedgerTest(unittest.TestCase):
    def test_the_difference_holds_only_what_happened_between_the_snapshots(self):
        diff = ledger.difference(BEFORE, AFTER)
        self.assertEqual(diff["seconds"], 2.0)
        self.assertEqual(diff["steps"], 200)
        self.assertEqual(diff["slots"]["decode.total"], {"n": 200, "sum": 1_000_000, "max": 120_000, "h": {13: 198, 17: 2}})
        self.assertEqual(diff["slots"]["graph.kept.reject.ubatch_shape_or_sequences"]["n"], 4)

    def test_a_gauge_keeps_its_last_value(self):
        self.assertEqual(ledger.difference(BEFORE, AFTER)["slots"]["cuda.mem.used_mib[0]"], {"g": 97107, "max": 97107})

    def test_a_slot_that_gained_nothing_is_left_out(self):
        self.assertNotIn("decode.total", ledger.difference(AFTER, AFTER)["slots"])

    def test_a_rare_large_value_shows_in_the_high_quantile_and_not_the_median(self):
        slot = ledger.difference(BEFORE, AFTER)["slots"]["decode.total"]
        self.assertEqual(ledger.quantile(slot, 0.5), 8192)
        self.assertEqual(ledger.quantile(slot, 0.999), 120_000)

    def test_a_step_is_a_server_iteration_where_the_ledger_has_them(self):
        diff = ledger.difference(BEFORE, AFTER)
        self.assertEqual(ledger.units(diff), ("server iteration", 100))
        self.assertEqual(ledger.summary(diff)["slots"]["decode.total"]["sum_per_step"], 10_000.0)

    def test_device_time_is_read_against_a_decodes_wall_time(self):
        out = io.StringIO()
        ledger.render(ledger.difference(BEFORE, AFTER), out)
        # 100,000 us over 25 timed decodes is 4,000 us a decode, of the 10,000 us a decode took: 40%
        self.assertIn("4000.0 us a decode, 40% of a decode's wall time", out.getvalue())

    def test_one_snapshot_reads_as_everything_since_the_process_began(self):
        diff = ledger.difference(None, AFTER)
        self.assertIsNone(diff["seconds"])
        self.assertEqual(diff["slots"]["decode.total"]["n"], 300)


if __name__ == "__main__":
    unittest.main()
