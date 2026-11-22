#!/usr/bin/env python3
"""The depth sweep around one trial: parsing --depths, keeping the worst depth, and one depth failing without the
trial's other depths. No server and no network: measure_one is a function in the test, returning records shaped like
search.py's measure() output."""
import unittest

import search_workload


def record(rate, **extra):
    """A trial record the shape search.py's measure() writes today: the rate and what a round and the ledger add."""
    return {"tokens_per_second": rate, "wall_tokens_per_second": rate * 2, "draft_accepted": 0.5,
            "tokens_sha256": "abc123", "slot": 0, "rounds": [rate - 1, rate, rate + 1], "prompt_tokens": 240,
            "ledger": {"seconds": 2.5, "steps": 3, "unit": "server iteration", "slots": {}}, **extra}


class DepthsFrom(unittest.TestCase):
    def test_an_empty_argument_is_the_request_alone(self):
        self.assertEqual(search_workload.depths_from(None), [0])
        self.assertEqual(search_workload.depths_from(""), [0])
        self.assertEqual(search_workload.depths_from("   "), [0])

    def test_one_depth(self):
        self.assertEqual(search_workload.depths_from("0"), [0])
        self.assertEqual(search_workload.depths_from("16384"), [16384])

    def test_depths_are_sorted(self):
        self.assertEqual(search_workload.depths_from("262144,0,16384"), [0, 16384, 262144])

    def test_a_depth_named_twice_is_measured_once(self):
        self.assertEqual(search_workload.depths_from("16384,0,16384,0"), [0, 16384])

    def test_whitespace_between_and_around_depths_is_kept_out(self):
        self.assertEqual(search_workload.depths_from(" 0 ,\t16384,\n 262144 "), [0, 16384, 262144])

    def test_a_garbage_entry_is_an_error_not_skipped(self):
        for spec in ("0,banana", "0,1.5", "0,0x10", "0,16384,"):  # the trailing comma leaves an empty entry
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                search_workload.depths_from(spec)

    def test_a_garbage_entry_says_which_it_is(self):
        with self.assertRaises(ValueError) as caught:
            search_workload.depths_from("0,banana,16384")
        self.assertIn("banana", str(caught.exception))

    def test_a_negative_depth_is_an_error(self):
        with self.assertRaises(ValueError):
            search_workload.depths_from("0,-16384")


class Aggregate(unittest.TestCase):
    def test_the_worst_depth_is_kept(self):
        out = search_workload.aggregate([record(100.0), record(60.0), record(80.0)])
        self.assertEqual(out["tokens_per_second"], 60.0)
        self.assertEqual(out["worst_depth"], 1)

    def test_per_depth_keeps_every_record_in_order(self):
        results = [record(100.0), record(60.0), record(80.0)]
        out = search_workload.aggregate(results)
        self.assertEqual([entry["tokens_per_second"] for entry in out["per_depth"]], [100.0, 60.0, 80.0])
        self.assertEqual([entry["depth_order"] for entry in out["per_depth"]], [0, 1, 2])

    def test_the_input_records_are_left_alone(self):
        results = [record(100.0), record(60.0)]
        search_workload.aggregate(results)
        for result in results:
            self.assertNotIn("depth_order", result)

    def test_a_tie_keeps_the_shallowest_depth(self):
        out = search_workload.aggregate([record(70.0), record(55.0), record(55.0)])
        self.assertEqual(out["worst_depth"], 1)

    def test_the_result_is_a_superset_of_todays_record(self):
        results = [record(100.0, prompt_tokens=200), record(60.0, prompt_tokens=300), record(80.0, prompt_tokens=400)]
        out = search_workload.aggregate(results)
        for key, value in results[1].items():
            self.assertEqual(out[key], value)

    def test_the_result_carries_the_worst_records_keys_not_anothers(self):
        results = [record(100.0, note="first"), record(60.0), record(80.0)]
        out = search_workload.aggregate(results)
        self.assertNotIn("note", out)
        self.assertEqual(out["rounds"], results[1]["rounds"])
        self.assertEqual(out["ledger"], results[1]["ledger"])

    def test_no_depths_is_an_error(self):
        with self.assertRaises(ValueError):
            search_workload.aggregate([])

    def test_a_record_without_a_rate_is_an_error(self):
        for result in ({"prompt_tokens": 10}, {"tokens_per_second": None}, {"tokens_per_second": True},
                       {"tokens_per_second": float("nan")}):
            with self.subTest(result=result), self.assertRaises(ValueError):
                search_workload.aggregate([record(100.0), result])


class RunDepths(unittest.TestCase):
    def test_every_depth_is_measured_in_order_and_the_worst_kept(self):
        calls = []
        rates = {0: 100.0, 16384: 55.0, 262144: 70.0}

        def measure_one(depth):
            calls.append(depth)
            return record(rates[depth], ledger={"depth": depth})

        out = search_workload.run_depths(measure_one, [0, 16384, 262144])
        self.assertEqual(calls, [0, 16384, 262144])
        self.assertEqual(out["tokens_per_second"], 55.0)
        self.assertEqual(out["worst_depth"], 1)
        self.assertEqual(out["depth"], 16384)
        self.assertEqual(out["ledger"], {"depth": 16384})
        self.assertEqual(out["failed_depths"], [])
        self.assertEqual([entry["depth"] for entry in out["per_depth"]], [0, 16384, 262144])
        self.assertEqual([entry["depth_order"] for entry in out["per_depth"]], [0, 1, 2])

    def test_one_depth_failing_costs_only_that_depth(self):
        def measure_one(depth):
            if depth == 16384:
                raise RuntimeError("the server left")
            return record(100.0 - depth // 4096)

        out = search_workload.run_depths(measure_one, [0, 16384, 262144])
        self.assertEqual(out["failed_depths"], [{"depth": 16384, "error": "the server left"}])
        self.assertEqual([entry["depth"] for entry in out["per_depth"]], [0, 262144])
        self.assertEqual(out["tokens_per_second"], 100.0 - 262144 // 4096)
        self.assertEqual(out["depth"], 262144)
        self.assertEqual(out["worst_depth"], 1)

    def test_a_depth_without_a_rate_keeps_its_own_error(self):
        def measure_one(depth):
            if depth:
                return {"tokens_per_second": None, "error": "the server exited 1"}
            return record(90.0)

        out = search_workload.run_depths(measure_one, [0, 16384])
        self.assertEqual(out["tokens_per_second"], 90.0)
        self.assertEqual(out["failed_depths"], [{"depth": 16384, "error": "the server exited 1"}])

    def test_a_depth_that_gives_no_record_is_a_failure(self):
        out = search_workload.run_depths(lambda depth: None, [0])
        self.assertIsNone(out["tokens_per_second"])
        self.assertEqual(len(out["failed_depths"]), 1)
        self.assertEqual(out["failed_depths"][0]["depth"], 0)

    def test_a_trial_where_every_depth_fails_keeps_its_error(self):
        def measure_one(depth):
            raise OSError(f"port {depth} is taken")

        out = search_workload.run_depths(measure_one, [0, 16384])
        self.assertIsNone(out["tokens_per_second"])
        self.assertIsNone(out["worst_depth"])
        self.assertEqual(out["per_depth"], [])
        self.assertEqual(len(out["failed_depths"]), 2)
        self.assertIn("16384", out["error"])

    def test_one_depth_is_todays_record(self):
        today = record(123.5)
        out = search_workload.run_depths(lambda depth: today, [0])
        for key, value in today.items():
            self.assertEqual(out[key], value)
        self.assertEqual(out["depth"], 0)
        self.assertEqual(out["worst_depth"], 0)
        self.assertEqual(out["failed_depths"], [])

    def test_no_depths_is_an_error(self):
        with self.assertRaises(ValueError):
            search_workload.run_depths(lambda depth: record(1.0), [])


if __name__ == "__main__":
    unittest.main()
