"""Screening spends two trials a knob and keeps the ones that move: the fake measures effects written in advance."""
import unittest

import search_screen


class Bench:
    """An evaluate over a fixed effect table: the baseline at a knob's low level, base*(1+effect) above it."""

    def __init__(self, baseline, effects, lows):
        self.baseline = baseline
        self.effects = effects
        self.lows = lows
        self.calls = []

    def __call__(self, name, value):
        self.calls.append((name, value))
        if value == self.lows[name]:
            return self.baseline
        return self.baseline * (1.0 + self.effects.get(name, 0.0))


class ScreenTest(unittest.TestCase):
    def test_keeps_movers_in_both_directions_and_drops_a_knob_that_does_nothing(self):
        space = {"alpha": ["off", "on"], "beta": ["off", "on"], "gamma": ["off", "on"]}
        evaluate = Bench(100.0, {"alpha": -0.10, "beta": 0.0, "gamma": 0.06},
                         {"alpha": "off", "beta": "off", "gamma": "off"})
        found = search_screen.screen(space, evaluate, baseline_rate=100.0)
        # A 10% loss is as much worth searching as a 6% gain; both are movers and the zero is not.
        self.assertEqual(found["survivors"], ["alpha", "gamma"])
        self.assertEqual(found["screened_out"], ["beta"])
        self.assertEqual(found["evaluations"], 6)
        self.assertEqual(set(evaluate.calls), {("alpha", "off"), ("alpha", "on"), ("beta", "off"),
                                               ("beta", "on"), ("gamma", "off"), ("gamma", "on")})

    def test_off_is_the_low_level_wherever_the_declaration_puts_it(self):
        space = {"flip": ["on", "off"]}
        evaluate = Bench(100.0, {"flip": 0.07}, {"flip": "off"})
        found = search_screen.screen(space, evaluate, baseline_rate=100.0)
        self.assertEqual(found["survivors"], ["flip"])
        self.assertEqual(evaluate.calls, [("flip", "off"), ("flip", "on")])

    def test_a_numeric_ladder_screens_from_its_first_value_to_its_last(self):
        space = {"depth": ["1", "3", "7"]}
        evaluate = Bench(100.0, {"depth": 0.10}, {"depth": "1"})
        found = search_screen.screen(space, evaluate, baseline_rate=100.0)
        self.assertEqual(found["survivors"], ["depth"])
        self.assertEqual(evaluate.calls, [("depth", "1"), ("depth", "7")])

    def test_zero_is_low_and_the_top_of_a_ladder_is_high(self):
        space = {"rows": ["0", "8", "64"]}
        evaluate = Bench(100.0, {"rows": 0.20}, {"rows": "0"})
        found = search_screen.screen(space, evaluate, baseline_rate=100.0)
        self.assertEqual(found["survivors"], ["rows"])
        self.assertEqual(evaluate.calls, [("rows", "0"), ("rows", "64")])

    def test_a_knob_without_contrast_is_screened_out_without_a_trial(self):
        space = {"fixed": ["3"], "off_only": ["off", "0"]}
        evaluate = Bench(100.0, {}, {"fixed": "3", "off_only": "off"})
        found = search_screen.screen(space, evaluate, baseline_rate=100.0)
        self.assertEqual(found["survivors"], [])
        self.assertEqual(found["screened_out"], ["fixed", "off_only"])
        self.assertEqual(found["evaluations"], 0)
        self.assertEqual(found["trace"], [])
        self.assertEqual(evaluate.calls, [])

    def test_a_swing_at_the_response_is_not_a_mover(self):
        # The gate is strict: exactly response is inside it, so the knob is screened out like the one below.
        space = {"edge": ["off", "on"], "inside": ["off", "on"], "outside": ["off", "on"]}
        evaluate = Bench(100.0, {"edge": 0.02, "inside": 0.019, "outside": 0.021},
                         {"edge": "off", "inside": "off", "outside": "off"})
        found = search_screen.screen(space, evaluate, baseline_rate=100.0, response=0.02)
        self.assertEqual(found["survivors"], ["outside"])
        self.assertEqual(found["screened_out"], ["edge", "inside"])

    def test_a_zero_baseline_compares_the_swing_to_the_response_itself(self):
        space = {"moves": ["off", "on"], "still": ["off", "on"]}
        rates = {("moves", "off"): 0.0, ("moves", "on"): 0.05, ("still", "off"): 0.0, ("still", "on"): 0.01}
        found = search_screen.screen(space, lambda name, value: rates[(name, value)],
                                     baseline_rate=0.0, response=0.02)
        self.assertEqual(found["survivors"], ["moves"])
        self.assertEqual(found["screened_out"], ["still"])

    def test_a_budget_that_cannot_afford_a_knob_leaves_it_a_survivor(self):
        space = {"alpha": ["off", "on"], "beta": ["off", "on"], "gamma": ["off", "on"]}
        evaluate = Bench(100.0, {"alpha": 0.0, "beta": -0.10, "gamma": 0.06},
                         {"alpha": "off", "beta": "off", "gamma": "off"})
        found = search_screen.screen(space, evaluate, baseline_rate=100.0, budget=2)
        self.assertEqual(found["evaluations"], 2)
        self.assertEqual(found["screened_out"], ["alpha"])
        self.assertEqual(found["survivors"], ["beta", "gamma"])
        self.assertEqual(found["trace"][-1]["reason"], "budget")
        self.assertEqual(found["trace"][-1]["unprobed"], [["beta"], ["gamma"]])

    def test_a_zero_budget_screens_nothing_and_drops_nothing(self):
        space = {"alpha": ["off", "on"], "beta": ["off", "on"], "gamma": ["off", "on"]}
        evaluate = Bench(100.0, {"alpha": -0.10, "beta": 0.0, "gamma": 0.06},
                         {"alpha": "off", "beta": "off", "gamma": "off"})
        found = search_screen.screen(space, evaluate, baseline_rate=100.0, budget=0)
        self.assertEqual(found["evaluations"], 0)
        self.assertEqual(found["survivors"], ["alpha", "beta", "gamma"])
        self.assertEqual(found["screened_out"], [])
        self.assertEqual(found["trace"][-1]["reason"], "budget")
        self.assertEqual(evaluate.calls, [])

    def test_an_odd_budget_spends_both_trials_of_earlier_knobs_only(self):
        space = {"alpha": ["off", "on"], "beta": ["off", "on"], "gamma": ["off", "on"]}
        evaluate = Bench(100.0, {"alpha": 0.0, "beta": 0.10, "gamma": 0.06},
                         {"alpha": "off", "beta": "off", "gamma": "off"})
        found = search_screen.screen(space, evaluate, baseline_rate=100.0, budget=5)
        # Five calls fit two knobs (four calls); the fifth cannot buy gamma's second level, so it stays unscreened.
        self.assertEqual(found["evaluations"], 4)
        self.assertEqual(found["screened_out"], ["alpha"])
        self.assertEqual(found["survivors"], ["beta", "gamma"])

    def test_a_small_space_is_measured_knob_by_knob_with_no_representative(self):
        # Twenty knobs is the exact-screen limit: every knob gets its own low and high reading, so a flat
        # neighbour cannot hide the one that moves.
        space = {f"knob_{index:02d}": ["off", "on"] for index in range(20)}
        effects = {name: (0.05 if name == "knob_07" else 0.0) for name in space}
        evaluate = Bench(100.0, effects, {name: "off" for name in space})
        found = search_screen.screen(space, evaluate, baseline_rate=100.0)
        self.assertEqual(len(found["trace"]), 1)
        self.assertEqual(len(found["trace"][0]["groups"]), 20)
        self.assertEqual(found["evaluations"], 40)
        self.assertEqual(found["survivors"], ["knob_07"])

    def test_a_moving_group_is_halved_and_its_zero_members_are_dropped(self):
        space = {f"grp_{index:02d}": ["off", "on"] for index in range(24)}
        effects = {name: (0.08 if name in ("grp_00", "grp_04", "grp_08") else 0.0) for name in space}
        evaluate = Bench(100.0, effects, {name: "off" for name in space})
        found = search_screen.screen(space, evaluate, baseline_rate=100.0)
        # Round 1 measures one representative a group of four; the three that move keep their groups, which are
        # halved. Round 2 measures each half's representative; the moving halves survive and the dead halves go.
        self.assertEqual(len(found["trace"]), 2)
        self.assertEqual(found["trace"][0]["groups"][0], ["grp_00", "grp_01", "grp_02", "grp_03"])
        self.assertEqual(found["trace"][1]["groups"][0], ["grp_00", "grp_01"])
        self.assertEqual(found["survivors"], ["grp_00", "grp_01", "grp_04", "grp_05", "grp_08", "grp_09"])
        for lost in ("grp_02", "grp_03", "grp_06", "grp_10", "grp_12", "grp_23"):
            self.assertIn(lost, found["screened_out"])
        self.assertEqual(found["trace"][-1]["reason"], "survivor_limit")

    def test_the_same_readings_give_the_same_trace(self):
        space = {f"grp_{index:02d}": ["off", "on"] for index in range(24)}
        effects = {name: (0.08 if name in ("grp_00", "grp_04", "grp_08") else 0.0) for name in space}
        lows = {name: "off" for name in space}
        one = search_screen.screen(space, Bench(100.0, effects, lows), baseline_rate=100.0)
        two = search_screen.screen(space, Bench(100.0, effects, lows), baseline_rate=100.0)
        self.assertEqual(one, two)
        self.assertEqual(one["survivors"], sorted(one["survivors"]))
        self.assertEqual(one["screened_out"], sorted(one["screened_out"]))

    def test_survivors_and_screened_out_partition_the_space(self):
        space = {f"grp_{index:02d}": ["off", "on"] for index in range(24)}
        effects = {name: (0.08 if name in ("grp_00", "grp_04", "grp_08") else 0.0) for name in space}
        found = search_screen.screen(space, Bench(100.0, effects, {name: "off" for name in space}),
                                     baseline_rate=100.0)
        self.assertEqual(sorted(found["survivors"] + found["screened_out"]), sorted(space))
        self.assertFalse(set(found["survivors"]) & set(found["screened_out"]))

    def test_a_nonfinite_trial_rate_is_an_error(self):
        space = {"alpha": ["off", "on"]}
        with self.assertRaises(ValueError):
            search_screen.screen(space, lambda name, value: float("nan"), baseline_rate=100.0)


class PlackettBurmanTest(unittest.TestCase):
    SUPPORTED = (3, 7, 11, 15, 19, 23, 31, 39, 47, 95)

    def test_each_supported_size_has_n_plus_one_runs_of_n_plus_minus_ones(self):
        for n in self.SUPPORTED:
            with self.subTest(n=n):
                design = search_screen.plackett_burman(n)
                self.assertEqual(len(design), n + 1)
                self.assertTrue(all(len(row) == n for row in design))
                self.assertTrue(all(value in (-1, 1) for row in design for value in row))

    def test_factor_columns_are_balanced_and_pairwise_orthogonal(self):
        for n in self.SUPPORTED:
            with self.subTest(n=n):
                design = search_screen.plackett_burman(n)
                columns = [tuple(row[column] for row in design) for column in range(n)]
                for column in columns:
                    self.assertEqual(sum(column), 0)
                for left in range(n):
                    for right in range(left + 1, n):
                        self.assertEqual(sum(a * b for a, b in zip(columns[left], columns[right])), 0)
                self.assertEqual([sum(value * value for value in column) for column in columns], [n + 1] * n)

    def test_the_baseline_run_is_all_ones(self):
        for n in self.SUPPORTED:
            with self.subTest(n=n):
                self.assertEqual(search_screen.plackett_burman(n)[0], [1] * n)

    def test_the_same_size_gives_the_same_design(self):
        self.assertEqual(search_screen.plackett_burman(11), search_screen.plackett_burman(11))

    def test_a_size_without_a_built_hadamard_order_is_refused(self):
        # 27 and 35 need Hadamard orders 28 and 36, which the other Paley construction builds and this one does
        # not; refusing is deliberate, not a guess at an invalid design.
        for n in (0, 1, 2, 4, 5, 6, 8, 9, 10, 27, 35):
            with self.subTest(n=n):
                with self.assertRaises(ValueError):
                    search_screen.plackett_burman(n)


if __name__ == "__main__":
    unittest.main()
