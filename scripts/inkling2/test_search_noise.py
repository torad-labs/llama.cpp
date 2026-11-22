"""Noise is part of the measurement: a trial earns repeats, its minimum is its rate, and disagreement is named."""
import statistics
import unittest

import search_noise


class Script:
    """A measure_one whose rates are written in advance, counting the calls it served and repeating its last rate."""

    def __init__(self, *rates):
        self.rates = list(rates)
        self.calls = 0

    def __call__(self):
        rate = self.rates[min(self.calls, len(self.rates) - 1)]
        self.calls += 1
        return {"tokens_per_second": rate, "error": None, "rounds": []}


class AggregateTest(unittest.TestCase):
    def test_the_minimum_not_the_median_nor_the_mean(self):
        rates = [12.0, 10.0, 11.0]
        self.assertEqual(search_noise.aggregate(rates), 10.0)
        self.assertEqual(statistics.median(rates), 11.0)
        self.assertEqual(statistics.mean(rates), 11.0)

    def test_repeats_that_agree(self):
        self.assertEqual(search_noise.aggregate([7.0, 7.0, 7.0]), 7.0)

    def test_no_repeats_is_an_error(self):
        with self.assertRaises(ValueError):
            search_noise.aggregate([])


class UnstableTest(unittest.TestCase):
    def test_zero_variance_is_stable(self):
        self.assertFalse(search_noise.unstable([100.0, 100.0, 100.0]))

    def test_a_single_draw_is_stable(self):
        self.assertFalse(search_noise.unstable([123.4]))

    def test_five_percent_spread_is_stable(self):
        self.assertFalse(search_noise.unstable([97.5, 100.0, 102.5]))

    def test_twelve_percent_spread_is_unstable(self):
        self.assertTrue(search_noise.unstable([94.0, 100.0, 106.0]))

    def test_the_threshold_itself_is_stable(self):
        self.assertFalse(search_noise.unstable([95.0, 100.0, 105.0], relative_spread=0.10))
        self.assertTrue(search_noise.unstable([95.0, 100.0, 105.0], relative_spread=0.099))

    def test_a_custom_spread_admits_what_the_default_rejects(self):
        self.assertFalse(search_noise.unstable([94.0, 100.0, 106.0], relative_spread=0.15))

    def test_all_zero_is_stable(self):
        self.assertFalse(search_noise.unstable([0.0, 0.0, 0.0]))

    def test_a_zero_median_with_a_spread_is_unstable(self):
        self.assertTrue(search_noise.unstable([0.0, 0.0, 5.0]))

    def test_no_repeats_is_an_error(self):
        with self.assertRaises(ValueError):
            search_noise.unstable([])


class ConfirmTest(unittest.TestCase):
    def test_a_quiet_trial_is_measured_exactly_repeats_times(self):
        script = Script(50.0, 50.0, 50.0, 50.0)
        found = search_noise.confirm(script, repeats=3)
        self.assertEqual(found, {"tokens_per_second": 50.0, "rates": [50.0, 50.0, 50.0], "stable": True, "repeats": 3})
        self.assertEqual(script.calls, 3)

    def test_the_minimum_is_the_trials_rate(self):
        script = Script(12.0, 10.0, 11.0)
        found = search_noise.confirm(script, repeats=3)
        self.assertEqual(found["tokens_per_second"], 10.0)
        self.assertEqual(found["rates"], [12.0, 10.0, 11.0])
        # Spread over the median is 2/11, over 10%, so the trial is named unstable; the default max_repeats is
        # repeats, so no extra draw is spent and the caller is told the rate rests on disagreement.
        self.assertFalse(found["stable"])
        self.assertEqual(found["repeats"], 3)

    def test_an_unstable_trial_draws_on_until_it_settles(self):
        # After three draws the median sits on the low draw, a 10.5% spread; a fourth lifts it to 105.25 and 9.98%.
        script = Script(100.0, 110.5, 100.0, 110.5)
        found = search_noise.confirm(script, repeats=3, max_repeats=4)
        self.assertTrue(found["stable"])
        self.assertEqual(found["repeats"], 4)
        self.assertEqual(found["tokens_per_second"], 100.0)
        self.assertEqual(script.calls, 4)

    def test_a_trial_that_never_settles_stops_at_max_repeats(self):
        script = Script(100.0, 120.0, 118.0)
        found = search_noise.confirm(script, repeats=3, max_repeats=6)
        self.assertFalse(found["stable"])
        self.assertEqual(found["repeats"], 6)
        self.assertEqual(found["tokens_per_second"], 100.0)
        self.assertEqual(script.calls, 6)

    def test_without_max_repeats_an_unstable_trial_spends_no_extra_draw(self):
        script = Script(100.0, 120.0, 118.0)
        found = search_noise.confirm(script, repeats=2)
        self.assertFalse(found["stable"])
        self.assertEqual(found["repeats"], 2)
        self.assertEqual(script.calls, 2)

    def test_the_returned_record_names_exactly_the_contract(self):
        found = search_noise.confirm(Script(9.0), repeats=1)
        self.assertEqual(set(found), {"tokens_per_second", "rates", "stable", "repeats"})

    def test_a_record_without_a_rate_is_an_error(self):
        with self.assertRaises(ValueError):
            search_noise.confirm(Script(None), repeats=2)

    def test_repeats_must_be_at_least_one(self):
        with self.assertRaises(ValueError):
            search_noise.confirm(Script(9.0), repeats=0)

    def test_max_repeats_cannot_be_under_repeats(self):
        with self.assertRaises(ValueError):
            search_noise.confirm(Script(9.0), repeats=3, max_repeats=2)


class NeedsConfirmationTest(unittest.TestCase):
    def test_a_candidate_inside_the_gain_earns_nothing(self):
        self.assertFalse(search_noise.needs_confirmation(101.9, 100.0, 0.02))
        self.assertFalse(search_noise.needs_confirmation(100.0, 100.0, 0.02))
        self.assertFalse(search_noise.needs_confirmation(90.0, 100.0, 0.02))

    def test_a_candidate_beyond_the_gain_earns_repeats(self):
        self.assertTrue(search_noise.needs_confirmation(102.5, 100.0, 0.02))

    def test_exactly_at_the_gain_boundary_earns_nothing(self):
        # 128 * 1.125 is 144 exactly in binary, so this is the boundary and not a float accident.
        self.assertFalse(search_noise.needs_confirmation(144.0, 128.0, 0.125))
        self.assertTrue(search_noise.needs_confirmation(144.0001, 128.0, 0.125))

    def test_a_zero_gain_still_needs_a_strictly_higher_rate(self):
        self.assertFalse(search_noise.needs_confirmation(100.0, 100.0, 0.0))
        self.assertTrue(search_noise.needs_confirmation(100.1, 100.0, 0.0))

    def test_a_cold_start_earns_repeats_on_any_positive_rate(self):
        self.assertTrue(search_noise.needs_confirmation(1.0, 0.0, 0.02))
        self.assertFalse(search_noise.needs_confirmation(0.0, 0.0, 0.02))


if __name__ == "__main__":
    unittest.main()
