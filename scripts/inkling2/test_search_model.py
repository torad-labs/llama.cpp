"""The model proposes settings near an interacting optimum, steers around crashes, and repeats itself for a seed."""
import itertools
import random
import statistics
import unittest

from search_model import CRASH_PENALTY_TOKENS_PER_SECOND, Model, decode, encode

# A small space of the real one's kinds: ordinal knobs of different lengths, a flag, a non-uniform knob, a single value.
ROUNDTRIP_SPACE = {
    "--draft-depth": ["1", "2", "3", "4"],
    "--draft": ["off", "on"],
    "GGML_TENSOR_N_SPLIT_ROWS": ["0", "8", "64"],
    "ONLY": ["one"],
}

# Two knobs that must agree: raising one alone loses more than it gains, so a walk over one knob at a time stalls at
# (0, 0) while the optimum sits at (4, 4). c and d pay a little each way; the interaction decides the best region.
SEARCH_SPACE = {
    "a": ["0", "1", "2", "3", "4"],
    "b": ["0", "1", "2", "3", "4"],
    "c": ["off", "on"],
    "d": ["1", "2", "3", "4", "5", "6"],
}
OPTIMUM = {"a": "4", "b": "4", "c": "on", "d": "6"}
TRIAL_COUNT = 20
TRIAL_SEED = 11
RANDOM_BASELINE_SEED = 99

# A launch that cannot start on rows 0 or 1, whatever its threads: the region the crash penalty must teach the model.
CRASH_SPACE = {"--rows": ["0", "1", "2", "3", "4", "5", "6"], "--threads": ["1", "2", "3", "4"]}
CRASH_ROWS = ("0", "1")


def rate(settings):
    """The synthetic objective, best when a and b agree and are high, c on, d long."""
    a, b = int(settings["a"]), int(settings["b"])
    return 1000 + 120 * a + 120 * b - 150 * abs(a - b) + 80 * (settings["c"] == "on") + 30 * int(settings["d"])


def distance(settings):
    """Index distance from the optimum, summed over the knobs."""
    return (abs(int(settings["a"]) - 4) + abs(int(settings["b"]) - 4)
            + (settings["c"] != "on") + abs(int(settings["d"]) - 6))


def combinations(space):
    keys = list(space)
    return [dict(zip(keys, values)) for values in itertools.product(*(space[key] for key in keys))]


def synthetic_trials(count=TRIAL_COUNT, seed=TRIAL_SEED):
    generator = random.Random(seed)
    return [{"settings": settings, "tokens_per_second": rate(settings)}
            for settings in generator.sample(combinations(SEARCH_SPACE), count)]


def crash_rate(settings):
    return 2000 + 60 * int(settings["--rows"]) + 25 * int(settings["--threads"])


class Encoding(unittest.TestCase):
    def test_roundtrip_for_every_value_of_every_knob(self):
        for values in itertools.product(*ROUNDTRIP_SPACE.values()):
            settings = dict(zip(ROUNDTRIP_SPACE, values))
            point = encode(ROUNDTRIP_SPACE, settings)
            self.assertEqual(len(point), len(ROUNDTRIP_SPACE))
            self.assertTrue(all(0.0 <= x <= 1.0 for x in point), point)
            self.assertEqual(decode(ROUNDTRIP_SPACE, point), settings)

    def test_ordinal_position_and_flag_ends(self):
        lowest = {"--draft-depth": "1", "--draft": "off", "GGML_TENSOR_N_SPLIT_ROWS": "0", "ONLY": "one"}
        highest = {"--draft-depth": "4", "--draft": "on", "GGML_TENSOR_N_SPLIT_ROWS": "64", "ONLY": "one"}
        middle = {"--draft-depth": "3", "--draft": "on", "GGML_TENSOR_N_SPLIT_ROWS": "8", "ONLY": "one"}
        self.assertEqual(encode(ROUNDTRIP_SPACE, lowest), [0.0, 0.0, 0.0, 0.0])
        self.assertEqual(encode(ROUNDTRIP_SPACE, highest), [1.0, 1.0, 1.0, 0.0])
        self.assertEqual(encode(ROUNDTRIP_SPACE, middle), [2 / 3, 1.0, 0.5, 0.0])

    def test_missing_knob_takes_its_first_value_and_unknown_knobs_are_ignored(self):
        self.assertEqual(encode(ROUNDTRIP_SPACE, {"--draft-depth": "2", "IGNORED": "x"}),
                         [1 / 3, 0.0, 0.0, 0.0])

    def test_value_outside_the_space_raises(self):
        with self.assertRaises(ValueError):
            encode(ROUNDTRIP_SPACE, {"--draft-depth": "9"})


class Proposal(unittest.TestCase):
    def test_fixed_seed_is_deterministic_and_settings_stay_in_the_space(self):
        model = Model()
        model.fit(synthetic_trials())
        first = model.propose(SEARCH_SPACE, n=3, seed=5)
        second = model.propose(SEARCH_SPACE, n=3, seed=5)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        self.assertEqual(len({tuple(sorted(settings.items())) for settings in first}), 3)
        for settings in first:
            self.assertEqual(list(settings), list(SEARCH_SPACE))
            for name, value in settings.items():
                self.assertIn(value, SEARCH_SPACE[name])

    def test_proposals_concentrate_near_the_interacting_optimum(self):
        proposals = []
        for seed in range(5):
            model = Model()
            model.fit(synthetic_trials())
            proposals.extend(model.propose(SEARCH_SPACE, n=3, seed=seed))
        distances = [distance(settings) for settings in proposals]
        generator = random.Random(RANDOM_BASELINE_SEED)
        random_distances = [distance({name: generator.choice(values) for name, values in SEARCH_SPACE.items()})
                            for _ in proposals]
        # A model that saw the interaction lands near (4, 4); random guessing averages about the space's middle.
        self.assertLess(statistics.mean(distances), statistics.mean(random_distances) / 2)
        self.assertLessEqual(statistics.median(distances), 2)
        self.assertLessEqual(min(distances), 1)
        self.assertGreater(statistics.mean([1.0 if settings["a"] == settings["b"] else 0.0 for settings in proposals]),
                           0.5)

    def test_fewer_than_three_trials_gives_random_distinct_proposals(self):
        model = Model()
        model.fit(synthetic_trials(count=2))
        first = model.propose(SEARCH_SPACE, n=5, seed=3)
        self.assertEqual(len(first), 5)
        self.assertEqual(len({tuple(sorted(settings.items())) for settings in first}), 5)
        self.assertEqual(first, model.propose(SEARCH_SPACE, n=5, seed=3))

    def test_unfitted_model_still_proposes(self):
        proposals = Model().propose(SEARCH_SPACE, n=2, seed=0)
        self.assertEqual(len(proposals), 2)
        self.assertEqual(len({tuple(sorted(settings.items())) for settings in proposals}), 2)

    def test_n_is_capped_by_the_space_and_zero_asks_for_nothing(self):
        space = {"x": ["0", "1"]}
        self.assertEqual(len(Model().propose(space, n=5, seed=0)), 2)
        self.assertEqual(Model().propose(space, n=0), [])


class RankedFill(unittest.TestCase):
    """The ranked path cannot always fill the request: few unseen lattice points remain, and the rest of the
    proposals come from the random path. That branch crashed on unpacking an index tuple (found by the Eli review)."""

    def test_a_nearly_walked_space_still_proposes_without_crashing(self):
        space = {"a": ["0", "1", "2", "3"], "b": ["x"]}
        trials = [{"settings": {"a": value, "b": "x"}, "tokens_per_second": float(10 - index)}
                  for index, value in enumerate(["0", "1", "2"])]
        model = Model()
        model.fit(trials)
        proposals = model.propose(space, n=2, seed=0)      # only one unseen point: the random path must fill the rest
        self.assertEqual(len(proposals), 2)
        values = [proposal["a"] for proposal in proposals]
        self.assertIn("3", values)                          # the unseen point is preferred over any seen one
        self.assertEqual(len(set(values)), len(values))     # and no combination repeats


class CrashPenalty(unittest.TestCase):
    def test_crashed_region_is_steered_around(self):
        everything = combinations(CRASH_SPACE)
        generator = random.Random(7)
        crashed = [settings for settings in everything if settings["--rows"] in CRASH_ROWS]
        live = generator.sample([settings for settings in everything if settings["--rows"] not in CRASH_ROWS], 10)
        trials = []
        for index, settings in enumerate(crashed):
            # Every record shape counts: the flag this module documents, the None rate search.py actually writes, and
            # a rate that is not finite, which says nothing usable either.
            if index % 3 == 1:
                trials.append({"settings": settings, "tokens_per_second": None, "error": "the server exited 1"})
            elif index % 3 == 2:
                trials.append({"settings": settings, "tokens_per_second": float("nan")})
            else:
                trials.append({"settings": settings, "crash": True})
        trials += [{"settings": settings, "tokens_per_second": crash_rate(settings)} for settings in live]
        self.assertLess(CRASH_PENALTY_TOKENS_PER_SECOND, min(crash_rate(settings) for settings in live))
        for seed in range(4):
            model = Model()
            model.fit(trials)
            proposals = model.propose(CRASH_SPACE, n=5, seed=seed)
            self.assertEqual(len(proposals), 5)
            for settings in proposals:
                self.assertNotIn(settings["--rows"], CRASH_ROWS)


if __name__ == "__main__":
    unittest.main()
