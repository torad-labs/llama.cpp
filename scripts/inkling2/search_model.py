#!/usr/bin/env python3
"""Propose the next Inkling 2 settings from the trials a search has run.

search.py walks one knob at a time. Two knobs that only pay off when they move together are found only if the walk
happens to stand on that pair, and a combination that kills the launch costs a full server start to learn one bit.
This module is the model that sees all the knobs at once: every value is projected onto [0, 1] (LlamaTune's
low-dimensional encoding), a Gaussian process with a squared-exponential kernel is fitted over the trials, and the
next settings are the points of highest expected improvement (SCOOT, CGPTuner). A launch that crashed enters the fit
as a penalized rate, so the acquisition learns to steer around the combinations that do not start.

encode() is the projection: a knob's values take index/(n-1) along its axis in the order its space lists them, so two
values always encode to 0 and 1 (flags included: off/on, 0/1) and decode() is the way back. Model.fit() only keeps the
trials; the encoding depends on the space, which propose() receives, so a space that changes between passes refits
without the caller doing more.

Cost: the fit picks a length scale and noise level over a small grid of Cholesky factorizations, O(n^3) each at n
trials; a proposal evaluates expected improvement over a few coordinate-ascent starts, O(n^2) a point, and at the tens
of trials a search runs it is milliseconds. Nothing here leaves the process, touches a GPU or reads the network.
"""
import itertools
import math
import random
import statistics

__all__ = ["Model", "encode", "decode", "CRASH_PENALTY_TOKENS_PER_SECOND"]

# A crashed launch (or one that exited before reporting a rate) is charged this rate. It sits below every rate a live
# server reports, so the process models the region as bad instead of unknown and expected improvement turns away from
# it. The value is a real observation, not a gap: a region tried and lost is known, where an untried one is not.
CRASH_PENALTY_TOKENS_PER_SECOND = 0.0

# Expected improvement must beat the best trial by this much, in units of the trials' spread, to count: it keeps a
# proposal from being a near-copy of a trial already run.
IMPROVEMENT_MARGIN = 0.01

# The kernel's length scale and noise level are read off the marginal likelihood over this grid, in encoded units.
LENGTH_SCALES = (0.15, 0.25, 0.4, 0.65, 1.0)
NOISE_LEVELS = (0.01, 0.05, 0.15)
JITTER = 1e-8

# Coordinate-ascent starts and knob sweeps per start. A start is the best trial's point or a seeded random point; the
# ascent climbs to the nearest lattice maximum of expected improvement. At the search's tens of trials this is cheap.
RESTARTS = 32
SWEEPS = 3

# A space no larger than this is enumerated when proposals must be filled, which keeps small synthetic spaces exact.
_ENUMERATE_LIMIT = 4096


def encode(space, settings):
    """Project one settings dict onto the unit cube: knob i by its value's index/(len-1), in the space's key order.

    Two-valued knobs (off/on, 0/1, any flag) therefore land on exactly 0.0 and 1.0, whichever order the space lists.
    A knob missing from settings takes its first listed value: search.py's best dict holds only the knobs a pass moved,
    so a caller that records trials should fill every knob that ran; otherwise the first listed value stands in for it.
    Knobs in settings that the space does not list are ignored, so a space narrowed between passes still encodes old
    trials. A value the space does not list raises ValueError, because that is a caller's mistake, not a default.
    """
    for name, values in space.items():
        if not values:
            raise ValueError(f"the space gives no values for {name}")
    point = []
    for name, values in space.items():
        value = settings.get(name, values[0])
        try:
            index = values.index(value)
        except ValueError:
            raise ValueError(f"{name}={value!r} is not one of {values}") from None
        point.append(index / (len(values) - 1) if len(values) > 1 else 0.0)
    return point


def decode(space, point):
    """The knob values nearest to a point in the cube, one per knob, in the space's key order.

    Nearest means the rounded index along each axis, so decode(encode(space, settings)) returns settings whenever the
    values are the space's own, and any other point in the cube maps to some settings the space allows.
    """
    if len(point) != len(space):
        raise ValueError(f"a point of {len(point)} dimensions cannot hold {len(space)} knobs")
    settings = {}
    for (name, values), x in zip(space.items(), point):
        index = 0 if len(values) == 1 else int(round(x * (len(values) - 1)))
        settings[name] = values[min(max(index, 0), len(values) - 1)]
    return settings


def _point_of(space, settings):
    """The unit-cube point, lattice indices and value tuple of one trial, or Nones when a value is outside the space.

    The index tuple is the proposal lattice's name for the point; the value tuple names the combination, for the
    random path's tried set.
    """
    point, indices, choice = [], [], []
    for name, values in space.items():
        value = settings.get(name, values[0])
        if value not in values:
            return None, None, None
        index = values.index(value)
        point.append(index / (len(values) - 1) if len(values) > 1 else 0.0)
        indices.append(index)
        choice.append(value)
    return point, tuple(indices), tuple(choice)


def _cholesky(matrix):
    """The lower factor L of a symmetric positive-definite matrix, or None where the matrix is not one."""
    size = len(matrix)
    lower = [[0.0] * size for _ in range(size)]
    for row in range(size):
        for column in range(row + 1):
            total = matrix[row][column] - sum(lower[row][k] * lower[column][k] for k in range(column))
            if row == column:
                if total <= 0.0:
                    return None
                lower[row][column] = math.sqrt(total)
            else:
                lower[row][column] = total / lower[column][column]
    return lower


def _forward(lower, vector):
    """Solve L y = vector for lower-triangular L."""
    size = len(lower)
    result = [0.0] * size
    for row in range(size):
        result[row] = (vector[row] - sum(lower[row][k] * result[k] for k in range(row))) / lower[row][row]
    return result


def _cho_solve(lower, vector):
    """Solve (L L^T) x = vector for lower-triangular L."""
    intermediate = _forward(lower, vector)
    size = len(lower)
    result = [0.0] * size
    for row in range(size - 1, -1, -1):
        result[row] = (intermediate[row] - sum(lower[k][row] * result[k] for k in range(row + 1, size))) / lower[row][row]
    return result


class _Surrogate:
    """The Gaussian process over one space: encoded trials, a tuned squared-exponential kernel, expected improvement."""

    def __init__(self, points, rates):
        self.points = points
        self.mean = statistics.fmean(rates)
        spread = statistics.pstdev(rates)
        # A spread of zero (every trial equal, or every trial crashed) leaves the process uninformative; a scale of 1
        # keeps the arithmetic finite and the acquisition falls back on its variance term.
        self.scale = spread if spread > 1e-12 else 1.0
        self.rewards = [(rate - self.mean) / self.scale for rate in rates]
        self.best = max(self.rewards)
        self.length_scale, self.noise, self.lower, self.alpha = self._tune(points, self.rewards)

    @staticmethod
    def _kernel(left, right, length_scale):
        squared = 0.0
        for a, b in zip(left, right):
            difference = a - b
            squared += difference * difference
        return math.exp(-squared / (2.0 * length_scale * length_scale))

    def _gram(self, points, length_scale, noise):
        size = len(points)
        gram = [[0.0] * size for _ in range(size)]
        for row in range(size):
            for column in range(row, size):
                value = self._kernel(points[row], points[column], length_scale)
                gram[row][column] = value
                gram[column][row] = value
        for row in range(size):
            gram[row][row] += noise * noise + JITTER
        return gram

    def _tune(self, points, rewards):
        """Pick the length scale and noise of largest marginal likelihood; ties go to the first in the grid order."""
        chosen = None
        for length_scale in LENGTH_SCALES:
            for noise in NOISE_LEVELS:
                lower = _cholesky(self._gram(points, length_scale, noise))
                if lower is None:
                    continue
                alpha = _cho_solve(lower, rewards)
                value = (-0.5 * sum(a * b for a, b in zip(rewards, alpha))
                         - sum(math.log(lower[i][i]) for i in range(len(lower))))
                if chosen is None or value > chosen[0] + 1e-12:
                    chosen = (value, length_scale, noise, lower, alpha)
        if chosen is None:
            # The kernel would not factor even at the largest grid noise; a wider prior always will, the squared
            # exponential's Gram matrix being positive semi-definite.
            length_scale, noise = LENGTH_SCALES[-1], NOISE_LEVELS[-1] * 10.0
            lower = _cholesky(self._gram(points, length_scale, noise))
            return length_scale, noise, lower, _cho_solve(lower, rewards)
        _, length_scale, noise, lower, alpha = chosen
        return length_scale, noise, lower, alpha

    def expected_improvement(self, point):
        """E[max(0, reward(x) - best - margin)] under the process posterior at x, in normalized reward units."""
        kernel = [self._kernel(point, known, self.length_scale) for known in self.points]
        mean = sum(a * b for a, b in zip(kernel, self.alpha))
        weighted = _forward(self.lower, kernel)
        variance = 1.0 - sum(a * a for a in weighted)
        improvement = mean - self.best - IMPROVEMENT_MARGIN
        if variance <= 1e-12:
            return max(0.0, improvement)
        deviation = math.sqrt(variance)
        z = improvement / deviation
        cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        pdf = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
        return improvement * cdf + deviation * pdf


class Model:
    """Proposes settings from completed trials: fit(trials), then propose(space, n, seed).

    Each trial is search.py's record: {"settings": dict[str, str], "tokens_per_second": float}, or a launch that did
    not start, marked {"crash": True} or carrying tokens_per_second None (the shape run_trial writes with an error
    string). Both enter the fit at CRASH_PENALTY_TOKENS_PER_SECOND, and so does a rate that is unreadable or not
    finite: such a record says nothing usable about its configuration. Trials whose settings hold a value the space
    does not list are skipped, since they cannot be placed in the cube; unknown knobs are ignored.

    propose() is deterministic for a fixed seed and the same trials and space. With fewer than three usable trials
    there is nothing for a process to see, so it proposes random distinct untried settings; in every path it returns
    at most one copy of a combination and prefers combinations not yet run.
    """

    def __init__(self):
        self.trials = []
        self._space = None
        self._surrogate = None
        self._seen = set()   # index tuples of usable trials, so a proposal repeats none of them
        self._tried = set()  # value tuples of usable trials, for the random path's untried preference
        self._best = []      # index tuples of the best trials, where coordinate ascent starts

    def fit(self, trials):
        """Take the completed trials; the process itself is built on the first propose() for a space."""
        self.trials = list(trials)
        self._space = None

    def _prepare(self, space):
        key = tuple((name, tuple(values)) for name, values in space.items())
        if key == self._space:
            return
        points, rates = [], []
        self._seen, self._tried = set(), set()
        for trial in self.trials:
            point, indices, choice = _point_of(space, trial.get("settings") or {})
            if point is None:
                continue
            rate = trial.get("tokens_per_second")
            try:
                rate = None if rate is None else float(rate)
            except (TypeError, ValueError):
                rate = None
            if trial.get("crash") is True or rate is None or not math.isfinite(rate):
                rate = CRASH_PENALTY_TOKENS_PER_SECOND
            points.append(point)
            rates.append(rate)
            self._seen.add(indices)
            self._tried.add(choice)
        self._space = key
        self._surrogate = _Surrogate(points, rates) if len(points) >= 3 else None
        order = sorted(range(len(rates)), key=lambda i: (-rates[i], points[i]))
        self._best = [_indices(points[i], space) for i in order[:3]]

    def propose(self, space, n=1, seed=0):
        """The n settings to try next from space: highest expected improvement, deterministic for a seed.

        Every returned dict carries every knob of the space, in the space's order, with one of its listed values. n is
        capped by the number of combinations the space holds. The fit and the acquisition are cached per space, so
        repeated calls with the same space and trials cost only the ascent.
        """
        if n <= 0:
            return []
        for name, values in space.items():
            if not values:
                raise ValueError(f"the space gives no values for {name}")
        if not space:
            return [{} for _ in range(n)]
        self._prepare(space)
        keys = list(space)
        sizes = [len(values) for values in space.values()]
        total = math.prod(sizes)
        count = min(n, total)
        generator = random.Random(seed)
        if self._surrogate is None:
            return self._random(space, sizes, count, generator)

        def point_of(indices):
            return [index / (size - 1) if size > 1 else 0.0 for index, size in zip(indices, sizes)]

        def settings_of(indices):
            return {name: space[name][index] for name, index in zip(keys, indices)}

        acquisition = {}

        def expected(indices):
            if indices not in acquisition:
                acquisition[indices] = self._surrogate.expected_improvement(point_of(indices))
            return acquisition[indices]

        starts = [tuple(indices) for indices in self._best]
        while len(starts) < RESTARTS:
            starts.append(tuple(generator.randrange(size) for size in sizes))
        for start in dict.fromkeys(starts):
            current, current_value = start, expected(start)
            for _ in range(SWEEPS):
                improved = False
                for axis, size in enumerate(sizes):
                    choice, choice_value = None, current_value
                    moved = list(current)
                    for index in range(size):
                        if index == current[axis]:
                            continue
                        moved[axis] = index
                        value = expected(tuple(moved))
                        if value > choice_value + 1e-12:
                            choice, choice_value = index, value
                    if choice is not None:
                        moved = list(current)
                        moved[axis] = choice
                        current, current_value = tuple(moved), choice_value
                        improved = True
                if not improved:
                    break

        ranked = sorted(acquisition.items(), key=lambda item: (-item[1], item[0]))
        chosen, proposals = set(), []
        for indices, _ in ranked:
            if len(proposals) == count:
                break
            if indices in self._seen or indices in chosen:
                continue
            proposals.append(settings_of(indices))
            chosen.add(indices)
        if len(proposals) < count:
            taken = {tuple(space[name][index] for name, index in indices) for indices in chosen} | self._tried
            proposals.extend(self._random(space, sizes, count - len(proposals), generator, taken))
        return proposals

    @staticmethod
    def _random(space, sizes, count, generator, exclude=frozenset()):
        """count distinct knob-value tuples, uniform over the product, preferring ones not in exclude."""
        if count <= 0:
            return []
        keys = list(space)
        proposals, chosen = [], set()

        def take(choice):
            proposals.append(dict(zip(keys, choice)))
            chosen.add(choice)

        if math.prod(sizes) <= _ENUMERATE_LIMIT:
            everything = [tuple(combo) for combo in itertools.product(*(space[key] for key in keys))]
            generator.shuffle(everything)
            for choice in everything:
                if len(proposals) == count:
                    return proposals
                if choice not in exclude and choice not in chosen:
                    take(choice)
            # Every combination has been tried. The space cannot give untried ones, so repeat tried ones, still
            # distinct from each other; with count capped by the space's size a combination is always left to take.
            for choice in everything:
                if len(proposals) == count:
                    return proposals
                if choice not in chosen:
                    take(choice)
            return proposals

        attempts, limit = 0, 200 * count + 2000
        while len(proposals) < count and attempts < limit:
            attempts += 1
            choice = tuple(generator.choice(space[key]) for key in keys)
            if choice in exclude or choice in chosen:
                continue
            take(choice)
        while len(proposals) < count:  # the untried part of the cube is nearly exhausted: repeat tried combinations
            choice = tuple(generator.choice(space[key]) for key in keys)
            if choice not in chosen:
                take(choice)
        return proposals


def _indices(point, space):
    """The lattice indices of an encoded point, the inverse of the encoding along each axis."""
    indices = []
    for (_, values), x in zip(space.items(), point):
        index = 0 if len(values) == 1 else int(round(x * (len(values) - 1)))
        indices.append(min(max(index, 0), len(values) - 1))
    return tuple(indices)
