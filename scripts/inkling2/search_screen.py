#!/usr/bin/env python3
"""Screen the launch space before the descent: two trials a knob, and only the knobs that move the rate go on.

search.py today walks every value of every setting and pays a server launch for each one. Sequential bifurcation
(Kleijnen, Bettonvil & van Groenendaal, 2006) screened 92 factors down to 11 before the costly experiment, and a
Plackett-Burman design asks about N+4 runs for the same question. The search gains by inserting that phase ahead
of the descent: measure each knob at its low and its high level, drop the knobs whose swing is inside the gain
threshold, and spend the passes on the rest.

Nothing here launches a server or reads a GPU. `evaluate(name, value)` is the caller's one trial, called once per
distinct (name, value) pair; the test fake stands in for it. The low and the high level of a knob come from its
declared values: "off" or "0" is low wherever it stands, the last other value is high, and a ladder without an off
value runs from its first value to its last. LOW_LEVEL_OVERRIDES is the one place a knob whose low level means
something else is declared.

A space of twenty or fewer knobs is measured knob by knob: two trials a knob is a price the descent pays back in
one pass, and an exact answer has no family representative's blind spot. Above that, knobs that belong to one
family -- the text before the first '-' or '_' past any leading dashes -- are grouped, at most four to a group,
and one representative is measured per group. That is the assumption Kleijnen's group test rests on: knobs that
belong together behave together, so a representative that does not move stands for a family that does not move.
The saving is what makes a 92-factor screen affordable; the risk is real and is why grouping starts only where
measuring every knob would cost more than the descent saves. A representative that moves keeps its group, which
is halved and measured again, until fewer than ten factors survive or the budget of trials is spent.
"""
import math
from collections.abc import Callable

# A declared value that says "this knob is not in play". The low level, wherever the declaration puts it.
_OFF_LEVELS = frozenset({"off", "0"})

# Knobs whose lowest declared value does not mean "off" map here to the value that is their low level. Example:
# a knob whose "0" means "automatic" and whose real low level is "8" would be declared {"KNOB": "8"}. Today's
# SPACE needs none: its flags spell their off state "off", and its numeric ladders start at their lowest value.
LOW_LEVEL_OVERRIDES: dict[str, str] = {}

# Below this many survivors, halving would keep nearly the whole space anyway, so the screen stops and reports
# them. A space up to twice this is measured knob by knob: two trials a knob costs less than the blind spot a
# family representative carries. Above it, a family group may hold at most MAX_GROUP knobs, bounding what one
# mistaken representative can discard.
_SURVIVOR_LIMIT = 10
_EXACT_LIMIT = 2 * _SURVIVOR_LIMIT
_MAX_GROUP = 4


def screen(space: dict[str, list[str]], evaluate: Callable[[str, str], float], baseline_rate: float,
           response: float = 0.02, budget: int | None = None) -> dict:
    """Measure each knob of `space` at its low and high level and keep the ones whose swing moves the rate.

    `evaluate(name, value)` returns one trial's rate with that knob set to `value` and the rest of the profile as
    the caller arranged. Each distinct (name, value) is measured at most once, so a representative met again in a
    later round costs nothing; each call is one engine trial for the caller.

    A knob moves when |high - low| exceeds `response` as a fraction of |baseline_rate| -- the descent's own gain
    gate, 2% by default -- in either direction. A knob that costs 10% is as much worth searching as one that gains
    5%, because the descent exists to move it the right way; signs are the descent's business, not the screen's.
    If baseline_rate is zero, the swing must exceed `response` itself, since no fraction is defined.

    `budget` caps evaluate() calls, one engine trial each. A knob is measured only when both of its levels fit in
    what remains, and each of its two levels is one call. A budget that runs out stops the screen with every
    unmeasured knob reported as a survivor, never as screened out: the screen may keep what it did not check, but
    it may not discard it.

    Returns {"survivors": names worth the descent, "screened_out": names that cannot move the rate, "evaluations":
    evaluate() calls made, "trace": one dict per round}. Both name lists are sorted and partition `space`, so a
    caller can feed survivors to the descent and leave the rest alone. The trace records every group, reading,
    kept group, dropped name and the reason the round was the last, in a deterministic order.
    """
    baseline_rate = float(baseline_rate)
    if not math.isfinite(baseline_rate):
        raise ValueError(f"baseline_rate must be finite, not {baseline_rate!r}")
    if not math.isfinite(response) or response < 0:
        raise ValueError(f"response must be a finite fraction of the baseline, not {response!r}")
    if budget is not None and budget < 0:
        raise ValueError(f"budget cannot be negative: {budget!r}")

    levels_of, screened = {}, set()
    for name in sorted(space):
        levels = _levels(name, space[name])
        if levels is None:
            screened.add(name)
        else:
            levels_of[name] = levels

    measured: dict[tuple[str, str], float] = {}

    def measure(name: str, value: str) -> float:
        key = (name, value)
        if key not in measured:
            rate = float(evaluate(name, value))
            if not math.isfinite(rate):
                raise ValueError(f"evaluate({name!r}, {value!r}) gave a rate that is not finite: {rate!r}")
            measured[key] = rate
        return measured[key]

    groups = _initial_groups(sorted(levels_of))
    trace, confirmed, survivors = [], [], []
    while groups:
        probed, kept, dropped, unprobed = [], [], [], []
        exhausted = False
        for index, group in enumerate(groups):
            name = group[0]
            low, high = levels_of[name]
            needed = sum(1 for level in (low, high) if (name, level) not in measured)
            if budget is not None and len(measured) + needed > budget:
                # This group and every group after it stay unscreened; that is a survivor, not a loss.
                exhausted = True
                unprobed = [list(later) for later in groups[index:]]
                break
            low_rate, high_rate = measure(name, low), measure(name, high)
            delta = high_rate - low_rate
            relative = delta / abs(baseline_rate) if baseline_rate else None
            moved = abs(relative) > response if relative is not None else abs(delta) > response
            probed.append({"name": name, "low": low, "high": high, "low_rate": low_rate, "high_rate": high_rate,
                           "delta": delta, "relative": relative, "moved": moved})
            if moved:
                kept.append(list(group))
            else:
                dropped.extend(group)
        confirmed.extend(group[0] for group in kept if len(group) == 1)
        refine = [group for group in kept if len(group) > 1]
        survivors = sorted(confirmed + [name for group in refine for name in group]
                           + [name for group in unprobed for name in group])
        if exhausted:
            reason = "budget"
        elif len(survivors) < _SURVIVOR_LIMIT:
            reason = "survivor_limit"
        elif not refine:
            reason = "resolved"
        else:
            reason = None
        trace.append({"round": len(trace) + 1, "groups": [list(group) for group in groups], "probed": probed,
                      "kept": [list(group) for group in kept], "dropped": sorted(dropped),
                      "unprobed": unprobed, "survivors": survivors, "reason": reason})
        screened.update(dropped)
        if reason is not None:
            break
        groups = [half for group in refine for half in _halve(group)]
    return {"survivors": sorted(survivors), "screened_out": sorted(screened),
            "evaluations": len(measured), "trace": trace}


def plackett_burman(n: int) -> list[list[int]]:
    """A Plackett-Burman screening design for n knobs: n+1 runs of n settings, each setting +1 or -1.

    A run is a row and a knob is a column. Every column is balanced (as many +1 as -1, which n = 3 mod 4 makes
    possible) and any two columns are orthogonal, so the design separates a knob's main effect from every other
    knob's with n+1 trials instead of a full grid. The first run is all +1, the baseline; the remaining runs are
    the screening trials. With n+1 runs there are n+1 vectors in n dimensions, so the runs themselves are not
    pairwise orthogonal; the PB condition is orthogonality between factor columns (X^T X = (n+1) I).

    Supported n are those where n % 4 == 3 and this module can build a Hadamard matrix of order n+1: n+1 a power
    of two, or n+1 = p+1 with p a prime that is 3 mod 4 (Paley's construction), or any such order doubled any
    number of times (Sylvester's). That covers 3, 7, 11, 15, 19, 23, 31, 39, 43, 47, 59, 63, 67, 71, 79, 83,
    95, 127, and so on -- n = 95 gives 96 runs, enough for the 92-factor screen that motivated this. n = 27 and
    n = 35 need Hadamard orders 28 and 36, which the other Paley construction builds and this one does not; they
    raise ValueError, as does any other unsupported n.
    """
    if not isinstance(n, int) or isinstance(n, bool):
        raise ValueError(f"a Plackett-Burman design is for a whole number of factors, not {n!r}")
    if n < 3 or n % 4 != 3:
        raise ValueError(f"no Plackett-Burman design of {n} factors: n must be 3 mod 4 and at least 3")
    hadamard = _hadamard(n + 1)
    if hadamard is None:
        raise ValueError(f"no Hadamard matrix of order {n + 1} is built here, so {n} factors is unsupported")
    return [row[1:] for row in hadamard]


def _levels(name: str, values) -> tuple | None:
    """A knob's (low, high) levels, or None when its declared values hold no contrast to measure.

    "off"/"0" is low wherever it is declared; the last value that is neither that nor off-like is high. A ladder
    with no off value runs from its first value to its last, the declaration order being the ladder. A single
    value, or only off-like values, has no high level and cannot move the rate.
    """
    levels = list(values)
    if not levels:
        return None
    if name in LOW_LEVEL_OVERRIDES:
        low = LOW_LEVEL_OVERRIDES[name]
        if low not in levels:
            raise ValueError(f"LOW_LEVEL_OVERRIDES sets {name} low to {low!r}, not one of its values {levels!r}")
    else:
        low = next((value for value in levels if _is_off(value)), levels[0])
    high = next((value for value in reversed(levels) if value != low and not _is_off(value)), None)
    return None if high is None else (low, high)


def _is_off(value) -> bool:
    """Whether a declared value means the knob is not in play."""
    return isinstance(value, str) and value.strip().lower() in _OFF_LEVELS


def _initial_groups(names: list[str]) -> list[list[str]]:
    """The knobs the first round measures a representative of.

    A space up to twice the survivor limit is measured knob by knob: it is small enough that the exact answer
    costs little, and a family representative could otherwise discard a moving knob behind a flat one. Above it,
    knobs are grouped by family -- names that belong to one subsystem, the text before the first '-' or '_' after
    any leading dashes -- and a family is chunked at MAX_GROUP. Sorting makes the family and member order
    deterministic.
    """
    if len(names) <= _EXACT_LIMIT:
        return [[name] for name in names]
    families: dict[str, list[str]] = {}
    for name in names:
        families.setdefault(_family(name), []).append(name)
    groups = []
    for family in sorted(families):
        members = families[family]
        for start in range(0, len(members), _MAX_GROUP):
            groups.append(members[start:start + _MAX_GROUP])
    return groups


def _family(name: str) -> str:
    """The subsystem a knob's name puts it in: "draft" for "--draft-depth", "GGML" for "GGML_CUDA_...".

    The grouping is a heuristic, not a claim about the kernels: it puts knobs that a person declared together in
    the space close together, which is the cheapest available guess at which knobs behave alike.
    """
    stem = name.lstrip("-")
    for separator in ("-", "_"):
        stem = stem.split(separator, 1)[0]
    return stem


def _halve(group: list[str]) -> list[list[str]]:
    """A group into two non-empty halves; both halves are measured again next round."""
    middle = len(group) // 2
    return [group[:middle], group[middle:]]


def _hadamard(order: int) -> list[list[int]] | None:
    """A Hadamard matrix of `order` (+/-1 rows orthogonal, order 4 or a multiple of 4) or None if none is built.

    Sylvester's doubling covers every power of two and every double of an order built here. Paley's construction
    covers order p+1 for a prime p that is 3 mod 4. Together they cover the Plackett-Burman sizes this search
    needs, including 96 runs for 95 factors; the remaining Hadamard orders (28, 36, ...) are refused rather than
    guessed at.
    """
    if order >= 4 and order & (order - 1) == 0:
        return _sylvester(order)
    quotient = order - 1
    if quotient >= 3 and quotient % 4 == 3 and _is_prime(quotient):
        return _paley(quotient)
    if order >= 8 and order % 2 == 0:
        half = _hadamard(order // 2)
        if half is not None:
            return [row + row for row in half] + [row + [-value for value in row] for row in half]
    return None


def _sylvester(order: int) -> list[list[int]]:
    """Sylvester's construction: H(2m) = [[H(m), H(m)], [H(m), -H(m)]] from H(2)."""
    matrix = [[1, 1], [1, -1]]
    while len(matrix) < order:
        matrix = ([row + row for row in matrix]
                  + [row + [-value for value in row] for row in matrix])
    return matrix


def _paley(prime: int) -> list[list[int]]:
    """Paley type I: order prime+1 from the Jacobsthal matrix of a prime that is 3 mod 4.

    H = [[1, 1^T], [1, J - I]] with J[i][j] the quadratic character of i - j. For prime = 3 mod 4, J is
    skew-symmetric, and J J^T = prime I - E, which is what makes the rows of H orthogonal.
    """
    residues = {pow(value, 2, prime) for value in range(1, prime)}

    def character(value: int) -> int:
        value %= prime
        if value == 0:
            return 0
        return 1 if value in residues else -1

    matrix = [[1] * (prime + 1)]
    for row in range(prime):
        matrix.append([1] + [character(row - column) - (1 if row == column else 0) for column in range(prime)])
    return matrix


def _is_prime(value: int) -> bool:
    """Whether `value` is prime, by trial division: the Hadamard orders here are small."""
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    divisor = 3
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 2
    return True
