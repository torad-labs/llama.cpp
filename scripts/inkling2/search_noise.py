#!/usr/bin/env python3
"""Treat a trial's noise as part of what it measured, never as something repeats average away.

A decode rate moves between measurements of the same settings: scheduler noise, clocks, a neighbour on the card.
TUNA (EuroSys 2025) found 5% noise slowed a search's convergence 2.5x, and that up to 63.3% of the configurations
a single-sample search keeps lose 30% or more of their rate when deployed. So this module supplies the search:

  unstable            do a trial's repeats disagree by more than a relative spread?
  aggregate           a trial's rate: the minimum of its repeats, never the median or the mean
  confirm             measure a trial, draw again while its repeats disagree, and say what it found
  needs_confirmation  should a candidate earn repeats at all, or is it a loser already?

The descent keeps spending repeats only where a repeat can change the answer: a candidate is measured once, and
only one that could beat the best by more than the search's gain is measured again.
"""
import statistics
from collections.abc import Callable


def unstable(rates: list[float], relative_spread: float = 0.10) -> bool:
    """Whether a trial's repeats disagree by more than relative_spread of their median: (max - min) / median.

    The threshold is strict: a spread exactly at it is stable. A single repeat, or repeats that all agree, are
    stable. Rates whose median is zero cannot be measured relative to it, so any spread among them is unstable.
    """
    if not rates:
        raise ValueError("a spread over no rates is unknown")
    low, high = min(rates), max(rates)
    if high == low:
        return False
    middle = statistics.median(rates)
    if middle <= 0:
        return True
    return (high - low) / middle > relative_spread


def aggregate(rates: list[float]) -> float:
    """A trial's rate from its repeats: the minimum (TUNA's min aggregation), not the median or the mean.

    A configuration is only as fast as its bad draws: the median hides the slow repeat that a deployment will meet.
    """
    if not rates:
        raise ValueError("a rate over no repeats is unknown")
    return float(min(rates))


def confirm(measure_one: Callable[[], dict], repeats: int, relative_spread: float = 0.10,
            max_repeats: int | None = None) -> dict:
    """Measure one trial `repeats` times, and keep drawing while its repeats disagree, up to max_repeats.

    measure_one is called once a repeat and returns a trial record whose "tokens_per_second" is read; a record
    without a rate is an error, never a zero. The trial's answer is {"tokens_per_second": the minimum of its rates,
    "rates": them in the order drawn, "stable": whether the last spread was inside relative_spread, "repeats": how
    many were drawn}. max_repeats defaults to repeats, so the caller that can afford more passes a higher one and
    one that cannot gets exactly the repeats it asked for; when the draws never settle, the last one is the
    answer and "stable" answers False.
    """
    if repeats < 1:
        raise ValueError("a confirmation draws at least once")
    max_repeats = repeats if max_repeats is None else max_repeats
    if max_repeats < repeats:
        raise ValueError("max_repeats cannot be under repeats")
    rates = []
    while True:
        rates.append(_rate_of(measure_one()))
        stable = not unstable(rates, relative_spread)
        if len(rates) >= repeats and (stable or len(rates) >= max_repeats):
            break
    return {"tokens_per_second": aggregate(rates), "rates": rates, "stable": stable, "repeats": len(rates)}


def needs_confirmation(candidate_rate: float, best_rate: float, gain: float) -> bool:
    """Whether a candidate's one measurement could beat the best by more than gain, so repeats are worth spending.

    The gate is the search's own, candidate > best * (1 + gain): a candidate exactly at that rate is no better than
    the best under the rule that decides what replaces it, and earns nothing.
    """
    return candidate_rate > best_rate * (1 + gain)


def _rate_of(record: dict) -> float:
    """The tokens a second a trial record measured, or an error saying the record has none."""
    rate = record.get("tokens_per_second") if isinstance(record, dict) else None
    if isinstance(rate, bool) or not isinstance(rate, (int, float)):
        raise ValueError(f"a trial record carries no tokens_per_second: {record!r}")
    return float(rate)
