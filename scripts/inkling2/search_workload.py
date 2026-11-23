#!/usr/bin/env python3
"""Measure one trial over several context depths and keep its worst.

  depths_from(spec)                parse --depths ("0,16384,262144") into the depths a trial is measured at
  aggregate(results)               one trial's records over depths into the record the search keeps
  run_depths(measure_one, depths)  measure a trial depth by depth; a depth that fails costs only that depth

The search measured a trial at one context depth and kept its median rounds. A configuration tuned on one trace is up
to twice as bad on another (Vidur, CGPTuner), and the request alone is one trace only: a draft depth that pays on a
cold prompt can fall over once a session decodes behind 128k tokens of context. So the workload is part of what the
search asks: a trial is measured at every depth the profile serves (--depths 0,16384,262144), and the rate the search
compares is the lowest of them, the worst depth the setting survives, not the best it was shown.

Measured at one depth, the result is exactly what search.py's measure() writes today: the worst depth's own record is
spread into the result, so its rounds, prompt_tokens and ledger are the trial's own and a reader of trials.jsonl or
MLflow sees the record it saw before. per_depth keeps every depth's record in depth order, each annotated with its
position, so a value that helps one depth and hurts another stays visible after the fact; failed_depths names the
depths that gave no record instead of hiding them.
"""
import math


def depths_from(spec):
    """The depths a --depths argument names, sorted, each measured once. None or an empty argument is the request
    alone: [0].

    An entry that is not a whole number of tokens, or is negative, is an error and not skipped: a search that quietly
    measured a depth nobody asked for would keep a setting for a workload nobody runs."""
    if spec is None or not spec.strip():
        return [0]
    depths = []
    for part in spec.split(","):
        token = part.strip()
        if not token:
            raise ValueError("--depths has an empty entry; write the depths separated by commas, as in 0,16384")
        try:
            depth = int(token)
        except ValueError:
            raise ValueError(f"--depths names {token!r}, which is not a whole number of tokens") from None
        if depth < 0:
            raise ValueError(f"--depths names {token!r}; a context depth cannot be negative")
        depths.append(depth)
    return sorted(set(depths))


def rate_of(result):
    """The rate a record kept, or None when it kept none a search could compare (missing, None, a bool or a NaN)."""
    rate = result.get("tokens_per_second")
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or math.isnan(rate):
        return None
    return rate


def aggregate(results):
    """One trial's records over depths as the record the search keeps: the worst depth's own record, its rate replaced
    by the lowest rate seen.

    results holds one record a depth, in depth order, each shaped like measure()'s output. The result carries every
    key of the worst record (rounds, prompt_tokens, ledger and whatever else a trial records), so a record kept in
    trials.jsonl and sent to MLflow keeps the shape it had before the sweep; per_depth carries every depth's record,
    annotated with its depth_order, and worst_depth is the position in per_depth of the one kept. The shallowest depth
    wins a tie, because the depths are sorted and the shallowest is the one served most."""
    results = list(results)
    if not results:
        raise ValueError("a trial needs one measurement a depth; none were given")
    for position, result in enumerate(results):
        if not isinstance(result, dict):
            raise ValueError(f"depth {position} gave no record: {result!r}")
        if rate_of(result) is None:
            raise ValueError(f"depth {position} has no rate to compare: {result.get('tokens_per_second')!r}")
    worst = min(range(len(results)), key=lambda position: rate_of(results[position]))
    return {**results[worst], "tokens_per_second": rate_of(results[worst]), "worst_depth": worst,
            "per_depth": [{**result, "depth_order": position} for position, result in enumerate(results)]}


def run_depths(measure_one, depths):
    """A trial measured once a depth, aggregated over what came back. measure_one(depth) is measure() with the context
    depth set, and returns its record.

    A depth whose measurement raises is left out of the aggregation and named in failed_depths with its message, so one
    depth that fails costs its own data and not the trial's. A record with no rate is a failed depth by the same rule,
    and its own error message is the one kept. When every depth fails, the trial's rate is None and the error says how
    each depth failed: the shape search.py's run_trial writes for a trial that could not be measured."""
    depths = list(depths)
    if not depths:
        raise ValueError("no depths to measure")
    results, failed = [], []
    for depth in depths:
        try:
            result = measure_one(depth)
            if not isinstance(result, dict):
                raise ValueError(f"the measurement gave no record: {result!r}")
            if rate_of(result) is None:
                raise ValueError(result.get("error") or f"the measurement gave no rate: {result.get('tokens_per_second')!r}")
        except Exception as error:  # one depth's failure is that depth's data, not the trial's
            failed.append({"depth": depth, "error": str(error)})
            continue
        results.append({**result, "depth": depth})
    if not results:
        return {"tokens_per_second": None, "per_depth": [], "worst_depth": None, "failed_depths": failed,
                "error": "no depth of " + ", ".join(str(depth) for depth in depths) + " gave a rate: "
                         + "; ".join(f"{entry['depth']}: {entry['error']}" for entry in failed)}
    return {**aggregate(results), "failed_depths": failed}
