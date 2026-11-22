"""The ledger of a search trial, read back: why a setting won, not only by how much it won by.

search.py records the engine's ledger summary with every trial (scripts/ledger.py, summary()), and nothing read it
back. This module reads it: features() turns one trial's summary into per-step evidence, explain() sets the fastest
trials' evidence against the rest as the "why a setting wins" table a report shows, and unstable_flags() names the
signs a window was not representative, so a rate measured on a moving target is not kept as a win.

  features(trial)        {"device_busy_share": ..., "captures_per_step": ..., ...}
  explain(trials, top=5) {"features": {name: {"top_mean", "rest_mean", "delta"}}, "trials": [{"why": ...}]}
  unstable_flags(trial)  [] for a steady window, else the names of the signs it was not

A trial is the record search.py writes: its rate under "tokens_per_second" and its summary under "ledger". A summary
is {"seconds", "steps", "unit", "slots"}, a slot is {"n_per_step", "sum_per_step", "mean", "p50", "p99", "max"} and a
gauge {"value", "max"}; a device's slots end in [device]. A missing slot, a missing ledger and a summary without a
wall time all read as 0.0: a trial with no evidence still ranks by its rate.
"""

# A capture every other decode step or more means the kept-graph cache was still being filled or kept changing shape
# through the measured window, so the rate was taken on moving graphs. Steady decode captures a graph once a shape and
# replays it, so captures per step sits near zero and a window that spends half its steps capturing is not that.
CAPTURES_PER_STEP_UNSTABLE = 0.5

# More than one kept graph turned down a step means the shape it was kept for almost never came back (drafts of
# varying length, mixed ubatches). Up to one a step is the ordinary cost of decode shapes that vary; above it the
# window was not a steady repeat of one step, and its rate is not the rate a repeated workload sees.
REJECTS_PER_STEP_UNSTABLE = 1.0

# A decode's p99 above three times its p50 is a window with stalls in it (a first capture, an allocation, a device
# sync), not the steady per-step cost whose change a search is comparing.
P99_OVER_P50_UNSTABLE = 3.0


def _summary(trial):
    """A trial's ledger summary, or {} for a trial that carries none."""
    found = trial.get("ledger") if isinstance(trial, dict) else None
    return found if isinstance(found, dict) else {}


def _slots(summary):
    found = summary.get("slots")
    return found if isinstance(found, dict) else {}


def _number(mapping, key):
    """A number in a slot or a summary; 0.0 when it is missing or is not a number (a gauge, a string, None)."""
    value = mapping.get(key) if isinstance(mapping, dict) else None
    return float(value) if isinstance(value, (int, float)) else 0.0


def _named(slots, name):
    """The slots named `name` or `name[device]`: a per-device entry carries its device in brackets."""
    return [slot for key, slot in slots.items()
            if isinstance(slot, dict) and (key == name or key.startswith(name + "["))]


def _slot(slots, name):
    found = _named(slots, name)
    return found[0] if found else {}


def _family(slots, prefix):
    """The slots whose name begins with `prefix`, a family the engine names by reason or by device."""
    return [slot for key, slot in slots.items() if isinstance(slot, dict) and key.startswith(prefix)]


def _round(value):
    return round(float(value), 6)


def features(trial):
    """One trial's ledger as the per-step evidence a comparison reads, a flat dict of floats, 0.0 when absent.

    device_busy_share is card time a decode's wall time, summed over the cards: 1.0 is one card's worth of work in
    every decode second, one per card when they all ran. The span slots are timed one decode in the ledger's period
    and a summary keeps no period, but `mean` is already per timed decode, and a decode is one graph evaluation a
    card, so the means are what a decode's wall time compares to. The rest are per decode step: captures
    (cuda.eval.capture_us additions), evictions (cuda.graph.evict.*), kept graphs turned down (graph.kept.reject.*;
    each addition carries in sum_per_step the rows it refused), host time between cards (meta.reduce_us and
    meta.gather_us), the decode's own graph_compute, and the rows it carried.
    """
    summary = _summary(trial)
    slots = _slots(summary)
    seconds = _number(summary, "seconds")
    steps = _number(summary, "steps")
    wall_us = seconds * 1e6 / steps if seconds > 0 and steps > 0 else 0.0
    cards_us = sum(_number(slot, "mean") for slot in _named(slots, "cuda.span_us"))
    meta_us = sum(_number(slot, "sum_per_step") for name in ("meta.reduce_us", "meta.gather_us")
                  for slot in _named(slots, name))
    return {
        "device_busy_share": _round(cards_us / wall_us) if wall_us > 0 else 0.0,
        "captures_per_step": _round(sum(_number(slot, "n_per_step") for slot in _named(slots, "cuda.eval.capture_us"))),
        "evictions_per_step": _round(sum(_number(slot, "n_per_step") for slot in _family(slots, "cuda.graph.evict."))),
        "rejects_per_step": _round(sum(_number(slot, "n_per_step") for slot in _family(slots, "graph.kept.reject."))),
        "meta_reduce_gather_us_per_step": _round(meta_us),
        "decode_graph_compute_us_per_step": _round(_number(_slot(slots, "decode.graph_compute"), "sum_per_step")),
        "rows_per_step": _round(_number(_slot(slots, "decode.rows"), "sum_per_step")),
    }


def unstable_flags(trial):
    """The signs a trial's window was not representative, each named; empty when it looked steady.

    A flag does not say the rate is wrong, only that it was measured on a moving target: a search should make a
    flagged trial pass its gain gate on evidence, not keep it as a win on its rate alone.
    """
    slots = _slots(_summary(trial))
    found = features(trial)
    total = _slot(slots, "decode.total")
    flags = []
    if _number(total, "p99") > P99_OVER_P50_UNSTABLE * _number(total, "p50"):
        flags.append("decode_total_p99_over_3x_p50")
    if found["captures_per_step"] > CAPTURES_PER_STEP_UNSTABLE:
        flags.append("captures_per_step_high")
    if found["rejects_per_step"] > REJECTS_PER_STEP_UNSTABLE:
        flags.append("rejects_per_step_high")
    return flags


def explain(trials, top=5):
    """Why the fastest trials won: their evidence against the rest's, per feature, and each one's leading reasons.

    Trials rank by "tokens_per_second"; the fastest `top` of them that have a rate are set against every other one,
    and each feature reports the two means and their delta, sorted by how far apart they are. Each top trial also
    gets a "why" line: the two features it beats the rest's mean by most, with the deltas, because a win the evidence
    cannot explain is a coin that landed well. A trial without a ledger has zero features and still ranks.
    Without a rest the rest's mean is 0.0; without a rated trial the result is {"features": {}, "trials": []}.
    """
    ranked = sorted((trial for trial in (trials or [])
                     if isinstance(trial, dict) and isinstance(trial.get("tokens_per_second"), (int, float))),
                    key=lambda trial: -trial["tokens_per_second"])
    if not ranked:
        return {"features": {}, "trials": []}
    count = max(1, min(int(top), len(ranked)))
    found = [features(trial) for trial in ranked]
    names = sorted({name for row in found for name in row})

    def mean_of(rows, name):
        return sum(row.get(name, 0.0) for row in rows) / len(rows) if rows else 0.0

    table = {}
    for name in names:
        top_mean = mean_of(found[:count], name)
        rest_mean = mean_of(found[count:], name)
        table[name] = {"top_mean": _round(top_mean), "rest_mean": _round(rest_mean), "delta": _round(top_mean - rest_mean)}
    table = {name: table[name] for name in sorted(table, key=lambda name: (-abs(table[name]["delta"]), name))}

    entries = []
    for trial, row in zip(ranked[:count], found[:count]):
        gains = sorted(((row.get(name, 0.0) - table[name]["rest_mean"], name) for name in names
                        if row.get(name, 0.0) - table[name]["rest_mean"] > 0),
                       key=lambda gain: (-gain[0], gain[1]))
        why = ", ".join(f"{name} {_round(gain):+g}" for gain, name in gains[:2])
        entries.append({"key": trial.get("key"), "tokens_per_second": trial["tokens_per_second"],
                        "settings": trial.get("settings", {}), "why": why})
    return {"features": table, "trials": entries}
