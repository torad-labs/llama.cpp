#!/usr/bin/env python3
"""Read the engine's ledger (ggml/include/ggml-ledger.h): where a step's time and a device's resources went.

  ledger.py show  SNAPSHOT            everything one snapshot holds, since the process started
  ledger.py diff  BEFORE AFTER        what happened between two snapshots, a step
  ledger.py watch URL [--seconds N]   two snapshots of a running server N seconds apart, and their difference

A snapshot is a file (GGML_LEDGER_FILE writes one when a process exits) or a server's address (GET /ledger). The
engine names what it counts, so this reader knows no model: it prints what is there.
"""
import argparse
import json
import sys
import time
import urllib.request


def load(source):
    if source.startswith(("http://", "https://")):
        with urllib.request.urlopen(source.rstrip("/") + "/ledger", timeout=30) as response:
            return json.load(response)
    with open(source) as file:
        return json.load(file)


def difference(before, after):
    """What the slots gained between two snapshots. A gauge keeps its last value; a slot that gained nothing is left out."""
    slots = {}
    for name, new in after["slots"].items():
        old = (before or {"slots": {}})["slots"].get(name, {})
        if "g" in new:
            slots[name] = {"g": new["g"], "max": new["max"]}
            continue
        n = new["n"] - old.get("n", 0)
        if n <= 0:
            continue
        hist = {int(k): c - old.get("h", {}).get(k, 0) for k, c in new["h"].items()}
        slots[name] = {"n": n, "sum": new["sum"] - old.get("sum", 0), "max": new["max"],
                       "h": {k: c for k, c in hist.items() if c > 0}}
    first = before or {"t_us": after["t_us"], "steps": 0, "timed_steps": 0}
    return {"seconds": (after["t_us"] - first["t_us"]) / 1e6 if before else None, "steps": after["steps"] - first["steps"],
            "timed_steps": after["timed_steps"] - first["timed_steps"], "period": after.get("period"), "slots": slots}


def quantile(slot, q):
    """The value q of the slot's entries are under, to the power of two its histogram keeps."""
    want, seen = q * slot["n"], 0
    for k in sorted(slot["h"]):
        seen += slot["h"][k]
        if seen >= want:
            return min(slot["max"], 2 ** k if k else 0)
    return slot["max"]


def summary(diff):
    """The difference as plain numbers a step, for a record of a run: {name: {per_step, mean, p50, p99, max}}."""
    steps = max(1, units(diff)[1])
    out = {}
    for name, slot in diff["slots"].items():
        if "g" in slot:
            out[name] = {"value": slot["g"], "max": slot["max"]}
        else:
            out[name] = {"n_per_step": round(slot["n"] / steps, 3), "sum_per_step": round(slot["sum"] / steps, 1),
                         "mean": round(slot["sum"] / slot["n"], 1), "p50": quantile(slot, 0.5), "p99": quantile(slot, 0.99),
                         "max": slot["max"]}
    return {"seconds": diff["seconds"], "steps": steps, "unit": units(diff)[0], "slots": out}


def units(diff):
    """What a step is: a server's iteration where the ledger has them, else a decode."""
    iteration = diff["slots"].get("stage.iteration")
    if iteration:
        return "server iteration", iteration["n"]
    return "decode", diff["steps"]


def render(diff, out=sys.stdout):
    unit, steps = units(diff)
    steps = max(1, steps)
    slots = diff["slots"]
    wall = diff["seconds"] * 1e6 / steps if diff["seconds"] else None
    print(f"{steps} steps (a step is one {unit})" + (f", {diff['seconds']:.1f} s, {wall / 1e3:.2f} ms a step" if wall else ""), file=out)

    def table(title, names, value="us"):
        rows = [(name, slots[name]) for name in names if name in slots and "n" in slots[name]]
        if not rows:
            return
        print(f"\n{title}", file=out)
        print(f"  {'':44} {'a step':>10} {'times':>8} {'mean':>9} {'p50':>9} {'p99':>9} {'max':>10}", file=out)
        for name, slot in sorted(rows, key=lambda row: -row[1]["sum"]):
            print(f"  {name:44} {slot['sum'] / steps:10.1f} {slot['n'] / steps:8.2f} {slot['sum'] / slot['n']:9.1f} "
                  f"{quantile(slot, 0.5):9} {quantile(slot, 0.99):9} {slot['max']:10}", file=out)

    def group(prefix):
        return [name for name in slots if name.startswith(prefix)]

    table("Server stages, microseconds", [n for n in group("stage.") if not n.endswith(".rows")])
    table("Rows a stage carried", [n for n in group("stage.") if n.endswith(".rows")])
    table("Decode phases on the host, microseconds", [n for n in group("decode.") if "rows" not in n and "seqs" not in n])
    table("Step shapes: rows and sequences", [n for n in group("decode.") if "rows" in n or "seqs" in n])
    table("Kept graphs turned down, by reason (rows of the step refused)", group("graph.kept.reject"))
    table("Graphs built where none kept was reused, by cause (rows of the step)", group("graph.kept.miss"))
    table("Between devices, microseconds on the host", [n for n in group("meta.") if n.endswith("_us")])
    table("Between devices, events", [n for n in group("meta.") if not n.endswith("_us")])
    table("Graph evaluations a device, microseconds on the host", [n for n in group("cuda.eval.") if n.split("[")[0].endswith("_us")])
    table("Graph evaluations run kernel by kernel, by reason", group("cuda.eval.direct."))
    table("Device graph events (nodes)", [n for n in group("cuda.graph.") if "held" not in n] + group("cuda.outer."))

    spans = [name for name in group("cuda.span_us")]
    if spans and diff["timed_steps"]:
        print(f"\nDevice time, from {diff['timed_steps']} timed decodes (one in {diff['period']})", file=out)
        decodes = max(1, diff["steps"])
        decode_wall = diff["seconds"] * 1e6 / decodes if diff["seconds"] else None
        for name in sorted(spans):
            span = slots[name]["sum"] / diff["timed_steps"]
            share = f", {100 * span / decode_wall:.0f}% of a decode's wall time" if decode_wall else ""
            print(f"  {name:44} {span:10.1f} us a decode{share}", file=out)

    gauges = [(name, slot) for name, slot in slots.items() if "g" in slot]
    if gauges:
        print("\nLast value and highest seen", file=out)
        for name, slot in sorted(gauges):
            print(f"  {name:60} {slot['g']:10} {slot['max']:10}", file=out)

    shown = ("stage.", "decode.", "graph.kept.reject", "graph.kept.miss", "meta.", "cuda.eval.", "cuda.graph.", "cuda.outer.", "cuda.span_us")
    table("Everything else", [name for name in slots if not name.startswith(shown)])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    verbs = parser.add_subparsers(dest="verb", required=True)
    show = verbs.add_parser("show")
    show.add_argument("snapshot")
    diff = verbs.add_parser("diff")
    diff.add_argument("before")
    diff.add_argument("after")
    watch = verbs.add_parser("watch")
    watch.add_argument("url")
    watch.add_argument("--seconds", type=float, default=10)
    for verb in (show, diff, watch):
        verb.add_argument("--json", action="store_true", help="Print the numbers a step as JSON in place of the tables")
    args = parser.parse_args(argv)
    if args.verb == "show":
        result = difference(None, load(args.snapshot))
    elif args.verb == "diff":
        result = difference(load(args.before), load(args.after))
    else:
        before = load(args.url)
        time.sleep(args.seconds)
        result = difference(before, load(args.url))
    if args.json:
        print(json.dumps(summary(result), indent=1))
    else:
        render(result)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError) as error:
        print(f"ledger: {error}", file=sys.stderr)
        raise SystemExit(2)
