#!/usr/bin/env python3
"""The launch budget of an Inkling decode step, counted from the engine ledger.

Each executed graph node is at least one dispatch, and the CUDA backend adds every evaluation's node count to
the ledger slot cuda.eval.nodes[device] (ggml/src/ggml-cuda/ggml-cuda.cu). The ledger's top-level "steps" counts
decode calls, so the launches a step costs on a card are the two snapshots' difference in that slot over the
difference in steps. The check holds the result to at most half the baseline, the same way on every card, and
the ncu count of a decode window to the same ceiling.

  launch_budget.py build --baseline BEFORE AFTER --result BEFORE AFTER --build SHA --profile tensor-4 --out PATH
  launch_budget.py --check --evidence PATH

A snapshot is a GGML_LEDGER_FILE the engine wrote at exit (GGML_LEDGER=1), or a server's /ledger saved to a
file by the capture. The build command records where each snapshot came from, so the evidence stands alone.
"""
import argparse
import datetime
import json
import sys

# the ledger's per-device node counter and the step counter
NODE_SLOT = "cuda.eval.nodes"
STEP_KEY = "steps"


def load(path):
    try:
        with open(path) as file:
            doc = json.load(file)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot read a ledger snapshot from {path}: {exc}")
    if not isinstance(doc, dict) or not isinstance(doc.get("slots"), dict):
        raise SystemExit(f"{path} is not a ledger snapshot: no slots")
    return doc


def slots_of(snapshot, prefix):
    """The device slots of a name, by device index: cuda.eval.nodes[0], [1], ..."""
    found = {}
    for name, slot in snapshot["slots"].items():
        if not name.startswith(prefix + "["):
            continue
        try:
            device = int(name[len(prefix) + 1:-1])
        except ValueError:
            continue
        found[device] = slot
    if not found:
        raise SystemExit(f"no {prefix}[device] slots: the ledger was off or no graph was evaluated")
    return found


def per_step(before, after):
    """The launches one decode step cost on each device between two snapshots."""
    steps = after.get(STEP_KEY, 0) - before.get(STEP_KEY, 0)
    if steps <= 0:
        raise SystemExit("the two snapshots hold no decode step between them")
    old = slots_of(before, NODE_SLOT)
    new = slots_of(after, NODE_SLOT)
    devices = sorted(set(old) & set(new))
    if not devices:
        raise SystemExit("the two snapshots share no device slot")
    counts = {device: (new[device].get("sum", 0) - old[device].get("sum", 0)) / steps for device in devices}
    return steps, counts


def build(args):
    base_steps, base_counts = per_step(load(args.baseline[0]), load(args.baseline[1]))
    res_steps, res_counts = per_step(load(args.result[0]), load(args.result[1]))
    if sorted(base_counts) != sorted(res_counts):
        raise SystemExit("baseline and result were counted on different devices")
    evidence = {
        "version": 1,
        "row": "inkling-launch-budget",
        "build": args.build,
        "profile": args.profile,
        "steps": {"baseline": base_steps, "result": res_steps},
        "per_device": {"baseline": [base_counts[d] for d in sorted(base_counts)],
                       "result": [res_counts[d] for d in sorted(res_counts)]},
        # one number for the row: the card that launches the most, which no average can hide
        "baseline_per_step": max(base_counts.values()),
        "result_per_step": max(res_counts.values()),
        "ncu": {"baseline_launches": args.ncu_baseline, "result_launches": args.ncu_result,
                "command": args.ncu_command},
        "sources": {"baseline": list(args.baseline), "result": list(args.result)},
        "measured_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
    if args.ncu_baseline is None or args.ncu_result is None:
        raise SystemExit("--ncu-baseline and --ncu-result name the ncu launch counts of the two arms")
    with open(args.out, "w") as file:
        json.dump(evidence, file, indent=2)
        file.write("\n")
    print(f"wrote {args.out}: baseline {evidence['baseline_per_step']:.1f} launches a step a card, "
          f"result {evidence['result_per_step']:.1f}, ratio {evidence['result_per_step']/evidence['baseline_per_step']:.2f}")
    return 0


def check(evidence):
    problems = []
    for field in ("build", "profile", "baseline_per_step", "result_per_step", "per_device", "ncu", "sources",
                  "steps", "measured_at"):
        if field not in evidence:
            problems.append(f"missing {field}")
    if problems:
        return problems
    if not str(evidence["build"]).strip():
        problems.append("build is empty")
    if evidence["profile"] != "tensor-4":
        problems.append(f"profile is {evidence['profile']!r}, the row is judged at tensor-4")
    base = evidence["baseline_per_step"]
    result = evidence["result_per_step"]
    if not base or base <= 0:
        problems.append("baseline_per_step is not positive")
    if result <= 0:
        problems.append("result_per_step is not positive")
    if base and base > 0 and not result * 2 <= base:
        problems.append(f"result {result:.1f} is above half the baseline {base:.1f}")
    per = evidence["per_device"]
    if len(per.get("baseline", [])) != len(per.get("result", [])):
        problems.append("per_device baseline and result differ in length")
    else:
        for device, (b, r) in enumerate(zip(per["baseline"], per["result"])):
            if not (b > 0 and r * 2 <= b):
                problems.append(f"device {device}: result {r:.1f} is above half the baseline {b:.1f}")
    ncu = evidence["ncu"]
    for field in ("baseline_launches", "result_launches", "command"):
        if not ncu.get(field):
            problems.append(f"ncu.{field} is missing")
            break
    if ncu.get("baseline_launches") and ncu.get("result_launches"):
        if not ncu["result_launches"] * 2 <= ncu["baseline_launches"]:
            problems.append(f"ncu launches {ncu['result_launches']} are above half the baseline "
                            f"{ncu['baseline_launches']}")
        # the ledger and ncu count different things (nodes against kernels), so only their ratios compare
        ledger_ratio = result / base if base else 0.0
        ncu_ratio = ncu["result_launches"] / ncu["baseline_launches"]
        if ledger_ratio and not 0.6 * ledger_ratio <= ncu_ratio <= 1.6 * ledger_ratio:
            problems.append(f"ncu ratio {ncu_ratio:.2f} and ledger ratio {ledger_ratio:.2f} disagree past 60 %")
    if not evidence["sources"].get("baseline") or not evidence["sources"].get("result"):
        problems.append("sources name no snapshot")
    return problems


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command")
    b = sub.add_parser("build", help="two snapshot pairs into the evidence file")
    b.add_argument("--baseline", nargs=2, metavar=("BEFORE", "AFTER"), required=True)
    b.add_argument("--result", nargs=2, metavar=("BEFORE", "AFTER"), required=True)
    b.add_argument("--build", required=True, help="the tree or commit the result arm ran")
    b.add_argument("--profile", default="tensor-4")
    b.add_argument("--ncu-baseline", type=int, help="launches in the baseline arm's decode window (ncu)")
    b.add_argument("--ncu-result", type=int, help="launches in the result arm's decode window (ncu)")
    b.add_argument("--ncu-command", default="ncu --set basic --launch-count 0 -c 1 ... (name the exact command)")
    b.add_argument("--out", default="reports/inkling2-launch-budget.json")
    parser.add_argument("--check", action="store_true", help="judge an evidence file")
    parser.add_argument("--evidence", default="reports/inkling2-launch-budget.json")
    args = parser.parse_args(argv)
    if args.check:
        try:
            with open(args.evidence) as file:
                evidence = json.load(file)
        except (OSError, ValueError) as exc:
            print(f"FAIL: no evidence at {args.evidence}: {exc}")
            return 1
        problems = check(evidence)
        if problems:
            for problem in problems:
                print(f"FAIL: {problem}")
            return 1
        print(f"ok: {evidence['result_per_step']:.1f} launches a step a card against a baseline of "
              f"{evidence['baseline_per_step']:.1f} ({evidence['result_per_step']/evidence['baseline_per_step']:.2f})")
        return 0
    if args.command == "build":
        return build(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
