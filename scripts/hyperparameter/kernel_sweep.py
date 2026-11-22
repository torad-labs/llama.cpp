#!/usr/bin/env python3
"""Sweep compile-time kernel constants as build defines, one constant at a time, at the serving shapes.

  kernel_sweep.py plan CONFIG [--constant NAME]...
        print the configure, build and timing command for every value
  kernel_sweep.py run CONFIG [--results DIR] [--constant NAME]...
        run the plan sequentially, one JSONL record per value in results/TIMESTAMP.jsonl
  kernel_sweep.py report DIR
        print the best value per constant and its delta against the default

A value reaches the kernel only when the source wraps its #define in #ifndef NAME: the harness passes
-DCMAKE_CUDA_FLAGS=-DNAME=value, and without the guard the source's own #define overrides the command line
(macro redefinition, the later one wins), so the sweep would silently measure the default again. plan and run
read the source and refuse a constant whose guard is missing, naming the file and line to fix. A value whose
configure or build fails, or whose timing binary exits nonzero, is recorded as invalid and the sweep goes on.

No timeout is ever added: every configure, build and timing run goes on until it finishes. The timing command
comes from the config ([timing], or a constant's own timing), so the box fills in the backend and shape flags;
the harness passes -o and every other argument through untouched.
"""
import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import tomllib

PROG = "kernel_sweep.py"

# "  MUL_MAT_ID(type_a=q3_K,...):   2048 runs -   123.45 us/run -   1.23 TFLOP/run -   9.96 TFLOPS"
PERF_LINE = re.compile(
    r"([A-Za-z0-9_]+)\((.*?)\):\s+(\d+) runs\s+-\s+([0-9.]+) us/run\s+-\s+(.*)"
)
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

BUILD_KEYS = {"source_dir", "build_dir", "target", "jobs", "configure", "build", "binary"}
CONSTANT_KEYS = {
    "define", "source", "source_line", "default", "values", "op", "needs_guard",
    "kind", "rules", "timing", "guard_hint", "note", "extra_flags",
}
RULE_KEYS = {"min", "max", "multiple_of", "power_of_two"}


class SweepError(Exception):
    """A config or input the sweep cannot use; the message says what to fix."""


@dataclass
class Rules:
    min: int | None = None
    max: int | None = None
    multiple_of: int | None = None
    power_of_two: bool = False


@dataclass
class Constant:
    name: str
    define: str
    source: str
    source_line: int
    default: object
    values: list
    op: str
    needs_guard: bool
    kind: str = "macro"
    rules: Rules = field(default_factory=Rules)
    timing: list | None = None
    guard_hint: str | None = None
    note: str | None = None
    extra_flags: list = field(default_factory=list)  # more configure arguments, {value} resolved; e.g. companions a static_assert pairs with this define


@dataclass
class Build:
    source_dir: Path
    build_dir: Path
    target: str
    jobs: int
    configure: list
    build: list
    binary: str


@dataclass
class SweepConfig:
    path: Path
    build: Build
    timing: list
    constants: dict


def _require(table, key, where):
    if key not in table:
        raise SweepError(f"{where}: missing '{key}'")
    return table[key]


def _check_keys(table, allowed, where):
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise SweepError(f"{where}: unknown key(s) {', '.join(unknown)}")


def _table(parent, key, where):
    value = _require(parent, key, where)
    if not isinstance(value, dict):
        raise SweepError(f"{where}: '{key}' must be a table")
    return value


def _string(table, key, where):
    value = _require(table, key, where)
    if not isinstance(value, str) or not value:
        raise SweepError(f"{where}: '{key}' must be a non-empty string")
    return value


def _int(table, key, where):
    value = _require(table, key, where)
    if isinstance(value, bool) or not isinstance(value, int):
        raise SweepError(f"{where}: '{key}' must be an integer")
    return value


def _string_list(table, key, where):
    value = _require(table, key, where)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise SweepError(f"{where}: '{key}' must be a list of strings")
    return list(value)


def _values_list(table, where):
    value = _require(table, "values", where)
    if not isinstance(value, list) or not value:
        raise SweepError(f"{where}: 'values' must not be empty")
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, str)):
            raise SweepError(f"{where}: every value must be an integer or a string")
    return list(value)


def _rules(raw, where):
    _check_keys(raw, RULE_KEYS, where)
    rules = Rules(**raw)
    for name in ("min", "max", "multiple_of"):
        value = getattr(rules, name)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            raise SweepError(f"{where}: '{name}' must be an integer")
    if rules.multiple_of is not None and rules.multiple_of <= 0:
        raise SweepError(f"{where}: 'multiple_of' must be positive")
    if not isinstance(rules.power_of_two, bool):
        raise SweepError(f"{where}: 'power_of_two' must be true or false")
    return rules


def load_config(path):
    path = Path(path).resolve()
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SweepError(f"config not found: {path}")
    except tomllib.TOMLDecodeError as exc:
        raise SweepError(f"{path}: {exc}")

    where = str(path)
    _check_keys(raw, {"build", "timing", "constants"}, where)
    build_raw = _table(raw, "build", where)
    timing_raw = _table(raw, "timing", where)
    constants_raw = _table(raw, "constants", where)
    _check_keys(build_raw, BUILD_KEYS, f"{where}: build")
    _check_keys(timing_raw, {"command"}, f"{where}: timing")

    base = path.parent
    jobs = build_raw.get("jobs", 8)
    if isinstance(jobs, bool) or not isinstance(jobs, int) or jobs < 1:
        raise SweepError(f"{where}: build.jobs must be a positive integer")
    target = build_raw.get("target", "test-backend-ops")
    if not isinstance(target, str) or not target:
        raise SweepError(f"{where}: build.target must be a non-empty string")
    build = Build(
        source_dir=(base / _string(build_raw, "source_dir", f"{where}: build")).resolve(),
        build_dir=(base / _string(build_raw, "build_dir", f"{where}: build")).resolve(),
        target=target,
        jobs=jobs,
        configure=_string_list(build_raw, "configure", f"{where}: build"),
        build=_string_list(build_raw, "build", f"{where}: build"),
        binary=_string(build_raw, "binary", f"{where}: build"),
    )
    if not build.configure or not build.build:
        raise SweepError(f"{where}: build.configure and build.build must not be empty")

    timing = _string_list(timing_raw, "command", f"{where}: timing")
    if not timing:
        raise SweepError(f"{where}: timing.command must not be empty")

    constants = {}
    for name, entry in constants_raw.items():
        where_c = f"{where}: constants.{name}"
        if not isinstance(entry, dict):
            raise SweepError(f"{where_c}: must be a table")
        _check_keys(entry, CONSTANT_KEYS, where_c)
        kind = entry.get("kind", "macro")
        if kind not in ("macro", "constexpr"):
            raise SweepError(f"{where_c}: kind must be 'macro' or 'constexpr', not {kind!r}")
        if not isinstance(entry.get("needs_guard"), bool):
            raise SweepError(f"{where_c}: 'needs_guard' must be true or false")
        constants[name] = Constant(
            name=name,
            define=_string(entry, "define", where_c),
            source=_string(entry, "source", where_c),
            source_line=_int(entry, "source_line", where_c),
            default=_require(entry, "default", where_c),
            values=_values_list(entry, where_c),
            op=_string(entry, "op", where_c),
            needs_guard=entry["needs_guard"],
            kind=kind,
            rules=_rules(_table(entry, "rules", where_c) if "rules" in entry else {}, f"{where_c}.rules"),
            timing=_string_list(entry, "timing", where_c) if "timing" in entry else None,
            guard_hint=entry.get("guard_hint"),
            note=entry.get("note"),
            extra_flags=_string_list(entry, "extra_flags", where_c) if "extra_flags" in entry else [],
        )
    if not constants:
        raise SweepError(f"{where}: no constants defined")
    return SweepConfig(path=path, build=build, timing=timing, constants=constants)


def injectability(constant, build):
    """Whether -D<define>=<value> can reach the kernel, and why not when it cannot."""
    if constant.kind != "macro":
        return False, (
            f"{constant.source}:{constant.source_line} declares {constant.name} as {constant.kind}, not a "
            f"preprocessor macro; a command line cannot replace it"
        )
    source = build.source_dir / constant.source
    if source.is_file():
        text = source.read_text(encoding="utf-8", errors="replace")
        guarded = re.search(
            rf"^[ \t]*#[ \t]*ifndef[ \t]+{re.escape(constant.define)}\b", text, re.MULTILINE
        )
        if guarded:
            return True, f"{constant.source} guards {constant.define} with #ifndef"
        return False, (
            f"{constant.source} defines {constant.define} at line {constant.source_line} without "
            f"#ifndef {constant.define}; the source's #define overrides the command line"
        )
    if constant.needs_guard:
        return False, (
            f"{constant.source} is not under the config's source_dir; the config declares "
            f"needs_guard = true for {constant.define}"
        )
    return True, (
        f"{constant.source} is not under the config's source_dir; trusting needs_guard = false"
    )


def timing_command(config, constant):
    return constant.timing if constant.timing is not None else config.timing


def blockers(config, constant):
    """Every reason this constant cannot be swept as configured, in the order they are printed."""
    reasons = []
    injectable, why = injectability(constant, config.build)
    if not injectable:
        reasons.append(why)
    if not timing_command(config, constant):
        reasons.append(
            f"no timing command: set constants.{constant.name}.timing or [timing].command"
        )
    return reasons


def rule_violation(value, rules):
    if isinstance(value, bool) or not isinstance(value, int):
        return "not an integer"
    if rules.power_of_two and value <= 0:
        return "not a positive power of two"
    if rules.power_of_two and value & (value - 1):
        return "not a power of two"
    if rules.multiple_of is not None and value % rules.multiple_of:
        return f"not a multiple of {rules.multiple_of}"
    if rules.min is not None and value < rules.min:
        return f"below the minimum {rules.min}"
    if rules.max is not None and value > rules.max:
        return f"above the maximum {rules.max}"
    return None


def prune(values, rules):
    kept, pruned = [], []
    for value in values:
        reason = rule_violation(value, rules)
        if reason:
            pruned.append((value, reason))
        else:
            kept.append(value)
    return kept, pruned


def _format(template, mapping, where):
    out = []
    for item in template:
        try:
            out.append(item.format(**mapping))
        except KeyError as exc:
            raise SweepError(
                f"{where}: unknown placeholder {{{exc.args[0]}}} in command template {item!r}"
            )
        except (IndexError, ValueError) as exc:
            raise SweepError(f"{where}: bad command template {item!r}: {exc}")
    return out


def value_commands(config, constant, value):
    """The three command lines a sweep of one value runs, with placeholders resolved."""
    mapping = {
        "source_dir": str(config.build.source_dir),
        "build_dir": str(config.build.build_dir),
        "target": config.build.target,
        "jobs": config.build.jobs,
        "macro": constant.define,
        "value": value,
    }
    mapping["binary"] = _format([config.build.binary], mapping, "build.binary")[0]
    configure = _format(config.build.configure, mapping, "build.configure")
    extra = _format(constant.extra_flags, mapping, f"constants.{constant.name}.extra_flags")
    if extra:
        # companions must ride the SAME CMAKE_CUDA_FLAGS string: a second -DCMAKE_*_FLAGS-looking -D is stored by
        # cmake as an unused cache entry and never reaches nvcc (the MMVQ/MMVF static_assert then still fails)
        for position, word in enumerate(configure):
            if word.startswith("-DCMAKE_CUDA_FLAGS="):
                configure[position] = word + " " + " ".join(extra)
                break
        else:
            raise SweepError(
                f"constants.{constant.name}: extra_flags need a -DCMAKE_CUDA_FLAGS= entry in build.configure to join"
            )
    return {
        "configure": configure,
        "build": _format(config.build.build, mapping, "build.build"),
        "run": _format(timing_command(config, constant), mapping, f"constants.{constant.name}.timing"),
    }


def _select(config, names):
    if not names:
        return list(config.constants.values())
    missing = [name for name in names if name not in config.constants]
    if missing:
        raise SweepError(
            f"unknown constant(s): {', '.join(missing)} (known: {', '.join(config.constants)})"
        )
    return [config.constants[name] for name in names]


def plan(config_path, constants=None, out=None):
    out = sys.stdout if out is None else out
    config = load_config(config_path)
    selected = _select(config, constants)
    blocked = []
    for constant in selected:
        kept, pruned = prune(constant.values, constant.rules)
        print(
            f"== {constant.name} ({constant.op}) default {constant.default} "
            f"at {constant.source}:{constant.source_line}",
            file=out,
        )
        print(f"   values: {', '.join(str(v) for v in kept) if kept else 'none'}", file=out)
        if pruned:
            print("   pruned: " + "; ".join(f"{v} ({reason})" for v, reason in pruned), file=out)
        reasons = blockers(config, constant)
        if reasons:
            blocked.append(constant)
        injectable, why = injectability(constant, config.build)
        print(f"   injectable: {'yes' if injectable else 'no'} - {why}", file=out)
        if not injectable and constant.guard_hint:
            print(f"   guard needed: {constant.guard_hint}", file=out)
        if not timing_command(config, constant):
            print(
                f"   timing: none - no timing command: set constants.{constant.name}.timing "
                f"or [timing].command",
                file=out,
            )
        if constant.note:
            print(f"   note: {constant.note}", file=out)
        for value in kept:
            commands = value_commands(config, constant, value)
            print(f"   [{constant.name}={value}]", file=out)
            for stage in ("configure", "build", "run"):
                if commands[stage]:
                    print(f"   {stage}: {shlex.join(commands[stage])}", file=out)
                else:
                    print(f"   {stage}: (no timing command configured)", file=out)
    if blocked:
        print(
            f"\n{len(blocked)} constant(s) cannot be swept as configured; run refuses them: "
            + ", ".join(c.name for c in blocked),
            file=out,
        )
    return 0


def parse_perf(text):
    """Every perf line the timing binary printed: the case, its runs and its microseconds a run."""
    rows = []
    for raw in text.splitlines():
        match = PERF_LINE.search(ANSI_RE.sub("", raw))
        if not match:
            continue
        op, params, runs, us_per_run, rest = match.groups()
        rows.append(
            {
                "op": op,
                "case": f"{op}({params})",
                "n_runs": int(runs),
                "us_per_run": float(us_per_run),
                "throughput": rest.strip(),
            }
        )
    return rows


def _timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _print_tail(text, out):
    lines = [line for line in (text or "").splitlines() if line.strip()]
    for line in lines[-8:]:
        print(f"   | {line}", file=out)


def _run_value(config, constant, value, runner, out):
    commands = value_commands(config, constant, value)
    record = {
        "timestamp": _timestamp(),
        "config": str(config.path),
        "constant": constant.name,
        "define": constant.define,
        "value": value,
        "default": constant.default,
        "op": constant.op,
        "status": "ok",
        "invalid_reason": None,
        "failed_stage": None,
        "wall_seconds": {},
        "commands": commands,
        "exit_codes": {},
        "perf": [],
        "score_us_per_run": None,
    }
    total = 0.0
    process = None
    for stage in ("configure", "build", "run"):
        print(f"{constant.name}={value} {stage}: {shlex.join(commands[stage])}", file=out)
        started = time.monotonic()
        try:
            process = runner(
                commands[stage], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
        except OSError as error:
            # a missing cmake, an unreadable binary: invalid this value, keep the sweep going
            record["status"] = "invalid"
            record["failed_stage"] = stage
            record["invalid_reason"] = f"{stage} could not run: {error}"
            record["wall_seconds"]["total"] = round(total + (time.monotonic() - started), 3)
            print(f"{constant.name}={value}: invalid ({record['invalid_reason']})", file=out)
            return record
        elapsed = time.monotonic() - started
        total += elapsed
        record["wall_seconds"][stage] = round(elapsed, 3)
        record["exit_codes"][stage] = process.returncode
        if process.returncode != 0:
            record["status"] = "invalid"
            record["failed_stage"] = stage
            record["invalid_reason"] = f"{stage} exited {process.returncode}"
            record["wall_seconds"]["total"] = round(total, 3)
            _print_tail(process.stderr, out)
            print(f"{constant.name}={value}: invalid ({record['invalid_reason']})", file=out)
            return record

    record["perf"] = parse_perf(process.stdout or "")
    record["wall_seconds"]["total"] = round(total, 3)
    if not record["perf"]:
        record["status"] = "invalid"
        record["failed_stage"] = "parse"
        record["invalid_reason"] = "the timing run exited 0 but printed no perf lines"
        _print_tail(process.stdout, out)
        print(f"{constant.name}={value}: invalid ({record['invalid_reason']})", file=out)
        return record

    record["score_us_per_run"] = round(
        sum(row["us_per_run"] for row in record["perf"]), 3
    )
    print(
        f"{constant.name}={value}: ok score={record['score_us_per_run']} us/run "
        f"({len(record['perf'])} case(s))",
        file=out,
    )
    return record


def _new_results_file(directory):
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = directory / f"{stamp}.jsonl"
    counter = 1
    while path.exists():
        path = directory / f"{stamp}-{counter}.jsonl"
        counter += 1
    return path


def _write_record(path, record):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()


def run(config_path, results_dir="results", constants=None, runner=subprocess.run, out=None):
    out = sys.stdout if out is None else out
    config = load_config(config_path)
    selected = _select(config, constants)

    refused = [(constant, blockers(config, constant)) for constant in selected]
    refused = [(constant, reasons) for constant, reasons in refused if reasons]
    if refused:
        print("refused: the sweep would measure defaults, not the values; no command was run.", file=out)
        for constant, reasons in refused:
            print(f"  {constant.name}:", file=out)
            for reason in reasons:
                print(f"    - {reason}", file=out)
            if constant.guard_hint:
                print(f"    guard needed: {constant.guard_hint}", file=out)
        return 2

    results = Path(results_dir)
    results.mkdir(parents=True, exist_ok=True)
    path = _new_results_file(results)
    path.touch()
    print(f"results: {path}", file=out)

    any_ok = False
    for constant in selected:
        kept, pruned = prune(constant.values, constant.rules)
        for value, reason in pruned:
            print(f"pruned {constant.name}={value}: {reason}", file=out)
        for value in kept:
            record = _run_value(config, constant, value, runner, out)
            _write_record(path, record)
            any_ok = any_ok or record["status"] == "ok"
    return 0 if any_ok else 1


def load_records(directory):
    directory = Path(directory)
    paths = sorted(directory.glob("*.jsonl"))
    records = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise SweepError(f"no such results file: {path}")
        for number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SweepError(f"{path}:{number}: {exc}")
    return records, paths


def _delta(score, base):
    if score is None or base is None or base == 0:
        return "n/a"
    return f"{(score / base - 1.0) * 100.0:+.1f}%"


def _value_sort_key(value):
    return (isinstance(value, str), value)


def report(directory, out=None):
    out = sys.stdout if out is None else out
    records, paths = load_records(directory)
    if not records:
        print(f"no records in {Path(directory)}", file=out)
        return 1
    print(f"# {len(records)} record(s) from {len(paths)} file(s) in {Path(directory)}", file=out)

    latest = {}
    for record in records:
        key = (record["constant"], record["value"])
        previous = latest.get(key)
        if previous is None or str(record.get("timestamp", "")) >= str(previous.get("timestamp", "")):
            latest[key] = record

    by_constant = {}
    for (name, value), record in latest.items():
        by_constant.setdefault(name, {})[value] = record

    header = f"{'constant':<28} {'value':>6} {'status':<8} {'score_us':>10} {'vs_default':>10}  best"
    print(header, file=out)
    for name in sorted(by_constant):
        rows = by_constant[name]
        default_value = next(iter(rows.values()))["default"]
        default_record = rows.get(default_value)
        default_score = None
        if default_record is not None and default_record.get("status") == "ok":
            default_score = default_record.get("score_us_per_run")
        scores = [
            record["score_us_per_run"]
            for record in rows.values()
            if record.get("status") == "ok" and record.get("score_us_per_run") is not None
        ]
        best_score = min(scores) if scores else None
        for value in sorted(rows, key=_value_sort_key):
            record = rows[value]
            status = record.get("status", "?")
            score = record.get("score_us_per_run") if status == "ok" else None
            if value == default_value and default_score is not None:
                versus = "default"
            else:
                versus = _delta(score, default_score)
            best = "*" if score is not None and score == best_score else ""
            print(
                f"{name:<28} {str(value):>6} {status:<8} "
                f"{(f'{score:.3f}' if score is not None else '-'):>10} {versus:>10}  {best}",
                file=out,
            )
        if best_score is None:
            print(f"best {name}: none (no valid value)", file=out)
        else:
            best_value = min(
                (
                    value
                    for value, record in rows.items()
                    if record.get("status") == "ok"
                    and record.get("score_us_per_run") == best_score
                ),
                key=_value_sort_key,
            )
            print(
                f"best {name}: {best_value} (score {best_score:.3f} us/run, "
                f"{_delta(best_score, default_score)} vs default {default_value})",
                file=out,
            )
    return 0


def main(argv=None, runner=subprocess.run):
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Sweep compile-time kernel constants as build defines and time them at the serving shapes.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan", help="print the commands a sweep would run")
    plan_parser.add_argument("config")
    plan_parser.add_argument("--constant", action="append", dest="constants", metavar="NAME")

    run_parser = subparsers.add_parser("run", help="run the sweep and record one JSONL line per value")
    run_parser.add_argument("config")
    run_parser.add_argument("--results", default="results", metavar="DIR")
    run_parser.add_argument("--constant", action="append", dest="constants", metavar="NAME")

    report_parser = subparsers.add_parser("report", help="best value per constant and its delta")
    report_parser.add_argument("directory", metavar="DIR")

    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            return plan(args.config, constants=args.constants)
        if args.command == "run":
            return run(args.config, args.results, args.constants, runner=runner)
        return report(args.directory)
    except SweepError as exc:
        print(f"{PROG}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
