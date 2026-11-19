#!/usr/bin/env python3
"""The knobs of tunables.toml.

  tunables.py gen     write ggml/src/ggml-tunables.inc, the table the engine reads knobs through
  tunables.py check   the table is current, every variable the source reads is declared, each knob's
                      `available` says how the source reads it, and every flag and option named exists
  tunables.py list    the params, one line each (--available, --scope, --kind, --layer, --set, --json)
  tunables.py status  how many params a search can set, by scope, kind and how they are set
"""

import argparse
import fnmatch
import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOML = ROOT / "tunables.toml"
INC = ROOT / "ggml/src/ggml-tunables.inc"

SETS = ("env", "cli", "build", "code")
TYPES = ("bool", "int", "float", "str")
KINDS = ("knob", "legacy", "diag", "setup")
SCOPES = ("shared", "inkling2", "other")
LAYERS = ("cuda", "cpu", "ggml", "llama", "model", "server", "common", "build")
OUTPUTS = ("same", "rounding", "model")
KEYS = {"name", "set", "type", "default", "min", "max", "values", "kind", "scope", "layer", "output", "where", "about",
        "launcher", "indirect", "available"}

# the source a knob can be read from: the engine as this fork serves it (other backends keep their own variables)
SOURCE_DIRS = ("ggml/include/", "ggml/src/ggml-cuda/", "ggml/src/ggml-cpu/", "src/", "common/", "tools/", "include/")
SOURCE_EXTS = (".c", ".cpp", ".h", ".hpp", ".cu", ".cuh")
REGISTRY = "ggml/src/ggml-tunables.cpp"  # reads its own two variables before any table exists
ARGS = ROOT / "common/arg.cpp"
LAUNCHER = ROOT / "scripts/inkling2/serve.py"

CHECKED = re.compile(r'\b(ggml_env_(?:switch|int|float)_or|ggml_env_(?:switch|int|float|str))\s*\(\s*"([A-Z0-9_]+)"')
RAW = re.compile(r'\b(getenv|getenv_s|common_get_env|dsv41_env_on|digit)\s*\(\s*"((?:GGML|LLAMA)_[A-Z0-9_]+)"')

READER_TYPE = {"switch": "bool", "int": "int", "float": "float", "str": "str"}


def fail(msg):
    print(f"tunables: {msg}", file=sys.stderr)
    sys.exit(1)


def load():
    with open(TOML, "rb") as f:
        doc = tomllib.load(f)
    params = doc.get("param", [])
    seen = set()
    for p in params:
        name = p.get("name")
        where = f"param {name!r}"
        if not name or name in seen:
            fail(f"{where}: a param needs a name of its own")
        seen.add(name)
        if set(p) - KEYS:
            fail(f"{where}: unknown keys {', '.join(sorted(set(p) - KEYS))}")
        for key, allowed in (("set", SETS), ("type", TYPES), ("kind", KINDS), ("scope", SCOPES), ("layer", LAYERS)):
            if p.get(key) not in allowed:
                fail(f"{where}: {key} is {p.get(key)!r}, not one of {', '.join(allowed)}")
        if "output" in p and p["output"] not in OUTPUTS:
            fail(f"{where}: output is {p['output']!r}, not one of {', '.join(OUTPUTS)}")
        if not isinstance(p.get("available"), bool):
            fail(f"{where}: available is true or false")
        if p["set"] == "env" and p["available"]:
            check_declared(p, where)
        if p["set"] == "code" and (p["available"] or "where" not in p):
            fail(f"{where}: a constant in the code is not available, and says where it is")
        if p["set"] in ("cli", "build") and not p["available"]:
            fail(f"{where}: a flag or a build option that exists is available")
    return doc, params


def check_declared(p, where):
    """A knob the engine checks says what unset means and what it accepts."""
    t = p["type"]
    if "default" not in p and t != "str":
        fail(f"{where}: an available knob declares its default (or \"auto\")")
    d = p.get("default")
    auto = d == "auto"
    if t == "bool" and not auto and not isinstance(d, bool):
        fail(f"{where}: the default of a bool is true, false or \"auto\"")
    if t in ("int", "float"):
        number = (int,) if t == "int" else (int, float)
        if not auto and (not isinstance(d, number) or isinstance(d, bool)):
            fail(f"{where}: the default of {t} is a number or \"auto\"")
        if "values" in p:
            if t != "int" or not p["values"] or not all(isinstance(v, int) for v in p["values"]):
                fail(f"{where}: values is a list of whole numbers")
            if not auto and d not in p["values"]:
                fail(f"{where}: the default {d} is not one of its values")
        else:
            if not all(isinstance(p.get(k), number) and not isinstance(p.get(k), bool) for k in ("min", "max")):
                fail(f"{where}: {t} needs min and max, or values")
            if p["min"] > p["max"] or (not auto and not p["min"] <= d <= p["max"]):
                fail(f"{where}: min <= default <= max does not hold")
            if t == "int" and max(abs(p["min"]), abs(p["max"])) > 2**53:
                fail(f"{where}: the range is beyond 2^53")
    if t == "str" and "values" in p and not all(isinstance(v, str) and "," not in v for v in p["values"]):
        fail(f"{where}: values is a list of words")


def c_number(v, t):
    if t == "int" or isinstance(v, int):
        return f"{int(v)}.0"
    return repr(float(v))


def render(doc, params):
    lines = [
        "// Generated from tunables.toml by scripts/tunables.py gen. Do not edit: change tunables.toml and generate again.",
        "// GGML_TUNABLE(name, type, flags, default, min, max, values), sorted by name.",
    ]
    for p in sorted((p for p in params if p["set"] == "env"), key=lambda p: p["name"]):
        t = p["type"]
        flags, d, lo, hi, values = [], "0.0", "0.0", "0.0", "nullptr"
        if not p["available"]:
            flags.append("RAW")
        else:
            default = p.get("default")
            if default == "auto":
                flags.append("AUTO")
            elif t == "bool":
                d = "1.0" if default else "0.0"
            elif t in ("int", "float"):
                d = c_number(default, t)
            if "values" in p:
                values = '"' + ",".join(str(v) for v in p["values"]) + '"'
            elif t in ("int", "float"):
                lo, hi = c_number(p["min"], t), c_number(p["max"], t)
        lines.append(f'GGML_TUNABLE("{p["name"]}", {t.upper()}, {" | ".join(flags) or "0"}, {d}, {lo}, {hi}, {values})')
    for prefix in doc.get("ignore_prefixes", []):
        lines.append(f'GGML_TUNABLE_IGNORE("{prefix}")')
    return "\n".join(lines) + "\n"


def tree_files(*patterns):
    """The tree's files a pattern names, as git matches one: a folder by its prefix, a * across folders.
    Git's own list in a checkout. A tree shipped without its repository is walked, less its build folders."""
    listed = subprocess.run(["git", "ls-files", "-z", "--", *patterns], cwd=ROOT, capture_output=True, text=True)
    if listed.returncode == 0:
        return [f for f in listed.stdout.split("\0") if f]
    found = []
    for folder, dirs, names in os.walk(ROOT):
        top = Path(folder) == ROOT
        dirs[:] = [d for d in dirs if not d.startswith(".") and not (top and d.startswith("build"))]
        for name in names:
            rel = (Path(folder) / name).relative_to(ROOT).as_posix()
            if any(rel.startswith(pat) if pat.endswith("/") else fnmatch.fnmatchcase(rel, pat) for pat in patterns):
                found.append(rel)
    return found


def scan():
    """name -> {"checked": {(reader, file)}, "raw": {file}} for every variable the source reads by name."""
    files = tree_files(*SOURCE_DIRS, "ggml/src/*.c", "ggml/src/*.cpp", "ggml/src/*.h")
    reads = {}
    for rel in sorted(set(files)):
        if not rel.endswith(SOURCE_EXTS) or rel == REGISTRY or (rel.startswith("ggml/src/") and rel.count("/") > 2 and not rel.startswith(SOURCE_DIRS)):
            continue
        text = (ROOT / rel).read_text(errors="replace")
        for reader, name in CHECKED.findall(text):
            reads.setdefault(name, {"checked": set(), "raw": set()})["checked"].add((reader, rel))
        for _, name in RAW.findall(text):
            reads.setdefault(name, {"checked": set(), "raw": set()})["raw"].add(rel)
    return reads


def check(doc, params):
    problems = []
    if not INC.exists() or INC.read_text() != render(doc, params):
        problems.append(f"{INC.relative_to(ROOT)} is not what tunables.toml gives: run scripts/tunables.py gen")

    ignore = tuple(doc.get("ignore_prefixes", []))
    env = {p["name"]: p for p in params if p["set"] == "env"}
    reads = scan()

    for name, r in sorted(reads.items()):
        if name.startswith(ignore):
            continue
        p = env.get(name)
        if p is None:
            if r["raw"] or name.startswith(("GGML_", "LLAMA_")):
                problems.append(f"{name} is read in {', '.join(sorted(r['raw'] | {f for _, f in r['checked']}))} and tunables.toml does not declare it")
            continue
        available = bool(r["checked"]) and not r["raw"]
        if p["available"] != available:
            how = "through the registry" if available else f"with getenv in {', '.join(sorted(r['raw']))}" if r["raw"] else "nowhere"
            problems.append(f"{name}: available = {str(p['available']).lower()}, and the source reads it {how}")
        for reader, rel in sorted(r["checked"]):
            kind = reader.removeprefix("ggml_env_").removesuffix("_or")
            if READER_TYPE[kind] != p["type"]:
                problems.append(f"{name}: declared {p['type']} and read with {reader} in {rel}")
            if p["available"] and reader.endswith("_or") != (p.get("default") == "auto"):
                problems.append(f"{name}: default {p.get('default')!r} and read with {reader} in {rel}: default \"auto\" goes with the _or readers")
    for name, p in sorted(env.items()):
        if name not in reads and not p.get("indirect", False):
            problems.append(f"{name} is declared and the source reads it nowhere by name (indirect = true if a table names it)")

    args = ARGS.read_text()
    launcher = LAUNCHER.read_text()
    cmake = "".join((ROOT / f).read_text(errors="replace") for f in tree_files("CMakeLists.txt", "*/CMakeLists.txt", "*.cmake"))
    for p in params:
        if p["set"] == "cli" and f'"{p["name"]}"' not in args:
            problems.append(f"{p['name']}: {ARGS.relative_to(ROOT)} has no such flag")
        if p["set"] == "build" and not re.search(rf"\b{re.escape(p['name'])}\b", cmake):
            problems.append(f"{p['name']}: no CMake file names this option")
        if "launcher" in p and f'"{p["launcher"]}"' not in launcher:
            problems.append(f"{p['name']}: {LAUNCHER.relative_to(ROOT)} has no flag {p['launcher']}")

    for msg in problems:
        print(msg, file=sys.stderr)
    if problems:
        fail(f"{len(problems)} problems")
    n = sum(1 for p in params if p["available"])
    print(f"tunables: {len(params)} params, {n} available, {len(params) - n} not yet")


def select(params, args):
    out = []
    for p in params:
        if args.available and not p["available"]:
            continue
        if any(getattr(args, k) and p[k] not in getattr(args, k).split(",") for k in ("scope", "kind", "layer", "set")):
            continue
        out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("gen", "check", "list", "status"))
    ap.add_argument("--available", action="store_true", help="only the knobs a search can set")
    for key in ("scope", "kind", "layer", "set"):
        ap.add_argument(f"--{key}", help="comma list")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    doc, params = load()
    if args.command == "gen":
        INC.write_text(render(doc, params))
    elif args.command == "check":
        check(doc, params)
    elif args.command == "list":
        chosen = select(params, args)
        if args.json:
            reads = scan()
            for p in chosen:
                r = reads.get(p["name"]) if p["set"] == "env" else None
                if r:
                    p["read_in"] = sorted(r["raw"] | {f for _, f in r["checked"]})
            json.dump(chosen, sys.stdout, indent=1)
            print()
            return
        for p in chosen:
            if "values" in p:
                span = "{" + ",".join(str(v) for v in p["values"]) + "}"
            elif "min" in p:
                span = f"{p['min']}..{p['max']}"
            else:
                span = ""
            print(f"{'[x]' if p['available'] else '[ ]'} {p['name']:<44} {p['set']:<5} {p['type']:<5} {str(p.get('default', '')):<8} {span:<22} {p['scope']:<8} {p['kind']:<6} {p['layer']}")
    else:
        rows = {}
        for p in params:
            row = rows.setdefault((p["scope"], p["kind"], p["set"]), [0, 0])
            row[0] += p["available"]
            row[1] += 1
        for (scope, kind, how), (n, total) in sorted(rows.items()):
            print(f"{scope:<9} {kind:<9} {how:<6} {n:>4} of {total}")
        print(f"{'all':<26} {sum(n for n, _ in rows.values()):>4} of {len(params)}")


if __name__ == "__main__":
    main()
