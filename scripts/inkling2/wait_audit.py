#!/usr/bin/env python3
"""Decode-path blocking-wait audit for campaign row inkling-counter-sync.

Every blocking synchronization wait and ordering-only stream fork-join wait on the
decode path is inventoried in reports/inkling2-wait-audit.json with one of three
dispositions:
  - counter:        ordering-only wait replaced with a device-side arrival counter
                    (producer release atomic/store, consumer acquire spin; host never blocks)
  - kept-host-data: real wait where the host CPU reads or writes data (host_data required)
  - out-of-path:    init, teardown, fallback, diagnostic, or non-Inkling-decode site

Usage:
  wait_audit.py --check --evidence reports/inkling2-wait-audit.json
  wait_audit.py --scan [--root .]
"""
import argparse
import json
import pathlib
import re
import sys

VALID_DISPOSITIONS = {"counter", "kept-host-data", "out-of-path"}
SYNC_CALL_RE = re.compile(
    r"\b(cudaStreamSynchronize|cudaEventSynchronize|cudaDeviceSynchronize|"
    r"ggml_backend_synchronize|ggml_backend_event_synchronize|ggml_backend_sched_synchronize)\s*\("
)
HOT_SCAN_FILES = [
    "ggml/src/ggml-cuda/allreduce.cu",
    "ggml/src/ggml-cuda/ggml-cuda.cu",
    "ggml/src/ggml-cuda/batch-copy.cu",
    "ggml/src/ggml-backend-meta.cpp",
    "src/llama-context.cpp",
    "src/llama-kv-cache.cpp",
    "src/llama-kv-cache-kpool.cpp",
    "src/llama-moe-cache.cpp",
    "src/llama-model.cpp",
]


def scan_sync_calls(root: pathlib.Path):
    """Scan repo files for host-blocking synchronization call sites."""
    hits = []
    for rel in HOT_SCAN_FILES:
        path = root / rel
        if not path.is_file():
            continue
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        if0_depth = 0
        for lineno, text in enumerate(lines, start=1):
            stripped = text.strip()
            if stripped == "#if 0":
                if0_depth += 1
                continue
            if if0_depth > 0:
                if stripped.startswith("#if"):
                    if0_depth += 1
                elif stripped.startswith("#endif"):
                    if0_depth -= 1
                continue
            if stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("*"):
                continue
            m = SYNC_CALL_RE.search(text)
            if m:
                hits.append({
                    "file": rel,
                    "line": lineno,
                    "api": m.group(1),
                    "text": stripped,
                })
    return hits


def check(evidence: dict, root: pathlib.Path):
    problems = []
    required_top = ("version", "row", "build", "measured_at", "summary", "sites", "verification")
    for field in required_top:
        if field not in evidence:
            problems.append(f"missing top-level field {field!r}")
    if problems:
        return problems

    if evidence["row"] != "inkling-counter-sync":
        problems.append(f"row is {evidence['row']!r}, expected 'inkling-counter-sync'")
    if not str(evidence["build"]).strip():
        problems.append("build is empty")
    if not isinstance(evidence["sites"], list) or not evidence["sites"]:
        problems.append("sites must be a non-empty list")
        return problems

    seen_ids = set()
    counts = {"counter": 0, "kept-host-data": 0, "out-of-path": 0}
    hot_counts = {"counter": 0, "kept-host-data": 0}
    file_cache = {}
    audited_locations = {}

    for idx, site in enumerate(evidence["sites"]):
        tag = f"sites[{idx}]"
        if not isinstance(site, dict):
            problems.append(f"{tag}: entry must be an object")
            continue
        for key in ("id", "file", "line", "symbol", "api", "hot_decode_path", "disposition", "reason"):
            if key not in site:
                problems.append(f"{tag}: missing {key!r}")
        if any(k not in site for k in ("id", "file", "line", "symbol", "api", "hot_decode_path", "disposition", "reason")):
            continue

        sid = site["id"]
        if sid in seen_ids:
            problems.append(f"{tag}: duplicate id {sid!r}")
        seen_ids.add(sid)

        disp = site["disposition"]
        if disp == "barrier":
            problems.append(f"{sid}: barrier disposition remains on decode path")
            continue
        if disp not in VALID_DISPOSITIONS:
            problems.append(f"{sid}: invalid disposition {disp!r} (expected one of {sorted(VALID_DISPOSITIONS)})")
            continue

        hot = site["hot_decode_path"]
        if not isinstance(hot, bool):
            problems.append(f"{sid}: hot_decode_path must be boolean")
            continue

        if hot and disp == "out-of-path":
            problems.append(f"{sid}: hot_decode_path=true cannot have disposition 'out-of-path'")

        if disp == "kept-host-data":
            host_data = str(site.get("host_data", "")).strip()
            if not host_data:
                problems.append(f"{sid}: kept-host-data site must name the host_data read or written")

        if disp == "counter":
            mech = str(site.get("counter_mechanism", "")).strip()
            if not mech:
                problems.append(f"{sid}: counter site must describe counter_mechanism")

        rel = site["file"]
        line = site["line"]
        if not isinstance(line, int) or line < 1:
            problems.append(f"{sid}: invalid line number {line!r}")
            continue

        fpath = root / rel
        if not fpath.is_file():
            problems.append(f"{sid}: source file {rel!r} does not exist")
            continue
        if rel not in file_cache:
            file_cache[rel] = fpath.read_text(encoding="utf-8", errors="replace").splitlines()
        lines = file_cache[rel]
        if line > len(lines):
            problems.append(f"{sid}: line {line} exceeds {rel} length ({len(lines)})")
            continue

        # Verify source anchor within +-12 lines of recorded line
        lo = max(0, line - 13)
        hi = min(len(lines), line + 12)
        window = "\n".join(lines[lo:hi])
        anchor = str(site.get("anchor", site["api"])).strip()
        if anchor and anchor not in window:
            problems.append(f"{sid}: anchor {anchor!r} not found near {rel}:{line}")

        counts[disp] += 1
        if hot:
            hot_counts[disp] = hot_counts.get(disp, 0) + 1
        audited_locations.setdefault(rel, []).append((line, site.get("line_end", line)))

    if counts["counter"] == 0:
        problems.append("no ordering-only wait has disposition 'counter'")
    if hot_counts["counter"] == 0:
        problems.append("no hot_decode_path site was converted to 'counter'")

    # Cross-check against live scan of allreduce.cu and ggml-cuda.cu so no host wait is silently omitted
    for hit in scan_sync_calls(root):
        if hit["file"] not in ("ggml/src/ggml-cuda/allreduce.cu", "ggml/src/ggml-cuda/ggml-cuda.cu",
                               "ggml/src/ggml-cuda/batch-copy.cu", "ggml/src/ggml-backend-meta.cpp"):
            continue
        ranges = audited_locations.get(hit["file"], [])
        covered = any((a - 5) <= hit["line"] <= (b + 5) for a, b in ranges)
        if not covered:
            problems.append(
                f"unaudited sync call at {hit['file']}:{hit['line']} ({hit['api']}): {hit['text']}"
            )

    summary = evidence["summary"]
    if not isinstance(summary, dict):
        problems.append("summary must be an object")
    else:
        for key in ("counter", "kept-host-data", "out-of-path"):
            if summary.get(key) != counts[key]:
                problems.append(f"summary[{key!r}]={summary.get(key)} disagrees with counted {counts[key]}")
        if summary.get("barrier", 0) != 0:
            problems.append(f"summary['barrier']={summary.get('barrier')} must be 0")

    verif = evidence["verification"]
    if not isinstance(verif, dict) or not verif.get("tests"):
        problems.append("verification.tests must record executed verification checks")

    return problems


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="judge an evidence JSON file")
    parser.add_argument("--scan", action="store_true", help="list host-blocking sync call sites in source")
    parser.add_argument("--evidence", default="reports/inkling2-wait-audit.json")
    parser.add_argument("--root", default=".", help="repository root")
    args = parser.parse_args(argv)

    root = pathlib.Path(args.root).resolve()
    if args.scan:
        hits = scan_sync_calls(root)
        for h in hits:
            print(f"{h['file']}:{h['line']} [{h['api']}] {h['text']}")
        print(f"total: {len(hits)} sites")
        return 0

    if args.check:
        ev_path = pathlib.Path(args.evidence)
        if not ev_path.is_absolute():
            ev_path = root / ev_path
        try:
            evidence = json.loads(ev_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"FAIL: cannot read evidence at {args.evidence}: {exc}")
            return 1
        problems = check(evidence, root)
        if problems:
            for p in problems:
                print(f"FAIL: {p}")
            return 1
        s = evidence["summary"]
        print(
            f"ok: {len(evidence['sites'])} audited sites "
            f"(counter={s['counter']}, kept-host-data={s['kept-host-data']}, "
            f"out-of-path={s['out-of-path']}, barrier=0)"
        )
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
