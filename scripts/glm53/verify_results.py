"""Qualification accounting for the pinned four-category sampled GLM workload.

This validates recorded responses; it does not start a server or rent compute.
"""
import argparse
import json
import math
from pathlib import Path

CATEGORIES = {"english", "chinese", "code", "tools"}

def _validate(rows, fixture_hashes):
    expected = {(p, c, r) for p in (0, 1) for c in CATEGORIES for r in range(3)}
    seen = set()
    for row in rows:
        try:
            key = row["pair"], row["category"], row["repeat"]
            timings = row["response"]["timings"]
            valid = (key in expected and key not in seen
                     and row["prompt_sha256"] == fixture_hashes[row["category"]]
                     and type(timings["predicted_n"]) is int and timings["predicted_n"] == 384
                     and math.isfinite(timings["predicted_ms"]) and timings["predicted_ms"] > 0)
        except (KeyError, TypeError, ValueError):
            valid = False
        if not valid:
            raise ValueError("incomplete, changed or invalid sampled workload")
        seen.add(key)
    if seen != expected:
        raise ValueError("exactly 24 unique requests are required for each arm")

def summarize(rows):
    tokens = sum(row["response"]["timings"]["predicted_n"] - 1 for row in rows)
    ms = math.fsum(row["response"]["timings"]["predicted_ms"] for row in rows)
    return dict(requests=len(rows), steady_tokens=tokens, decode_ms=ms, tok_s=tokens*1000/ms)

def evaluate(baseline, candidate, fixture_hashes, minimum_rate=200.0):
    if set(fixture_hashes) != CATEGORIES:
        raise ValueError("the four pinned fixture categories are required")
    for rows in (baseline, candidate):
        _validate(rows, fixture_hashes)
    base, cand = summarize(baseline), summarize(candidate)
    failures = []
    # Ignore only floating-point summation roundoff at the exact boundary.
    if cand["tok_s"] < minimum_rate and not math.isclose(cand["tok_s"], minimum_rate, rel_tol=1e-12):
        failures.append(f"combined rate below {minimum_rate:g} tok/s")
    categories = {}
    for category in sorted(CATEGORIES):
        a = summarize([r for r in baseline if r["category"] == category])
        b = summarize([r for r in candidate if r["category"] == category])
        change = 100*(b["tok_s"]/a["tok_s"] - 1)
        categories[category] = dict(baseline=a, candidate=b, change_percent=change)
        if change < -2 and not math.isclose(change, -2, abs_tol=1e-10):
            failures.append(f"{category} regressed more than 2%")
    pairs = []
    for pair in (0, 1):
        a = summarize([r for r in baseline if r["pair"] == pair])
        b = summarize([r for r in candidate if r["pair"] == pair])
        pairs.append(dict(baseline=a, candidate=b))
        if b["tok_s"] <= a["tok_s"]:
            failures.append(f"pair {pair} did not improve")
    return dict(passed=not failures, failures=failures, baseline=base, candidate=cand,
                categories=categories, pairs=pairs)

def validate_routes(record):
    try:
        rows, hist = record["rows"], record["multiplicity"]
        counts = [record[k] for k in ("calls", "pairs", "unique", "reused")]
        if len(rows) != 8 or len(hist) != 64 or any(type(v) is not int or v < 0 for v in counts + rows + hist):
            return False
        calls, pairs, unique, reused = counts
        return (calls > 0 and sum(rows) == calls and sum(hist) == unique
                and sum((i+1)*v for i, v in enumerate(hist)) == pairs
                and sum(hist[2:]) == reused)
    except (KeyError, TypeError):
        return False

def load_rows(paths):
    if len(paths) != 2:
        raise ValueError("two opposite-order legs per arm are required")
    result = []
    for pair, path in enumerate(paths):
        for line in (Path(path)/"responses.jsonl").read_text().splitlines():
            row = json.loads(line)
            row["pair"] = pair
            result.append(row)
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", nargs=2, required=True)
    parser.add_argument("--candidate", nargs=2, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--companion", action="store_true", help="report the companion rate without the primary 200 tok/s threshold")
    args = parser.parse_args()
    hashes = {v["category"]: v["prompt_sha256"] for v in json.loads(args.inputs.read_text())["fixtures"]}
    result = evaluate(load_rows(args.baseline), load_rows(args.candidate), hashes, 0 if args.companion else 200)
    print(json.dumps(result, indent=2, allow_nan=False))
    raise SystemExit(0 if result["passed"] else 1)
