#!/usr/bin/env python3
"""Ask a llama-server to fix a small repository at a specified prompt depth.

Run on a Linux client with git, bubblewrap and /usr/bin/python3. Point --url at
the inference server or a local SSH port forward. All grading runs on the client.
"""

import argparse
import difflib
import hashlib
import io
import json
import os
import random
import re
import shutil
import struct
import subprocess
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path


TARGET = "daily_units.py"
MARKER = "@@ROUTE_ARCHIVE_BODY@@"
WRITER = '''"""Packets emitted by the batch writer."""

TAG = {"idle": 3, "local": 0, "express": 2}


def pack(spans):
    previous_end = 0
    for start, end, service in spans:
        if start < previous_end or start >= end:
            raise ValueError("invalid span")
        yield (start - previous_end, ((end - start - 1) << 2) | TAG[service])
        previous_end = end
'''
POLICY = '''"""Current rate card. Idle time is free."""

UNITS_PER_TICK = {"idle": 0, "local": 3, "express": 7}
'''
STARTER = '''"""Usage totals for the reporting endpoint."""

from rate_card import UNITS_PER_TICK


def units_between(packets, start, stop):
    """Return billed units for the half-open window [start, stop)."""
    raise NotImplementedError("packet billing is missing")
'''
REFERENCE = STARTER.replace('    raise NotImplementedError("packet billing is missing")', '''    cursor = 0
    total = 0
    names = {3: "idle", 0: "local", 2: "express"}
    for gap, word in packets:
        first = cursor + gap
        cursor = first + (word >> 2) + 1
        overlap = max(0, min(stop, cursor) - max(start, first))
        total += overlap * UNITS_PER_TICK[names[word & 3]]
    return total''')
REQUEST = (
    "Implement units_between in daily_units.py for the wire format emitted by "
    "packet_writer.pack and the current rate card. packets is a one-pass iterable of valid "
    "packets. Bill only intersecting ticks. Gaps and empty or reversed windows cost zero. "
    "Keep the public API, use one pass and bounded extra space. "
    "Reply with one unified diff in a fenced diff block. Only daily_units.py may change."
)
SOURCE = (
    "Review the following shipping ledger repository and fix the billing endpoint.\n\n"
    "File: packet_writer.py\n```python\n" + WRITER + "```\n\n"
    "File: archived_route_samples.tsv (generated historical routing data)\n"
    "route\torigin\tdestination\tprior_distance\tsample_count\n" + MARKER +
    "\nEnd of archived_route_samples.tsv\n\nFile: rate_card.py\n```python\n" +
    POLICY + "```\n\nFile: daily_units.py\n```python\n" + STARTER +
    "```\n\nIssue from the reporting team:\n" + REQUEST
)


def post(url, route, body, timeout):
    req = urllib.request.Request(url.rstrip("/") + route, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            return json.load(res)
    except urllib.error.HTTPError as exc:
        detail = exc.read(4096).decode(errors="replace")
        raise RuntimeError(f"{route} returned HTTP {exc.code}: {detail}") from exc


def build_prompt(args, call):
    def messages(archive):
        return [{"role": "user", "content": SOURCE.replace(MARKER, archive)}]

    def render(chat):
        return call("/apply-template", {"messages": chat, "reasoning_effort": args.reasoning_effort})["prompt"]

    def tokenize(text, special=False):
        return call("/tokenize", {"content": text, "add_special": special, "parse_special": True})["tokens"]

    left, right = render(messages(MARKER)).split(MARKER)
    prefix, suffix = tokenize(left, True), tokenize(right)
    budget = args.prompt_tokens - len(prefix) - len(suffix)
    if not prefix or not suffix or budget < 1:
        raise ValueError("prompt budget cannot fit the two source sections and the archive")
    chunks = []
    rng = random.Random(53172)
    row = 0
    remaining = budget
    while remaining > 64:
        rows = []
        for _ in range(1024):
            rows.append(f"R{row:07d}\tH{rng.randrange(491):03d}\tH{rng.randrange(491):03d}"
                        f"\t{rng.randrange(1, 1401)}\t{rng.randrange(1, 250)}\n")
            row += 1
        block = "".join(rows)
        count = len(tokenize(block))
        if not count:
            raise ValueError("empty tokenization for archive")
        if count >= remaining - 64:
            chunks.append(block[:len(block) * (remaining - 64) // count])
            break
        chunks.append(block)
        remaining -= count

    archive = "".join(chunks)
    ids = tokenize(left + archive + right, True)
    while len(ids) > args.prompt_tokens and archive:
        archive = archive[:max(0, len(archive) - 4 * (len(ids) - args.prompt_tokens))]
        ids = tokenize(left + archive + right, True)
    if len(ids) != args.prompt_tokens:
        # Try short ASCII suffixes against this tokenizer, then verify the full native template.
        gap = args.prompt_tokens - len(ids)
        tail = (left + archive)[-256:]
        base = len(tokenize(tail + right))
        for unit in (" x", " 7", "!", "q", ".", "\n"):
            def increase(n):
                return len(tokenize(tail + unit * n + right)) - base

            low, high = 0, gap
            change = increase(high)
            if change < gap:
                while high < gap * 32 and change < gap:
                    low, high = high, high * 2
                    change = increase(high)
            if change < gap:
                continue
            if change > gap:
                while high - low > 1:
                    mid = (low + high) // 2
                    if increase(mid) >= gap:
                        high = mid
                    else:
                        low = mid
            for offset in (0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5):
                n = high + offset
                if n < 1:
                    continue
                trial = tokenize(left + archive + unit * n + right, True)
                if len(trial) == args.prompt_tokens:
                    archive += unit * n
                    ids = trial
                    break
            if len(ids) == args.prompt_tokens:
                break
    if len(ids) != args.prompt_tokens:
        raise ValueError("could not align the full prompt to the requested token budget")
    chat = messages(archive)
    if render(chat) != left + archive + right:
        raise RuntimeError("the chat template changed while inserting the archive")
    digest = hashlib.sha256(struct.pack(f"<{len(ids)}I", *ids)).hexdigest()
    return chat, {"prompt_tokens": len(ids), "early_section_tokens_approx": len(prefix),
                  "late_section_start_approx": len(ids) - len(suffix), "archive_tokens_approx": budget,
                  "token_ids_le_u32_sha256": digest}


class PatchError(ValueError):
    pass


def extract_patch(answer):
    if len(answer) > 256 * 1024:
        raise PatchError("answer exceeds patch size limit")
    blocks = re.findall(r"^```(?:diff|patch)[ \t]*\n(.*?)^```[ \t]*$", answer, re.M | re.S)
    if len(blocks) == 1:
        raw = blocks[0]
    elif not blocks and answer.lstrip().startswith(("diff --git ", "--- ")):
        raw = answer.lstrip()
    else:
        raise PatchError(f"expected one unified diff, found {len(blocks)} fenced diffs")
    lines = raw.splitlines()
    if lines[:1] == [f"diff --git a/{TARGET} b/{TARGET}"]:
        lines.pop(0)
    if lines and re.fullmatch(r"index [0-9a-f]+\.\.[0-9a-f]+(?: 100644)?", lines[0]):
        lines.pop(0)
    if lines[:2] != [f"--- a/{TARGET}", f"+++ b/{TARGET}"]:
        raise PatchError(f"expected a patch to {TARGET}")
    # Strip file metadata and submit only this fixed path and ordinary text hunks to git.
    hunks = lines[2:]
    if not hunks or not hunks[0].startswith("@@ "):
        raise PatchError("missing text hunk")
    for line in hunks:
        if re.fullmatch(r"@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@.*", line):
            continue
        if line.startswith(("--- ", "+++ ")) or not line.startswith((" ", "+", "-")):
            raise PatchError("unexpected file header or non-hunk line")
    return "\n".join(lines) + "\n"


def cases():
    # Derive expected totals from the original unpacked spans independently of the candidate.
    groups = {
        "empty": [([], -3, 100), ([(0, 4, "express")], 3, 3), ([(0, 4, "express")], 4, -2)],
        "wire_tags": [([(0, 1, mode)], 0, 1) for mode in ("idle", "local", "express")],
        "endpoints": [([(3, 7, "local"), (7, 10, "express")], a, b)
                      for a, b in [(0, 3), (3, 7), (7, 10), (10, 20), (5, 9), (-20, 30)]],
        "relative_gaps": [([(4, 15, "idle"), (18, 27, "express"), (50, 52, "local")], a, b)
                          for a, b in [(0, 60), (15, 18), (18, 27), (49, 51)]],
        "mixed_windows": [],
    }
    rng = random.Random(46911)
    for _ in range(30):
        spans, end = [], 0
        for _ in range(16):
            start = end + rng.randrange(6)
            end = start + rng.randint(1, 20)
            spans.append((start, end, rng.choice(("idle", "local", "express"))))
        left = rng.randrange(-9, end)
        groups["mixed_windows"].append((spans, left, left + rng.randrange(1, 74)))
    writer = {}
    exec(WRITER, writer)
    policy = {}
    exec(POLICY, policy)
    inputs, expected = [], []
    for group, data in groups.items():
        for spans, left, right in data:
            inputs.append([list(writer["pack"](spans)), left, right])
            expected.append((group, sum(max(0, min(right, b) - max(left, a)) *
                                        policy["UNITS_PER_TICK"][mode] for a, b, mode in spans)))
    return inputs, expected


DRIVER = '''import json, sys
sys.path.insert(0, "/fixture")
from daily_units import units_between
cases = json.load(sys.stdin)
results = []
for packets, start, stop in cases:
    try:
        value = units_between((tuple(row) for row in packets), start, stop)
        results.append(value if type(value) is int else {"type": type(value).__name__})
    except Exception as exc:
        results.append({"error": type(exc).__name__})
print(json.dumps(results))
'''


def command_for_sandbox(root):
    bwrap = shutil.which("bwrap")
    if not bwrap:
        raise RuntimeError("client grading requires Linux bubblewrap (bwrap)")
    return [bwrap, "--unshare-all", "--unshare-user", "--disable-userns", "--die-with-parent", "--new-session",
            "--cap-drop", "ALL", "--clearenv", "--ro-bind", "/usr", "/usr",
            "--tmpfs", "/usr/local", "--remount-ro", "/usr/local", "--symlink", "usr/bin", "/bin",
            "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
            "--ro-bind", str(root), "/fixture", "--chdir", "/fixture", "--remount-ro", "/",
            "/usr/bin/prlimit", "--cpu=4", "--nproc=8", "--fsize=65536", "--core=0",
            "--as=268435456", "--", "/usr/bin/python3", "-I", "-B", "-S", "-c", DRIVER]


def run_checks(root):
    inputs, expected = cases()
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        try:
            proc = subprocess.run(command_for_sandbox(root), input=json.dumps(inputs).encode(),
                                  stdout=out, stderr=err, timeout=10, check=False)
        except subprocess.TimeoutExpired:
            return {"passed": False, "error": "candidate timed out"}
        err.seek(0)
        stderr = err.read(4096).decode(errors="replace")
        if proc.returncode:
            if stderr.startswith(("bwrap:", "prlimit:")):
                raise RuntimeError(f"grading sandbox unavailable: {stderr.strip()}")
            return {"passed": False, "error": f"candidate exit {proc.returncode}", "stderr": stderr}
        out.seek(0)
        try:
            actual = json.loads(out.read(65537))
        except (ValueError, UnicodeError):
            return {"passed": False, "error": "candidate output was not valid JSON"}
    if not isinstance(actual, list) or len(actual) != len(expected):
        return {"passed": False, "error": "candidate returned an unexpected number of results"}
    suite = unittest.TestSuite()

    def test_value(index, want):
        def check():
            if type(actual[index]) is not int or actual[index] != want:
                raise AssertionError(f"expected {want}, got {str(actual[index])[:160]}")
        return check

    for i, (group, want) in enumerate(expected):
        suite.addTest(unittest.FunctionTestCase(test_value(i, want), description=f"{group}/{i}"))
    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=1).run(suite)
    failures = [{"case": t.shortDescription(), "failure": trace.splitlines()[-1]}
                for t, trace in result.failures + result.errors]
    return {"passed": result.wasSuccessful(), "checks": result.testsRun, "passed_checks": result.testsRun - len(failures),
            "groups": sorted({g for g, _ in expected}), "failures": failures}


def grade(answer):
    patch = extract_patch(answer)
    with tempfile.TemporaryDirectory(prefix="inkling-coding-") as tmp:
        root = Path(tmp)
        for name, source in [(TARGET, STARTER), ("rate_card.py", POLICY), ("packet_writer.py", WRITER)]:
            (root / name).write_text(source)
        # No repository config, hooks or checkout are exposed to the candidate.
        cmd = ["git", "-c", "apply.whitespace=nowarn", "apply", "--recount", "-p1", "-"]
        env = {"PATH": os.defpath, "HOME": tmp, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CEILING_DIRECTORIES": tmp}
        proc = subprocess.run(cmd, cwd=root, input=patch, text=True, capture_output=True, env=env)
        if proc.returncode:
            raise PatchError(proc.stderr.strip() or "git apply failed")
        return run_checks(root)


def run(args, report):
    def call(route, body):
        return post(args.url, route, body, args.timeout)

    props = call("/props", None)
    capacity = props["default_generation_settings"]["n_ctx"]
    report.update(slot_context=capacity, total_slots=props.get("total_slots"), model=props.get("model_alias"),
                  model_ftype=props.get("model_ftype"))
    available = capacity - args.prompt_tokens
    report["available_generation_tokens"] = available
    if available < 1 or (args.max_tokens is not None and args.max_tokens > available):
        raise RuntimeError(f"slot context {capacity} leaves {available} tokens after the requested prompt")
    chat, meta = build_prompt(args, call)
    report.update(meta)
    started = time.monotonic()
    body = {"messages": chat, "temperature": 0, "seed": 42,
            "reasoning_effort": args.reasoning_effort, "cache_prompt": False, "stream": False}
    if args.max_tokens is not None:
        body["max_tokens"] = args.max_tokens
    if args.model:
        body["model"] = args.model
    response = call("/v1/chat/completions", body)
    choice = response["choices"][0]
    report.update(generation_seconds=round(time.monotonic() - started, 3), model=response.get("model"),
                  usage=response["usage"], finish_reason=choice["finish_reason"], timings=response.get("timings"),
                  reasoning_chars=len(choice["message"].get("reasoning_content") or ""),
                  answer=choice["message"].get("content") or "")
    score_answer(report)


def score_answer(report):
    if report["usage"].get("prompt_tokens") != report["requested_prompt_tokens"]:
        raise RuntimeError("server did not confirm the complete requested prompt")
    if report["finish_reason"] != "stop":
        report.update(status="failed", error=f"generation ended with {report['finish_reason']!r}")
        return
    report["grade"] = grade(report["answer"])
    report["status"] = "passed" if report["grade"]["passed"] else "failed"


def self_test():
    diff = "".join(difflib.unified_diff(STARTER.splitlines(True), REFERENCE.splitlines(True),
                                        f"a/{TARGET}", f"b/{TARGET}"))
    answer = f"```diff\n{diff}```\n"
    result = grade(answer)
    if not result["passed"]:
        raise RuntimeError(str(result))
    wrong = grade(answer.replace("total += overlap", "total -= overlap"))
    if wrong["passed"]:
        raise RuntimeError("incorrect billing passed")
    for bad in (answer.replace("b/daily_units.py", "b/../other"), answer.replace("```\n", diff + "```\n")):
        try:
            extract_patch(bad)
        except PatchError:
            continue
        raise RuntimeError("unexpected file path accepted")
    print(json.dumps({"reference": result, "wrong_candidate_failures": len(wrong["failures"])}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="server model alias, when required")
    parser.add_argument("--regrade", type=Path, help="regrade a saved result locally without an inference request")
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--prompt-tokens", type=int, default=262144)
    parser.add_argument("--max-tokens", type=int, help="optional client output cap; by default the request leaves it unset")
    parser.add_argument("--reasoning-effort", choices=["none", "minimal", "low", "medium", "high", "xhigh", "max"], default="high")
    parser.add_argument("--timeout", type=float, default=7200, help="per-request timeout in seconds")
    parser.add_argument("--out", type=Path, help="result JSON; must be a new file")
    parser.add_argument("--self-test", action="store_true", help="test the isolated grader without an inference server")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not args.out or args.prompt_tokens < 1 or (args.max_tokens is not None and args.max_tokens < 1) or args.timeout <= 0:
        parser.error("supply --out and positive token budgets and timeout")
    report = {"fixture": "shipping-ledger-v1", "reasoning_effort": args.reasoning_effort,
              "requested_prompt_tokens": args.prompt_tokens, "generation_budget": args.max_tokens}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "x", opener=lambda path, flags: os.open(path, flags, 0o600)) as out:
        try:
            if args.regrade:
                report = json.loads(args.regrade.read_text())
                if report.get("fixture") != "shipping-ledger-v1":
                    raise ValueError("unexpected fixture in saved report")
                for key in ("error", "grade", "status"):
                    report.pop(key, None)
                score_answer(report)
            else:
                run(args, report)
        except PatchError as exc:
            report.update(status="failed", error=str(exc))
        except (OSError, RuntimeError, ValueError, KeyError, IndexError) as exc:
            report.update(status="error", error=str(exc))
        json.dump(report, out, indent=2, ensure_ascii=True)
        out.write("\n")
    print(f"{report['status']}: {args.out}")
    return {"passed": 0, "failed": 1, "error": 2}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
