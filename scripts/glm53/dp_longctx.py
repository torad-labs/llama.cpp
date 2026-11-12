"""Claude Code-shaped sessions at long context against a served GLM-5.3-Flash, several at once.

A session is what Claude Code sends: a system prompt, a tool list, the user's request, then Read calls and their
results (the files of a public source tree, --corpus) until the rendered prompt measures --context-tokens by the
server's own count (/apply-template then /tokenize). Turn 1 prefills that whole history. Every later turn appends the
reply, the user's next ask and one more Read step, and must reuse the cache: a turn that re-evaluates more than a
quarter of its context is reported as a lost cache. --streams sessions start their first turn together, as a burst of
Claude Code windows would, and then run their turns independently. Memory is sampled on every card while they run.
A response is waited for as long as the server works at it (--timeout sets a limit; twelve 400k first turns at once
take over an hour on four cards), and every turn is appended to <out>.partial.jsonl as it lands, so a run stopped
before its end keeps what it measured.

    python3 scripts/glm53/dp_longctx.py --server http://127.0.0.1:8080 --corpus . \\
        --context-tokens 400000 --streams 12 --turns 3 --out results/dp-12x400k.json

The results file keeps every response's timings; the summary printed at the end is also in it.
"""
import argparse
from collections import defaultdict
import concurrent.futures
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import threading
import time
import urllib.request

SYSTEM = """You are an interactive CLI tool that helps users with software engineering tasks. Use the instructions below
and the tools available to you to assist the user.

# Tone and style
You should be concise, direct, and to the point. When you run a non-trivial bash command, you should explain what the
command does and why you are running it. Your output will be displayed on a command line interface. Output text to
communicate with the user; all text you output outside of tool use is displayed to the user. Only use tools to complete
tasks. Never use tools or code comments as means to communicate with the user. If you cannot or will not help the user
with something, do not say why or what it could lead to. Offer helpful alternatives if possible, and otherwise keep your
response to one or two sentences. Keep your responses short, since they will be displayed on a command line interface.
Answer the user's question directly, without elaboration, explanation, or details. One word answers are best. Avoid
introductions, conclusions, and explanations.

# Following conventions
When making changes to files, first understand the file's code conventions. Mimic code style, use existing libraries and
utilities, and follow existing patterns. Never assume that a given library is available, even if it is well known. When
you write code that uses a library or framework, first check that this codebase already uses the given library. When you
create a new component, first look at existing components to see how they are written; then consider framework choice,
naming conventions, typing, and other conventions. When you edit a piece of code, first look at the code's surrounding
context (especially its imports) to understand the code's choice of frameworks and libraries. Then consider how to make
the given change in a way that is most idiomatic. Always follow security best practices. Never introduce code that
exposes or logs secrets and keys. Never commit secrets or keys to the repository.

# Code style
Do not add comments to the code you write, unless the user asks you to, or the code is complex and requires context.

# Doing tasks
The user will primarily request you perform software engineering tasks. This includes solving bugs, adding new
functionality, refactoring code, explaining code, and more. For these tasks the following steps are recommended: use
the available search tools to understand the codebase and the user's query; implement the solution using all tools
available to you; verify the solution if possible with tests; when you have completed a task, run the lint and
typecheck commands if they were provided to you to ensure your code is correct. Never commit changes unless the user
explicitly asks you to.

# Tool usage policy
When doing file search, prefer to use the Agent tool in order to reduce context usage. You have the capability to call
multiple tools in a single response. When multiple independent pieces of information are requested, batch your tool
calls together for optimal performance. You must answer concisely with fewer than 4 lines of text (not including tool
use or code generation), unless the user asks for detail.
"""

USER_OPENING = ("Read through the engine's sources, one file at a time, and keep notes on how the server's prompt "
                "cache, the CUDA backend's fusions and the model loader fit together. Read every file I point you at "
                "in full before answering any question about it.")


def tool(name, description, properties, required):
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required}}}


TOOLS = [
    tool("Read", "Reads a file from the local filesystem. The file_path parameter must be an absolute path. By default "
         "it reads up to 2000 lines from the start of the file; offset and limit select a range. Results are returned "
         "with line numbers in cat -n format.",
         {"file_path": {"type": "string"}, "offset": {"type": "integer"}, "limit": {"type": "integer"}}, ["file_path"]),
    tool("Write", "Writes a file to the local filesystem, overwriting it if it exists. The file must have been read "
         "in this conversation first.", {"file_path": {"type": "string"}, "content": {"type": "string"}},
         ["file_path", "content"]),
    tool("Edit", "Performs an exact string replacement in a file. old_string must match exactly and be unique unless "
         "replace_all is set.", {"file_path": {"type": "string"}, "old_string": {"type": "string"},
         "new_string": {"type": "string"}, "replace_all": {"type": "boolean"}},
         ["file_path", "old_string", "new_string"]),
    tool("MultiEdit", "Makes several edits to one file in one operation; each edit is applied in order.",
         {"file_path": {"type": "string"}, "edits": {"type": "array", "items": {"type": "object", "properties": {
             "old_string": {"type": "string"}, "new_string": {"type": "string"}}}}}, ["file_path", "edits"]),
    tool("Bash", "Executes a bash command in a persistent shell session with an optional timeout in milliseconds. "
         "Quote paths with spaces. Prefer dedicated tools for reading and searching files.",
         {"command": {"type": "string"}, "timeout": {"type": "integer"}, "description": {"type": "string"}},
         ["command"]),
    tool("Glob", "Fast file pattern matching over any codebase size; returns matching paths sorted by modification "
         "time.", {"pattern": {"type": "string"}, "path": {"type": "string"}}, ["pattern"]),
    tool("Grep", "Searches file contents with regular expressions, with ripgrep's syntax; supports include globs and "
         "output modes content, files_with_matches and count.",
         {"pattern": {"type": "string"}, "path": {"type": "string"}, "include": {"type": "string"},
          "output_mode": {"type": "string", "enum": ["content", "files_with_matches", "count"]}}, ["pattern"]),
    tool("LS", "Lists files and directories in a given absolute path, with optional ignore patterns.",
         {"path": {"type": "string"}, "ignore": {"type": "array", "items": {"type": "string"}}}, ["path"]),
    tool("Agent", "Launches a new agent with its own context to handle a multi-step task and report back.",
         {"description": {"type": "string"}, "prompt": {"type": "string"}, "subagent_type": {"type": "string"}},
         ["description", "prompt"]),
    tool("TodoWrite", "Creates and manages a structured task list for the current session.",
         {"todos": {"type": "array", "items": {"type": "object", "properties": {
             "content": {"type": "string"}, "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},
             "activeForm": {"type": "string"}}}}}, ["todos"]),
    tool("WebFetch", "Fetches content from a URL and processes it with the given prompt.",
         {"url": {"type": "string"}, "prompt": {"type": "string"}}, ["url", "prompt"]),
    tool("NotebookEdit", "Replaces, inserts or deletes a cell of a Jupyter notebook.",
         {"notebook_path": {"type": "string"}, "cell_id": {"type": "string"}, "new_source": {"type": "string"},
          "edit_mode": {"type": "string", "enum": ["replace", "insert", "delete"]}}, ["notebook_path", "new_source"]),
]

SUFFIXES = (".cpp", ".h", ".hpp", ".c", ".cu", ".cuh", ".py", ".md", ".txt")
FOLLOW_UPS = ["Now summarize what that file does in two sentences.",
              "Which function in it would you change first to add a new op, and why?",
              "List the symbols it defines that are used elsewhere.",
              "What would break if its first struct gained a field?"]


def corpus_files(root, suffixes=SUFFIXES, min_bytes=2000, max_bytes=400000):
    """The source files under root worth a Read: at least min_bytes, at most max_bytes, sorted for a stable order."""
    return sorted(p for p in Path(root).rglob("*") if p.is_file() and p.suffix in suffixes
                  and min_bytes <= p.stat().st_size <= max_bytes and ".git" not in p.parts)


def session_text(messages, tools):
    """What the server will tokenize, near enough to estimate from: the tools and every message's text."""
    parts = [json.dumps(tools)]
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        for call in message.get("tool_calls", []):
            parts.append(call["function"]["name"] + call["function"]["arguments"])
    return "\n".join(parts)


def read_step(index, path, text):
    call_id = f"call_{index:04d}"
    call = {"role": "assistant", "content": "", "tool_calls": [{"id": call_id, "type": "function", "function": {
        "name": "Read", "arguments": json.dumps({"file_path": str(path)})}}]}
    return call, {"role": "tool", "tool_call_id": call_id, "content": text}


def build_session(files, target_tokens, measure, tools, seed=0):
    """A history that measures between 98% and 100% of target_tokens by measure(messages, tools), the server's count.

    Files are read in a seeded order, whole, until the next one would pass the target; the last is cut to the room
    left. measure is called at the start, after the first step, every eighth step, and while the last step is cut."""
    order = list(files)
    random.Random(seed).shuffle(order)
    if not order:
        raise ValueError("the corpus has no files")
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER_OPENING}]
    size = measure(messages, tools)
    if size > target_tokens:
        raise ValueError(f"the system prompt and tools alone measure {size} tokens, over the target {target_tokens}")
    tokens_per_char = None
    index = 0
    steps = []
    while True:
        path = order[index % len(order)]
        text = path.read_text(errors="replace")
        call, result = read_step(index, path, text)
        trial = messages + [call, result]
        exact = tokens_per_char is None or index % 8 == 0
        trial_size = measure(trial, tools) if exact else size + int(len(text) * tokens_per_char) + 16
        if not exact and trial_size > target_tokens:
            # the estimate says this file is the one that passes the target: the count decides
            trial_size, exact = measure(trial, tools), True
        if exact:
            tokens_per_char = max(1e-3, (trial_size - size) / max(1, len(text)))
        if trial_size > target_tokens:
            break
        messages, size, index = trial, trial_size, index + 1
        steps.append((path, text))
    # the count decides what the estimates let through: a history over the target drops its last reads, and the
    # first read dropped is the one cut to the room left
    size = measure(messages, tools)
    while size > target_tokens and steps:
        path, text = steps.pop()
        messages, index = messages[:-2], index - 1
        size = measure(messages, tools)
    # the file that passes the target, cut to the room left: the cut is measured and rescaled by its own rate until
    # it lands between the floor and the target; a room too small for a cut leaves the history as it is
    floor = int(target_tokens * 0.98)
    if size >= floor:
        return messages
    keep = int((target_tokens - size) / tokens_per_char)
    for _ in range(12):
        if keep < 64:
            break
        call, result = read_step(index, path, text[:keep])
        trial = messages + [call, result]
        trial_size = measure(trial, tools)
        if floor <= trial_size <= target_tokens:
            return trial
        rate = max(1e-3, (trial_size - size) / keep)
        keep = int((target_tokens - size) / rate * (0.985 if trial_size > target_tokens else 0.995))
    return messages


def next_turn(messages, reply, path, question, step_chars=4000):
    """The reply, the user's next ask and one more Read of path appended; the earlier messages untouched."""
    index = sum(1 for m in messages if m["role"] == "tool")
    call, result = read_step(index, path, path.read_text(errors="replace")[:step_chars])
    return messages + [{"role": "assistant", "content": reply}, {"role": "user", "content": question}, call, result]


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def decode_rate(timing):
    if "predicted_per_second" in timing:
        rate = timing["predicted_per_second"]
        return rate if finite_number(rate) and rate >= 0 else None
    ms, n = timing.get("predicted_ms"), timing.get("predicted_n")
    return max(n - 1, 0) * 1000 / ms if finite_number(ms) and ms > 0 and finite_number(n) else None


def diagnostics(timings):
    fields = ("draft_n", "draft_n_accepted", "draft_rounds", "draft_delivered_n", "draft_verify_attempts",
              "draft_verify_rows_total", "draft_replay_attempts")
    totals = {"requests": len(timings), "observations": {}}
    for field in fields:
        values = [t[field] for t in timings if finite_number(t.get(field))]
        totals[field] = sum(values) if values else None
        totals["observations"][field] = len(values)
    histograms = [t["draft_verify_rows"] for t in timings if isinstance(t.get("draft_verify_rows"), dict)]
    histogram = defaultdict(int)
    for counts in histograms:
        for bucket, count in counts.items():
            histogram[bucket] += count
    totals["draft_verify_rows"] = dict(histogram) if histograms else None
    totals["observations"]["draft_verify_rows"] = len(histograms)
    # Ratios require paired observations; absent diagnostics are not observed zeros.
    for name, numerator, denominator in (("acceptance", "draft_n_accepted", "draft_n"),
                                         ("delivered_per_round", "draft_delivered_n", "draft_rounds")):
        paired = all(finite_number(t.get(numerator)) and finite_number(t.get(denominator)) for t in timings)
        totals[name] = totals[numerator] / totals[denominator] if paired and totals[denominator] else None
    return totals


def request_interval(rows):
    if not all(finite_number(r.get("request_start_s")) and finite_number(r.get("request_end_s"))
               and r["request_end_s"] >= r["request_start_s"] for r in rows):
        return None
    elapsed = max(r["request_end_s"] for r in rows) - min(r["request_start_s"] for r in rows)
    return elapsed if elapsed > 0 else None


def summarize(results):
    """Per turn, over the streams: prompt sizes, cache hits, prefill time, decode speed; and every turn after the
    first that re-evaluated more than a quarter of its context (its slot lost the cache)."""
    by_turn = defaultdict(list)
    for r in results:
        by_turn[r["turn"]].append(r)
    turns, cache_lost = [], []
    for turn in sorted(by_turn):
        rows = by_turn[turn]
        timings = [r["timings"] for r in rows]
        tps = [rate for t in timings if (rate := decode_rate(t)) is not None]
        interval = request_interval(rows)
        turns.append({
            "turn": turn,
            "streams": len(rows),
            "prompt_n_max": max(t["prompt_n"] for t in timings),
            "prompt_n_median": statistics.median(t["prompt_n"] for t in timings),
            "cache_n_median": statistics.median(t.get("cache_n", 0) for t in timings),
            "prompt_ms_median": statistics.median(t["prompt_ms"] for t in timings),
            "prompt_ms_max": max(t["prompt_ms"] for t in timings),
            "decode_tps_median": statistics.median(tps) if tps else None,
            "output_tps_aggregate": sum(t["predicted_n"] for t in timings) / interval if interval else None,
            "request_elapsed_s": interval,
            "diagnostics": diagnostics(timings),
            "wall_ms_max": max(r["wall_ms"] for r in rows),
        })
        if turn > 1:
            for r in rows:
                t = r["timings"]
                if t["prompt_n"] > 0.25 * (t["prompt_n"] + t.get("cache_n", 0)):
                    cache_lost.append({"stream": r["stream"], "turn": turn, "prompt_n": t["prompt_n"],
                                       "cache_n": t.get("cache_n", 0)})
    return {"turns": turns, "cache_lost": cache_lost}


def peak(samples):
    """The per-card maximum over samples, each a list of one value per card."""
    if not samples:
        return []
    return [max(column) for column in zip(*samples)]


class Server:
    def __init__(self, url, timeout=None):  # None: wait as long as the server is working on the response
        self.url = url.rstrip("/")
        self.timeout = timeout

    def post(self, route, body):
        data = json.dumps(body).encode()
        request = urllib.request.Request(self.url + route, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read())

    def measure(self, messages, tools):
        prompt = self.post("/apply-template", {"messages": messages, "tools": tools})["prompt"]
        return len(self.post("/tokenize", {"content": prompt, "add_special": False})["tokens"])

    def chat(self, messages, tools, max_tokens, server_sampling=False):
        body = {"messages": messages, "tools": tools, "max_tokens": max_tokens, "stream": False}
        if not server_sampling:
            body["temperature"] = 0.0
        return self.post("/v1/chat/completions", body)


def sample_memory(stop, samples, every):
    while not stop.is_set():
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=10).stdout
            samples.append([int(v.strip()) for v in out.split() if v.strip()])
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
        stop.wait(every)


def run_stream(server, number, files, args, record, log, start, clock_origin):
    name = f"s{number:02d}"
    try:
        session = build_session(files, args.context_tokens, server.measure, TOOLS, seed=number + 1)
        size = server.measure(session, TOOLS)
    finally:
        start.wait()  # every first turn leaves together; a failed build still releases the others
    log(f"{name}: history built, {size} tokens, {sum(1 for m in session if m['role'] == 'tool')} reads")
    for turn in range(1, args.turns + 1):
        started = time.monotonic()
        response = server.chat(session, TOOLS, args.reply_tokens, args.server_sampling)
        ended = time.monotonic()
        wall_ms = (ended - started) * 1000.0
        choice = response["choices"][0]["message"]
        reply = choice.get("content") or ""
        timings = response.get("timings", {})
        record({"stream": name, "turn": turn, "wall_ms": wall_ms, "timings": timings,
                "request_start_s": started - clock_origin, "request_end_s": ended - clock_origin,
                "finish_reason": response["choices"][0].get("finish_reason"), "reply_chars": len(reply),
                "tool_calls": len(choice.get("tool_calls") or [])})
        log(f"{name}: turn {turn} prompt_n={timings.get('prompt_n')} cache_n={timings.get('cache_n')} "
            f"prompt_ms={timings.get('prompt_ms')} predicted_n={timings.get('predicted_n')} "
            f"predicted_ms={timings.get('predicted_ms')} wall_ms={wall_ms:.0f}")
        if turn < args.turns:
            session = next_turn(session, reply or "(no reply)", files[turn % len(files)],
                                FOLLOW_UPS[(turn - 1) % len(FOLLOW_UPS)])


def format_rate(value):
    return f"{value:.2f}" if value is not None else "unavailable"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server", default="http://127.0.0.1:8080")
    parser.add_argument("--corpus", type=Path, required=True, help="a public source tree the sessions read")
    parser.add_argument("--context-tokens", type=int, default=400000)
    parser.add_argument("--streams", type=int, default=1)
    parser.add_argument("--turns", type=int, default=3)
    parser.add_argument("--reply-tokens", type=int, default=256)
    parser.add_argument("--server-sampling", action="store_true",
                        help="Send no sampling fields: the server's own sampling, what a client that sets none "
                             "(Claude Code) is served with; without it every turn is sent at temperature 0")
    parser.add_argument("--memory-every", type=float, default=2.0, help="seconds between nvidia-smi samples")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=None,
                        help="seconds to wait for one response (default: no limit; a burst of 400k prefills takes over an hour)")
    args = parser.parse_args(argv)

    files = corpus_files(args.corpus)
    if not files:
        parser.error(f"no source files under {args.corpus}")
    server = Server(args.server, timeout=args.timeout)
    results, samples, lines = [], [], []
    lock = threading.Lock()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix(".partial.jsonl")
    partial.unlink(missing_ok=True)

    def record(row):
        # each turn on disk as it lands: a run stopped before its end keeps what it measured
        with lock:
            results.append(row)
            with partial.open("a") as f:
                f.write(json.dumps(row, default=str) + "\n")

    def log(line):
        with lock:
            lines.append(f"{time.strftime('%H:%M:%S')} {line}")
            print(lines[-1], flush=True)

    stop = threading.Event()
    sampler = threading.Thread(target=sample_memory, args=(stop, samples, args.memory_every), daemon=True)
    sampler.start()
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.streams) as pool:
        start = threading.Barrier(args.streams)
        futures = [pool.submit(run_stream, server, i, files, args, record, log, start, started) for i in range(args.streams)]
        failures = []
        for future in futures:
            try:
                future.result()
            except Exception as error:  # a stream's failure is a result, not an abort of the others
                failures.append(repr(error))
                log(f"stream failed: {error!r}")
    stop.set()
    sampler.join(timeout=15)
    summary = summarize(results) if results else {"turns": [], "cache_lost": []}
    report = {"metrics_version": 2, "config": vars(args) | {"corpus": str(args.corpus), "out": str(args.out)},
              "elapsed_s": time.monotonic() - started, "results": results, "summary": summary,
              "vram_peak_mib": peak(samples), "memory_samples": len(samples), "failures": failures, "log": lines}
    args.out.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print("\n| turn | streams | prompt_n max | cache_n median | prefill ms median | prefill ms max | decode tok/s "
          "median | output tok/s aggregate (request interval) |")
    print("|---|---|---|---|---|---|---|---|")
    for t in summary["turns"]:
        print(f"| {t['turn']} | {t['streams']} | {t['prompt_n_max']} | {t['cache_n_median']:.0f} | "
              f"{t['prompt_ms_median']:.0f} | {t['prompt_ms_max']:.0f} | {format_rate(t['decode_tps_median'])} | "
              f"{format_rate(t['output_tps_aggregate'])} |")
    print(f"\ncache lost: {summary['cache_lost'] or 'none'}")
    print(f"VRAM peak MiB per card: {report['vram_peak_mib']} over {len(samples)} samples")
    if failures:
        print(f"failures: {len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
