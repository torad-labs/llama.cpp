#!/usr/bin/env python3
"""Per-session decode rate of a running server with N sessions decoding at once, from the server's own timings.

One line a session count: tokens/s a session (median, lowest, highest), tokens landed a step, the step's time those two
imply, and the drafts accepted. Nothing is written to disk and the server is not touched.
"""
import argparse
import concurrent.futures
import json
import statistics
import threading
import urllib.request

# Tasks whose answers run far past the token limit, so every session is still decoding when the others are.
TASKS = [
    "Write a Python module that parses a crontab file into a schedule object, with unit tests for every field type.",
    "Explain how a B-tree splits and merges nodes on insert and delete, with a worked example of order 4.",
    "Write a Rust function that streams a large CSV file and computes per-column statistics, then explain each part.",
    "Describe how TCP congestion control evolved from Tahoe to BBR, and what each version changed.",
    "Write a TypeScript rate limiter with a sliding window, its tests, and a section on its failure modes.",
    "Explain how a garbage collector with generations decides what to promote, with pseudocode.",
    "Write a Go HTTP middleware chain for logging, auth and retries, and walk through a request passing it.",
    "Review the design of a write-ahead log: what is written when, what recovery replays, and where it can lose data.",
    "Write a SQL schema and the queries for a double-entry ledger, with the constraints that keep it balanced.",
    "Explain how a transformer's attention is computed for a batch, shape by shape, and where the memory goes.",
    "Write a C function that implements a lock-free single-producer single-consumer ring buffer, and explain why it is safe.",
    "Describe how a Raft leader election proceeds through a network partition and its healing, step by step.",
]


def post(url, path, body, timeout):
    request = urllib.request.Request(url + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def session(args, index, barrier):
    task = TASKS[index % len(TASKS)]
    messages = [{"role": "system", "content": "You are a software engineering assistant."},
                {"role": "user", "content": f"Session {index}. {task} Be thorough and complete."}]
    prompt = post(args.url, "/apply-template", {"messages": messages}, args.timeout)["prompt"]
    body = {"prompt": prompt, "n_predict": args.tokens, "cache_prompt": False, "seed": 530 + index,
            "temperature": 0.0 if args.greedy else 1.0, "top_p": 0.95, "repeat_penalty": 1.1, "repeat_last_n": 256}
    barrier.wait(timeout=args.timeout)
    return post(args.url, "/completion", body, args.timeout)["timings"]


def read(args, sessions):
    barrier = threading.Barrier(sessions)
    with concurrent.futures.ThreadPoolExecutor(max_workers=sessions) as pool:
        timings = list(pool.map(lambda index: session(args, index, barrier), range(sessions)))
    rates = [t["predicted_per_second"] for t in timings]
    tokens = sum(t["predicted_n"] for t in timings)
    drafted = sum(t.get("draft_n", 0) for t in timings)
    accepted = sum(t.get("draft_n_accepted", 0) for t in timings)
    # every step lands one sampled token and the drafts it accepted
    per_step = tokens / max(1, tokens - accepted)
    median = statistics.median(rates)
    full = sum(t["predicted_n"] == args.tokens for t in timings)
    print(f"{sessions:3d} sessions: {median:6.1f} tok/s a session (low {min(rates):.1f}, high {max(rates):.1f}), "
          f"{per_step:.2f} tokens a step, {1000 * per_step / median:.1f} ms a step, "
          f"{accepted}/{drafted} drafts accepted, {sum(rates):.0f} tok/s in all, "
          f"{full}/{sessions} sessions reached {args.tokens} tokens", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:8102")
    parser.add_argument("--sessions", type=int, nargs="+", default=[23], help="Session counts, each read in turn")
    parser.add_argument("--tokens", type=int, default=256, help="Tokens each session decodes")
    parser.add_argument("--greedy", action="store_true", help="Temperature 0 in place of temperature 1, top-p 0.95")
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    for sessions in args.sessions:
        read(args, sessions)


if __name__ == "__main__":
    main()
