#!/usr/bin/env python3
"""One profiler window over steady decode: every session streams, and the window opens once each of them has tokens."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import subprocess
import sys
import time

here = Path(__file__).resolve().parent
# Beside search.py in the repo, or beside a kit's copy of it
kit = here / "repo/scripts/inkling2"
sys.path.insert(0, str(kit if kit.is_dir() else here))
import search

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://127.0.0.1:8103")
parser.add_argument("--sessions", type=int, default=23)
parser.add_argument("--context", type=int, default=0)
parser.add_argument("--tokens", type=int, default=400)
parser.add_argument("--after", type=int, default=24, help="Tokens every session has before the window opens")
parser.add_argument("--seconds", type=float, default=6)
parser.add_argument("--session", required=True, help="The nsys session the server was launched in")
parser.add_argument("--output", required=True)
parser.add_argument("--same", action="store_true", help="Every session sends the first one's request and seed, so a step "
                    "routes 23 copies of one token: the expert kernels do the same arithmetic over 8 experts' weights")
args = parser.parse_args()

rows = search.prompts(args.url, args.sessions, args.context)
if args.same:
    rows = rows[:1] * args.sessions
counts = [0] * args.sessions
ids = [None] * args.sessions


def client(slot):
    body = dict(prompt=rows[slot], id_slot=slot, seed=0 if args.same else slot, cache_prompt=True, ignore_eos=True,
                n_predict=args.tokens, stream=True, return_tokens=True)

    def seen(item):
        if isinstance(item["event"], dict):
            counts[slot] += len(item["event"].get("tokens") or [])

    with search.mixed_prefill.post(args.url, "/completion", body, 7200) as response:
        stream = search.glm_measure.read_completion_stream(response, args.tokens, on_event=seen)
    ids[slot] = tuple(stream["token_ids"])
    return stream["response"]["timings"]


def nsys(*words):
    done = subprocess.run(["nsys", *words], capture_output=True, text=True)
    return {"words": " ".join(words), "exit": done.returncode, "output": (done.stdout + done.stderr)[-1500:]}


begin = time.monotonic()
with concurrent.futures.ThreadPoolExecutor(max_workers=args.sessions) as pool:
    streams = [pool.submit(client, slot) for slot in range(args.sessions)]
    while min(counts) < args.after and not any(s.done() for s in streams):
        time.sleep(0.05)
    result = {"session": args.session, "context": args.context, "prompt_tokens": [len(row) for row in rows],
              "seconds_to_decode": round(time.monotonic() - begin, 1), "tokens_before": list(counts)}
    result["start"] = nsys("start", f"--session={args.session}", "--sample=none", "--cpuctxsw=none", "--force-overwrite=true",
                           "-o", args.output)
    opened, at = list(counts), time.monotonic()
    time.sleep(args.seconds)
    closed, seconds = list(counts), time.monotonic() - at
    result["stop"] = nsys("stop", f"--session={args.session}")
    result["window_seconds"] = round(seconds, 3)
    result["window_tokens"] = [b - a for a, b in zip(opened, closed)]
    result["window_tokens_per_second_a_session"] = round(sum(result["window_tokens"]) / seconds / args.sessions, 2)
    result["sessions_done_in_window"] = sum(1 for count in closed if count >= args.tokens)
    print(json.dumps(result), flush=True)
    timings = []
    for stream in streams:
        try:
            timings.append(stream.result())
        except Exception as error:  # a stream that failed still leaves the window's own numbers
            timings.append({"error": str(error)})
    print(json.dumps({"timings": timings, "distinct_streams": len(set(ids))}), flush=True)
sys.exit(0 if result["start"]["exit"] == 0 and result["stop"]["exit"] == 0 else 1)
