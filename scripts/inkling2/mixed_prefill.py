"""Measure the pauses seen by streaming clients while one cold prompt enters a running llama-server."""

import argparse
from contextlib import nullcontext
import concurrent.futures
import json
import math
import queue
import random
import statistics
import tempfile
from pathlib import Path
import threading
import time
import urllib.request


def post(url, route, body, timeout):
    request = urllib.request.Request(url.rstrip("/") + route, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(request, timeout=timeout)


def stream(args, prompt, predict, on_delivery, capture_text=False):
    body = {"prompt": prompt, "n_predict": predict, "stream": True, "return_tokens": True,
            "cache_prompt": False, "ignore_eos": True, "temperature": 0}
    text = []
    with post(args.url, "/completion", body, args.timeout) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            event = json.loads(line[5:])
            if "error" in event:
                raise RuntimeError(event["error"])
            if capture_text:
                text.append(event.get("content", ""))
            if event.get("tokens") and on_delivery(time.monotonic(), len(event["tokens"])):
                return
            if event["stop"]:
                if capture_text:
                    event["delivered_text"] = "".join(text)
                return event
    raise RuntimeError("stream closed before its terminal event")


def gaps(times):
    if not times:
        return {"count": 0, "p95_ms": None, "max_ms": None}
    ordered = sorted(times)
    return {"count": len(times), "p95_ms": 1000 * ordered[math.ceil(0.95 * len(times)) - 1],
            "max_ms": 1000 * ordered[-1]}


def compare_restore(args, vocab):
    def completion(slot, prompt, cached):
        body = {"id_slot": slot, "prompt": prompt, "cache_prompt": cached, "n_probs": 32,
                "n_predict": 1, "return_tokens": True, "ignore_eos": True, "temperature": 0}
        with post(args.url, "/completion", body, args.timeout) as response:
            result = json.load(response)
        timing = result["timings"]
        if result["truncated"] or timing["prompt_n"] + timing["cache_n"] != len(prompt):
            raise RuntimeError(f"slot {slot} did not process its complete {len(prompt)}-token input")
        return result

    prefix = random.Random(381).choices(vocab, k=args.reader_tokens)
    suffix = random.Random(51).choices(vocab, k=18)
    def distribution(reply):
        return {row["id"]: row["logprob"] for row in reply["completion_probabilities"][0]["top_logprobs"]}

    def compare(left, right):
        a, b = distribution(left), distribution(right)
        common = a.keys() & b.keys()
        return {"same_greedy_token": left["tokens"] == right["tokens"], "common_top_tokens": len(common),
                "max_common_logprob_difference": max((abs(a[t] - b[t]) for t in common), default=None)}

    def action(slot, name, filename=None):
        with post(args.url, f"/slots/{slot}?action={name}", {"filename": filename} if filename else {}, args.timeout) as response:
            return json.load(response)

    distractors = [token for token in vocab if token != prefix[0]]
    if not distractors:
        raise ValueError("need distinct tokenizer IDs for slot displacement")
    prompt = prefix + suffix
    reference_dir = getattr(args, "restore_reference_dir", None)
    file_control = None
    with (tempfile.NamedTemporaryFile(dir=reference_dir, prefix="inkling-reference-", suffix=".bin")
          if reference_dir else nullcontext(None)) as state:
        original = completion(0, prefix, False)
        if state:
            path = Path(state.name)
            saved = action(0, "save", path.name)
            if saved["n_saved"] != len(prefix) or path.stat().st_size != saved["n_written"]:
                raise ValueError("server and probe must share the same slot-save directory")
        completion(0, random.Random(55).choices(distractors, k=63), True)
        restored = completion(1, prompt, True)
        if state:
            # Reuse the RAM destination with the other slots untouched. Both continuations start from one prefill.
            action(1, "erase")
            loaded = action(1, "restore", path.name)
            if loaded["n_restored"] != len(prefix) or loaded["n_read"] != saved["n_written"]:
                raise ValueError("incomplete reference state")
            direct = completion(1, prompt, True)
            file_control = {"save": saved, "load": loaded, "timings": direct["timings"],
                            "top_logprobs": distribution(direct), "greedy_token": direct["tokens"][0],
                            "restored_vs_file": compare(restored, direct)}
    cold = completion(2, prompt, False)
    resident_prefix = completion(3, prefix, False)
    resident = completion(3, prompt, True)
    results = {"restored": restored, "resident": resident, "cold": cold}
    comparisons = {f"{left}_vs_{right}": compare(results[left], results[right]) for left, right in
                   (("restored", "cold"), ("resident", "cold"), ("restored", "resident"))}
    return {"greedy_token": {k: v["tokens"][0] for k, v in results.items()},
            "top_logprobs": {k: distribution(v) for k, v in results.items()},
            "timings": {k: v["timings"] for k, v in results.items()},
            "cached_tokens": {k: v["timings"]["cache_n"] for k, v in results.items()}, "comparisons": comparisons,
            "independent_prefixes": {"original": distribution(original), "resident": distribution(resident_prefix),
                                     "comparison": compare(original, resident_prefix)},
            "same_slot_file_control": file_control}


def measure(args, reader_prompt=None):
    with post(args.url, "/tokenize", {"content": " red blue green sky water cloud earth rain", "add_special": False},
              args.timeout) as response:
        vocab = json.load(response)["tokens"]
    if not vocab:
        raise ValueError("tokenizer returned no IDs")
    # Token IDs give each request its exact input length even when the tokenizer merges adjacent words.
    capture_text = reader_prompt is not None
    if reader_prompt is None:
        reader_prompt = random.Random(1).choices(vocab, k=args.reader_tokens)
    if len(reader_prompt) != args.reader_tokens:
        raise ValueError("reader prompt length differs from the requested token count")
    ready = queue.Queue()
    all_streaming = threading.Event()
    stop = threading.Event()

    def peer(index):
        arrivals = []
        warm_chunks = 0

        def record(now, count):
            nonlocal warm_chunks
            arrivals.append((now, count))
            if len(arrivals) == 1:
                ready.put((index, "first"))
            if all_streaming.is_set():
                warm_chunks += 1
                if warm_chunks == args.warm_chunks:
                    ready.put((index, "warm"))
            return stop.is_set()

        try:
            stream(args, random.Random(2 + index).choices(vocab, k=args.peer_prompt), args.peer_tokens, record)
        finally:
            ready.put((index, "finished"))
        return arrivals

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.sessions - 1) as pool:
        peers = [pool.submit(peer, i) for i in range(args.sessions - 1)]
        first = []
        try:
            for phase in ("first", "warm"):
                for _ in peers:
                    index, state = ready.get(timeout=args.timeout)
                    if state != phase:
                        peers[index].result()
                        raise RuntimeError(f"peer {index} finished during warmup, increase --peer-tokens")
                if phase == "first":
                    baseline_start = time.monotonic()
                    all_streaming.set()
            start = time.monotonic()
            start_epoch = time.time()

            def reader_first(now, _count):
                if not first:
                    first.append(now)
                    stop.set()
                return False

            completion = stream(args, reader_prompt, args.reader_output_tokens, reader_first, capture_text)
            if not first:
                raise RuntimeError("reader produced no token")
            timing = completion["timings"]
            if completion["truncated"] or timing["prompt_n"] != len(reader_prompt) or timing["cache_n"] != 0:
                raise RuntimeError(f"cold reader expected {len(reader_prompt)} tokens, evaluated {timing['prompt_n']}, "
                                   f"cached {timing['cache_n']}, truncated {completion['truncated']}")
            if timing["predicted_n"] != args.reader_output_tokens:
                raise RuntimeError(f"reader generated {timing['predicted_n']} of {args.reader_output_tokens} requested tokens")
        finally:
            stop.set()
        rows = [p.result() for p in peers]

    end = first[0]
    sessions = []
    overlap_gaps = []
    warm_rates = []
    for row in rows:
        warm = [(t, n) for t, n in row if baseline_start <= t <= start]
        before = [b - a for (a, _), (b, _) in zip(warm, warm[1:])]
        warm_rates.append(sum(n for _, n in warm) / (start - baseline_start))
        intervals = [(a, b) for (a, _), (b, _) in zip(row, row[1:]) if a < end and b > start]
        during = [b - a for a, b in intervals]
        observed = gaps(during)
        if intervals:
            a, b = max(intervals, key=lambda pair: pair[1] - pair[0])
            observed["max_interval_from_reader_start_ms"] = [1000 * (a - start), 1000 * (b - start)]
        overlap_gaps.extend(during)
        delivered = sum(n for t, n in row if start < t <= end)
        sessions.append({"spans_reader": row[0][0] <= start and row[-1][0] >= end,
                         "tokens_during_reader": delivered, "before": gaps(before), "overlapping": observed})
    seconds = end - start
    result = {"sessions": args.sessions, "reader_input_tokens": len(reader_prompt),
            "reader_evaluated_tokens": timing["prompt_n"], "reader_cached_tokens": timing["cache_n"],
            "reader_generated_tokens": timing["predicted_n"],
            "reader_ttft_ms": seconds * 1000,
            "reader_start_monotonic_s": start, "reader_start_epoch_s": start_epoch,
            "peers_spanning_reader": sum(p["spans_reader"] for p in sessions),
            "warm_median_peer_tokens_per_second": statistics.median(warm_rates),
            "reader_median_peer_tokens_per_second": statistics.median(p["tokens_during_reader"] / seconds for p in sessions),
            "reader_aggregate_peer_tokens_per_second": sum(p["tokens_during_reader"] for p in sessions) / seconds,
            "overlapping_delivery_gaps": gaps(overlap_gaps), "peers": sessions}
    if capture_text:
        result["reader_text"] = completion["delivered_text"]
    result["reader_timings"] = timing
    if args.check_restore:
        result["session_restore"] = compare_restore(args, vocab)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8103")
    parser.add_argument("--sessions", type=int, default=23, help="total clients, including the reader")
    parser.add_argument("--reader-tokens", type=int, default=262144, help="Cold input tokens (default: 262144)")
    parser.add_argument("--reader-output-tokens", type=int, default=1,
                        help="Tokens the reader generates; peer latency is measured until its first token (default: 1)")
    parser.add_argument("--peer-prompt", type=int, default=128)
    parser.add_argument("--peer-tokens", type=int, default=4096, help="maximum tokens per stream")
    parser.add_argument("--warm-chunks", type=int, default=16)
    parser.add_argument("--check-restore", action="store_true",
                        help="also compare RAM takeover, resident continuation and cold prefill on four slots")
    parser.add_argument("--restore-reference-dir", help="same directory as the local server's --slot-save-path for a temporary reference snapshot")
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    if (args.sessions < 2 or min(args.reader_tokens, args.reader_output_tokens, args.peer_prompt, args.peer_tokens) < 1 or
            not 2 <= args.warm_chunks < args.peer_tokens or args.timeout <= 0):
        parser.error("use at least two sessions, positive lengths and timeout, and 2 <= warm-chunks < peer-tokens")
    if args.check_restore and args.sessions < 4:
        parser.error("--check-restore needs at least four slots")
    result = measure(args)
    print(json.dumps(result, indent=2, allow_nan=False))
    if not all(row["spans_reader"] for row in result["peers"]):
        raise SystemExit("a peer finished before the reader's first token; increase --peer-tokens")
    if args.check_restore and result["session_restore"]["cached_tokens"] != {
            "restored": args.reader_tokens, "resident": args.reader_tokens, "cold": 0}:
        raise SystemExit("the restored, resident or cold request did not use the expected prefix")


if __name__ == "__main__":
    main()
