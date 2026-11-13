#!/usr/bin/env python3
"""Do sessions on a real server keep their own context? A planted code word per session, asked back under load.

  isolation.py --base-url http://127.0.0.1:8100/v1 [--model ID] [--sessions 8] [--turns 5] [--filler 6000]
               [--filler-max N] [--drop 0.25] [--max-tokens 400] [--seed 1] [--top-logprobs N] [--out FILE]
  The API key, when the server wants one, is read from the environment variable ISOLATION_API_KEY.

Every session is a conversation of its own with the SAME system prompt (as agents share one) and its own code word
in its first message, between paragraphs of filler of about --filler tokens (with --filler-max, a size of its own
between the two, so the sessions are of unlike lengths). Each later turn plants one more word and
asks for all of the session's words back. The sessions run at once, start at times of their own, and a share of the
requests is streamed and dropped by the client part-way, as an agent that is interrupted; a dropped turn is asked
again. With more sessions than the server has slots they take each other's slots and come back from the cache.

Every reply is checked for two things, which are counted apart:
  LEAK  the reply holds a code word that only another session was given
  LOST  the reply does not hold a word the session itself was given
The words are made of two halves that never occur in the filler, so a reply cannot hold one by chance.

A LOST turn is not always a lost context. The requests are at temperature 0, and a model split between two answers
(say 0.54 for starting with the newest word and stopping, 0.45 for starting with the first) loses a word with nothing
wrong in the server. With --top-logprobs N the server is asked for each reply token's N likeliest tokens, and a turn
that failed says the reply's closest pick, which tells the two apart.
Nothing private is sent: the filler and the words are generated here.
"""
import argparse
import http.client
import json
import math
import os
import random
import threading
import time
import urllib.parse

ap = argparse.ArgumentParser()
ap.add_argument("--base-url", required=True)
ap.add_argument("--model", default="default")
ap.add_argument("--sessions", type=int, default=8)
ap.add_argument("--turns", type=int, default=5)
ap.add_argument("--filler", type=int, default=6000, help="about this many tokens of filler in a session's first message")
ap.add_argument("--filler-max", type=int, default=0, help="each session draws its first message's filler between --filler and this")
ap.add_argument("--grow", type=int, default=600, help="about this many tokens of filler in each later message")
ap.add_argument("--drop", type=float, default=0.25, help="share of requests streamed and dropped part-way")
ap.add_argument("--max-tokens", type=int, default=400)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--skip", type=int, default=0, help="run only sessions --skip to --sessions - 1: the same conversations, fewer at once")
ap.add_argument("--top-logprobs", type=int, default=0, help="ask for each reply token's N likeliest tokens; a turn that failed says its closest pick")
ap.add_argument("--out", default="")
ap.add_argument("--extra", default="{}", help="JSON merged into every request, e.g. '{\"reasoning_effort\": \"low\"}'")
args = ap.parse_args()

url = urllib.parse.urlparse(args.base_url.rstrip("/"))
KEY = os.environ.get("ISOLATION_API_KEY", "")
EXTRA = json.loads(args.extra)

A = ["zarn", "quib", "vosk", "plim", "drax", "yeft", "gwol", "sneb", "trov", "jask", "hume", "krel", "bryx", "fenn", "olgo", "wixt"]
B = ["tavi", "moru", "lesk", "pody", "ching", "rauf", "bimo", "sted", "nual", "gorv", "ziph", "calt", "ewin", "drom", "xael", "uvik"]
WORDS = ("the of and a to in is that it was for on are as with his they at be this from have or by one had not but what all "
         "were when we there can an your which their said if do will each about how up out them then she many some so these "
         "would other into has more her two like him see time could no make than first been its who now people my made over "
         "did down only way find use may water long little very after words called just where most know river mountain "
         "garden window letter morning stone bridge market quiet season harvest lantern copper village meadow harbor").split()


def filler(rng, n_tokens):
    out = []
    while len(out) < n_tokens:  # about a token a word
        n = rng.randrange(8, 22)
        s = " ".join(rng.choice(WORDS) for _ in range(n))
        out.append(s[0].upper() + s[1:] + ".")
        out.extend([""] * (n - 1))
    return " ".join(x for x in out if x)


def code(s, k):
    r = random.Random(args.seed * 100003 + s * 1009 + k)
    return f"{r.choice(A)}{r.choice(B)}-{r.randrange(1000, 9999)}"


SYSTEM = ("You are a careful assistant. A conversation may hold code words. When asked, repeat every code word you were "
          "given in THIS conversation exactly, one a line, and nothing else. " + filler(random.Random(99), 1500))
codes = {s: [code(s, k) for k in range(args.turns)] for s in range(args.sessions)}
everyone = {c: s for s, cs in codes.items() for c in cs}
assert len(everyone) == args.sessions * args.turns, "two sessions drew the same word: take another --seed"


def chat(messages, drop_after=None):
    conn = (http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection)(url.netloc, timeout=3600)
    body = {"model": args.model, "messages": messages, "max_tokens": args.max_tokens, "temperature": 0,
            "stream": True, **({"logprobs": True, "top_logprobs": args.top_logprobs} if args.top_logprobs else {}), **EXTRA}
    head = {"Content-Type": "application/json"}
    if KEY:
        head["Authorization"] = "Bearer " + KEY
    conn.request("POST", url.path + "/chat/completions", json.dumps(body), head)
    resp = conn.getresponse()
    if resp.status != 200:
        text = resp.read()[:300]
        conn.close()
        raise RuntimeError(f"HTTP {resp.status}: {text!r}")
    text, think, n, done, read, probs = [], [], 0, False, {}, []
    for raw in resp:
        if not raw.startswith(b"data: "):
            continue
        if raw.strip() == b"data: [DONE]":
            done = True
            break
        d = json.loads(raw[6:])
        if d.get("error"):  # an error the server sends inside the stream
            conn.close()
            raise RuntimeError(f"server error: {str(d['error'])[:200]}")
        if d.get("timings"):  # llama-server: the prompt tokens it evaluated and the ones it took from its cache
            read = {k: d["timings"][k] for k in ("prompt_n", "cache_n", "predicted_n") if k in d["timings"]}
        for ch in d.get("choices") or []:
            delta = ch.get("delta") or {}
            if delta.get("content"):
                text.append(delta["content"])
                n += 1
                for t in (ch.get("logprobs") or {}).get("content") or []:  # the reply's tokens, not the thinking's
                    top = [[x.get("token"), round(math.exp(x["logprob"]), 4)] for x in t.get("top_logprobs") or [] if x.get("logprob") is not None]
                    probs.append(sorted(top, key=lambda x: -x[1]))
            if delta.get("reasoning_content"):
                think.append(delta["reasoning_content"])
                n += 1
            if ch.get("finish_reason"):
                done = True
                read["finish"] = ch["finish_reason"]  # "length": the reply was cut at --max-tokens, which reads as LOST
        if drop_after is not None and n >= drop_after and not done:
            break
    conn.close()
    if probs:
        read["probs"] = probs[:64]  # a reply token a row: its likeliest tokens and their probabilities
    return "".join(text), "".join(think), done, read


def closest(probs):  # the reply token whose two likeliest tokens are nearest: where the reply could have gone the other way
    rows = [(top[0][1] - top[1][1], i) for i, top in enumerate(probs) if len(top) > 1]
    if not rows:
        return ""
    i = min(rows)[1]
    (a, pa), (b, pb) = probs[i][:2]
    return f"; closest pick: reply token {i}, {a!r} at {pa:.3f} over {b!r} at {pb:.3f}"


log, lock = [], threading.Lock()


def session(s):
    rng = random.Random(args.seed * 7907 + s)
    time.sleep(rng.uniform(0.0, 2.0))
    first = rng.randrange(args.filler, args.filler_max + 1) if args.filler_max > args.filler else args.filler
    messages = [{"role": "system", "content": SYSTEM}]
    for k in range(args.turns):
        body = filler(rng, first if k == 0 else args.grow)
        half = len(body) // 2
        messages.append({"role": "user", "content": f"{body[:half]} Code word number {k + 1} of this conversation is "
                         f"{codes[s][k]}. {body[half:]} Now repeat every code word of this conversation, one a line."})
        while True:
            drop = rng.randrange(1, 6) if rng.random() < args.drop else None
            t0 = time.time()
            try:
                reply, think, done, read = chat(messages, drop)
            except Exception as err:  # the server gone, a timeout: said, and the turn asked again a few times
                with lock:
                    log.append({"session": s, "turn": k, "error": str(err)[:200]})
                    print(f"isolation: session {s} turn {k}: ERROR {str(err)[:160]}", flush=True)
                time.sleep(5)
                if sum(1 for e in log if e.get("session") == s and e.get("turn") == k and "error" in e) >= 3:
                    return
                continue
            if done:
                break
            if drop is None:  # the stream ended without a finish and was not dropped here: the server cut it
                with lock:
                    log.append({"session": s, "turn": k, "error": "stream ended without a finish"})
                    print(f"isolation: session {s} turn {k}: ERROR the stream ended without a finish", flush=True)
                if sum(1 for e in log if e.get("session") == s and e.get("turn") == k and "error" in e) >= 3:
                    return
                time.sleep(5)
                continue
            with lock:
                log.append({"session": s, "turn": k, "dropped": True})
            time.sleep(rng.uniform(0.0, 0.5))
        own = codes[s][:k + 1]
        leak = sorted(c for c, o in everyone.items() if o != s and (c in reply or c in think))
        lost = [c for c in own if c not in reply]
        with lock:
            log.append({"session": s, "turn": k, "seconds": round(time.time() - t0, 1), "leak": leak, "lost": lost,
                        "reply": reply[:400], "think_leak": [c for c in leak if c in think and c not in reply], **read,
                        **({"think": think[-600:]} if leak or lost else {})})  # what a turn that failed was thinking
            flag = ("  <== LEAK from session " + ",".join(str(everyone[c]) for c in leak) if leak else "") + \
                   ("  <== LOST " + ",".join(lost) if lost else "")
            size = f", read {read['prompt_n']} tokens and took {read.get('cache_n', 0)} from the cache" if "prompt_n" in read else ""
            print(f"isolation: session {s} turn {k} ({time.time() - t0:.1f} s): {len(own) - len(lost)} of {len(own)} own "
                  f"words back, {len(leak)} of another session's{size}{flag}"
                  f"{closest(read.get('probs', [])) if leak or lost else ''}", flush=True)
        messages.append({"role": "assistant", "content": reply})
        time.sleep(rng.uniform(0.0, 1.0))


threads = [threading.Thread(target=session, args=(s,)) for s in range(args.skip, args.sessions)]
t0 = time.time()
for th in threads:
    th.start()
for th in threads:
    th.join()
turns = [e for e in log if "leak" in e]
n_leak = sum(bool(e["leak"]) for e in turns)
n_lost = sum(bool(e["lost"]) for e in turns)
print(f"isolation: {len(turns)} turns of {args.sessions - args.skip} sessions in {time.time() - t0:.0f} s, "
      f"{sum(1 for e in log if e.get('dropped'))} requests dropped part-way, {sum(1 for e in log if 'error' in e)} errors")
print(f"isolation: turns with ANOTHER session's code word: {n_leak}; turns that lost a word of their own: {n_lost}")
if args.out:
    json.dump({"args": vars(args), "codes": codes, "log": log}, open(args.out, "w"), indent=1)
raise SystemExit(1 if n_leak or n_lost else 0)
