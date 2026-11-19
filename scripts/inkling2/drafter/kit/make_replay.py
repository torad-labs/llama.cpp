"""Turn the continuations of a recording into prompts, so they can be read back and compared.

A continuation is written by the model one token a step, after its request was restored to a cut. Whether its states
are right is asked here the direct way: for every continuation, a request made of the prompt up to the cut and the
tokens the model wrote. Read with no continuation at all, its last rows must be the states the recording holds.

Writes requests.jsonl ({id, tokens}) and plan.jsonl (the recording's plan header, one request a continuation, no cuts).

Usage: make_replay.py --rec DIR --plan PLAN --out DIR
"""
import argparse
import json
import os

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rec", required=True, help="the recording whose continuations are read back")
    p.add_argument("--plan", required=True, help="the plan it was recorded from")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    header = None
    with open(args.plan, encoding="utf-8") as f:
        for line in f:
            if line.strip() and json.loads(line)["type"] == "header":
                header = json.loads(line)
                break
    if header is None:
        raise SystemExit("%s has no header" % args.plan)

    os.makedirs(args.out, exist_ok=True)
    n = rows = 0
    with open(os.path.join(args.rec, "index.jsonl"), encoding="utf-8") as index, \
            open(os.path.join(args.out, "requests.jsonl"), "w", encoding="utf-8") as requests, \
            open(os.path.join(args.out, "plan.jsonl"), "w", encoding="utf-8") as plan:
        plan.write(json.dumps(header) + "\n")
        for line in index:
            if not line.strip():
                continue
            e = json.loads(line)
            tokens = np.fromfile(os.path.join(args.rec, "req", e["file"] + ".tokens"), dtype=np.int32)
            if tokens.size != e["n_tokens"]:
                raise SystemExit("%s: %d tokens on disk, %d in the index" % (e["id"], tokens.size, e["n_tokens"]))
            for s in e["segments"]:
                if s["n_states"] == 0:
                    continue
                # tokens[j] of a continuation stands at position cut + j; the last one was drawn and never read
                read = tokens[:s["cut"]].tolist() + [int(t) for t in s["tokens"][:s["n_states"]]]
                rid = "%s@%d" % (e["id"], s["cut"])
                requests.write(json.dumps({"id": rid, "tokens": read}) + "\n")
                plan.write(json.dumps({"type": "request", "id": rid, "file": "%s.c%d" % (e["file"], s["cut"]),
                                       "n_tokens": len(read), "shared_len": e["shared_len"], "cuts": []}) + "\n")
                n += 1
                rows += len(read) - e["shared_len"]
    print("replay: %d continuations as prompts, %d rows to read" % (n, rows))


if __name__ == "__main__":
    main()
