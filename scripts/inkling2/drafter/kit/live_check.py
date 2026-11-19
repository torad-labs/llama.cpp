"""Hold the trainer's copy of the draft to what the server's own draft proposes.

The trainer computes the draft in PyTorch. This asks the server itself: it starts llama-server as the launcher serves
Inkling with a draft (one session), has it accept every draft token it is offered (--spec-synth-rates of ones, so the
reply holds the draft's own tokens), and then asks the trainer's copy for the same rounds from a recording of the
same tokens. Where the two name the same tokens, what is trained is what is served.

  serve    starts the server, sends prompts cut from a held-out recording, and writes the replies as requests to
           read back (requests.jsonl, plan.jsonl) with the rounds the server's log names (rounds.json)
  compare  reads the recording of those requests and compares each round's draft tokens with the copy's

The server's log is at debug level for this, where each round says how many tokens it kept and where the session
stands after it. Neither that nor the accept-everything switch is ever used for a recording to train on.

Usage: live_check.py serve --source DIR --server BIN (--model GGUF | --packs DIR) --draft GGUF --rec DIR --plan PLAN
                           --out DIR --log FILE [--n 8] [--predict 128] [--port 8117]
       live_check.py compare --draft GGUF (--target GGUF | --tok-norm-eps E) --rec DIR --rounds rounds.json --out JSON
                             [--gguf-py DIR] [--vocab GGUF] [--device D] [--feats-device D]
"""
import argparse
import importlib.util
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ADDED = re.compile(r"add accepted tokens: sampled=(-?\d+), ids\.size=(\d+), n_draft=(\d+)")
KEPT = re.compile(r"accepted\s+(\d+)/\s*(\d+) draft tokens, new n_tokens = (\d+)")
ROTATION = re.compile(r"n_rot_[kv] = ")


def say(text: str) -> None:
    print(text, flush=True)


def spread(count: int, most: int) -> list:
    if most <= 0 or count <= most:
        return list(range(count))
    return sorted({int(round(i * (count - 1) / (most - 1))) for i in range(most)}) if most > 1 else [count // 2]


def plan_header(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip() and json.loads(line)["type"] == "header":
                return json.loads(line)
    raise SystemExit("%s has no header" % path)


def server_command(args):
    """The launcher's own command and environment for one session with this draft, accepting every draft token."""
    spec = importlib.util.spec_from_file_location("inkling2_serve", args.source / "scripts/inkling2/serve.py")
    serve = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(serve)
    asked = ["--profile", "tensor-4-draft", "--text-only", "--server", str(args.server), "--slots", "1",
             "--draft-model", str(args.draft), "--port", str(args.port), "--log-verbosity", "5"]
    asked += ["--model", str(args.model)] if args.model else ["--packs", str(args.packs)]
    argv, settings = serve.make_plan(serve.arguments(asked))
    serve.glm53.verify_assets({serve.TEMPLATE: serve.TEMPLATE_SHA256})
    depth = int(argv[argv.index("--spec-draft-n-max") + 1])
    argv = [*argv, "--spec-synth-rates", ",".join(["1"] * depth)]
    env = dict(serve.execution_environment(settings))
    env["LLAMA_TRACE"] = "1"
    for label, flag in (("server", None), ("model", "-m"), ("draft", "--spec-draft-model")):
        path = Path(argv[0] if flag is None else argv[argv.index(flag) + 1])
        if not path.is_file():
            raise SystemExit("missing %s: %s" % (label, path))
    return argv, env, depth


def get(url: str, timeout: float):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status, r.read()


def wait_ready(proc, port: int, minutes: float) -> None:
    end = time.time() + 60 * minutes
    while time.time() < end:
        if proc.poll() is not None:
            raise SystemExit("the server ended with %d before it was ready" % proc.returncode)
        try:
            status, _ = get("http://127.0.0.1:%d/health" % port, 5)
            if status == 200:
                return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(2)
    raise SystemExit("the server was not ready in %g minutes" % minutes)


def complete(port: int, prompt: list, predict: int, seed: int) -> dict:
    body = json.dumps({"prompt": prompt, "n_predict": predict, "return_tokens": True, "cache_prompt": False,
                       "stream": False, "seed": seed}).encode()
    request = urllib.request.Request("http://127.0.0.1:%d/completion" % port, data=body,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=None) as r:
        return json.loads(r.read())


def settled(path: str, since: int) -> bytes:
    """What the log gained after `since`, once it has stopped growing (the server writes it from a thread of its own)."""
    size = -1
    for _ in range(120):
        now = os.path.getsize(path)
        if now == size:
            break
        size = now
        time.sleep(0.5)
    with open(path, "rb") as f:
        f.seek(since)
        return f.read()


def rounds_of(log: bytes, full: list, n_prompt: int) -> tuple:
    """The rounds a reply's log names: the anchor's position, the draft tokens kept and how many the draft offered.

    A round that kept tokens logs their count (ids, the kept draft tokens and the token drawn after them) and then,
    unless the reply ended inside it, the draft tokens among them and the session's length, which is the position of
    the drawn token. So the anchor stands at that length less the count, and the draft's tokens right after it.
    """
    rounds, odd = [], 0
    pending = None
    for line in log.decode("utf-8", "replace").splitlines():
        m = ADDED.search(line)
        if m:
            pending = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
            continue
        m = KEPT.search(line)
        if not m or pending is None:
            continue
        drawn, size, offered = pending
        pending = None
        kept, total = int(m.group(1)), int(m.group(3))
        n = total - size
        if n < n_prompt or total >= len(full) or full[total] != drawn or kept > size - 1:
            odd += 1
            continue
        if kept:
            rounds.append({"n": n, "anchor": full[n], "drafts": full[n + 1: n + 1 + kept], "offered": offered})
    return rounds, odd


def serve(args) -> None:
    from dspark_data import Recording

    rec = Recording(str(args.rec))
    header = plan_header(str(args.plan))
    cuts = []
    for e in rec.entries:
        for s in e.get("segments", []):
            cuts.append((int(s["cut"]), e["id"], e))
    cuts.sort(key=lambda c: (c[0], c[1]))
    if not cuts:
        raise SystemExit("%s has no continuation to cut a prompt at" % args.rec)
    chosen = [cuts[i] for i in spread(len(cuts), args.n)]
    argv, env, depth = server_command(args)
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.log)), exist_ok=True)
    say("live: %d prompts of %s to %s tokens, %d draft tokens a round" % (len(chosen), chosen[0][0], chosen[-1][0], depth))
    with open(args.log, "wb") as log:
        proc = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    done = []
    try:
        wait_ready(proc, args.port, args.load_minutes)
        with open(args.log, "rb") as f:
            rotation = [line.strip() for line in f.read().decode("utf-8", "replace").splitlines() if ROTATION.search(line)]
        for number, (cut, rid, entry) in enumerate(chosen):
            shared = int(entry["shared_len"])
            if cut <= shared:
                raise SystemExit("%s: a cut at %d inside the shared %d" % (rid, cut, shared))
            prompt = rec.tokens(entry)[:cut].tolist()
            since = os.path.getsize(args.log)
            t0 = time.time()
            reply = complete(args.port, prompt, args.predict, args.seed + number)
            took = time.time() - t0
            if proc.poll() is not None:
                raise SystemExit("the server ended with %d during %s" % (proc.returncode, rid))
            if int(reply.get("tokens_evaluated", -1)) != len(prompt) or reply.get("truncated"):
                raise SystemExit("%s: the server read %s tokens of a prompt of %d (truncated: %s)" % (
                    rid, reply.get("tokens_evaluated"), len(prompt), reply.get("truncated")))
            written = [int(t) for t in reply.get("tokens", [])]
            if not written:
                raise SystemExit("%s: the reply holds no tokens (return_tokens)" % rid)
            full = prompt + written
            rounds, odd = rounds_of(settled(args.log, since), full, len(prompt))
            say("live: %s at %d: %d tokens in %.1fs, %d rounds with %d draft tokens, %d rounds set aside, stop %s" % (
                rid, cut, len(written), took, len(rounds), sum(len(r["drafts"]) for r in rounds), odd, reply.get("stop_type")))
            done.append({"id": "%s@%d.live" % (rid, cut), "file": "%s.live%d" % (entry["file"], cut), "cut": cut,
                         "shared_len": shared, "tokens": full, "rounds": rounds, "set_aside": odd})
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    total = sum(len(d["rounds"]) for d in done)
    if total == 0:
        raise SystemExit("the server's log names no round (it is read at debug level: add accepted tokens / accepted "
                         "N/M draft tokens, new n_tokens); see %s" % args.log)
    with open(os.path.join(args.out, "requests.jsonl"), "w", encoding="utf-8") as requests, \
            open(os.path.join(args.out, "plan.jsonl"), "w", encoding="utf-8") as plan:
        plan.write(json.dumps(header) + "\n")
        for d in done:
            requests.write(json.dumps({"id": d["id"], "tokens": d["tokens"]}) + "\n")
            plan.write(json.dumps({"type": "request", "id": d["id"], "file": d["file"], "n_tokens": len(d["tokens"]),
                                   "shared_len": d["shared_len"], "cuts": []}) + "\n")
    with open(os.path.join(args.out, "rounds.json"), "w", encoding="utf-8") as f:
        json.dump({"depth": depth, "rotation": rotation, "draft": str(args.draft),
                   "requests": [{k: d[k] for k in ("id", "file", "cut", "rounds", "set_aside")} | {"n_tokens": len(d["tokens"])}
                                for d in done]}, f)
    say("live: %d rounds of %d requests written to %s" % (total, len(done), args.out))
    for line in rotation:
        say("live: the server says %s" % line[-120:])


def compare(args) -> None:
    import torch

    import dspark_model
    from dspark_data import Batch, Loaded, Recording
    from dspark_step import Step

    with open(args.rounds, encoding="utf-8") as f:
        asked = json.load(f)
    depth = int(asked["depth"])
    by_id = {r["id"]: r for r in asked["requests"]}
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.backends.cuda.matmul.allow_tf32 = False
    eps = dspark_model.target_eps(args.gguf_py, args.target) if args.target else args.tok_norm_eps
    model = dspark_model.DSpark.from_gguf(args.draft, device, args.gguf_py, args.vocab, eps, args.kv_type,
                                          args.kv_type_swa, args.kv_rot, True)
    say(model.describe())
    rec = Recording(args.rec)
    if rec.taps != model.cfg.target_layers:
        raise SystemExit("%s holds the target's layers %s, the draft reads %s" % (rec.path, list(rec.taps), list(model.cfg.target_layers)))
    if depth + 1 > model.cfg.block_size:
        raise SystemExit("%d draft tokens a round, the block holds %d rows" % (depth, model.cfg.block_size))
    edges = (4096, 32768, 131072)
    modes = {}
    for name, round_x in (("bf16 inputs", True), ("f32 inputs", False)):
        model.round_x = round_x
        same = np.zeros(depth, dtype=np.int64)
        count = np.zeros(depth, dtype=np.int64)
        far_same = np.zeros(len(edges) + 1, dtype=np.int64)
        far_count = np.zeros(len(edges) + 1, dtype=np.int64)
        gap_same, gap_differ, rank_differ, per_request = [], [], [], []
        with torch.no_grad():
            for entry in rec.entries:
                want = by_id.get(entry["id"])
                if want is None or not want["rounds"]:
                    continue
                tokens = rec.tokens(entry)
                if tokens.size != want["n_tokens"]:
                    raise SystemExit("%s: %d tokens recorded, the server wrote %d" % (entry["id"], tokens.size, want["n_tokens"]))
                rounds = sorted(want["rounds"], key=lambda r: r["n"])
                ns = np.array([r["n"] for r in rounds], dtype=np.int64)
                drafts = np.zeros((len(rounds), depth), dtype=np.int64)
                kept = np.zeros((len(rounds), depth), dtype=bool)
                for i, r in enumerate(rounds):
                    if int(tokens[r["n"]]) != r["anchor"]:
                        raise SystemExit("%s: the recording's token at %d is not the round's anchor" % (entry["id"], r["n"]))
                    d = r["drafts"][:depth]
                    drafts[i, :len(d)] = d
                    kept[i, :len(d)] = True
                loaded = Loaded(rec.load(entry, prompt_rows=int(ns.max())), device, feats_device=args.feats_device or None)
                n = torch.from_numpy(ns).to(device)
                want_ids = torch.from_numpy(drafts).to(device)
                batch = Batch(cut=n, a=torch.zeros_like(n), row0=torch.full_like(n, loaded.ctx.c_last),
                              ids=torch.from_numpy(tokens[ns].astype(np.int64)).to(device), prev=want_ids[:, :depth - 1])
                step = Step(model, loaded.ctx, batch, "none", args.cells, args.max_batch, args.chunk)
                emb = model.embed(batch.ids)
                hit = np.zeros((len(rounds), depth), dtype=bool)
                for i, j in step.runs():
                    logits = step.logits(i, j, emb, depth + 1)
                    top = logits.topk(2, dim=-1)
                    mine = top.indices[..., 0]
                    ok = mine == want_ids[i:j]
                    gap = top.values[..., 0] - top.values[..., 1]
                    theirs = logits.gather(-1, want_ids[i:j, :, None])
                    rank = (logits > theirs).sum(-1)
                    k = torch.from_numpy(kept[i:j]).to(device)
                    hit[i:j] = (ok & k).cpu().numpy()
                    gap_same += gap[ok & k].tolist()
                    gap_differ += gap[~ok & k].tolist()
                    rank_differ += rank[~ok & k].tolist()
                same += (hit & kept).sum(0)
                count += kept.sum(0)
                where = np.searchsorted(np.array(edges), ns, side="right")
                for b in range(len(edges) + 1):
                    far_same[b] += int(hit[where == b].sum())
                    far_count[b] += int(kept[where == b].sum())
                per_request.append({"id": entry["id"], "cut": want["cut"], "tokens": int(kept.sum()), "same": int(hit.sum())})
                del loaded, step, batch
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        total = int(count.sum())
        if total == 0:
            raise SystemExit("no round of %s is in %s" % (args.rounds, args.rec))
        modes[name] = {
            "same": float(same.sum() / total), "tokens": total,
            "by_row": [float(s / c) if c else None for s, c in zip(same, count)],
            "by_position": {("below %d" % edges[b] if b < len(edges) else "from %d" % edges[-1]):
                            (float(far_same[b] / far_count[b]) if far_count[b] else None) for b in range(len(edges) + 1)},
            "margin_where_same": float(np.median(gap_same)) if gap_same else None,
            "margin_where_different": float(np.median(gap_differ)) if gap_differ else None,
            "server_token_rank_where_different": float(np.median(rank_differ)) if rank_differ else None,
            "requests": per_request,
        }
        m = modes[name]
        say("%s: the copy names the server's draft token in %.2f%% of %d (rows %s)" % (
            name, 100 * m["same"], total, " ".join("-" if r is None else "%.1f" % (100 * r) for r in m["by_row"])))
        say("    by position: %s" % ", ".join("%s %s" % (k, "-" if v is None else "%.1f%%" % (100 * v)) for k, v in m["by_position"].items()))
        say("    the copy's lead over its second choice: %s where the same, %s where different (the server's token then its choice number %s)" % (
            m["margin_where_same"], m["margin_where_different"], m["server_token_rank_where_different"]))
    best = max(modes, key=lambda k: modes[k]["same"])
    result = {"depth": depth, "draft": args.draft, "rotation": asked.get("rotation", []), "graph": model.describe(),
              "best": best, "least": args.least, "modes": modes}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    say("live: closest with %s, %.2f%% the same (%.0f%% asked)" % (best, 100 * modes[best]["same"], 100 * args.least))
    if modes[best]["same"] < args.least:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--source", required=True, type=Path, help="the engine source tree")
    s.add_argument("--server", required=True, type=Path, help="llama-server, beside the engine's libraries")
    where = s.add_mutually_exclusive_group(required=True)
    where.add_argument("--model", type=Path, help="a GGUF in place of the profile's pack (the rehearsal fixture)")
    where.add_argument("--packs", type=Path, help="the folder holding the profile's pack")
    s.add_argument("--draft", required=True, type=Path, help="the draft GGUF, with its own vocabulary tensors")
    s.add_argument("--rec", required=True, type=Path, help="the held-out recording the prompts are cut from")
    s.add_argument("--plan", required=True, type=Path, help="the plan it was recorded from, for its header")
    s.add_argument("--out", required=True, help="folder for requests.jsonl, plan.jsonl and rounds.json")
    s.add_argument("--log", required=True, help="the server's log")
    s.add_argument("--n", type=int, default=8, help="prompts, evenly apart by length")
    s.add_argument("--predict", type=int, default=128, help="tokens the server writes after each")
    s.add_argument("--port", type=int, default=8117)
    s.add_argument("--seed", type=int, default=20261119)
    s.add_argument("--load-minutes", type=float, default=90)
    c = sub.add_parser("compare")
    c.add_argument("--draft", required=True)
    c.add_argument("--vocab", default="")
    c.add_argument("--target", default="", help="the target GGUF, for the epsilon of its norms")
    c.add_argument("--tok-norm-eps", type=float, default=None)
    c.add_argument("--gguf-py", default="")
    c.add_argument("--rec", required=True, help="the recording of the requests serve wrote")
    c.add_argument("--rounds", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--least", type=float, default=0.9, help="share of draft tokens the copy has to name")
    c.add_argument("--kv-type", default="q8_0")
    c.add_argument("--kv-type-swa", default="f16")
    c.add_argument("--kv-rot", default="auto")
    c.add_argument("--cells", type=int, default=2_500_000)
    c.add_argument("--max-batch", type=int, default=64)
    c.add_argument("--chunk", type=int, default=4096)
    c.add_argument("--device", default="")
    c.add_argument("--feats-device", default="")
    args = parser.parse_args()
    if args.command == "serve":
        serve(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()
