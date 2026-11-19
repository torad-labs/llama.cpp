"""What a draft gets on the live server as sessions are added, and how fast the server writes with it.

Starts llama-server as the launcher serves Inkling (every slot it serves with, sharing one pool), with a draft or with
none, and sends it sessions cut from the held-out recording: the longest context of each of N requests that fits, each
on a slot it keeps. First every session reads its context at once and writes a reply, as a session's first turn does
(the server may copy a prefix another session has read, as it does when it serves). Then, for each level L, every
session writes a reply again, L of them at a time, with its context found in the slot's cache as a session's later
turns find theirs: the server goes back to the checkpoint it took near the context's end and reads the last few hundred
tokens again. A session that ends gives its place to the next, so L sessions are writing until the last few. Every
level runs the same sessions, so the levels compare like for like.

The reply's timings say how many draft tokens the server offered and kept, in how many rounds, and how fast it wrote.
The server samples as it is served (the launcher's temperature), so a reply differs from run to run: the shares compare
over the sessions together, not one by one. The same sessions with another draft, or with none, compare the drafts;
nothing here is used to train on.

Usage: live_bench.py --source DIR --server BIN (--model GGUF | --packs DIR) [--draft GGUF] --rec DIR --out JSON
                     --log FILE [--slots 23] [--levels 23,8,4,1] [--predict 256] [--depth N] [--most TOKENS]
                     [--port 8117]
"""
import argparse
import importlib.util
import json
import os
import random
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from live_check import spread, wait_ready  # noqa: E402

SUMS = ("prompt", "read", "cached", "written", "offered", "kept", "rounds")


def say(text: str) -> None:
    print(text, flush=True)


def server_command(args):
    """The launcher's own command and environment, with this draft or with none."""
    spec = importlib.util.spec_from_file_location("inkling2_serve", args.source / "scripts/inkling2/serve.py")
    serve = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(serve)
    asked = ["--profile", "tensor-4-draft", "--text-only", "--server", str(args.server), "--slots", str(args.slots),
             "--port", str(args.port)]
    asked += ["--draft-model", str(args.draft)] if args.draft else ["--no-draft"]
    if args.draft and args.depth:
        asked += ["--draft-depth", str(args.depth)]
    if args.verbosity is not None:
        asked += ["--log-verbosity", str(args.verbosity)]
    asked += ["--model", str(args.model)] if args.model else ["--packs", str(args.packs)]
    argv, settings = serve.make_plan(serve.arguments(asked))
    serve.glm53.verify_assets({serve.TEMPLATE: serve.TEMPLATE_SHA256})
    env = dict(serve.execution_environment(settings))
    for pair in args.env:
        name, sep, value = pair.partition("=")
        if not sep or not name:
            raise SystemExit("--env takes NAME=VALUE, not %s" % pair)
        env[name] = value
    paths = [("server", Path(argv[0])), ("model", Path(argv[argv.index("-m") + 1]))]
    if args.draft:
        paths.append(("draft", Path(argv[argv.index("--spec-draft-model") + 1])))
    for label, path in paths:
        if not path.is_file():
            raise SystemExit("missing %s: %s" % (label, path))
    return argv, env


def complete(port: int, prompt: list, predict: int, seed: int, slot: int) -> dict:
    body = json.dumps({"prompt": prompt, "n_predict": predict, "return_tokens": True, "cache_prompt": True,
                       "id_slot": slot, "stream": False, "seed": seed}).encode()
    request = urllib.request.Request("http://127.0.0.1:%d/completion" % port, data=body,
                                     headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(request, timeout=None) as r:
            reply = json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise SystemExit("slot %d: the server answered %d: %s" % (slot, e.code, e.read()[:600].decode("utf-8", "replace")))
    reply["seconds"] = time.time() - t0
    return reply


def pick(rec, count: int, cap: int) -> list:
    """The longest context of each request that fits `cap` tokens; `count` of them, from the shortest to the longest."""
    found = []
    for e in rec.entries:
        cuts = [int(s["cut"]) for s in e.get("segments", [])]
        cuts = [c for c in cuts if int(e["shared_len"]) < c <= cap]
        if cuts:
            found.append({"id": e["id"], "cut": max(cuts), "entry": e})
    found.sort(key=lambda s: (s["cut"], s["id"]))
    return [found[i] for i in spread(len(found), count)]


def run(port: int, prompts: list, at_once: int, predict: int, seed: int) -> tuple:
    """Every session once on its own slot, `at_once` of them at a time. The replies in session order, and the wall time."""
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=at_once) as pool:
        futures = [pool.submit(complete, port, prompt, predict, seed + slot, slot)
                   for slot, prompt in enumerate(prompts)]
        replies = [f.result() for f in futures]
    return replies, time.time() - t0


def row_of(slot: int, session: dict, prompt: list, reply: dict) -> dict:
    t = reply.get("timings", {})
    if reply.get("truncated"):
        raise SystemExit("%s: the server cut a prompt of %d tokens short" % (session["id"], len(prompt)))
    if int(reply.get("tokens_evaluated", -1)) != len(prompt):
        raise SystemExit("%s: the server read %s tokens of a prompt of %d" % (session["id"], reply.get("tokens_evaluated"), len(prompt)))
    return {"slot": slot, "id": session["id"], "prompt": len(prompt), "read": int(t.get("prompt_n", 0)),
            "cached": int(t.get("cache_n", 0)), "read_ms": float(t.get("prompt_ms", 0.0)),
            "written": int(t.get("predicted_n", 0)), "written_ms": float(t.get("predicted_ms", 0.0)),
            "per_second": float(t.get("predicted_per_second", 0.0)), "offered": int(t.get("draft_n", 0)),
            "kept": int(t.get("draft_n_accepted", 0)), "rounds": int(t.get("draft_rounds", 0)),
            "stop": reply.get("stop_type"), "seconds": reply["seconds"], "timings": t}


def total(rows: list, wall: float = 0.0) -> dict:
    s = {k: sum(r[k] for r in rows) for k in SUMS}
    s["sessions"] = len(rows)
    s["wall"] = wall
    s["kept_share"] = s["kept"] / s["offered"] if s["offered"] else 0.0
    s["kept_a_round"] = s["kept"] / s["rounds"] if s["rounds"] else 0.0
    s["drafted_share"] = s["kept"] / s["written"] if s["written"] else 0.0
    speeds = sorted(r["per_second"] for r in rows if r["written"] > 1)
    s["per_second_median"] = speeds[len(speeds) // 2] if speeds else 0.0
    s["per_second_least"] = speeds[0] if speeds else 0.0
    s["per_second_all"] = s["written"] / wall if wall else 0.0
    return s


def line(label: str, s: dict) -> str:
    text = "%-24s %2d sessions  kept %5d of %5d offered (%5.1f%%)  %.3f kept a round  %4.1f%% of %5d written came drafted  %6.1f tok/s a session (least %5.1f)" % (
        label, s["sessions"], s["kept"], s["offered"], 100 * s["kept_share"], s["kept_a_round"], 100 * s["drafted_share"],
        s["written"], s["per_second_median"], s["per_second_least"])
    if s["wall"]:
        text += "  %7.1f tok/s in all  read %7d cached %7d  %4.0fs" % (s["per_second_all"], s["read"], s["cached"], s["wall"])
    return text


def session_lines(label: str, rows: list) -> None:
    for r in sorted(rows, key=lambda r: r["prompt"]):
        say("%s  slot %2d  context %6d  kept %4d of %4d (%5.1f%%)  %3d of %3d written came drafted  %6.1f tok/s  stop %s" % (
            label, r["slot"], r["prompt"], r["kept"], r["offered"], 100.0 * r["kept"] / r["offered"] if r["offered"] else 0.0,
            r["kept"], r["written"], r["per_second"], r["stop"]))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, type=Path, help="the engine source tree")
    p.add_argument("--server", required=True, type=Path, help="llama-server, beside the engine's libraries")
    where = p.add_mutually_exclusive_group(required=True)
    where.add_argument("--model", type=Path, help="a GGUF in place of the profile's pack (the rehearsal fixture)")
    where.add_argument("--packs", type=Path, help="the folder holding the profile's pack")
    p.add_argument("--draft", type=Path, help="the draft GGUF, with its own vocabulary tensors (none: no drafting)")
    p.add_argument("--rec", required=True, type=Path, help="the held-out recording the sessions are cut from")
    p.add_argument("--out", required=True, help="the results, as JSON")
    p.add_argument("--log", required=True, help="the server's log")
    p.add_argument("--slots", type=int, default=23, help="the slots the server has, and the sessions sent")
    p.add_argument("--levels", default="23,8,4,1", help="sessions writing at once, in the order they are run")
    p.add_argument("--predict", type=int, default=256, help="tokens a reply, at most")
    p.add_argument("--depth", type=int, default=0, help="draft tokens a round (0: the launcher's own)")
    p.add_argument("--verbosity", type=int, help="the server's log level (4: the draft code's time totals at each reply's end)")
    p.add_argument("--env", action="append", default=[], metavar="NAME=VALUE",
                   help="a variable for the server's environment, for a probe (never for a run whose speed is quoted)")
    p.add_argument("--most", type=int, default=0, help="longest context taken, in tokens (0: what a slot holds)")
    p.add_argument("--port", type=int, default=8117)
    p.add_argument("--seed", type=int, default=20261119)
    p.add_argument("--load-minutes", type=float, default=90)
    args = p.parse_args()

    from dspark_data import Recording

    levels = [int(x) for x in args.levels.split(",") if x]
    if not levels or max(levels) > args.slots or min(levels) < 1:
        raise SystemExit("levels are from 1 to the %d slots" % args.slots)
    argv, env = server_command(args)
    pool, slot_cells = int(argv[argv.index("-c") + 1]), int(env["LLAMA_KV_SEQ_CELLS"])
    room = args.predict + 64
    cap = slot_cells - room
    if args.most:
        cap = min(cap, args.most)
    rec = Recording(str(args.rec))
    while True:
        sessions = pick(rec, args.slots, cap)
        if len(sessions) < args.slots:
            raise SystemExit("%s holds %d requests with a continuation within %d tokens, %d sessions asked" % (
                args.rec, len(sessions), cap, args.slots))
        if sum(s["cut"] + room for s in sessions) <= pool:
            break
        cap = int(cap * 0.8)
    # mixed, so the sessions a level starts with are of every length
    random.Random(args.seed).shuffle(sessions)
    prompts = [rec.tokens(s["entry"])[:s["cut"]].tolist() for s in sessions]
    say("bench: %d sessions of %d to %d tokens (%d in all, none over %d), levels %s, %d tokens a reply, draft %s" % (
        len(prompts), min(map(len, prompts)), max(map(len, prompts)), sum(map(len, prompts)), cap, levels, args.predict,
        args.draft or "none"))

    os.makedirs(os.path.dirname(os.path.abspath(args.log)), exist_ok=True)
    with open(args.log, "wb") as log:
        proc = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    result = {"draft": str(args.draft or ""), "slots": args.slots, "predict": args.predict, "argv": argv, "env": args.env,
              "depth": int(argv[argv.index("--spec-draft-n-max") + 1]) if args.draft else 0,
              "sessions": [{"slot": i, "id": s["id"], "cut": s["cut"]} for i, s in enumerate(sessions)], "levels": []}

    def measured(label: str, at_once: int, later: bool, seed: int) -> dict:
        replies, wall = run(args.port, prompts, at_once, args.predict, seed)
        if proc.poll() is not None:
            raise SystemExit("the server ended with %d during %s" % (proc.returncode, label))
        rows = [row_of(slot, sessions[slot], prompts[slot], reply) for slot, reply in enumerate(replies)]
        entry = {"label": label, "at_once": at_once, "later": later, "total": total(rows, wall), "rows": rows}
        say(line(label, entry["total"]))
        result["levels"].append(entry)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f)
        if later and 4 * entry["total"]["read"] > entry["total"]["prompt"]:
            session_lines(label, rows)
            raise SystemExit("%s: the server read %d of %d context tokens again, where the slots' caches should hold them" % (
                label, entry["total"]["read"], entry["total"]["prompt"]))
        return entry

    try:
        t0 = time.time()
        wait_ready(proc, args.port, args.load_minutes)
        result["load_seconds"] = time.time() - t0
        say("bench: the server was ready in %.0fs" % result["load_seconds"])
        first = measured("first turn, %d at once" % len(prompts), len(prompts), False, args.seed)
        session_lines("first turn", first["rows"])
        for number, level in enumerate(levels):
            entry = measured("later turn, %d at once" % level, level, True, args.seed + 1000 * (number + 1))
            if number == 0:
                session_lines("later turn, %d at once" % level, entry["rows"])
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    # by the context's length, over every later turn: does a long session keep less
    later = [r for entry in result["levels"] if entry["later"] for r in entry["rows"]]
    edges = sorted(len(prompt) for prompt in prompts)
    thirds = [edges[len(edges) // 3], edges[2 * len(edges) // 3]]
    for label, rows in (("contexts under %d" % thirds[0], [r for r in later if r["prompt"] < thirds[0]]),
                        ("contexts %d to %d" % (thirds[0], thirds[1]), [r for r in later if thirds[0] <= r["prompt"] < thirds[1]]),
                        ("contexts from %d" % thirds[1], [r for r in later if r["prompt"] >= thirds[1]])):
        if rows:
            say(line(label, total(rows)))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f)
    say("bench: written to %s" % args.out)


if __name__ == "__main__":
    main()
