"""Sessions sent the way a coding agent sends them, to the server as it is served: does the draft keep less as a session
goes on, or in the sessions that start last.

A session is a request of the held-out recording sent turn by turn: a turn's prompt is the request up to one of its
cuts, so each turn extends the one before it, as a conversation does. No slot is named. The server gives a turn the
slot whose cache holds most of its prompt, and where the launcher turns it on a new session starts from the prefix
another slot already read. The first session writes its first turn alone; the others then start a few seconds apart,
and each goes through its turns as fast as the server answers.

  live_replay.py --source ENGINE --server BIN/llama-server (--packs DIR | --model GGUF) [--draft GGUF] --rec DIR
                 --out JSON --log FILE [--slots 23] [--sessions 23] [--turns 8] [--predict 128] [--stagger 3]

Prints what the draft kept by the order the sessions started in, by the turn's number, by whether a session began
from another's prefix and by the context's length, a line a session, and the turns that kept least.
"""
import argparse
import json
import os
import random
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from live_bench import pick, row_of, say, server_command, total  # noqa: E402
from live_check import spread, wait_ready  # noqa: E402


def complete(port: int, prompt: list, predict: int, seed: int) -> dict:
    body = json.dumps({"prompt": prompt, "n_predict": predict, "return_tokens": True, "cache_prompt": True,
                       "stream": False, "seed": seed}).encode()
    request = urllib.request.Request("http://127.0.0.1:%d/completion" % port, data=body,
                                     headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(request, timeout=None) as r:
            reply = json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise SystemExit("the server answered %d: %s" % (e.code, e.read()[:600].decode("utf-8", "replace")))
    reply["seconds"] = time.time() - t0
    return reply


def line(label: str, rows: list) -> str:
    s = total(rows)
    return "%-42s %3d turns  kept %5d of %5d offered (%5.1f%%)  %.3f kept a round  %4.1f%% of %5d written came drafted  %6.1f tok/s a turn (least %5.1f)  read %7d cached %8d" % (
        label, len(rows), s["kept"], s["offered"], 100 * s["kept_share"], s["kept_a_round"], 100 * s["drafted_share"],
        s["written"], s["per_second_median"], s["per_second_least"], s["read"], s["cached"])


def share(r: dict) -> float:
    return 100.0 * r["kept"] / r["offered"] if r["offered"] else 0.0


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
    p.add_argument("--slots", type=int, default=23, help="the slots the server has")
    p.add_argument("--sessions", type=int, default=23, help="the sessions sent")
    p.add_argument("--turns", type=int, default=8, help="turns a session, at most: its cuts, evenly apart, the first and the last among them")
    p.add_argument("--predict", type=int, default=128, help="tokens a reply, at most")
    p.add_argument("--stagger", type=float, default=3.0, help="seconds between the starts of the sessions after the first")
    p.add_argument("--depth", type=int, default=0, help="draft tokens a round (0: the launcher's own)")
    p.add_argument("--verbosity", type=int, help="the server's log level")
    p.add_argument("--env", action="append", default=[], metavar="NAME=VALUE",
                   help="a variable for the server's environment, for a probe (never for a run whose speed is quoted)")
    p.add_argument("--most", type=int, default=0, help="longest context taken, in tokens (0: what a slot holds)")
    p.add_argument("--port", type=int, default=8117)
    p.add_argument("--seed", type=int, default=20261119)
    p.add_argument("--load-minutes", type=float, default=90)
    args = p.parse_args()

    from dspark_data import Recording

    if not 1 <= args.sessions <= args.slots:
        raise SystemExit("sessions are from 1 to the %d slots" % args.slots)
    argv, env = server_command(args)
    pool, slot_cells = int(argv[argv.index("-c") + 1]), int(env["LLAMA_KV_SEQ_CELLS"])
    room = args.predict + 64
    cap = slot_cells - room
    if args.most:
        cap = min(cap, args.most)
    rec = Recording(str(args.rec))
    while True:
        sessions = pick(rec, args.sessions, cap)
        if len(sessions) < args.sessions:
            raise SystemExit("%s holds %d requests with a continuation within %d tokens, %d sessions asked" % (
                args.rec, len(sessions), cap, args.sessions))
        if sum(s["cut"] + room for s in sessions) <= pool:
            break
        cap = int(cap * 0.8)
    # mixed, so the sessions that start first and last are of every length
    random.Random(args.seed).shuffle(sessions)
    tokens, turns = [], []
    for s in sessions:
        e = s["entry"]
        cuts = sorted({int(g["cut"]) for g in e.get("segments", []) if int(e["shared_len"]) < int(g["cut"]) <= s["cut"]})
        turns.append([cuts[i] for i in spread(len(cuts), args.turns)])
        tokens.append(rec.tokens(e)[:s["cut"]].tolist())
    say("replay: %d sessions of %d to %d turns, first turns of %d to %d tokens, last of %d to %d (%d in all), %d tokens a reply, draft %s" % (
        len(sessions), min(map(len, turns)), max(map(len, turns)), min(t[0] for t in turns), max(t[0] for t in turns),
        min(t[-1] for t in turns), max(t[-1] for t in turns), sum(t[-1] for t in turns), args.predict, args.draft or "none"))

    os.makedirs(os.path.dirname(os.path.abspath(args.log)), exist_ok=True)
    with open(args.log, "wb") as log:
        proc = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    result = {"draft": str(args.draft or ""), "slots": args.slots, "predict": args.predict, "argv": argv, "env": args.env,
              "sessions": [{"session": i, "id": s["id"], "turns": turns[i]} for i, s in enumerate(sessions)], "rows": []}
    rows, lock = result["rows"], threading.Lock()
    t_zero = [0.0]

    def play(i: int, k0: int) -> None:
        for k in range(k0, len(turns[i])):
            prompt = tokens[i][:turns[i][k]]
            began = time.time() - t_zero[0]
            reply = complete(args.port, prompt, args.predict, args.seed + 1000 * k + i)
            row = row_of(i, sessions[i], prompt, reply)
            row.pop("timings", None)
            row.update(session=i, turn=k, slot=int(reply.get("id_slot", -1)), began=began,
                       added=turns[i][k] - (turns[i][k - 1] if k else 0))
            with lock:
                rows.append(row)

    try:
        t0 = time.time()
        wait_ready(proc, args.port, args.load_minutes)
        result["load_seconds"] = time.time() - t0
        say("replay: the server was ready in %.0fs" % result["load_seconds"])
        t_zero[0] = time.time()
        # the first session's first turn alone, so the sessions after it find a prefix already read
        saved, turns[0] = turns[0], turns[0][:1]
        play(0, 0)
        turns[0] = saved
        say("replay: the first session read %d tokens and wrote its first turn in %.0fs" % (rows[0]["read"], time.time() - t_zero[0]))
        with ThreadPoolExecutor(max_workers=len(sessions)) as pool_:
            futures = []
            for i in range(len(sessions)):
                if i == 0:
                    if len(turns[0]) > 1:
                        futures.append(pool_.submit(play, 0, 1))
                    continue
                time.sleep(args.stagger)
                futures.append(pool_.submit(play, i, 0))
            for f in futures:
                f.result()
        result["seconds"] = time.time() - t_zero[0]
        if proc.poll() is not None:
            raise SystemExit("the server ended with %d during the replay" % proc.returncode)
    finally:
        with lock:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(result, f)
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    rows.sort(key=lambda r: (r["session"], r["turn"]))
    n = len(sessions)
    say("replay: %d turns in %.0fs" % (len(rows), result["seconds"]))
    say(line("every turn", rows))
    say("by the order the sessions started in:")
    for lo, hi in ((0, n // 3), (n // 3, 2 * n // 3), (2 * n // 3, n)):
        part = [r for r in rows if lo <= r["session"] < hi]
        if part:
            say(line("  sessions %d to %d" % (lo + 1, hi), part))
    say("by the turn's number in its session:")
    for k in range(max(r["turn"] for r in rows) + 1):
        part = [r for r in rows if r["turn"] == k]
        say(line("  turn %d" % (k + 1), part))
    firsts = [r for r in rows if r["turn"] == 0]
    say("by how a session began:")
    for label, part in (("  from another's prefix", [r for r in firsts if r["cached"] > 0]),
                        ("  reading all of its prompt", [r for r in firsts if r["cached"] == 0])):
        if part:
            say(line(label + ", first turn", part))
            ids = {r["session"] for r in part}
            later = [r for r in rows if r["session"] in ids and r["turn"] > 0]
            if later:
                say(line(label + ", later turns", later))
    edges = sorted(r["prompt"] for r in rows)
    thirds = [edges[len(edges) // 3], edges[2 * len(edges) // 3]]
    say("by the context's length:")
    for label, part in (("  under %d" % thirds[0], [r for r in rows if r["prompt"] < thirds[0]]),
                        ("  %d to %d" % (thirds[0], thirds[1]), [r for r in rows if thirds[0] <= r["prompt"] < thirds[1]]),
                        ("  from %d" % thirds[1], [r for r in rows if r["prompt"] >= thirds[1]])):
        if part:
            say(line(label, part))
    say("a line a session, in the order they started:")
    for i in range(n):
        part = [r for r in rows if r["session"] == i]
        s = total(part)
        slots = []
        for r in part:
            if not slots or slots[-1] != r["slot"]:
                slots.append(r["slot"])
        say("  session %2d  slot %-12s %2d turns  context %6d to %6d  first turn read %6d cached %6d  kept %5.1f%% then %s  in all %5.1f%% of %5d  %5.1f tok/s" % (
            i + 1, ">".join(map(str, slots)), len(part), part[0]["prompt"], part[-1]["prompt"], part[0]["read"], part[0]["cached"],
            share(part[0]), " ".join("%.0f" % share(r) for r in part[1:]), 100 * s["kept_share"], s["offered"], s["per_second_median"]))
    again = [r for r in rows if r["turn"] > 0 and r["read"] > r["added"] + 64]
    say("replay: %d later turns read more than the tokens their turn added" % len(again))
    for r in sorted(again, key=lambda r: r["added"] - r["read"])[:12]:
        say("  session %2d turn %2d slot %2d  context %6d  added %6d  read %6d  cached %6d  kept %5.1f%% of %4d" % (
            r["session"] + 1, r["turn"] + 1, r["slot"], r["prompt"], r["added"], r["read"], r["cached"], share(r), r["offered"]))
    say("the turns that kept least, of those offered 30 tokens or more:")
    for r in sorted((r for r in rows if r["offered"] >= 30), key=share)[:12]:
        say("  session %2d turn %2d slot %2d  context %6d  read %6d  cached %6d  kept %4d of %4d (%5.1f%%)  %5.1f tok/s  began at %4.0fs  stop %s" % (
            r["session"] + 1, r["turn"] + 1, r["slot"], r["prompt"], r["read"], r["cached"], r["kept"], r["offered"], share(r),
            r["per_second"], r["began"], r["stop"]))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f)
    say("replay: written to %s" % args.out)


if __name__ == "__main__":
    main()
