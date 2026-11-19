"""Carry the trials of a sweep on the rented box to MLflow and to plain logs, while they run.

  mlflow_relay.py SWEEP_DIR LOGS_DIR PREFIX EXPERIMENT OUT_DIR SWEEP_ARGS

SWEEP_DIR and LOGS_DIR are on the box (a folder a trial, each with train.jsonl and summary.json; sweep-TRIAL.log).
Every step's line becomes a point of the trial's MLflow run, PREFIX-TRIAL, and the trial's log is copied as it grows
to OUT_DIR/train-TRIAL.log. MLflow's address and login come from this machine's environment and go nowhere else.
"""
import json, math, os, subprocess, sys, time, traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlf  # noqa: E402

# the rented box: rig's id for it, and the ssh address `rig vast status` gives (user@host:port)
BOX = os.environ["DSPARK_BOX"]
BOX_USER_HOST, BOX_PORT = os.environ["DSPARK_BOX_SSH"].rsplit(":", 1)
KH = os.path.expanduser("~/.local/share/rig/local/rented-box/boxes/%s/known_hosts" % BOX)
SSH = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "UserKnownHostsFile=" + KH,
       "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2", "-p", BOX_PORT, BOX_USER_HOST]
MARK = "@@relay "
STEP_KEYS = {"loss": "train/loss", "kd": "train/kd", "mode": "train/first_choice", "lr": "train/lr",
             "grad_norm": "train/grad_norm", "seconds": "train/step_seconds", "anchors_per_s": "train/anchors_per_s",
             "context": "train/context", "epoch": "train/epoch", "elapsed": "train/elapsed_s"}


def remote(cmd: str) -> bytes:
    try:
        return subprocess.run(SSH + [cmd], capture_output=True, timeout=90).stdout
    except subprocess.TimeoutExpired:
        return b""


def sections(raw: bytes) -> dict:
    """{(kind, trial): bytes} of an answer whose parts each begin with a line '@@relay KIND TRIAL'."""
    out, key, buf = {}, None, []
    for line in raw.split(b"\n"):
        if line.startswith(MARK.encode()):
            if key:
                out[key] = b"\n".join(buf)
            words = line.decode().split()
            key, buf = (words[1], words[2]), []
        elif key:
            buf.append(line)
    if key:
        out[key] = b"\n".join(buf)
    return out


def number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def post(run_id: str, points: list) -> None:
    for s in range(0, len(points), 800):
        mlf.call("POST", "runs/log-batch", {"run_id": run_id, "metrics": points[s:s + 800]})


HERE = os.path.dirname(os.path.abspath(__file__))


def note(text: str) -> None:
    """A line on the screen and in relay.log, which outlives the seat."""
    print(text, flush=True)
    with open(os.path.join(HERE, "relay.log"), "a", encoding="utf-8") as f:
        f.write(time.strftime("%H:%M:%S ") + text + "\n")


def main() -> None:
    sweep, logs, prefix, experiment, out_dir, sweep_args = sys.argv[1:7]
    os.makedirs(out_dir, exist_ok=True)
    words = {}
    with open(sweep_args, encoding="utf-8") as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                name, *rest = line.split()
                words[name] = rest
    exp = mlf.experiment(experiment)
    print("experiment %s: %s/#/experiments/%s" % (experiment, mlf.BASE, exp), flush=True)
    state_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state-%s.json" % prefix)
    state = json.load(open(state_path)) if os.path.isfile(state_path) else {}
    quiet = 0
    while True:
        cmd = "cd %s 2>/dev/null || exit 0; for t in */; do echo '%slist' \"${t%%/}\"; done; " % (sweep, MARK)
        for t, st in state.items():
            cmd += ("echo '%sjsonl %s'; tail -c +%d %s/train.jsonl 2>/dev/null; echo; echo '%slog %s'; tail -c +%d %s/sweep-%s.log 2>/dev/null; echo; "
                    "echo '%ssummary %s'; cat %s/summary.json 2>/dev/null; echo; " % (
                        MARK, t, st["jsonl"] + 1, t, MARK, t, st["log"] + 1, logs, t, MARK, t, t))
        got = remote(cmd)
        moved = False
        for line in got.split(b"\n"):
            if line.startswith((MARK + "list ").encode()):
                t = line.decode().split()[2]
                if t not in state and t != "*":
                    params = [{"key": "trial", "value": t}, {"key": "sweep", "value": sweep}]
                    rest = words.get(t, [])
                    for k in range(len(rest)):
                        if rest[k].startswith("--"):
                            value = rest[k + 1] if k + 1 < len(rest) and not rest[k + 1].startswith("--") else "true"
                            params.append({"key": rest[k][2:], "value": value[:250]})
                    run = mlf.call("POST", "runs/create", {"experiment_id": exp, "run_name": "%s-%s" % (prefix, t),
                                                           "start_time": int(time.time() * 1000),
                                                           "tags": [{"key": "mlflow.runName", "value": "%s-%s" % (prefix, t)}]})
                    rid = run["run"]["info"]["run_id"]
                    mlf.call("POST", "runs/log-batch", {"run_id": rid, "params": params})
                    state[t] = {"run": rid, "jsonl": 0, "log": 0, "done": False, "steps": 0}
                    note("trial %s: run %s" % (t, rid))
                    moved = True
        parts = sections(got)

        def save():
            with open(state_path + ".tmp", "w") as f:
                json.dump(state, f)
            os.replace(state_path + ".tmp", state_path)

        def one(t, st):
            nonlocal moved
            raw = parts.get(("jsonl", t), b"")
            cut = raw.rfind(b"\n")            # whole lines only; the answer ends with a line of its own
            body = raw[:cut] if cut >= 0 else b""
            points = []
            used = 0
            for line in body.split(b"\n"):
                if not line.strip():
                    used += len(line) + 1
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    break
                used += len(line) + 1
                stamp = int(r.get("time", time.time()) * 1000)
                step = int(r.get("step", 0))
                if "eval" in r:
                    for k, v in (r["eval"] or {}).items():
                        if number(v):
                            points.append({"key": "eval/" + k, "value": v, "timestamp": stamp, "step": step})
                    continue
                for k, name in STEP_KEYS.items():
                    if number(r.get(k)):
                        points.append({"key": name, "value": r[k], "timestamp": stamp, "step": step})
                if number(r.get("eta_s")):
                    points.append({"key": "train/eta_min", "value": r["eta_s"] / 60, "timestamp": stamp, "step": step})
                if number(r.get("total")) and r["total"]:
                    points.append({"key": "train/progress", "value": step / r["total"], "timestamp": stamp, "step": step})
                st["steps"] = step
                st["total"] = r.get("total", 0)
            # the body's own bytes: never past the file's last whole line
            used = min(used, len(body) + 1) if body else 0
            if points:
                post(st["run"], points)
            if used:
                st["jsonl"] += used
                moved = True
                save()
            raw = parts.get(("log", t), b"")
            cut = raw.rfind(b"\n")
            body = raw[:cut] if cut >= 0 else b""
            whole = body + b"\n" if cut >= 0 else b""
            if whole:
                with open(os.path.join(out_dir, "train-%s.log" % t), "ab") as f:
                    f.write(whole)
                st["log"] += len(whole)
                moved = True
                save()
            raw = parts.get(("summary", t), b"").strip()
            if raw:
                try:
                    summary = json.loads(raw)
                except ValueError:
                    summary = {}
                if "after" in summary:
                    stamp = int(time.time() * 1000)
                    points = []
                    for side in ("before", "after", "walked_before", "walked_after"):
                        for k, v in (summary.get(side) or {}).items():
                            if number(v):
                                points.append({"key": "final/%s/%s" % (side, k), "value": v, "timestamp": stamp, "step": st["steps"]})
                    for k in ("steps", "seconds", "loss_first", "loss_last", "paired"):
                        if number(summary.get(k)):
                            points.append({"key": "final/" + k, "value": summary[k], "timestamp": stamp, "step": st["steps"]})
                    post(st["run"], points)
                    mlf.call("POST", "runs/update", {"run_id": st["run"], "status": "FINISHED", "end_time": stamp})
                    st["done"] = True
                    moved = True
                    print("trial %s: finished" % t, flush=True)
        for t, st in state.items():
            if st["done"]:
                continue
            try:
                one(t, st)
            except (Exception, SystemExit):
                note("trial %s: %s" % (t, traceback.format_exc().strip().splitlines()[-1][:300]))
        save()
        print(time.strftime("%H:%M:%S"), " ".join("%s %s/%s%s" % (t, st["steps"], st.get("total", "?"), " done" if st["done"] else "") for t, st in sorted(state.items())) or "no trial yet", flush=True)
        if state and all(st["done"] for st in state.values()):
            break
        quiet = 0 if moved else quiet + 1
        if quiet > 720:     # two hours with nothing new
            print("nothing new for two hours: ending", flush=True)
            break
        time.sleep(10)


if __name__ == "__main__":
    main()
