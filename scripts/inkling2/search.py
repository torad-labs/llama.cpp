#!/usr/bin/env python3
"""Search the settings an Inkling 2 profile launches with for the most decode tokens a second.

  search.py run    launch the profile once per trial, read its sessions' streams, keep each trial in <out>/trials.jsonl
  search.py load   read the same streams from a server that already runs (the load a profiler watches)
  search.py propose  fit the model to the trials so far and print the settings it would try next
  search.py explain  print why the fastest trials won, from the ledger each trial recorded
  search.py log    send the trials MLflow has not seen (MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD)
  search.py apply  write the settings the search kept into tuned.json, which serve.py launches the profile with

A setting is a launcher option (--draft-depth) or a variable of tunables.toml (LLAMA_SAMPLE_THREADS). Every setting is read
at start, so a trial is one server start: the search walks one setting at a time and keeps a value that beats the best by --gain.
The method around that walk is the search's own: --depths measures a trial at several context depths and keeps its worst
(search_workload.py), --screen drops the knobs that cannot move the rate before the walk starts (search_screen.py), and
--repeats re-measures a candidate that could win until its repeats agree, keeping their minimum (search_noise.py).
"""
import argparse
import base64
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request

import mixed_prefill
import serve

sys.path.append(str(serve.ROOT / "scripts/glm53"))
import measure as glm_measure

sys.path.append(str(serve.ROOT / "scripts"))
import ledger
import search_evidence
import search_model
import search_noise
import search_screen
import search_workload

# What a search tries, in the order it tries it: draft depth and whether to draft at all move a step's row count, so
# they go before the settings whose best value depends on it.
SPACE = {
    "--draft-depth": ["1", "2", "3", "4", "5", "6", "7"],
    "--draft": ["off", "on"],
    "--draft-layout": ["off", "on"],
    "LLAMA_SAMPLE_THREADS": ["1", "2", "4", "8", "16", "23"],
    "GGML_CUDA_FATTN_REL_DECODE_STREAMS": ["4", "8", "16", "32", "64"],
    "GGML_CUDA_FATTN_REL_DECODE_MEAN_STREAMS": ["64", "128", "256", "512", "1024"],
    "GGML_CUDA_FATTN_REL_DECODE_RUN_WORDS": ["1", "2", "4", "8", "16"],
    "--mean-mask-decode": ["on", "off"],
    "--sort-expert-rows": ["on", "off"],
    "--expert-row-limit": ["8", "16", "23", "32"],
    "--host-owner-rows": ["on", "off"],
}
SUBJECTS = ("merges overlapping time intervals", "parses an ISO 8601 duration", "finds the longest common prefix of a list",
            "rotates a square matrix in place", "counts the islands in a grid", "evaluates a postfix expression",
            "returns the k most frequent words of a text", "validates a Sudoku board", "flattens a nested dictionary",
            "implements a least recently used cache", "finds every anagram group in a word list", "computes a moving average",
            "topologically sorts a dependency graph", "converts an integer to Roman numerals", "finds the median of two sorted lists",
            "serializes a binary tree to a string", "schedules meetings into the fewest rooms", "compresses a string by run length",
            "checks a bracket sequence for balance", "finds the shortest path in a weighted graph", "debounces repeated calls",
            "diffs two text files line by line", "rate-limits requests with a token bucket")


def post(url, route, body, timeout=60):
    with mixed_prefill.post(url, route, body, timeout) as response:
        return json.load(response)


def prompts(url, sessions, context=0):
    """One coding request a session, as the template renders it: text the model continues, so drafts are kept as in use.
    With a context, the request follows about that many tokens of this repository's own scripts, a session its own part."""
    source = "\n".join(path.read_text() for path in sorted((serve.ROOT / "scripts/glm53").glob("*.py"))) if context else ""
    rows = []
    for slot in range(sessions):
        ask = f"Write a Python function that {SUBJECTS[slot % len(SUBJECTS)]}, with a docstring and three unit tests."
        if context:
            # Cut the part by how many characters this tokenizer spends a token on
            part = (source[slot * len(source) // sessions:] + source) * (1 + context * 8 // len(source))
            sample = part[:20000]
            per_token = len(sample) / len(post(url, "/tokenize", {"content": sample})["tokens"])
            ask = "Here is a module of the project.\n\n" + part[:int(context * per_token)] + "\n\n" + ask
        text = post(url, "/apply-template", {"messages": [{"role": "user", "content": ask}]})["prompt"]
        rows.append(post(url, "/tokenize", {"content": text, "add_special": False, "parse_special": True})["tokens"])
    return rows


def read_round(url, rows, tokens, timeout, sampling):
    """Every session generates `tokens` tokens at once; the rate is counted where all of them are decoding."""
    gate = threading.Barrier(len(rows))

    def client(slot):
        # The slot keeps its prompt between rounds, so a round after the first decodes without reading it again
        body = dict(prompt=rows[slot], id_slot=slot, seed=slot, cache_prompt=True, ignore_eos=True, n_predict=tokens,
                    stream=True, return_tokens=True, **sampling)
        gate.wait(timeout=60)
        with mixed_prefill.post(url, "/completion", body, timeout) as response:
            row = glm_measure.read_completion_stream(response, tokens)
        row["slot"] = slot
        return row

    start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(rows)) as pool:
        streams = list(pool.map(client, range(len(rows))))
    seconds = time.monotonic() - start
    window = glm_measure.common_decode_window(streams)
    if not window["available"]:
        raise ValueError("the sessions never decoded at the same time; raise --tokens")
    timings = [s["response"]["timings"] for s in streams]
    drafted = sum(t.get("draft_n", 0) for t in timings)
    ids = hashlib.sha256(json.dumps([s["token_ids"] for s in streams]).encode()).hexdigest()
    return {"tokens_per_second": window["tokens_per_second"], "wall_tokens_per_second": len(rows) * tokens / seconds,
            "draft_accepted": sum(t.get("draft_n_accepted", 0) for t in timings) / drafted if drafted else None,
            "tokens_sha256": ids}


def ledger_of(url):
    """The engine's ledger as the server has it now, or None from a server that keeps none."""
    try:
        found = ledger.load(url)
    except (OSError, ValueError):
        return None
    return found if isinstance(found, dict) and "slots" in found else None


def measure_at(url, sessions, args, context):
    """One context depth: a warm round the rate does not count, then the measured rounds; the rate is their median.
    The engine's ledger over the measured rounds goes with it: where a step's time went, whatever the model."""
    sampling = {} if args.temperature is None else {"temperature": args.temperature}
    rows = prompts(url, sessions, context)
    for _ in range(args.warm_rounds):
        read_round(url, rows, args.tokens, args.timeout, sampling)
    before = ledger_of(url)
    rounds = [read_round(url, rows, args.tokens, args.timeout, sampling) for _ in range(args.rounds)]
    after = ledger_of(url)
    middle = sorted(rounds, key=lambda r: r["tokens_per_second"])[len(rounds) // 2]
    return {**middle, "tokens_per_second": statistics.median(r["tokens_per_second"] for r in rounds),
            "rounds": [r["tokens_per_second"] for r in rounds], "prompt_tokens": statistics.median(len(row) for row in rows),
            "ledger": ledger.summary(ledger.difference(before, after)) if before and after else None}


def measure(url, sessions, args):
    """A trial over every depth --depths names, kept as its worst depth (search_workload.py: a configuration tuned
    on one trace was up to twice as bad on another). With no --depths, the trial is measured at --context alone.
    A depth that failed is a depth the trial did NOT survive: the trial's rate is None and its error says which
    depth failed, because scoring it on the depths that came back would favor the setting that dies deepest."""
    spec = getattr(args, "depths", "")
    depths = search_workload.depths_from(spec) if spec.strip() else [getattr(args, "context", 0)]
    aggregated = search_workload.run_depths(lambda depth: measure_at(url, sessions, args, depth), depths)
    failed = aggregated.get("failed_depths") or []
    if failed and aggregated.get("tokens_per_second") is not None:
        aggregated = {**aggregated, "tokens_per_second": None,
                      "error": "; ".join(f"depth {entry['depth']} failed: {entry['error']}" for entry in failed)}
    return aggregated


def plan(base, settings):
    """A trial's server command and settings: the profile as it launches now with the trial's settings on top."""
    options = [word for name, value in settings.items() if name.startswith("--") for word in serve.option_words(name, value)]
    argv, env = serve.make_plan(serve.arguments(base + options))
    trial_env = {**env, **{name: value for name, value in settings.items() if not name.startswith("--")}}
    # the ledger is off by default so served models pay nothing for it; a search reads it with every trial, so it
    # is asked for here by name (the same explicit opt-in a measurement arm uses)
    trial_env.setdefault("GGML_LEDGER", "1")
    return argv, trial_env


def run_trial(base, settings, folder, args):
    argv, env = plan(base, settings)
    folder.mkdir(parents=True, exist_ok=True)  # a resumed search finds its own repeat folders in place
    (folder / "plan.json").write_text(json.dumps({"argv": argv, "env": env}, indent=1) + "\n")
    url = "http://127.0.0.1:" + argv[argv.index("--port") + 1]
    start = time.monotonic()
    with (folder / "server.log").open("w") as log:
        server = subprocess.Popen(args.wrap + serve.glm53.resolve_server_arguments(argv, env), stdin=subprocess.DEVNULL,
                                  stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                  env=serve.execution_environment(env))
        try:
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f"the server exited {server.returncode} before it was ready")
                if time.monotonic() - start > args.startup_timeout:
                    raise TimeoutError(f"the server was not ready in {args.startup_timeout} s")
                try:
                    with urllib.request.urlopen(url + "/health", timeout=2) as response:
                        if json.load(response).get("status") == "ok":
                            break
                except (OSError, ValueError):
                    pass
                time.sleep(1)
            ready = time.monotonic() - start
            # the measurement's own error (a failed depth) must survive: overwriting it with None turned a
            # failed baseline into a TypeError and marked MLflow runs FINISHED that had measured nothing
            record = {**measure(url, int(argv[argv.index("-np") + 1]), args), "startup_seconds": round(ready, 1)}
            record.setdefault("error", None)
            return record
        except (OSError, RuntimeError, TimeoutError, ValueError) as error:
            return {"tokens_per_second": None, "error": str(error)}
        finally:
            glm_measure.terminate(server, timeout=60)


def key_of(build, settings, measurement=None):
    """What makes a trial already done: the build, the settings, and how it was measured -- re-running a folder with
    other --depths/--tokens/--rounds/--context measures again instead of silently reusing the old shape."""
    return hashlib.sha256(json.dumps([build, sorted(settings.items()),
                                      sorted((measurement or {}).items())]).encode()).hexdigest()[:12]


def trials_of(out):
    path = out / "trials.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.is_file() else []


def run(args):
    space = SPACE
    if args.knob:
        space = {name: values.split(",") for name, _, values in (knob.rpartition("=") for knob in args.knob)}
    if args.baseline:
        space = {}
    base = ["--profile", args.profile, "--server", str(args.server), "--port", str(args.port), "--tuned", str(args.tuned)]
    base += ["--packs", str(args.packs)] if args.packs else []
    base += args.option
    args.out.mkdir(parents=True, exist_ok=True)
    done = {t["key"]: t for t in trials_of(args.out)}
    registry = {p["name"]: p for p in tomllib.loads((serve.ROOT / "tunables.toml").read_text())["param"]}

    def launched_of(settings):
        """Every knob's effective value as this trial really launched it: `settings` holds only the knobs the walk
        moved, and a model fitted on those alone would place the baseline at the all-lowest corner and re-propose
        combinations the profile already ran. The launcher's OWN options are re-derived the way serve.py reads them
        (plan() renames them -- --draft-depth becomes --spec-draft-n-max, --expert-row-limit an env word -- so they
        cannot be looked up in plan()'s output); env knobs keep their names and are read from the plan's env."""
        _, env = plan(base, settings)
        launcher = serve.arguments(base + [word for name, value in settings.items()
                                           if name.startswith("--") for word in serve.option_words(name, value)])
        launched = dict(settings)
        for name in SPACE:
            if name in launched:
                continue
            if name.startswith("--"):
                value = getattr(launcher, name[2:].replace("-", "_"), None)
                if value is None:
                    continue  # the launcher has no such option or it is unset: nothing to record, not a lie
                launched[name] = "on" if value is True else "off" if value is False else str(value)
            elif name in env:
                launched[name] = env[name]
        return launched

    measurement = {"depths": getattr(args, "depths", ""), "context": getattr(args, "context", 0),
                   "tokens": getattr(args, "tokens", 0), "rounds": getattr(args, "rounds", 0)}

    def trial(settings):
        key = key_of(args.build, settings, measurement)
        if key not in done:
            print(f"trial {len(done)}: {settings or 'the profile as it launches'}", flush=True)
            record = {"key": key, "build": args.build, "profile": args.profile, "settings": settings,
                      "launched": launched_of(settings),
                      "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                      **run_trial(base, settings, args.out / f"{len(done):03d}-{key}", args)}
            with (args.out / "trials.jsonl").open("a") as file:
                file.write(json.dumps(record) + "\n")
            done[key] = record
            print(f"  {record['tokens_per_second'] and round(record['tokens_per_second'], 1)} tokens/s"
                  + (f", {record['error']}" if record["error"] else ""), flush=True)
            if os.environ.get("MLFLOW_TRACKING_URI"):
                send(args.out, args.experiment)
        return done[key]

    best, top = {}, trial({})
    if top["tokens_per_second"] is None:
        raise ValueError("the profile as it launches gave no rate: " + (top.get("error") or "no error recorded"))
    baseline = top["tokens_per_second"]

    def keep():
        # Written after every setting, so a search that is cut short still leaves what it found
        (args.out / "best.json").write_text(json.dumps(
            {"profile": args.profile, "build": args.build, "settings": best, "tokens_per_second": top["tokens_per_second"],
             "baseline_tokens_per_second": baseline}, indent=1) + "\n")

    keep()

    def persist_trial(key, merged):
        """Replace a trial's line in trials.jsonl with its confirmed record, so the repeat evidence outlives the run
        (log, propose and explain read the file, and a resumed search must see the rejection that lived in memory)."""
        path = args.out / "trials.jsonl"
        lines = [json.dumps(merged if json.loads(line).get("key") == key else json.loads(line))
                 for line in path.read_text().splitlines()]
        path.write_text("\n".join(lines) + "\n")

    def confirm_candidate(settings, found):
        """A candidate that could win is measured again until all its draws agree (search_noise.py is the rule; the
        first draw counts like every other, so --repeats=2 can dispute a first draw). The record keeps the minimum of
        the agreeing draws, an unstable candidate loses its rate, and a repeat that cannot run also loses it: no
        candidate wins the walk on the single draw that confirmation was called to check."""
        repeats = args.repeats
        max_repeats = args.max_repeats if args.max_repeats > 0 else repeats * 2
        if max_repeats < repeats:
            raise ValueError(f"--max-repeats {max_repeats} is under --repeats {repeats}")
        rates = [found["tokens_per_second"]]
        attempts, failed_repeat = 0, None
        while True:
            stable = not search_noise.unstable(rates)
            if len(rates) >= repeats and (stable or len(rates) >= max_repeats):
                break
            attempts += 1
            try:
                again = run_trial(base, settings, args.out / f"{found['key']}-repeat{attempts}", args)
            except (OSError, RuntimeError, TimeoutError, ValueError) as error:
                failed_repeat = str(error)
                break
            if search_workload.rate_of(again) is None:
                failed_repeat = again.get("error") or "a repeat gave no rate"
                break
            rates.append(again["tokens_per_second"])
        if failed_repeat is not None:
            merged = {**found, "tokens_per_second": None, "repeat_rates": rates, "stable": False,
                      "repeats": len(rates), "error": f"a repeat failed: {failed_repeat}"}
            print(f"  a repeat of {settings} failed: {failed_repeat}", flush=True)
        else:
            merged = {**found, "repeat_rates": rates, "stable": stable, "repeats": len(rates),
                      "tokens_per_second": search_noise.aggregate(rates) if stable else None}
            if not stable:
                merged["error"] = None
        persist_trial(found["key"], merged)
        done[found["key"]].update(merged)
        return merged

    if args.screen:
        baseline_plan = plan(base, {})

        def screened_rate(name, value):
            if plan(base, {name: value}) == baseline_plan:
                return top["tokens_per_second"]  # this level is the launching profile itself, already measured
            try:
                found = trial({name: value})
                return found["tokens_per_second"] or 0.0
            except ValueError:
                return 0.0
        verdict = search_screen.screen(space, screened_rate, top["tokens_per_second"],
                                       response=args.gain, budget=args.screen_budget or None)
        # keep the declared walk order: --draft-depth before --draft before --draft-layout is deliberate (drafting
        # moves a step's row count, so its depth is chosen with drafting on), and sorted survivors would undo it
        space = {name: values for name, values in space.items() if name in verdict["survivors"]}
        print(f"screen: {len(verdict['survivors'])} of {len(verdict['survivors']) + len(verdict['screened_out'])} knobs "
              f"can move the rate, {verdict['evaluations']} trials: {', '.join(verdict['survivors']) or 'none'}", flush=True)

    for _ in range(args.passes):
        for name, values in space.items():
            for value in values:
                settings = {**best, name: value}
                # The value the best already runs with is the best itself
                unset = name not in plan(base, best)[1] and not name.startswith("--")
                if plan(base, settings) == plan(base, best) or unset and str(registry[name].get("default")) == value:
                    continue
                try:
                    found = trial(settings)
                except ValueError as error:
                    print(f"  {name}={value} is no launch: {error}", flush=True)
                    continue
                if (args.repeats > 1 and found["tokens_per_second"]
                        and search_noise.needs_confirmation(found["tokens_per_second"], top["tokens_per_second"], args.gain)):
                    found = confirm_candidate(settings, found)
                    if not found.get("stable", True):
                        print(f"  {name}={value} did not settle over {found['repeats']} repeats: "
                              f"{[round(rate, 1) for rate in found['repeat_rates']]}", flush=True)
                if found["tokens_per_second"] and found["tokens_per_second"] > top["tokens_per_second"] * (1 + args.gain):
                    best, top = settings, found
            keep()
    print(f"best: {best or 'the profile as it launches'}, {top['tokens_per_second']:.1f} tokens/s "
          f"({baseline:.1f} as it launches)")
    return 0


def load(args):
    result = measure(args.url, args.sessions, args)
    if result.get("tokens_per_second") is None:
        raise ValueError(result.get("error") or "the load gave no rate")
    print(json.dumps(result))
    return 0


def send(out, experiment):
    """One MLflow run a trial, over the tracking server's REST API; mlflow.json keeps which trials it already holds."""
    site = os.environ["MLFLOW_TRACKING_URI"].rstrip("/") + "/api/2.0/mlflow/"
    # A client with a name: a tracking server behind a proxy can refuse urllib's own
    headers = {"Content-Type": "application/json", "User-Agent": "inkling2-search"}
    if os.environ.get("MLFLOW_TRACKING_USERNAME"):
        login = os.environ["MLFLOW_TRACKING_USERNAME"] + ":" + os.environ.get("MLFLOW_TRACKING_PASSWORD", "")
        headers["Authorization"] = "Basic " + base64.b64encode(login.encode()).decode()

    def call(route, body=None, query=""):
        request = urllib.request.Request(site + route + query, data=body and json.dumps(body).encode(), headers=headers)
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)

    try:
        found = call("experiments/get-by-name", query="?" + urllib.parse.urlencode({"experiment_name": experiment}))
        experiment_id = found["experiment"]["experiment_id"]
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
        experiment_id = call("experiments/create", {"name": experiment})["experiment_id"]
    sent_file = out / "mlflow.json"
    sent = json.loads(sent_file.read_text()) if sent_file.is_file() else {}
    for trial in trials_of(out):
        if trial["key"] in sent:
            continue
        now = int(time.time() * 1000)
        name = " ".join(f"{k.lstrip('-')}={v}" for k, v in trial["settings"].items()) or "as it launches"
        run_id = call("runs/create", {"experiment_id": experiment_id, "start_time": now, "run_name": name})["run"]["info"]["run_id"]
        metrics = [{"key": key, "value": trial[key], "timestamp": now, "step": 0}
                   for key in ("tokens_per_second", "wall_tokens_per_second", "draft_accepted", "startup_seconds", "prompt_tokens")
                   if trial.get(key) is not None]
        metrics += [{"key": "round_tokens_per_second", "value": value, "timestamp": now, "step": step}
                    for step, value in enumerate(trial.get("rounds", []), 1)]
        params = [{"key": key, "value": str(value)} for key, value in
                  {**trial["settings"], "profile": trial["profile"], "build": trial["build"], "trial": trial["key"],
                   "error": trial["error"] or ""}.items() if value != ""]
        call("runs/log-batch", {"run_id": run_id, "metrics": metrics, "params": params})
        call("runs/update", {"run_id": run_id, "status": "FAILED" if trial["error"] else "FINISHED", "end_time": now})
        sent[trial["key"]] = run_id
        sent_file.write_text(json.dumps(sent, indent=1) + "\n")
    return len(sent)


def log(args):
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        raise ValueError("MLFLOW_TRACKING_URI names no tracking server")
    print(f"{send(args.out, args.experiment)} trials in MLflow experiment {args.experiment}")
    return 0


def propose(args):
    """The settings the model (search_model.py) would try next, given the trials so far: the GP over every knob at
    once, so a combination two knobs only pay off in together can be found without walking the space to it."""
    space = SPACE
    if args.knob:
        space = {name: values.split(",") for name, _, values in (knob.rpartition("=") for knob in args.knob)}
    model = search_model.Model()
    # a trial records only the knobs the walk moved; "launched" holds every knob as it really ran, so the model
    # places the baseline where it launched, not at the all-lowest corner (records from before "launched" fall back)
    model.fit([{**trial, "settings": trial.get("launched") or trial.get("settings") or {}}
               for trial in trials_of(args.out)])
    print(json.dumps(model.propose(space, n=args.count, seed=args.seed), indent=1))
    return 0


def explain(args):
    """Why the fastest trials won, from the engine's ledger each trial recorded (search_evidence.py): the per-feature
    evidence of the top trials against the rest, and the trials whose window was not representative."""
    trials = trials_of(args.out)
    result = search_evidence.explain(trials, top=max(1, args.top))
    result["unstable"] = [{"key": trial.get("key"), "flags": flags}
                          for trial in trials if (flags := search_evidence.unstable_flags(trial))]
    print(json.dumps(result, indent=1))
    return 0


def apply(args):
    best = json.loads((args.out / "best.json").read_text())
    tuned = json.loads(args.tuned.read_text()) if args.tuned.is_file() else {}
    tuned[args.profile] = {**tuned.get(args.profile, {}), **best["settings"]}
    args.tuned.write_text(json.dumps(tuned, indent=1, sort_keys=True) + "\n")
    print(f"{args.profile} launches with {tuned[args.profile]}: {best['tokens_per_second']:.1f} tokens/s, "
          f"{best['baseline_tokens_per_second']:.1f} before")
    return 0


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    verbs = parser.add_subparsers(dest="verb", required=True)

    def verb(name, *, rounds=False):
        sub = verbs.add_parser(name)
        sub.set_defaults(act={"run": run, "load": load, "log": log, "apply": apply,
                              "propose": propose, "explain": explain}[name])
        if name != "load":
            sub.add_argument("--out", type=Path, required=True, help="The search's folder: trials.jsonl, best.json, a folder a trial")
        if name in ("run", "apply"):
            sub.add_argument("--profile", choices=sorted(serve.PROFILES), default="tensor-4")
            sub.add_argument("--tuned", type=Path, default=serve.TUNED, help="The settings file the launcher reads (default: its own)")
        if name in ("run", "log"):
            sub.add_argument("--experiment", default="llama.cpp/inkling2-decode", help="MLflow experiment the trials go to")
        if rounds:
            sub.add_argument("--tokens", type=int, default=192, help="Tokens each session generates in a round (default: 192)")
            sub.add_argument("--rounds", type=int, default=3, help="Measured rounds a trial; its rate is their median (default: 3)")
            sub.add_argument("--warm-rounds", type=int, default=1, help="Rounds before them that the rate does not count (default: 1)")
            sub.add_argument("--temperature", type=float, help="Sampling temperature (default: the server's, the model card's 1.0)")
            sub.add_argument("--timeout", type=float, default=900, help="Seconds a session's stream may take (default: 900)")
            sub.add_argument("--context", type=int, default=0,
                             help="Tokens of source text each session's request follows, about (default: the request alone)")
            sub.add_argument("--depths", default="",
                             help="Context depths a trial is measured at, comma separated; the trial keeps its worst "
                                  "(default: empty, --context alone)")
        return sub

    sub = verb("run", rounds=True)
    sub.add_argument("--server", type=Path, default=serve.ROOT / "build/bin/llama-server")
    sub.add_argument("--packs", type=Path, help="Folder of the profile's packs (default: the launcher's)")
    sub.add_argument("--port", type=int, default=8103)
    sub.add_argument("--build", default="", help="A name for the server build, kept with each trial: a new build is measured again")
    sub.add_argument("--knob", action="append", metavar="NAME=V1,V2", help="Search these values of a setting in place of the "
                     "default space; repeat for more settings (write an option as --knob=--draft-depth=2,4)")
    sub.add_argument("--baseline", action="store_true", help="Measure the profile as it launches and search nothing")
    sub.add_argument("--option", action="append", default=[], metavar="WORD", help="A launcher word every trial starts with; repeat")
    sub.add_argument("--wrap", action="append", default=[], metavar="WORD", help="A word in front of the server command; repeat")
    sub.add_argument("--gain", type=float, default=0.02, help="How far a value must beat the best to replace it (default: 0.02)")
    sub.add_argument("--passes", type=int, default=1, help="Walks over the space (default: 1)")
    sub.add_argument("--screen", action="store_true",
                     help="Screen the knobs first (search_screen.py): two trials a knob, and only the knobs whose swing "
                          "moves the rate by more than --gain enter the walk")
    sub.add_argument("--screen-budget", type=int, default=0, help="Most trials the screen may spend (default: 0, no cap)")
    sub.add_argument("--repeats", type=int, default=1,
                     help="Measurements of a candidate that could beat the best, drawn until they agree; their minimum "
                          "keeps the value (search_noise.py; default: 1, a single measurement)")
    sub.add_argument("--max-repeats", type=int, default=0,
                     help="Most draws a confirmation may take (default: 0, twice --repeats)")
    sub.add_argument("--startup-timeout", type=float, default=3600, help="Seconds a server may take to load (default: 3600)")
    sub = verb("load", rounds=True)
    sub.add_argument("--url", default="http://127.0.0.1:8103")
    sub.add_argument("--sessions", type=int, default=serve.SLOTS)
    sub = verb("propose")
    sub.add_argument("--knob", action="append", metavar="NAME=V1,V2",
                     help="Propose over these settings in place of the default space; repeat for more")
    sub.add_argument("--count", type=int, default=5, help="How many settings to propose (default: 5)")
    sub.add_argument("--seed", type=int, default=0, help="The proposal's seed (default: 0; a fixed seed is deterministic)")
    sub = verb("explain")
    sub.add_argument("--top", type=int, default=5, help="Trials treated as the fast ones (default: 5)")
    verb("log")
    verb("apply")
    return parser.parse_args(argv)


if __name__ == "__main__":
    try:
        options = arguments()
        raise SystemExit(options.act(options))
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2)
