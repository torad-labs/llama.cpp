"""Execute a frozen GLM bundle on an already approved, dedicated two-GPU rental.

The operator must arm the provider stop deadline before --execute. This script
never creates, starts, stops or destroys provider instances, or downloads weights.
All subprocesses have a deadline; only this script's children are terminated.
"""
import argparse
import concurrent.futures
import csv
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import subprocess
import sys
import threading
import time

from qualification import (BASELINE_ARCHIVE, BASELINE_COMMIT, check_quality, check_quality_context, check_run_identity, check_stop_at,
                           device_cases, verify_files, validate_arch_table, validate_device_csv)
from verify_results import evaluate, load_rows, validate_route_observations


def clean_env():
    return {k: v for k, v in os.environ.items() if not k.startswith(("GGML_", "LLAMA_", "TEST_"))}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


class SiblingStopped(RuntimeError):
    """A command stopped because another one failed; never the cause itself."""


class Window:
    def __init__(self, args):
        self.args = args
        self.root = args.root.resolve()
        self.out = self.root / "results"
        self.until = args.stop_at - 600  # 50 minutes mandatory, then collection/stop reserve.
        self.failed = threading.Event()
        self.stage = "preflight"
        self.verdicts = {}
        self.records = []
        self.lock = threading.Lock()
        self.env = clean_env() | {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "0,1",
                                  "GGML_CUDA_ALLREDUCE": "internal", "GGML_CUDA_GRAPH_MAX": "0"}

    def command(self, name, argv, env=None, limit=300, expected_failure=None, check=None):
        # check(stdout, stderr) runs before the receipt is written, so a failed
        # validation closes the window and is recorded against its own command.
        # On a failed exit it still runs, to name what failed.
        if self.failed.is_set():
            raise SiblingStopped("another command failed; not starting " + name)
        remaining = min(limit, self.until - time.time())
        if remaining <= 0:
            raise TimeoutError("mandatory window ended; collect and stop the retained rental")
        started = time.time()
        error = None
        with (self.out / (name + ".out")).open("w") as stdout, (self.out / (name + ".log")).open("w") as stderr:
            child = subprocess.Popen([str(x) for x in argv], env=env or self.env, cwd=self.root,
                                     stdout=stdout, stderr=stderr, start_new_session=True)
            try:
                end = time.monotonic() + remaining
                while child.poll() is None:
                    if self.failed.is_set():
                        raise SiblingStopped("another command failed; stopping the current check")
                    left = end - time.monotonic()
                    if left <= 0:
                        raise subprocess.TimeoutExpired(argv, remaining)
                    try:
                        child.wait(timeout=min(0.25, left))
                    except subprocess.TimeoutExpired:
                        pass
                rc = child.returncode
                if expected_failure is not None:
                    diagnostic = (self.out / (name + ".log")).read_text()
                    if not rc or rc != expected_failure["exit"] or expected_failure["message"] not in diagnostic:
                        raise RuntimeError(f"{name}: expected failure was not observed (exit {rc})")
                elif rc:
                    detail = ""
                    if check:
                        try:
                            check((self.out / (name + ".out")).read_text(), (self.out / (name + ".log")).read_text())
                        except Exception as exc:
                            detail = f"; {exc}"
                    raise RuntimeError(f"{name}: exit {rc}{detail}")
                if check:
                    check((self.out / (name + ".out")).read_text(), (self.out / (name + ".log")).read_text())
            except BaseException as exc:
                error = f"{type(exc).__name__}: {exc}"
                self.failed.set()
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                raise
            finally:
                with self.lock:
                    self.records.append(dict(name=name, argv=[str(x) for x in argv],
                                             env={k: v for k, v in (env or self.env).items()
                                                  if k.startswith(("GGML_", "LLAMA_", "CUDA_", "TEST_"))},
                                             started_epoch=started, ended_epoch=time.time(), exit=child.returncode,
                                             expected_failure=expected_failure, error=error))
                    write_json(self.out / "commands.json", self.records)
        return (self.out / (name + ".out")).read_text()

    def device_case(self, case):
        env = self.env | self.profile_env | case["env"]
        env["LD_LIBRARY_PATH"] = str(self.root / "candidate-tests") + ":" + str(self.root / "cuda-runtime")
        binary, *argv = case["argv"]

        def check(output, log):
            try:
                # The required line is a pattern for one whole line of the program's own output.
                if case["require"] and not any(re.fullmatch(case["require"], line) for line in output.splitlines()):
                    raise ValueError(f"required route was not exercised: {case['require']}")
                if binary == "test-backend-ops" and case.get("expected_failure") is None:
                    validate_device_csv(output, case["cases"])
                if case.get("table"):
                    validate_arch_table(output, **case["table"])
            except ValueError as exc:
                raise ValueError(f"{case['name']}: {exc}") from None
        self.command(case["name"], [self.root / "candidate-tests" / binary, *argv], env, 420,
                     expected_failure=case.get("expected_failure"), check=check)

    def preflight(self):
        self.out.mkdir(parents=True, exist_ok=False)
        check_stop_at(self.args.stop_at, time.time(), self.args.max_hours)
        processes = self.command("initial-processes", ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"])
        if processes.strip():
            raise ValueError("the dedicated GPU allocation is occupied")
        hardware = self.command("hardware", ["nvidia-smi", "--query-gpu=name,power.limit,memory.total", "--format=csv,noheader,nounits"])
        cards = list(csv.reader(hardware.splitlines()))
        if len(cards) != 2 or any("RTX PRO 6000" not in n or "Server Edition" not in n
                                  or float(w) < 599 or float(m) < 97000 for n, w, m in cards):
            raise ValueError("the declared two 600 W SM120 Server Edition cards are required")
        self.command("hardware-detail", ["nvidia-smi", "-q"])
        self.command("topology", ["nvidia-smi", "topo", "-m"])
        inputs = json.loads(self.args.inputs.read_text())
        verify_files(self.args.fixtures, inputs["files"])
        verify_files(self.args.model_root, {f["file"]: {k: f[k] for k in ("bytes", "sha256")} for f in inputs["model"]["files"]})
        self.inputs = inputs
        self.model = self.args.model_root / inputs["model"]["files"][0]["file"]
        self.manifests = {}
        for arm in ("baseline", "candidate", "candidate-tests"):
            manifest = json.loads((self.root / arm / "manifest.json").read_text())
            verify_files(self.root / arm, manifest["files"])
            if manifest["build"]["kind"] != "portable" or manifest["build"]["arch"] != "120a":
                raise ValueError(f"{arm}: a portable SM120a artifact is required")
            runtime = self.root / "cuda-runtime"
            verify_files(runtime, {name: dict(bytes=(runtime / name).stat().st_size, sha256=sha)
                                   for name, sha in manifest["runtime"].items()})
            self.manifests[arm] = manifest
        if (self.manifests["baseline"]["commit"] != BASELINE_COMMIT
                or self.manifests["baseline"]["tarball_sha256"] != BASELINE_ARCHIVE):
            raise ValueError("the approximately 141 tok/s baseline is not installed")
        candidate, tests = self.manifests["candidate"], self.manifests["candidate-tests"]
        if candidate["commit"] != tests["commit"]:
            raise ValueError("candidate/test source differs")
        for name, identity in candidate["files"].items():
            if name.startswith(("libggml", "libllama")) and tests["files"].get(name) != identity:
                raise ValueError(f"candidate/test library differs: {name}")
        export_env = self.env | {"LD_LIBRARY_PATH": f"{self.root}/candidate-tests:{self.root}/cuda-runtime"}
        text = self.command("resolved-profile", [self.root / "candidate-tests/test-glm53-profile", self.args.profile], export_env)
        text += (self.out / "resolved-profile.log").read_text()
        self.profile_env = dict(re.findall(r"GLM profile " + re.escape(self.args.profile) + r": ((?:GGML_|LLAMA_)\w+)=(\d+)", text))
        if len(self.profile_env) != 13:
            raise ValueError("complete profile did not resolve from the candidate library")
        write_json(self.out / "identity.json", dict(inputs=inputs, artifacts=self.manifests, profile=self.args.profile,
                                                    resolved=self.profile_env, stop_at_epoch=self.args.stop_at))

    def serve(self, name, arm, context, ubatch, mode="timing"):
        argv = [sys.executable, Path(__file__).with_name("measure.py"), "--root", self.root,
                "--fixtures", self.args.fixtures, "--run", name, "--binary", self.root / arm,
                "--model", self.model, "--model-sha256", self.inputs["model"]["files"][0]["sha256"],
                "--context", str(context), "--ubatch", str(ubatch), "--deadline", str(self.until), "--mode", mode]
        if arm == "candidate":
            argv += ["--profile", self.args.profile]
        self.command(name, argv, limit=900)

    def compare(self, label, context, ubatch):
        # ABBA: each arm contains two legs and each leg has all 12 fixture requests.
        for arm, pair in (("baseline", 1), ("candidate", 1), ("candidate", 2), ("baseline", 2)):
            self.serve(f"{label}-{arm}-{pair}", arm, context, ubatch)
        directories = {arm: [self.out / f"{label}-{arm}-{p}" for p in (1, 2)] for arm in ("baseline", "candidate")}
        for arm, paths in directories.items():
            expected = self.manifests[arm] | {"model_sha256": self.inputs["model"]["files"][0]["sha256"]}
            expected["resolved_env"] = self.profile_env if arm == "candidate" else {}
            for path in paths:
                check_run_identity(json.loads((path / "run.json").read_text()), expected, context, ubatch,
                                   self.args.profile if arm == "candidate" else "")
        result = evaluate(load_rows(directories["baseline"]), load_rows(directories["candidate"]),
                          {f["category"]: f["prompt_sha256"] for f in self.inputs["fixtures"]},
                          200 if label == "primary" else 0)
        write_json(self.out / f"{label}-comparison.json", result)
        return result

    def quality(self):
        quality = {}
        for ctx in (2048, 4096, 16384):
            baseline_run = f"{self.args.run_id}-baseline-{ctx}"
            for arm, operation in (("baseline", "ident"), ("candidate", "kld")):
                run = baseline_run if arm == "baseline" else f"{self.args.run_id}-candidate-{ctx}"
                engine_env = {"GGML_CUDA_GRAPH_MAX": "0", "GGML_CUDA_ALLREDUCE": "internal"}
                if arm == "candidate":
                    engine_env |= self.profile_env
                if ctx == 4096:
                    engine_env["LLAMA_KLD_FIRST"] = "0"
                argv = [self.args.rig, "engine", operation, "--tree", self.root / arm, "--gpu", "0,1",
                        "--model", self.model, "--text", self.args.fixtures / "final-rental-bundle/kld-public.txt",
                        "--ctx", str(ctx), "--chunks", "1", "--tag", arm, "--run", run,
                        "--env", ",".join(f"{k}={v}" for k, v in engine_env.items()), "--json"]
                if arm == "candidate":
                    argv += ["--base", self.args.rig_root / "local/engine-lab" / baseline_run / "baseline.kld"]
                argv += ["--", "-ngl", "99", "-sm", "tensor", "-ts", "1,1", "-fa", "on", "-ctk", "q8_0",
                         "-ctv", "q8_0", "-ctki", "q8_0", "-b", "4096", "-ub", "1024"]
                # Rig's explicit root keeps its result paths independent of this script's cwd.
                output = self.command(f"kld-{arm}-{ctx}", argv, self.env | {
                    "RIG_ROOT": str(self.args.rig_root),
                    "LD_LIBRARY_PATH": f"{self.root}/{arm}:{self.root}/cuda-runtime"}, 600)
                if arm == "candidate":
                    # Keep invalid numeric literals as text so the failed result still has a valid JSON record.
                    report = json.loads(output, parse_constant=str,
                                        parse_float=lambda value: float(value) if math.isfinite(float(value)) else value)
                    quality[str(ctx)] = report.get("summary") if isinstance(report, dict) else None
                    write_json(self.out / "quality.json", quality)
                    check_quality_context(ctx, quality[str(ctx)])
        check_quality(quality)

    def card_matrix(self, cases):
        def on_card(card):
            try:
                for case in cases:
                    if case["env"].get("CUDA_VISIBLE_DEVICES") == str(card) and case["name"] != "capture-duplicate":
                        self.device_case(case)
            except BaseException:
                self.failed.set()
                raise
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(on_card, card) for card in (0, 1)]
        # Both cards have finished; the cause is the failure that was not a stop.
        errors = [future.exception() for future in futures if future.exception()]
        if errors:
            raise next((e for e in errors if not isinstance(e, SiblingStopped)), errors[0])

    def run(self):
        self.stage = "preflight"
        self.preflight()
        cases = device_cases()
        # Verify genuine peer support before doing the longer per-card matrix.
        self.stage = "capture-peer"
        self.device_case(next(c for c in cases if c["name"] == "capture-peer"))
        self.stage = "card-matrix"
        self.card_matrix(cases)
        for case in cases:
            if case["name"] == "capture-peer":
                continue
            if case["env"].get("CUDA_VISIBLE_DEVICES") not in ("0", "1") or case["name"] == "capture-duplicate":
                self.stage = case["name"]
                self.device_case(case)
        # The distribution check and both comparisons come first, so a window that
        # runs out of time has already answered whether the candidate is faster.
        # A failed comparison is the window's answer and ends it.
        self.stage = "quality"
        self.quality()
        comparisons = {}
        for label, context, ubatch in (("primary", 1048576, 512), ("companion", 524288, 1024)):
            self.stage = label
            result = comparisons[label] = self.compare(label, context, ubatch)
            self.verdicts[label] = dict(passed=result["passed"], failures=result["failures"],
                                        baseline_tok_s=result["baseline"]["tok_s"],
                                        candidate_tok_s=result["candidate"]["tok_s"])
            if not result["passed"]:
                raise ValueError(f"{label} comparison failed: {', '.join(result['failures'])}")
        for name, context, ubatch, mode in (("serving-semantics", 1048576, 512, "semantics"),
                                            ("capacity-32k", 524288, 1024, "concurrency"),
                                            ("capacity-128k", 1048576, 512, "capacity"),
                                            ("route-diagnostics", 1048576, 512, "diagnostic")):
            self.stage = name
            self.serve(name, "candidate", context, ubatch, mode)
        routes = [json.loads(line.split("GLM_ROUTE_STATS ", 1)[1])
                  for line in (self.out / "route-diagnostics/server.log").read_text().splitlines()
                  if "GLM_ROUTE_STATS " in line]
        if not validate_route_observations(routes):
            raise ValueError("missing or inconsistent per-device route observations")
        write_json(self.out / "qualification.json", dict(passed=True, **comparisons, routes=routes))
        publication = getattr(self.args, "publication_check", None)
        if publication is not None:
            self.stage = "driver-only"
            def check_publication(stdout, stderr):
                receipt = json.loads((self.out / "driver-only/driver-only-receipt.json").read_text())
                if receipt.get("passed") is not True:
                    raise ValueError("driver-only publication check has no passing receipt")
            self.command("publication", [sys.executable, publication, "--root", self.root,
                         "--deadline", str(self.until)], limit=3600, check=check_publication)
        self.stage = "done"
        return True


def execute(window):
    # results/outcome.json says how the controller ended, whatever ended it, with each comparison's verdict.
    outcome = dict(passed=False, stage=None, error=None, verdicts=None)
    try:
        outcome["passed"] = bool(window.run())
        return outcome["passed"]
    except BaseException as exc:
        outcome["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        outcome["stage"], outcome["verdicts"] = window.stage, window.verdicts
        if window.out.is_dir():
            write_json(window.out / "outcome.json", outcome)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--rig-root", type=Path, default=Path("/workspace/rig"))
    parser.add_argument("--rig", default="/workspace/rig/dist/rig")
    parser.add_argument("--run-id", required=True, help="unique Rig KLD output prefix")
    parser.add_argument("--profile", choices=("sm120-balanced", "sm120-wide", "sm120-direct"), default="sm120-balanced")
    parser.add_argument("--stop-at", type=float, required=True, help="already armed provider-stop epoch; includes setup")
    parser.add_argument("--max-hours", type=float, default=1.0, help="the hours approved for the window, setup included")
    parser.add_argument("--publication-check", type=Path,
                        help="reviewed driver-only companion; runs only after G passes and shares its work deadline")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(profile=args.profile, cases=device_cases(), primary=[1048576, 512],
                              companion=[524288, 1024], stop_at=args.stop_at), indent=2))
        return 0
    # Configuration probes deliberately abort. Disable their core files in this
    # process and its children before starting worker threads, without preexec_fn.
    resource.setrlimit(resource.RLIMIT_CORE, (0, resource.getrlimit(resource.RLIMIT_CORE)[1]))
    window = Window(args)
    def interrupted(signum, frame):
        window.failed.set()
        raise InterruptedError(f"qualification interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    return 0 if execute(window) else 1


if __name__ == "__main__":
    raise SystemExit(main())
