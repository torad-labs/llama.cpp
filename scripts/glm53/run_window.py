"""Execute a frozen GLM bundle on an already approved, dedicated two-GPU rental.

The operator must arm the provider stop deadline before --execute. This script
never creates, starts, stops or destroys provider instances, or downloads weights.
All subprocesses have a deadline; only this script's children are terminated.
"""
import argparse
import concurrent.futures
import csv
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time

from qualification import (BASELINE_ARCHIVE, BASELINE_COMMIT, check_quality,
                           check_run_identity, device_cases, verify_files, validate_device_csv)
from verify_results import evaluate, load_rows, validate_routes


def clean_env():
    return {k: v for k, v in os.environ.items() if not k.startswith(("GGML_", "LLAMA_", "TEST_"))}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


class Window:
    def __init__(self, args):
        self.args = args
        self.root = args.root.resolve()
        self.out = self.root / "results"
        self.until = args.stop_at - 600  # 50 minutes mandatory, then collection/stop reserve.
        self.failed = threading.Event()
        self.records = []
        self.lock = threading.Lock()
        self.env = clean_env() | {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "0,1",
                                  "GGML_CUDA_ALLREDUCE": "internal", "GGML_CUDA_GRAPH_MAX": "0"}

    def command(self, name, argv, env=None, limit=300):
        remaining = min(limit, self.until - time.time())
        if self.failed.is_set() or remaining <= 0:
            raise TimeoutError("mandatory window ended; collect and stop the retained rental")
        started = time.time()
        with (self.out / (name + ".out")).open("w") as stdout, (self.out / (name + ".log")).open("w") as stderr:
            child = subprocess.Popen([str(x) for x in argv], env=env or self.env, cwd=self.root,
                                     stdout=stdout, stderr=stderr, start_new_session=True)
            try:
                end = time.monotonic() + remaining
                while child.poll() is None:
                    if self.failed.is_set():
                        raise RuntimeError("another command failed; stopping the current check")
                    left = end - time.monotonic()
                    if left <= 0:
                        raise subprocess.TimeoutExpired(argv, remaining)
                    try:
                        child.wait(timeout=min(0.25, left))
                    except subprocess.TimeoutExpired:
                        pass
                rc = child.returncode
                if rc:
                    raise RuntimeError(f"{name}: exit {rc}")
            except BaseException:
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
                                             started_epoch=started, ended_epoch=time.time(), exit=child.returncode))
                    write_json(self.out / "commands.json", self.records)
        return (self.out / (name + ".out")).read_text()

    def device_case(self, case):
        env = self.env | self.profile_env | case["env"]
        env["LD_LIBRARY_PATH"] = str(self.root / "candidate-tests") + ":" + str(self.root / "cuda-runtime")
        binary, *argv = case["argv"]
        output = self.command(case["name"], [self.root / "candidate-tests" / binary, *argv], env, 420)
        combined = output + (self.out / (case["name"] + ".log")).read_text()
        if case["require"] and case["require"] not in combined:
            raise ValueError(f"{case['name']}: required route was not exercised")
        if binary == "test-backend-ops":
            validate_device_csv(output, case["cases"])

    def preflight(self):
        self.out.mkdir(parents=True, exist_ok=False)
        if not 0 < self.args.stop_at - time.time() <= 3600:
            raise ValueError("stop-at must be the armed deadline, at most one hour away")
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
                    quality[str(ctx)] = json.loads(output)["summary"]
        check_quality(quality)
        write_json(self.out / "quality.json", quality)

    def run(self):
        self.preflight()
        cases = device_cases()
        # Verify genuine peer support before doing the longer per-card matrix.
        self.device_case(next(c for c in cases if c["name"] == "capture-peer"))
        def on_card(card):
            try:
                for case in cases:
                    if case["env"].get("CUDA_VISIBLE_DEVICES") == str(card) and case["name"] != "capture-duplicate":
                        self.device_case(case)
            except BaseException:
                self.failed.set()
                raise
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(on_card, (0, 1)))
        for case in cases:
            if case["name"] == "capture-peer":
                continue
            if case["env"].get("CUDA_VISIBLE_DEVICES") not in ("0", "1") or case["name"] == "capture-duplicate":
                self.device_case(case)
        self.serve("serving-semantics", "candidate", 1048576, 512, "semantics")
        self.serve("capacity-32k", "candidate", 524288, 1024, "concurrency")
        self.serve("capacity-128k", "candidate", 1048576, 512, "capacity")
        self.quality()
        primary = self.compare("primary", 1048576, 512)
        companion = self.compare("companion", 524288, 1024)
        self.serve("route-diagnostics", "candidate", 1048576, 512, "diagnostic")
        routes = [json.loads(line.split("GLM_ROUTE_STATS ", 1)[1])
                  for line in (self.out / "route-diagnostics/server.log").read_text().splitlines()
                  if "GLM_ROUTE_STATS " in line]
        if not routes or not all(validate_routes(row) for row in routes) or {r["device"] for r in routes} != {0, 1}:
            raise ValueError("missing or inconsistent per-device route observations")
        write_json(self.out / "qualification.json", dict(passed=primary["passed"] and companion["passed"],
                                                         primary=primary, companion=companion, routes=routes))
        return primary["passed"] and companion["passed"]


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
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(profile=args.profile, cases=device_cases(), primary=[1048576, 512],
                              companion=[524288, 1024], stop_at=args.stop_at), indent=2))
        return 0
    window = Window(args)
    def interrupted(signum, frame):
        window.failed.set()
        raise InterruptedError(f"qualification interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    return 0 if window.run() else 1


if __name__ == "__main__":
    raise SystemExit(main())
