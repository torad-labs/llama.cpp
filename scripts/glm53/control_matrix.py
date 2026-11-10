#!/usr/bin/env python3
"""Prepare same-workload comparisons or run bounded distribution controls; never rent."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from qualification import check_quality_context

QUALITY_CONTRACT = {"meanKld_abs_max": 1e-6, "sameTopP_min": 99.99}
CORPUS_SHA256 = "fca56be516ba5a724c9812619a3c0a31fcdda209a74fcd1080868e051630700d"
MODEL = "GLM-5.3-Flash-Uncensored-IQ3_XXS-00001-of-00003.gguf"
FROZEN_BASELINE = {
    "engine": "B",
    "commit": "24c77a0b5c8dcb3352e8027cd7e34367bdaeec13",
    "patch_sha256": "098fcb005ad4b0575ece6786e1da219d49c5b337a6e72a06404c0635b0014510",
    "source_record": "engines/B/SOURCE",
}


def quality_jobs(candidate, baseline):
    jobs = []
    for context in (2048, 4096, 16384):
        configurations = [(baseline, "f32", 512), (baseline, "q8", 512),
                          (candidate, "f32", 512), (candidate, "f32", 128), (candidate, "q8", 512)]
        references = {}
        for engine, query, ubatch in configurations:
            identifier = f"{context}-{engine}-{query}-ub{ubatch}"
            references[engine, query, ubatch] = identifier
            jobs.append(dict(id=identifier, action="reference", context=context,
                             engine=engine, query=query, ubatch=ubatch))
        comparisons = [
            ("engine-f32", candidate, "f32", 512, baseline, "f32", 512),
            ("engine-q8", candidate, "q8", 512, baseline, "q8", 512),
            ("chunk-f32", candidate, "f32", 128, candidate, "f32", 512),
            ("chunk-q8", candidate, "q8", 128, candidate, "q8", 512),
            ("query-512", candidate, "q8", 512, candidate, "f32", 512),
            ("query-128", candidate, "q8", 128, candidate, "f32", 128),
        ]
        for name, engine, query, ubatch, ref_engine, ref_query, ref_ubatch in comparisons:
            jobs.append(dict(id=f"{context}-{name}", action="compare", comparison=name, context=context,
                             engine=engine, query=query, ubatch=ubatch,
                             reference_id=references[ref_engine, ref_query, ref_ubatch]))
    return jobs


def prepare(candidate_engine="C", baseline_engine="B") -> dict[str, Any]:
    if (candidate_engine == baseline_engine or
            any(not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in (candidate_engine, baseline_engine))):
        raise ValueError("baseline and candidate need distinct safe engine labels")
    entries, controls = [], []
    transport = ["--transport", "internal-nccl-gather", "--indexer-query", "q8"]
    for depth in ("off", "1", "2"):
        controls.append(dict(id=f"mtp{depth}", mtp_depth=depth, indexer_query="q8", ubatch=512,
                             slots=36, contexts=[512, 8192], repeats_per_engine=2))
        for repeat, engine in enumerate((baseline_engine, candidate_engine, candidate_engine, baseline_engine)):
            entries.append([engine, depth, [[36, 512], [36, 8192]], [],
                            dict(label=f"mtp{depth}-repeat{repeat}", serve=list(transport))])
    return dict(
        schema="glm53-control-plan-v2",
        source_identity=dict(baseline=FROZEN_BASELINE | {
            "engine": baseline_engine, "source_record": f"engines/{baseline_engine}/SOURCE"},
            candidate=dict(engine=candidate_engine, source_record=f"engines/{candidate_engine}/SOURCE",
                           requirement="record exact commit and patch SHA-256 before the run")),
        serving_baseline=dict(mtp_depth="2", depth_policy="adaptive", adaptive_threshold=0.18,
                              indexer_query="q8", ubatch=512, slots=36),
        quality_contract=QUALITY_CONTRACT,
        quality_scope="Single-sequence DP4 teacher-forced distributions at2K/4K/16K; not36-sequence quality certification.",
        quality_jobs=quality_jobs(candidate_engine, baseline_engine),
        serving_controls=controls,
        profile_capture=dict(active_sessions=36, context_tokens=512, mtp_depth="2", indexer_query="q8",
                             ubatch=512, profile_requested=True, timing="gpu-cupti-only", expected_devices=[0, 1, 2, 3],
                             required_evidence=["36 distinct sequence markers with no prompt rows in a target phase",
                                                "same-process, phase-correlated CUDA kernels on every expected device",
                                                "raw report, SQLite export, exporter version and launch SHA-256"],
                             exporter_policy=dict(portable_nsys="2026.6.1.218-266139266636v0",
                                                  host_nsys="2026.1.3.425-261338342291v0",
                                                  requirement="Use the bundled capture/exporter version; do not export with the older host profiler.")),
        server_matrix_plan=[[candidate_engine, "2", [[36, 512]], [],
                             dict(label="gpu-36-decode", profile=True, tokens=1024, serve=list(transport))], *entries],
        execution=dict(program="scripts/glm53/server_matrix.py",
                       unprofiled_controls="ABBA at each MTP setting; unchangedQ8/ub512/36-slot recipe. Numerical work runs separately."),
    )


def matrix_command(root, deadline, tag="glm53-controls", candidate_engine="C", baseline_engine="B"):
    plan = prepare(candidate_engine, baseline_engine)
    return ["python3", str(Path(root) / "scripts/glm53/server_matrix.py"), "--root", str(Path(root)),
            "--deadline", str(deadline), "--tag", tag,
            "--plan", json.dumps(plan["server_matrix_plan"], separators=(",", ":"))]


class QualityGuardError(RuntimeError):
    """Stop numerical work when the selected-card reserve cannot be maintained."""


def quality_memory(devices, deadline):
    try:
        sampled = subprocess.run(
            ["nvidia-smi", "-i", ",".join(devices), "--query-gpu=uuid,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=min(10, max(0.001, deadline-time.time())))
        rows = [line.split(",") for line in sampled.stdout.splitlines() if line.strip()]
        free = {uuid.strip(): int(value.strip()) for uuid, value in rows}
        if len(rows) != len(devices) or set(free) != set(devices) or any(value < 0 for value in free.values()):
            raise ValueError("selected GPU memory readings are incomplete")
        return free
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        raise QualityGuardError(f"cannot observe the selected-card reserve: {error}") from error


def quality_command(argv, env, stream, devices, deadline, output, record):
    record["memory_log"] = record["id"] + ".memory.jsonl"
    record["minimum_free_mib"] = {}
    child = None
    with (output / record["memory_log"]).open("x") as samples:
        def sample():
            free = quality_memory(devices, deadline)
            samples.write(json.dumps(dict(epoch=time.time(), free_mib=free)) + "\n")
            samples.flush()
            minima = record["minimum_free_mib"]
            for uuid, value in free.items():
                minima[uuid] = min(minima.get(uuid, value), value)
            if min(free.values()) < 4096:
                raise QualityGuardError("selected GPU fell below the 4096 MiB reserve")

        sample()
        try:
            child = subprocess.Popen(argv, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            while True:
                sample()
                left = deadline - time.time()
                if left <= 0:
                    raise TimeoutError("quality work deadline reached")
                try:
                    return child.wait(timeout=min(1, left))
                except subprocess.TimeoutExpired:
                    pass
        finally:
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=10)


def run_quality(root, output, deadline, candidate="C", baseline="B", corpus_sha256=CORPUS_SHA256):
    root, output = Path(root).resolve(), Path(output).resolve()
    plan = prepare(candidate, baseline)
    output.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = dict(plan=plan["quality_jobs"], runs=[], completed=False, passed=False,
                  scope=plan["quality_scope"], contract=QUALITY_CONTRACT, started_epoch=time.time())

    def save():
        (output / "quality.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    try:
        if not math.isfinite(deadline) or deadline <= time.time():
            raise ValueError("quality work requires a finite future deadline")
        sources = {engine: (root / "engines" / engine / "SOURCE").read_text().strip()
                   for engine in (baseline, candidate)}
        report["sources"] = sources
        expected_baseline = FROZEN_BASELINE["commit"] + "+dirty-sha256:" + FROZEN_BASELINE["patch_sha256"]
        if sources[baseline] != expected_baseline:
            raise ValueError("baseline source does not match the frozen November 8 engine")
        if not sources[candidate]:
            raise ValueError("candidate source identity is empty")
        corpus = root / "public-corpus.txt"
        actual = hashlib.sha256(corpus.read_bytes()).hexdigest()
        if actual != corpus_sha256:
            raise ValueError("public corpus checksum mismatch")
        model = root / "model" / MODEL
        if not model.is_file():
            raise ValueError("missing pinned local model")
        selected = json.loads((root / "results/selected-gpus.json").read_text())["selected"]
        devices = [card["uuid"] for card in selected]
        if len(devices) != 4 or len(set(devices)) != 4:
            raise ValueError("quality controls require four distinct selected GPUs")
        report.update(corpus_sha256=actual, model=str(model), devices=devices,
                      initial_free_mib=quality_memory(devices, deadline))
        if min(report["initial_free_mib"].values()) < 4096:
            raise QualityGuardError("selected GPU starts below the 4096 MiB reserve")
        for job in report["plan"]:
            record = job | dict(log=job["id"] + ".log", started_epoch=time.time())
            report["runs"].append(record)
            try:
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise TimeoutError("quality work deadline reached")
                binary = root / "engines" / job["engine"] / "bin"
                reference_id = job["id"] if job["action"] == "reference" else job["reference_id"]
                reference = output / (reference_id + ".kld")
                record["distribution"] = reference.name
                if job["action"] == "compare" and (not reference.is_file() or not reference.stat().st_size):
                    raise ValueError("distribution reference unavailable")
                argv = [str(binary / "llama-perplexity"), "-m", str(model), "-f", str(corpus),
                        "-c", str(job["context"]), "--chunks", "1", "-ngl", "99", "-sm", "dp", "-ts", "1,1,1,1",
                        "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "-ctki", "q8_0", "-cts", "q8_0",
                        "-b", "2048", "-ub", str(job["ubatch"]), "--kl-divergence-base", str(reference)]
                if job["action"] == "compare":
                    argv += ["--kl-divergence"]
                env = {"PATH": os.environ["PATH"], "HOME": os.environ.get("HOME", "/root"),
                       "LD_LIBRARY_PATH": str(binary), "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                       "CUDA_VISIBLE_DEVICES": ",".join(devices), "GGML_CUDA_GRAPH_MAX": "0",
                       "GGML_CUDA_ALLREDUCE": "internal", "LLAMA_TOK_EMBD_GPU": "1",
                       "LLAMA_INDEXER_Q8_QUERY": "1" if job["query"] == "q8" else "0"}
                if job["context"] == 4096:
                    env["LLAMA_KLD_FIRST"] = "0"
                record.update(argv=argv, env=env)
                save()
                log = output / record["log"]
                with log.open("x") as stream:
                    record["exit"] = quality_command(argv, env, stream, devices,
                                                     min(deadline, time.time() + 600), output, record)
                if record["exit"]:
                    raise RuntimeError(f"perplexity exited {record['exit']}")
                if job["action"] == "reference":
                    if not reference.is_file() or not reference.stat().st_size:
                        raise ValueError("missing distribution output")
                    record["passed"] = True
                else:
                    text = log.read_text()
                    mean = re.search(r"Mean\s+KLD:\s*([-+0-9.eE]+)", text)
                    top = re.search(r"Same top p:\s*([-+0-9.eE]+)", text)
                    quality = dict(meanKld=float(mean.group(1)) if mean else None,
                                   sameTopP=float(top.group(1)) if top else None)
                    try:
                        check_quality_context(job["context"], quality)
                        record["quality"] = quality | dict(passed=True)
                    except ValueError as error:
                        record["quality"] = quality | dict(passed=False, error=str(error))
                    record["passed"] = record["quality"]["passed"]
            except (Exception, KeyboardInterrupt) as error:
                record.update(passed=False, error=f"{type(error).__name__}: {error}")
                if isinstance(error, (KeyboardInterrupt, TimeoutError, subprocess.TimeoutExpired, QualityGuardError)):
                    report["error"] = record["error"]
                    break
            finally:
                record["finished_epoch"] = time.time()
                save()
        report["completed"] = (len(report["runs"]) == len(report["plan"])
                               and all(row.get("exit") == 0 for row in report["runs"]))
        report["passed"] = report["completed"] and all(row.get("passed") is True for row in report["runs"])
    except (Exception, KeyboardInterrupt) as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        report["finished_epoch"] = time.time()
        save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Plan file, or a new result directory with --execute-quality")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--deadline", type=float)
    parser.add_argument("--tag", default="glm53-controls")
    parser.add_argument("--candidate-engine", default="C")
    parser.add_argument("--baseline-engine", default="B")
    parser.add_argument("--execute-quality", action="store_true")
    parser.add_argument("--corpus-sha256", default=CORPUS_SHA256)
    args = parser.parse_args()
    if args.execute_quality or args.root is not None or args.deadline is not None:
        if args.root is None or args.deadline is None:
            parser.error("--root and --deadline are required together")
    if args.execute_quality:
        report = run_quality(args.root, args.output, args.deadline, args.candidate_engine,
                             args.baseline_engine, args.corpus_sha256)
        print(json.dumps(dict(output=str(args.output), completed=report["completed"], passed=report["passed"])))
        return 0 if report["passed"] else 1
    plan = prepare(args.candidate_engine, args.baseline_engine)
    if args.root is not None:
        plan["execution"]["command"] = matrix_command(args.root, args.deadline, args.tag,
                                                    args.candidate_engine, args.baseline_engine)
        plan["execution"]["quality_command"] = [
            "python3", "-I", str(args.root / "scripts/glm53/control_matrix.py"), "--execute-quality",
            "--root", str(args.root), "--deadline", str(args.deadline),
            "--baseline-engine", args.baseline_engine, "--candidate-engine", args.candidate_engine,
            "--output", str(args.root / "results" / (args.tag + "-quality"))]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(plan, output, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps(dict(output=str(args.output), controls=len(plan["serving_controls"]), checked=True)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
