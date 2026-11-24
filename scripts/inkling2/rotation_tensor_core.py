#!/usr/bin/env python3
"""Hadamard rotation in the tensor cores: microbenchmark, evidence and its check.

Row inkling-rotation-in-tensor-cores. The question: a Hadamard rotation as a bf16 tensor-core matmul with
exact +-1 entries and fp32 accumulate is the FWHT butterfly's signed-sum arithmetic in one launch. At the MoE
q8_1 seam it can only lose in two ways:

  1. the bf16 input rounding changes the quantized result enough to matter,
  2. reading the sign basis at 23 rows costs more than the butterfly it replaces.

rotation_tensor_core.cu is the standalone benchmark (it copies fwht.cu's fwht_block_transform<512, 256> and
quantize.cu's q8_1 arithmetic). This script runs it on the RTX 5090 build box through on-box, writes the
evidence, and checks it:

  --run --evidence PATH      measure on the box, write the evidence file
  --measure                  on the box: build the .cu, measure, print evidence JSON
  --check --evidence PATH    verify the evidence and judge the row

The check passes only when the evidence is real and neither falsifier fired. It re-derives every timing
statistic from the recorded samples, replays the deterministic error figures from the recorded seed with a
CPU reference, executes the freshly built kernel on the GPU against the recorded checksums, and recomputes
the verdict from the numbers with bars fixed here.

Bars: the bf16 input may raise the q8_1 result's error against the f32 rotation by at most 11.8 %, the
growth an independent input error of half the q8_1 error gives (sqrt(1 + 0.5^2) - 1). The prototype first
judged the difference between the two quantized results against the q8_1 error (bar 0.5); that difference
counts every element whose q8_1 bin flips under a tiny input change as a full step, even where both bins are
equally close to the true value, so it is kept in the evidence as a figure and does not decide. The
tensor-core rotation must also be faster than the butterfly at 23 rows, in both the rotation-only and the
rotation-plus-q8 chains.
"""

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CU_REL = "scripts/inkling2/rotation_tensor_core.cu"
PY_REL = "scripts/inkling2/rotation_tensor_core.py"
EVIDENCE_DEFAULT = "reports/inkling2-rotation-tensor-core.json"

ROWS = [23, 128, 512, 2048]
SEED = 0x5EED1234
WIDTH = 4608
BLOCK = 512
NBLK = WIDTH // BLOCK
ROUNDS = 40

# fixed at the run's design; the check compiles with exactly these
NVCC_FLAGS = ["-O3", "-std=c++17", "-arch=sm_120a"]

# bars fixed before the measurement
BAR_Q8_EXTRA_RATIO = 0.5   # bf16 q8_1 result vs f32 q8_1 result / q8_1 quantization error (a figure only)
BAR_Q8_TOTAL_GROWTH = math.sqrt(1.0 + BAR_Q8_EXTRA_RATIO ** 2) - 1.0  # q8_1 error vs the f32 rotation, growth
BAR_ROTATION_REL = 1e-2    # tensor-core rotation against the f32 rotation, relative L2
BAR_TC_VS_SIM = 1e-4       # tensor-core against the bf16-input butterfly, relative L2

CONFIGS = ["bm16_bn64_w4", "bm16_bn128_w8", "bm32_bn128_w8"]


# --------------------------------------------------------------------------------------------------------------
# small helpers

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd, cwd=None, env=None, capture=True):
    return subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=capture)


def require(cond, msg):
    if not cond:
        raise SystemExit("FAIL: " + msg)


def close(a, b, rel, abs_=0.0):
    if a is None or b is None:
        return False
    return abs(a - b) <= max(abs_, rel * max(abs(a), abs(b)))


def find_nvcc():
    cand = [os.environ.get("NVCC"), shutil.which("nvcc"), "/usr/local/cuda/bin/nvcc"]
    cand += sorted(str(p) for p in Path("/usr/local").glob("cuda*/bin/nvcc"))
    for c in cand:
        if c and Path(c).is_file() and os.access(c, os.X_OK):
            return str(c)
    raise SystemExit("FAIL: nvcc not found; this check builds and runs the kernel on the RTX 5090 build box")


def nvcc_version(nvcc):
    r = run([nvcc, "--version"])
    if r.returncode != 0:
        raise SystemExit("FAIL: nvcc --version failed:\n" + r.stderr)
    m = re.search(r"release (\S+), V(\S+)", r.stdout)
    return (m.group(1) + " V" + m.group(2)) if m else r.stdout.strip().splitlines()[-1]


def build_binary(tmpdir):
    nvcc = find_nvcc()
    out = str(Path(tmpdir) / "rotation_tensor_core")
    cmd = [nvcc, *NVCC_FLAGS, str(ROOT / CU_REL), "-o", out]
    r = run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        raise SystemExit("build failed:\n" + r.stdout + r.stderr)
    return out, nvcc


def run_json(binary, args):
    r = run([binary, *args])
    if r.returncode != 0:
        raise SystemExit("benchmark failed (%d):\n%s\n%s" % (r.returncode, r.stdout, r.stderr))
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError as e:
        raise SystemExit("benchmark printed no JSON: %s\n%s" % (e, r.stdout[:2000]))


def gpu_state():
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    r = run([smi, "--query-gpu=name,utilization.gpu,memory.used,clocks.sm",
             "--format=csv,noheader,nounits"])
    if r.returncode != 0:
        return None
    out = []
    for line in r.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            out.append({"name": parts[0], "util_pct": int(parts[1]),
                        "mem_used_mib": int(parts[2]), "clock_sm_mhz": int(parts[3])})
    return out


class GpuWatch:
    """Samples the GPU while a measurement runs, so the evidence says whether the card was quiet."""

    def __init__(self):
        self.samples = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.wait(2.0):
            s = gpu_state()
            if s:
                self.samples.append({"t": time.time(), "gpus": s})

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=10)

    def summary(self):
        utils = [g["util_pct"] for s in self.samples for g in s["gpus"]]
        mems = [g["mem_used_mib"] for s in self.samples for g in s["gpus"]]
        if not utils:
            return None
        return {"util_max_pct": max(utils), "util_median_pct": sorted(utils)[len(utils) // 2],
                "mem_max_mib": max(mems), "samples": len(utils)}


GPU_LOCK = "/run/user/%d/gpu0.lock" % os.getuid()


def try_gpu_lock():
    """The box's seats serialize their GPU tests with flock on /run/user/<uid>/gpu0.lock; a timing run takes it too.
    The acquisition blocks, so this run queues with theirs instead of racing them for the card."""
    try:
        import fcntl
        fd = os.open(GPU_LOCK, os.O_CREAT | os.O_RDWR, 0o664)
        fcntl.flock(fd, fcntl.LOCK_EX)
        return fd
    except (OSError, ImportError):
        return None


def drop_gpu_lock(fd):
    if fd is None:
        return
    try:
        import fcntl
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def wait_for_idle(poll_s=20.0):
    """A timing run waits for the shared 5090 to be quiet, holding the box's card lock once it is free.
    No timeout: another seat's test may hold the card."""
    first = True
    while True:
        s = gpu_state()
        if s is None:
            return None, None  # no nvidia-smi: nothing to wait on
        busy = any(g["util_pct"] > 5 or g["mem_used_mib"] > 1024 for g in s)
        if not busy:
            fd = try_gpu_lock()
            if fd is not None:
                s2 = gpu_state()
                if not (s2 and any(g["util_pct"] > 5 or g["mem_used_mib"] > 1024 for g in s2)):
                    return s2, fd
                drop_gpu_lock(fd)
            elif s:
                return s, None
        if first:
            if s:
                print("waiting for the 5090: " + ", ".join(
                    "%s util %d%% mem %d MiB" % (g["name"], g["util_pct"], g["mem_used_mib"]) for g in s),
                    file=sys.stderr)
            else:
                print("waiting for the 5090 lock", file=sys.stderr)
            first = False
        time.sleep(poll_s)


# --------------------------------------------------------------------------------------------------------------
# statistics, recomputed identically by --measure and --check

def stats_of(samples):
    v = sorted(samples)
    n = len(v)
    return {
        "median_us": v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2]),
        "min_us": v[0],
        "p10_us": v[int(math.floor(0.10 * (n - 1)))],
        "p90_us": v[int(math.floor(0.90 * (n - 1)))],
    }


def arm_map(shape):
    return shape["arms"]


def median_of(shape, arm):
    st = arm_map(shape).get(arm)
    require(st is not None, "shape %d has no arm %s" % (shape["rows"], arm))
    return stats_of(st["samples_us"])["median_us"]


def primary_config(shape):
    """The quickest stored-basis tensor-core configuration at this shape. Chosen from the measurement."""
    best, best_us = None, None
    for cfg in CONFIGS:
        arm = "tc_basis_f32/" + cfg
        if arm in arm_map(shape):
            us = median_of(shape, arm)
            if best_us is None or us < best_us:
                best, best_us = cfg, us
    require(best is not None, "no tensor-core arm at rows %d" % shape["rows"])
    return best


def derived_for(shape):
    cfg = primary_config(shape)
    cfg2 = "bm16_bn128_w8"  # the configuration the error figures are computed with
    return {
        "primary_config": cfg,
        "tc_basis_f32_us": median_of(shape, "tc_basis_f32/" + cfg),
        "tc_gen_f32_us": median_of(shape, "tc_gen_f32/" + cfg),
        "tc_basis_q8_us": median_of(shape, "tc_basis_q8/" + cfg),
        "tc_gen_q8_us": median_of(shape, "tc_gen_q8/" + cfg),
        "fwht_f32_us": median_of(shape, "fwht_f32"),
        "fwht_q8_us": median_of(shape, "fwht_q8"),
        "basis_read_us": median_of(shape, "tc_basis_f32/" + cfg) - median_of(shape, "tc_gen_f32/" + cfg),
        "copy_floor_us": median_of(shape, "copy_f32"),
        "basis_stream_floor_us": median_of(shape, "basis_stream"),
        "error_config": cfg2,
    }


def basis_read_bytes(rows, cfg):
    bm = {"bm16_bn64_w4": 16, "bm16_bn128_w8": 16, "bm32_bn128_w8": 32}[cfg]
    return ((rows + bm - 1) // bm) * NBLK * BLOCK * BLOCK


# --------------------------------------------------------------------------------------------------------------
# verdict: one definition, used by --measure and --check

def judge(measure):
    by_rows = {s["rows"]: s for s in measure["shapes"]}
    require(23 in by_rows, "the decode shape (23 rows) was not measured")

    out = {"shapes": {}}
    for rows, shape in by_rows.items():
        out["shapes"][rows] = derived_for(shape)

    d = by_rows[23]
    der = out["shapes"][23]
    q8 = d["errors"]["q8"]
    q8h = d["errors"].get("q8_fp16", {})
    tc_rel = d["errors"]["tc_vs_f32"]["rel_l2"]
    growth = q8["tc_vs_f32"] / q8["ref_rel"] - 1.0
    growth16 = (q8h["tc_vs_f32"] / q8["ref_rel"] - 1.0) if q8h.get("tc_vs_f32") is not None else None

    error_pass = (growth <= BAR_Q8_TOTAL_GROWTH) and (tc_rel <= BAR_ROTATION_REL)
    fp16_pass = (growth16 is not None) and (growth16 <= BAR_Q8_TOTAL_GROWTH)

    cost_pass_rotation = der["tc_basis_f32_us"] < der["fwht_f32_us"]
    cost_pass_q8 = der["tc_basis_q8_us"] < der["fwht_q8_us"]
    cost_pass = cost_pass_rotation and cost_pass_q8
    generated_keeps_win = der["tc_gen_f32_us"] < der["fwht_f32_us"]
    # no stored-basis kernel can read the basis faster than one coalesced pass over it
    read_floor_loses = der["basis_stream_floor_us"] >= der["fwht_f32_us"]

    if not error_pass:
        decided_by = "bf16_input_rounding"
        reason = ("bf16 input rounding raises the q8_1 error against the f32 rotation by %.2f %% "
                  "(bar %.2f %%); the tensor-core rotation itself is %.4f relative against the f32 rotation"
                  % (100.0 * growth, 100.0 * BAR_Q8_TOTAL_GROWTH, tc_rel))
        if fp16_pass:
            reason += ("; the fp16 input path passes at %.2f %%, so the arithmetic is sound only with an "
                       "11-mantissa-bit input" % (100.0 * growth16))
    elif not cost_pass:
        decided_by = "basis_read_cost" if (generated_keeps_win or read_floor_loses) else "prototype_kernel_cost"
        reason = ("the tensor-core rotation with the basis read costs %.3f us against the butterfly's %.3f us "
                  "at 23 rows (q8 chain %.3f against %.3f)"
                  % (der["tc_basis_f32_us"], der["fwht_f32_us"], der["tc_basis_q8_us"], der["fwht_q8_us"]))
        if read_floor_loses:
            reason += ("; one coalesced pass over the basis alone costs %.3f us, so no stored-basis kernel can "
                       "beat the butterfly at this shape" % der["basis_stream_floor_us"])
        elif generated_keeps_win:
            reason += "; without the basis read it costs %.3f us, so the read is what loses" % der["tc_gen_f32_us"]
        else:
            reason += ("; one pass over the basis costs %.3f us, below the butterfly, so the loss is this "
                       "prototype kernel's (%.3f us without the read), not the read's" %
                       (der["basis_stream_floor_us"], der["tc_gen_f32_us"]))
    else:
        decided_by = "none"
        reason = ("the tensor-core rotation costs %.3f us against the butterfly's %.3f us at 23 rows and "
                  "raises the q8_1 error against the f32 rotation by %.2f %% (bar %.2f %%)"
                  % (der["tc_basis_f32_us"], der["fwht_f32_us"], 100.0 * growth, 100.0 * BAR_Q8_TOTAL_GROWTH))

    reason += ("; the butterfly sits %.3f us above a plain copy of the same bytes (%.3f us), the most any "
               "rotation kernel can save here" % (der["fwht_f32_us"] - der["copy_floor_us"], der["copy_floor_us"]))

    prompt = {}
    for rows in (128, 512, 2048):
        if rows in out["shapes"]:
            der_s = out["shapes"][rows]
            prompt[rows] = {
                "tc_over_fwht": der_s["tc_basis_f32_us"] / der_s["fwht_f32_us"],
                "tc_basis_f32_us": der_s["tc_basis_f32_us"],
                "fwht_f32_us": der_s["fwht_f32_us"],
            }

    return {
        "derived": out,
        "falsifiers": {
            "bf16_input_rounding": {
                "bar_q8_total_growth": BAR_Q8_TOTAL_GROWTH,
                "q8_total_growth": growth,
                "q8_ref_rel": q8["ref_rel"],
                "q8_tc_vs_f32_rel": q8["tc_vs_f32"],
                "fp16_q8_total_growth": growth16,
                "bar_ratio_not_deciding": BAR_Q8_EXTRA_RATIO,
                "q8_extra_ratio": q8["ratio"],
                "rotation_rel_l2": tc_rel,
                "bar_rotation_rel_l2": BAR_ROTATION_REL,
                "fired": not error_pass,
                "fp16_input_ratio": q8h.get("ratio"),
                "fp16_input_passes": fp16_pass,
            },
            "basis_read_cost": {
                "decode_rotation_us": der["tc_basis_f32_us"],
                "decode_fwht_us": der["fwht_f32_us"],
                "decode_generated_us": der["tc_gen_f32_us"],
                "decode_basis_read_us": der["basis_read_us"],
                "decode_basis_stream_floor_us": der["basis_stream_floor_us"],
                "decode_copy_floor_us": der["copy_floor_us"],
                "read_floor_alone_loses": bool(read_floor_loses),
                "fired": not cost_pass,
            },
        },
        "prompt": prompt,
        "verdict": {
            "pass": bool(error_pass and cost_pass),
            "decided_by": decided_by,
            "reason": reason,
            "error_pass": bool(error_pass),
            "cost_pass": bool(cost_pass),
        },
    }


# --------------------------------------------------------------------------------------------------------------
# measurement on the box

def cmd_measure(args):
    rows = args.rows_list()
    seed = args.seed
    tmpdir = tempfile.mkdtemp(prefix="rotation-tc-")
    try:
        binary, nvcc = build_binary(tmpdir)
        device = run_json(binary, ["--device"])
        gpu_before = gpu_state()
        gpu_ready, lock_fd = wait_for_idle()
        gpu_after = None
        try:
            with GpuWatch() as watch:
                measure = run_json(binary, ["--measure", "--rows", ",".join(str(r) for r in rows),
                                            "--seed", str(seed)])
            gpu_after = gpu_state()
        finally:
            drop_gpu_lock(lock_fd)
        if gpu_ready is None:
            gpu_ready = gpu_state()

        cu_sha = sha256_file(ROOT / CU_REL)
        py_sha = sha256_file(ROOT / PY_REL)
        build_hash = hashlib.sha256(
            "|".join([cu_sha, py_sha, ",".join(NVCC_FLAGS), device["name"], device["cc"]]).encode()
        ).hexdigest()

        replay = run_json(binary, ["--replay", "--rows", ",".join(str(r) for r in rows),
                                   "--seed", str(seed)])
        cross_check(measure, replay)

        judged = judge(measure)
        evidence = {
            "row": "inkling-rotation-in-tensor-cores",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "host": os.uname().nodename,
            "seam": {
                "name": "MoE activation rotation and q8_1 quantize at hidden width 4608",
                "baseline_kernel": ("fwht.cu fwht_cuda_block<512, 256, float, true>: the transform, scale and "
                                    "sign multiply copied verbatim, one block per (row, 512-block), one launch"),
                "candidate_kernel": ("bf16 wmma matmul against the signed block Hadamard, block 512, nine "
                                     "blocks, fp32 accumulate, one launch"),
                "moe_fuse_cu_present": (ROOT / "ggml/src/ggml-cuda/moe-fuse.cu").is_file(),
                "note": ("the butterfly body is the one fwht_siglu_q8_cuda runs (moe-fuse.cu "
                         "moe_fwht_block_transform is a copy of fwht.cu's block transform); the SiLU gate and the "
                         "q8_1 write are common to both arms, so only the q8_1 write is timed"),
            },
            "build": {
                "source_sha256": {CU_REL: cu_sha, PY_REL: py_sha},
                "build_hash": build_hash,
                "nvcc": nvcc_version(nvcc),
                "nvcc_flags": NVCC_FLAGS,
                "device": device,
                "git_head": run(["git", "-C", str(ROOT), "rev-parse", "HEAD"]).stdout.strip(),
            },
            "shape": {
                "width": WIDTH, "block": BLOCK, "blocks": NBLK, "rows": rows, "seed": seed,
                "rotation": "normalized block Hadamard, block 512, signed per input element",
                "q8_block": 32,
            },
            "basis": {
                "unique_bytes": NBLK * BLOCK * BLOCK,
                "unique_mib": NBLK * BLOCK * BLOCK / (1024 * 1024),
                "read_bytes_per_launch": {
                    str(r): {cfg: basis_read_bytes(r, cfg) for cfg in CONFIGS} for r in rows
                },
                "note": ("stored int8, nine 512x512 blocks = 2.36 MB, L2-resident; the generated variant "
                         "computes the signed Sylvester entry from the indices and reads no basis"),
            },
            "gpu_state": {"before": gpu_before, "ready": gpu_ready, "after": gpu_after,
                          "during_including_this_measurement": watch.summary()},
            "measure": measure,
            "derived": judged["derived"],
            "falsifiers": judged["falsifiers"],
            "prompt": judged["prompt"],
            "verdict": judged["verdict"],
        }
        return evidence
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# --------------------------------------------------------------------------------------------------------------
# CPU replay cross-check, shared by --measure and --check

def cross_check(measure, replay, tag="measure"):
    """The GPU's deterministic content must match an independent CPU replay of the same seed."""
    by_rows = {s["rows"]: s for s in measure["shapes"]}
    rep = {s["rows"]: s for s in replay["shapes"]}
    require(set(by_rows) == set(rep), "%s: replay shapes differ" % tag)

    for rows, m in by_rows.items():
        r = rep[rows]
        e = m["errors"]
        c = e["checksums"]

        def w(what, a, b, rel, abs_=0.0):
            require(close(a, b, rel, abs_),
                    "%s rows %d: %s is %.17g but the CPU replay says %.17g" % (tag, rows, what, a, b))

        # the f32 butterfly is bit-exact on the CPU (same operation tree)
        w("y_ref sum", c["y_ref"]["sum"], r["ref"]["sum"], 1e-9, 1e-9)
        w("y_ref sumsq", c["y_ref"]["sumsq"], r["ref"]["sumsq"], 1e-9)
        w("y_ref maxabs", c["y_ref"]["maxabs"], r["ref"]["maxabs"], 1e-9)
        for i, (a, b) in enumerate(zip(e["samples"]["y_ref"], r["ref"]["samples"])):
            w("y_ref sample %d" % i, a, b, 1e-6)
        # the bf16-input butterfly simulation against the tensor core (accumulation order only)
        w("y_tc sumsq", c["y_tc"]["sumsq"], r["sim"]["sumsq"], 2e-4)
        w("y_tc maxabs", c["y_tc"]["maxabs"], r["sim"]["maxabs"], 2e-4)
        w("tc_vs_f32 rel_l2", e["tc_vs_f32"]["rel_l2"], r["tc_vs_ref"]["rel_l2"], 0.02, 2e-5)
        w("tc_vs_f32 max_abs", e["tc_vs_f32"]["max_abs"], r["tc_vs_ref"]["max_abs"], 0.02, 2e-6)
        # the q8_1 of the bit-exact f32 reference is deterministic; the bf16 one follows its simulation
        w("q8 ref_rel", e["q8"]["ref_rel"], r["q8"]["ref_rel"], 1e-6)
        w("q8 tc_rel", e["q8"]["tc_rel"], r["q8"]["sim_rel"], 0.02)
        w("q8 tc_vs_ref", e["q8"]["tc_vs_ref"], r["q8"]["diff_rel"], 0.02)
        w("q8 ratio", e["q8"]["ratio"], r["q8"]["ratio"], 0.02)
        w("q8 tc_vs_f32", e["q8"]["tc_vs_f32"], r["q8"]["sim_vs_f32"], 0.02)
        if "q8_fp16" in e and "fp16" in r:
            w("fp16 rot rel_l2", e["tc16_vs_f32"]["rel_l2"], r["fp16"]["rel_l2"], 0.02)
            w("fp16 q8 tc_rel", e["q8_fp16"]["tc_rel"], r["fp16"]["q8_rel"], 0.02)
            w("fp16 q8 ratio", e["q8_fp16"]["ratio"], r["fp16"]["ratio"], 0.02)
            w("fp16 q8 tc_vs_f32", e["q8_fp16"]["tc_vs_f32"], r["fp16"]["vs_f32"], 0.02)
            w("y_h sumsq", c["y_h"]["sumsq"], r["sim16"]["sumsq"], 2e-4)
        require(e["gen_vs_tc_max_abs"] == 0.0,
                "%s rows %d: the generated and stored bases differ (max abs %.17g)" % (tag, rows, e["gen_vs_tc_max_abs"]))


# --------------------------------------------------------------------------------------------------------------
# check

def load_evidence(path):
    p = Path(path)
    require(p.is_file(), "evidence file %s does not exist" % p)
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError as e:
        raise SystemExit("FAIL: evidence is not JSON: %s" % e)


def check_structure(ev):
    for key in ("row", "seam", "build", "shape", "basis", "measure", "derived", "falsifiers", "verdict"):
        require(key in ev, "evidence lacks %s" % key)
    require(ev["row"] == "inkling-rotation-in-tensor-cores", "evidence is for another row")
    require(ev["shape"]["width"] == WIDTH and ev["shape"]["block"] == BLOCK, "unexpected rotation geometry")
    rows = [s["rows"] for s in ev["measure"]["shapes"]]
    require(23 in rows, "the decode shape (23 rows) is missing")
    require(512 in rows and 2048 in rows, "the prompt shapes are missing")
    for s in ev["measure"]["shapes"]:
        require(len(s["arms"]) >= 14, "rows %d has too few arms" % s["rows"])
        for name, st in s["arms"].items():
            require(len(st["samples_us"]) == ROUNDS,
                    "rows %d arm %s has %d samples, not %d" % (s["rows"], name, len(st["samples_us"]), ROUNDS))
            require(all(v > 0 for v in st["samples_us"]), "non-positive sample in %s" % name)


def check_build(ev):
    cu_sha = sha256_file(ROOT / CU_REL)
    py_sha = sha256_file(ROOT / PY_REL)
    require(ev["build"]["source_sha256"].get(CU_REL) == cu_sha,
            "the evidence was measured from a different %s" % CU_REL)
    require(ev["build"]["source_sha256"].get(PY_REL) == py_sha,
            "the evidence was measured from a different %s" % PY_REL)
    require(ev["build"]["nvcc_flags"] == NVCC_FLAGS, "the evidence used different compiler flags")


def check_quiet(ev):
    ready = ev["gpu_state"].get("ready")
    require(isinstance(ready, list) and ready, "the evidence does not record the card's state at measurement start")
    for g in ready:
        require(g["util_pct"] <= 5 and g["mem_used_mib"] <= 1024,
                "the timing run started while the card was busy (%s util %d%% mem %d MiB)"
                % (g["name"], g["util_pct"], g["mem_used_mib"]))


def check_timings(ev):
    for s in ev["measure"]["shapes"]:
        for name, st in s["arms"].items():
            rederived = stats_of(st["samples_us"])
            for k, v in rederived.items():
                require(close(st[k], v, 1e-6, 1e-9),
                        "rows %d arm %s: recorded %s %.9g, samples say %.9g" % (s["rows"], name, k, st[k], v))


def check_derived(ev):
    rec = ev["derived"]["shapes"]
    by_rows = {s["rows"]: s for s in ev["measure"]["shapes"]}
    for rows, shape in by_rows.items():
        key = str(rows)
        require(key in rec, "derived statistics for rows %d are missing" % rows)
        d = derived_for(shape)
        for k, v in d.items():
            if isinstance(v, str):
                require(rec[key][k] == v, "rows %d: derived %s differs" % (rows, k))
            else:
                require(close(rec[key][k], v, 1e-6, 1e-9),
                        "rows %d: derived %s recorded %.9g, samples say %.9g" % (rows, k, rec[key][k], v))


def cmd_check(args):
    ev = load_evidence(args.evidence)
    check_structure(ev)
    check_build(ev)
    check_quiet(ev)
    check_timings(ev)
    check_derived(ev)

    tmpdir = tempfile.mkdtemp(prefix="rotation-tc-check-")
    try:
        binary, _ = build_binary(tmpdir)
        device = run_json(binary, ["--device"])
        require(device["name"] == ev["build"]["device"]["name"] and device["cc"] == ev["build"]["device"]["cc"],
                "the evidence was measured on %s cc %s, this machine has %s cc %s"
                % (ev["build"]["device"]["name"], ev["build"]["device"]["cc"], device["name"], device["cc"]))
        want_hash = hashlib.sha256("|".join([
            ev["build"]["source_sha256"][CU_REL], ev["build"]["source_sha256"][PY_REL],
            ",".join(NVCC_FLAGS), device["name"], device["cc"]]).encode()).hexdigest()
        require(want_hash == ev["build"]["build_hash"],
                "build hash mismatch: evidence %s, this source and compiler %s" % (ev["build"]["build_hash"], want_hash))

        rows = [str(s["rows"]) for s in ev["measure"]["shapes"]]
        seed = str(ev["shape"]["seed"])
        replay = run_json(binary, ["--replay", "--rows", ",".join(rows), "--seed", seed])
        cross_check(ev["measure"], replay, tag="check")

        # a fresh execution of the built kernel against the evidence's recorded values
        fresh = run_json(binary, ["--selftest", "--rows", rows[0], "--seed", seed])
        fresh_shape = fresh["shapes"][0]
        ev_shape = [s for s in ev["measure"]["shapes"] if s["rows"] == fresh_shape["rows"]][0]
        for path in (("errors", "tc_vs_f32", "rel_l2"), ("errors", "q8", "ref_rel"),
                     ("errors", "q8", "tc_rel"), ("errors", "q8", "ratio"),
                     ("errors", "tc16_vs_f32", "rel_l2"), ("errors", "q8_fp16", "ratio"),
                     ("errors", "q8", "tc_vs_f32"), ("errors", "q8_fp16", "tc_vs_f32")):
            a, b = ev_shape, fresh_shape
            for p in path:
                a = a[p]
            b = fresh_shape
            for p in path:
                b = b[p]
            require(close(a, b, 2e-4, 1e-9),
                    "check: fresh execution gives %s %.17g, the evidence says %.17g" % ("/".join(path), b, a))
        for name, cs in fresh_shape["errors"]["checksums"].items():
            for k in ("sumsq", "maxabs"):
                require(close(ev_shape["errors"]["checksums"][name][k], cs[k], 1e-6, 1e-9),
                        "check: fresh %s %s differs" % (name, k))

        judged = judge(ev["measure"])
        require(judged["verdict"]["pass"] == ev["verdict"]["pass"], "the recorded verdict does not follow from the numbers")
        require(judged["verdict"]["decided_by"] == ev["verdict"]["decided_by"],
                "the recorded decided_by does not follow from the numbers")
        for k in ("q8_total_growth", "q8_extra_ratio", "rotation_rel_l2", "fired"):
            require(judged["falsifiers"]["bf16_input_rounding"][k] == ev["falsifiers"]["bf16_input_rounding"][k]
                    or close(judged["falsifiers"]["bf16_input_rounding"][k],
                             ev["falsifiers"]["bf16_input_rounding"][k], 1e-9),
                    "the recorded bf16 falsifier differs from the numbers")
        for k in ("fired",):
            require(judged["falsifiers"]["basis_read_cost"][k] == ev["falsifiers"]["basis_read_cost"][k],
                    "the recorded basis falsifier differs from the numbers")

        if not ev["verdict"]["pass"]:
            print("FAIL: the row is answered negatively by %s" % ev["verdict"]["decided_by"])
            print("      " + ev["verdict"]["reason"])
            print("      error_pass=%s cost_pass=%s" % (ev["verdict"]["error_pass"], ev["verdict"]["cost_pass"]))
            return 1

        print("PASS: evidence is real and neither falsifier fired")
        print("      seam: " + ev["seam"]["name"])
        print("      " + ev["verdict"]["reason"])
        for rows_s in ev["measure"]["shapes"]:
            d = ev["derived"]["shapes"][str(rows_s["rows"])]
            print("      rows %-4d  tc_basis %.3f us  tc_gen %.3f us  fwht %.3f us  q8 chain %.3f / %.3f us"
                  % (rows_s["rows"], d["tc_basis_f32_us"], d["tc_gen_f32_us"], d["fwht_f32_us"],
                     d["tc_basis_q8_us"], d["fwht_q8_us"]))
        f = ev["falsifiers"]["bf16_input_rounding"]
        print("      bf16: rotation error %.4f%%, q8 error growth vs the f32 rotation %.2f%% (bar %.2f%%); fp16 %.2f%%"
              % (100.0 * f["rotation_rel_l2"], 100.0 * f["q8_total_growth"], 100.0 * f["bar_q8_total_growth"],
                 100.0 * f["fp16_q8_total_growth"]))
        return 0
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# --------------------------------------------------------------------------------------------------------------
# --run: measure through on-box and write the evidence locally

def cmd_run(args):
    out_path = Path(args.evidence or EVIDENCE_DEFAULT)
    cmd = ["on-box", "python3", PY_REL, "--measure",
           "--rows", ",".join(str(r) for r in args.rows_list()), "--seed", str(args.seed)]
    print("running the measurement on the build box: " + " ".join(cmd), file=sys.stderr)
    r = run(cmd, cwd=str(ROOT))
    if r.returncode == 75:
        raise SystemExit("on-box could not reach the build box (exit 75); the measurement did not start")
    if r.returncode != 0:
        raise SystemExit("measurement failed (%d):\n%s\n%s" % (r.returncode, r.stdout, r.stderr))
    try:
        ev = json.loads(r.stdout)
    except json.JSONDecodeError as e:
        raise SystemExit("the box printed no evidence JSON: %s\n%s" % (e, r.stdout[:2000]))

    ev["run"] = {
        "invocation": "on-box",
        "local_git_head": run(["git", "-C", str(ROOT), "rev-parse", "HEAD"]).stdout.strip(),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(ev, indent=2) + "\n")

    v = ev["verdict"]
    print("evidence: %s" % out_path)
    print("verdict: %s (%s)" % ("PASS" if v["pass"] else "FAIL", v["decided_by"]))
    print(v["reason"])
    for s in ev["measure"]["shapes"]:
        d = ev["derived"]["shapes"][str(s["rows"])]
        print("rows %-4d tc_basis %.3f us  tc_gen %.3f us  fwht %.3f us  q8 chain %.3f / %.3f us"
              % (s["rows"], d["tc_basis_f32_us"], d["tc_gen_f32_us"], d["fwht_f32_us"],
                 d["tc_basis_q8_us"], d["fwht_q8_us"]))
    return 0


# --------------------------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true", help="measure on the build box and write the evidence")
    ap.add_argument("--measure", action="store_true", help="on the box: build, measure, print evidence JSON")
    ap.add_argument("--check", action="store_true", help="verify the evidence and judge the row")
    ap.add_argument("--evidence", default=None, help="evidence JSON path")
    ap.add_argument("--rows", default=",".join(str(r) for r in ROWS), help="row counts")
    ap.add_argument("--seed", type=lambda s: int(s, 0), default=SEED)
    args = ap.parse_args()

    def rows_list():
        return [int(x) for x in args.rows.split(",") if x]

    args.rows_list = rows_list

    if args.measure:
        ev = cmd_measure(args)
        print(json.dumps(ev))
        return 0
    if args.run:
        return cmd_run(args)
    if args.check:
        return cmd_check(args)
    ap.print_help()
    return 64


if __name__ == "__main__":
    sys.exit(main())
