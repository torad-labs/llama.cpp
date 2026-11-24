#!/usr/bin/env python3
"""FP8 pair accumulation on the relative decode: RTX 5090 evidence and its check.

Row inkling-fp8-accumulation. GGML_CUDA_FATTN_FP8_ACCUM=1 makes the relative decode's partial writers store e4m3 hi+lo
pair slots and the merge fold them on the sm_120a block-scaled MMA (ggml/src/ggml-cuda/fattn-rel-fp8-pair.cuh,
fattn-rel-decode.cuh). The falsifier is hi+lo quality against f16: the pair algebra's own error, the attention op
against the host on an f8_e4m3 cache, and the arch fixture's logits and greedy picks on an f8_e4m3 global cache with
the switch on (condition 3 of the row: switch ON and type_KV=f8_e4m3).

  --run --evidence PATH     measure on the build box through on-box, write the evidence file
  --measure --build-dir B   on the box: run the binaries of build B, print evidence JSON
  --check --evidence PATH   verify the evidence and judge the row (no GPU needed)

The evidence keeps the lines each binary printed that carry a figure; --check parses the figures again from those
lines, checks the kernel sources are the ones measured, and recomputes the verdict with the bars fixed here.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_DEFAULT = "reports/inkling2-fp8-accumulation.json"
ROW = "inkling-fp8-accumulation"

# the sources the evidence was measured from; a change to any of them makes the evidence stale
SOURCES = [
    "ggml/src/ggml-cuda/fattn-rel-fp8-pair.cuh",
    "ggml/src/ggml-cuda/fattn-rel-decode.cuh",
    "ggml/src/ggml-cuda/fattn-rel-mma.cuh",
    "tests/test-inkling-fp8-pair.cpp",
]

# the relative decode kernels whose partials the pair path carries (ggml_cuda_fattn's sub-kernel tags)
REL_DECODE_FAMILY = ("REL_RUNS", "REL_MMA", "REL_DECODE")

# bars fixed before the measurement
BAR_PAIR_GAIN = 4.0      # hi+lo error must be at least this many times below hi alone (the unit test's own bar)
BAR_MODEL_NMSE = 1e-4    # logits NMSE against the F16 cache, the fixture's tolerance for one-byte caches

ONE_BYTE_ENV = {
    "GGML_CUDA_FATTN_REL_DECODE_ONE_BYTE": "1",
    "LLAMA_INKLING2_MEAN_MASK_DECODE": "1",
    "LLAMA_INKLING2_MEAN_POOL_MIN": "8192",
}
MODEL_CASES = ["served", "restore"]

LINE_PATTERNS = re.compile(
    r"(encode:|fold32:|fold_slots|sum_terms|merge pattern|fold pattern|all checks passed|failure\(s\)|skipped|"
    r"tests passed|FAIL|NMSE|argmax agrees|ggml_cuda_fattn: kernel|OK inkling2|nodes ran|off the device)")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def require(cond, msg):
    if not cond:
        raise SystemExit("FAIL: " + msg)


def fnum(s):
    return float(s)


# --------------------------------------------------------------------------------------------------------------
# parsing, shared by --measure and --check

def parse_pair(lines):
    out = {"passed": any("all checks passed" in l for l in lines), "merge_worst_rel": [], "sum_terms": []}
    for l in lines:
        m = re.search(r"encode: (\d+) values, mean hi err (\S+), mean two-term err (\S+)", l)
        if m:
            out["encode_values"] = int(m.group(1))
            out["hi_err"] = fnum(m.group(2).rstrip(","))
            out["two_term_err"] = fnum(m.group(3).rstrip(","))
        m = re.search(r"fold32: (\d+)/32 single-lane exact, x32 worst rel (\S+), mixed worst rel (\S+)", l)
        if m:
            out["fold32_exact_lanes"] = int(m.group(1))
            out["fold32_worst_rel"] = fnum(m.group(2).rstrip(","))
            out["fold32_mixed_worst_rel"] = fnum(m.group(3).rstrip(","))
        m = re.search(r"fold_slots\((\d+)\): worst rel (\S+)", l)
        if m:
            out["fold_slots_parts"] = int(m.group(1))
            out["fold_slots_worst_rel"] = fnum(m.group(2))
        m = re.search(r"sum_terms (two-term|one-term) \((.*)\): worst rel (\S+)", l)
        if m:
            out["sum_terms"].append({"kind": m.group(1), "shape": m.group(2), "worst_rel": fnum(m.group(3))})
        m = re.search(r"merge pattern \(\s*(\d+) parts, mode (\d+)\): worst rel (\S+)", l)
        if m:
            out["merge_worst_rel"].append({"parts": int(m.group(1)), "mode": int(m.group(2)),
                                           "worst_rel": fnum(m.group(3))})
    if "hi_err" in out and out["two_term_err"] > 0:
        out["pair_gain"] = out["hi_err"] / out["two_term_err"]
    return out


def parse_ops(lines):
    out = {"passed": None, "total": None, "fail_lines": sum(1 for l in lines if "FAIL" in l)}
    for l in lines:
        m = re.search(r"(\d+)/(\d+) tests passed", l)
        if m:
            out["passed"], out["total"] = int(m.group(1)), int(m.group(2))
    return out


def parse_model(lines):
    out = {"ok": any(l.startswith("OK inkling2") for l in lines), "kernels": []}
    for l in lines:
        m = re.search(r"NMSE (\S+) \(at most (\S+)\)", l)
        if m:
            out["nmse"], out["nmse_bar"] = fnum(m.group(1)), fnum(m.group(2))
        m = re.search(r"argmax agrees on (\d+) of (\d+) rows", l)
        if m:
            out["argmax_agree"], out["rows"] = int(m.group(1)), int(m.group(2))
        m = re.search(r"ggml_cuda_fattn: kernel (\S+) on K (\S+)", l)
        if m:
            out["kernels"].append(m.group(1) + " " + m.group(2))
    return out


# --------------------------------------------------------------------------------------------------------------
# verdict: one definition, used by --measure and --check

def judge(ev):
    runs = ev["runs"]
    pair = parse_pair(runs["pair"]["lines"])
    ops = parse_ops(runs["tile_ops"]["lines"])
    model = {}
    for case in MODEL_CASES:
        on = parse_model(runs["model_on_" + case]["lines"])
        off = parse_model(runs["model_off_" + case]["lines"])
        rc_on, rc_off = runs["model_on_" + case]["rc"], runs["model_off_" + case]["rc"]
        model[case] = {
            "on": on, "off": off,
            "rc_on": rc_on, "rc_off": rc_off,
            # both runs finished and judged themselves OK with a figure each: the reference is no reference otherwise
            "both_ran": rc_on == 0 and rc_off == 0 and on["ok"] and off["ok"] and
                        on.get("nmse") is not None and off.get("nmse") is not None,
            # the pair path moved values: else the switch did not reach this case's merge
            "engaged": on.get("nmse") is not None and off.get("nmse") is not None and on["nmse"] != off["nmse"],
            "greedy_parity_vs_f32_accum": on.get("argmax_agree") is not None and
                                          on.get("argmax_agree") == off.get("argmax_agree") and
                                          on.get("rows") == off.get("rows") and bool(on.get("rows")),
            "rel_decode_on_e4m3": any(k + " f8_e4m3" in on["kernels"] for k in REL_DECODE_FAMILY),
        }

    checks = {
        "device_sm_120": str(ev["device"].get("compute_cap", "")).startswith("12."),
        "pair_test_passed": runs["pair"]["rc"] == 0 and pair["passed"],
        "pair_gain_at_least_bar": pair.get("pair_gain", 0.0) >= BAR_PAIR_GAIN,
        "single_lane_fold_exact": pair.get("fold32_exact_lanes") == 32,
        "tile_ops_all_passed": runs["tile_ops"]["rc"] == 0 and ops["total"] is not None and ops["total"] > 0
                               and ops["passed"] == ops["total"] and ops["fail_lines"] == 0,
        "archs_inkling2_passed": runs["archs_inkling2"]["rc"] == 0,
    }
    for case in MODEL_CASES:
        m = model[case]
        checks["model_%s_ok" % case] = m["both_ran"] and m["on"]["nmse"] <= BAR_MODEL_NMSE
        checks["model_%s_engaged" % case] = m["engaged"] and m["rel_decode_on_e4m3"]
        checks["model_%s_greedy_parity" % case] = m["greedy_parity_vs_f32_accum"]

    failed = sorted(k for k, v in checks.items() if not v)
    return {
        "pair": pair, "tile_ops": ops, "model": model, "checks": checks,
        "verdict": {
            "pass": not failed,
            "failed": failed,
            "bars": {"pair_gain": BAR_PAIR_GAIN, "model_nmse": BAR_MODEL_NMSE},
        },
    }


# --------------------------------------------------------------------------------------------------------------
# measurement on the box

def run_binary(cmd, env_add, cwd):
    env = dict(os.environ)
    for k in ("GGML_CUDA_FATTN_FP8_ACCUM",):
        env.pop(k, None)
    env.update(env_add)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=True, errors="replace")
    text = r.stdout + "\n" + r.stderr
    lines = [l.rstrip() for l in text.splitlines() if LINE_PATTERNS.search(l)]
    return {"cmd": cmd, "env": env_add, "rc": r.returncode, "seconds": round(time.time() - t0, 3),
            "lines": lines[-400:]}


def gpu_info():
    r = subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap,driver_version", "--format=csv,noheader"],
                       text=True, capture_output=True)
    parts = [p.strip() for p in r.stdout.strip().splitlines()[0].split(",")] if r.returncode == 0 and r.stdout else []
    return {"name": parts[0], "compute_cap": parts[1], "driver": parts[2]} if len(parts) >= 3 else {}


def cmd_measure(args):
    bdir = Path(args.build_dir)
    binp = bdir / "bin"
    require((binp / "test-inkling-fp8-pair").is_file(), "no test-inkling-fp8-pair in %s" % binp)
    accum = {"GGML_CUDA_FATTN_FP8_ACCUM": "1"}
    runs = {
        "pair": run_binary([str(binp / "test-inkling-fp8-pair")], {}, str(ROOT)),
        "tile_ops": run_binary([str(binp / "test-backend-ops"), "test", "-b", "CUDA0", "-o", "FLASH_ATTN_EXT",
                                "-p", "hs=128,.*type_KV=f8_e4m3"],
                               dict(accum, GGML_CUDA_FATTN_REL_DECODE_ONE_BYTE="1"), str(ROOT)),
    }
    for case in MODEL_CASES:
        cmd = [str(binp / "test-llama-archs"), "--inkling-path", "kv1b-e4m3/" + case,
               "--inkling-path-device", "CUDA0", "-s", "1234"]
        runs["model_on_" + case] = run_binary(cmd, dict(ONE_BYTE_ENV, **accum), str(ROOT))
        runs["model_off_" + case] = run_binary(cmd, dict(ONE_BYTE_ENV), str(ROOT))
    runs["archs_inkling2"] = run_binary([str(binp / "test-llama-archs"), "-a", "inkling2"], accum, str(ROOT))

    ev = {
        "row": ROW,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": os.uname().nodename,
        "device": gpu_info(),
        "build_dir": str(bdir),
        "git_head": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True,
                                   capture_output=True).stdout.strip(),
        "source_sha256": {s: sha256_file(ROOT / s) for s in SOURCES},
        "switch": "GGML_CUDA_FATTN_FP8_ACCUM=1",
        "what_is_measured": (
            "partial traffic stays 4 bytes a value (one u32 pair slot replaces one f32), so the win is the fp8 MMA "
            "rate and one rounding a fold; this evidence judges quality only. Prefill and decode time of the "
            "integrated path are not measured here."),
        "runs": runs,
    }
    ev.update(judge(ev))
    return ev


# --------------------------------------------------------------------------------------------------------------
# check

def cmd_check(args):
    p = Path(args.evidence)
    require(p.is_file(), "evidence file %s does not exist" % p)
    ev = json.loads(p.read_text())
    require(ev.get("row") == ROW, "evidence is for another row")
    for key in ("device", "runs", "source_sha256", "checks", "verdict"):
        require(key in ev, "evidence lacks %s" % key)
    for s in SOURCES:
        require(ev["source_sha256"].get(s) == sha256_file(ROOT / s), "the evidence was measured from a different %s" % s)
    for name in ["pair", "tile_ops", "archs_inkling2"] + ["model_%s_%s" % (w, c) for w in ("on", "off") for c in MODEL_CASES]:
        require(name in ev["runs"], "evidence lacks the %s run" % name)
        require(ev["runs"][name]["lines"] or name == "archs_inkling2", "the %s run kept no lines" % name)
    for c in MODEL_CASES:
        require(ev["runs"]["model_on_" + c]["env"].get("GGML_CUDA_FATTN_FP8_ACCUM") == "1",
                "the model %s run did not have the switch on" % c)
        require("GGML_CUDA_FATTN_FP8_ACCUM" not in ev["runs"]["model_off_" + c]["env"],
                "the model %s reference run had the switch on" % c)

    again = judge(ev)
    require(again["checks"] == ev["checks"], "the recorded checks do not follow from the recorded lines")
    require(again["verdict"]["pass"] == ev["verdict"]["pass"], "the recorded verdict does not follow from the lines")

    pair, model = again["pair"], again["model"]
    if not again["verdict"]["pass"]:
        print("FAIL: " + ", ".join(again["verdict"]["failed"]))
        return 1
    print("PASS: FP8 pair accumulation holds f16 quality on %s (cc %s)" % (ev["device"]["name"], ev["device"]["compute_cap"]))
    print("      pair: hi err %.3e, hi+lo err %.3e (%.1fx), fold32 %d/32 exact, 256-part fold %.3e"
          % (pair["hi_err"], pair["two_term_err"], pair["pair_gain"], pair["fold32_exact_lanes"],
             pair["fold_slots_worst_rel"]))
    print("      tile ops: %d/%d f8_e4m3 FLASH_ATTN_EXT cases against the host" % (again["tile_ops"]["passed"],
                                                                                    again["tile_ops"]["total"]))
    for c in MODEL_CASES:
        m = model[c]
        print("      model %-7s NMSE vs F16 %.3e on, %.3e off; argmax %d/%d on, %d/%d off"
              % (c, m["on"]["nmse"], m["off"]["nmse"], m["on"]["argmax_agree"], m["on"]["rows"],
                 m["off"]["argmax_agree"], m["off"]["rows"]))
    return 0


# --------------------------------------------------------------------------------------------------------------
# --run: measure through on-box and write the evidence locally

def cmd_run(args):
    out = Path(args.evidence or EVIDENCE_DEFAULT)
    cmd = ["on-box", "python3", "scripts/inkling2/fp8_accumulation.py", "--measure", "--build-dir", args.build_dir]
    r = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)
    if r.returncode != 0:
        raise SystemExit("measurement failed (%d):\n%s\n%s" % (r.returncode, r.stdout[-4000:], r.stderr[-4000:]))
    ev = json.loads(r.stdout)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ev, indent=1) + "\n")
    print("evidence: %s" % out)
    print("verdict: %s %s" % ("PASS" if ev["verdict"]["pass"] else "FAIL", ev["verdict"]["failed"]))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--evidence", default=EVIDENCE_DEFAULT)
    ap.add_argument("--build-dir", default="/home/riftuser/build-torad-120a")
    args = ap.parse_args()
    if args.measure:
        print(json.dumps(cmd_measure(args)))
        return 0
    if args.run:
        return cmd_run(args)
    if args.check:
        return cmd_check(args)
    ap.print_help()
    return 64


if __name__ == "__main__":
    sys.exit(main())
