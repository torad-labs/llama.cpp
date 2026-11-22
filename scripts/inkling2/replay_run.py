#!/usr/bin/env python3
"""Run fixed-input diagnostic arms on an already authorized host.

This program never rents, starts, or stops a machine. Prepare the input with
prepare_input.py (or replay_prepare.py) and use the original server plan for
engine and KV settings. It runs diagnostic readback, not a serving throughput
benchmark. --smoke cuts the prompts short and runs only the compact arm, to
check the whole pipeline on the real model before the long run.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_compare import ARMS, check_selected, compare
from replay_prepare import token_digest

VALUE_FLAGS = ("-m", "-sm", "-ts", "-ngl", "-fa", "-c", "-np", "-b", "-ub", "--cache-type-k", "--cache-type-v")


def engine_args(plan):
    # Do not send server-only cache, renderer, or scheduling flags to the example.
    argv = plan["argv"]
    return [item for flag in VALUE_FLAGS for item in (flag, argv[argv.index(flag) + 1])] + ["--attn-mask-bits"]


def query_first_exercised(arm_dir, ranks):
    """Rows whose recorded attention could take the query-first launch, and all rows.

    The launcher needs 23 queries, 12 query heads and 2 KV heads on a rank, a low-mean mask, and 256
    partitions (a low-mean step has no row bound, so that means at least 256*1024 cells). The callback may
    see logical shapes (heads split across the ranks) or per-rank ones, so accept both."""
    rows = json.loads((Path(arm_dir) / "rows.json").read_bytes())["rows"]
    heads = {(12, 2), (12 * ranks, 2 * ranks)}
    eligible = sum(any(a["queries"] == 23 and a["low_mean"] and a["kv_cells"] >= 256 * 1024 and
                       (a["query_heads"], a["kv_heads"]) in heads for a in row["attention_inputs"])
                   for row in rows)
    return eligible, len(rows)


def smoke_input(spec, n_common=4096, n_private=512):
    """The same prompts cut short, with their digests recomputed, for a quick check on the real model."""
    common = spec["common"][:n_common]
    private = [tail[:n_private] for tail in spec["private"]]
    return dict(spec, common=common, private=private, common_sha256=token_digest(common),
                private_sha256=[token_digest(tail) for tail in private])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--driver", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--cuda-libs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="new directory for the captures")
    p.add_argument("--smoke", action="store_true", help="short prompts, compact arm only")
    args = p.parse_args()
    plan = json.loads(args.plan.read_bytes())
    cmd = engine_args(plan)
    n_seq = int(cmd[cmd.index("-np") + 1])
    spec = json.loads(args.input.read_bytes())
    if args.smoke:
        spec = smoke_input(spec)
    elif (n_seq, len(spec["private"]), len(spec["common"])) != (23, 23, 196608) or any(
            len(tail) != 65536 for tail in spec["private"]):
        p.error("the input does not match the original 23-session workload")
    gpus = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).splitlines()
    if len(gpus) != 4 or not all("RTX PRO 6000 Blackwell" in g for g in gpus):
        p.error("this is not the original four-card hardware")

    env = {k: v for k, v in os.environ.items() if not k.startswith(("GGML_", "LLAMA_"))}
    env.update(plan["env"], LD_LIBRARY_PATH=str(args.cuda_libs.resolve()),
               LLAMA_INKLING2_EXPERT_SPLIT="0", GGML_CUDA_AR_HOST_OWNER_STREAM_ORDERED="0")
    args.out.mkdir()
    input_path = args.input
    if args.smoke:
        input_path = args.out / "smoke-input.json"
        input_path.write_text(json.dumps(spec, separators=(",", ":")))
    dirs = {arm: args.out / arm for arm in ARMS}
    for arm in (("greedy",) if args.smoke else ARMS):
        print(f"capturing {arm}", flush=True)
        arm_env = dict(env, GGML_CUDA_FATTN_REL_QUERY_FIRST=str(int(arm == "attention")))
        with (args.out / (arm + ".log")).open("wb") as out:
            subprocess.run([args.driver.resolve(), "--inkling-replay", input_path, dirs[arm],
                            "compact" if arm == "greedy" else "full", *cmd],
                           env=arm_env, stdout=out, stderr=subprocess.STDOUT, check=True)

    if args.smoke:
        matched, total, bad = check_selected(dirs["greedy"])
        print(f"smoke: {matched} of {total} compact rows chose the token their own scores pick", flush=True)
        if bad:
            raise SystemExit(f"smoke failed: compact rows disagree with their scores: {bad[:5]}")
        return

    result = compare(dirs)
    (args.out / "kl.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    ranks = len(plan["argv"][plan["argv"].index("-ts") + 1].split(","))
    used, total = query_first_exercised(dirs["attention"], ranks)
    (args.out / "attention-exercised.json").write_text(json.dumps({"rows_eligible": used, "rows": total}) + "\n")
    if used != total:
        print(f"WARNING: the attention arm could take the query-first launch on only {used} of {total} rows; "
              "its agreement with stock says nothing about that kernel", flush=True)
    selected = {arm: check_selected(dirs[arm]) for arm in ARMS}
    (args.out / "selected-check.json").write_text(json.dumps(
        {arm: {"matched": m, "rows": t, "mismatched": b} for arm, (m, t, b) in selected.items()}) + "\n")
    for arm, (matched, rows, bad) in selected.items():
        print(f"{arm}: {matched} of {rows} rows chose the token their own scores pick", flush=True)
    print(f"compared {len(result['rows'])} identical-input rows; no release decision applied", flush=True)


if __name__ == "__main__":
    main()
