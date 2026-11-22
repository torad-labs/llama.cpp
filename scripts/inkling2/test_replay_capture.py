#!/usr/bin/env python3
"""Exercise diagnostic capture on an existing synthetic Inkling model."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_compare import compare


def token_digest(tokens):
    return hashlib.sha256(",".join(map(str, tokens)).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bin_dir", type=Path, help="a build's bin directory (unused when --driver and --model are given)")
    parser.add_argument("out", type=Path, help="new directory for the synthetic model and records")
    parser.add_argument("--driver", type=Path, help="llama-eval-callback to run (default: in bin_dir)")
    parser.add_argument("--libs", type=Path, help="directory of the shared libraries the driver loads")
    parser.add_argument("--model", type=Path, help="an existing synthetic inkling2 GGUF (default: written with bin_dir/test-llama-archs)")
    parser.add_argument("--devices", help="two backend names such as CUDA0,CUDA0 for a GPU run (default: two CPU ranks)")
    parser.add_argument("--max-kl", type=float, default=0.0, help="largest KL accepted between arms")
    args = parser.parse_args()
    binary = args.bin_dir.resolve()
    out = args.out.resolve()
    out.mkdir()
    if args.model:
        model = args.model.resolve()
    else:
        subprocess.run([binary / "test-llama-archs", "-a", "inkling2", "-o", out, "-s", "1234"], check=True)
        model = next(out.glob("inkling2*.gguf"))
    driver = (args.driver or binary / "llama-eval-callback").resolve()
    common = list(range(3, 11))
    tails = [list(range(11, 19)), list(range(19, 27))]
    forced = [[27, 28, 29, 30], [31, 32, 33, 34]]
    path = out / "input.json"
    with model.open("rb") as f:
        model_hash = hashlib.file_digest(f, "sha256").hexdigest()
    path.write_text(json.dumps({
        "build_sha256": "one synthetic build", "model_sha256": model_hash,
        "common": common, "private": tails, "forced": forced, "logical_cpu_ranks": 0 if args.devices else 2,
        "common_sha256": token_digest(common), "private_sha256": [token_digest(t) for t in tails],
    }))

    arms = {name: out / name for name in ("stock_a", "stock_b", "attention", "greedy")}
    for arm, dst in arms.items():
        env = dict(os.environ, GGML_CUDA_FATTN_REL_QUERY_FIRST=str(int(arm == "attention")))
        if args.libs:
            env["LD_LIBRARY_PATH"] = str(args.libs.resolve())
        cmd = [driver, "--inkling-replay", path, dst,
               "compact" if arm == "greedy" else "full", "-m", model, "-fa", "on",
               "-c", "512", "-b", "64", "-ub", "64", "-np", "2",
               "--attn-mask-bits", "-t", "2", "-tb", "2", "-ngl", "99"]
        if args.devices:
            cmd += ["-dev", args.devices, "-sm", "tensor"]
        log = out / (arm + ".log")
        with log.open("wb") as f:
            result = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
        if result.returncode:
            print(log.read_text(errors="replace")[-8000:], file=sys.stderr)
            result.check_returncode()
        rows = json.loads((dst / "rows.json").read_bytes())["rows"]
        assert len(rows) == 10 and all(r["compact"] == (arm == "greedy") for r in rows)
    result = compare(arms)
    assert len(result["rows"]) == 10
    for row in result["rows"]:
        for name in ("stock_a/stock_b", "attention/stock_a", "greedy/stock_a", "greedy/stock_b"):
            match = row[name]
            assert max(match["kl_ab"], match["kl_ba"]) <= args.max_kl, (row["client"], row["token_position"], name, match)
            assert match["selected_a"] == match["selected_b"]
    print("PASS: full and compact readbacks match at all ten fixed-prefix CPU Meta rows")
    print("The CPU fixture does not exercise query-first CUDA dispatch.")


if __name__ == "__main__":
    main()
