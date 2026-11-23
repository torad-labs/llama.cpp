#!/usr/bin/env python3
"""Serve Inkling 2 Mini (the abliterated IQ3_S pack) on one 16 GB RTX 5080 for Claude Code; --dry-run prints JSON without starting a model."""
import argparse
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]

# The GLM launcher's start-up: the same dry run, file checks and removal of inherited engine overrides
_spec = importlib.util.spec_from_file_location("glm53_serve", ROOT / "scripts/glm53/serve.py")
glm53 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(glm53)
own_path = glm53.own_path
execution_environment = glm53.execution_environment

# The pack is AtomicChat's (huggingface.co/AtomicChat/Inkling-2-Mini-Uncensored-GGUF), not converted here, so its name is the one
# it is published under: 13.3 GB, a fixed loop (loop.count 3 over core layers 6 to 18, loop.varying false), and the chat template
# is inside the GGUF (6,868 characters; read from the public file's header), so the launcher passes no template file.
SOURCE = "Inkling-2-Mini-Uncensored"
PACKS = ROOT / "models" / SOURCE
PACK = f"{SOURCE}-AD-IQ3_S.gguf"
PACK_PASSES = 3  # loop.count of the pack

SLOTS = 4
CONTEXT = 233472
KV_TYPE = "q4_0"
BATCH_SIZE = 2048
UBATCH_SIZE = 512
MAX_SLOTS = 256  # the engine's sequence limit (LLAMA_MAX_SEQ)
MAX_LOOP_PASSES = 64  # the range the knob registry (tunables.toml) declares for LLAMA_LOOP_PASSES and _MAX
# The one serving configuration for the abliterated Inkling 2 Mini on a 16 GB RTX 5080, for Claude Code (long context, several
# sessions at once): what the 2026-11-21 Mini searches settled on (research_notes/inkling-mini-5090-results/results.tgz). They ran
# on a 32 GB RTX 5090 held to the 5080's 16,303 MiB by a ballast process, so a launch that does not fit there fails as it would
# on the 5080. A score is the geometric mean of solo, pair and four-session decode and agent-turn rate, in tokens a second. "Dev"
# is the build with this tree's kernel changes, "first" the one before them. Every variant tested against it lost or was not
# run, and their numbers are recorded here so nothing is re-derived:
#   q4_0 K and V. The dev frontier (out/f-dev/frontier.jsonl: np 4, a cell read with prompts filling 75% of its own pool):
#     q4_0/q4_0  pool 233,472  39,291 tokens a session  solo 50.9  pair  81.6  four 237.3  agent 116.4  score 103.5
#     q8_0/q8_0  pool 118,784  17,787 tokens a session  solo 67.1  pair 110.3  four 257.6  agent 134.2  score 126.5
#     q8_0 K, q4_0 V  126,976  19,323 tokens a session  solo 70.8  pair 120.8  four 186.1  agent 106.6  score 114.2
#     The scores rank them the other way round, but the cells are different loads. On one workload q4_0 on either side stayed within 4%
#     of q8_0 on four-session decode and was faster solo (out/kv, first build, pool 65,536: q8_0/q8_0 solo 72.8 four 209.7; q4_0 K
#     79.7 / 201.8; q4_0 V 80.5 / 204.5), and q4_0 holds 1.97 times the tokens in the same memory, which is the long context. f16 V was
#     fastest there (93.5 / 222.9) at 516 MiB more (15,462 against 14,946). Sliding-window layers at f16 (first build only, np 4): pool
#     131,072 against 167,936 (-22%), score 89.0 against 79.4 at 78% of the load; no dev-build arm used them.
#   KV quality settled q4_0 (2026-11-21, real RTX 5080, fixed build): top-16 next-token distributions at 464 positions
#     (prose and tool-call JSON) in 20 real Claude Code contexts, 8K to 51K tokens, against f16 as reference. The raw runs
#     again anywhere: research_notes/kv-quality.py with kv-quality-contexts.json (the 20 contexts saved as token ids),
#     results in inkling-mini-5090-results/loop/kv-quality.json. f16 twice TV 0 (deterministic). Perturbation floor (f16
#     at -ub 256, summation order only) TV 0.0136, match 0.983. q8_0/q8_0 0.0133 / 0.981: on the floor at every length.
#     q4_0/q4_0 0.0208 / 0.972: +0.0072 [+0.0051, +0.0093] over the floor, flat from 8K to 51K, mostly prose word choice;
#     no top token changed where f16 was sure (p >= 0.9, 324 positions incl. structural JSON), flips only at near-ties
#     (p <= 0.57); no systematic loss of the real next token's probability. K and V cost the same, so q4_0 stays for its
#     ~58K tokens a session at 4 sessions against q8_0's ~30K when one real Claude Code request carries ~18K of preamble.
#   -np 4: the workload the score measures. Dev frontier, q4_0/q4_0: np 1 249,856, np 2 245,760, np 4 233,472, np 8 212,992 cells,
#     about 5,100 a slot. np 8 read solo 52.7, pair 82.8, eight together 377.0, agent 151.3: eight together is not the four-session score.
#   -c 233,472: the largest pool of that frontier at np 4 (237,568 did not load; sampled peak 15,573 MiB of 16,303; the first build
#     fit 167,936). A pool cell costs 10.1 KiB at q4_0 (peaks of out/k1 and out/ab4: 14,651 MiB at 131,072, 14,974 at 163,840), and the
#     same 39,000-token sessions decode about 10% slower in this pool than in 163,840 (out/ab4: 57.0 / 90.6 / 266.5 / 128.9). Every
#     frontier cell ran LLAMA_KV_SEQ_CELLS = its pool, so a session may take all of it; a lower value caps a session's context, and
#     Inkling 2 has no pooled indexer for it to size.
#   -b 2048 -ub 512, the engine's defaults, written out so the launch does not move with them. out/k1 (dev, q4_0/q4_0 np 4 -c 131072,
#     noise 0.005, threshold 0.02), score against the base:
#     -ub 128 0.906, 256 0.960, 384 0.971, 768 1.007 (peak 14,812 MiB), 1024 1.022 with repeats 0.994, 1.023, 1.031, 0.995 (14,958 against
#     14,656: 302 MiB, 30,700 pool cells). The mean gain is 1.3% for 13% of a pool that is already at the memory limit. -b 1024 0.975, 4096 1.000.
#   In the base of every arm and not run against: -ngl 99, -fa on, --kv-unified, --cache-ram 0, --fit off, --metrics,
#     LLAMA_SLOT_PREFIX_COPY=1 (a new session copies the cells of a prefix another holds: a replay session's first request is 19,273
#     tokens, mostly the shared Claude Code prompt and tools), --jinja with the template in the GGUF, and sampling without
#     truncation at the GGUF's temperature (general.sampling.temp 1.0).
#   Nothing else was adopted. out/k1 ran 136 single settings on the dev build (graph, L2, MMQ, MMVQ, thread and batch knobs, and
#     the legacy switches of what the kernel changes replaced): none beat 0.02 but -ub 1024 above. --min-p 0.05 won out/s2b (final scores
#     1.24, 1.16) against a base that ran n-gram speculation and scored 0.92 to 1.25 on its own repeats; on the replay it scored 0.984
#     (out/r1). --attn-mask-bits, --prompt-step and --prompt-chunk, which the Small launcher carries, were never run in a Mini arm.
#   No speculation. out/r2 (the replay of real Claude Code sessions, a session's rate its tokens over its decode time): base 87.6 /
#     135.7 / 181.0 / 110.3, its repeats scored 0.986, 0.965, 1.001; n-gram speculation 0.931 to 0.997; the DSpark drafter at depths 2, 3, 4
#     0.967, 1.002, 0.988 (depth 3 with --backend-sampling 1.026), at 1,000 MiB more (15,150 against 14,140). out/s2 (synthetic, depths 1
#     to 7): four-session decode between 250.0 (depth 1) and 174.4 (depth 7) against 266.6. The +27% to +39% of the n-gram arms in out/r1
#     came from the replay as it was before its 19:07 UTC rewrite (r1 recorded no per-turn rows; the rewrite's comment says a mean of
#     per-turn rates lets one copied file decide a phase) and did not repeat in r2.


def loop_env(passes):
    """The variables a run-time pass count needs on this fixed-loop pack. Every count but its own is untrained
    (LLAMA_LOOP_PASSES_ANY, which the engine requires of a loop.varying false model), and the cache slots of more passes than the
    pack's own are built at load (LLAMA_LOOP_PASSES_MAX)."""
    if passes == 0:
        return {}
    env = {"LLAMA_LOOP_PASSES": str(passes)}
    if passes != PACK_PASSES:
        env["LLAMA_LOOP_PASSES_ANY"] = "1"
    if passes > PACK_PASSES:
        env["LLAMA_LOOP_PASSES_MAX"] = str(passes)
    return env


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=PACKS / PACK,
                        help=f"A local GGUF in place of the pack (default: models/{SOURCE}/{PACK})")
    parser.add_argument("--server", type=Path, default=ROOT / "build/bin/llama-server",
                        help="llama-server from this fork, with matching libraries beside it")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8103)
    parser.add_argument("--slots", type=int, default=SLOTS, help=f"Sessions sharing the pool, 1..{MAX_SLOTS} (default: {SLOTS})")
    parser.add_argument("--context", type=int, default=CONTEXT,
                        help=f"Tokens in the shared pool (default: {CONTEXT}, the most that fit 16,303 MiB at {SLOTS} sessions)")
    parser.add_argument("--loop-passes", type=int, default=0, metavar="N",
                        help=f"Passes a decode runs through the loop's core layers (default 0: the pack's own {PACK_PASSES}, neither "
                             f"variable set). Any other count is untrained on this fixed-loop pack, and more than {PACK_PASSES} builds "
                             "cache slots for the extra passes at load, which comes out of the pool's memory: lower --context to fit")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(sys.argv[1:] if argv is None else list(argv))


def make_plan(args):
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be 1..65535")
    if not 1 <= args.slots <= MAX_SLOTS:
        raise ValueError(f"slots must be 1..{MAX_SLOTS}")
    if args.context < 256:
        raise ValueError("context is a pool of 256 tokens or more")
    if not 0 <= args.loop_passes <= MAX_LOOP_PASSES:
        raise ValueError(f"loop-passes must be 0..{MAX_LOOP_PASSES}")
    if not args.host:
        raise ValueError("host must be a nonempty address")
    server = args.server.expanduser().resolve()
    argv = [str(server), "-m", own_path(args.model),
            "-ngl", "99", "--jinja", "-fa", "on", "-b", str(BATCH_SIZE), "-ub", str(UBATCH_SIZE),
            # Sampling without truncation, at the temperature the GGUF records
            "--temp", "1.0", "--top-p", "1.0", "--top-k", "0", "--min-p", "0.0",
            "--kv-unified", "--cache-ram", "0", "--metrics", "--host", args.host, "--port", str(args.port), "--fit", "off",
            "--cache-type-k", KV_TYPE, "--cache-type-v", KV_TYPE, "-c", str(args.context), "-np", str(args.slots)]
    env = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "0",
           # A session may take the whole pool, as in every cell of the frontier
           "LLAMA_KV_SEQ_CELLS": str(args.context),
           "LLAMA_SLOT_PREFIX_COPY": "1",
           "LD_LIBRARY_PATH": str(server.parent),
           **loop_env(args.loop_passes)}
    return argv, env


def main(args):
    argv, settings = make_plan(args)
    return glm53.launch(argv, settings, [("model", Path(argv[argv.index("-m") + 1]))], args.dry_run)


if __name__ == "__main__":
    try:
        raise SystemExit(main(arguments()))
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2)
