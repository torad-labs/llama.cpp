#!/usr/bin/env python3
"""Serve Inkling 2 Small on four cards from a local abliterated pack; --dry-run prints JSON without starting a model."""
import argparse
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "models/templates/thinkingmachines-Inkling-2-Small.jinja"
TEMPLATE_SHA256 = "e9cfd8e75d1272633c2e72b3cc76e99567e2bc8912b960483aed483ad35d54c7"

# The GLM launcher's start-up: the same asset check, dry run, file checks, overrides removed and --prompt-chunk probe
_spec = importlib.util.spec_from_file_location("glm53_serve", ROOT / "scripts/glm53/serve.py")
glm53 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(glm53)
own_path = glm53.own_path
execution_environment = glm53.execution_environment

# The packs are converted from the pinned abliterated revision (docs/torad/research/inkling-2-small/09-abliterated-target.md),
# and each name carries that revision and its recipe so no file can pass for a stock build of the same bit width.
SOURCE = "Inkling-2-Small-abliterated-e4853745"
PACKS = ROOT / "models" / SOURCE
MMPROJ = f"mmproj-{SOURCE}-F16.gguf"

# One session holds 262,144 tokens at most (the limit vLLM also keeps at tensor parallel 4), and 23 sessions are reserved
# against one shared pool in the GLM Tensor 4 pattern. The pools and cache types are the capacity study's
# (docs/torad/research/inkling-2-small/05-capacity-and-tensor4.md), sums from file sizes, not a load: the first box
# read replaces them.
SLOTS = 23
SLOT_CONTEXT = 262144
MAX_SLOTS = 256  # the engine's sequence limit (LLAMA_MAX_SEQ)
PROFILES = {
    # Q3-class pack, about 20 GiB a card left for KV: 2.65M tokens at F16. The pool starts at nine full sessions until
    # the load leg justifies more; F16 KV until a Q8_0 quality leg reads clean.
    "tensor-4": {"pack": f"{SOURCE}-Q3_K_M-imatrix.gguf", "context": 9 * SLOT_CONTEXT, "kv": "f16"},
    # IQ3-class pack, about 34 GiB a card for KV: 8.39M tokens at Q8_0. The pool holds all 23 sessions full at once.
    "tensor-4-capacity": {"pack": f"{SOURCE}-IQ3_XXS-imatrix.gguf", "context": SLOTS * SLOT_CONTEXT, "kv": "q8_0"},
    # Q4-class pack, about 4 GiB a card for KV: 0.54M tokens at F16. Two full sessions; a quality experiment only.
    "tensor-4-quality": {"pack": f"{SOURCE}-Q4_K_M-imatrix.gguf", "context": 2 * SLOT_CONTEXT, "kv": "f16"},
}


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="tensor-4",
                        help="Pack, pool and KV type on four cards with one shared pool (default: tensor-4)")
    parser.add_argument("--packs", type=Path, default=PACKS,
                        help="Folder holding the converted packs under their recipe names (default: models/" + SOURCE + ")")
    parser.add_argument("--model", type=Path, help="A local GGUF in place of the profile's pack")
    parser.add_argument("--mmproj", type=Path, help="A local vision projector in place of the packs' F16 one")
    parser.add_argument("--text-only", action="store_true", help="Load no vision projector")
    parser.add_argument("--server", type=Path, default=ROOT / "build/bin/llama-server",
                        help="llama-server from this fork, with matching libraries beside it")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8103)
    parser.add_argument("--slots", type=int, help=f"Sessions sharing the pool, 1..{MAX_SLOTS} (default: {SLOTS})")
    parser.add_argument("--context", type=int, help="Tokens in the shared pool (default: the profile's)")
    parser.add_argument("--slot-context", type=int, help=f"Tokens one session holds at most (default: {SLOT_CONTEXT})")
    parser.add_argument("--kv-cache", choices=["f16", "q8_0"], help="KV cache type (default: the profile's)")
    parser.add_argument("--cache-ram", type=int, default=32768, metavar="MIB",
                        help="RAM the prompt cache may hold, in MiB (default: 32768; -1 is no limit, 0 is no cache)")
    parser.add_argument("--prompt-step", type=int, default=1024,
                        help="Prompt tokens per step beside generation (default: 1024, GLM's first read; 0 removes the cap)")
    parser.add_argument("--prompt-step-ms", type=int, metavar="MS",
                        help="The time a step's prompt tokens may take beside replies; the server sets their count from "
                             "its last steps, up to --prompt-step (opt-in: not read for Inkling)")
    parser.add_argument("--reduction-wire", choices=["bf16", "q8"], default="bf16",
                        help="What a card's share of a reduction crosses as (default: bf16 until its quality leg: "
                             "reductions feed convolutions that write persistent sequence state)")
    parser.add_argument("--reduction-route", choices=["host", "engine"], default="host",
                        help="host sends reductions on the copy engines through host memory, the route GLM Tensor 4 "
                             "read fastest on rented hosts; engine leaves the routes the engine probes (default: host)")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def make_plan(args):
    profile = PROFILES[args.profile]
    slots = args.slots if args.slots is not None else SLOTS
    context = args.context if args.context is not None else profile["context"]
    slot_context = args.slot_context if args.slot_context is not None else SLOT_CONTEXT
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be 1..65535")
    if not 1 <= slots <= MAX_SLOTS:
        raise ValueError(f"slots must be 1..{MAX_SLOTS}")
    if not 256 <= slot_context < context:
        raise ValueError("slot-context is from 256 tokens to under the context")
    if args.prompt_step < 0 or args.prompt_step_ms is not None and args.prompt_step_ms < 1:
        raise ValueError("prompt-step must be nonnegative and prompt-step-ms 1 or more")
    if args.cache_ram < -1:
        raise ValueError("cache-ram is a size in MiB (-1: no limit)")
    if args.text_only and args.mmproj:
        raise ValueError("text-only loads no projector, so it takes no --mmproj")
    if not args.host or any(char in args.host for char in "\n\r\0"):
        raise ValueError("host must be a nonempty address without control characters")
    server = args.server.expanduser().resolve()
    model = own_path(args.model) if args.model else own_path(args.packs / profile["pack"])
    kv = args.kv_cache or profile["kv"]
    argv = [str(server), "-m", model]
    if not args.text_only:
        argv += ["--mmproj", own_path(args.mmproj) if args.mmproj else own_path(args.packs / MMPROJ)]
    # Attention splits by heads (2 KV heads a card), so the caches split with them and are never mirrored
    argv += ["-ngl", "99", "-sm", "tensor", "-ts", "1,1,1,1", "--jinja", "-fa", "on",
             "--cache-type-k", kv, "--cache-type-v", kv,
             "-b", "2048", "-ub", "2048", "--prompt-step", str(args.prompt_step), "--prompt-chunk", "128"]
    if args.prompt_step_ms is not None:
        argv += ["--prompt-step-ms", str(args.prompt_step_ms)]
    # The model card's sampling defaults; reasoning effort is left to the template's default of high
    argv += ["--temp", "1.0", "--top-p", "1.0", "--top-k", "0", "--min-p", "0.0",
             "--alias", Path(profile["pack"]).stem, "--chat-template-file", str(TEMPLATE),
             "-c", str(context), "-np", str(slots), "--kv-unified",
             "--cache-ram", str(args.cache_ram), "--no-cache-idle-slots",
             "--metrics", "--host", args.host, "--port", str(args.port), "--log-timestamps"]
    env = {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        # A session ends at slot-context tokens, and the graph reserved at load is sized for sessions that long
        "LLAMA_KV_SEQ_CELLS": str(slot_context),
        "LD_LIBRARY_PATH": str(server.parent),
    }
    if args.reduction_route == "host":
        env.update({"GGML_CUDA_ALLREDUCE": "internal", "GGML_CUDA_AR_LL_MAX_BYTES": "0", "GGML_CUDA_AR_P2P_MAX_BYTES": "0",
                    "GGML_CUDA_AR_COPY_THRESHOLD": "1", "GGML_CUDA_AR_COPY_PEER": "0",
                    "GGML_CUDA_AR_GATHER_OWNER_SLICES": "0"})
    if args.reduction_wire == "q8":
        env["GGML_CUDA_AR_WIRE_Q8"] = "1"
    return argv, env


def main(args):
    argv, settings = make_plan(args)
    glm53.verify_assets({TEMPLATE: TEMPLATE_SHA256})
    paths = [("model", Path(argv[argv.index("-m") + 1]))]
    if "--mmproj" in argv:
        paths.append(("mmproj", Path(argv[argv.index("--mmproj") + 1])))
    return glm53.launch(argv, settings, paths, args.dry_run)


if __name__ == "__main__":
    try:
        raise SystemExit(main(arguments()))
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2)
