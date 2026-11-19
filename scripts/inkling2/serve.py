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

SLOTS = 23
SLOT_CONTEXT = 262144
CODING_SLOT_CONTEXT = 294912  # Shared by prompt, reasoning, tool history and generated text.
BATCH_SIZE = 2048
MAX_SLOTS = 256  # the engine's sequence limit (LLAMA_MAX_SEQ)
PROFILES = {
    "tensor-4": {"pack": f"{SOURCE}-Q3_K_M-imatrix.gguf", "mmproj": "mmproj-F16.gguf",
                 "context": 3014656, "slot_context": CODING_SLOT_CONTEXT, "kv": "f16", "cache_ram": 262144,
                 "checkpoint_every": 32768, "expert_row_limit": 23, "bucket_prefill_kv": True,
                 "stable_prefill": False, "sort_expert_rows": True,
                 "prefix_copy": True, "mean_mask_decode": True, "reduction_route": "host-sliced", "host_owner_rows": True},
    # IQ3-class pack, about 34 GiB a card for KV: 8.39M tokens at Q8_0. The pool holds all 23 sessions full at once.
    "tensor-4-capacity": {"pack": f"{SOURCE}-IQ3_XXS-imatrix.gguf", "context": SLOTS * SLOT_CONTEXT, "kv": "q8_0"},
    # Q4-class pack, about 4 GiB a card for KV: 0.54M tokens at F16. Two full sessions; a quality experiment only.
    "tensor-4-quality": {"pack": f"{SOURCE}-Q4_K_M-imatrix.gguf", "context": 2 * SLOT_CONTEXT, "kv": "f16"},
}


def expert_row_limit(value):
    return value if value == "auto" else int(value)


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
    parser.add_argument("--log-verbosity", type=int, choices=range(6), help="Server log level (4 includes model and buffer placement)")
    parser.add_argument("--slots", type=int, help=f"Sessions sharing the pool, 1..{MAX_SLOTS} (default: {SLOTS})")
    parser.add_argument("--context", type=int, help="Tokens in the shared pool (default: the profile's)")
    parser.add_argument("--slot-context", type=int,
                        help=f"Maximum input plus generation per session (tensor-4: {CODING_SLOT_CONTEXT}, other profiles: {SLOT_CONTEXT})")
    parser.add_argument("--kv-cache", choices=["f16", "q8_0"], help="KV cache type (default: the profile's)")
    parser.add_argument("--swa-kv-cache", choices=["f16", "q8_0"],
                        help="Override K and V precision in sliding-window layers (default: follow --kv-cache)")
    parser.add_argument("--attn-mask-bits", action=argparse.BooleanOptionalAction, default=True,
                        help="Use a one-bit mask for supported flash attention paths (default: enabled)")
    parser.add_argument("--bucket-prefill-kv", action=argparse.BooleanOptionalAction,
                        help="Round the global F16 attention span to 8192 cells for single-reader prefill above 32k (enabled in tensor-4)")
    parser.add_argument("--stable-prefill", action=argparse.BooleanOptionalAction,
                        help="Use FP32 tensor-core accumulation and logical key order for eligible Blackwell prefills")
    parser.add_argument("--sort-expert-rows", action=argparse.BooleanOptionalAction,
                        help="Group the 23-token batch by expert for eligible Blackwell Q3_K and Q4_K projections")
    parser.add_argument("--slot-save-path", type=Path, help="Directory for explicit server slot save/restore requests")
    parser.add_argument("--mean-mask-decode", action=argparse.BooleanOptionalAction,
                        help="Try grouped SM120 F16 global attention for 23 decoding sessions with low average KV visibility")
    parser.add_argument("--reference-sinks", action="store_true",
                        help="Use the padded, shifted relative-position table for comparison with the original attention")
    parser.add_argument("--ubatch-size", type=int, default=BATCH_SIZE,
                        help=f"Maximum tokens in one physical microbatch (1..{BATCH_SIZE}, default: {BATCH_SIZE})")
    parser.add_argument("--keep-graphs", "--kept-graphs", type=int, metavar="N",
                        help="Keep up to N scheduler graphs (1..256) and reuse matching batch shapes across sessions "
                             "with fixed convolution rows. The default uses the engine's graph limits.")
    parser.add_argument("--expert-row-limit", type=expert_row_limit, metavar="N|auto",
                        help="Try the SM120 Q3_K/Q4_K/Q5_K routed matrix-vector kernel up to N rows (8..32); "
                             "auto uses the engine's dispatch (tensor-4: 23, otherwise auto)")
    parser.add_argument("--cache-ram", type=int, metavar="MIB",
                        help="RAM budget for displaced sessions and their checkpoints, in MiB. Idle sessions stay "
                             "resident until a slot is reused or KV is needed (tensor-4: 262144, otherwise 32768; -1 removes the byte "
                             "limit, 0 disables the cache)")
    parser.add_argument("--prefix-copy", action=argparse.BooleanOptionalAction,
                        help="Reuse matching KV cells across sessions, including simultaneous text requests with a long common prefix")
    parser.add_argument("--ctx-checkpoints", type=int, metavar="N",
                        help="Maximum rollback checkpoints per slot, also retained in RAM snapshots (engine default: 32)")
    parser.add_argument("--checkpoint-every", type=int, metavar="N",
                        help="Also save a checkpoint every N prompt tokens and at a fork (tensor-4: 32768, engine default: 0, off)")
    for direction in ("save", "restore"):
        mode = parser.add_mutually_exclusive_group()
        mode.add_argument(f"--{direction}-async", dest=f"{direction}_async", action="store_true", default=True,
                          help=f"Allow KV copies for slot exchanges to {direction} in the background (default)")
        mode.add_argument(f"--{direction}-sync", dest=f"{direction}_async", action="store_false",
                          help=f"Wait for all slot-exchange copies on the server thread during {direction}")
    parser.add_argument("--prompt-step", type=int, default=1024,
                        help="Prompt tokens per step beside generation (default: 1024, GLM's first read; 0 removes the cap)")
    parser.add_argument("--prompt-step-ms", type=int, metavar="MS",
                        help="The time a step's prompt tokens may take beside replies; the server sets their count from "
                             "its last steps, up to --prompt-step (opt-in: not read for Inkling)")
    parser.add_argument("--nccl-sliced-gather", action="store_true",
                        help="Gather contiguous channel shards with native NCCL on the engine route")
    parser.add_argument("--host-owner-rows", action=argparse.BooleanOptionalAction,
                        help="On a BF16 host route, receive only the local convolution channels for 23-row TP4 decode")
    parser.add_argument("--reduction-wire", choices=["bf16", "q8"], default="bf16",
                        help="What a card's share of a reduction crosses as (default: bf16 until its quality leg: "
                             "reductions feed convolutions that write persistent sequence state)")
    parser.add_argument("--reduction-route", choices=["host", "host-sliced", "engine"],
                        help="host sends reductions on the copy engines through host memory, the route GLM Tensor 4 "
                             "read fastest on rented hosts. host-sliced stages only owned gather values. engine uses the routes it probes (tensor-4: host-sliced, otherwise host)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    profile = PROFILES[args.profile]
    for key, fallback in (("cache_ram", 32768), ("prefix_copy", False), ("mean_mask_decode", False), ("reduction_route", "host"),
                          ("checkpoint_every", None), ("expert_row_limit", None), ("bucket_prefill_kv", False),
                          ("stable_prefill", False), ("sort_expert_rows", False)):
        if getattr(args, key) is None:
            setattr(args, key, profile.get(key, fallback))
    if args.expert_row_limit == "auto":
        args.expert_row_limit = None
    if args.host_owner_rows is None:
        args.host_owner_rows = (profile.get("host_owner_rows", False) and args.reduction_route == profile["reduction_route"]
                                and args.reduction_wire == "bf16")
    return args


def make_plan(args):
    profile = PROFILES[args.profile]
    slots = args.slots if args.slots is not None else SLOTS
    context = args.context if args.context is not None else profile["context"]
    slot_context = args.slot_context if args.slot_context is not None else profile.get("slot_context", SLOT_CONTEXT)
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be 1..65535")
    if not 1 <= slots <= MAX_SLOTS:
        raise ValueError(f"slots must be 1..{MAX_SLOTS}")
    if not 1 <= args.ubatch_size <= BATCH_SIZE:
        raise ValueError(f"ubatch-size must be 1..{BATCH_SIZE}")
    if args.keep_graphs is not None and not 1 <= args.keep_graphs <= 256:
        raise ValueError("keep-graphs must be 1..256")
    if args.expert_row_limit is not None and not 8 <= args.expert_row_limit <= 32:
        raise ValueError("expert-row-limit must be 8..32")
    if not 256 <= slot_context < context:
        raise ValueError("slot-context is from 256 tokens to under the context")
    if args.prompt_step < 0 or args.prompt_step_ms is not None and args.prompt_step_ms < 1:
        raise ValueError("prompt-step must be nonnegative and prompt-step-ms 1 or more")
    if args.cache_ram < -1:
        raise ValueError("cache-ram is a size in MiB (-1: no limit)")
    if args.ctx_checkpoints is not None and args.ctx_checkpoints < 0:
        raise ValueError("ctx-checkpoints must be nonnegative")
    if args.checkpoint_every is not None and args.checkpoint_every < 0:
        raise ValueError("checkpoint-every must be nonnegative")
    if args.text_only and args.mmproj:
        raise ValueError("text-only loads no projector, so it takes no --mmproj")
    if not args.host or any(char in args.host for char in "\n\r\0"):
        raise ValueError("host must be a nonempty address without control characters")
    server = args.server.expanduser().resolve()
    model = own_path(args.model) if args.model else own_path(args.packs / profile["pack"])
    kv = args.kv_cache or profile["kv"]
    argv = [str(server), "-m", model]
    if not args.text_only:
        argv += ["--mmproj", own_path(args.mmproj) if args.mmproj else own_path(args.packs / profile.get("mmproj", MMPROJ))]
    # Attention splits by heads (2 KV heads a card), so the caches split with them and are never mirrored
    argv += ["-ngl", "99", "-sm", "tensor", "-ts", "1,1,1,1", "--jinja", "-fa", "on",
             "--cache-type-k", kv, "--cache-type-v", kv,
             "-b", str(BATCH_SIZE), "-ub", str(args.ubatch_size), "--prompt-step", str(args.prompt_step), "--prompt-chunk", "128"]
    if args.prompt_step_ms is not None:
        argv += ["--prompt-step-ms", str(args.prompt_step_ms)]
    if args.attn_mask_bits:
        argv += ["--attn-mask-bits"]
    if args.swa_kv_cache:
        argv += ["--cache-type-k-swa", args.swa_kv_cache, "--cache-type-v-swa", args.swa_kv_cache]
    # The model card's sampling defaults; reasoning effort is left to the template's default of high
    argv += ["--temp", "1.0", "--top-p", "1.0", "--top-k", "0", "--min-p", "0.0",
             "--alias", Path(profile["pack"]).stem, "--chat-template-file", str(TEMPLATE),
             "-c", str(context), "-np", str(slots), "--kv-unified",
             "--cache-ram", str(args.cache_ram), "--no-cache-idle-slots",
             "--metrics", "--host", args.host, "--port", str(args.port), "--log-timestamps"]
    if args.ctx_checkpoints is not None:
        argv += ["--ctx-checkpoints", str(args.ctx_checkpoints)]
    if args.log_verbosity is not None:
        argv += ["--log-verbosity", str(args.log_verbosity)]
    if args.checkpoint_every is not None:
        argv += ["--checkpoint-every", str(args.checkpoint_every)]
    env = {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        # A session ends at slot-context tokens, and the graph reserved at load is sized for sessions that long
        "LLAMA_KV_SEQ_CELLS": str(slot_context),
        "LD_LIBRARY_PATH": str(server.parent),
    }
    if args.slot_save_path:
        argv += ["--slot-save-path", str(args.slot_save_path.resolve())]
    if args.prefix_copy:
        env["LLAMA_SLOT_PREFIX_COPY"] = "1"
    if args.cache_ram != 0 and args.save_async:
        env["LLAMA_STATE_SAVE_ASYNC"] = "1"
    if args.cache_ram != 0 and args.restore_async:
        env["LLAMA_STATE_RESTORE_ASYNC"] = "1"
    if args.bucket_prefill_kv:
        env["LLAMA_INKLING2_BUCKET_PREFILL_KV"] = "1"
    if args.stable_prefill:
        env["GGML_CUDA_FATTN_REL_PREFILL_F32"] = "1"
    if args.sort_expert_rows:
        env["GGML_CUDA_MMVQ_EXPERT_SORT_23"] = "1"
    if args.mean_mask_decode:
        env["LLAMA_INKLING2_MEAN_MASK_DECODE"] = "1"
    if args.reference_sinks:
        env["LLAMA_INKLING2_SHIFT_SINKS"] = "1"
    if args.keep_graphs is not None:
        env.update({"LLAMA_GRAPH_SLOTS": str(args.keep_graphs), "GGML_META_CAPTURE_GRAPHS": str(args.keep_graphs + 4),
                    "LLAMA_RS_FIXED_ROWS": "1", "LLAMA_GRAPH_ANY_SEQS": "1",
                    "GGML_CUDA_GRAPH_NODES": "262144", "GGML_CUDA_GRAPH_IDLE_MS": "600000"})
    if args.expert_row_limit is not None:
        env["GGML_CUDA_MMVQ_KQUANT_MMID_MAX"] = str(args.expert_row_limit)
    if args.reduction_route in ("host", "host-sliced"):
        env.update({"GGML_CUDA_ALLREDUCE": "internal", "GGML_CUDA_AR_LL_MAX_BYTES": "0", "GGML_CUDA_AR_P2P_MAX_BYTES": "0",
                    "GGML_CUDA_AR_COPY_THRESHOLD": "1", "GGML_CUDA_AR_COPY_PEER": "0",
                    "GGML_CUDA_AR_GATHER_OWNER_SLICES": "0"})
    if args.host_owner_rows:
        if args.reduction_route == "engine" or args.reduction_wire != "bf16":
            raise ValueError("host-owner-rows requires a BF16 host or host-sliced route")
        env["GGML_CUDA_AR_HOST_OWNER_ROWS"] = "1"
    if args.nccl_sliced_gather:
        if args.reduction_route != "engine":
            raise ValueError("nccl-sliced-gather requires the engine route")
        env["GGML_CUDA_NCCL_GATHER_SLICES"] = "1"
        env["LLAMA_INKLING2_GATHER_TRANSPOSE"] = "1"
    if args.reduction_route == "host-sliced":
        env["GGML_CUDA_AR_GATHER_HOST_SLICES"] = "1"
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
