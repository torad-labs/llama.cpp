#!/usr/bin/env python3
"""Serve Inkling 2 Small on four cards from a local abliterated pack; --dry-run prints JSON without starting a model."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "models/templates/thinkingmachines-Inkling-2-Small.jinja"
TEMPLATE_SHA256 = "e9cfd8e75d1272633c2e72b3cc76e99567e2bc8912b960483aed483ad35d54c7"
# The settings search.py kept for each profile: launcher options and variables of tunables.toml
TUNED = Path(__file__).with_name("tuned.json")

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
DRAFT = f"{SOURCE}-DSpark-BF16-own-head.gguf"

SLOTS = 23
SLOT_CONTEXT = 262144
CODING_SLOT_CONTEXT = 294912  # Shared by prompt, reasoning, tool history and generated text.
BATCH_SIZE = 2048
DRAFT_CONTEXT = 1572864
DRAFT_UBATCH_SIZE = 256
MAX_SLOTS = 256  # the engine's sequence limit (LLAMA_MAX_SEQ)
# The one serving configuration for the ablated model on four RTX PRO 6000 cards. Every variant tested
# against it lost or was an experiment, and their settings are recorded here so nothing is re-derived:
#   stock decode order (all switches off): 289.6/290.7 tok/s in the 2026-11-21 six-arm window, 292.0 in the
#     2026-11-22 arms - the combined profile beat it by +45.7% (425.6 tok/s, p99 92.5 -> 65.1 ms).
#   query-first attention +26.8% and compact greedy +11.5% over the stock control, each alone
#     (reports/inkling2-tensor4-implementation.md); the shared few-row quantize +0.75% on the serving arm
#     (fast 425.61 vs fast-legacyq8 422.44, 2026-11-22 arms). --no-query-first-decode / --no-compact-greedy /
#     GGML_CUDA_MMVQ_FEW_SHARED_Q8_1_LEGACY=1 restore the stock behavior; the parity capture remains pending
#     (reports/inkling2-tensor4-capture).
#   --expert-row-limit 32: +1.9% in one 2026-11-19 trial, -0.3% in the 2026-11-22 arms - not adopted.
#   GGML_CUDA_AR_GATHER_BLOCKS 4: -1.4% (default 8 stands; 16 pending).
#   Former profiles, removed 2026-11-22 (restore from git history if a test needs them):
#     tensor-4-draft    context 1572864, the DSpark drafter's reduced-pool layout; draft measured -23% at 23 clients.
#     tensor-4-capacity {SOURCE}-IQ3_XXS-imatrix.gguf, q8_0 KV, context SLOTS*SLOT_CONTEXT - the capacity tier.
#     tensor-4-quality  {SOURCE}-Q4_K_M-imatrix.gguf, f16 KV, context 2*SLOT_CONTEXT - a two-session quality box.
PROFILES = {
    "tensor-4": {"pack": f"{SOURCE}-Q3_K_M-imatrix.gguf", "mmproj": "mmproj-F16.gguf",
                 "context": 3014656, "slot_context": CODING_SLOT_CONTEXT, "kv": "f16", "swa_kv": "q8_0",
                 "cache_ram": 262144, "keep_graphs": 32, "keep_prompt_graphs": True, "prompt_shapes": 32,
                 "checkpoint_every": 32768, "checkpoint_rows": 2, "checkpoint_ram": 16384,
                 "expert_row_limit": 23, "bucket_prefill_kv": True,
                 "stable_prefill": False, "sort_expert_rows": True,
                 "prefix_copy": True, "mean_mask_decode": True, "reduction_route": "host", "host_owner_rows": False,
                 "query_first_decode": True, "compact_greedy": True,
                 "kv_runs": True, "packed_prefill": True, "f32_few": True, "moe_fuse": True,
                 "mmq_tma": True, "l2_persist": True, "l2_expert_warm": True,
                 "router_bake": True, "moe_route_gate": True, "ar_copy_capture": True},
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
    parser.add_argument("--draft", action=argparse.BooleanOptionalAction,
                        help="Use the paired native DSpark draft (default off; the flag also applies the draft "
                             "layout: context 1572864, ubatch 256, q8_0 target KV)")
    parser.add_argument("--draft-layout", action=argparse.BooleanOptionalAction,
                        help="Apply the draft layout without a draft: the matched control of a draft A/B")
    parser.add_argument("--draft-model", type=Path, help="An explicit Inkling DSpark GGUF with its own vocabulary tensors")
    parser.add_argument("--draft-depth", type=int, choices=range(1, 8), default=3,
                        help="Maximum DSpark proposals per verification (default: 3)")
    parser.add_argument("--draft-sampling", choices=["probabilistic", "greedy"], default="probabilistic",
                        help="probabilistic samples the draft at the request's temperature and verifies by rejection; "
                             "greedy proposes the draft's first choice and verifies by matching (default: probabilistic)")
    parser.add_argument("--draft-device", choices=["CUDA0", "CUDA1", "CUDA2", "CUDA3"], default="CUDA3",
                        help="Device for the draft model (default: CUDA3)")
    parser.add_argument("--dspark-trim-block", action=argparse.BooleanOptionalAction, default=True,
                        help="Trim DSpark draft blocks to n_max + 1 rows when drafting (default: enabled)")
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
    parser.add_argument("--kv-cache", choices=["f16", "q8_0", "q4_0"],
                        help="KV cache type (default: the profile's; q4_0 serves only the dense-path decode today)")
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
    parser.add_argument("--prefix-bundle", type=Path,
                        help="Verified shared-prefix KV bundle directory (doc 19); pins checked against pack/template/KV before launch")
    parser.add_argument("--mean-mask-decode", action=argparse.BooleanOptionalAction,
                        help="Try grouped SM120 F16 global attention for 23 decoding sessions with low average KV visibility")
    parser.add_argument("--query-first-decode", action=argparse.BooleanOptionalAction,
                        help="Order the 23 low-mean decode queries before the KV partitions so one prefix tile serves them all "
                             "(tensor-4 default; bit-identical operator output, parity at the server level not yet captured)")
    parser.add_argument("--compact-greedy", action=argparse.BooleanOptionalAction,
                        help="Read back a short candidate list instead of the whole vocabulary for greedy rows "
                             "(tensor-4 default; neutral greedy chains with static exclusions only)")
    parser.add_argument("--reference-sinks", action="store_true",
                        help="Use the padded, shifted relative-position table for comparison with the original attention")
    parser.add_argument("--ubatch-size", type=int,
                        help=f"Maximum tokens in one physical microbatch (1..{BATCH_SIZE}, default: {BATCH_SIZE} or {DRAFT_UBATCH_SIZE} with a draft)")
    parser.add_argument("--keep-graphs", "--kept-graphs", type=int, metavar="N",
                        help="Keep up to N scheduler graphs (0..256; 0 disables, tensor-4 default: 32) and reuse matching "
                             "batch shapes across sessions with fixed convolution rows.")
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
    parser.add_argument("--checkpoint-rows", type=int, metavar="N",
                        help="Spare device staging windows for async SWA+recurrent checkpoints, 0..64 (tensor-4: 2, 0 disables)")
    parser.add_argument("--checkpoint-ram", type=int, metavar="MIB",
                        help="Live slot checkpoint host RAM cap in MiB (tensor-4: 16384, 0 removes cap)")
    parser.add_argument("--kv-runs", action=argparse.BooleanOptionalAction,
                        help="Enable run-addressed global decode and shared-prefix read-once (tensor-4 default: on)")
    parser.add_argument("--packed-prefill", action=argparse.BooleanOptionalAction,
                        help="Enable GQA-6x packed relative prefill (tensor-4 default: on)")
    parser.add_argument("--f32-few", action=argparse.BooleanOptionalAction,
                        help="Enable custom F32 few-column kernels for router and relative bias table (tensor-4 default: on)")
    parser.add_argument("--moe-fuse", action=argparse.BooleanOptionalAction,
                        help="Enable fused MoE router+SITU+weighted-sum decode chain (tensor-4 default: on)")
    parser.add_argument("--mmq-tma", action=argparse.BooleanOptionalAction,
                        help="Enable SM120 TMA tile MMQ kernel for K-quant projections (tensor-4 default: on)")
    parser.add_argument("--l2-persist", action=argparse.BooleanOptionalAction,
                        help="Pin hot Inkling 2 conv-state rings in persisting L2 on large-L2 cards (tensor-4 default: on)")
    parser.add_argument("--l2-expert-warm", action=argparse.BooleanOptionalAction,
                        help="Warm routed expert weight tiles into L2 after router selection (tensor-4 default: on)")
    parser.add_argument("--router-bake", action=argparse.BooleanOptionalAction,
                        help="Bake paired block signs into Inkling 2 F32 router weights (tensor-4 default: on)")
    parser.add_argument("--moe-route-gate", action=argparse.BooleanOptionalAction,
                        help="Count prompt routes once and skip empty experts on Blackwell MMQ tiles (tensor-4 default: on)")
    parser.add_argument("--ar-copy-capture", action=argparse.BooleanOptionalAction,
                        help="Enable copy-engine whole-step CUDA graph capture with device-side pacing on host route (tensor-4 default: on)")
    parser.add_argument("--keep-prompt-graphs", action=argparse.BooleanOptionalAction,
                        help="Retain repeated prompt-shaped graphs in shared buffers when keep-graphs > 0 (tensor-4 default: on)")
    parser.add_argument("--prompt-shapes", type=int, choices=[0, 1, 2, 4, 8, 16, 32], metavar="ROWS",
                        help="Read prompts apart from decode replies in reusable power-of-two ubatches (tensor-4: 32, 0 disables)")
    parser.add_argument("--prompt-shape-rows", type=int, choices=[64, 128, 256, 512, 1024, 2048], metavar="ROWS",
                        help="Largest power-of-two prompt ubatch with --prompt-shapes (default: 256)")
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
                             "read fastest on rented hosts. host-sliced stages only owned gather values. engine uses the routes it probes (tensor-4: host, otherwise host)")
    parser.add_argument("--tuned", type=Path, default=TUNED,
                        help="Settings a search kept, by profile; an option given here wins (default: tuned.json beside this file)")
    parser.add_argument("--dry-run", action="store_true")
    argv = sys.argv[1:] if argv is None else list(argv)
    args = parser.parse_args(argv)
    text = args.tuned.read_text() if args.tuned.exists() else ""
    tuned = json.loads(text).get(args.profile, {}) if text.strip() else {}
    # The search's options go first, so the same option on the command line replaces them
    args = parser.parse_args([word for name, value in tuned.items() if name.startswith("--")
                              for word in option_words(name, value)] + argv)
    args.tuned_env = {name: value for name, value in tuned.items() if not name.startswith("--")}
    profile = PROFILES[args.profile]
    for key, fallback in (("cache_ram", 32768), ("prefix_copy", False), ("mean_mask_decode", False), ("reduction_route", "host"),
                          ("checkpoint_every", None), ("checkpoint_rows", 0), ("checkpoint_ram", 0),
                          ("keep_graphs", 0), ("keep_prompt_graphs", False), ("prompt_shapes", None),
                          ("expert_row_limit", None), ("bucket_prefill_kv", False),
                          ("stable_prefill", False), ("sort_expert_rows", False), ("query_first_decode", False),
                          ("compact_greedy", False), ("kv_runs", False), ("packed_prefill", False),
                          ("f32_few", False), ("moe_fuse", False), ("mmq_tma", False),
                          ("l2_persist", False), ("l2_expert_warm", False),
                          ("router_bake", False), ("moe_route_gate", False), ("ar_copy_capture", False)):
        if getattr(args, key) is None:
            setattr(args, key, profile.get(key, fallback))
    if args.prompt_shapes == 0:
        args.prompt_shapes = None
    if args.expert_row_limit == "auto":
        args.expert_row_limit = None
    if args.host_owner_rows is None:
        args.host_owner_rows = (profile.get("host_owner_rows", False) and args.reduction_route == profile["reduction_route"]
                                and args.reduction_wire == "bf16")
    return args


def option_words(name, value):
    """A setting as launcher words; on and off are the two forms of a switch."""
    return [name] if value == "on" else ["--no-" + name[2:]] if value == "off" else [name, value]


def make_plan(args):
    profile = PROFILES[args.profile]
    if args.draft_model is not None and args.draft is False:
        raise ValueError("draft-model cannot be combined with --no-draft")
    draft = args.draft_model
    if draft is None and args.draft:
        if args.model is not None:
            raise ValueError("a draft for a different target requires --draft-model")
        draft = args.packs / DRAFT
    draft_layout = draft is not None or args.draft_layout
    kv = args.kv_cache or ("q8_0" if draft_layout else profile["kv"])
    swa_kv = args.swa_kv_cache if args.swa_kv_cache is not None else profile.get("swa_kv")
    slots = args.slots if args.slots is not None else SLOTS
    # Use the reduced draft pool whenever draft layout is active so CUDA3 keeps headroom for CUDA graphs alongside DSpark.
    default_context = DRAFT_CONTEXT if draft_layout else profile["context"]
    context = args.context if args.context is not None else default_context
    slot_context = args.slot_context if args.slot_context is not None else profile.get("slot_context", SLOT_CONTEXT)
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be 1..65535")
    if not 1 <= slots <= MAX_SLOTS:
        raise ValueError(f"slots must be 1..{MAX_SLOTS}")
    ubatch_size = args.ubatch_size
    if ubatch_size is None:
        ubatch_size = DRAFT_UBATCH_SIZE if draft_layout else BATCH_SIZE
    if not 1 <= ubatch_size <= BATCH_SIZE:
        raise ValueError(f"ubatch-size must be 1..{BATCH_SIZE}")
    if not 0 <= args.keep_graphs <= 256:
        raise ValueError("keep-graphs must be 0..256")
    if not 0 <= args.checkpoint_rows <= 64:
        raise ValueError("checkpoint-rows must be 0..64")
    if args.checkpoint_ram < 0 or args.checkpoint_ram > 1048576:
        raise ValueError("checkpoint-ram must be 0..1048576")
    if args.prompt_shape_rows is not None and args.prompt_shapes is None:
        raise ValueError("prompt-shape-rows requires --prompt-shapes")
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
    argv = [str(server), "-m", model]
    if not args.text_only:
        argv += ["--mmproj", own_path(args.mmproj) if args.mmproj else own_path(args.packs / profile.get("mmproj", MMPROJ))]
    # Attention splits by heads (2 KV heads a card), so the caches split with them and are never mirrored
    argv += ["-ngl", "99", "-sm", "tensor", "-ts", "1,1,1,1", "--jinja", "-fa", "on", "--poll", "0",
             "--cache-type-k", kv, "--cache-type-v", kv,
             "-b", str(BATCH_SIZE), "-ub", str(ubatch_size), "--prompt-step", str(args.prompt_step), "--prompt-chunk", "128"]
    if args.prompt_step_ms is not None:
        argv += ["--prompt-step-ms", str(args.prompt_step_ms)]
    if args.attn_mask_bits:
        argv += ["--attn-mask-bits"]
    if swa_kv:
        argv += ["--cache-type-k-swa", swa_kv, "--cache-type-v-swa", swa_kv]
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
    if draft is not None:
        argv += ["--spec-type", "draft-dspark", "--spec-draft-model", own_path(draft),
                 "--spec-draft-n-max", str(args.draft_depth), "--spec-draft-n-min", "1",
                 "--spec-draft-p-min", "0", "--spec-draft-sampling", args.draft_sampling,
                 "--spec-draft-device", args.draft_device, "--spec-draft-ngl", "99",
                 "--spec-draft-ubatch", str(DRAFT_UBATCH_SIZE),
                 "--spec-draft-type-k", "q8_0", "--spec-draft-type-v", "q8_0"]
    env = {
        **args.tuned_env,
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        # A session ends at slot-context tokens, and the graph reserved at load is sized for sessions that long
        "LLAMA_KV_SEQ_CELLS": str(slot_context),
        "LLAMA_TOK_EMBD_GPU": "1",
        "LD_LIBRARY_PATH": str(server.parent),
    }
    slot_save_path = args.slot_save_path
    if args.prefix_bundle is not None:
        import importlib.util
        pa_path = Path(__file__).resolve().parent / "prefix_artifact.py"
        spec = importlib.util.spec_from_file_location("inkling2_prefix_artifact", pa_path)
        assert spec is not None and spec.loader is not None
        prefix_artifact = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prefix_artifact)
        bundle_dir = args.prefix_bundle.expanduser().resolve()
        try:
            _, reasons = prefix_artifact.verify_bundle(
                bundle_dir,
                pack_name=Path(model).name,
                template_sha256=TEMPLATE_SHA256,
                kv_type=kv,
                swa_kv_type=swa_kv or kv,
                max_pool_cells=context,
            )
        except prefix_artifact.ArtifactError as exc:
            raise ValueError(f"prefix-bundle invalid: {exc}") from exc
        if reasons:
            raise ValueError("prefix-bundle pin mismatch: " + "; ".join(reasons))
        if slot_save_path is None:
            slot_save_path = bundle_dir
    if slot_save_path:
        argv += ["--slot-save-path", str(slot_save_path.resolve())]
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
    if args.query_first_decode:
        env["GGML_CUDA_FATTN_REL_QUERY_FIRST"] = "1"
    if args.compact_greedy:
        env["LLAMA_INKLING2_COMPACT_GREEDY"] = "1"
    if args.reference_sinks:
        env["LLAMA_INKLING2_SHIFT_SINKS"] = "1"
    if args.kv_runs:
        env["LLAMA_INKLING2_KV_RUNS"] = "1"
        env["GGML_CUDA_FATTN_REL_RUNS_SHARED"] = "1"
    if args.packed_prefill:
        env["GGML_CUDA_FATTN_REL_PACKED_PREFILL"] = "1"
    if args.f32_few:
        env["GGML_CUDA_F32_FEW"] = "1"
    if args.moe_fuse and args.f32_few:
        env["GGML_CUDA_MOE_FUSE"] = "1"
    if args.mmq_tma:
        env["GGML_CUDA_MMQ_TMA"] = "1"
    if args.l2_persist:
        env["GGML_CUDA_INKLING2_L2_PERSIST"] = "1"
    if args.l2_expert_warm:
        env["GGML_CUDA_L2_EXPERT_WARM"] = "1"
    if args.router_bake:
        env["LLAMA_INKLING2_ROUTER_BAKE"] = "1"
    if args.moe_route_gate:
        env["GGML_CUDA_MOE_ROUTE_GATE"] = "1"
    if draft is not None and args.dspark_trim_block:
        env["LLAMA_DSPARK_TRIM_BLOCK"] = "1"
    elif not args.dspark_trim_block:
        env.pop("LLAMA_DSPARK_TRIM_BLOCK", None)
    if kv == "q8_0" or swa_kv == "q8_0":
        env["GGML_CUDA_FATTN_REL_DECODE_ONE_BYTE"] = "1"
    if kv == "q4_0" or swa_kv == "q4_0":
        env["GGML_CUDA_FATTN_REL_DECODE_Q4"] = "1"
    if args.keep_graphs > 0:
        prompt_headroom = 12 if args.keep_prompt_graphs else 4
        env.update({"LLAMA_GRAPH_SLOTS": str(args.keep_graphs),
                    "GGML_META_CAPTURE_GRAPHS": str(args.keep_graphs + prompt_headroom + args.checkpoint_rows),
                    "LLAMA_RS_FIXED_ROWS": "1", "LLAMA_GRAPH_ANY_SEQS": "1",
                    "GGML_CUDA_GRAPH_NODES": "262144", "GGML_CUDA_GRAPH_IDLE_MS": "600000"})
        if args.keep_prompt_graphs:
            env["LLAMA_GRAPH_PROMPT_KEPT"] = "1"
    if args.checkpoint_rows > 0:
        env["LLAMA_RS_STAGE_ROWS"] = str(args.checkpoint_rows)
    if args.checkpoint_ram > 0:
        env["LLAMA_CHECKPOINT_RAM_MIB"] = str(args.checkpoint_ram)
    if args.prompt_shapes is not None:
        env["LLAMA_PROMPT_SHAPES"] = "1"
        if args.prompt_shapes != 32:
            env["LLAMA_PROMPT_SHAPE_SMALL"] = str(args.prompt_shapes)
        if args.prompt_shape_rows is not None:
            env["LLAMA_PROMPT_SHAPE_ROWS"] = str(args.prompt_shape_rows)
    if args.expert_row_limit is not None:
        env["GGML_CUDA_MMVQ_KQUANT_MMID_MAX"] = str(args.expert_row_limit)
    if args.reduction_route in ("host", "host-sliced"):
        env.update({"GGML_CUDA_ALLREDUCE": "internal", "GGML_CUDA_AR_LL_MAX_BYTES": "0", "GGML_CUDA_AR_P2P_MAX_BYTES": "0",
                    "GGML_CUDA_AR_COPY_THRESHOLD": "1", "GGML_CUDA_AR_COPY_PEER": "0"})
        if args.ar_copy_capture:
            env["GGML_CUDA_AR_COPY_ENGINE_CAPTURE"] = "1"
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
        env["GGML_CUDA_AR_GATHER_Q8"] = "1"
    return argv, env


def main(args):
    argv, settings = make_plan(args)
    glm53.verify_assets({TEMPLATE: TEMPLATE_SHA256})
    paths = [("model", Path(argv[argv.index("-m") + 1]))]
    if "--mmproj" in argv:
        paths.append(("mmproj", Path(argv[argv.index("--mmproj") + 1])))
    if "--spec-draft-model" in argv:
        paths.append(("DSpark draft", Path(argv[argv.index("--spec-draft-model") + 1])))
    return glm53.launch(argv, settings, paths, args.dry_run)


if __name__ == "__main__":
    try:
        raise SystemExit(main(arguments()))
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2)
