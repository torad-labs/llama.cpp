#!/usr/bin/env python3
"""Serve a documented GLM-5.3 Flash recipe; --dry-run prints JSON without starting a model."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

from capacity import SLOT_CELLS, MLA_LAYERS, LATENT_ROW_BYTES, MIB

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "models/templates/GLM-5.3-Flash.jinja"
DRAFT_VOCAB = ROOT / "scripts/glm53/assets/mtp-draft-vocab-98304.i32"
DEFAULT_MTP_DEPTH = "2"
DP_SLOT_CONTEXT = 442368
# tensor-4's sessions and shared pool: the pool measured to load beside 23 slots (48.6 GiB used a card). The
# 6,029,312 tokens that 23 sessions of 262144 need fail in the graph reserve (docs/torad/glm53-goal.md, cause 3)
POOL_SLOTS = 23
POOL_CONTEXT = 524288
# the engine's sequence limit (LLAMA_MAX_SEQ)
MAX_SLOTS = 256
# Inherited engine, profile and diagnostic overrides are removed before the exec; the server's own access controls stay.
ENGINE_OVERRIDES = ("GGML_", "LLAMA_", "CUDA_", "NVIDIA_")
SERVER_ACCESS_CONTROLS = frozenset({"LLAMA_API_KEY", "LLAMA_ARG_API_KEY_FILE",
                                    "LLAMA_ARG_SSL_KEY_FILE", "LLAMA_ARG_SSL_CERT_FILE"})
ASSET_SHA256 = {
    TEMPLATE: "bb443c2772691a62b661da6515637c3ff70b2a790bf939b30220e337a6d9e723",
    DRAFT_VOCAB: "8e83de6161b0f75c212f7079b00e10af2c03701ae17afe62ee48bca35717e3e6",
}


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Local first GGUF shard; other shards must be alongside it")
    parser.add_argument("--mmproj", type=Path, help="Local vision projector; omit for text-only serving")
    parser.add_argument("--server", type=Path, default=ROOT / "build/bin/llama-server",
                        help="llama-server from this fork, with matching libraries beside it")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8102)
    parser.add_argument("--profile", choices=["tensor-2", "tensor-4", "dp-4"], default="tensor-2",
                        help="Two-card tensor recipe, four-card tensor recipe with one shared pool, or experimental "
                             "four-card DP capacity recipe")
    parser.add_argument("--transport", choices=["internal", "internal-nccl-gather"], help="DP collective transport (default: internal)")
    parser.add_argument("--slots", type=int, help="tensor-4: sessions sharing the pool, 1..256 (default: 23). "
                        "DP private caches of 442368 cells each: 1..40 resident, 1..256 host KV (default: 36)")
    parser.add_argument("--context", type=int, help="tensor-4: tokens in the shared pool (default: 524288)")
    parser.add_argument("--slot-context", type=int, help="tensor-4: tokens one session holds of the pool at most. The graph "
                        "reserved at load is then sized for sessions of that many, so a larger --context loads "
                        "(default: a session may take the whole pool)")
    parser.add_argument("--kv-host", action="store_true", help="Experimental DP/MTP-2: keep latent KV in pinned RAM and stage selected rows on the GPU")
    parser.add_argument("--host-budget-mib", type=float, help="RAM available for host KV after model, prompt-cache and system allocations")
    parser.add_argument("--mtp-depth", choices=["1", "2", "3", "4", "off"], default=DEFAULT_MTP_DEPTH,
                        help="Tokens the pack's own MTP head drafts a step; 3 and 4 on the tensor recipes")
    parser.add_argument("--mtp-adaptive", type=float, default=0.18,
                        help="The engine's --spec-draft-mtp-adaptive cost: a draft token's share of a one-token round, "
                             "measured at 0.18 for one session on two cards; 0 drafts --mtp-depth every step")
    parser.add_argument("--dflash", type=Path, help="tensor-4: a local DFlash draft GGUF that drafts in place of the MTP head, on the first card; "
                             "one that carries its own token embeddings and output head (dflash_own_head.py)")
    parser.add_argument("--dflash-depth", type=int, help="Tokens the DFlash draft drafts a step: its block size less the anchor row (required with --dflash)")
    parser.add_argument("--backend-sampling", action="store_true", help="tensor-4: sample on the cards (dp-4 always does)")
    parser.add_argument("--indexer-query", choices=["q8", "f32"], default="q8", help="Q8 is lossy; compare F32 for quality-sensitive workloads")
    parser.add_argument("--diagnostic-ubatch", type=int, choices=[128, 512],
                        help="Explicit numerical-control microbatch for DP; it never changes the serving default")
    parser.add_argument("--prompt-step", type=int, help="Prompt tokens per step alongside generation; default: 1024 tensor, 512 DP; 0 removes the cap")
    parser.add_argument("--trace-decode", action="store_true",
                        help="Opt-in host phase/batch markers, graph identity and CUDA NVTX; not an unprofiled speed run")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def resolve_transport(args, legacy=None):
    if legacy is not None and legacy not in ("internal", "internal-nccl-gather"):
        raise ValueError("unknown transport")
    if args.transport is not None and legacy is not None and args.transport != legacy:
        raise ValueError("conflicting transport choices")
    return args.transport or legacy or "internal"


def make_plan(args):
    dp = args.profile == "dp-4"
    four = args.profile == "tensor-4"
    if not dp and args.transport is not None:
        raise ValueError("transport is supported only with --profile dp-4")
    if args.diagnostic_ubatch is not None and not dp:
        raise ValueError("diagnostic-ubatch is supported only with --profile dp-4")
    diagnostic_ubatch = args.diagnostic_ubatch if args.diagnostic_ubatch is not None else 512
    prompt_step = args.prompt_step if args.prompt_step is not None else (512 if dp else 1024)
    if not 1 <= args.port <= 65535 or prompt_step < 0:
        raise ValueError("port must be 1..65535 and prompt-step must be nonnegative")
    limit = MAX_SLOTS if args.kv_host or four else 40
    if args.slots is not None and (not (dp or four) or not 1 <= args.slots <= limit):
        raise ValueError(f"slots is supported only with --profile tensor-4 or dp-4 and must be 1..{limit}")
    slots = (36 if dp else POOL_SLOTS) if args.slots is None else args.slots
    if not (dp or four):
        slots = 3
    context = POOL_CONTEXT if args.context is None else args.context
    if args.context is not None and (not four or context < slots):
        raise ValueError("context is supported only with --profile tensor-4 and holds at least one token a slot")
    if args.slot_context is not None and (not four or not 256 <= args.slot_context < context):
        raise ValueError("slot-context is supported only with --profile tensor-4, from 256 tokens to under the context")
    if dp and args.mtp_depth in ("3", "4"):
        raise ValueError("dp-4 is measured at MTP depth 1 and 2")
    if not 0 <= args.mtp_adaptive <= 1:
        raise ValueError("mtp-adaptive is a share of a round, 0..1")
    if (args.dflash or args.backend_sampling) and not four:
        raise ValueError("dflash and backend-sampling are supported only with --profile tensor-4")
    if (args.dflash is None) != (args.dflash_depth is None) or args.dflash_depth is not None and args.dflash_depth < 1:
        raise ValueError("dflash and a positive dflash-depth are given together")
    if args.kv_host:
        if not dp or args.mtp_depth != "2":
            raise ValueError("host KV requires --profile dp-4 and --mtp-depth 2")
        if args.host_budget_mib is None or not math.isfinite(args.host_budget_mib) or args.host_budget_mib < host_kv_mib(slots):
            raise ValueError(f"host KV needs a RAM budget of at least {host_kv_mib(slots):.0f} MiB")
    elif args.host_budget_mib is not None:
        raise ValueError("host-budget-mib requires --kv-host")
    if not args.host or any(char in args.host for char in "\n\r\0"):
        raise ValueError("host must be a nonempty address without control characters")
    server = args.server.expanduser().resolve()
    # The loader finds shards 2 and 3 from the first shard's own name, so a symlinked shard keeps its path (a Hugging
    # Face cache snapshot links it to a blob whose name carries no split suffix): absolute, the leaf link not followed.
    argv = [str(server), "-m", own_path(args.model)]
    if args.mmproj:
        argv += ["--mmproj", own_path(args.mmproj)]
    # tensor-4's latent cache is 4-bit: 288 bytes a row against 544; the indexer's keys stay q8_0, the one other
    # type the engine takes for them
    latent = "q4_0" if four else "q8_0"
    argv += ["-ngl", "99", "-sm", "dp" if dp else "tensor", "-ts", "1,1,1,1" if dp or four else "1,1",
             "--jinja", "-fa", "on",
             "--cache-type-k", latent, "--cache-type-v", latent, "-ctki", "q8_0",
             "-b", "2048", "-ub", str(diagnostic_ubatch) if dp else "2048", "--prompt-step", str(prompt_step),
             "--alias", "GLM-5.3-Flash-Uncensored-IQ3_XXS", "--chat-template-file", str(TEMPLATE)]
    if dp:
        argv += ["-fit", "off", "-cts", "q8_0", "--backend-sampling", "--prompt-chunk", "128"]
    elif args.backend_sampling:
        argv += ["--backend-sampling"]
    if args.dflash:
        # the draft is one small model reading the target's layer inputs: whole on the first card, never split four ways
        argv += ["--spec-type", "draft-dflash", "--spec-draft-model", own_path(args.dflash),
                 "--spec-draft-n-max", str(args.dflash_depth), "--spec-draft-device", "CUDA0", "--spec-draft-ngl", "99"]
    elif args.mtp_depth != "off":
        argv += ["--spec-type", "draft-mtp", "--spec-draft-n-max", args.mtp_depth,
                 "--spec-draft-mtp-vocab", str(DRAFT_VOCAB), "--spec-draft-mtp-adaptive", f"{args.mtp_adaptive:g}"]
        if dp:
            argv += ["-ubd", str(diagnostic_ubatch), "-ctkd", "q8_0", "-ctvd", "q8_0"]
    argv += ["-c", str(slots * DP_SLOT_CONTEXT if dp else context if four else 524288), "-np", str(slots)]
    if not dp:
        argv += ["--kv-unified"]
    argv += ["--cache-ram", "32768", "--no-cache-idle-slots"]
    argv += ["--metrics", "--host", args.host, "--port", str(args.port), "--log-timestamps"]
    env = {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "0,1,2,3" if dp or four else "0,1",
        "GGML_CUDA_GRAPH_MAX": "0", "GGML_CUDA_ALLREDUCE": args.transport or "internal",
        "LLAMA_INDEXER_Q8_QUERY": "1" if args.indexer_query == "q8" else "0",
        "LLAMA_MTP_DEVICE_H": "1", "LLAMA_TOK_EMBD_GPU": "1",
        "GGML_SCHED_DEBUG": "0",
        "LD_LIBRARY_PATH": str(server.parent),
    }
    if dp:
        # Keep infrequent shapes warm with a finite node budget; qualification still enforces 4096 MiB free per GPU.
        env.update({"GGML_CUDA_GRAPH_MEM_STATS": "0", "GGML_CUDA_GRAPH_NODES": "32768",
                    "GGML_CUDA_GRAPH_IDLE_MS": "0"})
    if not dp:
        env.update({"LLAMA_KPOOL_DIRECT_INDICES": "0", "GGML_CUDA_MMQ_MOE_NCOLS": "0", "GGML_CUDA_MMQ_L2_PF": "1"})
    if four and not args.dflash and args.mtp_depth != "off":
        # The MTP head proposes with the session's own sampler and the noise the target will draw with. A reply's
        # tokens are samples of the same distribution, though a seed's reply is another than without it. Four RTX PRO
        # 6000, 23 sessions (Nov 11, 2026): 2.17 to 2.21 tokens a step against 2.02 to 2.07.
        env["LLAMA_SPEC_COUPLED"] = "1"
    if args.slot_context is not None:
        # A session ends at slot-context tokens, and the load-time reserve holds a pooled indexer's scores for sessions
        # of that many, not for one that took every cell: 10 bytes a pool a row, the term that kept the pool for 23
        # sessions of 262144 from loading (docs/torad/glm53-goal.md, cause 3). The engine rounds it up to 256
        env["LLAMA_KV_SEQ_CELLS"] = str(args.slot_context)
    if args.kv_host:
        env["LLAMA_KV_HOST"] = "1"
    if args.trace_decode:
        argv += ["-lv", "4"]
        env.update({"LLAMA_DECODE_TRACE": "1", "GGML_CUDA_NVTX": "1",
                    "GGML_GRAPH_TRACE": "1", "LLAMA_TIME_DECODE": "1"})
    return argv, env


def host_kv_mib(slots):
    return slots * SLOT_CELLS * MLA_LAYERS * LATENT_ROW_BYTES / MIB


def available_memory_mib():
    available = next(int(line.split()[1]) / 1024 for line in Path('/proc/meminfo').read_text().splitlines()
                     if line.startswith('MemAvailable:'))
    # Vast containers may have a smaller cgroup limit than the machine's RAM.
    limit = Path('/sys/fs/cgroup/memory.max')
    if limit.is_file():
        if limit.read_text().strip() != 'max':
            available = min(available, (int(limit.read_text()) - int(limit.with_name('memory.current').read_text())) / MIB)
    else:
        limit_v1 = Path('/sys/fs/cgroup/memory/memory.limit_in_bytes')
        if limit_v1.is_file():
            available = min(available, (int(limit_v1.read_text()) -
                                       int(limit_v1.with_name('memory.usage_in_bytes').read_text())) / MIB)
    return available


def own_path(path):
    return os.path.abspath(path.expanduser())


def removed_overrides(inherited):
    """The inherited keys the exec removes: engine, profile and diagnostic overrides, never the server's access controls."""
    return sorted(key for key in inherited if key.startswith(ENGINE_OVERRIDES) and key not in SERVER_ACCESS_CONTROLS)


def execution_environment(settings, inherited=None):
    source = os.environ if inherited is None else inherited
    # Keep unrelated application settings and the server's key/TLS settings; remove implicit engine/profile/debug overrides.
    removed = set(removed_overrides(source))
    env = {key: value for key, value in source.items() if key not in removed}
    env.update(settings)
    return env


def resolve_server_arguments(argv, settings):
    if "--prompt-chunk" not in argv:
        return argv
    try:
        help_run = subprocess.run([argv[0], "--help"], env=execution_environment(settings),
                                  stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired as error:
        raise ValueError("server --help timed out: " + argv[0]) from error
    if help_run.returncode != 0:
        raise ValueError(f"server --help failed with exit {help_run.returncode}: {argv[0]}")
    options = help_run.stdout + "\n" + help_run.stderr
    if re.search(r"(?m)^[ \t]*(?:-\S+[ \t]+)*--prompt-chunk(?=[ \t=,]|$)", options):
        return argv
    index = argv.index("--prompt-chunk")
    print(f"{argv[0]} has no --prompt-chunk; keeping its original prompt scheduling", file=sys.stderr)
    return argv[:index] + argv[index + 2:]


def verify_assets():
    for path, expected in ASSET_SHA256.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("serving asset SHA256 mismatch: " + str(path))


def main(args):
    argv, settings = make_plan(args)
    verify_assets()
    if args.dry_run:
        if Path(argv[0]).is_file():
            argv = resolve_server_arguments(argv, settings)
        print(json.dumps({"argv": argv, "env": settings, "removed": removed_overrides(os.environ)}, indent=2))
        return 0
    for label, path in [("server", Path(argv[0])), ("model", args.model.expanduser()),
                        ("mmproj", args.mmproj.expanduser() if args.mmproj else None),
                        ("dflash draft", args.dflash.expanduser() if args.dflash else None)]:
        if path is not None and not path.is_file():
            raise ValueError("missing local " + label + ": " + str(path))
    if not os.access(argv[0], os.X_OK):
        raise ValueError("server is not executable: " + argv[0])
    if args.kv_host and available_memory_mib() < host_kv_mib(args.slots or 36) + 32768 + 8192:
        raise ValueError("insufficient available RAM for host KV, 32 GiB prompt cache and 8 GiB system reserve")
    argv = resolve_server_arguments(argv, settings)
    os.execve(argv[0], argv, execution_environment(settings))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(arguments()))
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2)
