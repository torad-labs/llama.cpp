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
    parser.add_argument("--kept-graphs", type=int, help="tensor-4: graphs a context keeps by shape, 1..256 (default: 128, "
                        "and 32 with --slot-context)")
    parser.add_argument("--restore-async", action="store_true",
                        help="A session brought back from the RAM prompt cache copies its rows on a thread of its own while "
                             "the other sessions decode (opt-in: not read on four cards)")
    parser.add_argument("--save-async", action="store_true",
                        help="A session whose slot another session takes is copied to the RAM prompt cache on a thread of "
                             "its own while the other sessions decode (opt-in: not read on four cards)")
    parser.add_argument("--sessions-dir", type=Path,
                        help="A directory the server leaves its sessions in when it is stopped with SIGTERM or SIGINT, and "
                             "takes them from when it starts with the same models: /dev/shm/... keeps them in memory. The "
                             "files hold the sessions' tokens (opt-in: not read on four cards)")
    parser.add_argument("--checkpoint-rows", type=int,
                        help="Spare state rows on the cards, 1..64: a session's context checkpoint is copied to one of them "
                             "on the cards, and from there to RAM on a thread of its own while the sessions decode "
                             "(opt-in: not read on four cards)")
    parser.add_argument("--fixed-rows", nargs="?", const="fused", choices=["fused", "unfused"],
                        help="Tensor profiles: a session's recurrent state keeps one row on the cards for the session's "
                             "whole life, so no step moves another session's state there and a copied session has a row "
                             "of its own at once. The conv-state update on those rows stays one kernel; 'unfused' "
                             "leaves it node by node, the rows alone (opt-in: four cards read the rows for isolation; "
                             "their cost was read on one card, 1 to 2%% of a decode fused and 4 to 6%% unfused)")
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
    parser.add_argument("--reduction-wire", choices=["q8", "bf16"],
                        help="tensor-4: what a card's share of a reduction crosses to the others as (default: q8, 8-bit "
                             "blocks; bf16 is the engine's own)")
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
    if args.kept_graphs is not None and (not four or not 1 <= args.kept_graphs <= 256):
        raise ValueError("kept-graphs is supported only with --profile tensor-4 and must be 1..256")
    if dp and args.fixed_rows:
        raise ValueError("fixed-rows is for the tensor profiles: under dp-4 every session has a row of its own already")
    if dp and args.mtp_depth in ("3", "4"):
        raise ValueError("dp-4 is measured at MTP depth 1 and 2")
    if not 0 <= args.mtp_adaptive <= 1:
        raise ValueError("mtp-adaptive is a share of a round, 0..1")
    if (args.dflash or args.backend_sampling or args.reduction_wire) and not four:
        raise ValueError("dflash, backend-sampling and reduction-wire are supported only with --profile tensor-4")
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
    if four:
        # Prompts are read in whole chunks, in an eval of their own after the step's replies: one forward pass of the
        # model, where prompt tokens shared out beside the replies are two or more. Fixture, 23 sessions of Claude
        # Code's sizes (Nov 11, 2026): every prompt-carrying eval is one pass, against 2.3 passes an eval with equal
        # shares and no chunk. The runs took 80.2 and 84.2 s on a card that was full and evicting graphs in both, so
        # the times do not isolate the chunk; four cards have not read it.
        argv += ["--prompt-chunk", "128"]
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
        env.update({"GGML_CUDA_MMQ_MOE_NCOLS": "0", "GGML_CUDA_MMQ_L2_PF": "1"})
    if four:
        # The cards' reductions go through the copy engines and host memory. On both four-card hosts read cleanly
        # (Nov 11, 2026) the engine's own routes, whose kernels store into a peer's memory, ran the server at 15.8 and
        # 13.4 to 14.7 tok/s a session where this route ran it at 33.0 and 42.8. On a third (Nov 10) a reduction took
        # 105 us on the engine's route and 109 on this one, in a benchmark whose cards were not shown to be free.
        env.update({"GGML_CUDA_AR_LL_MAX_BYTES": "0", "GGML_CUDA_AR_P2P_MAX_BYTES": "0", "GGML_CUDA_AR_COPY_THRESHOLD": "1",
                    "GGML_CUDA_AR_COPY_PEER": "0", "GGML_CUDA_AR_GATHER_OWNER_SLICES": "0"})
        # A share crosses in 8-bit blocks, 36 bytes to 32 elements where the BF16 wire takes 64: a 69-row target decode
        # is 40.2 ms against 42.5, a session's fourth pass after a load 45.7 tok/s against 41.5. Against an F32 wire's
        # logits on the pack (two chunks at a context of 8,192, 8,190 scored positions) the mean KLD is 0.0141 +- 0.0005
        # where the BF16 wire's is 0.0145 +- 0.0006: alike on that sample, which is all that was scored. A block that
        # holds a NaN or an infinity crosses as NaN in all 32 elements, where the BF16 wire carries the one.
        if args.reduction_wire != "bf16":
            env["GGML_CUDA_AR_WIRE_Q8"] = "1"
        # A graph is kept by its shape whichever sessions are in it, 128 a context, and a CUDA executable is not given
        # back while the server idles between turns: the steps sessions end in are built once, not every pass. A card
        # that fills gives the graphs used least recently back (LLAMA_GRAPH_DEVICE_RESERVE_MIB).
        # With --slot-context a kept graph is sized for sessions that long and 128 of them fill the cards: a pool of
        # 3,014,656 with sessions of 262,144 is 67.9 GB a card at load, and 23 sessions of 20,000 tokens took the cards
        # to 97.1 of 97.9 GB with both contexts at their reserve. With 32 they stood at 84.1 GB through the same
        # sessions, one of 250,000 tokens and four of 100,000 beside 19 short ones, and replies ran alike (four RTX PRO
        # 6000, Nov 11, 2026). What 32 costs a short-context read against 128 was not read.
        kept = args.kept_graphs if args.kept_graphs is not None else 32 if args.slot_context is not None else 128
        env.update({"LLAMA_GRAPH_ANY_SEQS": "1", "LLAMA_GRAPH_SLOTS": str(kept), "GGML_META_CAPTURE_GRAPHS": str(kept + 4),
                    "GGML_CUDA_GRAPH_NODES": "262144", "GGML_CUDA_GRAPH_IDLE_MS": "600000"})
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
    if args.restore_async:
        # A restore stops the server's thread only to place the cells; the rows are copied beside the other sessions' steps
        env["LLAMA_STATE_RESTORE_ASYNC"] = "1"
    if args.save_async:
        # A save for a slot that changes hands stops the server's thread only to write the state's layout; the request
        # that takes the slot waits its turn while the rows are copied beside the other sessions' steps
        env["LLAMA_STATE_SAVE_ASYNC"] = "1"
    if args.sessions_dir is not None:
        # At a stop every session is saved to the RAM prompt cache and the cache written to files; the next start maps
        # them, and a session comes back from its snapshot where it would be prefilled from its first token
        env["LLAMA_PROMPT_CACHE_DIR"] = str(args.sessions_dir)
    if args.checkpoint_rows is not None:
        # A context checkpoint stops the server's thread for the copy of one state row on the cards; the copy to RAM is
        # made beside the sessions' steps, and a checkpoint that finds every spare row busy is copied as before
        env["LLAMA_RS_STAGE_ROWS"] = str(args.checkpoint_rows)
        if "GGML_META_CAPTURE_GRAPHS" in env:
            # Each spare row has a graph of its own, which the tensor split keeps captured beside the decodes' graphs:
            # without room for it, a checkpoint would put a decode's graph out and that decode would capture again
            env["GGML_META_CAPTURE_GRAPHS"] = str(int(env["GGML_META_CAPTURE_GRAPHS"]) + args.checkpoint_rows)
    if args.fixed_rows:
        # A tensor-split context is one sequence device, whose recurrent memory gathers a step's sessions into a run of
        # rows and moves a state that is elsewhere before the decode. With this each session's row is its own from its
        # first token to its last and nothing is moved or shared. Four RTX PRO 6000, ten sessions whose first prompts
        # are 67,163 to 114,625 tokens taking each other's five slots (Nov 13, 2026): 0 turns of 60 lost a word of
        # their own or held another session's, in two runs, as without it.
        env["LLAMA_RS_FIXED_ROWS"] = "1"
        if args.fixed_rows != "unfused":
            # The fixed rows store a layer's conv-state snapshots by index, a form the fused conv-state update turned
            # away, so the chain ran node by node. With this the one kernel takes it. One RTX 5080, a 69-row decode
            # (Nov 13, 2026), in ms without the rows, with them, and with them and this: 15.5, 16.1 to 16.2 and 16.0
            # with two ranks on the card; 13.5, 14.4 and 13.7 to 13.8 with one. Four cards were not read for the cost.
            # An inherited GGML_CUDA_SSM_CONV_ROWS never reaches the server (removed_overrides), so the way to keep
            # the rows without the kernel is --fixed-rows unfused
            env["GGML_CUDA_SSM_CONV_ROWS"] = "1"
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
