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

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "models/templates/GLM-5.3-Flash.jinja"
DRAFT_VOCAB = ROOT / "scripts/glm53/assets/mtp-draft-vocab-98304.i32"
DEFAULT_MTP_DEPTH = "2"
# tensor-4's sessions and shared pool: the pool measured to load beside 23 slots (48.6 GiB used a card). The
# 6,029,312 tokens that 23 sessions of 262144 need fail in the graph reserve (docs/torad/glm53-goal.md, cause 3)
POOL_SLOTS = 23
POOL_CONTEXT = 524288
# tensor-2's goal is as many sessions as two cards hold with each able to reach 300K tokens. Its pool holds six
# sessions of 307,200 full at the same time and it reserves twelve slots, the shape the four-card deployment
# runs in (23 reserved, fewer decoding at once: decode cost follows the requests actually decoding, not the
# slots reserved). The counts are arithmetic, not a read on cards: the split's half of the weights is about
# 64 GiB a card, the mirrored caches 9,088 bytes a pool token (the four-card load of Nov 11, 2026), the two
# compute buffers about 6 GiB, and 4 GiB kept free. docs/torad/glm53-two-gpu.md has the sums and the box read
# that settles them.
TENSOR2_SLOTS = 12
TENSOR2_CONTEXT = 1843200
TENSOR2_SLOT_CONTEXT = 307200
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
    parser.add_argument("--profile", choices=["tensor-2", "tensor-4"], default="tensor-2",
                        help="Tensor recipe on two or four cards with one shared pool")
    parser.add_argument("--slots", type=int, help="Sessions sharing the pool, 1..256 (default: 12 on two cards, 23 on four)")
    parser.add_argument("--context", type=int, help="Tokens in the shared pool (default: 1843200 on two cards, 524288 on four)")
    parser.add_argument("--slot-context", type=int, help="Tokens one session holds of the pool at most. The graph "
                        "reserved at load is then sized for sessions of that many, so a larger --context loads "
                        "(default: 307200 on two cards when --context is not given; otherwise, and on four cards, a "
                        "session may take the whole pool)")
    parser.add_argument("--kept-graphs", type=int, help="Graphs a context keeps by shape, 1..256 (default: 128, "
                        "and 32 with --slot-context, so 32 on two cards)")
    parser.add_argument("--restore-async", action="store_true",
                        help="A session brought back from the RAM prompt cache copies its rows on a thread of its own while "
                             "the other sessions decode (the default; --restore-sync is its off switch)")
    parser.add_argument("--save-async", action="store_true",
                        help="A session whose slot another session takes is copied to the RAM prompt cache on a thread of "
                             "its own while the other sessions decode (the default; --save-sync is its off switch)")
    parser.add_argument("--restore-sync", action="store_true",
                        help="A restore copies its rows on the server's thread with every session waiting, as "
                             "before it was the default: the off switch, to compare the two on a box")
    parser.add_argument("--save-sync", action="store_true",
                        help="A save copies its rows on the server's thread with every session waiting, as "
                             "before it was the default: the off switch, to compare the two on a box")
    parser.add_argument("--state-copy-legacy", action="store_true",
                        help="A save to the RAM prompt cache and a restore from it copy a session's runs of cells one "
                             "by one, as before the batched copies: the off switch, to compare the two on a box")
    parser.add_argument("--cache-ram", type=int, default=32768, metavar="MIB",
                        help="RAM the prompt cache may hold, in MiB: a session whose slot another session took waits "
                             "there, and the oldest leave when it is full (default: 32768; -1 is no limit, 0 is no cache)")
    parser.add_argument("--ctx-checkpoints", type=int, metavar="N",
                        help="Context checkpoints a session keeps, the newest N: a turn whose prompt changed before its "
                             "end resumes from one, and a session saved to the prompt cache takes them all with it "
                             "(default: the engine's 32; 0 makes none)")
    parser.add_argument("--prefix-copy", action="store_true",
                        help="Tensor profiles: reuse a shared prefix from an idle or active session and preserve "
                             "edited branches in separate slots. A full bank chooses its victim separately from the "
                             "donor. Simultaneous starts can wait for one reader to reach their fork. KV cells are "
                             "shared; recurrent state is copied (opt-in: not read on four cards)")
    parser.add_argument("--shared-cells-legacy", action="store_true",
                        help="While two sessions hold the same cells (a prefix copy), every session's pool maps go the "
                             "packed way, as before the streams that share cells: the off switch, to compare the two on a box")
    parser.add_argument("--victim-lru-legacy", action="store_true",
                        help="With every slot in use, a new session takes the least recently used slot, whatever it holds, "
                             "as before the rule that weighs what a session holds against how long it has been idle: "
                             "the off switch, to compare the two on a box")
    parser.add_argument("--victim-idle-legacy", action="store_true",
                        help="With every slot in use, the session that is cut is the one with the fewest tokens for each "
                             "second it has been idle, as before the rule that weighs its tokens by the measured chance "
                             "of its return: the off switch, to compare the two on a box")
    parser.add_argument("--victim-return-s", type=float, metavar="SECONDS",
                        help="The seconds of idleness that halve the part of a session's chance of a return that fades "
                             "with its time away, in the rule for which session is cut (default: the engine's 90, the fit "
                             "to one deployment's records)")
    parser.add_argument("--sessions-dir", type=Path,
                        help="A directory the server leaves its sessions in when it is stopped with SIGTERM or SIGINT, and "
                             "takes them from when it starts with the same models: /dev/shm/... keeps them in memory. The "
                             "files hold the sessions' tokens (opt-in: not read on four cards)")
    parser.add_argument("--checkpoint-rows", type=int,
                        help="Spare state rows on the cards, 1..64: a session's context checkpoint is copied to one of them "
                             "on the cards, and from there to RAM on a thread of its own while the sessions decode "
                             "(opt-in: not read on four cards)")
    parser.add_argument("--prompt-shapes", type=int, choices=[1, 2, 4, 8, 16, 32], metavar="ROWS",
                        help="Tensor profiles: a prompt is read apart from the replies, in ubatches of a few row counts "
                             "whose graphs are kept: up to ROWS rows whole, then powers of two. 32 is the engine's "
                             "LLAMA_PROMPT_SHAPES=1 as it was; 8 leaves short prompts 8 row counts where 32 leaves 32 "
                             "(opt-in: not read on four cards)")
    parser.add_argument("--prompt-shape-rows", type=int, choices=[64, 128, 256, 512, 1024, 2048], metavar="ROWS",
                        help="With --prompt-shapes: the largest power of two a prompt's ubatch takes (default: 256)")
    parser.add_argument("--session-step", type=int, choices=[16384, 32768, 65536, 131072], metavar="TOKENS",
                        help="A session's graphs keep their shape while it grows by this many tokens, and "
                             "sessions inside one step share them (default: 256 tokens a slot, the next power of two: "
                             "8192 at 23 slots, 2048 at 6). The indexer scores the empty pools the wider step adds (opt-in: not "
                             "read on four cards)")
    parser.add_argument("--fixed-rows", nargs="?", const="fused", choices=["fused", "unfused"],
                        help="Tensor profiles: a session's recurrent state keeps one row on the cards for the session's "
                             "whole life, so no step moves another session's state there and a copied session has a row "
                             "of its own at once. The conv-state update on those rows stays one kernel; 'unfused' "
                             "leaves it node by node, the rows alone (opt-in: four cards read the rows for isolation; "
                             "their cost was read on one card, 1 to 2%% of a decode fused and 4 to 6%% unfused)")
    parser.add_argument("--mtp-depth", choices=["1", "2", "3", "4", "off"], default=DEFAULT_MTP_DEPTH,
                        help="Tokens the pack's own MTP head drafts a step")
    parser.add_argument("--mtp-depth-min", choices=["1", "2", "3", "4"],
                        help="The depth the adaptive choice falls back to: a session drafts this many tokens or "
                             "--mtp-depth, by what its drafts gain (default: 1). --mtp-depth 3 --mtp-depth-min 2 "
                             "chooses between 2 and 3")
    parser.add_argument("--mtp-adaptive", type=float, default=0.18,
                        help="The engine's --spec-draft-mtp-adaptive cost: a draft token's share of a one-token round, "
                             "measured at 0.18 for one session on two cards; 0 drafts --mtp-depth every step")
    parser.add_argument("--dflash", type=Path, help="A local DFlash draft GGUF that drafts in place of the MTP head, on the first card; "
                             "one that carries its own token embeddings and output head (dflash_own_head.py). The draft "
                             "sits whole on the first card beside its half of the target, so two cards hold fewer sessions with it")
    parser.add_argument("--dflash-depth", type=int, help="Tokens the DFlash draft drafts a step: its block size less the anchor row (required with --dflash)")
    parser.add_argument("--backend-sampling", action="store_true", help="Sample on the cards")
    parser.add_argument("--indexer-query", choices=["q8", "f32"], default="q8", help="Q8 is lossy; compare F32 for quality-sensitive workloads")
    parser.add_argument("--latent-cache", choices=["q4_0", "q8_0"],
                        help="The latent cache's type (default: q4_0, 288 bytes a row; q8_0 is 544 "
                             "and what the two-card recipe served with before it took the four-card settings)")
    parser.add_argument("--reduction-wire", choices=["q8", "bf16"],
                        help="What a card's share of a reduction crosses to the others as (default: q8, 8-bit "
                             "blocks; bf16 is the engine's own)")
    parser.add_argument("--reduction-route", choices=["host", "engine"],
                        help="host sends the cards' reductions on the copy engines through host memory; "
                             "engine leaves the routes the engine probes (default: host on four cards, where the "
                             "engine's routes were read three times slower on two hosts; engine on two, unread on "
                             "any pair: read it with test-backend-meta-allreduce-copy --bench 2 and choose)")
    parser.add_argument("--prompt-step", type=int, help="Prompt tokens per step alongside generation; default: 1024, and "
                        "with --prompt-shapes the largest prompt pass (256, or --prompt-shape-rows); 0 removes the cap")
    parser.add_argument("--prompt-step-ms", type=int, metavar="MS",
                        help="The time the prompt tokens of a step may take beside replies: the server sets their count "
                             "from the time its last steps took, 32 tokens or a power of two times that, up to "
                             "--prompt-step (then 1024 where it is not given, with "
                             "--prompt-shapes as well). A reply then waits about as long whatever the prompt's depth "
                             "(opt-in: not read on four cards)")
    parser.add_argument("--trace-decode", action="store_true",
                        help="Opt-in host phase/batch markers, graph identity and CUDA NVTX; not an unprofiled speed run")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def make_plan(args):
    devices = 4 if args.profile == "tensor-4" else 2
    # With prompt shapes a step's prompt tokens go in passes of at most --prompt-shape-rows rows, back to back, and the step's replies wait for all of them.
    # The step is then one pass, so a reply waits for one. Fixture, three sessions decoding beside a reader (Nov 14, 2026): 39.0 tok/s each at 256 against 34.5 at 1,024, the reader's prompts 4 % slower.
    shape_step = (args.prompt_shape_rows or 256) if args.prompt_shapes is not None else 1024
    # With --prompt-step-ms the step is the most the server's own count comes to, so it is not held to one pass
    if args.prompt_step_ms is not None:
        shape_step = 1024
    prompt_step = args.prompt_step if args.prompt_step is not None else shape_step
    if not 1 <= args.port <= 65535 or prompt_step < 0:
        raise ValueError("port must be 1..65535 and prompt-step must be nonnegative")
    if args.prompt_step_ms is not None and args.prompt_step_ms < 1:
        raise ValueError("prompt-step-ms must be 1 or more")
    if args.slots is not None and not 1 <= args.slots <= MAX_SLOTS:
        raise ValueError(f"slots must be 1..{MAX_SLOTS} for --profile {args.profile}")
    slots = args.slots if args.slots is not None else default_slots(args.profile)
    slot_context = args.slot_context
    if devices == 2 and slot_context is None and args.context is None:
        slot_context = TENSOR2_SLOT_CONTEXT
    context = args.context
    if context is None:
        context = TENSOR2_CONTEXT if devices == 2 else POOL_CONTEXT
    if context < slots:
        raise ValueError("context holds at least one token a slot")
    if slot_context is not None and not 256 <= slot_context < context:
        raise ValueError("slot-context is from 256 tokens to under the context")
    if args.kept_graphs is not None and not 1 <= args.kept_graphs <= 256:
        raise ValueError("kept-graphs must be 1..256")
    if args.cache_ram < -1 or args.ctx_checkpoints is not None and args.ctx_checkpoints < 0:
        raise ValueError("cache-ram is a size in MiB (-1: no limit) and ctx-checkpoints a count")
    if not 0 <= args.mtp_adaptive <= 1:
        raise ValueError("mtp-adaptive is a share of a round, 0..1")
    if args.mtp_depth_min is not None and (args.mtp_depth == "off" or args.dflash or int(args.mtp_depth_min) > int(args.mtp_depth)):
        raise ValueError("mtp-depth-min is a depth of the MTP head, no deeper than --mtp-depth")
    if args.prompt_shape_rows is not None and args.prompt_shapes is None:
        raise ValueError("prompt-shape-rows is given with --prompt-shapes")
    if args.restore_sync and args.restore_async or args.save_sync and args.save_async:
        raise ValueError("a save or a restore is async or sync, not both")
    if args.victim_return_s is not None and not (math.isfinite(args.victim_return_s) and args.victim_return_s > 0):
        raise ValueError("victim-return-s is a positive number of seconds")
    if args.victim_return_s is not None and (args.victim_lru_legacy or args.victim_idle_legacy):
        raise ValueError("victim-return-s sets the rule that the legacy switches turn off")
    if (args.dflash is None) != (args.dflash_depth is None) or args.dflash_depth is not None and args.dflash_depth < 1:
        raise ValueError("dflash and a positive dflash-depth are given together")
    if not args.host or any(char in args.host for char in "\n\r\0"):
        raise ValueError("host must be a nonempty address without control characters")
    server = args.server.expanduser().resolve()
    # The loader finds shards 2 and 3 from the first shard's own name, so a symlinked shard keeps its path (a Hugging
    # Face cache snapshot links it to a blob whose name carries no split suffix): absolute, the leaf link not followed.
    argv = [str(server), "-m", own_path(args.model)]
    if args.mmproj:
        argv += ["--mmproj", own_path(args.mmproj)]
    # The latent cache is 4-bit: 288 bytes a row against 544; the indexer's keys stay q8_0, the one other
    # type the engine takes for them. --latent-cache q8_0 keeps the wider rows the two-card recipe
    # served with before it took the four-card settings.
    latent = args.latent_cache or "q4_0"
    argv += ["-ngl", "99", "-sm", "tensor", "-ts", "1,1,1,1" if devices == 4 else "1,1",
             "--jinja", "-fa", "on",
             "--cache-type-k", latent, "--cache-type-v", latent, "-ctki", "q8_0",
             "-b", "2048", "-ub", "2048", "--prompt-step", str(prompt_step),
             "--alias", "GLM-5.3-Flash-Uncensored-IQ3_XXS", "--chat-template-file", str(TEMPLATE)]
    if args.prompt_step_ms is not None:
        argv += ["--prompt-step-ms", str(args.prompt_step_ms)]
    if args.backend_sampling:
        argv += ["--backend-sampling"]
    # Prompts are read in whole chunks, in an eval of their own after the step's replies: one forward pass of the
    # model, where prompt tokens shared out beside the replies are two or more. Fixture, 23 sessions of Claude
    # Code's sizes (Nov 11, 2026): every prompt-carrying eval is one pass, against 2.3 passes an eval with equal
    # shares and no chunk. The runs took 80.2 and 84.2 s on a card that was full and evicting graphs in both, so
    # the times do not isolate the chunk; four cards have not read it.
    argv += ["--prompt-chunk", "128"]
    if args.dflash:
        # the draft is one small model reading the target's layer inputs: whole on the first card, never split across them
        argv += ["--spec-type", "draft-dflash", "--spec-draft-model", own_path(args.dflash),
                 "--spec-draft-n-max", str(args.dflash_depth), "--spec-draft-device", "CUDA0", "--spec-draft-ngl", "99"]
    elif args.mtp_depth != "off":
        argv += ["--spec-type", "draft-mtp", "--spec-draft-n-max", args.mtp_depth,
                 "--spec-draft-mtp-vocab", str(DRAFT_VOCAB), "--spec-draft-mtp-adaptive", f"{args.mtp_adaptive:g}"]
        if args.mtp_depth_min is not None:
            # The engine's adaptive choice is between two depths, max(1, --spec-draft-n-min) and --spec-draft-n-max
            argv += ["--spec-draft-n-min", args.mtp_depth_min]
    argv += ["-c", str(context), "-np", str(slots)]
    argv += ["--kv-unified"]
    argv += ["--cache-ram", str(args.cache_ram), "--no-cache-idle-slots"]
    if args.ctx_checkpoints is not None:
        argv += ["--ctx-checkpoints", str(args.ctx_checkpoints)]
    argv += ["--metrics", "--host", args.host, "--port", str(args.port), "--log-timestamps"]
    env = {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "0,1,2,3" if devices == 4 else "0,1",
        "GGML_CUDA_GRAPH_MAX": "0", "GGML_CUDA_ALLREDUCE": "internal",
        "LLAMA_INDEXER_Q8_QUERY": "1" if args.indexer_query == "q8" else "0",
        "LLAMA_MTP_DEVICE_H": "1", "LLAMA_TOK_EMBD_GPU": "1",
        "GGML_SCHED_DEBUG": "0",
        "LD_LIBRARY_PATH": str(server.parent),
    }
    env.update({"GGML_CUDA_MMQ_MOE_NCOLS": "0", "GGML_CUDA_MMQ_L2_PF": "1"})
    if (args.reduction_route or ("host" if devices == 4 else "engine")) == "host":
        # The cards' reductions go through the copy engines and host memory. On both four-card hosts read cleanly
        # (Nov 11, 2026) the engine's own routes, whose kernels store into a peer's memory, ran the server at 15.8 and
        # 13.4 to 14.7 tok/s a session where this route ran it at 33.0 and 42.8. On a third (Nov 10) a reduction took
        # 105 us on the engine's route and 109 on this one, in a benchmark whose cards were not shown to be free.
        # No pair of cards has been read either way, so two cards keep the engine's routes until a box says
        # otherwise (--reduction-route host forces this route there).
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
    # 6000, Nov 11, 2026). What 32 costs a short-context read against 128 was not read. Two cards run the limit
    # by default, so they keep 32.
    kept = args.kept_graphs if args.kept_graphs is not None else 32 if slot_context is not None else 128
    env.update({"LLAMA_GRAPH_ANY_SEQS": "1", "LLAMA_GRAPH_SLOTS": str(kept), "GGML_META_CAPTURE_GRAPHS": str(kept + 4),
                "GGML_CUDA_GRAPH_NODES": "262144", "GGML_CUDA_GRAPH_IDLE_MS": "600000"})
    if devices == 2:
        # Two cards carry half the weights each and run fuller: a context gives a kept graph's buffers back
        # while 1 GiB is still free (the engine's default is 256 MiB), so making a graph's executables, which
        # a device out of memory cannot recover from, keeps its room.
        env["LLAMA_GRAPH_DEVICE_RESERVE_MIB"] = "1024"
    if not args.dflash and args.mtp_depth != "off":
        # The MTP head proposes with the session's own sampler and the noise the target will draw with. A reply's
        # tokens are samples of the same distribution, though a seed's reply is another than without it. Four RTX PRO
        # 6000, 23 sessions (Nov 11, 2026): 2.17 to 2.21 tokens a step against 2.02 to 2.07.
        env["LLAMA_SPEC_COUPLED"] = "1"
    if slot_context is not None:
        # A session ends at slot-context tokens, and the load-time reserve holds a pooled indexer's scores for sessions
        # of that many, not for one that took every cell: 10 bytes a pool a row, the term that kept the pool for 23
        # sessions of 262144 from loading (docs/torad/glm53-goal.md, cause 3). The engine rounds it up to 256
        env["LLAMA_KV_SEQ_CELLS"] = str(slot_context)
    # The tensor profiles save and restore beside the sessions' steps unless told not to. Four RTX PRO 6000 with the
    # pack, 23 slots, engine 417923e (#238, Nov 14 and 15, 2026), 30 sessions on 23 slots, 90 turns, a quarter of the
    # requests dropped part-way: 87 saves held the server's thread 4.7 s in all and 64 ms at most where 97 blocking ones
    # held it 99.4 s and 7,360 ms, 19 restores 4.4 s and 287 ms where 64 held it 66.6 s and 3,658 ms (the arms' prompt
    # caches differ, 256 and 32 GiB), and a session streaming beside them paused 1.34 s at most where it paused 8.66 s.
    # 182 turns of that churn: no leak, no lost context. 4.3 h of real traffic: 790 streamed answers, none waiting over
    # 3 s for a chunk, no server error. Not read on four cards with the batched state copies that came after that engine
    # (--state-copy-legacy is their off switch).
    if args.restore_async or not args.restore_sync:
        # A restore stops the server's thread only to place the cells; the rows are copied beside the other sessions' steps
        env["LLAMA_STATE_RESTORE_ASYNC"] = "1"
    if args.save_async or not args.save_sync:
        # A save for a slot that changes hands stops the server's thread only to write the state's layout; the request
        # that takes the slot waits its turn while the rows are copied beside the other sessions' steps
        env["LLAMA_STATE_SAVE_ASYNC"] = "1"
    if args.state_copy_legacy:
        # A session's state is one tensor range for each run of cells that follow each other, a layer: by default a device
        # copies them as batches through a scratch it keeps, and with this each range is a copy of its own
        env["GGML_TENSOR_N_STAGE_LEGACY"] = "1"
    if args.prefix_copy:
        # The empty slot starts as a copy of the idle session's slot, and the new session cuts the copy at the end of
        # the prefix it shares: it reads what follows, and the idle session keeps everything
        env["LLAMA_SLOT_PREFIX_COPY"] = "1"
    if args.shared_cells_legacy:
        # A cell two sessions hold sends the whole pool back to the packed maps, for every session, while it is shared
        env["LLAMA_KPOOL_SHARED_CELLS_LEGACY"] = "1"
    if args.victim_lru_legacy:
        # By default the session that is cut when no slot is empty is the one whose tokens times its chance of a return
        # are the fewest (what it has to bring back, against how likely it is to return); with this, the least recently used
        env["LLAMA_SLOT_VICTIM_LRU_LEGACY"] = "1"
    if args.victim_idle_legacy:
        # The chance of a return taken as one over the seconds idle, as the rule first was, in place of the measured shares
        env["LLAMA_SLOT_VICTIM_IDLE_LEGACY"] = "1"
    if args.victim_return_s is not None:
        env["LLAMA_SLOT_VICTIM_RETURN_S"] = f"{args.victim_return_s:g}"
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
    if args.prompt_shapes is not None:
        # A prompt beside replies is cut at the replies' length and its rest is a ubatch of a row count seen once: both
        # graphs are built, planned and launched kernel by kernel where a kept one replays. With this the replies keep
        # the graph they make alone and a prompt goes in a few row counts
        env["LLAMA_PROMPT_SHAPES"] = "1"
        if args.prompt_shapes != 32:
            env["LLAMA_PROMPT_SHAPE_SMALL"] = str(args.prompt_shapes)
        if args.prompt_shape_rows is not None:
            env["LLAMA_PROMPT_SHAPE_ROWS"] = str(args.prompt_shape_rows)
    if args.session_step is not None:
        # A graph is sized on the pools of the longest session in it, padded to the shared pad: each session's graphs
        # change shape every 8,192 tokens it grows (23 slots), and sessions in different steps share none
        env["LLAMA_KV_PAD_SHARED"] = str(args.session_step)
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
    if args.trace_decode:
        argv += ["-lv", "4"]
        env.update({"LLAMA_DECODE_TRACE": "1", "GGML_CUDA_NVTX": "1",
                    "GGML_GRAPH_TRACE": "1", "LLAMA_TIME_DECODE": "1"})
    return argv, env


def default_slots(profile):
    return {"tensor-2": TENSOR2_SLOTS, "tensor-4": POOL_SLOTS}[profile]


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
    argv = resolve_server_arguments(argv, settings)
    os.execve(argv[0], argv, execution_environment(settings))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(arguments()))
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2)
