# GLM-5.3 Flash serving goal

Read this before starting any GLM-5.3 work, and check the work against it when it is done.

## The goal

GLM-5.3 Flash, the BoldingBuilds IQ3_XXS pack, served by this fork on four RTX PRO 6000 cards for 36-40 concurrent agent sessions:

- **50-70 decode tokens/s per session, with all of them decoding at once.**
- Context: one shared pool of about 4.4M tokens under the layout below, about 120K a session at 36 with any one session able to pass 400K. `dp-4` holds 15.9M private cells (442K a slot, its caches on one card each instead of mirrored on four) and stays in the tree for the day every session must hold 400K at once; no engine on these cards has served that at speed.

Per-session decode rate at N concurrent sessions is the number. Aggregate is N times it. A single-session rate, a kernel time or a trace share is not progress toward the goal until it moves that number.

## Where it stands (measured)

| What | Value | Source |
| --- | --- | --- |
| One session, `dp-4` | 45-73 tok/s | Nov 4 profile; Nov 5 audit (71.2 uninstrumented) |
| 2 / 4 / 8 / 40 sessions, `dp-4`, per session | 28 / 12-16 / 6 / 0.9 tok/s | Nov 3-4 (`glm53-dp.md`) |
| 36 sessions, `dp-4`, one pass a step, equal drafts | 148 ms a round (121 ms the 108-row target pass, 21 ms a result-queue lock since removed), about 13 tok/s a session at ~1.9 tokens a round | Nov 5 box: `9ca2776d0`, `6384cd2ac` commit messages |
| The fork's own model of an optimized `dp-4`, 40 sessions (not measured) | 30.6 ms a round: 98-119 tok/s a session if every draft lands, 65-79 with one of two, 47-57 with one of two and 70 % of rated memory bandwidth | `glm53-decode-pcie.md` |
| Same model and cards, another engine (TensorFold, EXL3 4-bit experts, FP8 dense, DFlash2) | 70 tok/s per request at 16 concurrent (prose), 86 (code); 40 slots, shared 5.1M-token pool | mia-ai.net, measured Oct 4 |

The measured round is about 5x the modeled one, before slot gaps. The goal sits where the model puts an engine running at 70-100 % of rated memory bandwidth with ordinary draft acceptance: the low end is within reach, the high end needs near-roofline kernels or more tokens landed a round.

## Known causes, each a number

1. **Passes per step.** Every ubatch is a full weight read. Drafts of different lengths (fixed on main: `9ca2776d0`, `861d22f10`, `e18d70a0b`) and idle slots between busy ones (open: `split_equal`'s sequential split and `split_cache_view` need each card's busy streams consecutive; 8 passes a step measured on CPU once replies end at different times) multiply the step.
2. **Time per pass.** 121 ms measured for the target pass against about 28 ms of modeled memory work. Nov 4 trace: collectives 74 % of summed kernel time, cards idle 46-51 % of the step, hundreds of graph instantiations per request. Collective dispatch picks its transport by message size (≤128 KB peer packets, 128 KB-1 MB host-staged chunks, ≥1 MB copy path); a 36-session step's reductions sit at about 1.3 MB.
3. **Crashes at 4 and 12 sessions** (Nov 5). The split-state stack overflow has a fix on main (`ea2201d80`), not run on cards since.

## The layout being taken

Four-way tensor split (`-sm tensor`, every card on every request) with one shared unified pool, instead of `dp-4`'s
experts split across cards and private per-slot cells. Measured on the fork before any rental:

- Shapes divide four ways on whole IQ3_XXS blocks: expert and shared feed-forward 2,048, dense 12,288, 64 heads,
  32 indexer heads, vocabulary 154,880 (the pack's GGUF header).
- The DSA latent cache and the indexer cache are mirrored on every card under `-sm tensor` (`llama-model.cpp`'s split
  rules). At q8_0 the caches take 11.2 KiB a token (target, indexer and MTP, from the served pack's load logs), so the
  shared pool holds about 4.4M tokens: roughly 120K per session at 36, with any one session able to grow past 400K.
- `test-llama-archs --glm-tensor-device CUDA0,CUDA0,CUDA0,CUDA0`: four ranks against one device, mean KL 2.7e-15 with
  the default cache and 2.4e-15 with q8_0, no differing top tokens.
- A GLM fixture with the served pack's attention shape, served by `llama-server` over four ranks with `--kv-unified`
  and MTP depth 2, 36 slots, 20 of them ending early: 332 of 333 decode steps are one pass, no assert, replies identical
  to one card 36/36. The same server on `dp-4`'s private slots split 337 of 358 steps, mostly into 8 passes.

On paper the two layouts tie on round time (`dp-4` reads replicated dense weights and moves about 94 MB a card a round; four-way tensor reads every weight once and moves about 240 MB a card a round in reductions). What tensor buys is one pass a step whichever slots are busy, with no engine change, on the path this fork has served on two cards since September; what it costs is the pool above. Its round time at 36 sessions has never been measured on cards.

The next number is four RTX PRO 6000 cards serving the real pack this way at 1, 8, 16 and 36 short sessions, per-session decode rate read as each leg ends. Near the goal: this is the serving layout. Short: the box's own phase timers say whether the time is weight reads (kernel work, which `dp-4` needs too) or reductions (where `dp-4` moves less).

## Before any work

Write down which of the three it moves, the number expected, and how it will be read, then do it. Use a rented box only to read a number the local cards and CPU runs can't give, read the target number in its first minutes, and destroy it the moment the number is clearly short.
