# GLM-5.3 Flash serving goal

Read this before starting any GLM-5.3 work, and check the work against it when it is done.

## The goal

GLM-5.3 Flash, the BoldingBuilds IQ3_XXS pack, served by this fork on four RTX PRO 6000 cards for 36-40 concurrent agent sessions:

- **50-70 decode tokens/s per session, with all of them decoding at once.**
- Session histories that grow toward ~400K tokens. This is why the layout is `dp-4`: `tensor-2` holds about 1M cells per card pair, roughly 55K per session at 36 sessions.

Per-session decode rate at N concurrent sessions is the number. Aggregate is N times it. A single-session rate, a kernel time or a trace share is not progress toward the goal until it moves that number.

## Where it stands (measured)

| What | Value | Source |
| --- | --- | --- |
| One session, `dp-4` | 45-73 tok/s | Nov 4 profile; Nov 5 audit (71.2 uninstrumented) |
| 2 / 4 / 8 / 40 sessions, `dp-4`, per session | 28 / 12-16 / 6 / 0.9 tok/s | Nov 3-4 (`glm53-dp.md`) |
| 36 sessions, one pass a step, equal drafts | 121 ms a step, about 15 tok/s a session at ~1.8 tokens a step | `9ca2776d0` commit message |
| Bandwidth floor, 36 sessions × 3 rows | about 24 ms a step on `dp-4` (about 30 GB of experts plus 14 GB of dense weights per card at 1.79 TB/s) | arithmetic, not measured |
| Same model and cards, another engine (TensorFold, EXL3 4-bit experts, FP8 dense, DFlash2) | 70 tok/s per request at 16 concurrent (prose), 86 (code); 40 slots, shared 5.1M-token pool | mia-ai.net, measured Oct 4 |

The gap is about 5x on time per step, before slot gaps.

## Known causes, each a number

1. **Passes per step.** Every ubatch is a full weight read. Drafts of different lengths (fixed on main: `9ca2776d0`, `861d22f10`, `e18d70a0b`) and idle slots between busy ones (open: `split_equal`'s sequential split and `split_cache_view` need each card's busy streams consecutive; 8 passes a step measured on CPU once replies end at different times) multiply the step.
2. **Time per pass.** 121 ms against a ~24 ms floor. Nov 4 trace: collectives 74 % of summed kernel time, cards idle 46-51 % of the step, hundreds of graph instantiations per request. Collective dispatch picks its transport by message size (≤128 KB peer packets, 128 KB-1 MB host-staged chunks, ≥1 MB copy path); a 36-session step's reductions sit at about 1.3 MB.
3. **Crashes at 4 and 12 sessions** (Nov 5). The split-state stack overflow has a fix on main (`ea2201d80`), not run on cards since.

## Before any work

Write down which of the three it moves, the number expected, and how it will be read, then do it. Use a rented box only to read a number the local cards and CPU runs can't give, read the target number in its first minutes, and destroy it the moment the number is clearly short.
