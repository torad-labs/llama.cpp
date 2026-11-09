# GLM-5.3 Flash serving goal

Read this before starting any GLM-5.3 work, and check the work against it when it is done.

## The goal

GLM-5.3 Flash, the BoldingBuilds IQ3_XXS pack, served by this fork on four RTX PRO 6000 cards for 23 concurrent agent sessions of up to 256K tokens each (decided Nov 9; 36-40 sessions was the earlier target):

- **70 decode tokens/s per session, with all of them decoding at once** (50 is the floor worth serving).
- Context: one shared pool under the layout below, on the 4-bit latent cache (`--cache-type-k q4_0 --cache-type-v q4_0`, accepted Nov 9 with its Oct 30 measurement: same speed, KLD 1.26x the reorder floor at 196K-262K): 8,352 bytes a token mirrored on every card, about 6.0M tokens, so 23 sessions at 256K at once. At q8_0 it is 4.4M (17 sessions at 256K). `dp-4` holds 15.9M private cells (442K a slot, one copy of each cache) and stays in the tree; no engine on these cards has served that at speed.

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

1. **Passes per step.** Every ubatch is a full weight read. Drafts of different lengths (fixed on main: `9ca2776d0`, `861d22f10`, `e18d70a0b` for MTP depth and the output limit, `a3eff8478` pads any shorter draft, such as one cut at the end token, to the step's longest) and idle slots between busy ones (open: `split_equal`'s sequential split and `split_cache_view` need each card's busy streams consecutive; 8 passes a step measured on CPU once replies end at different times) multiply the step.
2. **Time per pass.** 121 ms measured for the target pass against about 28 ms of modeled memory work. Nov 4 trace: collectives 74 % of summed kernel time, cards idle 46-51 % of the step, hundreds of graph instantiations per request. Collective dispatch picks its transport by message size (≤128 KB peer packets, 128 KB-1 MB host-staged chunks, ≥1 MB copy path); a 36-session step's reductions sit at about 1.3 MB.
3. **Crashes at 4 and 12 sessions** (Nov 5). The split-state stack overflow has a fix on main (`ea2201d80`), not run on cards since.

## Tokens a step

Rate a session = tokens landed a step / time a step, and 23 sessions against 36 shortens the step only about 12 % (it still reads about 86 % of the experts). The fork's model at 23 sessions: 26 ms a step at rated memory bandwidth, 37 ms at 70 % of it, so 70 tok/s needs about 2.6 tokens a step at 37 ms.

| Drafter | Tokens a step | Source |
| --- | --- | --- |
| MTP depth 2 (served) | 2.0-2.35 | Oct 30 box, mean accepted length |
| MTP depth 1 / 2 / 3 / 4 | 1.7 / 2.1 / 2.4 / 2.5 from acceptance 0.73 / 0.55 / 0.47 / 0.38 | `evidence.md`, single session, greedy; deeper lost alone, unmeasured with many sessions where rows are nearly free |
| DFlash2 for GLM-5.3 Flash (block of 8: up to 7 drafts a pass; reads target layers 6, 15, 25, 34, 43) | not measured on the pack | runs beside a four-rank tensor split on a 45-layer fixture of the pack's width and vocabulary (8 sessions, 7 drafts each, every step one pass); a random target accepts nothing, so its acceptance needs the real pack; cc-by-nc-nd-4.0; trained on the original model, not the uncensored pack |

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

## Launch

The launch is in this fork and needs nothing outside it. `scripts/glm53/serve.py` holds every server argument and
environment setting of the layout; `scripts/glm53/rate.py` reads the number from the running server.

```sh
python3 scripts/glm53/serve.py --profile tensor-4 \
    --server <build>/bin/llama-server \
    --model <pack>/GLM-5.3-Flash-Uncensored-IQ3_XXS-00001-of-00003.gguf
python3 scripts/glm53/rate.py --sessions 23
```

- `tensor-4` is `-sm tensor -ts 1,1,1,1 --kv-unified`, 23 slots on one pool of 6,029,312 tokens, the 4-bit latent
  cache, MTP depth 2. `--slots`, `--context`, `--mtp-depth 1..4`, `--mtp-adaptive`, `--backend-sampling` and
  `--dflash FILE --dflash-depth N` change one thing each; `--dry-run` prints the argv and environment as JSON.
- The pack: `BoldingBuilds/orcarouter_GLM-5.3-Flash-Uncensored-GGUF` at revision
  `1bc0afadd92f4ff9fc2b1281a95a7921b058d35d`, the three `IQ3_XXS/` shards in one directory (134.3 GB).
- The engine: this fork with CUDA for the cards' compute capability, `120a` for RTX PRO 6000:
  `cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120a -DGGML_CUDA_FA=ON
  -DGGML_CUDA_GRAPHS=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF` and
  `cmake --build build --target llama-server`.
- The DFlash2 draft: `Anbeeld/GLM-5.3-Flash-DFlash2-GGUF` at `f3569acb7f9cee95e66d24442d75536eb00ab842`,
  `GLM-5.3-Flash-DFlash2-Q8_0.gguf`. As published it holds neither token embeddings nor an output head and reads the
  target's, which a tensor-split target keeps where no draft context can run: the server aborts at load.
  `scripts/glm53/dflash_own_head.py --draft FILE --target <the three shards> --out FILE` writes the draft with its own
  copies, and `--dflash` takes that file. Its cache is its own 2,048-token window a session plus one step, about 1 GiB
  at 23 sessions whatever the pool holds.

On vast.ai, rig rents the box, pulls the pack by its hashes and ships the engine build; the launch is still the two
lines above.

## The read (written before the box, Nov 9)

Four RTX PRO 6000 cards, the real pack, `tensor-4`, 23 short sessions decoding at once, `rate.py` run as soon as
`/health` answers. It moves the per-session rate at 23 sessions, which has no measurement on this layout.

| Leg | Expected | From |
| --- | --- | --- |
| Load, 6,029,312-token pool | loads, a few GiB free a card | 8,352 bytes a token and the q8_0 load logs |
| MTP depth 2, as served | 2.0-2.35 tokens a step greedy, fewer sampled; 26-37 ms a step, so 54-90 tok/s a session | the tables above |
| MTP depth 3, 4, fixed depth | more tokens a step than depth 2 if a row is nearly free at 23 sessions | acceptance 0.47 / 0.38 alone |
| Backend sampling on | a shorter step: 69 rows of 154,880 logits stay on the cards | `10d6bd57d`, two cards |
| DFlash2, depth 7 | unknown: never run here, and trained on the original model | |
| 1 and 36 sessions | 1: near `tensor-2`'s 155-174 tok/s; 36: a step about 12 % longer than at 23 | Oct 22 box; the 86 % / 95 % expert reads |

Under 30 tok/s a session at depth 2, no drafter measured anywhere lands enough tokens to reach the 50 floor on that step
(it would take 3.5 a step): the other legs are dropped and the box is destroyed once three things are read from it,
which say where that step goes: the step at 1 session against 23 (what a step costs before its rows), each card's
utilization during the 23-session read (busy cards are weight reads, waiting cards are reductions or the host), and one
leg with `LLAMA_TIME_DECODE=1` for the host's share. At or over it, the legs run in the order above, each read as it
ends, and the box is destroyed after the last.

## Before any work

Write down which of the three it moves, the number expected, and how it will be read, then do it. Use a rented box only to read a number the local cards and CPU runs can't give, read the target number in its first minutes, and destroy it the moment the number is clearly short.
