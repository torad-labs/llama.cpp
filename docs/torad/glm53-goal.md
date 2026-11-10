# GLM-5.3 Flash serving goal

Read this before starting any GLM-5.3 work, and check the work against it when it is done.

## The goal

GLM-5.3 Flash, the BoldingBuilds IQ3_XXS pack, served by this fork on four RTX PRO 6000 cards for 23 concurrent agent sessions of up to 256K tokens each (decided Nov 9; 36-40 sessions was the earlier target):

- **70 decode tokens/s per session, with all of them decoding at once** (50 is the floor worth serving).
- Context: one shared pool on the 4-bit latent cache (`--cache-type-k q4_0 --cache-type-v q4_0`, accepted Nov 9 with its Oct 30 measurement: same speed, KLD 1.26x the reorder floor at 196K-262K), 8,352 bytes a token mirrored on every card. The pool that holds 23 sessions of 256K (6,029,312 tokens) does not load yet: see cause 3.

Per-session decode rate at N concurrent sessions is the number. Aggregate is N times it. A single-session rate, a kernel time or a trace share is not progress toward the goal until it moves that number.

## Where it stands (measured)

Four RTX PRO 6000 Max-Q, the real pack, engine `d9626d7`, `serve.py --profile tensor-4 --context 524288`, MTP depth 2, `rate.py` (sampled, temperature 1, top-p 0.95), Nov 9:

| Sessions | tok/s a session | Tokens a step | ms a step | Cards busy |
| ---: | ---: | ---: | ---: | --- |
| 1 | 149 | 2.23 | 15 | |
| 2 | 101 | 2.33 | 23 | 25-44 % |
| 4 | 51 | 2.18 | 43 | 34-46 % |
| 8 | 35 | 2.10 | 60 | 29-33 % |
| 16 | 20 | 2.11 | 104 | 27-30 % |
| 23 | 13.8-14.6 (15.4 greedy) | 2.14-2.16 (2.23) | 148-156 (144) | 23-30 % |

| What else | Value | Source |
| --- | --- | --- |
| The same server on two of the cards (`-ts 1,1`), 23 sessions / 1 session | 13.5 tok/s, 156 ms a step / 155 tok/s, 15.5 ms | same box |
| `--backend-sampling`, 23 sessions / 1 session | 11.4 tok/s, 186 ms a step / 118 tok/s | same box |
| 36 sessions, `dp-4`, one pass a step, equal drafts | 148 ms a round, about 13 tok/s a session | Nov 5 box: `9ca2776d0`, `6384cd2ac` commit messages |
| One session, `dp-4` | 45-73 tok/s | Nov 4 profile; Nov 5 audit |
| The fork's own model of the memory work at 23 sessions (not measured) | 26 ms a step at rated memory bandwidth, 37 ms at 70 % of it | `glm53-decode-pcie.md`, `estimate_decode_pcie.py` |
| Same model and cards, another engine (TensorFold, EXL3 4-bit experts, FP8 dense, DFlash2) | 70 tok/s per request at 16 concurrent (prose), 86 (code); 40 slots, shared 5.1M-token pool | mia-ai.net, measured Oct 4 |

## Where a step goes

A step at 23 sessions is about 150 ms on four cards, on two cards and on `dp-4`. The layout does not move it, and neither do card-to-card reductions or weight reads:

- The cards are busy 23-30 % of the step: about 40 ms of work, which is the model's 26-37 ms. One host thread, the server's loop, is at 100 % of a core for the whole step, and every other thread is idle.
- The step grows by about 5.5 ms for each session added, in a straight line from 2 to 23.
- `LLAMA_TIME_DECODE=1` at 23 sessions: about four `llama_decode` calls a step (the target's pass and the MTP head's), about 25 ms each: `graph_compute` 69 %, `alloc_graph` 15.5 % (12-13 ms each time it runs, in under half the calls), 13 % in none of its phases, `build_graph` and `set_inputs` 1 % each. That is about 100 ms of the step inside `llama_decode` and about 60 ms outside it, in the server's own work for each slot.
- The sessions share one pool, the pool's padded length sets every graph's shapes, and 23 sessions cross a 256-cell padding every 5 steps where one session crosses it every ~115. Each crossing builds the step's four graphs again (the target's, the MTP head's two proposals and its catch-up), plans their memory, runs them uncaptured for that step and captures them the next. That is the `alloc_graph` share above (a graph planned in 34 of 64 decodes) and a session's target graph reused in 41 of its ~61 steps. Found Nov 9 on a fixture of the pack's block layout and vocabulary (46 blocks, 154,880 tokens, 23 sessions, two ranks on one local card, `GGML_GRAPH_TRACE=1`): 22 of 90 steps built a graph, and a crossing's two steps took 76 ms more than a 31 ms step.
- With every host cost gone the step would be the cards' 40 ms: 54 tok/s a session at 2.16 tokens a step. 70 needs that and either 3 tokens a step at the same rows or less work on the cards.
- The routed experts are about 21-24 ms of the cards' 40, and the cards' memory rate sets them, not the kernel: 69 rows read 86 % of every bank when the sessions route apart. Read Nov 9 on an RTX 5080 (960 GB/s) at a card's quarter of a layer's bank (`test-backend-ops perf -o MUL_MAT_ID`, 288 experts, 8 used, 69 rows): gate or up 258 us and down 338 us in IQ3_XXS (768 and 586 GB/s), 344 and 358 us in NVFP4 (846 and 812 GB/s), 658 and 716 us in Q8_0 (835 and 768 GB/s). A layer is 854 us in IQ3_XXS and 1,047 us in NVFP4: NVFP4's 4.5 bits are 1.47 times IQ3_XXS's bytes and its kernel moves them 1.1 to 1.4 times as fast. By memory rate to an RTX PRO 6000 (1,597 GB/s Server, 1,792 Workstation), the target's 42 banks are 19-22 ms in IQ3_XXS and 24-26 ms in NVFP4. A pack with NVFP4 experts (the expert format of `orcarouter/GLM-5.3-Flash-Uncensored-NVFP4`; `conversion/base.py` repacks `nvfp4-pack-quantized` experts into `GGML_TYPE_NVFP4`, not run on that checkpoint) would be 4.3-4.9 ms a step slower at 23 sessions and 13.7 GB a card larger. The MTP block's bank is Q8_0: 2.2-2.5 ms a step (two proposals at 23 rows, the catch-up at 69), and 1.2-1.3 in NVFP4. The smaller formats do not bank their bytes in this engine: at the same 69 rows IQ2_S takes 387 and 489 us (1,262 us a layer) and IQ2_XXS 311 and 306 us (927), 35-45 % of the card's memory rate where IQ3_XXS's tile reaches 60-80 %, so IQ3_XXS is the fastest of the five formats here (another job shared the card for part of that run; its IQ3_XXS rows read as the clean ones). Unmeasured: the sessions' real routing, and whether a 4.5-bit target accepts more drafts (NVFP4 experts break even at 70 % of drafts accepted against today's 58-62).

## Known causes, each a number

1. **Host work that grows with the sessions, on one thread** (the cause of the 150 ms; about 5.5 ms a session a step). Two parts:
   - **The step's graphs are built again as the shared pool grows** (fixed in the engine, not read on cards yet). A pool that sequences share is padded to 256 cells a sequence, the next power of two, so it crosses as often a step as one sequence does (`LLAMA_KV_PAD_SHARED_LEGACY=1` keeps 256). On the local fixture at 23 sessions one step of 91 builds a graph where 22 of 90 did, and a step averages 30-34 ms where it averaged 37-46 (four and five runs; other work shares that card and CPU).
   - **A rollback read every cell of the pool, for each session, each step** (fixed in the engine, not read on cards yet). The first session to roll back in a step lowers the pool's head to its freed cell; the cells the others placed in that step are then above the head, and the walk for them went down to cell 0 and came back from the pool's last cell. On the local fixture at 23 sessions that was 3.7 ms of a step's host work at 131,072 cells over a 11,776-cell pool's (2.5 ms verifying, 1.2 drafting), and it grows with the pool: in a straight line from those two pools, about 16 ms a step at the 524,288 cells the box ran and about 190 ms at the 6,029,312 the goal needs. Now the cells just above the head are read first and none above the last used cell (`LLAMA_KV_SEQ_RM_HEAD_LEGACY=1` keeps the old walk): the same host work at 131,072 cells as at 11,776.
   - Expected on four cards with both: `alloc_graph` gone from the step, and 23 sessions at 45-60 ms a step, 36-48 tok/s a session at MTP depth 2.
   - **A verified row's candidates, read on the server's thread** (open). Of the fixture's 32 ms step at 23 sessions and short histories the verify stage is 5 ms, and 3 ms of it is `common_sampler_top_k_candidates` for one row a session: 131 us a row, 62 % of it keeping the heap of the 296 candidates the served chain reads (top-k 40 behind a penalty window of 256), the rest reading the row's 154,880 logits. The box verifies about two rows a session. The step's two draft stages (3.1 and 2.3 ms) are not sampling: the thread waits there for the proposal's decode on the card (stack samples of the server's thread by stage, Nov 9). Read and not the fix: sampling on the cards (`--backend-sampling`) takes the verify stage to 0.4 ms and adds 6.5 ms to the target's decode, a step of 33.3 ms against 32.1; the backend's top-k alone, for a chain with no penalty, adds 3.0-3.6 ms to the decode at k 40 and 1.4 ms at k 128 for 1.5 ms of verify taken; reading the rows' candidates on worker threads ahead of the loop took longer than the loop on a CPU other work shares two to three ways (verify 6.4-8.6 ms against 5.0-5.4) and was not kept.
   - **The MTP head's draft samplers took a seed from the system's random device at every reset**, one a sequence a draft (fixed in the engine): 0.56 ms a step at 23 sessions, the step's first host stage 0.65 ms and 0.09 with a fixed seed. A draft is its row's first candidate, so no draw decides it.
2. **Passes per step** (fixed on main for the shared pool). Every ubatch is a full weight read: `9ca2776d0`, `861d22f10`, `e18d70a0b` for MTP depth and the output limit, `a3eff8478` pads any shorter draft to the step's longest. `dp-4`'s private streams still split on idle slots between busy ones.
3. **The pool for 23 sessions of 256K does not load.** At `-ub 2048` the graph reserve asks for a compute buffer that grows with the pool: 31.8 GiB on a card at 6,029,312 tokens, on top of 48 GiB of cache and 32 GiB of weights. A 524,288-token pool loads with 48.6 GiB used a card. What the reserve costs a pool token, and at which batch step, has two measurements and needs a third to be a rule.
4. **`dp-4` crashes at 4 and 12 sessions** (Nov 5). The split-state stack overflow has a fix on main (`ea2201d80`), not run on cards since.
5. **Host work that grows with the cells the pool holds** (open; larger than all of cause 1 at any real history). A shared pool's sequences are more than one to a stream, so `llama_kpool_set_input` takes none of them from its resident view (one sequence a stream only) and builds the indexer's pool maps and mask facts from the cells at every decode: for each session it walks every cell of the pool several times and writes two integers a cell (`kpool_mask_cells` is n_kv x 2 x sessions). On the fixture at 23 sessions (Nov 9, `LLAMA_TIME_DECODE=1`, the last 64 decodes of a run) `set_inputs` is 0.85 ms a decode at 8,192 cells (48-token prompts), 1.62 ms at 32,768 (1,000-token prompts) and 6.38 ms at 131,072 (4,000-token prompts, the fixture's longest), four decodes a step: steps of 32.4, 34.8 and 63.8 ms. Inside it at 24,576 cells: each session's walks of the cells 66 %, the facts and the tails 18 %, the runs' bounds 13 %. That is about 48 ns a cell a decode: by that line 25 ms a decode and 100 ms a step at the 524,288 cells of 23 sessions of 22,800 tokens, on the fixture's CPU. The Nov 9 box read short prompts, so its 150 ms step holds none of it, and the 45-60 ms expected above is for short histories only.

## Tokens a step

Rate a session = tokens landed a step / time a step.

| Drafter | Tokens a step | Source |
| --- | --- | --- |
| MTP depth 2, 23 sessions | 2.14-2.16 sampled, 2.23 greedy (58-62 % of drafts accepted) | Nov 9 box |
| MTP depth 1 / 2 / 3 / 4, one session | 1.7 / 2.1 / 2.4 / 2.5 from acceptance 0.73 / 0.55 / 0.47 / 0.38 | `evidence.md`, greedy; depth 3 and 4 unmeasured at 23 sessions |
| DFlash2 for GLM-5.3 Flash, 7 drafts a step, 23 sessions | 2.58 sampled, 2.89 greedy (23-28 % of drafts accepted) | Nov 9 box, first run on the real pack; the draft was trained on the original model, not the uncensored pack |
| DFlash2, one session on a coding task | 3.66 | Nov 9 box |

DFlash2 verifies 8 rows a session a step against MTP depth 2's 3, and while a step is bound by host work for each row it pays for them: 166-210 ms a step at 23 sessions against 144-156.

## The layout

`tensor-4`: four-way tensor split (`-sm tensor -ts 1,1,1,1`, every card on every request) with one shared unified pool. It is the layout the reads run on because it is one pass a step whichever slots are busy, with no engine change. It ran on four real cards for the first time on Nov 9 (a fixture of the pack's attention shape, then the pack). `dp-4` (experts split across cards, private 442,368-cell slots) stays in the tree.

- Shapes divide four ways on whole IQ3_XXS blocks: expert and shared feed-forward 2,048, dense 12,288, 64 heads, 32 indexer heads, vocabulary 154,880. A fixture with fewer heads than ranks asserts in the meta backend; the pack does not.
- The DSA latent cache and the indexer cache are mirrored on every card under `-sm tensor`.
- Load takes 67-74 s from a warm page cache.
- The split does not move the routed experts, which are most of a step's reads: every split reads each touched expert once. A card's 72 whole experts under `-sm dp` take what its quarter of all 288 takes under `-sm tensor` (Nov 9, RTX 5080, IQ3_XXS, `test-backend-ops perf -o MUL_MAT_ID_PARTITION`: gate or up 289 us and down 291 us at 36 rows, 346 and 354 us at 108, 654-777 GB/s; about 837 us a layer at 69 rows against the quarter's 854). What a split moves, by `scripts/glm53/estimate_decode_pcie.py`'s byte model at 23 sessions, MTP depth 2, 400K histories and the cards' rated 1,597 GB/s: `dp-4` reads for 26.0 ms a step (its dense weights are read on every card, 9.2 GB a pass against a quarter's 2.3) and transfers for 1.7 ms as designed or 6.4 as coded; `tensor-4` reads for 21-23 ms by the same model (23 if every card scans every session's index keys, which its mirrored caches allow); shared dense weights with a session's caches on one card would read for 21. The experts are 17.8 ms of each. `tensor-4`'s cards measured 40 ms. `dp-4` keeps a session's caches on one card, which is the memory the 23 x 256K pool lacks (cause 3), not a faster step.
- Open, off the goal's path: on the CPU, four tensor-split ranks return NaN logits when the cache length is padded well past its used cells, with the prefill padding as with the shared pool's (`test-speculative-mtp`'s prepared-weights check, which keeps 256 for that reason). One and two CPU ranks, and four ranks on a GPU, do not.

## Launch

The launch is in this fork and needs nothing outside it. `scripts/glm53/serve.py` holds every server argument and
environment setting of the layout; `scripts/glm53/rate.py` reads the number from the running server.

```sh
python3 scripts/glm53/serve.py --profile tensor-4 \
    --server <build>/bin/llama-server \
    --model <pack>/GLM-5.3-Flash-Uncensored-IQ3_XXS-00001-of-00003.gguf
python3 scripts/glm53/rate.py --sessions 23
```

- `tensor-4` is `-sm tensor -ts 1,1,1,1 --kv-unified`, 23 slots on one pool of 524,288 tokens (the pool measured to
  load), the 4-bit latent cache, MTP depth 2. `--slots`, `--context`, `--mtp-depth 1..4`, `--mtp-adaptive`, `--backend-sampling` and
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

## The next number

The step at 23 sessions with the pool holding the sessions' histories. With short prompts the shared pool's padding and rollback fixes predict 45-60 ms on four cards against the 150 measured; with histories the indexer's inputs (cause 5) are the step, 64 ms on the fixture at 4,000 tokens a session against 32 at 48. So first cause 5 on the local fixture, read as `set_inputs` a decode at 8,192, 32,768 and 131,072 cells (`LLAMA_TIME_DECODE=1`, 23 sessions with prompts of 48, 1,000 and 4,000 tokens), until it no longer grows with the pool; then the candidates' read of cause 1. No box is rented before the fixture's step at 4,000 tokens a session is its step at 48 and the step that predicts for the cards clears the floor. Then the box, with prompts of the sessions' real length, read as `rate.py --sessions 23` reads it, with `LLAMA_TIME_DECODE=1` saying whether `alloc_graph` and `set_inputs` left the step. The engine's own markers (`LLAMA_DECODE_TRACE=1`) give a step by stage, and `GGML_GRAPH_TRACE=1` says which decode built a graph and why.

A box read starts with the pack already on the box's disk and the step read in the first two minutes after load. Under 30 tok/s a session at MTP depth 2, no drafter lands enough tokens to reach the 50 floor (it would take 3.5 a step), and nothing else is read from that box but where the step goes.

## Before any work

Write down which of the three it moves, the number expected, and how it will be read, then do it. Use a rented box only to read a number the local cards and CPU runs can't give, read the target number in its first minutes, and destroy it the moment the number is clearly short.
