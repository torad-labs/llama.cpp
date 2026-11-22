# Inkling 2 Small on four RTX PRO 6000: kernels shaped to the architecture

Scope: changes that are only possible because the engine knows it is running Inkling 2 Small. They use its attention geometry, its bounded K and V, its fixed window, its shared Q input and its router. Levers that would help any model (synchronization structure, L2 warming, the ring redesign, the layer fusions) are already campaign rows and are only referenced here.

Marking used below. Verified: read in code or in a measured report of the fork. External: another team's published result. Inferred: my reading, to be checked before it is relied on.

---

## 1. Facts this plan stands on

### 1.1 Where a decode step goes

Rented four RTX PRO 6000 box, 23 sessions, Q3_K_M, F16 KV, no draft, one sequential arm each (verified, from the tensor-4 seat's session):

| Arm | tok/s | p99 gap |
|---|---:|---:|
| Stock | 292.01 | 92.5 ms |
| Combined (query-first attention, compact greedy, shared quantize) | 425.61 | 65.1 ms |
| Combined, `--expert-row-limit 32` | 424.23 | 67.6 ms |
| Combined, per-reader quantize | 422.44 | |

425.61 is the total of 23 sessions (18.5 tok/s a session). The step is about 54 ms.

No trace of the combined arm exists yet. The shares below come from the stock Nsight trace (GPU 0, 82.6 ms profiled step) and from the arms:

- Global attention 34.6 ms in eight launches. About 18 ms after query-first.
- Sliding-window attention 1.7 ms.
- Experts 12.8 ms on average, 16.5 ms on the critical path.
- Owner gather 3.8 ms. Idle 18.8 ms.

The workload: every session holds 262,144 tokens, 196,608 shared with the others and 65,536 private. 1,703,936 physical cells.

Byte arithmetic for global attention, per card and step:

- One cell is 2 local KV heads x (128 K + 128 V values) x 2 bytes = 1,024 bytes a layer, 8,192 bytes over the eight global layers.
- Shared cells read once: 1.61 GB. Private cells: 12.35 GB. Together 13.96 GB.
- As the stock kernel reads, every query re-reading the shared cells: 49.4 GB.
- The fork's own figures for the same thing: 49.4 GB against 13.9 GB, 27.6 ms against 7.7 ms at about 1.8 TB/s.

Two consequences. First, global attention at 18 ms is a little over twice its read-once floor of 7.7 ms. Second, private cells are 88% of the read-once bytes, so reading the shared prefix once removes re-reads but does not lower the floor. Only fewer bytes a cell does that.

### 1.2 Architecture facts used (verified in code)

- **The relative bias is exactly zero beyond 1,023 tokens.** The table is `{16, extent}` a layer, with extent 1,024 for sliding and global layers alike. `llama_kv_cache::set_input_relative_pos` sends any distance outside `[0, extent)` to index `extent`, the unfused path pads the table with a zero column for it, and the fused kernels compute `distance >= 0 && distance < n_dist ? table[...] : 0.0f` (`fattn-rel-decode.cuh`, `fattn-rel-mma.cuh`). `02-architecture.md` says "a single learned value beyond that". The code says zero.
- **Stored K and V are bounded by construction.** K is stored after its convolution and its L2 clip: a head vector's norm is at most 22.6274, so no element exceeds 22.63. V is stored after its convolution and an RMS norm with no gain: RMS exactly 1 over 128 values, so no element exceeds sqrt(128) = 11.31. Q is clipped the same way, then multiplied by tau in global layers (tau is 1 below 128,000 tokens and 1 + 0.1 ln(position / 128,000) above).
- **Four products read the Q input in every layer.** `wq`, `attn_rel` (the 16-wide relative feature), `attn_sink_proj` and `wqkv_gate` all read `inputs[0]`, in sliding and global layers. K and V read the same input on sliding layers and their own norms on global layers, where two more small products predict the dynamic conv taps.
- **Two F32 products a layer.** `table = ggml_mul_mat(bias_weight, r)` builds the bias table for every query head and row. The router is `{4,608, 384}` with logits in F32.
- **The sliding window equals the bias extent**, 1,024, on the 40 sliding layers. Inside the window every cell has an exact bias. Nothing is far.
- **The sink joins the denominator only.** The direct-sink path adds it at the merge of the partials (`fattn_rel_decode_merge`).

### 1.3 How the global decode kernel works today (verified, server tree)

`fattn_rel_mma_rows` in `fattn-rel-mma.cuh`:

- One warp per query, local KV head and stream. It scans packed mask words (4 words a lane, 4,096 cells a warp a scan step), lists the visible cells, gathers 16 visible KV rows into a shared-memory tile reading K and V 16 bytes at a time, and multiplies the tile by the six query heads of its KV head on the tensor cores.
- For every cell it reads the cell's position, forms the distance to the query and takes the table entry or zero.
- F16 K and V only. A Q8_0 cache takes the one-cell kernel.
- Query-first reorders independent blocks so the blocks of one cell range run close in time. Every stream computes the same partial and the merge is unchanged. Inferred: the +26.8% comes from the second-level cache serving the re-reads. Reading 49.4 GB in about 18 ms would be 2.7 TB/s, more than the card delivers from memory, so part of it is already served from cache.

So three things are general where the model is specific. Each of 23 queries gathers the shared cells again. Cells are found by scanning a bit mask, not by address. The bias branch runs on every cell although 99.6% of the cells a query sees here are beyond the extent.

### 1.4 What others built for this architecture (external)

| Who, hardware | What bears on this plan |
|---|---|
| vLLM, GB200 | The 6:1 head ratio silently disabled head packing in their FlashAttention-4 decode path. Their decode kernel packs the six heads in one tile and reads KV once: a batch-1 decode step at 100K context went 15.1 to 4.8 ms. All-reduce, short conv, residual update, depth mix and RMS norm are one kernel. Activation quantization sits inside the router and SwiGLU kernels. |
| SGLang, GB300 | A relative-bias decode kernel that splits the KV across up to 64 blocks and reads a compact distance-indexed table: 25 to 9 us a layer at batch 1. One prologue kernel before attention. Router sigmoid, bias and top-k in one kernel. |
| TokenSpeed, B300 | One launch clips Q, K and the relative heads, norms V, optionally fuses the relative projection, the KV store and the decode ring convolution. Gate and sink are rows of the query projection. KV as E4M3 with a scale per 32 values, dequantized in shared memory bit for bit with the BF16 kernel. BF16 and FP8 KV give the same curve up to 32 sessions; at 64 FP8 gives 34% more output a GPU. Five sliding-window groups keep only the last 1,024 tokens. |
| meldavy, 3x RTX PRO 6000 (the same card) | FP8 KV as E4M3 with scale 1.0, cast in registers on store and back to BF16 in registers after each load. NLL 1.0363 / 1.0366 against 1.0351 / 1.0353. Top-1 agreement with a BF16 reference 98.0 to 98.1%, where two BF16 runs agree 98.6%. KV capacity 858,852 to 1,729,548 tokens. Decode unchanged at one stream (106.5 against 106.3 tok/s). sm_120 allows 99 KB of shared memory a block against 227 KB on sm_100. The cuBLAS fallback for products of 16 rows or fewer "runs well below memory bandwidth"; a split-K GEMV replaced it, router included. DSpark adds nothing at four sessions. |
| vLLM RFC 54704, Inkling 1, B200 | FP8 Q and KV fed straight to the kernel: best kernel speedup 1.45x at batched decode, 64K cells, 32 rows. At unit scale V outliers clipped (largest value 1,272 against E4M3's 448) and MATH-500 lost 4.2 points; K never clipped. |
| ds4-dfm-rs, DGX Spark | Global-layer KV as NVFP4 pages. Staging the key tile once for all verify rows inside one block ran slower (113.9 to 120.8 ms): the rows' scores serialize inside one block. Filling the tensor-core query slots with the rows' heads instead ran faster (113.8 to 102.7 ms). |

Chinese-language sources (Exa, seven queries aimed at Zhihu, linux.do, V2EX, CSDN, Juejin, WeChat, Bilibili, ModelScope and KTransformers): no first-hand kernel work on this architecture turned up. What came back were translations and summaries of the vLLM post (知微, 余广坝, AgentLensHQ), a relay of X posts (AGI Hunt), Unsloth's Chinese docs and two Taiwanese deployment guides. Two points were useful:

- A deep read of Inkling 1 (ricardokevins) notes that the global layers' learned distance bias covers only the nearest 1,024 positions and is zero past them while QK stays global. That is what sent me to check the fork's kernel, which does the same for Inkling 2.
- OrcaRouter's Chinese post says Mini was made by keeping the 128 most-used of 384 experts a layer. Secondary source, and about Mini, but it says expert use in this family is long-tailed.

KTransformers' README and site do not list Inkling.

### 1.5 Corrections to what I said earlier

- I said the far region carries one value per query. Wrong: it carries no bias at all.
- I proposed feeding FP8 K and V to the FP8 tensor-core instruction first. The lower-risk variant goes first: store one byte, convert while staging the tile, keep Q and the multiply in F16 (3.4).
- I ranked the K-quant expert ring first in my first answer. The fork's byte counts do not support that (3.8).

---

## 2. Measurements to take first

Each one decides something in section 3. They read; they change nothing. Write the hypothesis for each before it runs. Most come out of the Nsight trace of the combined arm that the rental runbook already schedules.

**M1. Distinct experts a layer in a traced step.** 184 routes a layer at 23 rows. Uniform routing gives about 147 distinct experts; the coding probe's median was 102. Decides 3.8.

**M2. Cell classes a query reads in a global layer.** Shared or private, near (distance under 1,024) or far. On this workload the answer should be 196,608 shared far, 64,512 private far and 1,024 private near. Confirm on real traffic, where sessions fork at other depths. Decides how much of 3.1 to 3.3 real traffic sees.

**M3. Memory bytes actually read by the global attention kernel under query-first**, against the 49.4 GB it reads by count, with its second-level-cache hit rate and its tensor-core and ALU time. Decides whether what is left of the 18 ms is bytes or arithmetic, so it sizes 3.1, 3.2 and 3.4 separately.

**M4. Stored K and V.** Largest element and RMS a layer as written to the cache (expected: K at most 22.63, V RMS 1, V at most 11.31), and the share of elements below 2^-6. Then the KL of each one-byte candidate against F16 on the parity capture. Decides 3.4.

**M5. Name the two cuBLAS kernels and the one full-MMQ product a layer.** `gemmSN_TN_kernel` and `Kernel2` run 47.9 times a step each; `mul_mat_q` with its fixup also runs 47.9 times. Inferred: the bias-table product, the router, and one quantized product whose shape misses the few-row guards. Decides 3.6 and 3.7.

**M6. Router weights.** For the eight selected experts of a token, the distribution of the normalized weights, smallest to largest, a layer. Decides the second half of 3.8.

**M7. Runs a sequence in the pools.** How many contiguous runs of cells a sequence holds in the global pool and in the window pool, in the served shape and after a session has been restored. Decides 3.3 and 3.5.

---

## 3. What to build

Every change is a switch that is off by default, with a `tunables.toml` line and a `TORAD.md` row, as the fork's other changes are.

### 3.1 Far cells without a bias

**Objective.** Stop computing position, distance and bias for cells that cannot have one.

**Build.**
- In a global layer, a query's near region is the cells within 1,023 positions of it. Everything else is far and its logit is the scaled QK product alone.
- Give the kernel the boundary instead of a position for every cell: the far region takes no `kv_pos`, no distance and no table.
- The near region keeps today's code path. It is at most 1,024 cells a query.

**What follows from it.** When a session's private tail is 1,024 tokens or longer, the whole shared prefix is far for every query. The shared-prefix kernel of 3.2 then has no bias, no positions and no table. A session that has just forked has near cells inside the prefix; let the per-query near kernel take the last 1,024 cells whoever owns them, and let the prefix kernel take the prefix minus those.

**Exactness.** The same values. The partials are merged in a different order, so results move by the reduction-order drift the fork already tracks. `test-backend-ops` cases with more than 1,024 cells against the CPU reference.

**Gain.** No bytes: a position is 4 bytes beside 1,024 bytes of K and V. It removes a load, a compare and a select per cell and head, and it is what makes 3.2 and 3.3 simple. M3 gives the arithmetic share.

### 3.2 The shared prefix read once

The fork's review lists this as its largest unbuilt item, up to about 10 ms. What the architecture adds is in 3.1 and in the constraints below.

**Build.**
- A decode kernel for the cells that several queries see: one read of a K and V tile serves a group of queries, each query's six heads filling tensor-core columns. Private cells keep the per-query path.
- Partials `(max, sum, accumulator)` a query and head, merged with the private and near partials by the existing merge. The sink joins at the merge, as today.

**Constraints, from measurements.**
- A layer's shared prefix is 196,608 x 1,024 bytes = 201 MB a card, more than the card's 128 MB second-level cache. Reuse has to happen inside a block or between blocks that run next to each other.
- ds4-dfm-rs found that staging a tile once for all rows inside one block serialized the rows' scores and ran slower. Filling the query slots of each multiply ran faster. Spread the queries over warps and blocks; do not loop over them in one.
- sm_120 gives a block 99 KB of shared memory. A 64-row tile of K and V is 32 KB in F16 and 16 KB at one byte a value (3.4).
- Choose the number of queries a block serves by a sweep on the card, as the fork's other geometry is chosen.

**Exactness.** Summation order changes. The KL parity capture is the gate, as for the two switches already in the tree.

**Gain bound.** From about 18 ms toward the 7.7 ms read-once floor.

### 3.3 Cells by run, not by mask

**Objective.** Let the kernel address K and V directly.

**Build.**
- Where a sequence's cells are contiguous (M7), describe what a query reads as a short list of runs (base, length) instead of a bit mask over the pool. The shared prefix is one run. A private tail is one run.
- The kernel then copies whole tiles (the ring kernel already uses `cp.async.bulk` on this card) instead of scanning mask words, building a list and gathering rows 16 bytes at a time.
- Keep the mask path for sequences that are not contiguous.

**Exactness.** The same cells in the same order.

**Gain.** Removes the scan and the gather from the far path, and replaces the per-microbatch upload of the 6.85 MB position array and the 4.92 MB mask with a few integers a sequence. The fork's review puts that upload at about 2.6 ms a step.

### 3.4 One byte per K and V value in the global layers

**Objective.** Halve the bytes of the largest memory-bound cost. Private cells are 88% of the read-once bytes and nothing else touches them.

**Why this model allows it.** A format with no scale needs bounded values. Inkling 1 did not have them: vLLM measured V up to 1,272 and a quality loss at unit scale. Inkling 2 clips K and normalizes V before the store, so both sit far inside E4M3's range of 448 with no clamp and no calibration (1.2).

**Candidates, picked by M4.**
- E4M3 with no scale: 32 bytes per 32 values, three mantissa bits, error relative to each value.
- Q8_0, which the cache already writes: 34 bytes per 32 values, error relative to the block's largest value, a scale to apply on load.

**Build.**
- Decode while the tile is staged, so the multiply and Q stay F16 and nothing downstream changes. Make the element decode a parameter of the tile kernel so both candidates run in the same kernel.
- Global layers only. The sliding layers hold 1,024 cells a session and are near their floor.
- The fork's campaign row `inkling-optimize-kv-precision` already covers proving Q8_0 and Q4_0; this adds the tile kernel as the reader and the scale-free candidate.

**Evidence.** On the same card, meldavy's E4M3 cache with conversion in registers: NLL up 0.1%, top-1 agreement 98.0 to 98.1% where two BF16 runs agree 98.6%, capacity doubled, single-stream decode unchanged. TokenSpeed: no difference up to 32 sessions, +34% a GPU at 64. Neither is this workload: 23 sessions over a long shared context is where bytes bind.

**Exactness.** Approximate. Gate on the KL capture, with the budget taken from the drift two stock runs already show against each other.

**Gain bound.** The read-once floor goes from 7.7 to about 3.85 ms, and the pool holds twice the tokens.

**Later, only if M3 shows arithmetic binding after this.** Feed FP8 K and V to the FP8 tensor-core instruction. Both operands must be FP8, so Q is rounded too. On sm_120 the plain form of the narrow-precision instructions runs well below the block-scaled form, so it would use the block-scaled form with unit scales.

### 3.5 The sliding window as a ring

**Objective.** A fixed-shape read for the 40 sliding layers.

**Build.**
- A ring of 1,024 rows a session (plus one prefill chunk, as ds4-dfm-rs keeps), contiguous a session.
- The window equals the bias extent, so the distance of a slot is its offset from the ring head. The kernel needs the head index and nothing else: no mask, no positions.
- First read how the fork's window cache lays cells today (M7); upstream gathers cells in position order once its ring wraps.

**Exactness.** The same cells and distances.

**Gain bound.** Sliding attention is 1.7 ms against a byte floor near 0.5 ms (23 x 1,024 cells x 1,024 bytes x 40 layers = 0.97 GB), plus its share of the metadata upload.

### 3.6 The products that read the Q input as one matrix

**Objective.** One product where there are four.

**Build.**
- Stack `wq`, `attn_rel`, `attn_sink_proj` and `wqkv_gate` at load into one weight and view the four results, as the fork already does for the merged gate and up product under the tensor split. TokenSpeed runs gate and sink as rows of the query projection.
- Only where the four tensors share a quantization type in the pack (check the Q3_K_M pack; they may not).
- On sliding layers K and V read the same input and can join. On global layers they cannot.

**Exactness.** The same weights and the same quantized input.

**Gain bound.** In the Nov 19 trace the dense few-row products were 304 launches and 2.70 ms a step a card, with 0.54 ms of quantize, against roughly 0.3 to 0.5 ms of weight bytes by my estimate (inferred; the pack's types for these tensors are not read). They are bound by fixed cost. Three fewer launches a layer is 144 a step; at the 3.3 us fixed cost the fork recorded per product that is about 0.5 ms.

### 3.7 The two F32 products

**Bias table (inferred to be `gemmSN_TN_kernel`, 0.475 ms a step a card, plus `k_compute_batched_ptrs`).** The product writes 1,024 x 48 heads x 23 rows of F32 a layer, which attention then reads. Only the near region uses it. Compute the bias inside the near kernel as the 16-term dot of the table row with the query's relative feature, or build the table in the attention prologue. The product, its output and its read go away.

**Router (inferred to be `Kernel2` 0.371 ms plus `splitKreduce_kernel` 0.123 ms).** Move the 4,608 to 384 product into the fused top-k kernel, which already holds sigmoid, bias, selection and weights. meldavy measured cuBLAS at these row counts well below memory bandwidth on this card and replaced it, router included. The fork's review also found about ten layers a step declining the top-k fusion; the same trace names them.

**Exactness.** F32 with a different summation order; fixture tolerance.

### 3.8 Experts: what the architecture gives and does not

- **The kernel.** By the fork's own byte count (`TORAD.md` row 342: 556 MB a card a layer under uniform routing) and the probe's 102 distinct experts, a step reads about 18.6 GB of expert weights a card, 10 to 12 ms of pure reads, against 12.8 ms measured. Reuse between routes exists only where routes share an expert, and the Q3_K block-reuse attempt measured -7.5%. M1 settles it with one number: near 70 distinct experts a layer the ring pays, near 100 it does not.
- **Routing weights.** The eight weights of a token are normalized by their sum and known before any expert runs. If M6 shows the smallest weights are small, skipping those experts cuts expert bytes in proportion. It changes the output, so it stands behind the KL capture like 3.4.
- **Nothing else.** The native NVFP4 experts are 4.5 bits a weight against Q3_K's 3.4, so the native format is more bytes at 23 rows, where bytes are the cost.

### 3.9 Smaller

- The depth mix (`attn_res_mix_4608`, 0.49 ms a step a card) takes raw sources and an epsilon, so it appears to normalize every finished block summary again in each of the 96 mixes (inferred from the op's signature). A finished summary never changes: normalize it once when its block closes.
- Tau multiplies Q and the relative feature in global layers. Fold it into the stacked product of 3.6 or the attention prologue.

---

## 4. Order

1. The trace of the combined arm, read for M1, M3, M5 and M7. M2, M4 and M6 are host-side reads on the same box.
2. 3.1 and 3.3 together: the far path with run addressing. It is exact, and 3.2 and 3.4 are built on it.
3. 3.4, both candidates in the tile kernel, gated by KL. It is the only change that lowers the floor for private cells.
4. 3.2, with the group size from a sweep.
5. 3.7, then 3.6 if the pack's types allow it.
6. 3.5.
7. 3.8 only as M1 and M6 direct.

If the remaining global attention is bandwidth-bound, 3.2 and 3.4 together take it from about 18 ms to about 3.9 ms. With everything else unchanged the step goes from about 54 to about 40 ms, about 570 tok/s. That is a bound from byte arithmetic, not an estimate of what will be measured, and the other items are not added to it because their numbers come from an older trace of a different workload.

## 5. Not covered here

- Prefill, which is most of this workload's wall time.
- The DSpark draft under tensor-4.
- Synchronization structure, L2 warming, the ring redesign and the layer and conv fusions, which are campaign rows already.

## 6. Open questions

- Does the reference model use zero beyond the extent, as the engine does? The GGUF table is `{16, 1,024}` and the engine pads with zero; the architecture chapter's wording differs.
- How much of the 18 ms is bytes and how much arithmetic (M3)? It sets the order of 3.2 and 3.4.
- Are a session's cells contiguous after restores and slot reuse (M7)?
- Does E4M3's three-bit mantissa hold on long shared context, where vLLM saw Inkling 1 lose accuracy on long derivations?
- Real traffic shares less than this benchmark does. With little shared prefix, 3.2 gives little and 3.4 gives the same.

## Sources

- vLLM, Inkling 2 Small day-0 support: https://vllm.ai/blog/2026-11-12-inkling2-small
- SGLang and Miles, day-0 support for Inkling 2: https://www.sglang.io/blog/inkling-2-small-day0-support
- TokenSpeed, Inkling 2 Small at day 0: https://lightseek.org/blog/tokenspeed-inkling-2.html and commit c485a46: https://github.com/lightseekorg/tokenspeed/commit/c485a46a13641db6a1a5f95154cb9202fd657a81
- meldavy, Inkling 2 Small on 3x RTX PRO 6000 (sm120) at TP3: https://github.com/meldavy/inkling2-small-sm120-tp3
- vLLM RFC 54704, fp8 KV cache for Inkling: https://github.com/vllm-project/vllm/issues/54704
- ds4-dfm-rs, Inkling-2-Small VQ: https://github.com/Baekpica/ds4-dfm-rs/blob/main/docs/inkling2-vq.md
- ricardokevins, Inkling 深读: https://ricardokevins.github.io/notes/tech-analysis/thinking-machines-inkling-open-weights-model.html
- 余广坝, Inkling 2 Small 混合精度之外: https://yuxiaoba.github.io/tech/zh/vllm-inkling2-small-day0/
- OrcaRouter, Inkling 2 Mini 与 Inkling 2 Small: https://www.orcarouter.ai/zh-CN/blog/inkling-2-mini-vs-inkling-2-small
- Lei Mao, tensor core MMA peak performance: https://leimao.github.io/blog/Benchmarking-NVIDIA-Tensor-Core-MMA-Peak-Performances
- Code: `src/models/inkling2.cpp`, `src/llama-kv-cache.cpp`, `ggml/src/ggml-cuda/fattn-rel-mma.cuh`, `ggml/src/ggml-cuda/fattn-rel-decode.cuh`
- Reports on torad-server: `reports/inkling2-tensor4-inefficiency-review.md`, `reports/inkling2-tensor4-implementation.md`, `.campaign/ledger.toml`; the tensor-4 seat's session for the arm results
- Fork docs: `docs/torad/research/inkling-2-small/02-architecture.md`, `12-four-card-serving.md`, `data/four-card-decode-kernels-20261119.csv`, `TORAD.md` row 342

---

## 7. Review, 2026-11-22 (eli subagent, code-checked)

Every byte count above recomputed cleanly (49.4 GB re-read against 13.96 GB read-once, 201 MB shared prefix a layer, the 0.97 GB sliding floor, 184 routes -> about 147 distinct experts, 54 -> 40 ms -> about 570 tok/s), and the code claims of 1.2 verified at their lines. The amendments, most valuable first:

1. **3.4 must name its guard and its store path.** `fattn_rel_mma_shapes` requires F16 K (`fattn-rel-mma.cuh:40`) and the dispatch sends every other cache type to the scalar one-cell kernel (`fattn.cu:1463-1466`). A one-byte cache needs the guard extended, a store-side cast kernel (prefill writes the cache for 14 of 15 minutes; restores and slot reuse rewrite cells) and a snapshot-format migration. Without them 3.4 is decode-only and falls to the slow kernel on the first shape the served path does not use.
2. **Run 3.4 first or in parallel with 3.1/3.3, not third.** The tile staging it parameterizes exists today (`staged_k`/`staged_v`, `fattn-rel-mma.cuh:81`), so it depends on neither. Halving the bytes also halves the re-read bytes (49.4 -> 24.7 GB by count), so it attacks the 18 ms directly. Its decider (M4) is a host-side read available now and its gate, the KL capture, is the longest-lead item.
3. **3.2's bound double-counts query-first.** Query-first already serves part of the re-reads from L2 (49.4 GB in 18 ms exceeds DRAM bandwidth), so 3.2 is the strong form of a reuse the kernel gets opportunistically. Re-measure its bound after 3.4 lands, and run the group-size sweep with query-first on.
4. **3.1's near/far split has a multi-run edge and two dispatch sites.** For a just-forked session the near window spans the shared-prefix run and the private-tail run; the prefix kernel must subtract exactly that window. The mode must be added in the model-side `classify_mask` (`inkling2.cpp:193-211`) and in the backend guards (`fattn.cu:1459-1466`, `fattn-rel-mma.cuh:34-48`) - the classifier is a co-equal work item.
5. **3.3 is not "the same cells in the same order" bit for bit.** The online-softmax rescale runs per 16-cell tile; whole-tile bulk copies change the cadence. Keep 16-cell tiles for bit equality at a copy-efficiency cost, or take larger tiles and gate on the KL capture like 3.1/3.2.
6. **3.6 undercounts.** On the 40 sliding layers six products read the same input (wq, wk, wv, attn_rel, attn_sink_proj, wqkv_gate): about 224 launches a step, about 0.74 ms at 3.3 us, plus quantize launches. Read the pack's tensor types first; a type-split stack returns some launches.
7. **3.8's floor is efficiency-adjusted, and the shard imbalance is unaddressed.** "10 to 12 ms of pure reads" assumes near-peak bandwidth; the fork measured expert reads at about 65-80% of peak (TORAD `8e13bd73f`), which is consistent with the 12.8 ms. The 3.74 ms expert-shard imbalance has no entry; adopt the untried per-layer rotation or reject it with a reason.
8. **The parity gate needs a number and a budget.** Predeclare the KL threshold before the capture (the drift budget on record is a token spread, not a KL), budget two capture arms (E4M3 and Q8_0) in the runbook, and run at the benchmark's 262K depth, where E4M3's three-bit mantissa question lives.
9. **Add a device-capability preflight.** One call reading `l2CacheSize`, `sharedMemPerBlockOptin` and `persistingL2CacheMaxSize` (A8), and sustained DRAM bandwidth under the four-card power budget, before the 1.8 TB/s floors and the 128 MB L2 are used again.
10. **3.5's ring needs monotone positions, not just contiguous cells.** Extend M7 to assert per-stream position monotonicity, restores and forks included.
11. **Citation rot:** "TORAD.md row 342" does not resolve in the current table; re-cite by commit hash.
12. **The combined-arm trace is a re-ranking gate, not just a measurement.** The idle and gather shares are stock-trace numbers; qgre's crossing (two synchronization changes, +14.5% with no bandwidth change) says the non-attention rocks may be bigger than the trace suggests. Re-derive the 570 bound from the trace before 3.2's sweep is designed. Every kernel item must also name the guard and classifier it updates: the most likely failure of the whole plan is a finished kernel the dispatcher never sends work to.

**Techniques (doc 16).** Only the speculative bound can pass the 570 bound, to about 600-850 tok/s depending on acceptance; its binding constraint at 23 sessions is expert-byte saturation (about 102 distinct experts at 23 rows against about 330 at 92), not the verify kernel. KIVI's asymmetry is 3.4's designated fallback if E4M3 fails the KL gate on long context. SmoothQuant's migration is already applied by the architecture (K clipped to 22.63, V normalized to RMS 1) and is the reason 3.4 works. GPTQ/AWQ do not move tok/s at a fixed pack. llama.cpp's MMQ shape is the body of the `inkling-optimize-moe-shapes` row (the Blackwell table has no K-quant entry). L2 carveout is 0-6% class and conflicts with the fork's own finding that a flat half-L2 budget took DRAM from the experts. Tactic tables and lab constants add nothing the sweeps do not.

**Verdict.** The order is right in its bones. Three amendments: promote 3.4 to first-or-parallel, make the combined-arm trace an explicit re-ranking gate, and attach to every kernel item the guard and classifier it must update.

---

## 8. Pack facts, read from the GGUF header (2026-11-22, spot-checked)

File: `Inkling-2-Small-abliterated-e4853745-Q3_K_M-imatrix.gguf`, 283,538,490,336 bytes, 1,092 tensors, file type 12 (Q3_K_M).

- **Types.** `attn_q`, `attn_k`, `attn_rel`, `attn_sink_proj` and `attn_gate` (the plan's `wqkv_gate`) are Q3_K in all 48 layers. `attn_v` is Q5_K in layers 0-1 and Q4_K in layers 2-47. `attn_output` is Q4_K, the router `ffn_gate_inp` is F32, `ffn_gate_up_exps` is Q3_K, `ffn_down_exps` is Q5_K in layers 0-2 and Q4_K in layers 3-47. Bytes by type: Q3_K 169.9 GB, Q4_K 104.1 GB, Q5_K 8.4 GB, Q6_K 0.76 GB, F32 0.36 GB; no F16 or BF16 tensor.
- **3.6 is viable.** The four products that read the Q input share Q3_K in every layer, and `attn_k` joins them on the 40 sliding layers; `attn_v` stays separate (Q4_K and Q5_K use a different activation layout in the few-row path). That is about 184 fewer product launches a step (160 sliding, 24 global), about 0.6 ms at 3.3 us a product. Row: `inkling-optimize-q-input-stack`.
- **Expert shard bytes.** One expert is 9.12 MB of gate/up plus 5.97 MB of down (7.30 MB in layers 0-2); a quarter is 3.77 MB a card in layers 3-47 (4.10 MB in layers 0-2), so the plan's 3.8 MB holds. The down projection's 2,304 inputs are 9 superblocks of 256, which four cards cannot split evenly: shards of 2, 2, 2 and 3 superblocks, so per-card expert bytes run from about 3.6 to 4.3 MB. This is the 3/2/2/2 imbalance the stock trace measured, in bytes.
- **The pack's importance matrix is borrowed.** `quantize.imatrix.source = stock-bartowski`: computed on the stock model with calibration set `thinkingmachines_Inkling-2-Small-calibration-v6.txt` (590 chunks, 496 entries), not on the abliterated weights and not on a coding-agent workload; `quantize.imatrix.unobserved_expert_rows = 12`. Addendum item 4's cheapest lever is therefore not GPTQ or AWQ but an importance matrix computed on the abliterated model over the served traffic (Claude Code traces). It could hold equal KL at a smaller tier; like all of item 4 it waits for M1 (are the experts byte-bound) and for attention to come down.
