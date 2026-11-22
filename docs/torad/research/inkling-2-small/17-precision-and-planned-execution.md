# 17. Two targets added 2026-11-22: precision and planned execution

Added to the plan of `15-kernels-shaped-to-the-architecture.md` at Marcos's direction: no part of the served
step carries more precision than the model needs, and the step runs as a plan made ahead of time rather than one the
host improvises card by card. Same setting: four RTX PRO 6000 (sm_120), 23 sessions of 262,144 tokens, Q3_K_M,
425.6 tok/s on the combined profile, a step of about 54 ms. Marking as in 15: verified (read in code or measured),
external, inferred.

## A. Precision: nothing wider than the model needs

Owned since 2026-11-22 by the session "Inkling 2 Small global KV cache" (local_4207d667), which works in the shared
checkout. The precision findings of the section D search were handed to it. This section stays as the plan's record.

**Facts.**
- The weights average about 3.4 bits (Q3_K_M mix, doc 15 section 8). The KV cache is F16, 16 bits, global and
  sliding layers alike (launcher, verified). Global attention is about 18 ms of the 54 ms step, and private cells are
  88% of its read-once bytes (doc 15 section 1.1): only fewer bytes a cell lowers that floor.
- Already built and off by default: one-byte global KV (q8_0, f8_e4m3) in the relative decode kernel and the
  run-addressed kernels (attn and runs pieces, reports in inkling-plan/reports). q8_0 and q4_0 correctness for global
  and sliding layers is proven on the synthetic fixture (campaign row inkling-optimize-kv-precision, done). The
  launcher offers f16 and q8_0 only (`--kv-cache`, `--swa-kv-cache`).
- Real-weight K/V statistics (Inkling 2 Mini, scripts/inkling2/kv_stats.py, illustrative): K relative error Q8_0 0.59%
  (0.51% with a static per-channel scale), E4M3 2.6%, Q4_0 9.5% (8.2% with the scale). Stored K sits at its L2 clip
  and V at RMS 1, so scale-free and per-channel formats both have bounded inputs.
- The reductions cross PCIe in BF16 (`--reduction-wire bf16`); a q8 wire exists (`--reduction-wire q8`,
  `GGML_CUDA_AR_WIRE_Q8`) and is off "until its quality leg": reductions feed the convolutions' persistent state.

**Targets, by bytes saved.**
1. KV: q8_0 for global and sliding layers as the default, then 4-bit for the global layers' far cells (q4_0, or e2m1
   with an E4M3 scale per 16) with a static per-channel K scale (addendum 16 item 1, the KIVI/SmoothQuant migration:
   K stored divided by s_c, Q multiplied by s_c in the prologue) and the near cells (last 1,024 positions) kept at
   8 bits. Read-once floor of global attention 7.7 ms -> about 3.9 (8-bit) -> about 2 ms (4-bit); the pool holds 2x
   or 4x the tokens in the same memory. Builds: the sliding kernels take q8_0 on their fast paths; a 4-bit element
   decode in fattn-rel-elem.cuh used by both relative kernels; the per-channel scale table made by a calibration pass
   from kv_stats and applied at store and in the prologue; launcher options for the types and switches.
2. Reduction wire BF16 -> q8: half the PCIe bytes of every reduction and gather, in decode and in prefill, where the
   96 reductions and gathers are estimated at about 80 ms of a 578 ms step (reports/inkling2-prefill-roofline.md).
3. Output head (Q6_K): a K-quant one tier lower is a small byte saving, read every step; last, and only if the KL
   capture finds it free. The router, bias tables and depth-mix summaries are small: keep F32 where routing stability
   needs it.

**Gate.** Approximate by design: the KL parity capture at 262K depth (reports/inkling2-tensor4-capture), budget taken
from the drift two stock runs show, for each arm: global q8_0, global 4-bit with and without the channel scale,
sliding q8_0, q8 wire. Sequence state (convolution tails) must round-trip under the q8 wire.

## B. Planned execution: one captured graph a step across the four cards

**Facts.**
- The meta backend already captures a whole evaluation, every device's subgraphs and every all-reduce, into one CUDA
  graph a device, replayed with one launch a device and kept per cgraph uid (ggml/src/ggml-backend-meta.cpp near 3214,
  `GGML_META_CAPTURE_GRAPHS`). It does so only when every all-reduce is capturable.
- `ggml_cuda_ar_capturable` returns `!use_copy_engine` (ggml/src/ggml-cuda/allreduce.cu near 2790): the served
  host-sliced route over the copy engines is never capturable, so capture is refused and the step runs as segments
  with the reductions driven live by the host. The stock trace counts about 196 segments a card a step, 4.0 ms in 33
  waits plus 4.3 ms in about 860 short gaps idle a step, and the graphs-off window spent about 16 ms a step in host
  `cudaEventSynchronize`, four a reduction (14-decode-stalls-and-knob-search.md).
- The capturable engine route measured slower than the host route on these PCIe boxes (481 against 496.7 tok/s), so
  switching routes is not the answer.
- The host route's last host-side step is its pacing wait. Main's eb249e0aa drops it (`GGML_CUDA_AR_SLOT_NO_SYNC=1`)
  and measured slower on two local cards: without the wait the receive started late, and "only the wait two back was
  fast". The owner-row route's `GGML_CUDA_AR_HOST_OWNER_STREAM_ORDERED` removed the same wait: +0.2% tok/s, p99
  -8 to -10 ms, one arm.

**Target: make the copy-engine route capturable, and capture the whole step.**
1. Hardware plan. The route's copy-engine copies (to and from pinned staging), its add kernels and the cross-card
   event edges become graph nodes. The two-back pacing that made the host route fast becomes a graph dependency
   between reductions, so it survives without the host in the loop. Copy directions and compute get planned streams
   rather than whatever the driver picks.
2. Software plan. The capture covers the whole decode step, all 48 layers, the reductions and the logits: one launch
   a card a step. Between steps the graph is updated in place, not captured again; shape stability (run addressing,
   doc 15 item 3.3: no per-step pool-sized inputs) keeps it reusable.
3. `ggml_cuda_ar_capturable` accepts the copy-engine route when the route runs without host waits.

**Gate.** Bit-identical: the same operations in the same order with the same reductions. Correctness through
test-backend-meta-split, the allreduce tests and test-llama-archs' Meta columns on logical ranks; the four-card group
window measures it. The known risk is pacing: if the graph's dependency between reductions does not reproduce the
two-back wait, the receives start late as in eb249e0aa. Bound: the host launches and live waits of about 196 segments
a card a step; the idle and host-sync shares above are the ceiling, not added to the other bounds.

## Order

The merges landed 2026-11-22: origin/main into draft-speed as `ea83c6dd8`, the review fixes as `ad51d800a` on top,
pushed. B, C and D are built on that sha, and the group window measures them with everything else. A belongs to the
KV-cache session, and its items are gated by the KL capture in that same window.

## C. Launch count: fuse until the step is a few hundred kernels, not three thousand

Marcos: the step does an insane amount of launches; fuse more, and it is fine to fork or copy kernels into new fused
ones rather than keep the generic ones tidy.

**Facts.** The four-card decode trace (data/four-card-decode-kernels-20261119.csv, per step per card) counts 3,215
kernel launches a step a card for 33.8 ms of kernel time. Graph capture (target B) removes their host cost; their
device-side gaps and their small-kernel inefficiency remain, so fewer kernels matters either way. Already landed
tonight: the Q-input products as one launch per layer (304 product launches -> about 120, their quantize launches
folded into the shared copy), the F32 router and bias table off cuBLAS, the top-k fusion on all 48 layers.

**What is left, grouped by what fuses (calls a step a card).**
1. The convolution chain, 4 sites a layer: cpy_scalar 143.6, concat_cont 143.6, short_conv_window 130.6, pad_f32
   96.7, k_set_rows 79.8, short_conv 60.9, concat_non_cont 54.9 (0.83 ms on its own). About 710 launches that are one
   stateful kernel a site: read the ring and the current rows, apply static and dynamic taps and the scaled residual,
   write the advanced ring (campaign row inkling-optimize-conv-kernel). Fusing the K and V sites of a layer, and the
   two hidden-stream sites with their neighbours, goes further.
2. The attention prologue: attn_res_mix 95.7, rms_norm 113.7, norm_clip 143.6, k_bin_bcast 123.7 (the tau and gate
   multiplies), convert 95.7. One kernel per layer from the depth mix through the input norms to the clipped Q/K/R and
   normed V (row inkling-optimize-layer-fusions); vLLM and TokenSpeed both run this as one launch.
3. The MoE chain: the router product, top-k, mmvq_sort_routes 92.7 and quantize_q8_1 95.7 before the experts; glu_fn
   95.7 and the weighted sum after. Router through sorted routes and the quantized input as one kernel; the GLU and
   weighted sum folded into the expert kernels' epilogues.
4. The reduction epilogue: ggml_cuda_ar_owner_gather 113.7 and ggml_cuda_ar_add 95.7 followed by the residual add
   and the next norm. Fold the add, residual and norm into one kernel after each reduction (vLLM fuses all-reduce,
   residual and RMS norm).

**Bound.** Groups 1-4 cover about 1,900 launches a step a card. Target: under 1,000 a step a card, bit-identical
where the arithmetic order is kept (most of the chain is elementwise), KL-gated where an accumulation order changes.
Forked kernels may duplicate code; they carry the fork's switch convention (LEGACY to the old chain) and ride the
group window's single A/B with everything else.

## D. Prefill, from a search of Chinese and English sources (2026-11-22)

Marcos asked for reductions where the step is compute-bound. **Fact first: almost nothing in this workload is.**
reports/inkling2-prefill-roofline.md (I, from verified step times) puts tensor-core work at peak rate at 34 ms of a 578 ms
1,024-row follower step (6%). The experts run at 99 int8 ops a weight byte against a ridge of 631, so their weight stream
bounds them. Decode is bandwidth, launch and sync bound. The ranked prefill step: experts 100 ms, collectives 80,
pool-scaled inputs 68, global attention 62, glue 45, dense/sliding 30, donor pass 20, host 20, and about 155 unexplained.
The compute-bound levers the search found (SageAttention INT8/FP8/FP4 attention, FP4 MMA) are precision work and went to
target A's owner. What follows reduces work or bytes without changing precision, ordered by expected gain.

**D1. Sequence-parallel prefill: run the glue on a quarter of the rows, and all-gather the MMQ input already quantized.**
- External: vLLM's Inkling 2 support (vllm PR 67325, zhidaily.com summary): each card does the convolution and norms for its
  own token shard, then all-gathers for the next GEMM. The 8K prefill step is 22% faster at TP4 and 26% at TP8, and it saves
  1.3-1.9 GiB a card. Tencent Hunyuan Hy3 (zhuanlan.zhihu.com/p/2053138680768943935) measured TP+SP with communication
  quantization at -30% TTFT at 16K and -25% at 32K. SGLang's LayerNorm-SP (draft PR 43048, author's own unaudited numbers):
  +15-18% TTFT from 32K to 235K. The vllm-ascend docs (Chinese) say where the gain really comes from: quantize after the
  norm, then all-gather, which halves the gather's bytes.
- Ours (I): today every card runs the post-reduction glue for all 1,024 rows (45 ms mid), and the gather half of the 96
  reductions crosses PCIe in BF16 (part of the 80 ms). Reduce-scatter instead, run the glue (residual, conv, mix, norm) on the
  owned share, quantize to q8_1 there, and all-gather q8_1 at about 1.06 bytes a value instead of 2. Moving q8_1 before the
  gather is bit-identical, because q8_1 blocks are local to 32 channels and MMQ quantizes the same values either way. A norm
  needs a full row: a row partition keeps it local but needs a 3-token conv halo, while a channel partition (the
  host-owner-rows route already gives each card its own conv channels in 23-row decode) keeps the conv rings where decode
  expects them but needs a 4-byte-per-row partial-sum exchange for the RMS. The builder picks, and the summation order of the
  RMS is the only arithmetic that may change. Estimate 40-55 ms of 578 (7-10%).

**D2. GQA-packed relative prefill attention: read K/V once for the six heads that share it.**
- Verified: served prefill runs global attention in the generic fattn-mma-f16, one Q head a tile, so K/V is re-read 6x
  (22.5 GB a layer L2 to SM, roofline row). fattn-rel-prefill.cuh (behind GGML_CUDA_FATTN_REL_PREFILL_F32, off in tensor-4)
  runs one warp, one head and 16 queries a block on WMMA. The decode kernel already packs (fattn-rel-mma.cuh:168,
  n_cols = group*gqa).
- External: vLLM's Inkling Gluon kernel packs the 6 query heads into one 64-row tile and reads KV once: 100K batch-1 decode
  15.1 -> 4.8 ms. SGLang's Inkling decode kernel splits KV over up to 64 CTAs and reads a compact distance-indexed bias
  table: 25 -> 9 us a layer at batch 1.
- Ours (I): a prefill tile of about 10 tokens x 6 heads in 64 rows, mma.sync with ldmatrix, and K/V tiles shared in shared
  memory by cp.async or TMA. Bit-identical per head when the KV order is kept. This is the one prefill part near its compute
  floor (23 ms at f16 peak against 47-100 estimated). Estimate 20-40 ms of 578. Pairs with A's int8/fp8 MMA if that lands.

**D3. Fused prologue and epilogues (confirms C2 and C4, and extends C2).**
- External: Tencent Hy3 fused RoPE+norm+Hadamard+quantize+KV store into one kernel (about 5x on that chain) and
  AllReduce+residual+RMSNorm (up to 1.68x over NCCL and FlashInfer). vLLM's fuse_allreduce_rms gives 5-20% end to end at low
  token counts. SGLang's Inkling: everything before attention is one fused prologue kernel, router sigmoid+bias+top-k is
  another, quantization is folded into the expert GEMMs, and nearly every launch uses PDL.
- Ours: C2's prologue also takes the KV store (k_set_rows, 79.8 calls a step a card), and C4 as written.

**D4. One persistent MoE kernel a layer (extends C3).**
- External: SGLang's MegaMoE for Inkling 2 (dispatch, both expert GEMMs, capped SwiGLU, combine in one DeepGEMM kernel) cut
  TTFT 20-27% against grouped GEMM, and DSpark decode time per token 17% at batch 64.
- Ours (I): gate/up, GLU and down as one persistent expert kernel a layer is C3's far end. In prefill it overlaps the phases
  of a stream-bound kernel, and in decode it removes launches.

**D5. Expert MMQ tile loads (only if the capture shows MMQ far below its stream floor).**
- External: llama.cpp discussion 28514 found stock NVFP4 MMQ on a 5090 with its tensor pipe 42% active, stalled on 4-byte
  cp.async because 36-byte blocks are only 4-byte aligned. TMA with mbarriers gave 1.16x.
- Ours (I): Q3_K blocks are 110 bytes and a 4,608 row is 1,980 bytes, 4-byte aligned. Padding rows to a 16-byte multiple at
  load (1,984 bytes) lets TMA bulk-copy whole tiles: the qgre lesson. It matters if part of the roofline's 155 ms
  unexplained is expert MMQ.

**D6. GEMM + reduce-scatter overlap (I; hardest).**
- External: Tencent's tile-level GEMM+ReduceScatter fusion is 1.68-1.81x over serial on prefill shapes (76-85% overlap).
- Ours (I): the attention output projection writes rows in tile order, so its reduction can start per finished tile. A MoE
  down-projection row is complete only after its 8 experts. Two-batch overlap is rejected because it re-reads the expert
  stream (58 ms floor).

**D7. Block-sparse prefill attention (approximate; last, after D2).**
- External: XAttention gives up to 13.5x attention at 256K, at 7% density at 128K, with RULER 128K at 72.3 against 76.9
  full. MInference drops to 54.1 at 128K. SpargeAttn (thu-ml) takes Llama 3.1 128K end to end from 52 s to under 30 s.
- Ours (I): the relative bias is zero beyond 1,023, so the far tiles are pure content attention and the near window stays
  dense. Agentic coding needs exact recall of far file contents, and the external tables lose the most at 128K and beyond,
  so the gate is the KL capture at 262K plus retrieval checks drawn from real sessions. Bound: the 62 ms global attention
  share, less what D2 saves.

**Checked here, no action.** SageAttention3 lost about 23% on a 110-SM card to a grid hardcoded at 170 (the 5090's
count; github.com/thu-ml/SageAttention/issues/378). Ours (V): no hardcoded 170 or 188 in ggml-cuda; persistent grids size
from `ggml_cuda_info().devices[].nsm` (mmq-few.cu:819, 899, 1220; mmvq-moe.cu:1033). Thresholds tuned on the 5090
(170 SMs) still meet the 188-SM cards for the first time in the group window.

**Sources that taught nothing new here:** linux.do and weisunit write-ups of four RTX PRO 6000 cards over PCIe (vLLM TP4 works
without NVLink; PCIe switch topology matters), lcz.me's dual PRO 6000 grouped expert prefill (2.6x from grouping, which our
MMQ already does).
