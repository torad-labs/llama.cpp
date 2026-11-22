# 21 · Architecture and routing, read from the weights

*Wrote Eli (weight-read seat), 2026-11-22. Every number below is computed from
`Inkling-2-Small-abliterated-e4853745-Q3_K_M-imatrix.gguf`
(283,538,490,336 bytes, 1,092 tensors — byte length and tensor count match the head pin),
read once on the rented four-card box over CPU mmap, or from this repo's own kernel sources.
Dequantizers were self-tested bit-for-bit against scalar transcriptions of
`ggml-quants.c`'s reference loops before use (`research_notes/inkling2-weight-read.py`,
golden blocks in `golden-selftest.npz`). Nothing here ran on a GPU.*

## 1. The surprise, stated plainly

**The router gate rows are unit-length by construction, except exactly eleven per layer,
in every one of the 48 layers, at norm ≈ 0.686 — while the experts themselves carry no
duplicate structure at all: within a layer their sampled weight directions are mutually
near-orthogonal (mean cosine 0.0004, max |cos| 0.228 across 73,728 expert pairs × 48
layers, zero pairs above 0.99), and the same expert index has uncorrelated weights across
layers (cross-layer fingerprint cosine mean 0.0004).**

Three numbers behind it:

1. **Router row norms:** 373 of 384 rows sit at 1.00000 ± 0.00013; 11 rows sit at
   0.6863 ± 0.001 — the *same count* and *same target norm* in every layer, though which
   expert IDs they belong to differs layer to layer (`routers.npz`, computed from the F32
   `ffn_gate_inp` tensors read exactly).
2. **Router cone:** mean inter-expert cosine between router rows is +0.19..0.33 depending
   on layer (max 0.90); random unit vectors in R^4608 would average ~0.015. The logit
   space spans as few as 8 effective dimensions by participation ratio in some layers.
3. **Expert orthogonality:** within a layer, expert gate/up weight fingerprints are
   orthogonal enough that no two experts could be confused even approximately — there is
   nothing to merge and no redundant bank to drop.

What this means for serving:

- There is **no amplitude prior** in routing: with all rows ≈ unit length, selection
  depends only on direction, so a quantized or reordered router cannot shift rankings via
  row-scale drift — but it also means the router's *output margin structure* is entirely
  carried by the narrow cone, making router-path precision more consequential than its
  byte share suggests (§5, precision map).
- The eleven short rows per layer look like a deliberate per-layer construction (the
  constant count/norm is too precise for training noise). Stock-vs-ablated provenance was
  probed with ranged upstream reads and answered in **§11**: they are a fingerprint of the
  derivative's own production pipeline, absent from stock.
- No duplicate-expert merging lever exists in this pack. Whatever whole-expert placement
  wins must win through routing statistics, not similarity structure (§3).

## 2. What was read, and how

| Step | Detail |
| --- | --- |
| Locate pack | Only extant copy: `/workspace/rig/dspark/pack/model/…Q3_K_M-imatrix.gguf` on vast box 61473783 (4× RTX PRO 6000 S). buildbox, mini-search and both Macs hold only Mini variants and the stock Bartowski imatrix. Marcos authorized opening the box to read weights, then stopping it. |
| Container parse | GGUF v3 header, **kv value-type field stored as u32** (verified byte-by-byte against ggml.h's u8 spec — this pack is nonstandard there); tensor table standard, `ne[]` lowest-index first. |
| Types | `ggml.h` canonical ids: gate/up = Q3_K (id 11) all 48 layers; down = Q5_K (13) layers 0–2, Q4_K (12) layers 3–47; routers F32 (0). A first-pass type table shifted by one produced NaN fingerprints — regenerated after fix (`inkling2-weight-stage3.py`). Doc 15 §8's per-layer list matches the corrected reading exactly. |
| Exact geometry | One expert = 9.12 MB gate/up + 5.98 MB down in L3+ (15.10 MB total), 16.42 MB in L0–2; spans verified against gap-to-next-tensor offsets for every layer. |
| Self-tests | Vectorized batch dequant vs scalar oracle transcribed from `dequantize_row_q3_K/q4_K/q5_K`: EQUAL on random bytes and on real file blocks (max diff 0.0). |

## 3. Routing statistics and whole-expert placement

Existing evidence (measured serving traces, not weights): MoE kernel cost scales near-
linearly with distinct experts touched (~137 µs at 16, ~633 µs at 150+;
`rtx5070ti-experts-23.csv`); a route probe recorded median **102 distinct experts per
layer-step** where uniform routing over 184 draws predicts ~147.

Weight-level additions this study contributes:

- Expert weight norms are essentially flat across experts (fingerprint magnitude medians
  differ by <1% between experts) — there is no "big" expert class whose residency would
  buy more than another's.
- Expert identities do NOT correlate across layers (§1), so a per-layer hot set cannot be
  amortized into a global one; any residency policy must be per-layer keyed, which cuts
  its bookkeeping value sharply.
- Router logits are dominated by ~8–25 effective directions (participation ratio range
  8.2–81.8 over the 48 layers). A load-time bake of the router as `[top-r PCA] × x`
  products is structurally available: it would fold the F32 SGEMM trio
  (`gemmSN_TN_kernel` + `Kernel2` + `mul_mat_q`, ≈1.20 ms/step/card combined from the
  four-card CSV) down to a small GEMV against r≈16..24 directions with bounded ranking
  error, gateable by an exact-top-k match rate on captured routes.
- **Verdict on placement:** whole-expert placement (96 experts/card, `LLAMA_INKLING2_EXPERT_SPLIT=1`)
  remains the right default given orthogonality + near-linear cost in distinct experts:
  routing locality (102 ≪ 147 uniform) already means each card touches far fewer experts
  than sharding-width would; the untried A/B should still run once because the down
  projection's nine-superblock split makes width-sharding byte-asymmetric (shards of
  2/2/2/3 superblocks), but nothing in the weights adds a reason beyond what serving
  traces already gave.

Measured from the assembled `expert-blocks.npz` (scale-level proxies over **every**
block of both expert stacks; method and limits in §10):

| Quantity | Gate/up (Q3_K) | Down (L0–2 Q5_K / L3+ Q4_K) |
| --- | --- | --- |
| top-16 experts' share of Σd²·(Σscale−32)² | **4.2 %** mean | **4.3 %** mean |
| top-32 share | 8.4 % | 8.5 % |
| top-64 share | 16.7 % | 16.8 % |
| Gini coefficient of per-expert energy | 0.0012 mean | 0.0031 mean |

Uniform to a degree that removes the weight-side argument for ANY hot-set residency:
no expert class exists in the bytes.

Cross-layer stability of per-expert energy vectors: mean off-diagonal correlation
between layers **−0.0009** (gate/up), i.e. an expert heavy in one layer predicts nothing
about another layer's weights; best pair r = +0.05. Router row-norm vs expert
energy-weighted norm within a layer: r = +0.008 — even the eleven quiet-router outliers
carry normal-size expert weights.

Zero-code fraction (share of dequantized values exactly 0 after quantization, sampled
exactly over strided rows): **20.63 % mean**, range 20.47–21.17 % across the 17 layers
measured before the sweep was cut (L0–L16, both down-quant regimes); first six layers at
stride 16, remainder at stride 64, sampling error ≈ ±0.05 pp on the fraction. The engine
cannot skip these zeros' bandwidth today (Q3_K has no run coding), but any future
value-skip fast path inside `vec_dot` starts from this floor.

## 4. Hot-set / residency

Hotness lives in activations, not weights: expert scale-energy is uniform to Gini
≈0.001 (top-16 of 384 hold 4.2%), per-expert energy decorrelates across layers
(r̄ = −0.0009), and router norms are flat by construction. A **hot-set residency policy
is NOT justified by weights** — only by route-frequency evidence at depth, which does
not exist anywhere yet: no capture holds per-route IDs across context positions (the M1
probe recorded counts only; drafter-g7 walk data carries draft, not target, routing).
Recorded as a gap rather than extrapolated. Cheapest decider stays the planned trace at
the next four-card window (M1 line): per-layer top-expert histograms over ~1000 decode
steps at shallow AND deep context before any LRU design.

## 5. Precision map (storage dtype vs compute accumulator)

Read straight from this fork's kernel sources + measured time shares from
`four-card-decode-kernels-20261119.csv` (23 sessions, ms/step/card). Ranked by
bytes-and-microseconds, waste column first where it exists.

| Subsystem | Storage dtype (as served) | Compute path & accumulator (source) | ms/step/card | Candidate |
| --- | --- | --- | --- | --- |
| MoE gate/up+down GEMV | Q3_K / Q4_K·Q5_K imatrix pack | y re-quantized q8_1 in-kernel (`mmvq.cu:833` `float tmp[][]`, `fuse-moe.cu:94` `float acc`) — f32 accumulation throughout | 14.22 (moe vec) | up-cast not needed; **already wide** |
| Owner-gather wire | F32 values → **BF16 wire** (`allreduce.cu:116`, BF16 round-trip comments at :391,:1158) | f32 accumulate at owners | 4.10 | **down-cast candidate**: GLM serve sets `GGML_CUDA_AR_WIRE_Q8=1`; Inkling does not. The switch exists today; needs only a serving-profile change + quality capture, no kernel work |
| Router product | **F32 weights + F32 input**, cuBLAS SGEMM (`gemmSN_TN_kernel` .475 + `Kernel2` .371 + `mul_mat_q` .357) | f32 | ≈1.20 | down-cast to BF16 inputs bounded by architecture only if activation range captured first (clip thresholds bound qk/rel paths, NOT router input); safer: §3's PCA bake |
| Relative attn decode rows | KV pool F16; sinks f32 (`fattn-rel-decode.cuh:269`) | softmax+accum all f32 (`:120` `float query_value…sum`) | 3.63 (+merge .136) | one-byte K/V = plan item 3.4 (needs store-side cast + snapshot migration — already scoped there, not restated here) |
| Rel mma rows (global layers) | Q/K f16 staged (`staged_k/v`) | mma f32 accum | 1.63 | same as above |
| Few-row MMQ dense stack | Q3_K/Q4_K attn weights; shared q8_1 inputs | `mmq-few.cu` half2 scale loads → f32 math; f32 acc | 2.71 (+quantize .54) | quantize calls are the dedup'd set keyed on src tensor (`ggml-cuda.cu:10189..10290`) — residual conversion churn is counted in §6 instead of re-folded here |
| Norms / GLU / conv / res-mix / top-k / sort | f32 everywhere | f32 | ≈2.0 combined | none worth it at these sizes |

Down-cast waste summary: the engine's *arithmetic* is already F32-accumulated end to
end; the two real byte-reductions available without new kernels are (a) the AR wire to
Q8 with a measured precedent on GLM and (b) KV pool narrowing per plan item 3.4.
The L2 clip thresholds (`qk_l2_clip_threshold` 22.627, rel 8.0) bound value ranges on
the attention paths they name; they do **not** cover router or MoE activations, so any
narrowing there waits on captured ranges.

Up-cast payoffs: no narrow accumulator found in any hot kernel — `mmvq`, `fuse-moe`,
both fattn-rel paths, norms and conv all accumulate f32. The only half-width math still
inside hot loops is scale handling (`half2 ds/dm` loads in `mmq-few.cu:125..131,
240`), which converts per block and cannot dominate at 15 µs-class kernels.

## 10. Method limits, stated

Scale-energy proxies count d²·Σ(scale−32)² per K-quant block (down adds the mn²·Σmin²
term). They rank experts by their stored spread energy exactly as quantized; they are
NOT full Frobenius norms of the dequantized tensors (block codes add a bounded,
data-dependent term treated as uniform across blocks within a tensor family). Zero-code
fractions are exact counts over sampled rows (stride documented above), not whole-tensor
counts. Router statistics are exact F32 reads. All comparisons between layers hold
within the same tensor family, which is where every decision above lives.

## 6. Folded-arithmetic census

Candidates ranked by evidence bar (what repeats × why constant under structure × where
it folds × µs saved from CSV × GLM-green check):

1. **Per-device dirty-range delta upload of window cells.** Every microbatch re-uploads
   cell positions over the padded extent from host data. What recurs: position/mask
   uploads sized to the whole padded window each step; constant because slot geometry is
   static across steps within a session. Fold locus: engine upload path keyed on
   per-device dirty ranges (delta-only). Saves: bounded by `pad_f32`+`cpy_scalar`
   ≈0.53 ms/step/card plus hidden latency in the 33 graph-segment waits. GLM: copy-path
   only → green by construction. *Attribution gap:* exact bytes need one trace with
   upload-size counters; flagged, not assumed.
2. **Final-layer expert results computed for never-selected experts in captured prompt
   graphs.** A captured prefill reduces ALL 384 experts including those top-k later
   discards. Constant under structure: during long-prefill replays the route set varies,
   but the kernel shape is baked at capture. Fold: index-driven gating inside the fused
   MoE chain (route-aware early-out) or smaller capture tiles per expert bucket. Save:
   up to ~30% of the 14.22 ms expert term on prompt-heavy steps (measured bound from
   M1's distinct-expert counts: 102 of 384 active ⇒ up to 73% dead work if the captured
   shape covers all). GLM shared-code risk: touches decode/prefill fusion both models
   run — needs a model-guarded switch, default off, like the existing few-row legacy.
3. **Router layout re-read every step.** The F32 router tensor keeps its cuBLAS-hostile
   layout and gets re-read each microbatch; a load-time swizzle into TC-friendly form
   (or the §3 PCA bake) removes the repeated transpose-shaped access. Saves within the
   ≈1.20 ms trio; exact split between layout tax and arithmetic needs one Nsight window.
4. **`quantize_mmq_few_q8_1` operand recurrence across tokens.** 304 calls/step
   (0.535 ms + `quantize_q8_1` .251 ms): inputs are per-token activations so values do
   NOT repeat, but *which tensors need conversion* is static per graph — the remaining
   lever is folding these launches into producers at capture time (graph-level fusion),
   not caching payloads. The few-shared machinery already dedups across products sharing
   a src (`ggml-cuda.cu:10189+`); what remains is launch-count, ~0.79 ms class total.

## 7. The 256k serving shape, priced first-class

Constants from `model-facts.json` (global KV f16 = 32,768 B/token all cards =
8,192 B/token/card; sliding floor 0.15625 GiB/sequence all cards; conv state
6.1875 MiB/seq f32; weights 70.9 GB/card measured on this pack = 283.5 GB / 4;
planning reserve 8 GiB/card). Pool cells today: 3,014,656 F16 ⇒ at 23 slots caps
~131 k tokens/slot — hence the coordinator's flag that 256 k needs fewer slots or a
bigger pool.

Global-KV per card = S · 262,144 · 8,192 B = **S × 1.994 GiB** (f16).
Usable after weights + reserve ≈ 95.6 − 70.9 − 8 = **16.7 GiB/card** for everything else.

| Slots | KV dtype | Global KV/card | Window+conv/card | Verdict vs 16.7 GiB |
| --- | --- | --- | --- | --- |
| 23 | F16 | 45.9 GiB | 0.93 GiB | no |
| 12 | F16 | 23.9 | 0.49 | no |
| 8 | F16 | 16.0 | 0.33 | marginal ✗ |
| 23 | Q8_0 (17,408 B/tok all cards) | 24.4 | 0.49 | no |
| 14 | Q8_0 | 14.9 | 0.30 | yes — 14 sessions |
| 23 | Q4_0 (9,216 B/tok) | 13.0 | 0.26 | yes — but engine's decode path for q4 KV needs plan item 3.4's kernel extension; quality gate required |

Profile knobs that move each term: `--slot-context` (advertised context), the KV cache
type flags (`-ctk/-ctv` equivalents in serve.py profiles), `GGML_CUDA_FATTN_REL_DECODE_Q4`
(already landed under a switch per doc 18 §2 / kv-q40 report), and doc 19's prefix-KV
artifact to cut restore traffic at slot churn. At 256 k history, shared-prefix attention
re-reads grow linearly with depth, so ranking by *total session cost* promotes:
(1) AR wire → Q8 (§5a), (2) prompt-graph route-aware expert gating (§6.2 — its saving
scales WITH prefill length), (3) one-byte KV via item 3.4, over any decode-tok/s-only win.

## 8. Regime boundary interaction

`log_scaling_n_floor` 128,000 / alpha 0.1 marks where position scaling changes slope.
Weight-side, nothing in the read interacts with it directly: the relative bias table is
looked up per distance (`rel_extent` 1024) and sinks are projections of the current token;
both live outside persisted state. The only weight-adjacent note: router logits' narrow
cone means softmax margins stay comparably tight deep into context, so routing noise
from narrower KV/quant paths would not be amplified by a sharper logit distribution —
the amplification risk sits in attention numerics past the scaling knee, which the KL
capture gates anyway.

## 9. Recommendation

1. **Run the whole-expert-placement A/B once** on the next four-card window
   (`LLAMA_INKLING2_EXPERT_SPLIT=1` vs tensor width-sharding), because it has never been
   measured and the down-superblock asymmetry gives it a concrete mechanism; expect
   routing locality to favor it.
2. **Do NOT build a hot-set residency policy yet** — weights give it no leg (flat expert
   norms, cross-layer decorrelation); wait for the M1 route histogram at depth.
3. **Take the AR-wire Q8 profile change + capture arm first among byte levers** (exists
   today, GLM precedent, 4.10 ms/step/card class).
4. **Fold the prompt-graph expert gating** (§6.2) as the highest-leverage new kernel
   under the hold — proposed diff only until Marcos releases the search-method freeze.
5. **Router**: prefer §3's PCA bake over naive BF16 downcast; gate both on exact-top-k
   match rate against captured routes.
6. ~~Open question worth ~340 MB of HF range reads~~ **answered in §11**: stock shows
   none of them; they belong to the derivative's own production pipeline.

## 11. Provenance of the eleven rows

*Added 2026-11-22 by the follow-up probe (`research_notes/inkling2-upstream-router-probe.py`):
CPU-only ranged HTTP reads against huggingface.co, one [384×4608] tensor held at a time,
artifacts under `~/.cache/inkling2-weightread/provenance/`. Upstream side is
`thinkingmachines/Inkling-2-Small @ main` (commit pinned at read time below); served side
is the cached exact F32 routers from §appendix. This answers §9.6.*

**Verdict: conversion-fingerprint of the derivative — not architecture, and not our
converter either.** Stock Inkling-2-Small has no trace of the pattern. The 528 rows were
made inside the production of `benthecarman/Inkling-2-Small-abliterated` rev e4853745
(doc 09: a fine-tune with a recorded "steering recipe"), and their membership is exactly
the map of which experts' routing priority that steering suppressed.

Branches stated before any fetch, so the experiment could fail honestly:

1. Stock shows the same count/norm with overlapping indices → **architecture**, abliteration ruled out.
2. Ablated-only pattern → **conversion fingerprint** plus the touched-expert map. ← measured here
3. Overlap beyond chance but short of identity → both contributions, shares to be stated.
4. Tensors unreachable → inconclusive-with-evidence-pending, blockers recorded.

### What was read

| Step | Detail |
| --- | --- |
| Format found | Upstream ships safetensors shards (24 × ~12–20 GB), NOT gguf; header = u64 length + JSON table, parsed from ≤256 KiB windows per shard. Router gates are `model.llm.layers.<L>.mlp.gate.weight`, bf16 [384, 4608] (`n_routed_experts=384` ✓ config). |
| Byte discipline | Ranged GETs only: header windows, each router tensor's own 3,538,944 B span, and later row-level 9 KB gapfills. Router payload total 169,869,312 B; nothing full-tensor was materialized host-side; results checkpointed per layer as fetched (`fetched-layers.jsonl`). |
| Commit pin | `X-Repo-Commit` at read time: `9d8efa6cf3c3c91953a8d3a…`; index/config/quant-config cached under `provenance/`. |
| Served side | `routers.npz` exact F32 from the served pack (§appendix), compacted once to norms + probe rows so no stage re-materializes it. |

### The numbers behind the verdict

1. **Stock has zero of them.** All 18,432 upstream router rows sit in one tight family:
   norm mean 70.3976 ± 0.3106 (min 64.4158, max 70.8420) — an unconstrained trained
   matrix's signature (~1.04·√4608). Rows below 0.9: **0 of 48 layers**, and no bimodal
   subpopulation anywhere (p0.1 percentile = 67.83). Served side: exactly 11 rows/layer ×
   48 = 528 at 0.68633 ± 0.00005.
2. **Same-index correspondence is perfect; directions are not.** Full 384×384 check on
   layer 23: every upstream row's best match is its own index (identity 384/384), matched
   cosine mean 0.850 spread [0.75, 0.87]; per-layer unit-sample cosines run 0.827–0.861
   across all 48 layers. The difference matrix is broad-spectrum (top singular value share
   3%, then ~1%), which is what trained drift looks like — a surgical projection would be
   rank-1 dominant. Confirms doc 09's "fine-tune of the base model" relation from weights
   alone.
3. **The eleven carry no selection signature.** Upstream norm ranks of those indices are
   flat (mean rank 179 vs chance 191; decile fractions ≈ chance). Drift magnitude AT those
   indices equals background (matched cosine 0.8502 vs 0.8521 at ordinary rows; per-layer
   gap −0.002 ± noise). Within the served pack they are not cone outliers either (within-set
   pairwise cosine ≈ within-unit average, sampled layers), and cross-layer set overlap is
   chance-level (mean 0.320 vs expected 0.315) with no recurring expert.

### What the bytes say about how they were made

- Served router values are **bit-exact on the fp16 lattice**: 99.967% of 3,847,680 sampled
  values survive an f32→fp16→f32 round trip unchanged, and median adjacent-value gaps are
  3.05e-5 = one fp16 ulp at |v|≈0.03. Stock routers are bf16 (100% by construction). So an
  fp16 step exists somewhere in the derivative's production path despite its card claiming
  "BF16 elsewhere" — a dtype fingerprint independent of everything above.
- Unit rows sit at 1.00000 ± 0.00013 and short rows at 0.68633 ± 0.00005, same count in
  every layer, arbitrary membership. That is the signature of a deterministic post-hoc rule
  applied to already-normalized rows: standardize all router rows to unit norm, then scale a
  chosen subset by a fixed factor α = 0.68633 (its complement √(1−α²) = 0.7272 would be the
  implied removed component if one insists on reading it as projection; plain projection
  cannot produce this, because it ties each touched row's norm to that row's own cosine and
  leaves untouched rows near their raw ~70 norms).
- Membership chosen from information outside router weights — activation/usage-style
  attribution is the natural candidate. The exact rule is recorded in the derivative repo's
  `trial.json`; that repository now gates anonymous reads (401 as of today), so the rule
  itself stays unread and the claim stops where the evidence does.

### What this changes for serving

- There is **no designed routing prior** to exploit in placement or gating; §3's
  amplitude-prior note stands, and any hot-set idea still waits on the M1 route histogram.
- The eleven sets ARE the touched-expert map: those experts' gate logits were deliberately
  reduced by ≈0.686 relative scale, making them lose top-8 contests more often. Treat them
  as low-priority lanes for expert residency/placement until measured otherwise at depth.
- Precision-map consequence (§5/§6.3): served routers carry only fp16 information content
  though stored F32 — an f16 router path loses nothing measurable versus today's F32 SGEMM
  reads and pairs with the layout-swizzle lever already proposed there.

### Per-layer table

Upstream short-row indices are empty in all 48 layers (count column = 0). Ablated-side
eleven indices per layer (served pack, exact F32 read):

| L | upstream | ablated eleven |
| --- | --- | --- |
|  0 | 0 | 45, 49, 102, 160, 186, 195, 319, 348, 366, 371, 373 |
|  1 | 0 | 10, 22, 46, 71, 128, 142, 164, 199, 226, 235, 344 |
|  2 | 0 | 14, 49, 98, 111, 150, 197, 241, 271, 276, 367, 375 |
|  3 | 0 | 5, 90, 116, 203, 221, 247, 270, 287, 297, 344, 355 |
|  4 | 0 | 24, 91, 164, 217, 249, 274, 283, 286, 313, 339, 344 |
|  5 | 0 | 8, 39, 50, 64, 68, 124, 224, 243, 267, 327, 333 |
|  6 | 0 | 104, 105, 109, 137, 223, 260, 282, 292, 298, 348, 382 |
|  7 | 0 | 5, 27, 41, 127, 178, 223, 229, 269, 274, 301, 351 |
|  8 | 0 | 66, 68, 74, 79, 112, 124, 132, 180, 301, 307, 364 |
|  9 | 0 | 61, 75, 98, 183, 244, 256, 259, 270, 285, 322, 365 |
| 10 | 0 | 2, 45, 55, 132, 145, 146, 151, 245, 269, 314, 360 |
| 11 | 0 | 79, 98, 159, 201, 244, 258, 270, 289, 297, 316, 349 |
| 12 | 0 | 34, 85, 105, 114, 179, 241, 267, 331, 345, 363, 372 |
| 13 | 0 | 8, 44, 51, 144, 156, 199, 229, 264, 293, 347, 357 |
| 14 | 0 | 68, 89, 105, 122, 142, 174, 179, 201, 312, 364, 369 |
| 15 | 0 | 29, 54, 96, 101, 105, 117, 174, 192, 231, 245, 309 |
| 16 | 0 | 9, 10, 51, 85, 90, 112, 215, 243, 363, 367, 377 |
| 17 | 0 | 2, 7, 30, 67, 99, 100, 162, 168, 201, 334, 358 |
| 18 | 0 | 46, 68, 118, 185, 207, 209, 229, 260, 275, 310, 331 |
| 19 | 0 | 45, 81, 104, 193, 201, 232, 258, 318, 346, 370, 373 |
| 20 | 0 | 55, 88, 163, 210, 218, 224, 237, 299, 300, 310, 329 |
| 21 | 0 | 33, 64, 84, 158, 183, 189, 221, 227, 291, 348, 349 |
| 22 | 0 | 30, 42, 86, 88, 94, 156, 168, 299, 332, 343, 345 |
| 23 | 0 | 3, 5, 14, 70, 122, 144, 154, 156, 219, 284, 309 |
| 24 | 0 | 4, 106, 130, 182, 185, 231, 235, 309, 351, 354, 358 |
| 25 | 0 | 8, 9, 144, 164, 216, 237, 261, 289, 293, 349, 378 |
| 26 | 0 | 29, 67, 70, 76, 118, 126, 150, 193, 241, 267, 335 |
| 27 | 0 | 31, 38, 162, 173, 199, 212, 218, 283, 332, 357, 370 |
| 28 | 0 | 7, 15, 22, 34, 72, 78, 111, 133, 190, 279, 373 |
| 29 | 0 | 75, 95, 97, 156, 177, 210, 231, 321, 357, 372, 373 |
| 30 | 0 | 32, 46, 89, 101, 114, 125, 161, 326, 338, 357, 378 |
| 31 | 0 | 44, 88, 90, 107, 176, 242, 268, 278, 284, 318, 364 |
| 32 | 0 | 33, 38, 50, 51, 82, 134, 231, 255, 302, 360, 372 |
| 33 | 0 | 35, 76, 112, 154, 178, 196, 216, 225, 252, 305, 332 |
| 34 | 0 | 96, 97, 101, 134, 158, 181, 187, 275, 316, 340, 351 |
| 35 | 0 | 16, 66, 73, 116, 165, 173, 200, 281, 301, 325, 346 |
| 36 | 0 | 36, 83, 107, 120, 122, 186, 275, 316, 320, 345, 353 |
| 37 | 0 | 9, 60, 74, 81, 115, 150, 171, 216, 256, 292, 367 |
| 38 | 0 | 33, 43, 53, 69, 190, 211, 287, 326, 335, 336, 382 |
| 39 | 0 | 6, 34, 35, 72, 74, 142, 153, 183, 222, 257, 273 |
| 40 | 0 | 63, 73, 90, 224, 230, 279, 281, 339, 351, 354, 357 |
| 41 | 0 | 25, 28, 55, 64, 169, 237, 295, 316, 318, 357, 376 |
| 42 | 0 | 43, 63, 173, 175, 235, 249, 285, 293, 305, 308, 331 |
| 43 | 0 | 18, 49, 81, 89, 107, 108, 134, 136, 161, 320, 362 |
| 44 | 0 | 14, 22, 108, 109, 116, 135, 250, 314, 317, 342, 353 |
| 45 | 0 | 3, 13, 17, 44, 76, 98, 116, 133, 196, 275, 346 |
| 46 | 0 | 23, 83, 134, 168, 212, 236, 275, 307, 313, 320, 340 |
| 47 | 0 | 23, 59, 68, 125, 139, 153, 203, 319, 347, 357, 378 |

### Method limits, stated

Upstream was read once per tensor from byte ranges; its bf16→f32 cast is exact, so upstream
norms are exact up to bf16 storage granularity. No stock-vs-derivative comparison of EXPERT
FFN weights was attempted: experts are NVFP4 (L0–39) / MXFP8 (L40–47) in both checkpoints,
and a dequantizing drift study over ~390 GB is a separate probe. Correspondence checks ran
fully on layer 23 and through fixed seeded probe-row samples (24 unit rows/layer, same seed
both sides) elsewhere; the fp16-lattice census sampled 12 of 48 layers deterministically.

## Provenance appendix

Scripts (this repo): `research_notes/inkling2-weight-read.py`,
`inkling2-weight-stage2.py`, `inkling2-weight-stage3.py`, `zf_one.py`,
analyzer `inkling2-weight-analyze.py`. Artifacts cached outside the repo at
`~/.cache/inkling2-weightread/` (routers.npz 339.7 MB F32 exact, fingerprints.npz with
regenerated NaN-free down block, tensor-inventory.json, router-stats.json,
energy_{gateup,down}.npy over every expert, zero-frac-progress.json, assembled
expert-blocks.npz, gguf-metadata.json incl. layout-width note, golden-selftest.npz).

Run history for honesty: vast box 61473783 opened 07:44 UTC with Marcos's instruction to
open it only to read weights; long single-process jobs were killed twice by the host
with no traceback, so the reader was split into resumable stages persisting each output;
the per-layer zero-code sweep ran as one ssh process per layer from here and was cut at
17 measured layers once its range stabilized (both down regimes covered). Artifacts were
rsynced home and the box stopped immediately after, 09:08 UTC (~1.42 h billed; disk kept
for serving). GLM RTX PRO 6000 box never touched.
