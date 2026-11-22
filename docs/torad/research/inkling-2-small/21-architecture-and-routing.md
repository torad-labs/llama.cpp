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
  constant count/norm is too precise for training noise). Whether they exist in the stock
  checkpoint — i.e. architecture vs abliteration artifact — is testable from HF range
  reads of both checkpoints' `ffn_gate_inp` tensors alone (~340 MB total) and is recorded
  as an open question; no copy of the stock model exists on any allowed machine.
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

*PENDING-TABLE-1: per-layer concentration stats from expert-blocks.npz (top-k energy shares, Gini, gate↔down correlation).*
*PENDING-TABLE-2: sampled zero-code fraction per layer.*
*PENDING: cross-layer expert-energy correlation matrix summary.*

## 4. Hot-set / residency

Hotness lives in activations, not weights: router structure is flat across experts, and
expert weight identity decorrelates across layers. Therefore a **hot-set residency policy
is NOT justified by weights alone** — it can only be justified by route-frequency evidence
at depth, which does not exist yet: no capture on any machine holds per-route IDs across
context positions (the M1 probe recorded counts only; drafter-g7 walk data carries draft,
not target, routing). Recorded as a gap rather than extrapolated. The cheapest first step
stays the planned trace at the next four-card window (M1 line): log per-layer top-expert
histograms for ~1000 decode steps at shallow AND deep context before any LRU design.

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
6. **Open question worth ~340 MB of HF range reads:** are the 11 short router rows per
   layer present in stock `thinkingmachines/Inkling-2-Small`, or an abliteration artifact?

## Provenance appendix

Scripts (this repo, uncommitted here): `research_notes/inkling2-weight-read.py`,
`inkling2-weight-stage2.py`, `inkling2-weight-stage3.py`, analyzer
`inkling2-weight-analyze.py`. Artifacts cached outside the repo at
`~/.cache/inkling2-weightread/` (routers.npz 339.7 MB F32 exact, fingerprints.npz,
router-stats.json, tensor-inventory.json, gguf-metadata.json incl. kv_type_width note,
expert-blocks.npz, golden-selftest.npz). Box 61473783 opened 07:44 UTC and stopped
immediately after artifacts landed.
