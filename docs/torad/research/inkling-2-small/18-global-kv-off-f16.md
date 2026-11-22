# 18. Taking the global KV cache off F16

Scope: the K and V cache of the 8 global layers on the four-card `tensor-4` profile, 23 sessions. Companion to
`15-kernels-shaped-to-the-architecture.md` (section 3.4) and `16-six-techniques-addendum.md` (item 1); the plan's record
in `17-precision-and-planned-execution.md` section A stays as its summary, this document is the working plan. Same
setting: four RTX PRO 6000 (sm_120), 23 sessions, Q3_K_M pack, a step of about 54 ms with global attention near 18 ms.
Marking as in 15: verified (read in code or measured), external, inferred.

Status 2026-11-22 CST: owned by the session "Inkling 2 Small global KV cache" (`local_4207d667`). The mid-merge hold of
2026-11-21 ended: origin/main merged as `ea83c6dd8`, so the freeze on conflicted files is lifted and nothing has been
committed by this work yet. Built since, off by default behind `GGML_CUDA_FATTN_REL_DECODE_Q4` (correctness argument in
`reports/kv-q40-dispatch-pending.md`): the Q4_0 element decode instance in `fattn-rel-elem.cuh` with dispatch in
the one-cell kernel and rel-runs, the fattn.cu gate hunk applied — GPU validation pending the CloudRift box.
`scripts/inkling2/kv_stats.py` records rotation-on-store fields instead of refusing them, but only f32/f16/bf16 row
types parse today — a quantized dump still exits 2 at the type filter until its row decoder lands in the reader.
`--kv-cache q4_0` accepted by the launcher.
Builds and correctness runs go over ssh to the CloudRift RTX 5090, not this machine.

Incident closed 2026-11-22: an overnight red of all eight `test-inkling-kv1b-*` model arms (batch ctest at Nov 21
23:26 CT) reproduces green from this disk with every rider active; its signature was a node-class coverage assertion —
logits parity passing everywhere (NMSE 3e-8..1.3e-6 against a 1e-4 bound), zero low-mean decode nodes routed — on a
mid-merge tree whose fattn.cu/ggml-cuda.cu resolution state predates the merges that settled it. Recorded as
non-reproductive-fixed-by-merge (`reports/kv-q40-dispatch-pending.md`, "kv1b attribution"). Standing rule taken
from it: validation arms for our new kernel paths assert their node class appears, never numerics alone.

## Summary

The target is a 4-bit global cache, with 8 bits as the fallback. The engine can already store both and read one-byte K
and V in the fast decode kernel. The served profile is still F16 for two reasons, and neither is a missing format:

1. The quality leg for a quantized KV cache on this model was never run.
2. Several fast paths the profile depends on accept F16 only, so switching today would lose them.

So the work is: run one quality leg with Q4_0 and Q8_0 side by side, give the chosen format the fast paths, switch the
default, and spend the freed memory.

## 1. Facts

### 1.1 What F16 costs

A cell holds 2 KV heads a card, K and V, 128 values each: 4 rows of 128 a layer a card.

| Format | Bytes per 128 values | Bytes a cell a card (8 global layers) | Pool of 3,014,656 cells, a card | Freed a card |
|---|---:|---:|---:|---:|
| F16 | 256 | 8,192 | 24.70 GB | |
| Q8_0 | 136 | 4,352 | 13.12 GB | 11.58 GB |
| E4M3 | 128 | 4,096 | 12.35 GB | 12.35 GB |
| Q4_0 | 72 | 2,304 | 6.95 GB | 17.75 GB |

- Memory. Weights take 70.9 GB a card and the F16 pool 24.7 GB, leaving 1.7 to 2.5 GB (`05-capacity-and-tensor4.md`,
  `10-serving-campaign-review.md`). That is why the launcher runs without kept graphs, and why the draft layout cuts
  the pool to 1,572,864 cells.
- Bandwidth. Global attention is about 18 ms of a 54 ms step. Its bytes are K and V; private cells are 88% of its
  read-once bytes (doc 15 section 1.1).
- Host copies. RAM snapshots, slot exchanges and slot saves move the same bytes.

### 1.2 What the engine already has

- Q8_0 and Q4_0 caches. The launcher exposes `--kv-cache q8_0`, with `--swa-kv-cache` to keep the sliding layers
  apart; `q4_0` is now accepted by the launcher flag for the dense-path decode (verified). The draft layout already
  serves the target shape with Q8_0 global and F16 sliding (`serve.py:231`).
- A quantized cache is stored rotated (orthonormal Walsh-Hadamard applied on store: K mixes its whole head,
  n_rot_k 128 at head size 128, V mixes within groups of 64 values, n_rot_v 64; `src/llama-kv-cache.cpp:28` and the
  `LLAMA_ATTN_ROT_DISABLE` escape). This is the engine's answer to hot channels, and it is what makes block scales on
  K reach their floor without a per-channel scale (doc 16 item 1, A1/A2). Per-vector norms survive either rotation,
  so bounds stated later hold on stored values too.
- One-byte global KV in the fast kernels, built and off by default (verified): q8_0 **and** f8_e4m3 in the relative
  decode kernel and the run-addressed kernels behind `GGML_CUDA_FATTN_REL_DECODE_ONE_BYTE`; measured reports live at
  `inkling-plan/reports/attn.md` and `runs.md`. The tensor-core relative decode reads one byte in place (`1454bba3a`):
  one card's slice on an RTX 5080, 23 rows over 16,384 cells a row, 316.0 us against 497.1 us for F16; at 92 rows,
  383.4 against 517.4. It ran at 68% of the card's bandwidth on Q8_0 and 81% on F16, so part of the byte saving is
  spent on conversion.
- The relative element decode is a template over the type: F16, Q8_0 and E4M3 instances behind
  `GGML_CUDA_FATTN_REL_DECODE_ONE_BYTE`, and Q4_0 beside them behind its own `GGML_CUDA_FATTN_REL_DECODE_Q4`
  (off by default; one-cell and runs dispatch included, bit-for-bit store argument in
  `reports/kv-q40-dispatch-pending.md`). Untested on real hardware until the box window validates it.
- An E4M3 cache type with a scale per layer (`is_type_scaled`, `src/llama-kv-cache.cpp`), read by the tile kernel under
  the ONE_BYTE switch. The launcher deliberately keeps it unexposed: over Q8_0 it saves only about 0.77 GB more a card,
  its Mini proxy error is worse (2.6% against 0.59%), and the dense path has no bit-for-bit reference for it — so it
  sits out of the ranking below. It remains the free dry-run candidate if the switch plumbing wants exercise before
  anything is built, since rel-mma reads it today.
- The raw flash-attention path multiplies K on int8 tensor cores for q4_0/q8_0 at head size 256 (`15a502bc5`) and now
  at head size 128 for q8_0 (`4e5911055`); its K*Q is int8 with Q quantized per column, "not the f16 copy's bit for
  bit". Precedent in-tree for section 3.4.
- Tools: `kv_stats.py` (value statistics and a format simulation from a slot state; rotation-on-store fields are
  recorded when present rather than refused — n_rot_k 128 for K, n_rot_v 64 for V — while only f32/f16/bf16 rows
  parse today, so simulated q4/q8 errors come from F16 values in the unrotated basis until the quantized row decoder
  lands: per-vector bounds survive rotation exactly, per-channel figures do not carry across),
  `replay_compare.py` (full next-token
  scores at identical forced inputs, two stock arms), `coding_probe.py` (a repository fix at a chosen prompt depth).

### 1.3 Paths that accept F16 only

- Bucketed prefill graph reuse (`LLAMA_INKLING2_BUCKET_PREFILL_KV`, `src/llama-kv-cache.cpp:1497`, verified:
  the gate tests both types against GGML_TYPE_F16).
- The short-rows relative kernel (`BEST_FATTN_KERNEL_REL_SHORT`, `fattn.cu`; line numbers drift once the open merge
  resolves).
- Grouped low-visibility decode (`--mean-mask-decode`, described as F16).
- Query groups for drafted rows on E4M3 (the tile kernel allows groups on F16 and on Q8_0 read by words).
- Query-first decode: not checked on a quantized cache.

### 1.4 Evidence for 4 bits

- GLM 5.3 Flash is served from this engine with a Q4_0 latent cache and IQ3_XXS weights, in daily use for coding work.
- ds4-dfm-rs runs this model's global-layer KV as NVFP4 (4 bits with a scale per 16 values) with top-1 equal on its
  short checks.
- SGLang ships block-scaled MXFP8 KV for Inkling 2 on Blackwell, where the native expert precision is NVFP4 for layers
  0-39 and MXFP8 for 40-47 (sglang.io blog, Hugging Face model card). External precedent that this architecture class
  ships below-F16 state caches.
- K is clipped to an L2 norm of 22.63 and V is RMS-normalized before the store (`src/models/inkling2.cpp:636-637`,
  verified), so both are bounded with no calibration. Inkling 1 had no such bound, which is why vLLM lost quality
  there at unit scale. The clip plus the Hadamard rotation on store attack exactly what KIVI found — fixed outlier
  channels in K — which is why a block-scale-only format has a fair chance here.

### 1.5 What the test has to cover that GLM does not

- GLM picks 2,048 cells with an indexer kept at 8 bits, then attends over those cells with the 4-bit latent. Selection
  over the long context never sees the 4-bit error.
- Inkling's global layers score every visible cell with the stored K itself, up to 262K cells. So the 4-bit error sits
  in the selection. The place it would show is late in a long session, which is where K3 looks.
- `kv_stats.py` on Inkling 2 Mini (illustrative, not Small): 99% of K vectors sit at the clip. Error of the q.k proxy:
  Q8_0 0.59%, E4M3 2.6%, Q4_0 9.5%, 4-bit float 9.4%. A per-channel scale moved Q8_0 to 0.51% and Q4_0 to 8.2%.
  Whether 9.5% matters to the output is what K3 measures; the proxy does not say.
- External evidence leans against uniform 4-bit K and toward late-depth failure being the central scenario, not the
  tail: llama.cpp PR #7412 ("the K cache seems to be much more sensitive to quantization than the V cache"); KIVI
  Table 2, per-token K quantization costs about 5x the attention-score error of per-channel; KVtuner found models
  hurt even by int4 keys; and one needle-margin study measured retrieval flips go from 1/200 at 8 bits to 155/200 at
  4 when logit margin is thin — which is what zero bias beyond distance 1,024 gives you at scale. The counterweights
  (clip, rotation) are this model's own.

## 2. Measurements

Each one states what is expected and what the result decides.

**K1. The budget.** Run `replay_compare.py` on the served pack with F16 KV twice (its two stock arms). Hypothesis: the
two arms differ by a small, stable amount. This is the floor: a format inside it is indistinguishable from F16. If the
Q4_K_M pack is at hand, capture it too: its divergence from Q3_K_M is the cost already accepted for the weights.

**K2. Format ranking on Small.** Save one F16 slot state of Small and run `kv_stats.py`. Hypothesis: the Mini ordering
holds, and channel scales add little because K sits at the clip and the cache is rotated. Rule: if a channel scale
changes a format's error by less than the K1 floor, drop scale migration from the plan.

**K3. Quality at depth, Q4_0 and Q8_0 side by side.** Same captures as K1 with each cache, at early, middle and late
positions of a long session — late depth weighted heaviest, per 1.5 — plus `coding_probe.py` at the same depths. Both
run on the dense path, so no new kernel is needed for this leg. Run the q8 reduction wire as a further arm.

Arms prepared for inkling-small's shared four-card window (each = one serve over the identical prompt set with these
flags, then its capture stack; `replay_compare.py` takes the stacks as files): A stock F16 + wire bf16; B second
identical run (the K1 drift floor); C `--kv-cache q8_0 --swa-kv-cache f16`; D `--kv-cache q4_0 --swa-kv-cache f16`
(the explicit sliding flag matters: unset, type_k_swa follows the global type at `src/llama-context.cpp:950`, so a
bare `--kv-cache q4_0` would quantize the sliding layers too against section 3.3's target); E stock F16 +
`--reduction-wire q8`, kept factorially separate from C/D in every verdict per the rule below. The kv_stats state
comes from one slot-save under A (`scripts/inkling2/kv_stats.py` USAGE block has the exact flow). Interpretation of
C/D/E deltas and KL verdicts sits with this seat; inkling-small holds the no-loss check and the group arms.
- The wire arm is captured in the same window but analyzed factorially from the KV-format arms (verified:
  doc `08-risks-and-questions.md` risk 6, reduction-wire error enters persistent state; external guidance is transmit
  low precision but add in FP32, since the error compounds token by token). A KV conclusion may never depend on the
  wire choice.
- Sequence state (convolution tails) must round-trip under the wire arm.
- Hypothesis: Q8_0 sits inside the K1 floor at every depth. Q4_0 sits outside the floor on scores but shows no loss on
  `coding_probe.py`, and its divergence is below the pack's own from K1.
- Rule: the smallest format that meets its hypothesis at every depth becomes the default. If Q4_0 fails only at late
  depth, section 3.5 applies. If it fails at every depth, Q8_0 is the default.

**K4. Speed on four cards.** 23 sessions at the served shape, F16 against the chosen format, on the merged engine.
- Q4_0: global attention from 18 ms to between 5.1 ms (the byte ratio 72/256) and 6.0 ms (the same conversion loss
  Q8_0 showed), a step of 41 to 42 ms, about 545 to 560 tok/s against 425.6.
- Q8_0: 9.6 to 11.4 ms, a step of 45.6 to 47.4 ms, about 485 to 505 tok/s.
- The query-first ridge cuts into these bounds (inferred): per KV head the shared prefix serves 138 query rows (23 x
  6 heads); at f16 KV that is about 138 flops/byte under the f16 MMA ridge (~223 on these cards, external measured:
  fp16 399 TFLOPS, int8/fp8 755, NVFP4 1,531 on an RTX PRO 6000 class part); at 8-bit KV it is about 276, past the
  ridge. So 8-bit global KV gains less than 2x on the shared cells unless QK and PV move to int8 or fp8 MMA — q8_0's
  32-value blocks match `mma.m16n8k32.s8` exactly with one block scale per k-step. Private cells (88% of bytes, 6
  query rows) stay bandwidth-bound, so dequantizing them to f16 is enough. A conversion cost bound from outside:
  BitDecoding-FP4 measured NVFP4-to-BF16 dequantization at 7-21% of decode kernel time at 128K; QServe found a 4-bit
  KV dequant can make the fused kernel ALU-bound, with bit tricks near 2 ops per element.
- Rule: above the upper figure, the conversion is binding and section 3.4 applies.

**K5. Memory.** Free bytes a card after the switch, then the bytes kept graphs take at 23 sessions.

## 3. What to build

### 3.1 The fast decode kernel for the chosen format

If K3 picks Q4_0: add its element decode to the tile kernel as a template instance beside F16/Q8_0/E4M3 (a row of 128
values is 72 bytes, four blocks of a scale byte-pair and 16 bytes), with query groups for drafted rows. While the merge
is open this lands as copies beside the shared kernels, wiring into `fattn.cu`/`fattn-rel-mma.cuh` deferred until it
resolves. Check it against the dense path, which is the bit-for-bit reference.

New-kernel guardrail from outside (external): SageAttention3's FP4 kernel measured its published gain only after its
hardcoded 170-SM grid (the 5090's count) was replaced with the real SM count — end to end 1.13x where kernel-level
promised 1.38x over INT8; our cards have 188 SMs, so nothing gets sized off another card's shape. SageAttention2's
INT8-QK/FP8-PV with two-level FP32 accumulation measured 1.91x BF16 SDPA on the same sm_120 class.

### 3.2 Lift the F16-only gates

Objective: switching the cache type must not change which kernels run. Take each path in 1.3 to the chosen format, or
show by measurement that the path it falls to is as fast. The bucketed prefill and the short-rows kernel come first,
since the profile turns them on.

### 3.3 Switch the profile, then spend the freed memory in order of measured gain

`tensor-4`: global KV in the chosen format, sliding KV F16. The sliding layers hold 1,024 cells a session, about
1 GiB a card at 23 sessions, and are near their floor. vLLM keeps small sliding-window layers in higher precision
because the quantization overhead does not amortize over a 128-token window (external); ours is 1,024 — measure rather
than assume.

1. Kept graphs. Without them a step is rebuilt, not replayed. GLM's 69-row evaluation takes 44 ms replayed and 164 ms
   built.
2. The draft on card 3 with the full pool, where the draft layout now cuts it.
3. Pool cells with what remains.

Slot states across the switch: flipping the profile default strands existing F16 snapshots and saves (and vice versa),
and host copies of rotated quantized states need an un-rotate on load before they can be restored into an F16 cache or
re-parsed by the tooling. This belongs to the same commit that switches the default, not after.

### 3.4 Conversion cost, only if K4 says so

Compare in the same kernel: wider reads of the quantized row, and K multiplied raw on the int8 tensor cores with Q
quantized per block, as the raw path already does for head size 256 and for q8_0 at head 128 (`15a502bc5`,
`4e5911055`). The ridge math under K4 decides which parts are worth it.

### 3.5 Mixed precision, likely needed, budgeted now

Keep the cells written last — about the newest 1,024 positions a session — at 8 bits or F16 and the rest at 4 bits,
because storage cannot split by query distance; state plainly that forked sessions' near cells inside a prefix read at
the low bits until their private tail grows past the bias extent, the case doc 15 section 3.1 handles kernel-side.
The relative bias is exactly zero from 1,024 on, so the kernel boundary comes from the model. Repeat K3 on this arm.
This matches KIVI's recent-tokens-in-full-precision result and doc 16 item 1; external evidence (1.5) makes late-depth
Q4 failure plausible enough that the build is carried with the plan instead of behind an "only if".

## 4. Order

1. K1 and K2 from one F16 window on the four cards.
2. K3 with Q4_0, Q8_0 and the q8 wire, factorially separated.
3. Sections 3.1 and 3.2 for the format K3 picks.
4. Section 3.3, with K4 and K5.
5. Document 15, sections 3.1 to 3.3. As read today the step moves 49.4 GB where 13.96 GB would do, and that saving
   multiplies with this one.
6. Sections 3.4 and 3.5 as their conditions hold — 3.5 built alongside regardless.

## 5. Open

- Whether the state written on four cards parses in `kv_stats.py`. It was checked on CPU states and on Mini only;
  rotated-on-store acceptance is now in the reader, so the remaining question is only the four-card layout itself.
- The cost of the rotation on store and on each query at 23 sessions. Not measured here.
- Whether the launcher's `--kv-cache` should take more than `q4_0` next (e4m3 stays out by choice).

## Sources

- `TORAD.md` rows `1454bba3a`, `4e5911055`, `15a502bc5`
- `src/llama-kv-cache.cpp` (28, 395-399, 1497-1500, 1693), `ggml/src/ggml-cuda/fattn.cu`, `fattn-rel-mma.cuh`,
  `fattn-rel-elem.cuh`
- `scripts/inkling2/serve.py`, `kv_stats.py`, `replay_compare.py`, `coding_probe.py`
- `docs/torad/research/inkling-2-small/05-capacity-and-tensor4.md`, `08-risks-and-questions.md`,
  `10-serving-campaign-review.md`, `15-kernels-shaped-to-the-architecture.md`, `16-six-techniques-addendum.md`;
  `/home/marcos/Documents/dev/infra/inkling-plan/17-precision-and-planned-execution.md` section A and its
  `reports/attn.md`, `runs.md`
- External: KIVI arXiv 2402.02750; KVtuner arXiv 2502.04420; llama.cpp PR #7412; SageAttention issues (thu-ml,
  issue 378); vLLM FP8 KV cache post (2026-04-22) and inference-parallelism notes on low-precision wires;
  BitDecoding-FP4; QServe; aosudh's RTX 6000-class measurements; sglang.io MXFP8-KV-on-Blackwell blog; the
  needle-margin KV quantization study (ml.co.ke, 2026-09-26)
