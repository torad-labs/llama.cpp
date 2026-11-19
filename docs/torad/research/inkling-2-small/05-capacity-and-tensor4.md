# 05. Capacity and the Tensor 4 serving design

Goal, stated plainly: Inkling 2 Small served by this fork on four RTX PRO 6000 cards with
the same operational shape as GLM-5.3 Flash Tensor 4. That means one tensor-split pass per
step, a unified KV pool that reserved sessions draw from as they grow, kept CUDA graphs,
prompts read in chunks beside live replies, prompt cache across turns, images through
mtmd, and a launcher whose plan is pinned by tests. It does not mean the native checkpoint,
and it does not mean 1M-token sessions. Both points follow from arithmetic below.

Every number in this file is sums from the published configuration and file sizes, not a
load. The first box read replaces them, as it did for GLM.

## The cards and the packs

One RTX PRO 6000 has 97,887 MiB, which is 95.6 GiB. Four cards hold 382.4 GiB, or
410.6 GB in the decimal units file sizes use. The table converts published build sizes to
GiB and shows an ideal even four-way split. Real splits are not perfectly even because
small tensors mirror and split rounding rotates by layer, so treat the last column as a
planning figure with a few GiB of slack either way.

| Build | Total GiB | Per card, even split | Left of 95.6 GiB |
|---|---:|---:|---:|
| Native mixed checkpoint files | 366.5 | 91.6 | 4.0 |
| Native MXFP8 checkpoint files | 580.0 | 145.0 | does not fit |
| Unsloth BF16 GGUF | 451.1 | 112.8 | does not fit |
| UD-Q8_K_XL | 360.0 | 90.0 | 5.6 |
| UD-Q4_K_XL | 329.8 | 82.5 | 13.1 |
| UD-Q3_K_XL | 265.5 | 66.4 | 29.2 |
| UD-IQ3_XXS | 210.5 | 52.6 | 43.0 |
| UD-IQ2_XXS | 177.5 | 44.4 | 51.2 |
| UD-IQ1_S | 159.9 | 40.0 | 55.6 |

Why the native numbers land where they do: experts are 587.1B of the 594B parameters.
Stored as released, the 40 NVFP4 layers take about 275.2 GB and the 8 MXFP8 layers about
100.9 GB, then roughly 17 GB of floating-point remainder, vision tower and draft head
bring the repository to its 393.5 GB. There is no packing trick that changes the
conclusion: on these four cards the model is served from a requantized pack or not at all.

## KV cache and sequence state

Only the 8 global layers grow with context. The 40 sliding layers are capped by the
window and the session count in unified mode.

Global KV per token, all 8 layers, before splitting:

- F16 or BF16: 32,768 bytes. One full 1M-token sequence is 32.0 GiB.
- Q8_0: about 17,408 bytes including scales. One 1M sequence is about 17.0 GiB.
- Q4_0: about 9,216 bytes including scales. One 1M sequence is 9.0 GiB.

With 8 KV heads and four cards, the natural tensor split puts 2 KV heads per card, so a
card carries a quarter of those figures: 8,192 bytes per token at F16. This split is an
explicit decision to encode and test in the split-state table, not an assumption. If the
global cache were mirrored instead, a 2.65M-token pool at F16 would need about 81 GiB per
card for cache alone and the server would not load beside any useful pack.

Fixed per-sequence state is small: sliding caches total 0.156 GiB per sequence at F16
across all cards, and convolution state is 6.19 MiB per sequence at F32. At 23 reserved
sessions that is roughly 1 GiB per card for sliding caches after the head split, and tens
of MiB for convolution state.

## Pool scenarios

Reserve 8 GiB per card for compute buffers, kept graphs, the vision tower on its card,
and driver room. That figure is an estimate from GLM's measured compute buffers, widened
for Inkling's nine live block summaries per mix; the load leg replaces it. What remains
buys pool tokens at the per-card rates above.

| Pack | Per card for KV | Pool at F16 KV | Pool at Q8_0 KV | Pool at Q4_0 KV |
|---|---:|---:|---:|---:|
| UD-IQ3_XXS | about 34 GiB | 4.46M | 8.39M | 15.8M |
| UD-Q3_K_XL | about 20 GiB | 2.65M | 4.99M | 9.41M |
| UD-Q4_K_XL | about 4 GiB | 0.54M | 1.01M | 1.91M |

Read against a 262,144-token session limit, a locally produced abliterated pack in the
UD-Q3_K_XL size class holds about ten simultaneously full sessions at F16 KV or about
nineteen at Q8_0 KV, with 23 sessions reserved against the shared pool in the GLM Tensor 4
pattern. The Unsloth names in the table supply stock-model size brackets until the local
abliterated artifacts exist. A Q4-class pack is a quality experiment for shorter work, not
a default. A single 1M-token session is possible only by
sizing the whole pool for it and giving up concurrent depth, which contradicts the serving
goal; vLLM reaches the same conclusion on larger cards and caps tensor parallel 4 at
262K per request.

KV quantization quality for this model is unmeasured anywhere found in this study. SGLang
serves an MXFP8 KV cache on Blackwell, which suggests 8-bit KV is acceptable to the model,
but our Q8_0 and Q4_0 caches are different encodings and need their own quality leg before
either becomes a default.

## Split design

- Attention splits by heads: 12 query heads and 2 KV heads per card. The relative
  projection, clip scales and sink projection follow the query-head split. The relative
  bias table is 16 x 1,025 values per layer and can mirror.
- K and V convolution kernels and their dynamic projections follow the KV-head split.
  The two hidden-stream convolutions and the convolution state's hidden channels split
  consistently with hidden partials, and each sequence's state stays whole per stream on
  the card that owns those channels.
- Experts split the way GLM's tensor layout splits them: gate and up by intermediate rows,
  down by input columns, one all-reduce after the down projection. SGLang's note that
  NVFP4 experts cannot be tensor-sharded applies to native NVFP4 activation scales, which
  a GGUF pack does not use. If a later custom pack keeps bit-exact NVFP4 experts, this
  decision must be revisited before adopting it.
- The embedding and output head are 0.93B parameters each. Split the output head by vocab
  rows as the standard table does, remembering the graph slices logits to the unpadded
  vocabulary after the softcap.
- The vision tower is not tensor split. It lives on one card inside the reserve above.
  Encoder disaggregation, which SGLang documents, is out of scope for parity.

## Reductions

The layout implies two all-reduces per layer, 96 per step, each carrying rows x 4,608
values. At BF16 that is 0.84 MiB of logical partials per row per step: about 19 MiB at 23
rows and 58 MiB at 69 rows, before the chosen route multiplies traffic. GLM Tensor 4
learned that the right route is the rented host's, not the engine's default, and the box
read starts there for Inkling too. Two differences from GLM: reduction outputs feed
convolutions that write persistent sequence state, so the 8-bit wire needs a fresh quality
read; and there is no indexer or latent cache, so no GLM-specific reduction settings
carry over.

## Launcher profiles

The launcher in `scripts/inkling2/serve.py` pins these plans in its dry-run tests.

- `tensor-4` (default): a locally converted Q3_K_M pack from the pinned
  `benthecarman/Inkling-2-Small-abliterated` revision, plus the F16 mmproj converted from
  the same source; `-sm tensor` with equal splits; flash attention on; F16 KV until the
  Q8_0 leg reads clean. The 23 slots share 3,014,656 tokens. Each slot has one 294,912-token window shared across the conversation, with no separate reply quota.
  The pool holds ten such independent sessions at once. The profile uses unified KV,
  262,144 MiB of host prompt cache, a 2,048-token batch, a 1,024-token prompt step, 128-token
  prompt chunks, and sliced host staging with a BF16 reduction wire. Sampling defaults to temperature 1.0, top-p 1.0, top-k 0 and min-p 0, with
  reasoning effort left to the template default of high.
- `tensor-4-capacity`: a locally converted IQ3-class abliterated pack with Q8_0 KV and a
  larger pool, for when session depth matters more than the last quality points.
- `tensor-4-quality`: a locally converted Q4-class abliterated pack with a small pool,
  kept only if the load leg shows real free memory and a quality leg shows it buying
  something.

No profile points at a stock Unsloth pack. Stock files are bring-up and comparison inputs
only, as [09-abliterated-target.md](09-abliterated-target.md) records.

Settings that do not carry from the GLM launcher: q4_0 latent and q8_0 indexer cache
types, every indexer and MTP environment switch, the coupled-draft flags, and the 8-bit
reduction wire as a default. Settings that do carry as candidates to read, not defaults:
kept-graph counts and node budgets, any-seqs graph reuse once the distance input stops
changing shape, MMQ prefetch switches, `--session-step` with `--prompt-shapes`, and
`--prompt-step-ms`.

## What speed to expect, honestly labeled

No Inkling number exists on RTX PRO 6000 anywhere found in this study. The published
figures are GB200, GB300, B200, H200 and Vera Rubin systems with NVLink-class interconnect
or its equivalent, running the native checkpoint with DSpark. They bound what the model
can do, not what four workstation cards on PCIe will do with a 3-bit pack.

Two structural facts shape our expectation. Decode is a 17B-active model whose expert
reads dominate, closer in spirit to GLM Flash than its 594B total suggests, but every step
also pays 96 residual mixes, 192 convolutions and 96 reductions, which penalizes small row
counts. Prefill is grouped expert GEMM work at top-8 of 384, where this fork's MMQ expert
pair path applies, plus attention over a per-head bias that today is materialized. The
first box read exists to replace this paragraph with measurements; any rate promised
before it would be invention.
