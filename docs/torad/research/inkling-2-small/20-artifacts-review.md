# 20. Independent review of the two unverified serving artifacts

Scope: adversarial, no-stake review of two artifacts other seats delivered on 2026-11-22:

1. `prefill-window-fill-proposal.md` (side-session notes, `local_df3473f5`): the analysis that concurrent
   prefills already share physical ubatches and the staged launcher patch making `--prompt-chunk` optional.
2. `19-prefix-kv-artifact.md` (this directory): the persisted shared-prefix KV deployment spec.

Written by the review lane only. Nothing was built or run; every claim below was re-derived by reading the
current working tree (draft-speed, head `065a3fe83` plus the uncommitted edits present at review time) and the
recorded data files. Verdicts: CONFIRMED, REFUTED, UNVERIFIABLE. Marking as in 15/18/19: verified (read in
code or data), inferred, external.

## Overall judgments

**Artifact 1: PASS on the engine-mechanics analysis and on the central correction; PASS-WITH-AMENDMENTS on
the validation plan.** Every admission-path claim re-derived true: the chunk flag floors, never caps; a lone
prompter takes the whole free window; equal-share division is the engine default. One sub-claim is refuted
(LLAMA_PROMPT_SHAPES does not multiply passes "beside replies alike" once more than two prompts are pending),
and one experiment-design flaw follows from it (the S arm as specified does not exercise the mechanism it
predicts on). One sequencing conflict with the launcher-one-profile rule: the diff flips the tensor-4 default
before the arms measure it.

**Artifact 2: PASS on facts, sizes and architecture to the byte; PASS-WITH-CONDITIONS as a buildable spec.**
All headline figures re-derive exactly (4.47 GB vs 25.77 GB at 131,072; 160 MiB window set; 1.00 GiB/card;
q8_0 to ~2.37 GB). Three conditions before anyone builds on it: (a) its default restore mapping (a shared,
read-only, refcounted G region) requires engine support that does not exist today and the spec does not say
so; (b) it needs a restore-parity acceptance gate, because the repo's own recorded experiments show restored
prefixes can differ from resident ones in top-32 distributions; (c) a Part C unit error and several manifest
pin gaps need fixing. The correctness-risks section misses the seq-id rebinding and restore-parity classes.

## Artifact 1: claim-by-claim

### A1.1 The `--prompt-chunk` floor, not cap

CONFIRMED. `tools/server/server-context.cpp:5744-5745`:

```
const int32_t chunk = std::min(params_base.n_prompt_chunk, n_ubatch);
n_share = std::max(chunk, n_share / chunk * chunk);
```

A floor with chunk rounding. The only later cap is the free window itself (`n_fill = min(n_batch_prompt,
batch.size() + n_share)`, and `read_prompt` returns early at `batch.size() >= n_batch`), never the chunk
value. A lone prompter's share starts at the whole free window and the floor cannot reduce it. The artifact's
cited lines (`:5743-5744`) are one line early; the statement is at 5744-5745.

### A1.2 Omitting the flag selects equal-share division

CONFIRMED. `common/common.h:726`: `int32_t n_prompt_chunk = 0; // prefer useful per-sequence prompt chunks
(0 = share across all pending slots)`. `server-context.cpp:5713-5734` (`share_step`): returns `-1` when
`prompt_share_legacy()` or `n_prompt_chunk != 0`, otherwise sorts remaining counts ascending, divides the
free window by the count of slots still needing an equal share, and stops when a smaller slot's whole
remainder fits. Matches the artifact's description exactly.

### A1.3 A lone prompter takes the whole free window per step

CONFIRMED. With one pending prompt, `share_step` returns `n_avail = n_batch_prompt - batch.size()`.
`n_batch_prompt = n_generated + min(n_batch - n_generated, n_prompt_step)` when a slot generates, else
`n_batch` (`server-context.cpp:5679-5681`). Idle: `n_batch = 2048` (`scripts/inkling2/serve.py:32`,
`BATCH_SIZE = 2048`; ubatch defaults to the batch size, `serve.py:207-210`). Beside replies with
`--prompt-step 1024`: the prompt budget is 1,024 rows. So "idle: up to 2048 rows/pass; beside replies: the
full --prompt-step budget" is correct, and the earlier "~78 slices of 128" verdict was indeed wrong.

### A1.4 Wait-for-full-chunk stalls

CONFIRMED. `server-context.cpp:5765-5771`: `available = n_batch_prompt - batch.size()`; when
`slot.can_split() && n_prompt_chunk > 0` and `available < chunk && remaining > available`, the slot returns
and waits. Removing the flag removes this branch entirely.

### A1.5 Tail slice copies / 803 copies for 46 requests

CONFIRMED as recorded history. `server-context.cpp:6371-6379` carries the comment: with sliced reads every
slice of the last `n_ubatch` tokens was near the end and each copied the slot's state; "on four RTX PRO 6000
with GLM-5.3 Flash 146 MiB and 13 to 40 ms a copy, 803 copies for 46 requests (Nov 11, 2026). So it is lifted
for the first batch near the end and for the one that ends the prompt". The near-end lift already exists, so
this is a residual, small effect, not a current 803-copy cost. The artifact cites `:6355-6373`; the number is
at `:6375`.

### A1.6 Roofline composition and the cheaper lever

CONFIRMED. `reports/inkling2-prefill-roofline.md:13-15`: "A step = one 1,024-row pass (8 sessions x 128) plus
a 1-row donor decode pass", with donor arithmetic 22 x 65,536 / 1,024 = 1,408. `:42`: "Cheaper lever (I):
halving rows leaves ~120 ms of per-pass fixed cost, so `--prompt-step 2048` (15 x 128 rows) cuts ~55 ms a
1,024 rows (10%)". The artifact's "~up to 2x predicted" applies to the expert component only; the recorded
whole-step expectation in the roofline is about 10%.

### A1.7 Distinct-expert saturation arithmetic

CONFIRMED as arithmetic, with two caveats. Top-8 over 384 experts, Poisson model: lambda = 128 x 8 / 384 =
2.667, touched fraction 1 - e^-2.667 = 93.1%; at 512 rows lambda = 10.67, touched 99.998%. The same model
appears in the roofline ("P(miss) 4e-10" at 1,024 rows). Caveats: (a) the exact per-row without-replacement
(hypergeometric) model saturates faster, about 100% at 128 rows, so the artifact is conservative; (b) real
routing skew is unmodeled and pushes the other way; (c) the cited kernel times are minima at 23 rows only
(`docs/torad/research/inkling-2-small/data/rtx5070ti-experts-23.csv`: q3_K, m=1024, rows=23, active 16 ->
min 136.76 us / median 143.21; active 150+ -> min 632.61 / median 686.38), so the 128-2048-row extrapolation
is qualitative, not measured.

### A1.8 LLAMA_PROMPT_SHAPES mechanics

CONFIRMED. `src/llama-batch.cpp:768-782` (`prompt_shape_rows`: remainder pieces first, then powers of two up
to n_max, then whole ubatches; n_max default 256 via `LLAMA_PROMPT_SHAPE_ROWS`, `:739-748`); `:802-831`
(`split_equal` comment and logic); `prompt_shape_sets` defaults to 2 (`:74`); `prompts_alone = shapes &&
n_prompt_sets <= prompt_shape_sets` (`:811`). A 906-row lone tail re-derives as 10, 128, 256, 256, 256: five
passes instead of one, so the pass multiplication claim holds for a lone (or at most two) pending prompt(s).

### A1.9 "LOSES on prefill throughput in idle bursts and beside replies alike"

REFUTED as stated for the many-prompt case. With more than `prompt_shape_sets` (2) pending prompts,
`prompts_alone` is false and prompts are read together as before (`llama-batch.cpp:811, 802-808`); only the
reply/prompt row separation remains. The proposed S arm runs on the 23-slot workload, where pending prompts
far exceed 2, so it would mostly measure the separation change, not the pass multiplication it predicts. The
prediction is right only for <=2 pending prompts and the arm must be redesigned to test what it claims
(for example `LLAMA_PROMPT_SHAPE_SETS` raised to the pending count, or a small-pending-prompt arm).

### A1.10 Graph rebuild cost and build counts

CONFIRMED. `src/llama-context.h:607-609`: "~5 ms to build, split and allocate it, and ~5 ms more to launch
its kernels uncaptured". `:635-637`: "110 of the 229 builds in the turns of 23 sessions of 20,000 tokens on
four RTX PRO 6000 (Nov 11, 2026), over 35 shapes" (that sentence is about prompt-step ubatches rebuilt on
the main scheduler, which is the graph-reuse motivation the artifact cites).

### A1.11 The proposed diffs

CONFIRMED to apply to the tree modulo one hunk. `serve.py:143-144` (prompt-step argument), `:169-175`
(fallback tuple; the tree has `("reduction_route", "host")` on the same line as the first three keys, so the
second hunk is illustrative only, as the artifact itself notes), `:240` (argv line), and
`test_serve.py:31,180-186` all match the diff context. One design wrinkle: with a profile carrying
`prompt_chunk`, there is no way to request the engine default (0) from the command line, because the
validation rejects 0; harmless while no profile sets the key, worth a comment.

### A1.12 Launcher-one-profile compatibility

Compliant as flags. The four-arm validation list (P0, W, B, S) is arm flags/env on the one tensor-4 profile,
which is exactly the allowed A/B shape. The conflict is sequencing only: the diff makes "no floor" the
DEFAULT. Under launcher-one-profile, an improvement lands in tensor-4 with recorded numbers, so the change
should land with `--prompt-chunk 128` still the default, run P0/W/B on the box, then flip the default in a
second commit carrying the numbers. Nothing in the proposal implies a second profile.

## Artifact 2: claim-by-claim

### A2.1 Recorded constants

CONFIRMED. `model-facts.json`: `hidden_size 4608, num_key_value_heads 8, head_dim 128, d_rel 16, rel_extent
1024, sliding_window_size 1024, global_layers [5,11,17,23,29,35,41,47], sconv_kernel_size 4,
qk_l2_clip_threshold 22.6274..., rel_l2_clip_threshold 8.0`; `tensor4_capacity` block at lines 203-221
carries `global_kv_bytes_per_token_all_cards.f16 32768`, `..._per_card_head_split.f16 8192`,
`sliding_kv_gib_per_sequence_f16_all_cards 0.15625`, `conv_state_mib_per_sequence_f32 6.1875`,
`conv_state_channels_per_layer 11264`.

### A2.2 Headline sizes at P = 131,072

CONFIRMED, every term re-derived independently:

- Global KV F16: 8 layers x 2 (K+V) x 8 heads x 128 dim x 2 B = 32,768 B/token; x 131,072 = 2^32 B = 4.000 GiB.
- Window set: 40 layers x 2048 values/cell x 1,024 cells x 2 B = 167,772,160 B = 160.0 MiB.
- Sconv tails: 3 x 11,264 x 48 x 4 B = 6,488,064 B = 6.1875 MiB (see A2.5 for the unit error the doc states here).
- Bundle = 4.162 GiB = 4.47 GB; naive = 48 layers x 2048 values x 2 B x 131,072 = 24.00 GiB = 25.77 GB;
  share 17.3%. The 262,144 and 1,048,576 rows re-derive too (8.162 / 32.162 GiB, shares 17.0% / 16.8%).

### A2.3 q8_0 figures

CONFIRMED. q8_0 is 34 B per 32 values = 1.0625 B/value; 16,384 values/token -> 17,408 B/token (matches the
recorded constant); x 131,072 + window 85 MiB (160 MiB x 0.53125) + half tails = 2.211 GiB = 2.374 GB, so
"halves it to about 2.37 GB" is exact. Naive q8_0 at 131,072 = 12.75 GiB, matching the table.

### A2.4 Per-card and pool figures

CONFIRMED except one internal inconsistency. Head split (2 of 8 KV heads a card): 8,192 B/token/card x
131,072 = exactly 1.00 GiB/card. The pool claim re-derives from `18-global-kv-off-f16.md:43`
("F16 | ... | 24.70 GB a card") and `10-serving-campaign-review.md:28` (3,014,656 tokens, F16 = 92.00 GiB
across four cards, so 23.0 GiB = 24.70 GB a card). But section 7 of the artifact says the shared Part G
region is "about 1.06 GiB/card", which contradicts section 3's exact 1.00 GiB. One of the two must go; 1.00
GiB is the correct figure.

### A2.5 Sconv size and the section 8 open question

REFUTED as stated. The doc reports "about 6.19 MB" (sections 1.1, 2, 3) and its section 8 note claims the
recorded constant "does not re-derive exactly from channels x timesteps x layers x 4 B (the arithmetic gives
about 6.19 MB)". The arithmetic re-derives EXACTLY: 3 x 11,264 x 48 x 4 = 6,488,064 B = 6.1875 MiB (the
recorded field is MiB, as its name says), which is 6.488 MB in decimal units. The bundle should say 6.19 MiB
or 6.49 MB; the open question about the constant is a units confusion, not a mismatch. It does not move the
headline (0.006 GiB at any P). Related finding, useful to the exporter: 11,264 channels/layer is not one
stream. `src/models/inkling2.cpp:100-101` sets `n_sconv_channels = 2 x n_embd_k_gqa = 2048` (K/V conv
history, the R state) and `n_sconv_hidden = 2 x n_embd = 9216` (hidden streams, the S state) when
`hidden_apart` (the default; `LLAMA_INKLING2_HIDDEN_CONV_LEGACY` off). The sum is 11,264, so the size is
right, but Part C's exporter must save both state streams per layer. The doc's own "pin the exact tensor
list" note covers this; keep it.

### A2.6 Engine state stream claims

CONFIRMED. `scripts/inkling2/kv_stats.py:61-67`: `u32 'ggsq', u32 version 4, ...; global cache: ...; per layer
the same for V. The window cache, the convolution state and the server's appendix follow: not read.` The save
paths exist: `src/llama-context.cpp:5956-5973` (`state_seq_save_file`), `src/llama-kv-cache.cpp:2926`
(`state_write`). Two precision gaps for the exporter: the v4 data order is all K blocks per layer, then all V
blocks per layer, with `v_trans`, `n_rot_k`, `n_rot_v`, row types and row sizes written ahead of the blocks
(`src/llama-kv-cache.cpp:3112-3150`), so Part G's "K then V per layer" phrasing is wrong as a layout
description and must follow `state_write_data` literally; and the per-sequence file carries no model identity
today (`src/llama-context.cpp:5997`: "TODO: add more model-specific info which should prevent loading the
session file if not identical"), which is exactly the gap the artifact's manifest pins fill.

### A2.7 Rotation

CONFIRMED. `src/llama-kv-cache.cpp:396` honors `LLAMA_ATTN_ROT_DISABLE`; `:420-421` sets `n_rot_k = n_rot_v
= 64`; `18-global-kv-off-f16.md:203-204` states host copies of rotated states need an un-rotate on load. The
manifest rotation fields and the record-capture-rotation rule are the right response.

### A2.8 Relative bias, sinks, sconv reshaping

CONFIRMED. `src/models/inkling2.cpp:146` creates `attn_rel_bias` as {n_rel, n_extent} (extent 1,024 for
global layers, n_swa for sliding); `:142` creates `attn_sink_proj` {n_embd, n_head}; `:626` computes sinks
from the current token's input every forward; `:583` reshapes the sconv history as
`n_shortconv_l_cache - 1 x n_sconv_channels` (kernel 4 -> 3 timesteps). "No cached per-position bias
component" holds; also `inkling2.cpp:77`: "no RoPE, the relative-position bias carries the positions", so
absolute position bookkeeping is the only position term, as claimed. `18-global-kv-off-f16.md:217`: "The
relative bias is exactly zero from 1,024 on".

### A2.9 Doc 18 cross-references

CONFIRMED. Q8_0 Mini proxy error 0.59% at `18-global-kv-off-f16.md:113`; the retrieval-flip evidence
(1/200 at 8 bits to 155/200 at 4 bits) and "late-depth failure being the central scenario, not the tail" at
`:117-119`; the F16-only paths exactly as listed at `:83-90` (bucketed prefill gate, short-rows relative
kernel, grouped low-visibility decode); pool cut to 1,572,864 cells at `:50`. Nuance: the "central scenario"
and flip findings are about 4-bit K, not q8_0; the artifact's sentence reads as if they were both about
q8_0. It is motivating K3 correctly (K3 runs Q4_0 and Q8_0 side by side), so this is wording, not error.

### A2.10 Kernel-dependent KV-type result (GQA density study)

CONFIRMED. `reports/inkling2-tensor4-inefficiency-review.md:38`: "The shipped F16 KV beats q8_0 on the
grouped kernel but loses ~35% on the blocked kernel at the 2.36M pool - which kernel runs decides which KV
type wins". The artifact's separable-store/conversion-on-import consequence follows.

### A2.11 Draft facts

CONFIRMED. `-23% measured at 23 clients` (`reports/inkling2-tensor4-inefficiency-review.md:39`, `doc 16:78`);
draft layout pool 1,572,864 cells (`18-global-kv-off-f16.md:50`, `scripts/inkling2/serve.py:33`); markov head
`markov_rank 256`, `rnn_hidden`, block 8 (`model-facts.json` mtp_config_dspark). One addition from today's
tree: `065a3fe83` made the draft LAYOUT its own switch (`--draft-layout`, `serve.py:75,198-209,232-233`),
default off, and `draft_layout` also selects `kv = q8_0`. So "draft-layout boot" is no longer synonymous with
`--draft`; the manifest geometry must record the layout switch and the pool cell partition it implies, not
infer them from whether a draft was loaded.

### A2.12 Prefill context

CONFIRMED. `reports/inkling2-prefill-roofline.md:4-5` ("No prefill trace exists") and
`reports/inkling2-tensor4-inefficiency-review.md:40` ("Prefill is untraced and is 14 of the workload's 15
minutes").

### A2.13 Security posture

NOT OVERCLAIMED. "Offline by construction" and "loaded without any network call" are stated as design
requirements of the loader, which is the right register for a spec. The corpus-privacy bullet is
immediately qualified by the derived-data bullet (K/V are one attention away from text; input-recovery
attacks marked inferred), which is the honest framing and the right operator decision point. "Hash pins
catch truncation, corruption and cross-model confusion" is defensible given the model-identity pin in the
manifest (and closes exactly the engine TODO at `llama-context.cpp:5997`). The one wording risk is comparing
the 1 GiB/card Part G with the 24.7 GB/card pool as "small against" it: at profile defaults the free headroom
is only 1.7-2.5 GB/card (`18-global-kv-off-f16.md:47-49`), and the shared region's value is that it REPLACES
per-slot duplicated prefix cells rather than being additive. Say that, so nobody double-counts or assumes
headroom.

## Correctness risks the artifact misses (attacks that survive)

1. **The default restore mapping needs engine support that does not exist.** Section 7's shared mapping
   ("Part G occupies read-only shared pool cells once", refcounted) has no counterpart in the engine: the KV
   cache copies cells between slots and has no foreign read-only region or refcounted cell class (the only
   refcounts in the tree are in `src/llama-memory-recurrent.cpp:1045-1056`, inside recurrent memory state
   dedup). `LLAMA_SLOT_PREFIX_COPY` is a COPY: "the request that takes an empty slot starts from a copy of
   the slot it shares the prefix with" (`tools/server/tests/unit/test_kv_keep_only_active.py:718`). So the
   claim that the shared mapping "matches how prefix_copy semantics already treat forked sessions'
   near-window cells" overstates what exists. Either scope the engine feature (shared region, refcount
   ownership, eviction protection, seq-id rebinding) or make the isolated per-slot mapping v1. This is a
   build-plan blocker, not a correctness footnote.

2. **Restore parity is not gated, and the repo's own data says it must be.** The spec's only offline check is
   `kv_stats.py` structural parsing. The recorded 12-four-card-serving.md:40 experiment shows the restored
   prefix chose the same greedy token but differed from the resident prefix in top-32 (30 of 32 common, max
   logprob difference 2.686), and `:91` records a cross-layout case with only 26 of 32 common (max 1.0833)
   and a failed strict reproducibility check. A restored bundle can therefore present a distribution that a
   cold prefill of the same prefix would not. A greedy-token plus top-k/logprob parity gate (restored vs
   cold-prefilled, same prompt, two trials of each) must be part of acceptance, at early and late depth.

3. **Seq-id rebinding on restore is unlisted.** ggsq cell metadata carries `n_seq_id` and seq ids (kv_stats
   cell layout). Restored cells belong to the capture session; the importer must rewrite them to the
   restoring slot's sequence id. The section 7 invalidation list does not name it, and a miss would alias
   slots or leak visibility across sessions.

4. **Quantized-prefix drift is only half-covered.** Canonical-F16 shards converted on import, and
   rotation-recorded shards, are covered. Not covered: capture format vs store format (capturing from a q8_0
   pool and re-storing as canonical F16 bakes q8_0 rounding into the "canonical" bundle; the manifest should
   record the capture KV type and wire beside the store format), and re-save/re-import cycles across
   deployments.

5. **Engine version pinning has concrete gaps.** The commit pin misses: the uncommitted-tree state this
   campaign actually runs (git status at review time: 20+ modified files including `src/llama-context.cpp`,
   `src/models/inkling2.cpp`, the CUDA attention kernels); a built-binary/build-info identity, which is what
   actually ran; the engine's state format version (`LLAMA_STATE_SEQ_VERSION`, `llama-context.cpp:5910`);
   and the served GGUF hash (recipe name is self-declared; the recorded pack SHA-256 is
   `afb72c67dad0c1c34424de631a9290595b6d1db55e346c001fc38db3efc59e2d`, `12-four-card-serving.md:20`).

6. **Smaller missing modes, in order of payoff:** checkpoint/fork behavior inside a restored prefix (a
   restored session has no intermediate checkpoints, so a fork at depth inside the prefix silently costs a
   cold read where a live session would not); eviction protection of the shared region against the pool's
   reclaim paths and the RAM slot-save path; multi-slot restore ordering (all slots must bind before any
   appends, or window copies race private tokens); and the P = 131,072 case sits just above the model's
   log-scaling floor (128,000) recorded in model-facts, so include a P above and below the floor in
   validation even though the scale is computed at query time.

## What must change before anyone builds on these artifacts

Ordered by cost of getting it wrong:

1. **Artifact 2, mapping:** decide and state whether the shared read-only G region is engine work (scoped
   with seq-id rebinding, refcount, eviction protection) or deferred, with the isolated per-slot mapping as
   v1; correct the prefix_copy "matches" sentence either way.
2. **Artifact 2, acceptance:** add the restore-parity gate (greedy token + top-k/logprob tolerance vs cold
   prefill, early and late depth, layout-matched), citing `12-four-card-serving.md:40,91` as its
   justification.
3. **Artifact 2, sizes:** fix the Part C unit error (6.19 MiB / 6.49 MB, and delete the "does not re-derive"
   open question, which is wrong), and fix section 7's 1.06 GiB to 1.00 GiB.
4. **Artifact 2, manifest:** pin build/binary identity and the engine state format version, the GGUF SHA-256,
   the capture KV type and reduction wire, and the layout switch (`--draft-layout`) plus pool partition;
   record Part C as two state streams per layer.
5. **Artifact 2, risk list:** add seq-id rebinding, eviction/refcount transitions, checkpoint/fork behavior
   inside restored prefixes, and capture-vs-store format drift.
6. **Artifact 1, S arm:** restrict the pass-multiplication prediction to <=2 pending prompts and redesign the
   S arm so it exercises `prompt_shape_sets` (or drop it to a latency-variance probe as the artifact already
   half-suggests); do not run it as designed against the 23-slot workload and read the result as a test of
   the mechanism.
7. **Artifact 1, sequencing:** land the `--prompt-chunk` flag with the 128 default intact, measure P0/W/B on
   the box, then flip the tensor-4 default in a second commit with the numbers recorded, per
   launcher-one-profile.
8. **Artifact 1, citations:** correct line references (floor 5744-5745, stall 5765-5771, share_step
   5713-5734, prompt_shape_rows 768, copies 6371-6379, roofline 13-15).

Nothing else in either artifact failed re-derivation. The prefill correction (floor, never cap; lone
prompters already fill the window) is right, and the prefix-KV size model is exact. The conditions above are
what stands between "analytically right" and "safe to build on".
