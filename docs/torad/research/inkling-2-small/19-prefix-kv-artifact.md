# 19. The persisted shared-prefix KV artifact

Scope: a deployment artifact that ships the key-value state of a pinned common corpus prefix, so a self-hosted
serving box starts near zero instead of prefilling the same corpus again at every boot, restart or slot churn.
The serving shape it targets is the one this campaign already serves: four RTX PRO 6000 cards, Inkling 2 Small
(abliterated, tensor-4 profile), about 23 concurrent coding sessions over one shared pool with a huge common
prefix. Prefill holds 14 of every 15 minutes of workload time there and was never traced end to end
(`reports/inkling2-prefill-roofline.md` carries the roofline work); this document specifies the product
primitive that removes most of that cost without re-running it. Companion documents:
`02-architecture.md`, `05-capacity-and-tensor4.md`, `12-four-card-serving.md`,
`15-kernels-shaped-to-the-architecture.md`, `16-six-techniques-addendum.md`,
`18-global-kv-off-f16.md`. Facts come from `data/model-facts.json` (hidden_size 4608, head_dim 128,
num_key_value_heads 8, d_rel 16, rel_extent 1024, sliding_window_size 1024, global_layers [5, 11, 17, 23,
29, 35, 41, 47]) unless marked otherwise.

Status 2026-11-22 CST: specification only, no implementation in this pass. Written by the prefix-artifact spec
seat under the coordinator session for the Inkling serving group (`local_df3473f5`). Every byte figure below is
analytical, computed from recorded constants; no GPU run backs it yet. The final F16-versus-q8_0 prefix-quality
claims wait on the end-of-work validation pass on the four cards.
Marking as in 15 and 18: verified (read in code or data), inferred, external.

## Summary

Of the model's 48 layers only 8 (the global layers) carry long-context KV. The other 40 are sliding-window
attention with window 1,024 cells and a relative bias that is exactly zero beyond extent 1,024, so their deep
prefix contributes nothing to any future token. Deep-prefix state therefore lives almost entirely on those 8
global layers, plus two small fixed-size terms: one trailing window of K/V per sliding layer, and the
short-convolution tails. A bundle holding exactly those parts is about one sixth of naive full-KV storage at
any interesting depth, converging down to the 8/48 deep-term ratio as the prefix grows:

| Prefix P | Bundle F16 | Naive full-KV F16 (48 layers) | Share |
|---:|---:|---:|---:|
| 131,072 tokens | 4.16 GiB (4.47 GB) | 24.00 GiB (25.77 GB) | 17.3% |
| 262,144 tokens | 8.16 GiB (8.76 GB) | 48.00 GiB (51.54 GB) | 17.0% |
| 1,048,576 tokens | 32.16 GiB (34.53 GB) | 192.00 GiB (205.6 GB) | 16.8% |

The engine already serializes every component this needs - `ggsq` v4 slot states carry global-layer K blocks,
then V, then "the window cache, the convolution state and the server's appendix" (`scripts/inkling2/kv_stats.py`
state-file notes; `src/llama-context.cpp` `state_seq_save_file`, `src/llama-kv-cache.cpp` `state_write`). The
artifact is the portable, content-addressed, version-pinned form of that stream for the shared prefix, stripped
of private tails.

## 1. Facts

### 1.1 What persists after a prefill ends

For positions [0, P) to be reusable by any later session that continues from P, exactly these are needed:

- Global layers [5, 11, 17, 23, 29, 35, 41, 47]: full K and V over all P positions. This is the only growing
  term (verified: model architecture; `src/models/inkling2.cpp`).
- The 40 sliding-window layers: K and V for the trailing min(P, 1024) positions only. Attention in those layers
  never reaches further back than `sliding_window_size`.
- Short-convolution ("sconv") tails per layer: a recurrent history of the last kernel-1 = 3 timesteps per
  channel (`n_shortconv_l_cache - 1` x `n_sconv_channels` reshaped at `src/models/inkling2.cpp:576`; recorded
  channels-per-layer 11,264). Engine-stored F32, about 6.19 MB per sequence copy (recorded:
  `conv_state_mib_per_sequence_f32` 6.1875; derivation note in section 8).
- Absolute position bookkeeping through P.

Nothing else carries across the boundary (verified by reading the graph code):

- Relative bias is looked up each step from weight tensors `attn_rel_bias` {d_rel 16, extent 1,024} plus an
  `attn_rel_clip` table per layer (`src/models/inkling2.cpp:146`). It has no cached per-position component; it
  is zero from delta 1,024 on (doc 18, section 3.5). Restored state needs correct absolute positions and
  nothing more.
- Attention sinks are projections of the current token's input (`attn_sink_proj`, a weight of shape
  {hidden_size, n_head}; computed per query at `src/models/inkling2.cpp:622`). Not accumulated state.

### 1.2 What naive serving stores for the same prefix

A deployment that does not exploit the architecture pays full K/V on all 48 layers: 196,608 B/token F16
(192 KiB), of which 163,840 B/token (five sixths) sits on sliding-window layers whose deep cells are never
read again once the window moves past them. The served profile already knows better - its shared pool holds
global-layer KV only to full depth (32,768 B/token across all four cards, verified:
`global_kv_bytes_per_token_all_cards.f16`) plus one 1,024-cell window set per sequence on the sliding layers
(160 MiB, verified: `sliding_kv_gib_per_sequence_f16_all_cards` 0.15625). The artifact prices the same state,
not less, so its size claim is conservative against every optimization doc 15 and doc 18 still plan.

## 2. Contents of the bundle

The bundle holds the engine state after a completed prefill of positions [0, P) over the common corpus.
Four parts:

**Part G - global-layer K and V, positions [0, P).** All 8 global layers, K then V per layer in the same block
order as `ggsq` v4 states. Canonical storage is F16 at 2048 values/layer/token (K+V, 8 KV heads x 128 head
dim), 32,768 B/token across the deployment. Optional companion shards at q8_0 (17,408 B/token) or the approximated
q4_0 figure (9,216 B/token) may ship beside it; section 5 holds the format decision open. Quantized shards must
state their rotation parameters explicitly: the engine stores quantized caches rotated with Walsh-Hadamard,
n_rot 64 (`src/llama-kv-cache.cpp`; escape hatch `LLAMA_ATTN_ROT_DISABLE`), so a shard must say whether its
values are canonical pre-rotation or rotated-on-store, and pin n_rot if the latter.

**Part W - sliding-window cells.** Per sliding layer, K and V for positions [P - min(P, 1024), P). One logical
copy; every restored slot starts from identical window content because those positions belong to the shared
corpus. Fixed size: 160 MiB F16 for all 40 layers (85 MiB at q8_0), independent of P once P >= 1,024.

**Part C - convolution tails.** The sconv history per layer at position P: 3 timesteps x recorded channel count
x 48 layers in the engine's F32 storage (about 6.19 MB). Also one logical copy at restore time; after the first
private token each slot's tail diverges, so this part seeds, not shares, ongoing state.

**Part M - manifest and appendix.** Absolute prefix length P; the hash chain of section 4; the server-appendix
fields a `ggsq` v4 file would carry forward (layout descriptors, cell counts per class) so an importer can map
parts onto pool regions without guessing.

Explicitly out of scope for v1: draft-model (DSpark/MTP) state; expert/router caches; any host-side reduction
wire state. Reasons in sections 7 and 8.

## 3. Size

Per-token terms (all four cards combined):

| Term | F16 | Q8_0 | Growth |
|---|---:|---:|---|
| Global deep KV | 32,768 B/token | 17,408 B/token | linear in P |
| Sliding window set (fixed) | 160 MiB | 85 MiB | none past 1,024 cells |
| Sconv tails (fixed) | ~6.19 MB F32 | ~3.5 MB if halved on store | none |

Totals against naive full-KV (48 layers, same format):

| P | Bundle F16 | Naive F16 | Share | Bundle Q8_0 | Naive Q8_0 | Share |
|---:|---:|---:|---:|---:|---:|---:|
| 131,072 | 4.162 GiB | 24.00 GiB | 17.3% | 2.211 GiB | 12.75 GiB | 17.3% |
| 262,144 | 8.162 GiB | 48.00 GiB | 17.0% | 4.336 GiB | 25.50 GiB | 17.0% |
| 1,048,576 | 32.162 GiB | 192.00 GiB | 16.8% | 17.086 GiB | 102.00 GiB | 16.8% |

Headline: at the served shape's natural 128K boundary the F16 bundle is 4.47 GB where naive serving of the same
prefix is 25.77 GB - one sixth and a fraction, and the fraction shrinks with depth. The q8_0 shard halves it to
about 2.37 GB.

Per card under the deployed head-split (each card holds 2 of 8 KV heads for every position): Part G costs
8,192 B/token/card F16 (verified: `global_kv_bytes_per_token_per_card_head_split.f16`), so P = 131,072 puts exactly
1.00 GiB on each card - small against the 24.7 GB/card the full 3,014,656-cell shared pool already occupies at
profile defaults (23 GiB binary).

What restore costs inside the live pool: Part G lands once as a read-only shared region; Part W materializes per
the slot mapping chosen in section 7 - either one shared copy (160 MiB) or 23 copies (3.59 GiB across slots,
matching the per-sequence accounting the capacity docs already charge). Neither grows the steady-state pool;
both replace prefill compute.

## 4. Content addressing and version pinning

- **Addressing.** A bundle's name is the sha256 of its canonical byte stream (parts concatenated in a fixed
  order with lengths). Large parts ship as shards with their own sha256 and length so four-card restores verify
  and pull in parallel and a truncated transfer fails loudly instead of silently mis-serving.
- **Manifest fields** (Part M carries them):
  - format_version; part list with offsets, lengths, hashes.
  - Model identity: repository, pinned revision `e4853745aa9cb4587e8959d3c80b92e1bef99a4a`, pack recipe name -
    pack names already carry revision and recipe "so no file can pass for a stock build" (`serve.py`).
  - Tokenizer and chat template identity: the launcher already pins the template by sha256
    (`TEMPLATE_SHA256` in `scripts/inkling2/serve.py`); the manifest repeats it plus a tokenizer hash, because
    KV state is only as reproducible as the token stream that made it.
  - Engine pin: torad-llama.cpp git commit and the profile switches that change cache semantics
    (rotated-store parameters incl. n_rot if any quantized shard is rotated-on-store; clip thresholds
    qk 22.627 / rel 8.0 from model-facts).
  - Geometry: P, per-class cell counts, layer ids, head split, slot count it was captured for.
  - Prefix provenance: sha256 of the exact decoded token ids of [0, P), so an importer checks a corpus before
    spending pool on it.
- **Pinning rule.** A bundle loads only into an engine/pack/template matching its manifest; anything else
  refuses and falls back to prefill, which is always correct. Never silently approximate.

## 5. Correctness risks: quantized-for-prefix versus F16

- **The recorded result is kernel-dependent.** The shipped F16 global KV beats q8_0 on one kernel family but
  loses about 35% on another at the 2.36M-cell pool; "which kernel runs decides which KV type wins"
  (`reports/inkling2-tensor4-inefficiency-review.md`, the GQA density study). Consequence for this spec: the
  artifact's canonical shard format must stay separable from the runtime pool type - store F16 canonically,
  convert on import if the deployment's pool runs something else.
- **The quality leg never ran on Small.** Doc 18 schedules K3 (Q4_0/Q8_0 side by side at early/middle/late
  depth) and records the Mini proxy error for q8_0 as 0.59%, with late-depth retrieval flips named the central
  scenario rather than a tail risk. Inkling's global layers score every visible cell with the stored K itself,
  so prefix-wide quantization error sits inside selection, not outside it (doc 18 section 1.5).
- **Rotation complicates canonical form.** Quantized caches are rotated-on-store; host copies of such states
  need an un-rotate before re-parse when formats flip (doc 18 section 3.3). The manifest's rotation fields in
  section 4 exist for exactly this.
- **The reduction wire writes persistent state.** Error in the q8 exchange path compounds token by token into
  stored KV (doc `08-risks-and-questions.md` risk 6). Capture bundles quiescently - after a completed step -
  and record the capture profile, so a bundle never freezes mid-wire state.
- **Restored state must present like post-prefill state to every kernel.** F16-only paths (bucketed prefill
  reuse gate, short-rows relative kernel, grouped low-visibility decode - doc 18 section 1.3) bypass restored
  prefixes because no prefill runs over them, but slot saves taken later from restored sessions must still
  parse as ordinary states; `kv_stats.py` is the offline check.
- **Final claims wait on hardware.** Per the coordinator's constraint: F16-versus-q8_0 prefix-quality statements
  stay open until the end-of-work validation pass runs the real arms on the four RTX PRO 6000 cards.

## 6. Security posture

- **Offline by construction.** The artifact is produced on the deployment's own box from its own serving run,
  verified locally by sha256 before load, loaded without any network call. Nothing phones home; nothing needs to.
- **The corpus stays private.** A team self-hosting an ablated model keeps its corpus private precisely because
  serving it does not require sending anything anywhere: the bundle contains derived tensors only, and it lives
  and loads inside their perimeter.
- **Treat bundles as corpus-derived data anyway.** K/V values are one attention away from the text that made
  them; input-recovery attacks against cached activations are demonstrated in adjacent settings (inferred).
  Handle and share a bundle with at least the care the corpus itself deserves - this belongs in the operator's
  decision, not the loader's assumption.
- **Integrity and supply chain.** Hash pins catch truncation, corruption and cross-model confusion at load;
  the manifest's engine/pack/template pins block the quiet mismatch class (wrong revision, re-rendered
  template) that would otherwise serve plausible garbage from stale state. Multi-tenant note: a bundle serves
  the slots of one deployment; sharing between deployments is deliberate copy, not shared cache.

## 7. Slot mapping and invalidation on restore

At restore, all SLOTS sessions resume from the same boundary P with identical logical state:

- **Default mapping (shared).** Part G occupies read-only shared pool cells once (about 1.06 GiB/card F16 at
  P = 131,072 under the head split); each slot's window and sconv tail copy in from Parts W/C and diverge as
  private tokens arrive. This matches how `prefix_copy` semantics already treat forked sessions' near-window
  cells (`serve.py`; doc 15 section 3.1 discusses exactly those near cells inside a prefix).
- **Isolated mapping (fallback).** If a deployment needs strict per-slot isolation for auditability, materialize
  full copies: deep term x 23 slots against today's pool accounting - workable only because naive serving could
  not fit even one session of it. Costs are the table in section 3 multiplied by SLOTS.
- **Lifetime.** Shared G-region cells are refcounted by the slots using them; released when the last dependent
  slot closes or the pool is repartitioned. An explicit rebind replaces the region when a new bundle version
  lands. Nothing stays owned by no one.
- **Invalidation on restore** - refuse, then fall back to prefill:
  - manifest pin mismatch (engine commit, pack recipe/revision, template/tokenizer hashes);
  - provenance hash of the corpus prefix does not match what the importer computes;
  - P exceeds the shared region available under the current profile (for example a draft-layout boot cuts the
    pool to 1,572,864 cells);
  - any part fails its shard hash.
- **Partial restores** stay out of v1; a segmented bundle (checkpoints at fixed cuts, in the spirit of
  `checkpoint_every` 32,768) would let sessions continue from a shorter validated boundary and corpora grow by
  appending segments. Listed as an open question, not designed here.

## 8. Open questions

1. **Canonical KV format for Part G: F16 or q8_0?** The quality leg never ran on Small, the recorded speed
   result flips with kernel choice (section 5), and the deciding GPU validation pass has not run yet. Until it
   does, ship canonical F16 and keep conversion-on-import available both ways.
2. **Window and tail materialization granularity.** One shared copy (160 MiB, seeds all slots until their tails
   diverge) versus per-slot copies (3.59 GiB across 23 slots today's accounting already charges per sequence)
   changes restore-time pool pressure and fork semantics alike; the serve side must pick, and the sconv tail
   divergence point after restore deserves a parity capture like doc 18's K3, including under the wire arm -
   sequence state must round-trip through any save path.
3. **Draft-state checkpointability.** v1 excludes DSpark/MTP prefix state: the draft is off in the served
   profile today (-23% measured at 23 clients anyway), its layout cuts the pool to 1,572,864 cells, and its
   recurrent markov head (`markov_rank` 256, `rnn_hidden`, blocks of 8 drafted tokens) may only be recoverable
   by replaying trailing tokens rather than restoring - measure before promising cold-start for the draft too.

Two smaller notes recorded so they are not lost: the recorded conv constant (`conv_state_mib_per_sequence_f32`
6.1875) does not re-derive exactly from channels x timesteps x layers x 4 B (the arithmetic gives about
6.19 MB); pin the exact tensor list when Part C's exporter exists. And bundle segmentation for growing corpora
(section 7) is designed nowhere yet.

## Sources

- `docs/torad/research/inkling-2-small/data/model-facts.json` (all recorded constants cited inline)
- `src/models/inkling2.cpp` (rel bias and clip tables :146, sink projection :142/:622, sconv history :576)
- `src/llama-context.cpp` `state_seq_save_file`, `src/llama-kv-cache.cpp` `state_write` (ggsq v4 stream);
  `scripts/inkling2/kv_stats.py` state-file notes; `scripts/inkling2/serve.py` profile and template pin
- `reports/inkling2-tensor4-inefficiency-review.md` (kernel-dependent KV-type result),
  `reports/inkling2-prefill-roofline.md` (prefill cost context)
- `docs/torad/research/inkling-2-small/02-architecture.md`, `05-capacity-and-tensor4.md`,
  `08-risks-and-questions.md`, `12-four-card-serving.md`, `15-kernels-shaped-to-the-architecture.md`,
  `16-six-techniques-addendum.md`, `18-global-kv-off-f16.md`
