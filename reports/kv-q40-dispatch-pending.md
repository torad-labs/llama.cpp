# Q4_0 relative decode — built, gate applied, hardware validation pending

Shared-code law (be60f24ef) compliance for this slice, stated once here rather than per file: the fattn-rel-*
headers also serve GLM 5.3's build, so every change this plan made to them takes the guard strategy — a new
Q4_0 template instance and dispatch cases behind `GGML_CUDA_FATTN_REL_DECODE_Q4`, default off, with the q8_0/
E4M3/F16 paths byte-for-byte what they were; GLM serves none of these types through the Inkling kernels, so its
behavior is unchanged by construction as well as by switch.
Released hold (goal-sharpening directive, same day): upstream-shaped caution no longer applies. Once the box
window validates the kernel on real weights, the switch may collapse into unconditional q4 acceptance like
q8_0's today — dropping knob ceremony entirely — and doc 18 section 3.3's default flip lands straight in the
tensor-4 profile with its measurement in the row note, since GLM stays green either way.

Companion to `docs/torad/research/inkling-2-small/18-global-kv-off-f16.md` section 3.1 and ledger rows
`1454bba3a` (rel-mma reads q8_0 one byte), `15a502bc5`/`4e5911055` (raw path). State as of the merge landing
and this seat's commit pass: the merge resolved as `ea83c6dd8` + review fixes `ad51d800a`, the dispatcher patch
below was applied at `fattn.cu:1397-1398` in the meantime, everything here is committed (`3280492de` kernel
slice, `69b948642` relocation, `0075b6d62` staging-hygiene note), and a no-stake DeepSeek V4.1 Flash review
(`reports/q40-review-report-deepseek.md`) found four importants afterwards — the one-cell indexing bug it caught
is fixed in this file's follow-up commits; nothing here awaits a merge any more.

## Routing caveat owned by inkling-mini's rider

The uncommitted `BEST_FATTN_KERNEL_REL_ROWS` hunk (Inkling 2 Mini's unified-pool rows decode,
`ggml/src/ggml-cuda/fattn-rel-rows.cuh`, sits after the REL_DECODE gate) accepts q4_0/q8_0. On a tree carrying
that rider with `GGML_CUDA_FATTN_REL_DECODE_Q4` unset, a q4_0 cache may route to it instead of the dense
deq-dense path — which would change the meaning of serve.py's "dense-path decode today" until that rider lands
or is reverted. Mini owns its landing; validation arms should assert their node class runs regardless.

## What landed already (committed)

- `ggml/src/ggml-cuda/fattn-rel-elem.cuh`: template instance `fattn_rel_elem<GGML_TYPE_Q4_0>` beside
  F16/Q8_0/E4M3, helper `fattn_rel_elem_q4(type)` behind env switch `GGML_CUDA_FATTN_REL_DECODE_Q4`
  (off by default; declared in tunables.toml since), and
  `fattn_rel_elem_align(Q4_0) = 8` beside q8_0's (rows of 72 bytes are not multiples of 16).
  `fattn_rel_elem_one_byte` itself is unchanged, so ONE_BYTE keeps its exact old meaning.
- `ggml/src/ggml-cuda/fattn-rel-decode.cuh`: `fattn_rel_decode_value<GGML_TYPE_Q4_0>` branch and a
  Q4_0 case in the one-cell kernel's pick switch. This is the fallback that also takes query groups.
  The DeepSeek review caught the block indexing reading without `col % QK4_0` — wrong nibbles for columns 32-127
  and reads up to 95 bytes past a row; fixed to take the within-block column like the q8_0 arm does.
- `ggml/src/ggml-cuda/fattn-rel-runs.cuh`: Q4_0 clause in `fattn_rel_runs_type` and a Q4_0 case in
  `ggml_cuda_fattn_rel_runs`'s dispatch switch (its default would have read the cache as F16).

Correctness argument: the tile store is transcribed from the dense path's reference decoder
`flash_attn_ext_raw_dequant_pair<GGML_TYPE_Q4_0>` (`fattn-mma-f16.cuh`, applied to the two blocks of
the lane's half): same nine words, same byte_perm selectors, same f16 bias trick (nibble | 0x6400 is
1024+nib; minus 1032 removes the 1024 exponent base and q4_0's offset 8), one rounding by `__hmul2`.
q4_0 packs value j into byte j low and value j+16 high (`qs[j] = L[j] | L[j+16]<<4`,
`quantize_row_q4_0_impl`); both the row reference `dequantize_row_q4_0` (`ggml-quants.c:517`,
registered `.to_float` at ggml.c:723) and the pair decoder write lows to positions 0-15 then highs to
16-31, so the staged tile equals an f16 copy's values. A host simulation emulating `__byte_perm`
(default prmt semantics) and IEEE-correct `_Float16` ops checked all of this bit for bit across a
spread of scales including subnormals — see `q40-store-sim.cpp` in this session's notes folder
(`.../side-session-notes/local_4207d667-fe67-48ea-b492-7fec449a4b84/q40-store-sim.cpp`), result "OK:
store matches dequantize_row_q4_0 halves and the dense pair decode, bit for bit". Reads stay inside
each 72-byte row (windows cover bytes [32p..32p+39], p in {0,1}), so no over-read past the pool.

Routing today with the switch on: REL_DECODE is chosen only through the fattn.cu gate below, which
still refuses Q4_0 until that hunk lands; rel-mma declines by construction because its frozen chain
reads ONE_BYTE (`fattn_rel_mma_elem` → `fattn_rel_mma_type`), so setting both flags cannot reach the
frozen pick switch and abort — it falls to rel-runs or the one-cell kernel once wired.

## Patch 1 (required to activate): fattn.cu line 1395

In `ggml_cuda_get_best_fattn_kernel`, the BEST_FATTN_KERNEL_REL_DECODE condition:

    k->type == v->type && (k->type == GGML_TYPE_Q8_0 || fattn_rel_mma_type(k->type)) && softcap == 0.0f) {

becomes:

    k->type == v->type && (k->type == GGML_TYPE_Q8_0 || fattn_rel_mma_type(k->type) ||
        fattn_rel_elem_q4(k->type)) && softcap == 0.0f) {

No other change in fattn.cu: everything else the branch tests (head sizes, strides, mask kind,
`n_kv_max`) is type-agnostic for q4_0's shape, and the dense deq-dense gate that serves as reference
already takes Q4_0/Q8_0 at head size 128.

## Patch 2 (optional, only if rel-mma should stage q4_0 tiles): fattn-rel-mma.cuh

In `ggml_cuda_fattn_rel_mma`'s pick switch (after the Q8_0 case), needed only if a future step wants
query-grouped rows on the tile kernel at q4; today `fattn_rel_mma_shapes` never passes for q4, so
this hunk changes nothing on its own:

    case GGML_TYPE_Q4_0:
        kernel = pick(std::integral_constant<ggml_type, GGML_TYPE_Q4_0>{}, std::true_type{});
        break;

(With it, group==1 steps stage through the new template instance; group>1 stays on the one-cell
kernel, which supports groups natively.)

## tunables.toml / ggml-tunables.inc

The merge freeze ended, so the deviation is closed: `GGML_CUDA_FATTN_REL_DECODE_Q4` is declared in
tunables.toml beside its ONE_BYTE sibling (bool knob, off by default, output rounding). If doc 18
section 3.3 later switches the profile default, revisit whether the knob stays or the acceptance goes
unconditional like q8_0's.

## What test-backend-ops proves once builds are possible

- The existing relative Q4_0 case (`test_flash_attn_ext_rel_pos(128, 8, 6, 1024, nb, 1, 1025,
  GGML_TYPE_Q4_0, false)`, tests/test-backend-ops.cpp:17630) keeps covering the dense deq-dense
  reference path unchanged.
- Extend the group loops (the `{GGML_TYPE_Q8_0, GGML_TYPE_F16}` sets at tests/test-backend-ops.cpp:17661,
  17673 and 17699) with GGML_TYPE_Q4_0 arms: they exercise rel-runs staging and the
  one-cell kernel at groups 1-4 and GQA 2/8 against the CPU oracle, the same matrix row
  `1454bba3a` ran for q8_0. Expected pass counts should match those runs' shape (all attention cases
  green under both the switch set and unset, plus `GGML_TUNABLES_STRICT=1`).

## Build/validation status

The CloudRift RTX 5090 ssh box (buildbox via ~/.ssh/config, 195.189.62.247:50127) refused
connections when tried once on 2026-11-22; nvcc syntax check, the test-backend-ops extension above
and a real compile of the instance are deferred to it post-merge.

## What remains after the merge lands

1. ~~Apply Patch 1~~ — done by the coordinating seat inside the merged tree (fattn.cu:1395 carries
   `|| fattn_rel_elem_q4(k->type)`); Patch 2 stays optional future work.
2. Rebase this working-tree state as its own commit naming the ledger row it serves.
3. Run the extended test-backend-ops matrix on the box with and without
   `GGML_CUDA_FATTN_REL_DECODE_Q4`, then K3/K4 quality and speed legs per doc 18 section 2.

## kv1b attribution

Overnight batch red, closed non-reproductive-fixed-by-merge:

- **Signature** (not numerics, not state): node-class coverage assertion — "31 K/V nodes ran,
  0 low-mean decode nodes, 0 off device → FAIL"; logits parity passed with NMSE 3e-8..1.3e-6
  against a 1e-4 bound on all sampled arms.
- **Timing**: batch artifacts stamped Nov 21 23:06–23:56 CT, ctest at 23:26 — before ad51d800a
  landed (~00:05), mid-merge with unstaged conflict-resolution riders in fattn.cu (+194/−1) and
  ggml-cuda.cu; corroborated by the 22:39 safety snapshot hunks and attn.md's green-arm shape
  ("27 prompt and 4 low-mean decode nodes"). A pre-finalization REL_DECODE selection routed the
  low-mean class nowhere; that content is inside ea83c6dd8 today.
- **Closure**: current disk runs green dirty (every ledger hook active) and clean on gpu-5090-02 —
  `ctest -R '^test-inkling-kv1b-'`: dirty tree 80% (2 ops cases Not Run for an unbuilt binary),
  clean ad51d800a floor `100% tests passed, 0 tests failed out of 10`. Nothing to fix in any rider.

Standing rule from the incident: validation arms assert their node class RUNS (`N low-mean decode
nodes` > 0 when the switch routes them there), never numerics alone. When extending the group loops
with Q4_0 per this file's plan, run those arms WITH `GGML_CUDA_FATTN_REL_DECODE_Q4=1` and add a
coverage assertion for the q4 path's kernel class alongside the logits check.

Buildbox note for whoever resumes here: gpu-5090-02 moved endpoints after its recreate — it answers
as `riftuser@180.189.55.43` port 12800 (`rig cloudrift status`; mini-search box same IP, port
40500); ~/.config/rig/ssh_config's buildbox block was rewritten accordingly with fresh host keys,
and the fresh image needed `-DCMAKE_CUDA_ARCHITECTURES=120`.

## Staging hygiene, from the 847e4c638 sweep

Committing by named `git add <paths>` does not make a commit contain only those paths: everything already in
the shared index goes too. 847e4c638 originally carried inkling-mini's `fattn-rel-rows.cuh` (unified-pool rows
decode kernel for Mini, q4_0/q8_0, GGML_CUDA_FATTN_REL_ROWS_LEGACY), staged by another seat's session before
this one committed its docstring fix on top of it. Fixed by plumbing while nothing was pushed: the branch moved
to d8d6b5607 (kv_stats change alone) and the rows file returned to disk untracked as its owner left it.
Standing rule adopted here: every future commit in this checkout inspects `git diff --cached --stat`
immediately before committing — the index never starts empty.

## section 3.5 skeleton

Landed 2026-11-22 CST behind `LLAMA_INKLING2_KV_MIXED` (off by default; declared in tunables.toml with
`LLAMA_INKLING2_KV_MIXED_HOT`, extent 1,024 as served):

- `src/llama-kv-mixed.{h,cpp}` — the hot pool object: F16 rows ringed over writes on the cold buffers'
  backend type, stamps saying whose value each slot holds (`mirror()` routes stores by recency,
  `merge_rows()` materializes one source per cell, `reset_stamps()` clears after restores).
- `src/llama-kv-cache.{h,cpp}` plumbing — Inkling 2 global caches (swa none, not the indexer) allocate an
  F16 K/V pair of pools per stream and layer sized to every sequence's newest extent;
  `apply_ubatch` records the batch's written cells and mirrors them at the head of the NEXT metadata pass
  (previous graph final there), validating each against live positions first; state reads and clear reset
  the picture so restores never serve stale stamps.
- `tests/test-llama-kv-mixed.cpp` arms `test-inkling-kvmix-{cpu,cuda0}` — all-hot and all-cold fixtures read
  bit for bit like their single pool, mixed windows take exactly one source per cell, evictions fall back
  to cold exactly, and the LLAMA_KV_STATE_MIGRATE helper units (block layouts, Hadamard composition, the
  converted row reader over shuffled destination runs) ride the same binary.

The mirror today carries the COLD pool's own values decoded and re-encoded F16, so hot precision equals
cold until true pre-quantization capture lands. Reads do not consume the split yet. What remains, in the
order it must land:

1. **Pre-store dual write** — beside `cpy_k`/`cpy_v`'s set_rows into the q4/q8 pool, a second scatter of the
   same pre-store k_cur/v_cur rows into the F16 pool. The call sites in llama-graph.cpp need no edit: chain
   the two scatters (hot scatter fed by the cold node's passthrough output keeps both ordered through one
   returned tensor), with a recency index I32 input built host-side from ubatch positions, on the
   llm_graph_input pattern this file already uses (`set_input_k_idxs`, `llm_graph_input_k_shift`). Ring
   slots then stamp at write time and the flush path retires.
2. **Kernel staging for split reads** — per query row, near cells read the F16 pool and far cells the
   staged low-bit rows; the boundary is the model's own (relative bias exactly zero from 1,024 back). Dense
   path first (concat of the two sources into one f16 staging buffer per span), then rel-runs / one-cell /
   rel-mma variants behind the existing switches, each asserting its node class runs per the kv1b standing
   rule. Forked sessions inside a prefix read their near cells at the low bits until the private tail grows
   past the extent - doc 15 section 3.1's case, stated here as accepted semantics.
3. **Dispatch wiring** — only after 1 and 2, gate `build_attn` to assemble dual-pool inputs under the knob;
   today the knob changes allocation and mirrors alone, which the fixture arms pin.

## kv_stats q4_0 nibble order

While writing the migration decoder against real bytes: ggml's canonical Q4_0 packs byte j low = value j,
high = value j+16 (`quantize_row_q4_0_ref`, ggml-quants.c; device store identical in
ggml/src/ggml-cuda/cpy-utils.cuh `quantize_f32_q4_0_block`; both kernels verified bit-for-bit earlier).
`scripts/inkling2/kv_stats.py` `_dequant_q` interleaved instead (`q[:, 0::2] = lo; q[:, 1::2] = hi`,
"byte j holds values 2j low"), so quantized dumps decoded with half the values on the wrong lanes there.
F16/bf16 rows were unaffected. Fixed 2026-11-22: `_dequant_q` reads the canonical order (values 0-15 from the
low nibbles, 16-31 from the high ones); the test encoder `pack_q4_0`, the hand-built order test, and a new q4_0
round-trip test (`test_q4_0_rows_decode_to_the_encoded_values`) pin that order, and the engine-side claims were
re-verified in ggml-quants.c, cpy-utils.cuh, set-rows.cu and the device readers (fattn-rel-decode.cuh,
fattn-rel-elem.cuh — both already canonical). tests/test-llama-kv-mixed.cpp pins the engine layout with
handcrafted blocks.
