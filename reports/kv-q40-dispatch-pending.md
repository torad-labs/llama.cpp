# Q4_0 relative decode — dispatcher patch pending the open merge

Companion to `docs/torad/research/inkling-2-small/18-global-kv-off-f16.md` section 3.1 and ledger rows
`1454bba3a` (rel-mma reads q8_0 one byte), `15a502bc5`/`4e5911055` (raw path). Written while
`MERGE_HEAD` a2861a62 (origin/main into draft-speed) is unresolved; nothing here commits or touches
the frozen files.

## What landed already (uncommitted, editable files only)

- `ggml/src/ggml-cuda/fattn-rel-elem.cuh`: template instance `fattn_rel_elem<GGML_TYPE_Q4_0>` beside
  F16/Q8_0/E4M3, helper `fattn_rel_elem_q4(type)` behind env switch `GGML_CUDA_FATTN_REL_DECODE_Q4`
  (off by default; plain env reading applies — the name declares no knob in tunables), and
  `fattn_rel_elem_align(Q4_0) = 8` beside q8_0's (rows of 72 bytes are not multiples of 16).
  `fattn_rel_elem_one_byte` itself is unchanged, so ONE_BYTE keeps its exact old meaning.
- `ggml/src/ggml-cuda/fattn-rel-decode.cuh`: `fattn_rel_decode_value<GGML_TYPE_Q4_0>` branch and a
  Q4_0 case in the one-cell kernel's pick switch. This is the fallback that also takes query groups.
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

## tunables.toml / ggml-tunables.inc — none now

Both are frozen. `ggml_env_switch` reads undeclared names straight from the environment
(`ggml-tunables.cpp:427`), so no table entry is needed to run. When doc 18 section 3.3 switches the
profile default, decide then whether `GGML_CUDA_FATTN_REL_DECODE_Q4` becomes a knob or disappears
behind an unconditional acceptance like q8_0's.

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
