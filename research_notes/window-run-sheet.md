# Four-card window run sheet (2026-11-22)

## Window result (box 61473783)

Window closed early: Marcos ordered the box stopped at 06:24 CT and it is being destroyed; the q4 arm died mid-run with the disk.
Arms measured at window-base 2aea42018 (clean checkout, meta LEGACY pin, f16 KV, 23-session probe):
- stock (raw defaults): 307.53 tok/s, p99 80.8 ms
- profile (query-first + compact greedy): 459.71 tok/s, p99 57.0 ms — against the old 425.605 / 65.096 floor: +8.0% tps, -12.4% p99
- profile-wire-q8 (profile + GGML_CUDA_AR_WIRE_Q8=1): 460.57 tok/s, p99 52.8 ms — +0.19% tps (inside single-arm noise), p99 -4.2 ms
- profile-q4 (q4_0 global KV + swa f16 + FATTN_REL_DECODE_Q4=1): ABANDONED mid-prefill, no result; KL vs f16 not run.
Review finding accepted (coordinator, 06:2x CT): at the window base the fusion, TMA, attention-prep and route kernels have no connectors in ggml-cuda.cu, so those arms could not change serving; the three arms above measure the existing serving path only.
Post-window: no GPU time until the joint carve lands; every switch ships with a behavioral check that proves ON vs OFF changes results.

One rental, five blocks, every arm prints `window-base=<sha>` and its full `GGML_CUDA_*` switch block.
Baseline pins on every stock-comparable arm: `GGML_LEDGER=0`, `GGML_CUDA_ATTN_RES_MIX_LEGACY=1`.
Tree patch (tracked diff + untracked files) saved beside the outputs of each arm.
Pre-run checks at rental: `git rev-parse HEAD` recorded, `git status --porcelain` empty of
non-window files, `git diff HEAD -- ggml/src/ggml-cuda/fattn.cu` empty (REL_ROWS absence),
box ref `refs/heads/window-base` at the frozen sha (anchor moved to ed36e2572 on 2026-11-22: 672f9f0a9 migrate-on-load + ed36e2572 llama-kv-mixed registration; 92705c440 lineage superseded).

## A0. No-loss floor (block 0)
- replay_compare stock arms, two runs (expected ≈425.6 tok/s at 16K, p99 ≈65 ms)
- tensor-4 launcher sanity: profile serves, one 23-session step, logits finite, kv_stats state ok

## A1. Group arm (block 1) — every staged switch at its landed default, then row pairs
- capture on/off: GGML_CUDA_AR_COPY_ENGINE_CAPTURE=1 / 0 (=_LEGACY=1 reverts)
- packed prefill on/off: GGML_CUDA_FATTN_REL_PACKED_PREFILL=1 / 0
- fusions pair (when their staging lead hands the list)
- prefill D1 pair (prefill-sp + AR_GATHER_Q8_1 arms, when staged)
- experts pair: GGML_CUDA_MMQ_TMA=0/1 (mmq-tma-tiles.*, staging to qgre-engine-labs pipe)
- mini fusion E1 arms (corrected seven, verified live in tunables.toml): GGML_CUDA_ATTN_PREP_Q_LEGACY, GGML_CUDA_ATTN_PREP_REL_LEGACY, GGML_CUDA_ATTN_PREP_KV_LEGACY, GGML_CUDA_ATTN_PREP_STATE_STORE_LEGACY, GGML_CUDA_ROUTE_MATVEC_TOPK_LEGACY, GGML_CUDA_FWHT_SIGLU_Q81_STAGED_LEGACY, LLAMA_INKLING2_MOE_MERGED_LEGACY — each sets 1 for the legacy arm, unset for the new path (all default to the new path)

## K1/K2 (block 2) — doc 18, names frozen by inkling-quant (5130b84bc)
- A stock-f16-wire-bf16 (floor), B stock-repeat (floor noise), E wire-q8-on-f16 (factorial arm)
- F fp8-accum-on-f16 (COUNTERSIGNED; factorial beside E, KL-parity discipline; model-level fixture runs with the switch ON and type_KV=f8_e4m3)
- kv_stats two-prompt state saves ride arm A only (--slot-save-path per script USAGE)
- K2 format ranking from those saved states

## K3 (block 3) — doc 18
- C q8-global-swaF16, D q4-global-swaF16 (dense-path decode, Q4 switch off: measures the cache not the kernel)
- early/middle/late depth with the late-third captured at twice early/middle weight
- coding_probe.py on the winning format
- q4_0 arms re-check fattn.cu == HEAD at run start (REL_ROWS absence)

## KQ4 (conditional, rides K2/K3 when landed)
- q4_0 global KV + GGML_CUDA_FATTN_REL_DECODE_Q4=1 vs q8-global-swaF16, inkling-quant's rel-mma q4 dispatch is
  committed at 31047b0ba; GREEN GATE SATISFIED 2026-11-22: GGML_CUDA_FATTN_REL_DECODE_Q4=1
  test-backend-ops FLASH_ATTN_EXT type_KV=q4_0 → 17/17 vs host reference, rc 0 (sources byte-identical
  to 31047b0ba); kv_stats q4 nibble order fixed at 5cea311fd, 55 tests + 6 subtests green; the arm
  rides K2/K3

## Context: prefix read-once (for interpreting block 1)
- 23-session storage is already shared: one unified pool, prefix_copy on, 196,608 of 262,144 tokens
  shared in the modeled workload; the READ is not shared. Read-once is worth up to ~10 ms a step.

## Search (block 4)
- hyperparameter search on the same merged green tree, window-base printed per trial arm

Parked: K4 (F16 vs chosen format at 23 sessions) until inkling-optimize-kv-gates lands.

## Ownership
- Driver: one hand (mine or rig-lead) — the same person for every block
- KV interpretation: inkling-quant-glm (K1-K3 delta + KL verdicts)
- Group-arm interpretation: me (no-loss vs 425.6, per-row deltas)
- Search setup: hyperparam seat per the search-methods-upgrades gate

## Staging record
- tests/test-backend-ops.cpp: TWO sequential hunk-level commits, each named per lane (coordinator's 2026-11-22 revision; rel_rows kernel promoted at c6d40f736): mini cuts prep/registrations/rider-block first, then I cut my four gqa cases + the whitespace hunk (extracted as a hunk-filtered patch applied with git apply --cached against mini's landed version). No joint commit, no exclusions — the rider is approved content now.
- tests/test-backend-meta-split.cpp: capture lane's, in my staging hash.
- REL_ROWS rider: APPROVED for inclusion (serving verdict +5.2% apart / +9.9% shared-prefix over f16-mask; +2.7/+4.4% over packed dense; --kv-unified ON load-bearing). It re-enters the joint commit from research_notes/rel-rows-rider-fattn.patch + rel-rows-rider-tests.patch, and ggml/src/ggml-cuda/fattn-rel-rows.cuh becomes a tracked new file; fattn.cu wire re-applied by mini under its staging hash. Exclusion no longer applies.
- Post-window handoff: send rig-lead the frozen window-base sha + the rental's `rig build --prebuilt` tarball path, for gate/publish/pin of the head entry (rig-dev#472).
- Post-window handoff #2: the arm-A saved slot state (.state path) goes to inkling-quant for the --eff-dim participation-ratio pass (9f4226d43) in the same exchange as the window numbers.

## Known issues (external to staging)
- test-llama-archs -a inkling2 rc=134 on the META-BACKEND row only: ggml-backend-meta.cpp:939 GGML_ASSERT(ggml_backend_buffer_is_meta(tensor->buffer)) through llama_init_from_model; CUDA0/CPU rows pass, -a inkling2_loop passes. Owned by the META lane (ggml-backend-meta.cpp, reserve sizing from f0735a148 is the prime suspect), NOT capture/prefill/fusion; verbatim text supplied by inkling-mini 2026-11-22. Acceptance runs treat that row as a known external red and report it, not reopen it.
- Staging hash (this lane): 4427251e4 (mmvq overload completion so the branch links — mmvq group's half-applied refactor unblocked, theirs to absorb post-window) + d0d0b2e03 (capture: allreduce.cu de-staticked route + --ar-capture test) and 38107c8a2 (gqa op cases, hunk-filtered); on top of mini's 05e2c889f kernel set and the a3faae9c4 lineage; branch HEAD at cut includes all committed staging.
- Capture end state 2026-11-22 03:5x: LEGACY leg green; CAPTURE leg accepted the route then failed at capture_end 'capturing stream has unjoined work' (CUDA text); builder bisecting by disabling the intra-session two-back pacing edge; acceptance re-runs both legs at the frozen hash.
- Execution integrity: every window arm builds and runs from a CLEAN checkout at the frozen window-base sha (rig build --sha / detached window-base ref), never from the dirty shared working tree; switch values applied as env at run time only.
- FP8 lane on origin: commits 5c34493ee and 8e764290b are inert with GGML_CUDA_FATTN_FP8_ACCUM off; any window-base at or after them builds arms that never enter the new paths. Wiring stays uncommitted (post-window). Builders reaching for block-scaled MMA must target sm_120a, never plain sm_120 (ptxas refuses).
- Conditional rider (approved): chat-inkling's route-ID JSONL dump behind a default-off switch in the sorted-dispatch path; if staged before the cut it rides the window arms — one pass answers the route-histogram row, doc 21 §4's depth gap and EXPERT_SPLIT placement. Off by default; only its dedicated arm turns it on.
- Packed-prefill served arm: DEFERRED to post-window integration (admission bound coupled to partial-alloc; widening is its own audited change). Window evidence = direct ctest arm green + one ncu case naming fattn_rel_prefill_gqa_packed_tiles<128>.
- Duration estimate (coordinator): full sheet 11-12 h; measurement blocks alone (0-3) about 6 h. Option: one rental for blocks 0-3 then stop (policy) and run the search block in a second rental at the same frozen sha.
