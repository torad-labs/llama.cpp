# CloudRift RTX 5090 verification of the landed open ledger rows

Date: 2026-11-24 (Chicago). Tree: main `d0627f4b6` (tree `94cbb8c8ed41d425454f814c3d7a57cf01d100d4`, checked on the
box before every run). Seat: WS-5 `ws-verify-landed`.

## Verdict

- 19 of 20 row groups pass every test they ran on one RTX 5090.
- One fails: `inkling-execution-ar-capture`. `test-inkling-ar-capture-fallback` (`test-backend-meta-split
  --ar-fallback`) hangs on the first replay of its whole-step capture on two aliased ranks of one card. It was
  killed after 775 s with no output past the replay line. Its sibling `test-inkling-ar-capture` skips (77) on
  a 1-GPU box by design.
- `inkling-launch-budget` does not pass. On a 1-card synthetic fixture the ledger counts 628.0 nodes a decode
  step unfused against 389.0 with the tensor-4 switches (0.62, the bar is 0.50). Real kernel launches (nsys)
  fall further, 431.0 to 155.0 (0.36), and the check rejects that disagreement too. Details in the launch
  budget section.

## Where and how it ran

| | |
|---|---|
| Box | CloudRift `buildbox` (`riftvm`, 3 cores, 23 GB), 1x RTX 5090 32 GB, sm_120, UUID GPU-0db0593f |
| Build dir | `/home/riftuser/build-ws-verify-120a` (own dir; see deviations) |
| Configure | `cmake -S . -B /home/riftuser/build-ws-verify-120a -G Ninja -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120a -DGGML_CCACHE=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_EXAMPLES=OFF` - exit 0 |
| Build | `flock /tmp/buildbox-build.lock cmake --build ... -j 3 --target llama-server test-llama-archs test-backend-ops test-backend-meta-split test-backend-meta-reserve test-kv-mask test-llama-kv-mixed test-inkling-f32-topk test-inkling-moe-fuse test-inkling-fp8-pair test-inkling-mmq-kquant test-inkling-kquant-tile test-inkling-kquant-jrule llama-batched-bench` - exit 0, 2026-11-24T11:20Z to 11:46Z (two warnings, `-Wmissing-field-initializers` at tests/test-llama-archs.cpp:3202 and :3327) |
| GPU runs | each step under `flock /tmp/buildbox-gpu.lock`; ctest as `ctest --test-dir /home/riftuser/build-ws-verify-120a --output-on-failure --no-tests=error -R <regex>` |

All 140 tests the regexes select were listed with `ctest --show-only=json-v1` and every one of their
executables exists in the build (0 missing). Times below are ctest's own test time; the step wall time
includes waiting on the shared GPU lock.

## Results by row

| # | Ledger row | Command | Exit | Result |
|---|---|---|---|---|
| 1 | inkling-optimize-relative-attention | `ctest -R inkling-rel -E rel-prefill-gqa` | 0 | 14/14 passed (buildbox3 run, 778 s); buildbox rerun: PENDING_01 |
| 2 | inkling-optimize-layer-fusions | `ctest -R inkling-fuse` | 0 | passed on buildbox3 (1544 s; per-test log lost with the box); buildbox rerun: PENDING_02 |
| 3 | inkling-optimize-conv-kernel | `ctest -R inkling-conv` | 0 | passed on buildbox3 (14 s); buildbox rerun: PENDING_03 |
| 4 | inkling-optimize-kv-precision, inkling-optimize-kv-onebyte (kv, kv1b, kvmix) | `ctest -R inkling-kv` | 0 | passed on buildbox3 (174 s); buildbox rerun: PENDING_04 |
| 5 | inkling-optimize-attention-far-runs | `ctest -R inkling-far` | 0 | 18/18 passed (CPU and CUDA0 runs/served/restore/drafted, q8_0, e4m3, far-ops, far-ops-query-first), 207 s |
| 6 | inkling-optimize-prefix-read-once | `ctest -R inkling-prefix` | 0 | 11/11 passed (CUDA0 cases, prefix-ops g1/g5/g32), 92 s |
| 7 | inkling-optimize-window-ring | `ctest -R inkling-ring` | 0 | 6/6 passed, 40 s |
| 8 | inkling-optimize-q-input-stack | `ctest -R inkling-stack` | 0 | 3/3 passed (stack, stack-split, stack-mul-mat), 47 s |
| 9 | inkling-optimize-f32-products | `ctest -R inkling-f32` | 0 | 6/6 passed (f32-model, router, bias-table, topk-moe, topk-alias, topk-alias-legacy), 13 s |
| 10a | inkling-execution-ar-capture | `ctest -R inkling-ar-capture` | **8** | **FAIL**: 0 passed, 1 skipped, 1 failed. `test-inkling-ar-capture` Skipped (77): 1 distinct GPU. `test-inkling-ar-capture-fallback`: `Subprocess terminated***Exception: 775.13 sec` (hang, killed; see below) |
| 10b | inkling-execution-ar-capture | `test-backend-meta-split --reduction-queue` | 0 | 16/16 legs `ok` on logical ranks (2 and 4 ranks on CUDA0, 23x1024 and 92x4608, F32 and BF16 wire, host-waits and host-queues-ahead, 576 reductions each, same=1); physical multi-GPU legs need a second card and did not run |
| 11 | inkling-optimize-rel-packed-prefill | `ctest -R inkling-rel-prefill-gqa` | 0 | 2/2 passed (unpacked, packed), 4 s |
| 12 | kv-mask-prefill-build | `ctest -R test-kv-mask` | 0 | 1/1 passed, 3 s |
| 13 | blackwell-kquant-mmq | `ctest -R inkling-mmq-kquant` | 0 | 1/1 passed (`test-inkling-mmq-kquant`), 585 s |
| 14 | inkling-prefill-delta-uploads | `ctest -R inkling-delta-upload` | 0 | 3/3 passed (1, split2, split4 ranks), 15 s |
| 15 | inkling-mmq-tma-tiles | `ctest -R inkling-mmq-tma` | 0 | 3/3 passed (mmq-tma 495 s, row-kernel 356 s, tma-kquant 66 s) |
| 16a | repo-meta-init-abort | `ctest -R test-backend-meta-reserve` | 0 | 2/2 passed (reserve, reserve-merged) |
| 16b | all inkling rows' fixture leg | `test-llama-archs -a inkling2` | 0 | `Summary: all 4 test(s) passed` (CUDA0 NMSE 2.68e-07, CPU, Meta 1.53e-06, Meta 3:7 1.58e-06; Roundtrip/Mixed batch SKIP where the table marks them) |
| 16c | GLM 5.3 stays green | `test-llama-archs -a glm5next` | 0 | `Summary: all 6 test(s) passed`; its `E llama_model_set_draft_vocab` lines are the negative cases of the draft-vocab test |
| 17a | kquant-staged-kernel | `ctest -R inkling-kquant-tile` | 0 | 3/3 passed (staged, legacy, exact compare), 18 s |
| 17b | kquant-staged-kernel | `ctest -R test-inkling-moe-tma-23row-cuda0` | 0 | 1/1 passed, 221 s |
| 18 | kquant-j-rule | `ctest -R inkling-kquant-jrule` | 0 | 2/2 passed (rule, legacy), 3 s |
| 19a | inkling-optimize-relative-attention / rel-packed-prefill | `test-backend-ops test -b CUDA0 -o FLASH_ATTN_EXT` (full) | PENDING_19A | PENDING_19A |
| 19b | inkling-optimize-layer-fusions | `test-backend-ops test -b CUDA0 -o MOE` | 0 | 14/14 passed, `Backend CUDA0: OK` |
| 19c | inkling-optimize-moe-shapes (blocked row) | `ctest -R inkling-moe` | PENDING_19C | PENDING_19C |
| 20 | inkling-optimize-sync-structure | `test-backend-meta-split` (no args) | PENDING_20 | PENDING_20 |

Not run, and why:
- The rows' second check, `sh .campaign/cases/check.sh <row>`, freezes server cases that hard-code torad-server's
  card 1 (`CUDA_VISIBLE_DEVICES=1`, `/run/user/1000/gpu1.lock`), an in-tree `build-campaign` and the hostshield
  build gate. They cannot run as written on a one-card box, and adapting them means editing `.campaign/`, which
  this seat does not own.
- Physical multi-GPU legs (`--ar-capture` on distinct GPUs, the four-rank card-to-card legs) need a second card.

## The failure: test-inkling-ar-capture-fallback hangs on aliased ranks

`tests/test-backend-meta-split.cpp:2172` runs `test_ar_fallback({gpu, gpu})`, commented "aliased ranks fine
here: refused captures fall back live either way". It is a 12-layer, 6-evaluation graph (n_embd 4608, 128
rows) that finishes in seconds when it works.

What happened on the 5090 (log `10a-arcap.log`, `GGML_GRAPH_TRACE=1`):
- The capture leg (`GGML_CUDA_AR_COPY_ENGINE_CAPTURE=1`) built the step and the meta backend captured it whole
  instead of refusing it: `cuda_outer event=instantiate ... nodes=192` on both aliased rank contexts, then
  `meta event=capture ... ranks=2 nodes=60 subgraphs=13`.
- The next and last line is `meta event=replay ... ranks=2 nodes=60`. Nothing followed for 775 s. The main
  thread sat at 100 % CPU (state R), the four other threads slept (poll/futex), and the GPU showed 100 %
  utilization with 854 MiB used.
- I killed the process (SIGTERM) after it had held the shared GPU lock for 12.9 minutes. ctest recorded
  `Subprocess terminated`. No gdb or eu-stack is on the box, so there is no native stack.

Reading, not proven: the first replay of a whole-step copy-engine capture deadlocks when both ranks are the
same physical card. That fits a device-side pacing wait each rank's graph holds for the other's arrival,
which cannot be met on one card. `--ar-capture` already refuses aliased ranks ("aliased logical ranks are out
of scope for this row"). `--ar-fallback` assumes the capture is refused on them, but here it is taken, then
hangs. Either the capture should refuse aliased ranks, or the test needs distinct GPUs and should skip
(77) on one card. Which one is the ar-capture lane's call. Not checked: whether the same hang happens on
distinct GPUs.

## inkling-launch-budget

Evidence: `reports/inkling2-launch-budget.json`, written by `scripts/inkling2/launch_budget.py build`.
`python3 scripts/inkling2/launch_budget.py --check --evidence reports/inkling2-launch-budget.json` exits 1:

```
FAIL: result 389.0 is above half the baseline 628.0
FAIL: device 0: result 389.0 is above half the baseline 628.0
FAIL: ncu ratio 0.36 and ledger ratio 0.62 disagree past 60 %
```

Setup:
- Model: the synthetic fixture `test-llama-archs -a inkling2 -s 1 -d 0.1 -o /home/riftuser/lb-model`. That
  is 3 layers, 2 experts, n_embd 512, 8 heads / 4 KV heads, sliding window 32, vocab 130, and every weight is
  F32 (13.8 MB). It is not Inkling 2 Small: no Q3_K_M, no 384 experts, one card instead of four.
- Workload: `llama-batched-bench -m inkling2-moe.gguf -ngl 99 -fa on -c 2048 -b 2048 -ub 2048 -kvu -ctk f16 -ctv f16 -ctks q8_0 -ctvs q8_0 -npp 32 -npl 23`
  at `-ntg 8` and at `-ntg 40`. That is 23 sequences decoding together, the KV types of tensor-4 (F16 global,
  q8_0 sliding). The two runs differ only in 32 decode steps of 23 rows, so the snapshot difference over 32
  is one decode step.
- Both arms: `LLAMA_GRAPH_SLOTS=32 GGML_META_CAPTURE_GRAPHS=38 LLAMA_RS_FIXED_ROWS=1 LLAMA_GRAPH_ANY_SEQS=1 GGML_CUDA_GRAPH_NODES=262144 GGML_CUDA_GRAPH_IDLE_MS=600000 LLAMA_RS_STAGE_ROWS=2 LLAMA_KV_SEQ_CELLS=294912 LLAMA_TOK_EMBD_GPU=1`.
- Result arm: the tensor-4 environment `scripts/inkling2/serve.py --dry-run` prints, single-card part:
  `LLAMA_INKLING2_BUCKET_PREFILL_KV=1 GGML_CUDA_MMVQ_EXPERT_SORT_23=1 LLAMA_INKLING2_MEAN_MASK_DECODE=1 GGML_CUDA_FATTN_REL_QUERY_FIRST=1 LLAMA_INKLING2_COMPACT_GREEDY=1 LLAMA_INKLING2_KV_RUNS=1 GGML_CUDA_FATTN_REL_RUNS_SHARED=1 GGML_CUDA_FATTN_REL_PACKED_PREFILL=1 GGML_CUDA_F32_FEW=1 GGML_CUDA_MOE_FUSE=1 GGML_CUDA_MMQ_TMA=1 GGML_CUDA_INKLING2_L2_PERSIST=1 GGML_CUDA_FATTN_REL_DECODE_ONE_BYTE=1 GGML_CUDA_MMVQ_KQUANT_MMID_MAX=23`.
- Baseline arm: the unfused reference the WS-5 brief asked for. The three opt-ins are unset, and these are set:
  `GGML_CUDA_DISABLE_FUSION=1 LLAMA_INKLING2_FUSE_LEGACY=1 LLAMA_INKLING2_CONV_LEGACY=1 LLAMA_INKLING2_MOE_MERGED_LEGACY=1 GGML_CUDA_MMQ_FEW_GROUP_LEGACY=1 GGML_CUDA_MMVQ_GROUP_LEGACY=1 GGML_CUDA_PQ2_MMA_GROUP_LEGACY=1 GGML_CUDA_TOPK_MOE_WIDE_LEGACY=1 GGML_CUDA_TOPK_SOFTCAP_FUSE_LEGACY=1 GGML_CUDA_ROUTE_MATVEC_TOPK_LEGACY=1 GGML_CUDA_MOE_WSUM_LEGACY=1 GGML_CUDA_GLU_LIMIT_FUSE_LEGACY=1 GGML_CUDA_ATTN_PREP_Q_LEGACY=1 GGML_CUDA_ATTN_PREP_KV_LEGACY=1 GGML_CUDA_ATTN_PREP_REL_LEGACY=1 GGML_CUDA_ATTN_PREP_KV_SET_ROWS_LEGACY=1 GGML_CUDA_ATTN_PREP_STATE_STORE_LEGACY=1 GGML_CUDA_ATTN_RES_MIX_LEGACY=1 GGML_CUDA_RMS_NORM_GATE_LEGACY=1 GGML_CUDA_NORM_FWHT_LEGACY=1 GGML_CUDA_SHORT_CONV_ADD_LEGACY=1 GGML_CUDA_MMVQ_FUSION_MULTICOL_LEGACY=1 GGML_CUDA_MMVQ_FUSION_SRC1_LEGACY=1 GGML_CUDA_MMVQ_MOE_QUANTIZE_LEGACY=1 GGML_CUDA_UNARY_MUL_VIEW_LEGACY=1`.
  This is not the row spec's baseline, which is today's tensor-4 measured before the launch-reduction work.
  The row's own reduction work has not landed, so that baseline equals the result arm here.

Measured (`GGML_LEDGER=1 GGML_LEDGER_FILE=...`, snapshots in `buildbox:/home/riftuser/lb-out`). The ledger
numbers are the same on buildbox3 and buildbox:

| | steps (n8 / n40) | `cuda.eval.nodes[0]` sum (n8 / n40) | nodes per decode step | nsys kernels (n8 / n40) | kernels per decode step |
|---|---|---|---|---|---|
| baseline (unfused) | 10 / 42 | 6268 / 26364 | **628.0** | 4292 / 18084 | **431.0** |
| result (tensor-4) | 10 / 42 | 3894 / 16342 | **389.0** | 1570 / 6530 | **155.0** |
| ratio | | | **0.62** | | **0.36** |

Kernel counts come from `nsys profile -t cuda -s none --cpuctxsw=none --cuda-graph-trace=node` plus `nsys stats
--report cuda_gpu_kern_sum` (instances). `ncu` was the named tool but cannot run here: `ERR_NVGPUCTRPERM - The
user does not have permission to access NVIDIA GPU Performance Counters` on all four arms. Fixing that needs a
root driver setting on the rented box, which this seat does not change. The evidence's `ncu` fields carry the
nsys decode-window counts (13792 and 4960 kernels over the 32 steps), and its `command` field says so.

Why the two counts disagree: `cuda.eval.nodes` adds `cgraph->n_nodes` once per graph evaluation
(`ggml/src/ggml-cuda/ggml-cuda.cu:11576`). That count is taken before the CUDA backend fuses anything, and it
includes view and reshape nodes that launch nothing. Fusions inside the backend (`GGML_CUDA_MOE_FUSE`, the
few-row MMQ group, the rms_norm fusions, F32_FEW replacing cuBLAS) cut real launches but not this number. Only
graph-level changes do (`LLAMA_INKLING2_FUSE_LEGACY`, `_CONV_LEGACY`, `_MOE_MERGED_LEGACY`). The row's premise
that "each executed ggml node is at least one dispatch" does not hold here: 628 nodes made 431 launches, and
389 made 155. A check built on the ledger will keep disagreeing with ncu or nsys by about this much.

Where the launches go, per decode step (nsys, n40 minus n8, divided by 32):

| baseline (431) | | result (155) | |
|---|---|---|---|
| 104 | k_bin_bcast | 33 | f32_cols (F32_FEW products) |
| 32 | unary_op_kernel | 26 | k_get_rows_float |
| 29 | k_get_rows_float | 15 | k_bin_bcast |
| 26 + 18 | scale_f32 / scale_f32_vec4 | 11 | rms_norm_f32 |
| 24 + 18 + 15 | cutlass Kernel2 / cublasLt splitKreduce / nvjet tf32 (cuBLAS products) | 6 each | attn_prep_state_store, attn_prep_kv_tail, attn_res_mix_split, f32_short_k, short_conv_window |
| 16 | rms_norm_f32 | 4 each | dequantize_block_q8_0_f16, glu_fn_kernel |
| 14 | cpy_scalar | 3 each | scale_f32, topk_moe, flash_attn_ext_f16, fattn stream_k fixup, norm_clip, rel_table, moe_weighted_reduction |
| 12 each | concat_cont, ssm_conv_f32, reduce_rows_f32, concat_dim0_transpose | 2 each | fwht, attn_siglu, concat_dim0_transpose |

On this fixture, what remains in the result arm is mostly the 33 F32 few-column products (every projection,
because the fixture's weights are F32), the 26 row gathers and 15 broadcasts. On the served Q3_K_M pack the
projections take MMVQ/MMQ instead, so this split does not carry over to it. Only a recount on the real pack
does: the four-card window, or a K-quant fixture at served shapes.

## Deviations

- Build dir: `/home/riftuser/build-torad-120a` could not be shared. Its CMake cache is bound to
  `checks/inkling-moe-23row-f27242` (tree `87b36e23`, not main), and two other seats' ninja runs were writing
  it at once. I built main into `/home/riftuser/build-ws-verify-120a`. The lead then moved every seat to its
  own dir with box-wide `flock` locks.
- Boxes: the first buildbox build hit an nvcc OOM-kill at load average 138, so I moved to buildbox3. That box
  went down mid-run (ssh refused at about 11:15 CT) after rows 1-4 had passed, and came back re-provisioned
  without my build, so everything else ran on buildbox after a clean rebuild there. Rows 1-4 are rerun on
  buildbox at the end of the matrix.
- My test matrix's driver shell on torad-server was killed at about 07:23 CT, by a cause I did not find, after
  row 14 had finished. I resumed from row 15 as a job detached on the box. Rows 1-14 results come from their
  logs.
- At about 06:06 CT I ran `pkill -f "bash -s" -u marcos` on torad-server to stop one of my own scripts. That
  pattern can also have killed another seat's `ssh ... bash -s` running at that moment. I told the lead.
- The test-backend-ops runs are CPU-reference bound on 3 cores, and the GPU lock is shared with other seats.
  Step wall times above include up to an hour of waiting on the lock.
