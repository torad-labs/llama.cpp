# Inkling 2 Small - 4x RTX PRO 6000 S Combined Stack v3 (PRs #305-#308) Validation

- **Date:** 2026-11-27 (01:24 - 01:56 AM CT).
- **Host:** Vast instance `62821112`, machine 57498 (4x NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB each, AMD EPYC 9554 64-Core, 256 threads visible, driver 610.57.04, Ubuntu 24.04). Destroyed after the pull: `0.53 h`, `~$3.76` at `$7.100/h`, gone from `rig vast status`.
- **Build:** `main` at `bab61e7ee` (PRs #305, #306, #307, #308), the same binaries that passed the 13 fast checks on buildbox (`build-main-120a`, sm_120a, CUDA 13.0, GNU 13.3, `GGML_NATIVE=ON`), copied to the box (sha256 of `llama-server` and `libggml-cuda.so` equal on both). The version string reads `36d47669f`: that is on-box's commit of the tree as it was, and its tracked files do not differ from `bab61e7ee`.
- **Model pack:** as v2 (`Inkling-2-Small-abliterated-e4853745-Q3_K_M-imatrix.gguf`, its mmproj and the tuned DSpark BF16 draft), byte lengths checked.
- **Workload:** as v2: `cc_turns_bench.py` over 40 real Claude Code 2.1.258 sessions, 25 tool schemas, 1/4/8/23 users x 4 turns at `max_tokens=256`, then 23 users x 2 turns at `max_tokens=128`. One trial an arm, no repeats.
- **Arms:** (1) `tensor-4` as it is now, where `f8_e4m3` + `--kv-mixed` is the default (#305); it matches v2 arm 2. (2) TP4 DSpark + kv-mixed with v2 arm 3's flags. (3) one `nsys` capture at 23 users on arm 1's plan.
- **Compared with:** `reports/inkling2-4card-v2-combined-toks.md` (2026-11-25, Vast `62493016`, engine `e76f2ad`).

## What differs from v2 besides the four PRs

Read every delta below with these in mind; none was controlled.

- Another Vast machine: EPYC 9554 here, EPYC 9555 in v2. Host-side costs a call were higher on this box: `stage.cpu_sample` 449 us against 152 us, `stage.cpu_verify` 1,617 us against 442 us, `cuda.eval.replay_us` 39 us against 26 us on ranks 1-3.
- Another build of the engine: buildbox's native build here, rig's portable prebuilt in v2.
- v2's `nsys` capture ran the F16 plan; this one ran the kv-mixed default. Their decode rates are not comparable; their kernel and API shares are.
- One trial an arm in both runs.

## 1. Later-turn aggregate decode throughput (tok/s)

| Users | v2 kv-mixed | v3 default (kv-mixed) | Change | v2 TP4 DSpark | v3 TP4 DSpark | Change |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 50.36 | 51.39 | +2.0% | 78.85 | 75.79 | -3.9% |
| 4 | 106.46 | 120.08 | +12.8% | 155.29 | 159.11 | +2.5% |
| 8 | 139.95 | 147.51 | +5.4% | 178.71 | 180.18 | +0.8% |
| 23 (4 turns) | 101.40 | **126.63** | **+24.9%** | 77.20 | 74.24 | -3.8% |
| 23 (warm, 2 turns) | 225.93 | 221.42 | -2.0% | 150.04 | 177.38 | +18.2% |

### Full v3 metrics

| Arm | Level | Per-stream (tok/s) | Aggregate (tok/s) | E2E (tok/s) | ITL p50 / p95 / p99 (ms) | TTFT turn 0 / later (ms) | Cache hit | Valid tools | Wall (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| default (kv-mixed) | 1u x 4t | 51.39 | 51.39 | 22.23 | 16.44 / 16.78 / 432.49 | 7,663 / 821 | 97.4% | 4 / 4 | 17.59 |
| | 4u x 4t | 30.02 | 120.08 | 47.38 | 20.00 / 143.38 / 654.02 | 13,457 / 1,630 | 96.5% | 15 / 16 | 37.08 |
| | 8u x 4t | 18.44 | 147.51 | 88.79 | 21.20 / 395.50 / 783.76 | 3,446 / 1,967 | 96.7% | 29 / 31 | 39.48 |
| | 23u x 4t | 5.51 | 126.63 | 80.90 | 35.06 / 926.29 / 1,390.44 | 14,145 / 3,485 | 97.2% | 81 / 85 | 116.48 |
| | 23u warm 2t | 9.63 | 221.42 | 101.66 | 36.84 / 538.76 / 1,434.73 | 12,821 / 2,139 | 99.9% | 46 / 46 | 32.32 |
| TP4 DSpark + kv-mixed | 1u x 4t | 75.79 | 75.79 | 21.03 | 0.07 / 36.97 / 185.70 | 12,389 / 968 | 97.4% | 4 / 4 | 21.58 |
| | 4u x 4t | 39.78 | 159.11 | 38.43 | 0.06 / 228.47 / 730.75 | 15,534 / 3,013 | 96.5% | 16 / 16 | 35.34 |
| | 8u x 4t | 22.52 | 180.18 | 92.46 | 0.06 / 485.41 / 749.16 | 3,795 / 2,486 | 96.7% | 31 / 32 | 34.71 |
| | 23u x 4t | 3.23 | 74.24 | 57.40 | 0.05 / 1,477.88 / 2,198.82 | 12,106 / 8,964 | 95.7% | 84 / 85 | 159.05 |
| | 23u warm 2t | 7.71 | 177.38 | 64.43 | 0.06 / 1,426.08 / 2,670.98 | 20,538 / 4,492 | 99.6% | 46 / 46 | 49.32 |

- Sanity probe `19*207 = 3933`: exact. Tool-call validity over the 4-turn sweeps: default 129 / 136 (94.9%; v2 kv-mixed 133 / 135), DSpark 135 / 137 (98.5%); 46 / 46 on both warm passes.
- DSpark acceptance: 12,119 of 15,057 draft tokens (80.5%; v2 78.3%).
- VRAM at the end of the sweep: default 88,809 / 88,081 / 88,081 / 88,081 MiB used (8.3-9.0 GiB free a card); DSpark 91,743 / 91,015 / 91,015 / 91,015 MiB (5.4-6.1 GiB free). No `no_memory` line in either ledger. No error line in any server log.

## 2. What each PR shows on four cards

**#305, kv-mixed as the `tensor-4` default: holds.** The profile with no KV flags loads `f8_e4m3` + mixed, with 10.0-10.7 GiB free a card at load and 8.3-9.0 GiB at the end of the sweep, and no `no_memory` fallback.

**#306, batched checkpoint restore: the sync storm is gone.** In the `nsys` capture, which again covers the start of turn 2's restores:

| | v2 | v3 |
|---|---:|---:|
| `cudaStreamSynchronize` | 91,536 calls, 3,085.5 ms | 1,988 calls, 230.5 ms |
| `cudaMemcpy2DAsync` | 73,604 calls, 422.0 ms | not in the top API calls |
| Host-to-device copies | 146,411 calls, 13.15 GB, 2,268.5 ms | 64,109 calls, 14.80 GB, 366.6 ms |
| Longest stretch with no GPU kernel (same query on both captures) | 1.46 s | 0.49 s |

The 23-user 4-turn aggregate rose 24.9% (101.40 to 126.63 tok/s), which is what a shorter all-slot freeze at each turn start would give. The later-turn TTFT median at 23 users did not move (3,485 ms against 3,490 ms), so the wait a user sees at a turn start is held by something other than these copies.

**#307, part 1, the relative-bias table: holds.** `attn_prep_rel_table` went from 356.1 ms (7.1% of GPU kernel time, 40.5 us a call) to 29.4 ms (0.6%, 3.0 us a call) as `attn_prep_rel_table_n16_f32`.

**#307, part 2, batched CE waits: no gain measured.** `ar_ce_wait_geq` is gone (48,663 calls before), but `ar_ce_wait_new` grew (41,751 calls and 294.0 ms to 50,337 calls and 397.7 ms) and `ar_ce_wait_peers` is new (19,179 calls, 103.7 ms). All `ar_ce_*` kernels together: 614.7 ms (12.2%) to 688.8 ms (14.2%). A decode step's worth, taking calls of `mul_mat_q_tma_q3_K` as the count of steps captured: 37.2 us to 36.2 us for each such call, flat. `ggml_cuda_arrival_wait_kernel` fell from 128.0 ms to 1.3 ms.

**#308, DSpark at 23 users: the gain it aimed for is not there.** 23 users x 4 turns: 74.24 tok/s against 77.20, and still under the undrafted default (126.63). From the ledger:

| DSpark arm, a step | v2 | v3 |
|---|---:|---:|
| `decode.alloc_graph` rate | 0.98 | 0.75 |
| `decode.alloc_graph` mean | 27.6 ms | 41.4 ms |
| `decode.alloc_graph` a step | 27.0 ms | 30.9 ms |
| `graph.kept.miss.no_shape` | 0.44 | 0.20 |
| `graph.kept.miss.refused` | 0.54 | 0.55 |
| `cuda.eval.direct.properties_changed[0]` | 37.5 | 36.2 |
| `cuda.outer.instantiate_us[0]` a step | 13.1 ms | 17.2 ms |
| `stage.target_eval` a step | 220.2 ms | 255.4 ms |
| `stage.iteration` a step | 341.3 ms | 353.6 ms |

The pow2 feed chunks cut the misses for a shape that was never kept by half. The refusals of a kept graph did not change, a graph is still allocated on three steps in four, and `properties_changed` did not move. The warm 2-turn pass did gain (150.04 to 177.38 tok/s), and 1, 4 and 8 users are within 4% of v2. This box's slower host calls weigh most on this arm, so the -3.8% itself is not evidence of a loss; the unchanged refusal and allocation rates are the finding.

## 3. `nsys` capture at 23 users (default plan)

659,257 kernels and 4,842 ms of GPU kernel time over 4.40 s on four cards. As in v2, the server under `nsys launch --cuda-graph-trace=node` stopped answering when turn 2 was admitted (the bench got connection refused; the server log ends on `launch_slot_` lines with no error). v2 put that on the F16 plan's 61 MiB of free memory on `CUDA0`; this run had over 8 GiB free a card, so that cause does not explain it here. The two arms without tracing ran their 4-turn sweeps to the end. Cause not found.

Steady decode in the traced pass before that: 22.68 tok/s a stream on 23 streams, 44.10 ms a token (v2 on F16: 26.35 tok/s, 37.95 ms).

| Share | Total (ms) | Calls | Avg (us) | Kernel |
|---:|---:|---:|---:|---|
| 16.3% | 790.2 | 19,014 | 41.6 | `mul_mat_q_tma_q3_K` |
| 9.4% | 453.9 | 19,011 | 23.9 | `mul_mat_q_tma_q45_K` |
| 8.2% | 397.7 | 50,337 | 7.9 | `ggml_cuda_ar_ce_wait_new` |
| 7.8% | 377.1 | 48,662 | 7.7 | `mul_mat_q_few` |
| 6.0% | 291.4 | 7,484 | 38.9 | `fattn_rel_runs_swa_direct` |
| 5.7% | 275.1 | 9,507 | 28.9 | `l2_warm_experts_tma_kernel` |
| 4.8% | 230.2 | 19,014 | 12.1 | `quantize_mmq_few_id_bcast_q8_1` |
| 4.1% | 199.4 | 18,917 | 10.5 | `attn_prep_kv_tail_f32` |
| 3.8% | 182.5 | 73,260 | 2.5 | `ggml_cuda_ar_ce_bump` |
| 2.8% | 133.2 | 9,507 | 14.0 | `f32_cols` |
| 2.3% | 110.7 | 2,178 | 50.8 | `mul_mat_q` |
| 2.1% | 103.7 | 19,179 | 5.4 | `ggml_cuda_ar_ce_wait_peers` |

CUDA API time: `cudaGraphLaunch` 2,651.1 ms over 1,134 calls (2.34 ms a call; v2 933.4 ms over 1,238 calls, 0.75 ms a call), `cudaGraphInstantiate` 1,608.4 ms over 686 calls (max 110.2 ms; v2 1,269.7 ms over 563), `cudaLaunchKernelExC` + `cudaLaunchKernel` 728.7 ms over 136,903 launches (v2 647.3 ms over 183,981), `cudaStreamSynchronize` 230.5 ms.

## 4. Open after this window

1. DSpark at 23 users: what refuses a kept graph on 0.55 of the steps (`graph.kept.miss.refused`) and what changes a property on 36 subgraphs a step (`properties_changed`). #308 did not reach either.
2. The AllReduce CE waits are still 14.2% of GPU kernel time; the batching moved the time from `wait_geq` to `wait_new` and `wait_peers`.
3. `cudaGraphLaunch` cost three times more a call than in v2 under the same tracing (0.75 ms to 2.34 ms). Not explained: fewer kernels were launched outside graphs (183,981 to 136,903), so more nodes a graph under node tracing is one reading; it was not checked without tracing.
4. Later-turn TTFT at 23 users is unchanged at about 3.5 s with the restore copies batched.
5. The server stops when turn 2 is admitted under `nsys` node tracing, on both runs.

Raw results (logs, ledgers, bench JSON, `nsys` report and SQLite): `~/.gemini/antigravity-cli/brain/c86bdf83-e09d-48b6-9d81-c4dfcbc3816d/scratch/4card-v3-results/`.
