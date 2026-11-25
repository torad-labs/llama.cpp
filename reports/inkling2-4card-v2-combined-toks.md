# Inkling 2 Small - 4x RTX PRO 6000 S Combined Stack v2 (PRs #298-#304) Benchmark & nsys Profile

- **Date:** 2026-11-25 (02:47 - 04:05 AM CT)
- **Host:** Vast instance `62493016` (4x NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB VRAM each, AMD EPYC 9555 64-Core, 756 GB RAM, driver 610.57.04) - destroyed immediately after pull (`1.31 h`, `~$9.30` at `$7.100/h`, confirmed gone via `rig vast status`).
- **Build:** Prebuilt `sm_120` release engine commit `e76f2ad` (`PRs #298, #299, #301, #302`) driven by `scripts/inkling2/serve.py` from `main` (`9da84c172`, `PR #304`, wiring `--expert-layout rotated` default and `--ledger`).
- **Model pack:** `Inkling-2-Small-abliterated-e4853745-Q3_K_M-imatrix.gguf` (264.07 GiB), `mmproj-Inkling-2-Small-abliterated-e4853745-F16.gguf` (658 MB), `Inkling-2-Small-abliterated-e4853745-Q3_K_M-DSpark-BF16-tuned.gguf` (4.59 GB).
- **Combined stack enabled (all arms):** `tensor-4` defaults including `LLAMA_INKLING2_EXPERT_LAYOUT=rotated` (Lever 3 C1 rotated K-split), `GGML_CUDA_MMQ_TMA=1` (`OCC=2` Q4_K down + 4-warp `gate_up_situ`), 4-warp `fattn_rel_runs_rows`, `GGML_CUDA_L2_EXPERT_WARM=1`, `LLAMA_INKLING2_ROUTER_BAKE=1`, `GGML_CUDA_MOE_ROUTE_GATE=1`, fixed 256B-aligned `GGML_CUDA_INKLING2_L2_PERSIST=1`, `LLAMA_PROMPT_SHAPES=1`, `LLAMA_GRAPH_PROMPT_KEPT=1`, `GGML_META_CAPTURE_GRAPHS=46`, `GGML_CUDA_AR_COPY_ENGINE_CAPTURE=1`, WS-3 device counter sync, 23-thread parallel tool-grammar CPU sampling, `alloc_graph` epoch fix + permanent meta split-state memo, and `--ledger`.
- **Workload:** 100% live 4-card GPU decoding against 40 real SWE-smith Claude Code 2.1.258 multi-turn sessions (`cc_turns_bench.py`, 25 tool schemas, 1/4/8/23 concurrency x 4 turns at `max_tokens=256` plus 23u warm 2-turn at `max_tokens=128`). Zero baseline re-runs; compared against the Nov 24 4-card combined run in `reports/inkling2-4card-combined-toks.md`.

## 1. Headline Comparison vs Nov 24 Combined Run

### Later-Turn Aggregate Decode Throughput (tok/s, higher is better)

| Concurrency | Nov 24 Undrafted (F16) | Nov 24 Draft (CUDA3) | v2 Arm 1: Undrafted (F16) | v2 Arm 2: KV-Mixed (FP8+F16) | v2 Arm 3: TP4 DSpark + KV-Mixed | Best v2 vs Nov 24 Best |
|---:|---:|---:|---:|---:|---:|---:|
| **1u** | 46.04 | 26.89 | 49.60 | 50.36 | **78.85** | **+71.3%** (+193% vs old draft) |
| **4u** | 124.22 | 74.03 | 123.14 | 106.46 | **155.29** | **+25.0%** (+110% vs old draft) |
| **8u** | 148.70 | 87.48 | 96.18* | 139.95 | **178.71** | **+20.2%** (+104% vs old draft) |
| **23u (4-turn)** | 116.09 | - | 89.15* | **101.40** | 77.20 | -12.7% (see Sec. 3-4) |
| **23u (warm 1-turn, 32t, no ledger)** | 475.60 | - | **480.46** | - | - | **+1.0%** |
| **23u (steady GPU decode, nsys log)** | - | - | **607.0** (26.39/u) | - | - | 37.89 ms/step @ 20.4K ctx |

*\*Arm 1 (`combined-undrafted` with F16 global KV) hit a VRAM cliff on `CUDA0` (`97,231 / 97,887 MiB`, only `61 MiB` free) after `keep_prompt_graphs=True` bumped `GGML_META_CAPTURE_GRAPHS` from 36 to 46, forcing `cuda.eval.direct.no_memory[0] = 34.1x/step` (`16.75 ms/step`). Arm 2 (`combined-kvmixed`) freed `8.2-9.0 GiB/card` and eliminated all `no_memory` fallbacks.*

### Full Per-Arm Live Multi-Turn Metrics

| Arm | Level | Per-Stream (tok/s) | Aggregate (tok/s) | E2E (tok/s) | ITL p50 / p95 / p99 (ms) | TTFT t0 / later (ms) | Cache Hit | Valid Tools | Wall (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Arm 1: undrafted (F16)** | 1u x 4t | 49.60 | 49.60 | 23.69 | 18.01 / 18.28 / 404.41 | 6,653 / 593 | 97.4% | 4 / 4 | 16.08 |
| | 4u x 4t | 30.79 | 123.14 | 30.54 | 22.14 / 445.07 / 690.19 | 31,675 / 1,950 | 96.5% | 14 / 16 | 55.70 |
| | 8u x 4t | 12.02 | 96.18 | 48.83 | 23.08 / 434.38 / 777.59 | 17,249 / 2,509 | 96.7% | 31 / 32 | 68.89 |
| | 23u x 4t | 3.88 | 89.15 | 63.14 | 39.88 / 921.59 / 1,469.11 | 57,626 / 4,017 | 97.2% | 82 / 85 | 149.47 |
| | 23u warm 2t (128t) | 12.62 | 290.22 | 92.71 | 35.40 / 835.12 / 1,626.46 | 9,299 / 4,637 | 99.9% | 46 / 46 | 35.09 |
| **Arm 2: kvmixed (FP8+F16)** | 1u x 4t | 50.36 | 50.36 | 25.88 | 18.24 / 20.75 / 413.76 | 6,701 / 609 | 97.4% | 5 / 5 | 17.54 |
| | 4u x 4t | 26.61 | 106.46 | 47.89 | 22.32 / 171.37 / 580.50 | 13,586 / 1,365 | 96.5% | 14 / 14 | 35.06 |
| | 8u x 4t | 17.49 | 139.95 | 90.29 | 22.44 / 434.88 / 729.65 | 3,322 / 1,420 | 96.7% | 30 / 30 | 38.75 |
| | 23u x 4t | 4.41 | 101.40 | 75.55 | 37.24 / 972.89 / 1,542.53 | 15,779 / 3,490 | 96.9% | 84 / 86 | 121.30 |
| | 23u warm 2t (128t) | 9.82 | 225.93 | 101.91 | 65.21 / 655.19 / 2,104.46 | 8,653 / 2,648 | 99.9% | 46 / 46 | 32.49 |
| **Arm 3: TP4 DSpark + kvmixed** | 1u x 4t | **78.85** | **78.85** | 22.36 | 0.02 / 36.89 / 128.06 | 11,191 / 832 | 97.4% | 4 / 4 | 19.01 |
| | 4u x 4t | **38.82** | **155.29** | 43.95 | 0.02 / 212.32 / 652.36 | 15,514 / 2,820 | 96.5% | 16 / 16 | 30.83 |
| | 8u x 4t | **22.34** | **178.71** | 88.03 | 0.02 / 357.88 / 940.17 | 4,588 / 2,425 | 96.7% | 30 / 30 | 38.23 |
| | 23u x 4t | 3.36 | 77.20 | 56.24 | 0.01 / 1,496.95 / 2,364.35 | 15,436 / 9,671 | 95.4% | 82 / 84 | 164.58 |
| | 23u warm 2t (128t) | 6.52 | 150.04 | 60.66 | 0.02 / 1,440.88 / 2,601.36 | 21,497 / 4,627 | 99.5% | 46 / 46 | 52.62 |

- **Sanity check:** `19*207 = 3933` exact; tool-call validity across all 4-turn sweeps: Arm 1 `131/137` (95.6%), Arm 2 `133/135` (98.5%), Arm 3 `132/134` (98.5%), plus `46/46` (100%) on all three 23u warm runs.
- **TP4 DSpark acceptance & VRAM:** `12,205 / 15,588` draft tokens accepted (**78.3%**, `+2.35` draft tokens/step, `3.35` total tokens/step; pos 0: `87.9%`, pos 1: `77.8%`, pos 2: `69.3%`) across `5,194` draft rounds with **0 `no_memory` evictions**, `13,291` graphs held per card, and `5.2-6.0 GiB` free VRAM on every GPU (`92,057 / 91,329 / 91,329 / 91,329 MiB` used out of `97,887 MiB`).
- **PR #299 fixes confirmed live:**
  - `stage.cpu_sample` dropped to **`0.08 ms/step`** (`77-82 us/step`; `stage.cpu_verify = 0.30 ms/step`).
  - `GGML_CUDA_INKLING2_L2_PERSIST=1` initialized cleanly with zero `131072 KiB` clamp warnings.
  - `graph.kept.reject.ubatch_shape_or_sequences` dropped from `15.45/step` (Nov 24) to **0** across all 3 arms.
  - `graph reallocs` stayed at `0-9` across `3,342` reused graphs (Arm 1) and `1` across `1,941` reused graphs (Arm 3).

## 2. Engine Ledger Breakdown Across All 3 Arms

| Ledger Metric (per step unless noted) | Nov 24 Undrafted | v2 Arm 1 (F16) | v2 Arm 2 (KV-Mixed) | v2 Arm 3 (TP4 DSpark) |
|---|---:|---:|---:|---:|
| Steps recorded | 2,204 | 2,700 | 2,354 | 869 |
| `stage.iteration` (ms/step) | 86.9 | 117.2 | 101.1 | 341.3 |
| `stage.target_eval` (ms/step) | 66.1 | 96.5 | 75.2 | 220.2 |
| Outside `target_eval` (ms/step) | 20.8 | 20.7 | 25.9 | **121.1** |
| `stage.cpu_sample` / `cpu_verify` (ms/step) | - | **0.08** | **0.08** | **0.30** |
| `decode.alloc_graph` (ms/step, rate, mean) | 14.3 (0.23x, 61.3 ms) | 10.3 (0.26x, 39.8 ms) | 8.8 (0.25x, 35.2 ms) | **27.0** (0.98x, 27.6 ms) |
| `meta.subgraph_us` (ms/step) | 42.2 | 51.7 | 31.8 | 88.7 |
| `meta.reduce_us` (ms/step) | 5.2 | 9.4 | 4.5 | 11.5 |
| `cuda.eval.direct_us[0]` (ms/step) | 8.9 | **16.75** (`no_mem` 34.1x) | 4.71 (`no_mem` 0) | 12.53 (`no_mem` 0) |
| `cuda.eval.direct.properties_changed` (/step) | - | 15.23 | 14.35 | **37.45** |
| `cuda.outer.instantiate_us` (ms/step, mean) | 2.7 (34 ms) | 3.9 (39 ms) | 4.9 (34 ms) | **14.4** (25 ms, 0.57x) |
| `cuda.span_us` (GPU time / decode, ms) | 35.8 | 57.3 | 46.4 | 40.9 |
| `cuda.mem.used_mib[0..3]` (MiB) | - | **97,230 / 97,100** | 89,036 / 88,308 | 92,056 / 91,328 |
| `mem.ctx0.context_mib` (per card, MiB) | - | 24,262 | **12,670** (-11.6 GiB) | 13,002 (+1,618 draft) |

## 3. Direct `nsys` Steady-State 23u Profile (First Real 4-Card Kernel Capture)

Arm 4 launched `llama-server` directly under `nsys launch -t cuda,nvtx --cuda-graph-trace=node` (no Python wrapper, no `GGML_LEDGER`). The 23u warm-up pass hit **`480.46 tok/s`** (`23.28 ms` p50 ITL), and during Turn 1 of the profiled 23u pass all 23 slots decoded in parallel at **`26.39 tok/s` per stream = `607.0 tok/s` aggregate** (`37.89 ms/step` at ~20,400 ctx). After 19 slots finished Turn 1 and began Turn 2 admission, GPU 0 (which only had `61 MiB` free under F16 KV) exited under CUPTI graph-node tracing overhead; `nsys` still captured `625,100` CUDA kernel executions (`5.05 s` GPU time across 4 cards) cleanly.

### Top GPU Kernels (`cuda_gpu_kern_sum`, 4 GPUs combined = 5,046 ms)

| Share | Total (ms) | Calls | Avg (us) | Med (us) | Kernel | Category |
|---:|---:|---:|---:|---:|---|---|
| **13.2%** | 663.9 | 16,539 | 40.1 | 29.2 | `mul_mat_q_tma_q3_K` | MoE Q3_K TMA gate/up |
| **7.1%** | 356.1 | 8,799 | 40.5 | 39.4 | `attn_prep_rel_table_f32` | Relative-bias table prep |
| **5.8%** | 294.0 | 41,751 | 7.0 | 3.4 | `ggml_cuda_ar_ce_wait_new` | AllReduce CE wait spin |
| **5.6%** | 283.7 | 7,744 | 36.6 | 34.3 | `mul_mat_q_tma_q45_K<Q4_K, OCC=2>` | MoE Q4_K TMA down |
| **5.1%** | 255.8 | 6,682 | 38.3 | 38.3 | `fattn_rel_runs_swa_direct` | SWA attention |
| **4.6%** | 231.3 | 8,462 | 27.3 | 27.9 | `l2_warm_experts_tma_kernel` | L2 expert prefetch |
| **4.0%** | 199.9 | 16,539 | 12.1 | 10.2 | `quantize_mmq_few_id_bcast_q8_1` | MoE Q8_1 input quant (bcast) |
| **3.9%** | 198.1 | 26,076 | 7.6 | 8.1 | `mul_mat_q_few<Q3_K, 3>` | Dense Q3_K projections |
| **3.8%** | 191.0 | 62,580 | 3.1 | 2.5 | `ggml_cuda_ar_ce_bump` | AllReduce CE signal |
| **2.5%** | 128.0 | 1,542 | 83.0 | 1.0 | `ggml_cuda_arrival_wait_kernel` | AllReduce arrival spin (max 41.5 ms) |
| **2.5%** | 125.7 | 48,663 | 2.6 | 2.6 | `ggml_cuda_ar_ce_wait_geq` | AllReduce CE wait spin |
| **2.3%** | 113.8 | 7,820 | 14.5 | 14.6 | `f32_cols<3, 6>` | Router / rel-bias F32 matmul |
| **2.1%** | 105.0 | 282 | 372.3 | 302.6 | `mul_mat_q<Q3_K, 16>` | Prefill/large-M MoE MMQ |
| **2.0%** | 100.1 | 15,317 | 6.5 | 7.0 | `mul_mat_q_few<Q4_K, 3>` | Dense Q4_K projections |
| **1.7% + 1.6%** | 166.1 | 14,670 | 11.3 | 11.2 | `attn_prep_kv_tail_f32` (128/64) | Attention KV prep tail |
| **1.5%** | 75.7 | 16,538 | 4.6 | 4.1 | `quantize_mmq_few_id_q8_1` | MoE Q8_1 input quant (down) |
| **1.3% + 0.9%** | 113.5 | 2,732 | 41.5 | 42.4 | `fattn_rel_runs_rows` + `_shared` | Global F16 attention (only 2.2%!) |

### Top CUDA API & Memcpy Overhead (`cuda_api_sum` + `CUPTI_ACTIVITY_KIND_MEMCPY`)

- `cudaStreamSynchronize`: **45.8%** (`3,085.5 ms`, `91,536` calls, `33.7 us` avg, `20.4 ms` max)
- `cudaGraphInstantiate_v12000`: **18.8%** (`1,269.7 ms`, `563` calls, `2.26 ms` avg, `64.0 ms` max)
- `cudaGraphLaunch_v10000`: **13.9%** (`933.4 ms`, `1,238` calls, `754 us` avg, `12.0 ms` max)
- `cudaMemcpy2DAsync`: **6.3%** (`422.0 ms`, `73,604` calls)
- `cudaLaunchKernelExC` + `cudaLaunchKernel`: **9.6%** (`647.3 ms`, `183,981` uncaptured kernel launches)
- `CUPTI_ACTIVITY_KIND_MEMCPY`: HtoD `146,411` calls (`13.15 GB`, **`2,268.5 ms`**), DtoH `16,941` calls (`4.01 GB`, `93.3 ms`), DtoD `8,794` calls (`4.72 GB`, `9.1 ms`).

### Hardware Telemetry (`nvidia-smi dmon`, 5,217 active GPU samples)

- **SM util:** p50 `35%`, p95 `99%`, max `100%` (up from `31%` p50 on Nov 24)
- **Mem controller util:** p50 `11%`, p95 `42%`, max `67%`
- **Power:** p50 `165 W`, p95 `314 W`, max `376 W`

## 4. Ranked Next Bottlenecks (Exact Evidence From This Run)

1. **Synchronous `cudaMemcpy2DAsync` + `cudaStreamSynchronize` storm during Turn-2 checkpoint restore freezes the server for `3.2 s` at 23u (BIGGEST MULTI-TURN WIN):**
   - `nsys-server.log` proves all 23 slots decoded in parallel at **`607.0 tok/s` (`26.39 tok/s/stream`, `37.89 ms/step`)**, and `slot release` / `slot launch_slot` logging itself took only `0.3-0.6 ms` per slot.
   - Instead, `nsys` SQLite (`CUPTI_ACTIVITY_KIND_RUNTIME` + `CUPTI_ACTIVITY_KIND_MEMCPY`) caught **3.2 full seconds (`t=0s..3.2s`) of zero GPU kernels** on Turn-2 admission: **`73,604` `cudaMemcpy2DAsync` + `91,536` `cudaStreamSynchronize` calls (`3,085.5 ms` on 1 thread!)** moving `1,973.3 MB` (`~85.8 MB` x 23 slots) in `81,020` `4KB-64KB` HtoD transfers.
   - **Root cause:** `tools/server/server-context.cpp:6096` calls `common_prompt_checkpoint::load_tgt()` -> synchronous `llama_state_seq_set_data_ext()` -> per-run `ggml_backend_tensor_set()` -> `ggml_backend_meta_buffer_set_tensor()` -> `ggml_backend_tensor_set_2d()` -> `cudaMemcpy2DAsync` + `cudaStreamSynchronize` per layer/run/GPU (`~3,200` syncs per slot). Even `ggml_backend_meta_buffer_tensor_n` caps `rows <= part_rows` (8), falling back to `set_tensor_2d` on 1,024-cell SWA runs.
   - **Fix:** Pack split-rank contiguous-device rows (`pitch == width`) through 4 MiB CPU scratch into batched `ggml_backend_tensor_set_n` (1 H2D + scatter + 1 sync per chunk per GPU instead of 73,604 syncs), and use `llama_state_seq_set_data_async` at `server-context.cpp:6096` when `LLAMA_STATE_RESTORE_ASYNC=1`. Cuts 23u Turn-2 restore from `~3.2 s` to `<60 ms` (>50x).
2. **Fix `CUDA0` VRAM exhaustion in default F16 `tensor-4` profile (IMMEDIATE CONFIG FIX):**
   - `keep_prompt_graphs=True` bumped `GGML_META_CAPTURE_GRAPHS` from `36` to `46`, leaving only `61 MiB` free on `CUDA0` under F16 KV and causing `cuda.eval.direct.no_memory[0] = 34.1x/step` (`16.75 ms/step` penalty!).
   - Making `--kv-cache f8_e4m3 --kv-mixed` the `tensor-4` default (saving `11.6 GiB/card`, leaving `8.2-9.0 GiB` free!) or lowering `-c` under F16 by ~256K cells fixes this immediately.
3. **Eliminate DSpark 23u variable-shape graph thrashing (`0.98` `alloc_graph`/step -> unlock TP4 DSpark at 23u):**
   - TP4 DSpark already crushed 1u (`78.85 tok/s`), 4u (`155.29 tok/s`), and 8u (`178.71 tok/s`) with 78.3% acceptance and 0 OOMs.
   - At 23u, variable verification batch row counts (`sum(n_draft_i + 1)` across active slots) missed all 32 kept graphs **98% of steps**, spending `27.0 ms/step` in `decode.alloc_graph`, `14.4 ms/step` in `cuda.outer.instantiate_us`, `12.5 ms/step` in `cuda.eval.direct.properties_changed`, and `121.1 ms/step` outside `target_eval`.
   - Padding DSpark verification batches to fixed power-of-two/bucketed row counts (just like `--prompt-shapes` did for prompts) will let 23u DSpark hit kept CUDA graphs every step.
4. **Cut top non-MoE GPU kernels revealed by `nsys` (`26.3%` of GPU time!):**
   - **AllReduce CE spin/signal kernels:** `14.6%` of total GPU time (`ce_wait_new` 5.8% + `ce_bump` 3.8% + `ce_wait_geq` 2.5% + `arrival_wait` 2.5%), plus `2.27 s` HtoD copy time.
   - **`attn_prep_rel_table_f32`:** `7.1%` of total GPU time (`40.5 us` x 8,799 calls) - huge surprise and likely low-hanging fusion/caching fruit.
   - **`l2_warm_experts_tma_kernel`:** `4.6%` of total GPU time (`27.3 us` x 8,462 calls) - worth A/B checking on 4 cards whether its `27.3 us` launch sits on the critical path vs overlapped on a side stream.
   - **MoE Q8_1 input quant (`quantize_mmq_few_id_*`):** `5.5%` of total GPU time (`12.1 us` + `4.6 us`).
5. **Fix `cuda.eval.direct.properties_changed` (`14.4-37.5x/step`) and `cudaStreamSynchronize` (`45.8%` of CUDA API time):**
   - Even with 0 `no_memory` (Arm 2), `14.35` subgraphs/step still fell back to uncaptured direct execution due to `properties_changed` (tensor pointer/stride churn between shapes), driving `183,981` direct kernel launches and `563` `cudaGraphInstantiate` calls (`1.27 s`).
