# Inkling 2 Small - 4x RTX PRO 6000 S Combined Stack & Real Claude Code Benchmark

- **Date:** 2026-11-24
- **Host:** Vast instance `62379820` (4x NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB VRAM each, 256 CPU threads, 756 GB RAM) - destroyed after pull (`1.87 h`, `~$13.19`).
- **Build:** Integrated 6-workstream commit `8515d2e8a` (`sm_120a`, Release) plus the 5-line `ggml_gallocr_needs_realloc` measure-epoch fix in `ggml/src/ggml-alloc.c` (see below).
- **Model pack:** `Inkling-2-Small-abliterated-e4853745-Q3_K_M-imatrix.gguf` (264.07 GiB), `mmproj-Inkling-2-Small-abliterated-e4853745-F16.gguf`, and `Inkling-2-Small-abliterated-e4853745-Q3_K_M-DSpark-BF16-tuned.gguf` (4.59 GB).
- **Combined stack enabled:** `serve.py --profile tensor-4` defaults (`query_first_decode`, `compact_greedy`, `kv_runs`, `packed_prefill`, `f32_few`, `moe_fuse`, `mmq_tma`, `l2_persist`, `ar_copy_capture`, `keep_graphs=32`, host copy-engine AllReduce, default-on WS-3 device arrival-counter AllReduce pacing `400 -> 0` host syncs) plus `--tuned` (`GGML_CUDA_L2_EXPERT_WARM=1`, `LLAMA_INKLING2_ROUTER_BAKE=1`, `GGML_CUDA_MOE_ROUTE_GATE=1`, `GGML_LEDGER=1`).
- **Workload:** 100% live 4-card GPU decoding against 40 normalized SWE-smith Claude Code 2.1.258 multi-turn sessions (6.5K-char system prompt, 25 real tool schemas, 4 cumulative multi-turn progressions per session, `max_tokens=256`, `stream=True`, `cache_prompt=True`).

## 1. Critical 4-Card Bug Found and Fixed Live (`ggml/src/ggml-alloc.c`)

Before the fix, every new graph allocation on 4 cards stalled on CPU at 0% GPU utilization for **18.71 seconds** (`GGML_LEDGER` recorded `decode.alloc_graph = 18,709,211 us` out of `19,108,441 us` total, while `decode.graph_compute` for 2,048 tokens took `191 ms`).

- **Root cause:** `ggml_gallocr_reserve_n_impl` sets `g_gallocr_measure_epoch` so `ggml_backend_meta_measure_op_splits` caches split states across a graph in `O(N)`. `ggml_gallocr_needs_realloc` (called on every `ggml_gallocr_alloc_graph`) did not set `g_gallocr_measure_epoch`, leaving it `0`. For every node and source (~15,000 calls per graph), `ggml_backend_meta_measure_sync_epoch(prov, 0)` cleared the cache and re-walked all 48 layers `O(N^2)`.
- **5-line fix:** Save, bump (`__atomic_add_fetch(&g_gallocr_next_epoch, 1, __ATOMIC_RELAXED)`), and restore `g_gallocr_measure_epoch` around `ggml_gallocr_needs_realloc_inner`.
- **Measured impact on 4x RTX PRO 6000:**
  - `decode.alloc_graph`: `18,709 ms` -> `65.5 ms` p50 (`61.3 ms` mean across 2,204 steps) - **285x faster**.
  - Server cold/warm ready time: `62 s` -> `16 s` (undrafted) / `24 s` (draft).

## 2. Live Real Claude Code 2.1.258 Multi-Turn Results (Undrafted Combined Stack)

OpenAI `/v1/chat/completions` and Anthropic `/v1/messages` tool-call roundtrip sanity checks both passed (`answer='3927'`). Zero router-bake warnings on startup.

| Concurrency | Turns / sess | Live decode / stream | Aggregate decode | End-to-end tok/s | ITL p50 / p95 / p99 (ms) | Turn-0 / Later TTFT (ms) | Later KV cache hit | Valid tools | Wall time |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 46.04 tok/s | 46.04 tok/s | 26.17 | 19.34 / 19.51 / 329.0 | 5,519 / 600 | 97.4% | 4 / 4 | 15.55 s |
| 4 | 4 | 31.05 tok/s | 124.22 tok/s | 60.18 | 21.16 / 112.13 / 458.0 | 10,142 / 1,601 | 96.5% | 15 / 16 | 29.55 s |
| 8 | 4 | 18.59 tok/s | 148.70 tok/s | 95.33 | 23.31 / 335.17 / 544.5 | 3,485 / 1,462 | 96.7% | 30 / 32 | 37.44 s |
| 23 | 4 | 5.05 tok/s | 116.09 tok/s | 83.13 | 36.08 / 804.83 / 1333.7 | 14,381 / 3,982 | 97.2% | 83 / 85 | 112.11 s |
| 23 (warm 1-turn, 32 tok) | 1 | 20.68 tok/s | **475.60 tok/s** | 34.45 | 23.00 / 688.96 / 693.7 | 20,218 / - | - | - | 21.36 s |
| 23 (warm 2-turn, 64 tok) | 2 | 16.67 tok/s | **383.35 tok/s** | 128.81 | 31.33 / 310.96 / 1339.1 | 4,059 / 10,811 | 98.7% | 24 / 45* | 20.95 s |

*Note: 32/64-token caps intentionally truncate long tool-call JSON payloads mid-call; at `max_tokens=256`, valid tool calls were 132/137 (96.4%). Across the full 1/4/8/23 undrafted sweep, `/metrics` recorded `154,155` uncached prompt tokens at `1,298.4 tok/s`, `2.945M` cached prompt tokens, `15,074` generated tokens, and `31,585` max sequence length.*

For reference against already-recorded 4-card RTX PRO 6000 S baselines (`scripts/inkling2/serve.py:38-44`, not re-run): stock decode was `289.6 / 290.7 tok/s` (`91.3 ms` p99) and Nov 22 `tensor-4` was `425.6 tok/s` (`65.1 ms` p99) on steady 23-slot decode. On warm 23-slot real Claude Code decode, the combined stack reached **`475.60 tok/s`** (`23.0 ms` p50 ITL) and **`36.08 ms` p50 ITL (~637 tok/s instantaneous steady-state decode)** during the 4-turn 23-slot sweep.

## 3. Live Real Claude Code Results with Tuned DSpark Draft

| Concurrency | Turns / sess | Live decode / stream | Aggregate decode | End-to-end tok/s | ITL p50 / p95 / p99 (ms) | Turn-0 / Later TTFT (ms) | Later KV cache hit | Valid tools | Wall time |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 26.89 tok/s | 26.89 tok/s | 12.14 | 0.02 / 120.46 / 188.8 | 17,862 / 1,098 | 97.4% | 4 / 4 | 38.13 s |
| 4 | 4 | 18.51 tok/s | 74.03 tok/s | 26.25 | 0.02 / 399.08 / 1139.5 | 27,542 / 3,361 | 96.5% | 15 / 16 | 61.30 s |
| 8 | 4 | 10.94 tok/s | 87.48 tok/s | 45.69 | 0.02 / 841.32 / 1345.1 | 7,064 / 3,895 | 97.1% | 30 / 31 | 64.52 s |

- **Draft quality was very high:** `/metrics` recorded **79.3% acceptance** (`3,499 / 4,411` draft tokens accepted across `1,469` verification steps = **+2.38 accepted draft tokens/step**, or `3.38` total tokens/step; pos 0: `88.4%`, pos 1: `78.7%`, pos 2: `71.1%`).
- **Why draft was slower than undrafted (proved by `GGML_LEDGER`):**
  1. `CUDA3` ran out of graph headroom (`cuda.mem.used_mib[3] = 97,258 MiB` out of `97,887 MiB`), triggering `cuda.eval.direct.no_memory[3] = 7.46x/step`, `meta.capture.defer.no_memory`, and evicting `cuda.graph.held[3]` from `10,769` down to `240`!
  2. `stage.iteration = 325.5 ms/step` with `4.45` `llama_decode` calls/step, `decode.alloc_graph = 21.8 ms/step`, `graph.kept.reject.ubatch_shape_or_sequences = 31.68/step`, `meta.subgraph_us = 55.0 ms/step`, and `cuda.outer.instantiate_us = 7.6 ms/step`.

## 4. Hardware & Engine Ledger Bottleneck Analysis (Where the Next Wins Are)

1 Hz `nvidia-smi dmon` telemetry across the combined runs (`1,467` active GPU-samples over `407 s`):
- **SM utilization:** median `31.0%`, p95 `99.0%`, max `100.0%`
- **Memory controller utilization:** median `10.0%`, p95 `40.0%`, max `52.0%`
- **Power:** median `167 W`, max `378 W`
- **PCIe TX bandwidth:** median `522 MB/s`, p95 `12,137 MB/s`, max `37,984 MB/s` (RX counter reported 0 by driver)

`GGML_LEDGER` across 2,204 server iterations of the undrafted 1/4/8/23 Claude Code run:
- `stage.iteration`: `86.9 ms/step` (`stage.target_eval`: `66.1 ms/step`, leaving **`20.8 ms/step` = 24% outside `llama_decode`**)
- `cuda.span_us` (actual GPU device time): **`35.8 ms/decode`** (only 41% of iteration wall time!)
- `cuda.eval.replay_us`: **`20.9 us`** when captured vs `cuda.eval.direct_us`: **`8.85 ms/step` per card** (`19.15` uncaptured subgraph evals/step)
- `decode.alloc_graph`: **`14.3 ms/step`** (`61.3 ms` mean, `262 ms` p99, `512 ms` max; triggered on `23%` of steps!)
- `meta.subgraph_us`: **`42.2 ms/step`** host, `meta.reduce_us`: **`5.16 ms/step`** host
- `graph.kept.reject.ubatch_shape_or_sequences`: **`15.45` rejections/step**
- `cuda.outer.instantiate_us`: **`2.7 ms/step`** (`34 ms` mean, `322 ms` max)

### Ranked Next Bottlenecks

1. **Eliminate mixed prompt+decode graph thrashing (biggest multi-turn win):** `23%` of steps miss all 32 kept graphs because arbitrary real-turn prompt lengths mix into decode batches, causing `61-512 ms` `alloc_graph` + `8.85 ms/card` uncaptured direct launches and driving conc-23 p95/p99 ITL to `805 / 1,334 ms` even though pure decode ITL p50 is `36.1 ms`. Enable `--prompt-shapes` (fixed power-of-two prompt ubatches separated from decode) and cut the remaining `61 ms` cost in `ggml_gallocr_needs_realloc` (avoid re-evaluating `ggml_backend_meta_get_split_state` when shapes/pointers are unchanged).
2. **Cut the `20.8 ms/step` server-thread overhead outside `target_eval`:** 24% of total iteration time is outside `llama_decode` (prefix copy, slot management, JSON/tokenization).
3. **Give DSpark draft `CUDA3` VRAM headroom + graph retention:** Free ~3-4 GiB on `CUDA3` (adjust `-c` under `--draft`) so `CUDA3` stops hitting `no_memory` graph evictions (`10,769 -> 240`), unlocking the already-measured `79.3%` acceptance (+2.38 tokens/step) on real Claude Code turns.
4. **Fix `GGML_CUDA_INKLING2_L2_PERSIST=1` window clamp on 23 slots:** Startup logged `131072 KiB from the base of their 136800 KiB buffer` on all 4 cards, silently disabling L2 persistence because the 23-slot recurrent state buffer exceeded Blackwell's 128 MiB max window by 5.7 MiB. Clamp/split the persisted window so 23-slot serving gets L2 persistence.

*Instrumentation caveat:* `nsys launch python3 serve.py` ran without `--trace-fork-before-exec=true`, so `inkling2_cc_4card.nsys-rep` did not capture CUDA kernels from the spawned `llama-server` child. All engine-level, API, and hardware telemetry above comes from `GGML_LEDGER=1`, `/metrics`, `cc_turns_bench.py`, and 1 Hz `nvidia-smi dmon`.
