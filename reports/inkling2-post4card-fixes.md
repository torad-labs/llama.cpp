# Inkling 2 Post-4-Card Bottleneck Audit and Fixes

Date: 2026-11-24
Branch: `torad/inkling2-post4card-fixes` (stacked on `campaign/inkling2-integrate-and-bench`)
Source telemetry: `reports/inkling2-4card-combined-toks.md` (Vast `62379820`, 4x RTX PRO 6000 Blackwell Server Edition)

## 1. Independent Source Audit of the 4-Card Telemetry

1. **`ggml_gallocr_needs_realloc` split-state cache thrash (`ggml/src/ggml-alloc.c`):**
   - Confirmed. `ggml_gallocr_needs_realloc` queried `ggml_backend_buft_get_alloc_size` across ~15,000 node/src/leaf checks per 5,900-node 4-card graph with `g_gallocr_measure_epoch == 0`, wiping the meta split-state cache on every call (`18,709 ms -> 65.5 ms` p50 on 4 cards, `285x` faster).
   - Fix: wrap `ggml_gallocr_needs_realloc_inner` in a single monotonic `g_gallocr_measure_epoch` scope.

2. **Residual `61.3 ms` mean (`512 ms` max) `decode.alloc_graph` and DSpark `graph slot memory limit` (`src/llama-context.cpp`, `scripts/inkling2/serve.py`):**
   - Correction to initial hypothesis: the remaining `61.3 ms` mean `alloc_graph` on 23% of steps was not `needs_realloc` CPU overhead; `combined-cc-server.log` logged `graph reallocs = 24 / 0` on the single-graph `main scheduler` as unretained prompt shapes (`1024 -> 413 -> 2048`) forced blocking 4-GPU `ggml_backend_synchronize` + full `reserve_n` re-plans.
   - In addition, `sched_reserve()` in `src/llama-context.cpp` read `ggml_backend_sched_get_buffer_size(sched.get(), ...)` right after `graph_reserve(..., split_only=true, sizes)` had reset `sched`'s buffers to `NULL` (`0` bytes), and did not account for single-sequence or `embd`-fed DSpark ubatches on CPU/GPU (`W process_ubatch: retained shape with 53/67/71/87/157/256 rows uses the main scheduler (graph slot memory limit)`).
   - Fixes:
     - Capture `pp_buf_size` while `sched` holds its allocated `pp` buffers (including single-sequence DSpark reservation), add float `embd` input margin for DSpark draft slots, and bound `slot_buf_size` up to `pp_buf_size` when `graph_shared` retains prompt or MTP/DSpark shapes.
     - Enable `prompt_shapes: 32` (`LLAMA_PROMPT_SHAPES=1`) and `keep_prompt_graphs: True` (`LLAMA_GRAPH_PROMPT_KEPT=1`, `GGML_META_CAPTURE_GRAPHS=46`) in `PROFILES["tensor-4"]`.

3. **20.8 ms/step server-thread gap (`stage.iteration 86.9 ms` vs `stage.target_eval 66.1 ms`) (`tools/server/server-context.cpp`):**
   - Correction to initial hypothesis: this gap was not generic HTTP/JSON overhead. On 133 of 144 real Claude Code turns, the 25 tool schemas activated lazy tool grammar (`!common_sampler_backend_passive`).
   - `sample_cpu_batch` (`server-context.cpp:6821`) and `verify_cpu_batch` (`server-context.cpp:6887`) explicitly skipped `!common_sampler_backend_passive(slot.smpl.get())`, forcing all 23 slots to sample 200,064-vocab tool grammar serially on 1 CPU thread (`stage.cpu_sample` had 0 entries in the 4-card ledger). In addition, `params_base.cpuparams.n_threads` (`9` from single-CCD `common_bind_`) clamped `LLAMA_SAMPLE_THREADS` (`64`) down to 9 threads.
   - Fixes:
     - Allow `!common_sampler_backend_passive` slots with full host logits (`row.sampled_logits == nullptr`) in both `sample_cpu_batch` and `verify_cpu_batch` (each slot mutates only its private `common_sampler`).
     - Size the parallel sampling pool from `max(cpuparams.n_threads, cpuparams_batch.n_threads)` capped by `LLAMA_SAMPLE_THREADS`.

4. **DSpark `CUDA3` VRAM starvation (`97,258 / 97,887 MiB`) and block length (`scripts/inkling2/serve.py`):**
   - Confirmed. `serve.py:234` gated `DRAFT_CONTEXT` (`1,572,864`) on `kv == "f16"`, which was unreachable because `--draft` sets `kv = "q8_0"`. Keeping `3,014,656` cells left `629 MiB` free on `CUDA3`, triggering `cuda.eval.direct.no_memory[3] = 7.46x/step` and evicting `CUDA3` CUDA graphs (`10,769 -> 240`).
   - Fixes: apply `DRAFT_CONTEXT` (`1,572,864`) unconditionally under `draft_layout` (freeing ~6.5 GiB across the pool) and enable `LLAMA_DSPARK_TRIM_BLOCK=1` by default with `--draft` (`--[no-]dspark-trim-block`). Also promote the proven 4-card flags (`GGML_CUDA_L2_EXPERT_WARM=1`, `LLAMA_INKLING2_ROUTER_BAKE=1`, `GGML_CUDA_MOE_ROUTE_GATE=1`) into `PROFILES["tensor-4"]`.

5. **Persisting L2 window on `>128 MiB` recurrent buffers (`ggml/src/ggml-cuda/ggml-cuda.cu`):**
   - Calibrated confirmation: on 4 cards with checkpoints, the recurrent buffer was `514,368 KiB` per card vs Blackwell's `131,072 KiB` (`128 MiB`) `l2_window_bytes`, so layers past `128 MiB` got no window and logged `no L2 window`.
   - Fix: when `!arena.magic && buf_size > dev.l2_window_bytes`, window each layer's recurrent state tensor `[p0, p1]` (256-byte aligned) individually with `active_bytes = buf_size` so all 48 layers share the granted persisting L2 budget proportionally.

## 2. Verification Status

- Local Python unit suite (`scripts/inkling2/test_*.py`, 281/281 tests OK), `wait_audit.py --check` (OK), `scripts/tunables.py check` (562 params OK), and ASCII checks: **PASS**.
- Cross-lab static verification (`claude-codex--gpt-6.2-sol` on `xhigh`, `verify-post4card`): **VERDICT: PASS** (0 critical/blocker issues; incorporated both nonblocking notes: `hparams.n_embd_inp_enc()` for DSpark feed margin in `src/llama-context.cpp` and explicit `--no-dspark-trim-block` override clearing in `scripts/inkling2/serve.py`).
- CloudRift RTX 5090 (`on-box`, SM120a Ninja build) C++/CUDA build and test suite: **PASS** (`ONBOX_VERIFY=PASS`):
  - `llama-server` + libraries built cleanly with zero warnings.
  - `test-tunables`: 444 knobs listed, 378 checked: OK.
  - `test-backend-meta-reserve`: 2-rank, uneven 2-rank, and 4-rank meta reserve sizing and split probes: PASS.
  - `test-llama-archs -b CUDA0 -a inkling2` (with `GGML_CUDA_L2_EXPERT_WARM=1 LLAMA_INKLING2_ROUTER_BAKE=1 GGML_CUDA_MOE_ROUTE_GATE=1 GGML_CUDA_INKLING2_L2_PERSIST=1 LLAMA_GRAPH_PROMPT_KEPT=1 LLAMA_PROMPT_SHAPES=1`): `CUDA0` MoE NMSE `3.32e-07` (Roundtrip OK), `Meta 3:7` MoE NMSE `1.78e-06` (2/2 passed).
  - `test-llama-archs -b CUDA0 -a glm5next`: NMSE `8.65e-08`, mixed batch `3.26e-14`, shared pool reuse `mtp=0/1` OK (4/4 passed).
