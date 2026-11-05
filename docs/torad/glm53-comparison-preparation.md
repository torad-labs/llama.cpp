# One-session decode comparison preparation

## Goal

Prepare a reproducible, CPU-testable path to compare the same decode work on four GPUs, beginning with one active session. Preparation never starts inference, reserves hardware, or claims GPU qualification. The existing DP4 launcher is the configuration source of truth.

The first target is **one active 400,000-token session across all four GPUs**, with one reserved slot and resident KV. One session does not mean one GPU or only one measured cycle. Collect multiple identified decode cycles before expanding concurrency.

The launcher uses **adaptive MTP, maximum depth two, threshold 0.18**, not fixed MTP2. The case records that fact. It reserves 442,368 context cells per slot; the workload's 400,000 populated tokens are a separate quantity. Slots and active sessions are separate too.

The old one-session/4096-context capture is an offline plumbing fixture, not evidence for the 400k target. Its 40 reserved slots, host KV and historical runtime remain attached to it.

## What is implemented

1. **Encoded case.** `compare_decode.py prepare` derives the launch settings through `serve.make_plan`, preserving the pinned model revision, real cache types, adaptive policy, owners and batch sizes. Unbound model paths or host-memory budgets stay unresolved. It does not load weights or execute the generated command.
2. **Host collection hooks.** `serve.py --trace-decode` explicitly enables host markers, CUDA NVTX, graph identity and host-phase timers. Without that flag, these new markers are off. There are markers for server iterations, target evaluations, per-sequence batch row counts/position ranges/prompt counts, MTP proposals, catch-up, tentative acceptance and committed draft acceptance. No token text or token IDs are emitted.
3. **Rich trace recovery.** `decode_lab.py profile --include-trace` retains process/context/stream IDs, launch correlations, graph identities, explicit-copy endpoints, memsets, CUDA events and synchronization records. Optional fields stay null when absent. Compact browser exports remain the default.
4. **Semantic launch matching.** Whitelisted `glm53.v1` NVTX markers identify host phases. Kernels join through process-scoped CUDA launch correlations, not because their GPU timestamps happen to fall inside a CPU range. This works for asynchronous work that finishes after the host range ends. Target evaluations carry the server iteration ID across the queue thread boundary.
5. **Offline dependency scheduler and comparer.** Logical steps have cycle/phase/layer/device/work identities, explicit dependencies and exclusive resources. The slowest dependency releases the next step. Unknown costs prevent a complete timeline; memory-only models cannot claim a complete timing comparison. The comparer accepts exactly one identified cycle at a time, uses recorded wall span rather than summed GPU durations, and names intentional implementation differences.

The scheduler is a deterministic **feasible list schedule**, not a proof of globally optimal execution. It supports a supplied complete cost model; it is not itself a calibrated CUDA or PCIe simulator. The old byte-budget estimate is not relabeled as complete.

## Prepare without a GPU

From the repository root, choose a new output file:

```sh
python3 scripts/glm53/compare_decode.py prepare \
  --output .Codex/analysis/glm53-comparison-preparation/case.json

python3 scripts/glm53/compare_decode.py prepare --trace-decode \
  --output .Codex/analysis/glm53-comparison-preparation/capture-case.json
```

Defaults are one active session and 400,000 initial context tokens. Use `--model` and optionally `--server` / `--mmproj` to bind paths on the eventual GPU host. Even with bound paths, **prepare only writes JSON**. The model and projector files still need verification against their pinned artifact hashes before a real run.

The baseline is text-only unless a projector is supplied. A served configuration with a loaded projector must keep that explicit because it changes available memory. Host KV can be selected with `--kv host`; its required allocation is not treated as proof of available RAM.

The capture recipe preserves `nsys --trace=cuda,nvtx --cuda-graph-trace=node`. Capture the appropriate warmed workload, then select actual steady-decode cycles using their records; do not call startup, prefill or a whole request one cycle. Compare profiled evidence with an unprofiled baseline before making speed claims.

## Recover the existing evidence

```sh
python3 scripts/glm53/decode_lab.py profile \
  --sqlite .Codex/tmp/dp-profile-20261104/receipts/results/timeline-4k.sqlite \
  --label timeline-4k --active 1 --context 4096 \
  --runtime 82ad56cb8b67c83c0977ae9e4d23da9b5779d469 \
  --slots 40 --depth-max 2 --depth-policy adaptive --kv host --scope request \
  --measurements .Codex/tmp/dp-profile-20261104/receipts/results/measurements.json \
  --include-trace --output .Codex/analysis/glm53-comparison-preparation/recovered.json.gz

python3 scripts/glm53/compare_decode.py inspect \
  --profile .Codex/analysis/glm53-comparison-preparation/recovered.json.gz \
  --output .Codex/analysis/glm53-comparison-preparation/readiness.json
```

Rich exports can exceed the browser's expanded-file limit; they are for offline matching. They preserve only whitelisted profiler fields, not arbitrary SQLite tables, environment, prompts or process command lines.

The executed recovery is `timeline-4k-rich-v2.json.gz` in that directory, with `timeline-4k-readiness-v2.json`. It preserves 167,321 kernels, 12,190 explicit copies and 98,600 CUDA API records, plus the previously omitted scheduling data. The old capture predates these semantic markers, so its semantic cycle coverage remains missing. Recovery is not fabrication.

## Semantic work journal

The comparison API consumes `glm53-work-journal-v1` JSON with:

- `case`: the case metadata, adjusted from actual run facts rather than assumed defaults.
- `evidence`: `captured`, `source-model` or `synthetic-fixture`.
- `cost_coverage`: `complete` only for a complete execution-cost model, not the existing memory budget.
- `steps`: one identified cycle's logical work. Each step has `id`, `cycle`, `phase`, `layer`, `operation`, `device`, `work`, `depends_on`, `resources`, and `duration_ns`. Unknown ideal duration is null.
- Recorded steps additionally have `start_ns` and `end_ns` on one capture clock. `work` describes the actual rows, sequences, contexts and relevant routing/KV selections. Unknown workload identity is not a match.

A semantic step need not equal a CUDA kernel: fusion and optimized communication can change kernel boundaries. Preserve user-visible work identity while making implementation differences explicit.

```sh
python3 scripts/glm53/compare_decode.py compare \
  --ideal ideal-cycle.json --recorded recorded-cycle.json \
  --output matched-cycle.json
```

The executable synthetic example lives in `scripts/glm53/test_compare_decode.py`. It tests real CLI round trips and is always labeled synthetic. It is not a GLM benchmark.

## Marker interpretation

- A server iteration is **not automatically a decode cycle**. It can contain prefill, retries, multiple microbatches or several sequences. Batch metadata and phase coverage must establish the selected cycle.
- `target_eval` is deliberately not called verification: the recorded prompt-row counts distinguish prompt, generation and mixed work.
- `accept_candidate` occurs before checkpoint/replay decisions. `accept` records committed draft acceptance. Neither substitutes for a delivered-token count after output limits/cancellation.
- Host ranges identify launches, not GPU execution duration. Text journal timestamps use a host monotonic clock and are not silently aligned with Nsight's clock.
- NVTX host markers require the CUDA toolkit's NVTX3 headers in the build. The text journal reports `nvtx=0` when they were unavailable; a text-only journal does not claim CUDA-launch correlation.
- The normal graph trace is diagnostic metadata; framework graph UIDs are not automatically CUPTI graph IDs.

## Still requires GPU bring-up and evidence

This preparation does **not** establish:

- Occupied one-session/400k capacity, numerical correctness or decode speed.
- Per-cycle/per-layer actual expert routing, KV selection/reuse, or collective arrival skew. Existing route counters are cumulative and path-dependent; selected KV IDs are usually device-resident. Device-side collection must avoid forced readbacks/synchronization in the timed path and quantify its overhead.
- Hardware DRAM/PCIe bytes for kernel-driven transfers. Logical payload, hardware traffic and kernel duration remain distinct.
- Sustained simultaneous four-way bandwidth, latency distributions, or calibrated compute/dequantization and temporary-memory costs.
- A fully populated ideal timeline or a valid 1:1 comparison of the old 4k capture against 400k.

These gaps remain visible in the readiness report. They are not turned into zeros, estimated measurements or invented traces. The first GPU run is a one-active-session bring-up, not an immediate jump to 40.

## Verification

- All **149 GLM Python tooling tests passed**, including generated DAG/resource and marker-parser properties, real SQLite export, real CLI comparison and compilation/execution of the C++ marker emitter.
- A **CPU-only llama-server build passed** with CUDA disabled. The CUDA/NVTX build path and execution were not validated on GPUs.
- Existing capture recovery ran locally, read-only.
- No GPU inference, hardware rental, external publication or production restart occurred.

```sh
python3 -m unittest discover -s scripts/glm53 -p 'test_*.py'
cmake -S . -B .Codex/builds/glm53-comparison-cpu \
  -DGGML_CUDA=OFF -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=ON -DCMAKE_BUILD_TYPE=Release
cmake --build .Codex/builds/glm53-comparison-cpu --target llama-server --parallel 2
```
