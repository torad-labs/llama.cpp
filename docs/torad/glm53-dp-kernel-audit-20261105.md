# Four-card DP kernel audit, November 5

The candidate fixes an expert-routing bounds error and removes a separate expert-output clearing launch in the normal partitioned path. It also keeps infrequent CUDA graph shapes alive within the existing node budget and repairs profiling gaps. Runtime `136589f16066dc02c5eb5fb5fd324acf2fe1ed0d` now passes the focused GPU kernel tests, but full-model serving still crashes during four-session decode. It is not qualified for concurrent serving, and no throughput improvement is established.

## Physical test of the candidate

Vast instance 59286015, machine 44690, supplied four RTX PRO 6000 Blackwell Server Edition cards, each with 97,887 MiB, 600 W limits and PCIe 5 x16 capability, on driver 580.95.05. Rig provisioned one lab box from the existing CUDA image, fetched the pinned public pack and verified its hashes. The binary archive and all 46 manifested files were checked remotely. The text-only serving configuration matched the preceding short run: 36 reserved private slots, resident Q8 caches/state and adaptive MTP maximum depth 2. No projector was loaded; no 400K fill was attempted.

| Check | Result |
|---|---|
| Peer access and four-card direct collectives | Pass |
| Expert kernels under Compute Sanitizer | All 84 cases pass: 72 new routing cases and 12 existing fused cases; zero reported memory errors |
| 36 reserved slots and initial arithmetic response | Pass; 442,368 cells per slot |
| One active session, 512-token prompt, 128-token cached continuation | 71.203 tokens/s uninstrumented; cache retained |
| Four active sessions, same configuration | All four initial 16-token responses complete; the server segfaults during the cached continuation after one request finishes |
| Twelve and 36 active sessions | Not reached |
| Nsight replay and CUDA-GDB replay | Both reproduce the four-session failure |

The baseline observed at least 7,724 MiB free on every card; the trace and debugger runs observed at least 7,706 and 7,700 MiB respectively. None breached the 4,096 MiB reserve. These are periodic samples, not a proof of continuous-time minima. The single-session rate does not improve on the prior 73.216 tokens/s observation; different hosts/drivers and short windows prevent a controlled speedup or regression claim.

The debugger stops on SIGSEGV in `ggml_backend_meta_get_split_state`, with repeated recursive frames through its source-analysis lambda. Matching call-return addresses against the exact packaged ELF places the fault at offset `0x65fd4`: the stack-page probe `orq $0x0,(%rsp)` in the function prologue. This supports CPU stack exhaustion during recursive split-state traversal. The 20-frame backtrace does not establish whether an excessively deep valid dependency chain or a dependency cycle caused that recursion. In the traced failure, the last target evaluation has seven rows across sequences 0, 2 and 3, with row counts 3/3/1 after sequence 1 has completed. This still requires a source fix and regression coverage; the expert-kernel changes do not resolve the full-model failure.

The first trace launch was refused because the image's Nsight Systems 2025.3.1 does not accept `--discard-environment=true`. The diagnostic harness was corrected while the box was stopped, then the same box resumed for a bounded replay using its existing model files. The successful capture exports locally and passes the integrity check: 834,978 kernel events across all four GPUs and 485 model phase markers. Nsight Compute reports `ERR_NVGPUCTRPERM`; no hardware-counter result is claimed and host permissions were not changed.

Within 55 completed twelve-row target-evaluation host scopes, the two collective kernels account for 60.46% of summed per-device kernel duration. This includes waiting, overlaps across devices and profiler overhead; it is not wall time or a removable-overhead estimate. Collective coordination is the next performance investigation after the stack failure. The graph log records 62 Meta LRU evictions and no CUDA node-cap or idle evictions, so this run does not justify raising the CUDA node budget.

All evidence is local under `.Codex/analysis/glm53-dp-candidate-20261105/`: `results-manifest.json` hashes 263 collected files (56,786,533 bytes); `capture-integrity.json`, `nsys-target12-summary.json`, `graph-events.json` and `split-state-fault.json` record the offline analysis. The box was requested stopped about 1.5 seconds after the first remote window ended and about 2.5 seconds after the replay ended, allowing only file collection before those requests. Stop confirmations were 15:47:55 and 15:56:42 UTC. It was destroyed and confirmed absent afterward; unrelated stopped instance 59024660 remains untouched. Rig reports $1.40 + $0.42 = $1.82 of running cost, excluding storage/transfer and provider invoice reconciliation.

## Earlier recorded failure

The tested runtime was `e2435726a4eeb4d9b51520fe52625d81620c93be`, with the BoldingBuilds uncensored IQ3_XXS pack, four-way DP, resident Q8 target/indexer/recurrent/draft caches and adaptive MTP maximum depth 2. Forty reserved slots left only 2,989 MiB free on the most constrained card, below the 4,096 MiB requirement. Thirty-six reserved slots fit the short tests. Neither configuration qualified populated concurrent 400K sessions.

With 36 reserved slots and a 512-token prompt, one uninstrumented active session delivered a cached continuation at 73.216 tokens/s. The instrumented sequence measured 56.981 tokens/s for one active session and a median 28.583 tokens/s per session at four, with 116.394 aggregate. Twelve active sessions then failed during target verification. These short, differently instrumented measurements must not be mixed into a speedup claim or compared directly to the ideal model for 400K contexts.

Iterations 116-119 completed twelve-session MTP rounds. Iteration 119 accepted different draft depths across sequences. Iteration 120 completed both 12-row host proposal scopes and entered a 36-row target evaluation that never ended. CUDA reported an unspecified launch failure at `ggml_backend_cuda_synchronize` on device 0. The diagnostic observed at least 7,660 MiB free on every card and logged no allocation failure. The synchronization point does not identify the offending asynchronous kernel. Several earlier rounds used the same 36-row shape successfully; neither a first-use shape bug nor a rollback bug is proved.

The saved evidence is in `.Codex/analysis/glm53-concurrency-trace-20261105/REPORT.md`, `host-phase-summary.json` and `results/`. No full-model Nsight report survived that run. The complete preflight probe is not a model profile. A fresh twelve-session CUDA-GDB run completed, but did not replay the failing one-to-four-to-twelve history and is not evidence ruling out a state-dependent failure. All temporary instances from that investigation were destroyed.

## TensorFold comparison

Exa research and direct source inspection used TensorFold commit `018305ac4886c6c8c238a190aeb4e453dd1b7987`. Its native CUDA implementation contains useful scheduling patterns, but its NVFP4 and affine-4 weight paths do not implement this GGUF IQ3_XXS pack. No matched four-RTX-PRO-6000, 36-session benchmark for this pack was found.

| TensorFold implementation | Corresponding fork path and consequence |
|---|---|
| [Grouped expert rows and fused gate/up](https://github.com/ashhart/TensorFold/blob/018305ac4886c6c8c238a190aeb4e453dd1b7987/zig/kernels/cuda/experts_rows.cu): row tiles share weight loads and combine gate/up with SwiGLU. | `mmvq-moe.cu` already groups routed rows and fuses gate/up for at most eight token rows and 64 expert pairs. The larger MMQ path also groups rows by expert. Replacing it with a differently quantized TensorFold kernel would change the model and arithmetic. |
| [Persistent MoE scratch](https://github.com/ashhart/TensorFold/blob/018305ac4886c6c8c238a190aeb4e453dd1b7987/zig/src/families/glm/cuda_moe.zig) and [expert plans](https://github.com/ashhart/TensorFold/blob/018305ac4886c6c8c238a190aeb4e453dd1b7987/zig/src/families/glm/cuda_experts.zig). | The fork already uses pooled buffers and captured graph bindings. This audit does not establish allocation as the dominant cost. The separate expert zero-fill launch can be removed without changing allocation lifetime. |
| [Routing kernels](https://github.com/ashhart/TensorFold/blob/018305ac4886c6c8c238a190aeb4e453dd1b7987/zig/kernels/cuda/route_rows.cu) use staged loads and explicit row bounds. | Reviewing the fork's partial routing tiles exposed three unguarded destination-ID loads in `mmq.cuh`. The candidate guards them. |

Twelve sessions at full MTP depth 2 verify 36 rows, outside the eight-row fused ring path. For IQ3_XXS, the intervening vector path handles up to sixteen rows before MMQ takes over; Q8_0 switches after eight. This explains which code needs profiling, not why the run crashed or how much time any kernel consumed. A wider fused IQ3_XXS expert kernel remains a candidate for a later measured change. No TensorFold kernel was copied or model pack substituted.

## Implemented changes

- `mmq.cuh` bounds the destination-ID loads in the ordinary, stream-K and stream-K-fixup paths. Padding lanes use zero instead of reading past the last valid routing tile. Existing output bounds still suppress padding lanes.
- `ggml_cuda_mul_mat_id_partition` maps expert IDs and zeros only non-local expert contributions in one launch. Local expert kernels overwrite their own complete outputs. If output storage overlaps the input IDs, it retains the original map-then-clear ordering to avoid a read/write race. Hit counters still count each owned pair once. The operation count decreases on the non-overlapping path; its GPU latency has not been measured.
- `GGML_CUDA_GRAPH_IDLE_MS` controls graph expiry, retaining the engine's 10,000 ms default. The `dp-4` launcher sets it to zero and explicitly keeps `GGML_CUDA_GRAPH_NODES=32768`, with memory diagnostics off. Node-based eviction remains possible; increasing the budget requires evidence of eviction and memory headroom. Other serving profiles retain their existing policy.
- `--trace-decode` sets verbosity 4 so the enabled backend graph diagnostics reach the log. Normal serving does not enable trace logging.
- `finish_nsys_capture` waits for an owned exporter after a failed stop or missing report, bounded by the supplied deadline, before caller cleanup. It never treats report presence as capture qualification. The prepared, unexecuted `diagnose_v2.py` harness calls this helper and reserves export time within its deadline; the general measurement CLI does not automatically launch Nsight.
- `decode_trace.py` can check an exported SQLite capture offline. It requires model phase markers and post-marker kernel events on all expected devices in the same process. Runtime-only exports, preflight kernels from another process and startup kernels on a missing decode device do not pass. This is an integrity check, not causal phase attribution or model qualification.

## Local validation

The new partition matrix passes all 72 CPU-reference cases: IQ3_XXS and Q8_0, broadcast and per-expert inputs, wholly local and wholly non-local routing, token counts 1/3/8/9/16/17/36/108, and a partial 256-thread output tile. The existing CPU Meta split and Meta sampler suites also pass. CPU references exercise the expected operation and routing cases; they do not execute the modified CUDA kernels.

The launcher, qualification, trace, comparison and measurement Python suites pass 72 tests in total. The capture check's startup-only false positive was reproduced before the final guard and passes afterward. The new graph expiry test fails without the zero-timeout implementation; default expiry and disabled expiry both pass as host-only policy tests in a CUDA build, with no device initialization. The CUDA server, backend-ops and graph-cap targets compile for SM120a. No local GPU was used.

Build and test logs are under `.Codex/analysis/glm53-concurrency-trace-20261105/local-fixes-*`. Record the actual candidate commit and runtime artifact hash for the next run; the old `e2435726` package does not contain these changes.

## Planned GPU gate before the physical test

Keep the same resident 36-slot serving configuration, MTP maximum depth 2, pack, projector and 4,096 MiB minimum free-memory requirement. First run the focused expert cases under Compute Sanitizer on the remote box, including existing fused cases and the new routing matrix. Then reproduce one, four and twelve active sessions in that order on the same server at 512 populated prompt tokens. Only proceed to 36 active sessions after the shorter sequence passes. A fresh twelve-session server alone does not reproduce the observed history.

Keep uninstrumented throughput separate from Nsight timing. Save the graph log and completed capture, export SQLite on the analysis machine and check it with:

```sh
python3 scripts/glm53/decode_trace.py concurrency.sqlite --devices 0 1 2 3
```

Use the actual kernel/device timeline to decide whether a wider fused expert path or graph-budget adjustment is justified. If the concurrency failure remains, isolate the failing operator before another long-context fill. Preserve raw capture streams if export fails, finish collection within the bounded rental window and stop the machine as soon as GPU work ends. Populated 400K concurrency, quality and throughput remain subsequent gates.
