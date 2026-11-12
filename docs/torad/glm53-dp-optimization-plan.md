# DP4 throughput implementation - November 4, 2026

Status: PRs [118](https://github.com/torad-labs/llama.cpp/pull/118) and [119](https://github.com/torad-labs/llama.cpp/pull/119) are merged into main (`3e15411a5`). Their local and remote branches were removed after verifying ancestry. The next source fixes are on `perf/glm53-decode-followups`, directly from main. CPU regressions and the SM120a CUDA build pass. GPU correctness, throughput gains and the transport decision remain unqualified; no speedup is claimed from source inspection.

Start with owner-local sampling, then repair graph reuse where a cause is demonstrated, then optimize remaining collective latency. Keep the current BoldingBuilds IQ3_XXS pack, four-way DP with partitioned experts and replicated dense weights, private sequence caches, host latent KV, Q8 caches/state and MTP2. Preserve the two-card tensor serving path. Throughput comes before a 400K capacity run.

## Baseline and limits

Runtime: `82ad56cb8b67c83c0977ae9e4d23da9b5779d469`. Evidence: [November 4 profiling](glm53-dp.md#november-4-single-session-profiling-and-scaling-controls) and [measurements](glm53-dp-profile-20261104.json). Documentation head at inspection was `301547c2039b9321d9afab4371f3bdab1285ba94`. PR 118's startup fix is now on main, but the saved measurements still describe that earlier runtime. They do not measure the subsequent optimizations.

One active session with 40 configured slots measured 45.3 tokens/s at 512 prompt tokens and 31.7 at 24,576. Reducing configured slots to one barely changed the short-context rate. One-slot host/GPU KV measured 36.7/36.8 at 24K with matching token IDs. These controls do not identify host KV or configured slot count as the main short-context bottleneck.

The traces show an extra full-vocabulary sampling gather and repeated graph instantiation. Collectives occupy 73-75% of summed GPU kernel durations, including rank waiting. This is neither wall time nor a removable percentage. The large unsigned gather family accounts for 21-22% of that sum, but not every captured invocation was attributed to a tensor; that percentage is not the predicted sampling speedup.

Concurrent 2/4/8-session replies already scale poorly. Their 64-token replies differ from the 128-token single-session controls. The measured common HTTP interval includes prefill and queueing; it is not aggregate steady decode throughput. MTP-off follows different output tokens and is also not an exact-work ablation.

## External results checked through Exa and primary sources

| Source | What was actually demonstrated | Application here |
| --- | --- | --- |
| [SGLang PR 5558](https://github.com/sgl-project/sglang/pull/5558), merged | DP attention's LM head can avoid gathering logits across DP groups by limiting vocabulary parallelism to the attention TP group. | Strong architectural precedent. Our DP owner already holds its full vocabulary row, so sampling need not mirror it across all four devices. |
| [llama.cpp PR 30662](https://github.com/ggml-org/llama.cpp/pull/30662), merged November 3 | Preserves the requested graph during idle eviction and gives cached graphs a reuse grace period after idle. | Already present in our `common.cuh:2079-2103`; do not repeat it as a new fix. |
| [Unsloth fork PR 232](https://github.com/unslothai/llama.cpp/pull/232), open | Author reports TP decode recovering when a graph-count cache smaller than its working set is enlarged; tested on V100/NVLink with another model. | Establishes a plausible failure mode, not our diagnosis. Our CUDA graph-count cap is already unlimited. Node limits, Meta cache replacement and new graph identities need separate attribution. |
| [SGLang graph design](https://www.lmsys.org/blog/2026-08-17-advanced-cuda-graph/) | Capture management separates execution phases, stable input buffers and padded shapes. | Use those invariants when fixing reuse. Copying a larger cache limit does not make incompatible shapes reusable. |
| [PCIe oneshot report](https://github.com/local-inference-lab/rtx6kpro/blob/master/optimization/pcie-oneshot-allreduce.md) | Author measures small-collective improvements on four RTX PRO 6000s with Qwen/NVFP4, with results sensitive to topology and message size. | Our internal collective already uses direct exchange. Benchmark our SM-issued remote accesses and rank-arrival skew before selecting a different transport. A fast DMA copy does not establish fast kernel-issued peer access. Driver override claims are not independently qualified here. |
| [GLM on four RTX PRO 6000s](https://helix.ml/blog/glm53-flash-on-rtx-pro-6000-part-1) and [follow-up](https://helix.ml/blog/glm53-flash-tp2-rtx-pro-6000) | Same model family/hardware, different NVFP4 packs and SGLang layouts. The follow-up uses TP2 replicas with a shared 500K token pool per replica. | Useful feasibility evidence, not proof of forty simultaneous 400K sessions, our pack's performance or a replacement for the requested capacity architecture. |

Chinese forum searches included Linux.do, V2EX and Hostloc. The returned first-person reports include [four-card DeepSeek deployments](https://linux.do/t/topic/2754979?page=2), but do not reproduce our model, pack and DP/MTP layout. No exact-recipe solution was established by this search. [DeepEP's current requirements](https://github.com/deepseek-ai/DeepEP/blob/main/README.md#requirements) specify NVLink for intranode communication and RDMA for internode communication; it is not a ready-made PCIe-only replacement for our GGML path.

Search queries and fetched pages are retained locally under `.Codex/tmp/dp-research-20261104/`. Published speedups are not forecasts for this fork.

## 1. Remove the unnecessary sampling gather

Source evidence: `src/llama-graph.cpp:4773-4801` appends a dummy row before per-sequence sampling. `ggml/src/ggml-backend-meta.cpp:1784-1792` interprets padding along the split axis as a gather. DP logits are split by output row, so this gathers every vocabulary row onto every device. NVTX confirms this path executes on the full model.

Make the DP sampler consume the full vocabulary row on its owning device and preserve ownership through sampler outputs. Use explicit row ownership and an owner-local padding/view path; preserve the existing vocabulary-gather semantics for tensor split and the intentional MoE gathers. Return the selected token and only the outputs the caller actually requests from the owner. Keep the dummy/inactive-row mechanism's graph stability and alignment requirements.

Correctness constraints: preserve per-sequence RNG and penalty history, inactive samplers, empty ranks, uneven output rows, multiple MTP verification rows, candidates/probabilities and requested logits. Do not replace the real sampling chain with a greedy shortcut. A caller explicitly requesting full logits still receives them.

Extend existing `tests/test-backend-meta-sampler.cpp` and DP/MTP fixtures in `tests/test-speculative-mtp.cpp`. Add CPU-capable structural coverage that demonstrates the unwanted gather before the change, checks row ownership afterward, and retains tensor-split gather coverage. Same-device sampler comparisons must preserve greedy tokens and seeded chain results, including history, candidates and probabilities. Physical CUDA sampler equivalence remains a remote check; CPU routing success does not qualify CUDA arithmetic.

Done: no full-vocabulary cross-DP collective for owner-local sampling, passing ownership/reservation/MTP regressions, and an SM120a build ready for remote qualification. This removes demonstrated work; the speed gain is unmeasured.

## 2. Fix graph reuse using separate cache identities and eviction reasons

There are multiple independent layers:

- `src/llama-context.h:438-458`: the current graph and up to three retained graph/scheduler slots, with bounds on retained rows and buffers.
- `ggml/src/ggml-backend-meta.cpp:2592-2645`: four whole-evaluation capture entries keyed by graph UID, replaced by LRU. This bound matches the nominal context slots; its existence alone is not evidence it is too small.
- `ggml/src/ggml-cuda/common.cuh:2079-2162`: individual CUDA graphs, an unlimited count cap in this recipe, a 32,768-node default cap, and idle eviction with a grace period.

Add opt-in counters/trace events for graph UID, target versus draft phase, rows and per-rank ownership, reuse rejection reason, capture refusal, eviction reason, nodes retained, instantiate versus update versus replay. Record events at the existing decisions, without adding synchronization or `cudaMemGetInfo`. Preserve graph identity ownership and scheduler-buffer lifetime.

Use CPU fixtures to exercise repeated target/verify/draft shapes and identify unnecessary graph rebuilds where possible. Fix deterministic invalidation mistakes first. Resize a bounded cache only after establishing that live reusable entries exceed its capacity. If shapes or addresses change legitimately, correct the key/buffer lifetime or introduce bounded shape retention; do not replay a stale graph. Keep rank launches concurrent: the current per-device launcher prevents collective replay deadlocks.

Done: each warm repeated shape has an explainable replay/capture path; no repeated capture solely from a cache too small for the selected working set; retained graph/device memory is accounted for. A short remote event capture is allowed if CPU fixtures cannot identify the remaining cause. Do not disable all limits and call that a qualification.

## 3. Prepare the measurement harness before another rental

Extend `scripts/glm53/measure.py` and its existing tests. Use equal output lengths for 1/2/4/8 active sessions, fixed prompts/seeds, identical slot IDs and owner assignments, and retained token IDs. Report both native per-request timings and actual delivered-token counts over a common interval. Separate prefill/queueing from a window where all requested streams are decoding. MTP emits bursts, so preserve token counts and timestamps rather than assuming one SSE event equals one token. If streams have no common steady interval, report that explicitly.

Retain MTP draft/acceptance counters and target/draft service times. For code changes expected to preserve arithmetic, compare matching token trajectories; otherwise use a controlled token replay for attribution and report ordinary generation separately. Use repeated baseline/candidate order (ABBA) for the important cells. Instrumented runs do not supply the throughput baseline.

The MTP implementation already batches active sequences before each `llama_decode` (`common/speculative.cpp:2019-2062`). Do not propose adding batching that already exists. Investigate batching losses at scheduler boundaries, per-sequence output consumption, idle ranks and changes in accepted draft lengths only where evidence identifies them.

## 4. Optimize remaining collectives only after the first fixes

Extend the existing collective benchmark/test to measure actual decode payload sizes on four ranks, both simultaneous arrivals and controlled owner delays, with CUDA graphs on. Compare our LL/direct-peer and fallback paths, and measure kernel-issued peer accesses separately from copy-engine transfers. Include topology/NUMA placement and latency distributions. No model download is needed for this probe.

Choose the smallest supported change from those results: payload thresholds, unnecessary barriers, redundant gather data, or a transport change. Actual all-gather/reduce-scatter or routed dispatch/combine remains a later candidate if the all-reduce emulation is demonstrably the bottleneck. Preserve fixed reduction order, exact gather bits, empty ranks, BF16/F32 behavior, host fallback and two-device behavior. Do not adopt speculative driver overrides, new dependencies or expert replication without evidence and memory accounting.

MTP hidden-state transfers and expert-load imbalance remain secondary candidates. Attribute their cost before expanding this patch series. Keep MTP enabled for the serving candidate; the existing MTP-off short-context result does not justify removing it.

## Remote acceptance and stop conditions

Finish source changes, focused CPU regressions, CUDA compilation, review and packaging locally. Use one four-GPU machine for the next bounded qualification. Verify counter permissions before collecting any Nsight Compute data. Keep an independent provider stop timer and collect only the artifacts needed to decide the next step.

First qualify CUDA correctness, then warmed unprofiled 512/4K requests at 1/2/4/8 active sessions with 40 slots configured. Compare baseline and individual candidate changes as well as the combined candidate. Check that the sampling gather is absent and repeated graph instantiations fall for shapes already warmed. Extend the existing 24K control only after short-context throughput and scaling improve. Stop the machine as soon as collection finishes; analysis and documentation are local.

Do not launch a 400K/full-capacity fill while the short concurrent workload remains unusable. The eventual requirement is useful per-stream throughput at the accepted active-session count, followed by gradual 400K capacity qualification, tool-cache continuity and numerical checks. Forty allocated slots are not proof that forty filled sessions perform acceptably. The user's 70-80 tokens/s team target remains a target, not a predicted result from these fixes.

## Implementation and local validation

- `87c9ce015`: DP sampling copies each vocabulary row into an aligned owner-local allocation. Tensor-split padding/gather remains. Four logical CPU ranks pass uneven/empty ownership layouts and three sampling chains, including the 154,880-token vocabulary, probabilities, candidates, seeded draws and unchanged source logits.
- `230ace57b`: graph reuse exposed an allocator lifetime defect. A temporary becoming an output could keep its old overlapping allocation. Allocation plans now include input/output lifetime flags. The regression fails without the check and passes with it.
- `8fb096af9`: retained graphs snapshot built-in sampler tensor bindings while sharing live RNG and penalty history. Unknown custom input callbacks retain the current-graph-only behavior. Three repeated shapes reuse without rebuilding, preserve seeded tokens and sampled outputs through partial acceptance, and survive eviction/owner changes. CPU DP/MTP cases pass on two and four logical ranks with default and Q8 caches. This does not imply arbitrary concurrent shapes fit the bounded cache.
- `9ed90db1b`: `measure.py --mode scaling` records native timings, raw SSE, each burst's token IDs and receive time, and the common decode interval. `--baseline-binary DIR` runs A1/B1/B2/A2 with one shared deadline. Seventeen harness tests pass, including HTTP streaming and all four ABBA legs.
- `b998962b1`: the existing collective test adds `--bench 4 [--delay 0..64]`, with 1/4/8/40-row payloads, F32/BF16 reduction, bitwise gather, peer/host paths and raw latency samples. Whole-evaluation capture is used where supported. Copy-engine reduction synchronizes the host and cannot join that capture; its result explicitly reports `whole_eval_capture=false`. The replay-only control and wall latency include launch/synchronization overhead; they are not GPU kernel latency.

`GGML_GRAPH_TRACE=1` reports llama graph UID/phase/ownership, reuse rejection, Meta capture/LRU decisions and CUDA instantiate/update/replay/eviction. It is disabled for throughput measurements. Its lines are logged at `-lv 4`. When the pooled indexer's input turns a kept graph down, `event=kpool_reject why=<condition>` names the condition under that graph's `event=reject reason=input` line, and for `why=dims` gives the step's dimensions beside the kept graph's. A card under its reserve gives kept graphs back (`event=evict reason=reserve`): a run of builds after those lines is the card's memory, not a change of shape. No cache limits were raised. Whole-evaluation capture now also honors `GGML_CUDA_GRAPH_MEM_STATS=0`, avoiding two accounting-only memory queries that the earlier change missed.

Focused local checks: `test-alloc`; `test-backend-meta-sampler --cpu`; `test-speculative-mtp --test sampler-graphs`; `test-speculative-mtp --dp-device CPU`; tensor CPU split cases `(0,1)`, `(0,3)`, `(1,8)`; and `python3 -m unittest discover -s scripts/glm53 -p test_measure.py`. CUDA compilation covered the server, sampler/MTP tests, collective probe and split tests. Raw local logs are under `.Codex/tmp/dp-impl-20261104/`.

The collective probe prepares the fourth item; a production transport change still requires measured evidence. The next bounded rental must establish physical CUDA correctness before the full-model ABBA matrix. Keep the startup-fix baseline (`82ad56cb8`), a sampling-only library from `87c9ce015`, and the combined candidate separately identified by artifact hashes.

## Qualification availability, November 4 at 18:58 UTC

At that time, PR 119 held this series, stacked on the startup fix in PR 118. That local candidate archive had SHA256 `668275c7b92e5e843559fb9b948e45cee5edbad5424b8cbd15037032d4cdfcdc`, with source head `75f8d57a2` and an independently built sampling-only library. Historical artifacts and the bounded controller are in `.Codex/tmp/dp-impl-20261104/`.

No physical GPU test ran in this attempt. A direct restart of retained instance 59024660 reached SSH, but Rig's old hard-stop deadline interrupted the initial upload. The controller confirmed the box stopped after 30 seconds. A second attempt used `rig vast keep --hours 0.6666666666666666` to renew its deadline; the box did not restart within the controller's two-minute readiness budget. The start queue was cancelled and both provider states confirmed stopped. A fresh search found no four-card, >=90 GB/card, >=360 GB RAM offer at <=$8.50/hour and >=98% reliability. No new instance was created.

GPU correctness, the ABBA rates and collective transport selection remain pending availability. The machine is stopped; its retained disk still incurs storage charges. The controller stops the instance immediately after necessary artifact collection and also arms an independent provider stop timer.

## Follow-ups after the offline profile audit

`ed45cde93` removes the hidden-row getter after terminal MTP samples while keeping the timing boundary after sampling, which already synchronizes. It also replaces the per-row sequence scan with a range-checked lookup. Sampling-output reordering now swaps only valid logits/probability/candidate prefixes; raw vocabulary rows are skipped only when no raw logits were copied for that batch. Mixed raw/backend output batches keep the raw permutation.

The same commit prepares the constant KDA convolution layout and indexer APE transpose once at model load, preserving their element types. Tensor split distributes each device's Q/K/V segments; DP mirrors these dense weights. The derived allocations belong to the model, are included in memory accounting and logs, and are excluded from GGUF export. `LLAMA_GLM5NEXT_PREPACK_LEGACY=1` retains the old graph construction for qualification. The canonical weights remain allocated as well.

`b46d39855` keeps DP draft selections in a persistent bank partitioned by sequence owner. Captures copy matching owner slices directly; graph reads select the local sequence slot without a host selection download or a cross-device gather. The existing round, position, pool-boundary and memory-generation checks still govern reuse. Graph inputs retain bank ownership across allocation changes. Matching Meta copies validate complete split layouts and skip empty ranks; different ownership retains the fallback. `LLAMA_KPOOL_SELECTION_HOST_LEGACY=1` remains the control.

Local validation:

- All 19 `test-speculative-mtp` groups pass. New coverage includes unequal sampling output lengths, raw-logit fallback, and exact legacy/prepared prefill, draft-token and verification-logit parity with one CPU device and two/four logical tensor and DP ranks.
- Two/four logical-rank DP MTP with default and Q8 caches pass, including 48 sequence-rounds per configuration and controlled zero/one/two-token acceptance. The tiny fixture has no naturally accepted greedy drafts; controlled acceptance supplies that coverage.
- `test-kpool-input` passes, including selection-bank width changes, reordered/shrinking batches, idle ranks, graph-storage destruction, memory invalidation and direct-copy admission/rejection. The initial direct-interface checks covered the first row of each round; the independent review below extends that coverage to every production capture.
- SM120a CUDA compilation passes for the server and MTP/k-pool tests. No CUDA execution or paid machine was used for these follow-ups.
- `test-llama-archs -a glm5next -s 1234` passes the DP KL checks and GGUF roundtrip. The full command still fails the prefill-reuse checks: the CPU compute reserve grows from 1,854,528 to 1,907,424 bytes, and logits differ at NMSE 1.70334e-12. The same failures occur with the legacy weight path; the diagnostic also expects sparse inputs where the short-context dense path has none. This is not reported as a green architecture suite.

The next hardware window must qualify these changes together with PR 119 before attributing a throughput gain. Start with short prompts and equal output lengths, then stop after collection. Remaining separate work is persistent accepted hidden-state carry for small MTP batches; measured concurrent MTP policy, host-KV staging generations, collective transport and graph retention above 32 target rows. The terminal-getter change does not remove eager hidden-state downloads. No 400K capacity fill or 70-80 tokens/s result is established here.

## Final local readiness check

A separate build of merged main (`3e15411a5`) reproduces both architecture-test failures with seed 1234: prefill logits NMSE 1.70334e-12 and the decode-input diagnostic reading -1 instead of 4352 cells. These are baseline failures, not regressions introduced by these follow-ups. The assertion has not been relaxed. Build provenance and both logs are under `.Codex/tmp/glm-readiness-20261104/`.

The selection-bank fixture can now run on named devices: `test-kpool-input --draft-bank CUDA0 CUDA1 CUDA2 CUDA3`. It exercises owner-local asynchronous and synchronous copies, rejects mismatched owners, destroys source storage, then reads the retained selections through a graph. Width changes, inactive ranks and invalidation are included. The same entry point passes with four logical CPU devices; physical CUDA remains untested. Its payloads exceed device allocation alignment so the partitioned-memory assertion remains meaningful on CUDA.

The current bundle and controller are together under `.Codex/tmp/glm-followups-20261104/`. The earlier controller still identifies an older artifact and must not be used for this candidate. The current package includes the remote runner, supporting scripts, prompt fixture and sampling-only control. Its controller defaults to a local archive/member hash check, checks that the packaged source matches the checkout, and requires `--execute` to start the existing bounded window. The GPU preflight includes the selection-bank fixture before full-model measurements. Stop and collection behavior remains unchanged.

Independent correctness, security, test and selection-lifetime reviews of PR 120 at `c75d0e787` found two test coverage gaps and no confirmed production defect. The test findings survived the separate filter review. The mechanical stage had no applicable C++ rules. Reviewer timeouts were resumed to final verdicts, not counted as approvals. The local review artifact is `/tmp/llama-pr120-review-lo2jgI/review.hif.json`.

The selection-bank fixture now observes the actual asynchronous copy calls made by every production capture and requires direct admission for each active row, including non-zero offsets and different owners. The end-to-end DP MTP fixture observes bank-backed graph reads before round teardown clears the rows, and runs a host-selection control that must produce no bank reads. Disabling matching owner copies or DP bank capture makes the respective test fail; restoring the optimized code makes both pass on logical CPU ranks. These checks establish path coverage, not physical CUDA stream ordering or performance.

Remaining work is not described as completed: small-batch accepted hidden carry and reusable host-KV staging require explicit lifetime designs; concurrent MTP policy, larger graph retention, collective routes and L2 prefetch require current measurements. Use the short qualification to choose among those, before any 400K capacity fill.

## Post-merge GPU qualification

PR 120 merged as `d54ea817b`. Two bounded windows reused instance 59024660, with four 300 W RTX PRO 6000 Blackwell Max-Q cards and the same BoldingBuilds IQ3_XXS pack. Both windows collected their results and confirmed the instance stopped immediately afterward. Their combined duration was 1,253 seconds, approximately $2.24 in compute at $6.437/hour, excluding retained storage. Machine-readable results, token hashes and artifact receipts are in [the qualification record](glm53-dp-qualification-20261104.json).

Four-card sampler and selection-bank checks, two/four-card DP MTP with default/Q8 caches, the two-card tensor check and both collective probes passed. All 128 collective cases reported correct results. The peer path was faster than host staging for small decode payloads. Copy-engine reduction did not support whole-evaluation capture, so its isolated latency is not a recommendation to change serving transport.

The old baseline (`82ad56cb8`) completed 512-token prompts at one, two and four active sessions, then exited with signal 11 during the eight-session warmup. The cause is not established. The combined candidate was tested in a second window. Each completed cell below had one 128-token warmup and one measured 128-token reply per stream, with 40 slots configured and MTP2 enabled. Values are means of native per-stream decode rates, not aggregate HTTP throughput.

| Prompt tokens | 1 session | 2 sessions | 4 sessions | 8 sessions |
| --- | ---: | ---: | ---: | ---: |
| 512 | 81.97 | 59.97 | 48.64 | 5.56 |
| 4096 | 55.08 | 53.65 | 39.66 | Incomplete qualification |

The eight-session 4096-token warmup generated 128 tokens but returned only 126 token IDs in its SSE stream. The strict measurement guard rejected it and stopped the run before profiling. The source buffers text until a UTF-8 character is complete, while the partial result included only the latest token ID; the final streaming result also omitted IDs held by an incomplete tail. The pending-token fix retains those IDs until a partial or terminal result sends them. Six tiny-model regressions reproduce the omission before the fix; they cover complete split characters, malformed leads, incomplete tails, both return_tokens settings and slot reuse. They pass after the fix, along with the existing malformed-text and basic completion/streaming tests (14 cases total). Text handling and the measurement guard are unchanged.

The completed candidate data shows a substantial four-to-eight-session cliff. During the 512-token requests, sampled GPU utilization averaged roughly 20-27% at eight sessions, versus 84-94% at four; these coarse samples include prefill and are not a kernel attribution. The next trace must include eight sessions. Current source still has the 32,768-node default CUDA graph budget and hard-coded idle eviction; their role in this cliff has not been measured.

No complete ABBA comparison or performance gain is claimed. Nsight Compute was denied GPU counter access (`ERR_NVGPUCTRPERM`); the Nsight Systems probe succeeded, but neither window reached a model trace. No 400K capacity fill ran, and forty configured slots remain distinct from forty filled sessions.
