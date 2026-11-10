# GLM-5.3 Flash: local-first concurrent inference work

## Goal

People using about 36 simultaneous GLM-5.3 Flash sessions should receive useful responses faster without losing context, changing the selected model, weakening target sampling, or breaking conversation restore and speculative rollback. The deployment target is four RTX PRO 6000 cards, with 50-70 delivered tokens per second per active session as the performance goal, not an established result.

Do everything that can be built and checked on the existing machine before spending on another rental. No rental or remote run is part of this implementation. Keep the resident recipe's 442,368 private cells per slot and its 4,096 MiB free-memory qualification reserve. CPU fixtures and the local GPUs cannot establish throughput or populated long-context capacity on the four-card deployment.

## Agreed changes and acceptance

1. **Collective choice.** The launcher can explicitly choose the existing internal or internal-with-NCCL-gather transport for the four-card recipe. Its dry-run plan and comparison records identify the effective requested choice. Defaults remain reproducible, unrelated inherited engine overrides remain removed, and a comparison cannot silently describe a fallback as the requested transport.
2. **Concurrent MTP policy.** A concurrent round uses one equal draft depth chosen for the batch's delivered-token service rate, including draft, verification and catch-up costs. One sequence's periodic probe must not force repeated batch-wide probes. Preserve output limits, target sampling, rollback and existing single-session behavior; invalidate timing samples when work cannot be attributed to the measured round.
3. **Expert preparation.** Compatible routed gate/up projections share routing and activation preparation at the widths used by 36 sessions. Outputs, clamps, nonlocal-expert zeros, alias safety and captured replay remain equivalent to the existing path. Unsupported types and layouts retain the existing implementation. No second copy of the model weights is introduced.
4. **MTP hidden state.** Pending and accepted hidden rows stay in owned backend storage across ordinary draft/verify rounds. Replacing, restoring, clearing or interleaving sequences must never expose a stale row. Existing host checkpoint serialization remains correct, and unsupported paths retain a correct fallback.
5. **Prefill scheduling.** Long prompts make fair progress in useful per-sequence chunks rather than unnecessarily thin slices, while existing decode streams retain their prompt-work budget and speculative rows remain intact. Preserve prefix reuse, output accounting, short-prompt admission and the other serving profiles. Any remaining cache-layout restriction on noncontiguous active slots must be reported rather than removed unsafely.

## Review follow-up goal

An embedding request that must be evaluated whole keeps the same pooled result with prompt chunking on or off. Capture storage retains the four most recently used scheduler graph generations and their actual fragments, not four times a historical fragment maximum; once a wide generation leaves that set, all its captured executables are released.

GPU residency here means target carry and accepted/verification rows are retained in backend storage. The per-sequence retained-row copies are currently synchronous, and proposal continuation after the first draft step still uses host hidden rows. This is not an end-to-end device-resident proposal chain. These remain optimization opportunities, not measured slowdowns.

The pooled acceptance estimator and probe phase currently have the lifetime of the MTP implementation: they persist across idle periods and fresh cohorts. Timing evidence is separately invalidated on request boundaries and changes in participating sequences. This is service-wide adaptation, not a guarantee of fresh-cohort warmup; no lifecycle change is included in this follow-up.

## November 8 audit corrections

Goal: operators can compare the current candidate with historical engines without an unsupported launch argument, concurrent users get a draft policy based on correctly attributed cost and valid acceptance results, retained hidden rows use bounded metadata without per-sequence synchronization, and Rig rejects an incompatible selected draft cache before renting.

Acceptance:

- Historical baseline binaries can start through the comparison launcher; capable candidates retain the requested prompt chunk and each launch record identifies the actual arguments.
- A depth's continuation preparation belongs to that depth's timing sample. Invalid timing does not discard valid acceptance evidence, including sustained mixed prefill and verification.
- Retaining rows across repeated requests keeps view metadata bounded. Copies complete before any consumer, replacement, checkpoint, or release can observe them, without synchronizing every individual row copy.
- Rig rental preflight and serving resolve the same selected profile's draft cache and reject unsupported formats before creating a paid instance.
- Capture reuse retains the four recent graph generations without rebuilding a parent list or scanning all fragments on every cache hit.
- Retained-row packing has one source of sequence boundaries, and paired CUDA eligibility reuses the existing layout comparison without changing supported layouts or results.

These corrections do not change the model, numerical-quality contract, production pins, or rental authorization. Behavioral regressions are tested before their fixes; output-preserving cleanups retain the existing equivalence checks.

### November 8 correction results

All five audited defects and four cleanups are implemented in the shared engine and Rig checkouts:

| Item | Change | Verification |
| --- | --- | --- |
| Historical engine launch | Probe the chosen binary's help before adding the automatic prompt chunk; record the resolved arguments. Failed capability detection refuses startup. | Legacy/current engines, failed help, live/dry-run agreement and matrix receipts passed; new regressions failed before the fix. |
| Concurrent MTP cost | Start the service interval before the first draft evaluation, so a round pays for its own first step and continuation preparation. | The real driver with a deterministic clock keeps cheap full drafts and rejects expensive ones at identical acceptance. |
| MTP acceptance learning | Keep acceptance results independently of timing; lifecycle changes invalidate only an outstanding member, not unrelated slots or results already collected. | Mixed-prefill lifecycle regression failed before the parent correction, then passed across generated row orders, all spare slots, and accepted-slot release/restore. |
| Bounded retained-row metadata | Reuse one fixed-capacity copy-view arena per sequence, preserving metadata until queued copies complete. | Generated repeated requests with 1–4 retained rows, duplicates and reordered rows passed on CPU and CUDA. |
| Rig selected draft cache | Preflight uses the same selected-profile cache resolver as serving. | Generated supported/unsupported K/V pairs and dry/live preflight cases failed first, then passed; unsupported configurations create no instance. |
| Asynchronous retained-row copies | Distinct sequences no longer drain each other's copies. Arena reuse and host access wait; target/head consumers use event dependencies. | CUDA retention passed over 1/2/4 logical ranks, default and Q8 caches; reads occur after subsequent target/head consumption. |
| Capture-cache hits | Direct fragment/generation lookup and generation-level recency; eviction removes the oldest generation's complete fragment set. | CPU/CUDA compilation and CUDA split transition/reuse passed over two logical ranks. |
| Retained-row packing | Build each sequence's prefix/last-row list and derive its count once. | Existing host/device state, logits and row-copy equivalence coverage retained. |
| CUDA layout checks | Reuse `ggml_are_same_layout` for paired routing eligibility/assertions. | All 97 supported local expert-routing cases and captured source-type replay passed. |

Other completed checks: all 171 tests from the 11 GLM Python test files; Rig lint, repository rules, TypeScript and build. Rig's full suite had 1,038 passes, one skip and seven default-five-second timeouts; all 28 tests in the two affected files passed with a 120-second invocation timeout. No assertion or test-source timeout was weakened. At the end of that phase, the dirty Rig binary remained deliberately ineligible for rental preflight until committed; the later authorized rental used the clean local commit documented below.

The integrated CPU run passed 9/10 checks. The broader `test-glm5next` failures below remain red; unchanged checks were not repeatedly rerun. After the lifecycle correction, CPU and CUDA server/MTP builds passed, all 19 ordinary MTP test cases passed, and the CPU and CUDA DP groups passed with 2/4 logical ranks and default/Q8 caches. These include 36-way equal-depth rounds, output-limit handling, and graph reuse across 33/40-sequence verifications. CUDA used only the RTX 5080; it does not establish multi-physical-GPU performance. Final logs: `/tmp/glm53-audit-fixes.3ZqjB8/final-*.log`.

Verification limits: the retained-row worker's original pre-fix failure was a Q8 target-logit mismatch before the retention assertions, not a demonstrated metadata-growth reproduction. The MTP worker's intended red run hung in a newly introduced recursive clock helper; the parent fixed that helper and a two-versus-three-sequence fixture error. Consequently, no original timing-defect red-to-green claim is made. The later acceptance-lifecycle failure was reproduced and fixed. Real-clock jitter in a cadence assertion was removed by using the permitted fake clock, without changing the expected cadence.

At the end of the audit-fix phase, no commits, pushes, rentals, deployments or production-pin changes had been made. That phase made no throughput, full-model quality, populated long-context, NCCL or four-physical-GPU qualification claim.

### November 8 authorized rental - completed

After Marcos funded and authorized a test rental, Rig's tested two-file correction was committed locally as `3ffba54` and rebuilt; nothing was pushed or production-repinned. Disposable lab **59699559** ran four physical RTX PRO 6000 Server Edition GPUs, driver 580.178.04, at $7.537777/hour including disk. All owned model processes had exited before cleanup. Rig destroyed the lab at **6:37 AM CT**, well before the **7:28 AM CT** hard stop, and a subsequent complete provider listing confirmed it absent. Rig estimated **2.15 hours / $16.21** for rental time including disk; transfer charges were not reconciled to a final invoice. The existing kept box 59024660 was untouched and still listed stopped.

The portable candidate is HEAD `24c77a0b5` plus dirty patch SHA-256 `098fcb005ad4b0575ece6786e1da219d49c5b337a6e72a06404c0635b0014510`. Source identity was checked after the build. It uses CUDA 13.3, glibc 2.35, `120a-real`, native CPU tuning off, and NCCL 2.28.9+cuda13.0. All 95 upload-manifest entries verified on the rental; 14 binaries passed the bundled-library resolution check.

**All 10 declared four-card component gates passed**, including physical DP/MTP in internal and NCCL-gather modes. The NCCL gather test recorded four native captures for each of three shapes and zero fallbacks. The pinned three-shard model fetch passed its streamed checksums. Component results and all six completed serving stages are collected and hash-verified locally.

Median per-session delivered tokens/s during the common decode overlap, with 36 active sessions:

| Engine / transport | 512-token prompt, repeat A / B | 8,192-token prompt, repeat A / B |
| --- | --- | --- |
| Retained baseline D, internal | 16.409 / 16.588 | 15.351 / 15.473 |
| Current candidate C, internal | 26.776 / 27.101 | 24.631 / 24.220 |
| Current candidate C, NCCL gather | 28.385 / 28.529 | 26.024 / 25.120 |

C means this run's HEAD plus patch above, not the historical November 6 C. D is the retained `e18d70a0bf69873e2b39785cd17b1a83270e8f9d` binary, not Rig's production pin. Using the mean of each pair of medians, candidate internal is 63.3% faster at 512 and 58.5% faster at 8,192; NCCL gather adds 5.6% and 4.7% respectively. This is a small repeated comparison, not a confidence interval or attribution to one optimization. The candidate transport order was internal, hybrid, hybrid, internal; baseline repeats followed.

All six short-answer checks passed, and all 96 greedy token IDs plus content matched between both engines and transports. This is not full-distribution or semantic quality certification. The lowest sampled free memory across these six serving runs was 5,116 MiB, above the 4,096 MiB guard; these are one-second samples, not continuous minima. Candidate internal with one active 512-token session delivered 95.298 tokens/s. The 36-session target of 50-70 tokens/s remains unmet.

Decode begins after 16 warmup tokens, at populated lengths 528 or 8,208. The 8,192-token group warmup took about 618 seconds for baseline, 102 seconds for candidate internal and 89 seconds for hybrid. Those warmups include generated tokens and are not a dedicated mixed-prefill/decode benchmark. The single-sequence DP4 distribution check completed at contexts 2,048, 4,096 and 16,384. At matched microbatch 512, both candidate transports reported mean KL 0.000000 and same-top-p 100.000% against baseline D at all three contexts. The separate microbatch-128 probes failed the unchanged absolute-mean-KL <= 1e-6 and same-top-p >= 99.99% contract in both transports: KL 0.016772 / 0.020505 / 0.011963 and same-top-p 95.797% / 94.774% / 96.594%, respectively. The quality stage therefore failed; its failures and distribution references were collected and hash-verified. These are single-sequence probes, not a direct quality certification of 36 concurrent serving streams. The 2,048-token control reproduced exactly the same discrepancy in baseline D at microbatch 128 versus D at 512 (KL 0.016772; same-top-p 95.797%). C at 128 versus D at 128 reported KL 0.000000 and same-top-p 100.000%. Thus this 2K discrepancy already exists in the retained engine, rather than being introduced by the candidate; this does not waive the failed contract or establish the cause at all contexts. The guarded long-context run passed with four simultaneous 400,000-token prompts. After 16 warmup tokens, all four streams decoded from 400,016 populated tokens at a median 47.024 tokens/s per stream (188.479 aggregate common-overlap tokens/s); the group warmup took 781.938 seconds. Minimum sampled free memory was 7,452 MiB. This establishes four populated long contexts, not 36, and does not override the numerical-quality failure. 

The instrumented full-model hybrid run passed. Its whole-process trace records 145 meta captures and 1,434 meta replays, plus 3,654 CUDA captures and 8,762 CUDA replays. One CUDA graph update failed and was followed immediately by a new instantiation of the same graph; serving completed without a memory-guard failure. The trace also records 42,856 direct executions caused by changed graph properties. These counts include initialization, prefill and decode and are not GPU-time attribution or proof of the throughput bottleneck. No Nsight capture was requested; the matrix's `capture_verified: false` refers to that profiling receipt, not an absence of runtime graph replay. There were no transport-fallback warnings.

Final coverage, counted from the plans: **10/10 component gates passed; all 8 serving matrices and their 15 workload legs passed; all 15 numerical processes completed, but 6/12 distribution comparisons failed; all 3 diagnostic-control processes completed.** The original mixed-prefill/decode measurement remains a gap: steady-decode warmups cannot fill it. Also unqualified: 36 populated long contexts, multi-sequence and long-context numerical/retrieval quality, DP-versus-tensor equivalence, the 50-70 tokens/s goal, and the broader local GLM failures above. No production promotion was performed.

Final state and receipts: `.Codex/analysis/glm53-qualification-20261108.QQAX0I/STATE.json`. Each remote step ran detached in tmux with automatic local collection and hash verification, including failures. All **5,928 result files (4,951,529,303 bytes)** were collected and hash-verified; the final provider-side inventory matched the verified manifest before destruction. `provider-down.json` records teardown and `provider-absence.json` records the subsequent provider listing.

## Verification before renting

Use existing test files and the normal project runners through the host build gate. New behavior starts with a failing behavioral test; output-preserving performance refactors start with the existing equivalence tests. Tests assert caller-visible results and invariants, using generated cases where input combinations make individual examples insufficient. Do not manufacture failures or change expected outputs merely to make a test pass.

Run focused checks while building, then one integrated CPU build/test pass and the relevant CUDA compilation. Any safe local GPU correctness checks are serial and leave existing workloads untouched. Record skipped or unavailable hardware coverage explicitly. A future paid run should compare the exact candidate and baseline on the same fixed workload, separately reporting steady decode, prefill overlap, delivered tokens, quality and active memory. That run requires separate authorization.

## Original implementation state (before the November 8 audit)

All five implementation fronts are implemented in the shared checkout and personally reviewed by the parent. CPU and CUDA builds and focused integration checks passed. Prompt-test reader cleanup also passed with thread warnings treated as errors. Nothing has been committed or deployed; the broader GLM test failures below remain unresolved.

| Files | Planned change | Current result |
| --- | --- | --- |
| `scripts/glm53/serve.py`, `server_matrix.py`, `compare_decode.py`; their existing tests | Explicit collective choice and honest comparison receipts | Implemented; parent review found and fixed unknown actual transport on the captured ideal side as well. All 49 launcher/comparison tests passed. |
| `common/speculative.cpp/.h`; `tests/test-speculative-mtp.cpp` | One concurrent depth/probe policy using delivered-token service rate | Implemented; focused CPU and local CUDA correctness passed. |
| `ggml/src/ggml-cuda/ggml-cuda.cu`, `mmq.cu/.cuh`; existing backend/replay tests | Shared routed gate/up preparation | Implemented; 97 local CUDA equivalence cases and captured replay passed. |
| `src/llama-context.cpp/.h`, `llama-ext.h`, `llama-graph.cpp/.h`; `tests/test-llama-archs.cpp` | Owned backend carry/accepted rows and lazy host serialization | Implemented; CPU fixtures and local CUDA store/MTP fixtures passed. |
| `common/arg.cpp`, `common/common.h`, `include/llama.h`, `src/llama-model.cpp`, `tools/server/server-common.h`, `server-context.cpp`; existing prompt/server tests | Useful, owner-balanced, fair prompt chunks with decode protection | Implemented; sparse-owner starvation and short-tail thinning were reproduced and fixed. All nine prompt integration cases passed cleanly after fixing reader teardown; the four reader-dependent cases passed again after the final type refinement. |
| `tests/CMakeLists.txt` | Register focused existing fixtures | CPU registrations and the capture-transition regression added. |
| `ggml/src/ggml-backend-meta.cpp`, `ggml-backend.cpp`, `ggml-impl.h`, `ggml.c`; `tests/test-backend-meta-capture.cpp` | Bound captures by actual recent scheduler graph generations | Wide-to-narrow retention reproduced (0 executables retired instead of 22). Generation-based eviction passed the transition regression and existing 5/11-fragment replay checks on the RTX 5080 (two logical ranks). |

The review follow-up reproduced all three pooled-embedding failures before the fix. Respecting `can_split()` passed all 12 prompt tests, followed by the three embedding cases on the integrated CPU build. Four CPU meta-backend tests passed after the graph-identity change. The CUDA build and both capture tests passed: complete eviction of the old 11-fragment generation and continued reuse of four active 5/11-fragment generations. No OOM or throughput experiment was run; the prior broader GLM failures below were not addressed or rerun in this follow-up.

The initial final gated CPU set passed 7/7: batch allocation, server response/admission fairness, NextN store, ordinary MTP, 36-session logical DP MTP, expert preparation, and the CPU replay reference. The CUDA build and all four local device groups passed.

The local CUDA checks ran serially on the otherwise idle RTX 5080, with the desktop's RTX 5070 Ti excluded by UUID. Four logical ranks on one card exercise ownership and state correctness, not four-card communication, memory capacity or throughput.

The broader `test-glm5next` run is red: an unbound context's CPU compute buffer grows from 1,854,528 to 1,907,424 bytes, and the prefill-padding fixture reports non-bit-identical logits (NMSE 1.70334e-12) and an unobserved cache-mask length. These failures are not waived or counted as passes. Their paths include earlier resident-table/direct-index graph changes; they are not evidence that the new changes are numerically qualified on the deployment model.

The sequential allocator's consecutive-local-stream guard remains in place. Noncontiguous active slots on one owner can still require separate microbatches; this work does not claim to remove that cache-layout restriction. The two temporary Terra 6.1 worker sessions were closed after their scoped work and corrections were reviewed.

At the end of the initial implementation, the remaining hardware qualifications were four physical RTX PRO 6000s, both transports' runtime evidence, populated 36-way long contexts, and matched throughput/quality against baseline. The historical November 6 measurements belonged to the older engine, not this candidate; no improvement was established at that point. The November 8 rental section above records the subsequent evidence and remaining gaps.

Initial local build/test commands assumed nonexistent CMake/CTest/Python PATH shims. After Host Shield's owner identified the gap, outstanding work was stopped and remaining builds use `hostshield run --tenant torad-llama.cpp --class build -- cmake ...`; checks use `hostshield work -- ...`. No safeguards or machine settings were changed.

