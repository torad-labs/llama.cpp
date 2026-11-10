<!-- The rig lead's report of its Nov 9 four-card run, as it wrote it (its evidence folder is not in the repo). -->

# Focused run complete; throughput target not met

The candidate shows a small short-context improvement, but not a consistent overall improvement. The real four-card GPU timeline is preserved and validated. The disposable rental is destroyed and verified absent; estimated cost was about $8.4 against the approved $15 ceiling.

## Delivered throughput

Both engines used 36 concurrent sessions, 1024 measured output tokens per stream, adaptive MTP maximum 2, Q8 indexer queries, microbatch 512, resident cache and internal-nccl-gather transport. Context reservation stayed at 442368 tokens per slot. Rates are the common-overlap client-delivered aggregate divided by 36, not summed server-local rates or instrumented timings.

| Initial prompt | B repeats, tokens/s/session | C repeats, tokens/s/session | Mean B -> C | Mean change |
|---|---|---|---|---|
| 512 tokens | 25.28, 24.81 | 25.54, 25.66 | 25.05 -> 25.60 | +2.21% |
| 8192 tokens | 19.63, 19.31 | 17.74, 19.66 | 19.47 -> 18.70 | -3.96% |

Paired changes were +1.01%/+3.44% at 512 and -9.62%/+1.79% at 8192. The long-context result is inconsistent, including one clearly slower candidate leg. Two repeats do not establish confidence intervals. That slower leg also had lower draft acceptance: 61.39% versus 63.65% in the other candidate leg. Both had a sampled median SM clock of 2295 MHz. This does not establish the cause of the variation.

All eight legs completed. The 50-70 tokens/s/session target was not reached. C differs from B in nine engine files, including earlier correctness, graph-lifetime and projection changes; the measured difference cannot be attributed solely to the two new kernel optimizations. The older 28.5 tokens/s result used 256 output tokens, versus 1024 here, and is not this experiment's before arm.

## Quality and memory

- All 13 physical-card component gates passed.
- The matched 4096-token Q8/ubatch-512 teacher-forced check passed: printed mean KLD -0.000000 and same-top-token agreement 100%. Maximum per-position KLD was 0.000059; this is not a claim of bit-identical full-model logits.
- All four unprofiled servers returned identical 96-token greedy sequences.
- No reserve guard tripped. Minimum sampled free memory was 4934 MiB for B and 5292 MiB for C, above the 4096 MiB reserve. Samples were once per second, not continuous minima.

This single quality check does not certify all contexts, query formats or 36-sequence quality. Cross-query/cross-microbatch discrepancies and 36 simultaneously populated 400K contexts remain unqualified. Nothing was promoted.

## GPU timeline and validator repair

The raw report contains all four selected physical GPUs and no collector diagnostics. Our validator initially rejected it because it counted CUDA launches only on the coordinating CPU thread. GPU 1-3 work was also launched by same-process backend workers.

The fix accepts those workers' API launches only when they begin and finish inside the decode phase, retaining process/correlation-ID matching and the existing device, sequence and diagnostic requirements. A generated regression failed in 24 cases before the fix. Afterward, those cases and 216 invalid-launch probes passed. The capture test module ran 20 tests: 19 passed and one opt-in GPU/Nsight integration was skipped rather than rerun.

The retained real trace then validated offline: 36 distinct decoding sequences, 108 target rows, and 5590 phase-correlated kernels on each GPU. The original failed reports remain unchanged; corrected validation is recorded separately.

Across 46 complete instrumented rounds:

- Mean NVTX round: 75.97 ms. All-device activity union: 62.77 ms/round (82.6%); observed idle: 13.20 ms/round.
- The three largest quantized-matmul signatures total about 23.17 summed GPU-ms/round on device 0. Native NCCL and internal owner reduction contribute about 5.64 and 3.91 respectively; gated-delta recurrence contributes about 2.03.
- These are attribution measurements, not additive elapsed-time savings. Kernels overlap, and the collective groupings do not establish a critical path.
- The full instrumented leg delivered 23.02 tokens/s/session, about 10.1% below unprofiled C's short-context mean. The report's 30.51 figure describes only selected NVTX rounds, not full-leg delivered throughput; it is not used in the comparison.

Quantized matrix multiplication remains the largest named GPU cost, alongside material collective and idle time. Further small recurrence/state-write tuning alone is not a demonstrated route to 50-70. This short-context capture does not diagnose the 8K variation.

## Cost and cleanup

Instance 59921522 on machine 57498 was allocated at $7.028148148/hour including storage, with a verified two-hour timer and earlier destruction deadline. It was destroyed at 12:35 CT on November 9, 2026; provider absence was independently confirmed. Retained instance 59024660 remained listed as exited and was not targeted by any action.

Rig reported approximately $8.36 compute/storage for 1.19 hours; transfers are estimated at roughly one cent. No final provider invoice was retrieved, so about $8.4 is an estimate, not an exact billed total. No top-up, extension or second box occurred.

All 4044 remote result files were collected and hash-verified before destruction. My per-file SCP collection caused an avoidable delay; one compressed archive fixed the transfer. The one-off quality script also overescaped its optional maximum-KLD parser. Its original null field is preserved, and the actual maximum is documented from the raw log. Neither issue changed the throughput measurements or quality contract.

## Evidence and changes

- `THROUGHPUT.json`: every leg, pairs and means.
- `PROFILE-VALIDATED-OFFLINE.json`: corrected validation, raw hashes, launch binding and GPU UUIDs.
- `GPU-REPORT.txt`, `GPU-OBSERVATIONS.json`: timeline details and scope caveats. The report's source display accepts only a bare commit; exact dirty identities remain in the launch and sealed engine receipts.
- `checks/final-collection.json`, `results-profile.sha256`: all 4044 files and their verification.
- `cleanup-confirmed.json`, `teardown.stderr`: provider absence and estimated cost.
- `DECODE-TELEMETRY.json`, `checks/greedy-comparison.json`, `checks/quality-interpretation.json`: supporting observations.
- `stage/engines/{B,C}/SOURCE` and `source.json`: exact engine identities. Candidate patch: `8e9f7c7066fbc943a27bcf79bedef8ad1b5c5f88d2d2173c26beaa2a24446761`.

The offline fix touches only `scripts/glm53/decode_trace.py` and its existing test file. No engine code changed during the rental. No commit, push or promotion was made. The separate QGRE removal request still lacks an identified check; no QGRE files were changed.
