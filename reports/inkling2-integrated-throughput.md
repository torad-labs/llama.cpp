# Inkling 2 integrated throughput: baseline vs integrated switches (RTX 5090)

Measured 2026-11-24, 18:24-19:40 UTC (12:24-13:40 Chicago time), on the CloudRift
RTX 5090 box (ssh host `buildbox`, host `riftvm`) through `on-box`. Every number below
is single-card and synthetic. Read the limits section before quoting any of it.

## Headline

- The one clear gain is the TMA expert path at 23 decode rows. In the stable
  backend-ops tier it cuts the 23-row Q3_K gate/up expert product from 239.8 to
  205.5 us/run (median, -14.3%). The integrated arm measures the same 205.6 us/run.
- In the served-shape MoE timer, every TMA-bearing arm reaches a 220-230 us/step
  floor at 23 rows with 8 of 384 experts occupied, against a 387-408 us/step floor
  for every non-TMA arm (-41 to -43% by best observed). The route gate alone and
  L2 warming alone do not move that floor.
- The shipping default (device sync on, every opt-in off) shows no single-sequence
  regression. Two calmer reruns settled the noisy first fixture run: pooled across
  22 paired repetitions the default arm sits at 0.996x the legacy-sync baseline
  (slower in 11 of 22), and TMA alone at +0.08% median against baseline. See Tier BB.
- With every opt-in on, the integrated arm pays a real, reproducible cost on the
  1-sequence toy fixture only: -9.3% decode throughput against the default arm
  (paired ratio 0.908, slower in 14 of 15). It follows `GGML_CUDA_L2_EXPERT_WARM`
  (about -7 points) plus `LLAMA_INKLING2_ROUTER_BAKE` (about -2 points) - the
  costs that follow those switches in a zero-benefit case (2 tiny F32 experts, a
  step that is almost all host overhead). No such cost appears at served shapes
  (Tiers RG, BO).
- Device arrival-counter sync (WS-3) shows no measurable single-card effect in any
  tier here. Its benefit was host sync calls on multi-rank paths, which a single
  card does not exercise.

## Build and box

- Build commit: `c41aacd3063766f72fb2cf8d5699642d2943a31d` (the integrated build of
  reports/inkling2-integrated-verification.md), reused as-is. No rebuild for this run.
- Build dir: `/home/riftuser/build-campaign-inkling2` (Ninja, Release,
  CMAKE_CUDA_ARCHITECTURES=120a, GGML_CUDA=ON). Binaries dated 2026-11-24 17:04 UTC.
- GPU: NVIDIA GeForce RTX 5090, compute capability 12.0, 32607 MiB, driver 580.159.03,
  max SM clock 3090 MHz, power limit 575 W. CUDA toolkit 13.3.73.
- libggml-cuda.so.1.0.0 sha256 70f12ba267d2c5338be7356b1d5a2e8f50adac5d00c1615d8896d3eddd62a2ef.
- Synthetic fixture: `inkling2-moe.gguf` written by `test-llama-archs -a inkling2 -s 1234`,
  sha256 688a62ec8c8717ee0a041aece447120a7055d9390bdfec716dc540c8c3993330, 14 MB, 119 tensors.
  Its dimensions are toy-sized: 3 blocks, embedding 512, context 256, vocab 130,
  2 experts (both used), expert feed-forward 24, all tensors F32.

## Arms

Each arm is one environment, set explicitly per process. Diagnostics are off in every
timed run (`GGML_LEDGER=0 GGML_CUDA_MMQ_ROUTE_HIST=0`; the route log defaults off).

| Arm | GGML_CUDA_MOE_ROUTE_GATE | LLAMA_INKLING2_ROUTER_BAKE | GGML_CUDA_L2_EXPERT_WARM | GGML_CUDA_MMQ_TMA | GGML_CUDA_AR_COUNTER_SYNC_LEGACY | GGML_CUDA_COUNTER_SYNC_LEGACY |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 0 | 0 | 0 | 0 | 1 | 1 |
| sync | 0 | 0 | 0 | 0 | 0 | 0 |
| route | 1 | 0 | 0 | 0 | 0 | 0 |
| bake | 0 | 1 | 0 | 0 | 0 | 0 |
| warm | 0 | 0 | 1 | 0 | 0 | 0 |
| tma | 0 | 0 | 0 | 1 | 0 | 0 |
| integrated | 1 | 1 | 1 | 1 | 0 | 0 |

Baseline means every Inkling 2 opt-in off plus the two legacy sync switches on, because
WS-3 made device arrival-counter sync the default. The `sync` arm isolates that one
difference (baseline vs sync is the WS-3 sync delta and also serves as a noise control).
Each per-switch arm is measured against `sync`. A second short run added `tma-warm` and
`route-tma` arms (same columns, the two named switches on) to check one interaction.
Two later 1-sequence fixture reruns reused these arms; the second attributing run
compared `sync`, `bake`, `warm`, and `integrated`.

Method: fresh process per repetition, arms round-robin within each tier, one timed
command at a time under its own `flock /tmp/buildbox-gpu.lock`, and `nvidia-smi`
checked idle before every timed command (no other compute app was ever seen; the
cross-lab verifier's GPU work and this bench hold the same lock). The two 1-sequence
reruns rotated the starting arm each round. 184 timed runs in the main run, 36 in
the interaction run, 75 in the 1-sequence rerun, and 60 in the attribution run, all
exit 0. Logs on the box: `/tmp/inkbench/run.7fD7qW1n` (main),
`/tmp/inkbench/rg-interaction` (interaction), `/tmp/inkbench/bb-pl1-rerun`
(1-sequence rerun), and `/tmp/inkbench/bb-pl1-attr` (attribution), drivers
`/tmp/inkbench/bench2.log` through `/tmp/inkbench/bench5.log`.

## Exact commands (on-box)

Launch (all runs detached on the box, marker at the end):

```sh
on-box sh -c 'rm -f /tmp/inkbench/DONE2; nohup sh -c "bash /tmp/inkbench/run-throughput-bench.sh > /tmp/inkbench/bench2.log 2>&1; touch /tmp/inkbench/DONE2" >/dev/null 2>&1 & echo launched'
on-box sh -c 'rm -f /tmp/inkbench/DONE3; nohup sh -c "bash /tmp/inkbench/run-rg-interaction.sh > /tmp/inkbench/bench3.log 2>&1; touch /tmp/inkbench/DONE3" >/dev/null 2>&1 & echo launched'
on-box sh -c 'rm -f /tmp/inkbench/DONE4; nohup sh -c "bash /tmp/inkbench/run-bb-pl1-rerun.sh > /tmp/inkbench/bench4.log 2>&1; touch /tmp/inkbench/DONE4" >/dev/null 2>&1 & echo launched'
on-box sh -c 'rm -f /tmp/inkbench/DONE5; nohup sh -c "bash /tmp/inkbench/run-bb-pl1-attr.sh > /tmp/inkbench/bench5.log 2>&1; touch /tmp/inkbench/DONE5" >/dev/null 2>&1 & echo launched'
```

The 1-sequence rerun ran 19:29-19:34 UTC and the attribution run 19:36-19:40 UTC.

Fixture written once, under the GPU lock, fixed seed:

```sh
B=/home/riftuser/build-campaign-inkling2
flock /tmp/buildbox-gpu.lock $B/bin/test-llama-archs -a inkling2 -s 1234 -o /tmp/inkbench/fixture1234
```

Each timed command ran as `flock /tmp/buildbox-gpu.lock env <ARM-ENV> <CMD>` where
`<ARM-ENV>` is the arm's row above plus `GGML_LEDGER=0 GGML_CUDA_MMQ_ROUTE_HIST=0`.
Example, the integrated arm: `GGML_CUDA_MOE_ROUTE_GATE=1 LLAMA_INKLING2_ROUTER_BAKE=1
GGML_CUDA_L2_EXPERT_WARM=1 GGML_CUDA_MMQ_TMA=1 GGML_CUDA_AR_COUNTER_SYNC_LEGACY=0
GGML_CUDA_COUNTER_SYNC_LEGACY=0 GGML_LEDGER=0 GGML_CUDA_MMQ_ROUTE_HIST=0`.

Tier RG, served-shape MoE timer (hidden 4608, intermediate 2304, 384 experts, top-8;
Q3_K gate/up, Q4_K or Q5_K down; 5 timed steps after 1 warmup step per process; 9
processes per arm):

```sh
flock /tmp/buildbox-gpu.lock env <ARM-ENV> $B/bin/test-inkling-route-gate /dev/null
```

Tier BB, end-to-end decode on the synthetic fixture (built-in 16-token warmup;
7 processes per arm and shape in the main run, then two 15-process-per-arm reruns
at pl=1 with the starting arm rotated each round):

```sh
flock /tmp/buildbox-gpu.lock env <ARM-ENV> $B/bin/llama-batched-bench -m /tmp/inkbench/fixture1234/inkling2-moe.gguf -ngl 99 -c 2048 -b 2048 -ub 512 -npp 128 -ntg 512 -npl 1
flock /tmp/buildbox-gpu.lock env <ARM-ENV> $B/bin/llama-batched-bench -m /tmp/inkbench/fixture1234/inkling2-moe.gguf -ngl 99 -c 8192 -b 2048 -ub 512 -npp 64 -ntg 256 -npl 23
```

Tier BO, served-shape expert products (perf mode times at least 1 s per process,
about 4000-7700 internal runs; 3 processes per arm):

```sh
flock /tmp/buildbox-gpu.lock env <ARM-ENV> $B/bin/test-backend-ops perf -b CUDA0 -o MUL_MAT_ID -p 'type_a=q3_K,type_b=f32,n_mats=384,n_used=8,b=1,m=1024,n=23,k=4608'
flock /tmp/buildbox-gpu.lock env <ARM-ENV> $B/bin/test-backend-ops perf -b CUDA0 -o MUL_MAT_ID -p 'type_a=q4_K,type_b=f32,n_mats=384,n_used=8,b=0,m=4608,n=23,k=512'
```

Output-identity check (not timed): same RG command with a real output file instead of
`/dev/null`, then `sha256sum`. Graph-launch diagnostic (not timed): the pl1 BB command
with `GGML_CUDA_TIME_LAUNCH=1` added, for baseline and integrated. It emitted no
`cudaGraphLaunch` lines, so launch host cost and kernel launch counts were not
collected; nothing below claims them.

## Results: backend-ops served expert products (Tier BO)

us/run, median of 3 per-process means [min, max]. This is the stable tier: spreads
are a few percent and the sync control sits within -2.4% / +0.7% of baseline.

| Case | Arm | us/run median [min, max] | Delta vs baseline |
| --- | --- | --- | --- |
| gate/up Q3_K, m=1024 n=23 k=4608 | baseline | 239.83 [232.19, 255.67] | - |
| | sync | 234.00 [227.22, 243.46] | -2.43% |
| | warm | 238.20 [227.84, 248.05] | -0.68% |
| | tma | 205.50 [202.07, 211.31] | -14.31% |
| | integrated | 205.60 [201.97, 208.04] | -14.27% |
| down Q4_K, m=4608 n=23 k=512 | baseline | 141.46 [130.82, 147.30] | - |
| | sync | 142.48 [136.86, 144.52] | +0.72% |
| | warm | 135.20 [134.85, 145.25] | -4.43% |
| | tma | 141.77 [134.86, 142.93] | +0.22% |
| | integrated | 143.98 [136.25, 146.15] | +1.78% |

Reading: TMA cuts the Q3_K gate/up slice by about 14% at 23 rows, well outside the
control noise, and the integrated arm equals TMA alone. On the Q4_K down slice
nothing separates from baseline (the warm delta sits inside fully overlapping
ranges). The route gate cannot fire in this tier (its dispatch requires the served
weight tensor names, `ggml/src/ggml-cuda/fuse-moe.cu:791`), and the bake does not
apply (no model load), so this tier attributes TMA and warm only.

## Results: served-shape MoE step (Tier RG)

us/step from `test-inkling-route-gate` (source timer at `tests/test-inkling-route-gate.cpp:93`).
This tier is noisy in a specific way: each process's samples land in one of two or
three discrete levels (for example about 1170 or about 1830 us at 128 rows), and the
box preamble shows the card idling at 127 MHz, so short 3-second processes time
whichever clock state they land in. Both the median and the best observed (min, the
fast-clock floor) are reported. n=9 processes per arm.

Baseline vs integrated, all cases:

| Case (rows, occupied of 384, down) | Baseline median [min, max] | Integrated median [min, max] | Delta median | Delta min |
| --- | --- | --- | --- | --- |
| 23, 8, q4_K | 579.7 [387.4, 1556.6] | 247.7 [229.6, 902.1] | -57.3% | -40.7% |
| 23, 184, q4_K | 1952.0 [1748.1, 2749.2] | 2282.0 [1940.0, 2805.4] | +16.9% | +11.0% |
| 128, 8, q4_K | 1829.1 [1170.3, 2996.2] | 2217.6 [1433.3, 2638.4] | +21.2% | +22.5% |
| 128, 384, q4_K | 3904.4 [3700.3, 5769.3] | 3873.6 [3742.2, 4345.7] | -0.8% | +1.1% |
| 128, 0 local, q5_K | 581.2 [188.7, 1532.4] | 576.8 [188.6, 828.5] | -0.8% | -0.1% |
| 128, 96 local, q5_K | 1829.1 [1117.1, 2486.1] | 1829.9 [1125.6, 2652.1] | +0.0% | +0.8% |
| 512, 8, q4_K | 3999.6 [2681.8, 5656.4] | 3946.6 [2672.2, 4152.7] | -1.3% | -0.4% |
| 512, 384, q4_K | 4404.1 [4287.8, 5353.4] | 4943.2 [4354.8, 5469.1] | +12.2% | +1.6% |

Per-switch floors (min us/step, same run):

| Case | baseline | sync | route | warm | tma | integrated |
| --- | --- | --- | --- | --- | --- | --- |
| 23, 8 | 387.4 | 388.4 | 395.3 | 408.1 | 220.2 | 229.6 |
| 23, 184 | 1748.1 | 1762.6 | 1750.7 | 1745.2 | 1696.8 | 1940.0 |
| 128, 8 | 1170.3 | 1175.9 | 1183.8 | 1172.2 | 1183.9 | 1433.3 |
| 128, 384 | 3700.3 | 3696.0 | 3713.3 | 3688.5 | 3687.7 | 3742.2 |
| 512, 384 | 4287.8 | 4279.0 | 4385.4 | 4312.2 | 4297.3 | 4354.8 |

Reading: only the 23-row sparse case separates. Its fast floor exists only in
TMA-bearing arms (220.2 tma, 229.6 integrated; 219.9 route-tma and 240.0 tma-warm in
the interaction run), while route alone (395.3) and warm alone (408.1) sit at the
baseline floor. So the sparse decode gain in this harness is the TMA path
(`GGML_CUDA_MMQ_TMA`, few-row dispatch capped at 32 tokens,
`ggml/src/ggml-cuda/mmq-tma-tiles.cu:39`), not the route gate. Two median red flags
did not reproduce: integrated at 23 rows/184 occupied (+16.9% median, +11.0% min)
and at 128 rows/8 occupied (+21.2%). In the interaction rerun the floors at
23 rows/184 occupied were 1934.6 (tma), 1938.9 (tma-warm), 1942.1 (route-tma) and
1939.0 (integrated) - equal - and at 128 rows/8 occupied three of four arms floored
at the same 1831-1834 us level (the fourth caught the 1192 us fast level once),
both levels already present inside the main run's spreads. The inert
`tma` arm at 512 rows (TMA cannot fire above 32 tokens) still shows a +7.6% median
delta in the main run, which sizes this tier's median noise. Treat the dense-case
medians (128-512 rows, or 184-384 occupied) as no measurable change, with a
possible small dense cost (0-2% by floor at 512 rows/384 occupied) that this
harness cannot resolve.

## Results: end-to-end decode on the synthetic fixture (Tier BB)

Decode throughput from `llama-batched-bench` (S_TG), median of 7 processes [min, max];
ms/step is pl / S_TG (one batched decode step across pl sequences).

| Shape | Arm | TG t/s median [min, max] | ms/step | Delta vs baseline |
| --- | --- | --- | --- | --- |
| pl=1, pp128 tg512 | baseline | 466.78 [267.20, 726.51] | 2.142 | - |
| | sync | 320.78 [217.47, 624.75] | 3.117 | -31.3% |
| | route | 389.60 [219.05, 634.37] | 2.567 | -16.5% |
| | bake | 435.44 [254.18, 667.51] | 2.297 | -6.7% |
| | warm | 313.43 [177.50, 538.60] | 3.191 | -32.9% |
| | tma | 431.17 [264.24, 534.17] | 2.319 | -7.6% |
| | integrated | 385.17 [250.18, 469.65] | 2.596 | -17.5% |
| pl=23, pp64 tg256 | baseline | 6406.04 [3675.68, 9847.41] | 3.590 | - |
| | sync | 6548.69 [4012.72, 11262.63] | 3.512 | +2.2% |
| | route | 6558.00 [2996.36, 9481.89] | 3.507 | +2.4% |
| | bake | 6536.39 [3635.76, 11318.70] | 3.519 | +2.0% |
| | warm | 4691.20 [2361.06, 7334.47] | 4.903 | -26.8% |
| | tma | 6287.15 [5295.58, 10185.72] | 3.658 | -1.9% |
| | integrated | 7620.82 [3628.91, 11223.25] | 3.018 | +19.0% |

The main run alone cannot settle pl=1. Per-arm ranges span 2-3x, and on this fixture
`route` and `tma` are functionally identical to `sync` (neither switch can fire on
2 F32 experts at these dims), yet their three medians span 320.78 to 431.17 t/s
(1.34x) - wider than the integrated arm's -17.5% against baseline. The median in
this report that did not reproduce was a Tier RG dense-case median, not this 1-seq
median; pl=1 had no rerun, so it got two. Per-repetition pl=1 values from the main
run, in repetition order (S_TG t/s):

- baseline: 466.78, 278.50, 267.20, 726.51, 519.77, 574.39, 402.39
- sync (the shipping default): 369.76, 289.18, 320.78, 369.87, 624.75, 313.51, 217.47
- route (identical-config control): 389.60, 270.18, 326.07, 455.54, 634.37, 522.83, 219.05
- tma: 297.71, 264.24, 464.69, 531.30, 431.17, 351.94, 534.17
- integrated: 301.09, 400.61, 385.17, 469.65, 250.18, 294.29, 467.76

1-sequence rerun (15 fresh processes per arm, starting arm rotated each round,
19:29-19:34 UTC). Paired ratio is the median of arm/reference S_TG within the same
repetition round, which cancels round-level host and clock state:

| Arm | TG t/s median [min, max] | ms/step | Delta vs baseline | Paired vs baseline |
| --- | --- | --- | --- | --- |
| baseline | 587.51 [307.24, 678.69] | 1.702 | - | - |
| sync (shipping default) | 558.16 [288.60, 700.58] | 1.792 | -5.00% | 1.002, faster in 8 of 15 |
| tma | 587.98 [392.2, 673.5] | 1.701 | +0.08% | - |
| integrated | 528.53 [353.04, 643.25] | 1.892 | -10.04% | 0.903, slower in 10 of 15 |

The `route` control median was 567.07 t/s (-3.48%), so the three identical-config
arms (sync, route, tma) span 1.053x by median in this rerun. Per-repetition values
(S_TG t/s, repetition order):

- baseline: 587.5, 506.6, 456.2, 307.2, 359.9, 428.7, 543.8, 661.1, 674.1, 599.8,
  665.5, 591.2, 609.2, 567.3, 678.7
- sync: 501.5, 533.1, 512.8, 525.3, 659.3, 288.6, 617.2, 700.6, 544.9, 594.3,
  558.2, 539.4, 610.2, 573.9, 661.1
- tma: 392.2, 427.1, 392.9, 653.0, 591.2, 664.8, 644.6, 570.9, 611.6, 515.4,
  588.0, 494.7, 635.1, 577.7, 673.5
- integrated: 485.6, 353.0, 535.7, 408.1, 436.0, 643.2, 574.8, 485.2, 443.4, 541.5,
  546.8, 551.9, 528.5, 529.8, 486.9

Pooling the main run and this rerun (22 paired rounds) settles the two arms that
matter for the merge: the shipping default sits at 0.996x baseline (slower in 11
of 22, a dead heat), and tma at 1.013x the shipping default (slower in 10 of 22).
Neither shows a 1-sequence regression. The integrated arm still pointed slower,
but its main-run paired comparison against sync had pointed the other way (1.201,
slower in 3 of 7), so a calm-box attribution run separated the two switches that
can act on this fixture (19:36-19:40 UTC, 15 processes per arm):

| Arm | TG t/s median [min, max] | ms/step | Delta vs sync | Paired vs sync |
| --- | --- | --- | --- | --- |
| sync (shipping default) | 802.65 [699.56, 828.19] | 1.246 | - | - |
| bake | 788.76 [747.17, 806.96] | 1.268 | -1.73% | 0.977, slower in 13 of 15 |
| warm | 744.78 [711.74, 758.46] | 1.343 | -7.21% | 0.927, slower in 14 of 15 |
| integrated | 727.86 [711.41, 746.10] | 1.374 | -9.32% | 0.908, slower in 14 of 15 |

The two single-switch paired ratios multiply to the integrated ratio
(0.977 x 0.927 = 0.906 vs 0.908 measured), so the cost is additive:
`GGML_CUDA_L2_EXPERT_WARM` carries about 7 of the 9 points and
`LLAMA_INKLING2_ROUTER_BAKE` about 2. Pooling main and attribution runs against
sync gives bake 0.982 (slower in 14 of 22), warm 0.925 (18 of 22), integrated
0.912 (17 of 22). This is a fixture-only result: the step is about 1.25 ms of
mostly host and launch overhead for a 3-block, 512-wide model whose 2 tiny F32
experts are always resident, so the warm plan's per-step side work is pure
overhead and the bake has no benefit to offset the small cost that follows it. At
served shapes the same
switches show no cost (Tier RG floors, Tier BO above). The reruns cover pl=1 only;
pl=23 remains unresolved noise (integrated +19.0% against warm alone at -26.8% in
the main run) and no claim is made from it.

## Output identity

- Route gate alone is bit-identical to baseline on the served-shape timer output:
  both sha256 632c66ad9ee2fed4ca485906dd077ce13a73bea52d39183ba3e4b3c0e2c797b2.
- Integrated differs: sha256 b9a34681cb66b9b67fca4e3cb2fb5899b3320238fb5504cd3346c117e4e0d8ca.
  Expected: `GGML_CUDA_MMQ_TMA` is declared `output = "rounding"` in tunables.toml:176,
  so bit-identity is not part of its contract. Its numerical agreement is covered by
  the parity tests in the integrated verification report, not by this bench.

## Honest reading, switch by switch

- **GGML_CUDA_MMQ_TMA moved.** -14.3% on the stable 23-row Q3_K gate/up tier and the
  220-230 us sparse floor in the MoE step. No effect on the Q4_K down slice. This is
  the only switch with a measurable single-card decode gain in this bench. It also
  shows no 1-sequence regression: +0.08% median against baseline in the 15-rep
  rerun and a pooled paired 1.013x against the shipping default over 22 rounds.
- **GGML_CUDA_MOE_ROUTE_GATE did not move anything measurable here.** Its floors match
  baseline at every occupancy in the MoE step (395.3 vs 387.4 at 23 rows/8 occupied;
  3713.3 vs 3700.3 at 128 rows/384 occupied), it cannot fire in the other two tiers,
  and its output is bit-identical. That does not prove it is worthless - its case is
  prompt-route counting at served batch shapes - but this bench gives it no win.
- **GGML_CUDA_L2_EXPERT_WARM: no served-shape effect, a real toy-fixture cost.**
  Floors and the BO deltas (-0.7% gate, -4.4% down inside overlapping ranges) match
  baseline. In these harnesses the expert weights are re-read every step from a
  small working set, which is not the cold-weight regime the prefetch targets. On
  the 1-sequence toy fixture, where it can only pay its fixed per-step cost, it
  measures -7.21% against the default arm (paired 0.927, slower in 14 of 15;
  pooled 0.925 over 22 rounds). That prices its overhead in the zero-benefit case,
  not a served-shape regression.
- **LLAMA_INKLING2_ROUTER_BAKE: no served-shape effect, a small toy-fixture cost.**
  Bake alone measures -1.73% against the default arm at 1 sequence on the fixture
  (paired 0.977, slower in 13 of 15; pooled 0.982 over 22 rounds). On the served
  model its purpose is keeping the fused router chain at 4608 wide; this fixture's
  router is 512 by 2, so again only fixed cost shows.
- **WS-3 device arrival-counter sync did not move anything measurable here.**
  Baseline vs sync is inside noise in every tier (BO: -2.4% / +0.7%; 1-sequence
  fixture pooled paired 0.996 over 22 rounds, slower in 11 of 22). The shipping
  default carries no 1-sequence regression. A single card has no cross-rank
  arrival to pace, so this bench cannot price its benefit.
- **Combined (integrated) is TMA at sparse decode and unresolvable at dense shapes.**
  Where routing is sparse at 23 rows it matches TMA alone. Where most experts are
  occupied, the main run's medians suggested a cost that the interaction rerun did
  not reproduce; by floor the dense difference is at most about 2%. On the
  1-sequence toy fixture the all-on arm pays -9.32% against the default arm
  (paired 0.908), the sum of the warm and bake fixed costs above. That arm is not
  the shipping default.

## Limits

- **Synthetic everything.** Tier BB runs a 3-block, 512-wide, 2-expert F32 toy whose
  experts cannot trigger the route gate (needs 384 experts and K-quant named
  weights) or the TMA path (needs K-quant expert shapes), and give L2 warming and
  the router bake no possible benefit to offset their fixed costs (2 tiny
  always-resident experts, a 512 by 2 router). Tier RG and BO use served shapes but
  formula-generated routes and weights, no attention, no KV, no sampler. No number
  here is a served Inkling 2 decode tok/s.
- **Single card.** Tensor-split serving, cross-rank all-reduce, and the arrival
  counters' pacing benefit are all untested by design on this box.
- **Not everything can be turned off.** WS-1's fixes and the WS-4 f8_e4m3 KV gate
  have no off switch in this build, so "baseline" here means opt-ins off in a build
  that already contains them. This bench compares switch states of one build, not
  the integrated tree against its parent.
- **Clock and host noise.** The card idles at 127 MHz and the box CPU was shared
  with other sessions' builds during parts of the run. GPU access was serialized by
  the per-command flock and idle-checked before every run, but short-process tiers
  (RG, BB) in the main run still show discrete clock levels and 2-3x spreads. Their
  medians are reported with spreads and floors so the reader can see the noise. The
  two later 1-sequence BB reruns ran on a calmer box with tight spreads (the
  attribution run's default arm spans 699.56-828.19 t/s) and rotated starts, so
  their paired ratios resolve fixture-level fixed costs of a few percent. Only
  Tier BO (about 3900-7700 internal runs per process) resolves served-shape
  differences under about 10%.
- **Occupancy is synthetic.** The sparse case (8 of 384 experts) and dense case
  (184-384 of 384) bracket served decode; where a real 23-session step lands
  between them depends on the served router's correlation, which this bench does
  not model.
- No CUDA graph launch cost or kernel launch counts: the `GGML_CUDA_TIME_LAUNCH`
  diagnostic pair emitted no lines, so none are claimed.

## Bottom line

For the combined-flags question: on the shapes that look like served Inkling 2
decode, turning everything on buys the TMA win (about 14% on the 23-row Q3_K
gate/up expert product, and the matching sparse-step floor drop) and costs nothing
measurable at those shapes. The one cost this bench does resolve is on the toy
fixture at 1 sequence, where the all-on arm pays about 9% against the shipping
default in overhead that follows L2 warming and the router bake, with no benefit
those switches could return on that fixture. The shipping default itself,
and TMA alone, show no 1-sequence regression (pooled paired 0.996 and 1.013 over
22 rounds). The route gate, L2 warming, router bake, and device sync may each earn
their keep under served serving conditions, but at served shapes on this
single-card synthetic bench they are indistinguishable from noise.
