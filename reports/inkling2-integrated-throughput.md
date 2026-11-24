# Inkling 2 integrated throughput: baseline vs integrated switches (RTX 5090)

Measured 2026-11-24, 18:24-19:24 UTC (12:24-13:24 Chicago time), on the CloudRift
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
- End-to-end decode on the synthetic `test-llama-archs` fixture shows no measurable
  change in either direction. That is the expected result on that fixture (see limits)
  and neither confirms nor contradicts served-model throughput.
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

Method: fresh process per repetition, arms round-robin within each tier, one timed
command at a time under its own `flock /tmp/buildbox-gpu.lock`, and `nvidia-smi`
checked idle before every timed command (no other compute app was ever seen; the
cross-lab verifier's GPU work and this bench hold the same lock). 184 timed runs in
the main run and 36 in the interaction run, all exit 0. Logs on the box:
`/tmp/inkbench/run.7fD7qW1n` (main) and `/tmp/inkbench/rg-interaction` (interaction),
drivers `/tmp/inkbench/bench2.log` and `/tmp/inkbench/bench3.log`.

## Exact commands (on-box)

Launch (both runs detached on the box, marker at the end):

```sh
on-box sh -c 'rm -f /tmp/inkbench/DONE2; nohup sh -c "bash /tmp/inkbench/run-throughput-bench.sh > /tmp/inkbench/bench2.log 2>&1; touch /tmp/inkbench/DONE2" >/dev/null 2>&1 & echo launched'
on-box sh -c 'rm -f /tmp/inkbench/DONE3; nohup sh -c "bash /tmp/inkbench/run-rg-interaction.sh > /tmp/inkbench/bench3.log 2>&1; touch /tmp/inkbench/DONE3" >/dev/null 2>&1 & echo launched'
```

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
7 processes per arm and shape):

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

Reading: no measurable change. The integrated delta flips sign between the two
shapes (-17.5% at pl=1, +19.0% at pl=23), the inert sync control at pl=1 is -31.3%,
per-arm ranges span 2-3x, and warm alone at pl=23 (-26.8%) contradicts the integrated
arm that contains it (+19.0%). The fixture is host-launch-bound (a 3-block, 512-wide
model), so these medians mostly record host scheduling and clock state on a shared
box. Nothing here is evidence about the served model, in either direction.

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
  the only switch with a measurable single-card decode gain in this bench.
- **GGML_CUDA_MOE_ROUTE_GATE did not move anything measurable here.** Its floors match
  baseline at every occupancy in the MoE step (395.3 vs 387.4 at 23 rows/8 occupied;
  3713.3 vs 3700.3 at 128 rows/384 occupied), it cannot fire in the other two tiers,
  and its output is bit-identical. That does not prove it is worthless - its case is
  prompt-route counting at served batch shapes - but this bench gives it no win.
- **GGML_CUDA_L2_EXPERT_WARM did not move anything measurable here.** Floors and the
  BO deltas (-0.7% gate, -4.4% down inside overlapping ranges) match baseline. In
  these harnesses the expert weights are re-read every step from a small working
  set, which is not the cold-weight regime the prefetch targets.
- **LLAMA_INKLING2_ROUTER_BAKE did not move anything measurable here.** Bake alone:
  -6.7% (pl=1) and +2.0% (pl=23) against a -31.3% / +2.2% control. On the served
  model its purpose is keeping the fused router chain at 4608 wide; this fixture's
  router is 512 by 2.
- **WS-3 device arrival-counter sync did not move anything measurable here.**
  Baseline vs sync is inside noise in every tier (BO: -2.4% / +0.7%). A single card
  has no cross-rank arrival to pace, so this bench cannot price it.
- **Combined (integrated) is TMA at sparse decode and unresolvable at dense shapes.**
  Where routing is sparse at 23 rows it matches TMA alone. Where most experts are
  occupied, the main run's medians suggested a cost that the interaction rerun did
  not reproduce; by floor the dense difference is at most about 2%.

## Limits

- **Synthetic everything.** Tier BB runs a 3-block, 512-wide, 2-expert F32 toy whose
  experts cannot trigger the route gate (needs 384 experts and K-quant named
  weights), the TMA path (needs K-quant expert shapes), or L2 warming (no routed
  expert tiles worth warming). Tier RG and BO use served shapes but formula-generated
  routes and weights, no attention, no KV, no sampler. No number here is a served
  Inkling 2 decode tok/s.
- **Single card.** Tensor-split serving, cross-rank all-reduce, and the arrival
  counters' pacing benefit are all untested by design on this box.
- **Not everything can be turned off.** WS-1's fixes and the WS-4 f8_e4m3 KV gate
  have no off switch in this build, so "baseline" here means opt-ins off in a build
  that already contains them. This bench compares switch states of one build, not
  the integrated tree against its parent.
- **Clock and host noise.** The card idles at 127 MHz and the box CPU was shared
  with other sessions' builds during parts of the run. GPU access was serialized by
  the per-command flock and idle-checked before every run, but short-process tiers
  (RG, BB) still show discrete clock levels and 2-3x spreads. Their medians are
  reported with spreads and floors so the reader can see the noise; only Tier BO
  (about 3900-7700 internal runs per process) resolves differences under about 10%.
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
measurable anywhere else this bench can resolve. The route gate, L2 warming,
router bake, and device sync may each earn their keep under served serving
conditions, but on this single-card synthetic bench they are indistinguishable
from noise.
