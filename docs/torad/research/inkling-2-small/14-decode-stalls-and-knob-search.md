# Where a four-card decode step goes, and the stalls taken out of it

Every number here comes from four RTX PRO 6000 Blackwell Server Edition cards on one rented box on
2026-11-19, running the `tensor-4-draft` profile of `scripts/inkling2/serve.py` against the
abliterated Q3_K_M pack: `-sm tensor -ts 1,1,1,1`, 23 sessions, `--kv-unified`, a 1,572,864-cell
shared pool, Q8_0 KV. Two reduction routes are compared throughout: the profile's default, which
sums the hidden rows through host memory, and `--reduction-route=engine`, the card-to-card pipeline.
Throughput is the aggregate of 23 simultaneous decoders as `scripts/inkling2/search.py` measures it;
each search trial is one server launch, and the figures below are one trial each unless stated.

## A decode step, kernel by kernel

The step breakdown comes from an Nsight Systems capture with CUDA graphs switched off, so every
kernel is its own trace span, taken in a 6.06-second window while all 23 sessions decoded at a
16,384-token context. The window held 111 steps, 54.6 ms each. The server's own decode trace splits
a step into 44.16 ms of evaluation and 7.28 ms of CPU sampling, in series, with nothing measurable
between them.

| A step on one card | ms | calls a step | mean us |
| --- | ---: | ---: | ---: |
| `mul_mat_vec_q_moe`, Q3_K merged gate/up | 8.81 | 48 | 184.1 |
| `mul_mat_vec_q_moe`, Q4_K down | 4.98 | 45 | 111.1 |
| `ggml_cuda_ar_owner_gather_kernel_n` | 4.10 | 114 | 36.1 |
| `fattn_rel_decode_rows` | 3.63 | 8 | 454.2 |
| `mul_mat_q_few` | 2.70 | 304 | 8.9 |
| `fattn_rel_mma_rows` | 1.63 | 40 | 40.9 |
| `concat_non_cont` | 0.83 | 55 | 15.1 |
| `mul_mat_vec_q_moe`, Q5_K down | 0.43 | 3 | 143.3 |
| everything else, 39 kernels | 6.67 | | |
| every kernel | 33.79 | | |
| card busy, kernels and copies | 34.7 | 3,679 | |
| card idle | 19.9 | | |

The expert projections are 14.2 ms of every step on every card, 26% of the step and 41% of the time
the card is busy. CPU sampling of the step's 23 rows is 7.28 ms during which all four cards are
idle. The remaining 12.6 ms of idle is launch and dependency gaps inside the evaluation: 2,585 gaps
under 20 us a step come to 2.4 ms, and 35 gaps between 0.1 and 1 ms come to 4.1 ms.

The Q3_K gate/up kernel is the sorted row kernel (`GGML_CUDA_MMVQ_EXPERT_SORT_23`), one launch a
layer. Its cost tracks how many distinct experts the step's routes name, not how many routes there
are: the earlier single-card measurements in chapter 10 gave 100.0 us at 61 distinct experts and
273.5 us at 150, a ratio of 2.74 against 2.46 distinct. Each warp reads its expert's rows from
global memory, and the sort only groups routes within a block, so an expert whose routes straddle a
block boundary is read again. At one shard's 1,024 merged rows by 4,608 columns an expert's gate/up
is 2.03 MB, which puts the 184 us launch at roughly half of the card's memory bandwidth. Switching
the sort off costs only 0.8% of the aggregate rate (465.9 against 469.7), so grouping routes inside
a block is not what the kernel is short of.

## The graph-capture stall

`llama_kv_cache::get_n_kv` padded the shared pool to 256 cells. With 23 sessions in one pool the
pool's used length crosses a 256-cell boundary every 11 steps, and each crossing changes every
decode graph's shape, so the backend captured the graphs again. In a 6-second window of 16,384-token
sessions that was 7,720 captures through host memory, the graphs of 10 steps in 110, and card to
card 44 graph instantiations at 17.7 ms each.

The pool now pads as what it is, a dense shared pool: powers of two to 8,192 cells for 23 sequences
and multiples of that above. `LLAMA_KV_PAD_SHARED_LEGACY=1` restores the 256-cell padding and
`LLAMA_KV_PAD_SHARED=N` widens it.

| 23 sessions, no draft | tokens/s |
| --- | ---: |
| through host memory, wide pad | 496.7 |
| through host memory, 256-cell pad, same build | 474.9 |
| through host memory, 256-cell pad, before the build | 469.7 |
| card to card, wide pad | 481.0 |
| card to card, 256-cell pad, before the build | 424.4 |

With the draft on the same comparison is 376.1 against 370.9. At a 16,384-token context the window
rate per session went from 18.17 to 20.0 tokens/s. The card-to-card route was 11% behind the host
route before the fix and is 3% behind it after, because its cost was dominated by instantiating
graphs rather than capturing them.

A capture also calls `cudaMemGetInfo`, which the earlier window showed 7,796 times for 214.9 ms.

The two routes and the two paddings each produce a different `tokens_sha256` on the search load. The
load samples at temperature 1.0 from a fixed per-slot seed, so identical logits give identical
tokens; a different padded width changes how many masked cells attention sums over and in what
order, as a different reduction route changes the order of the hidden sum. Neither changes what the
operator checks compare.

## The owner sum under the draft

The chunked owner-reduce kernel sums only each card's own 1,152 convolution channels and skips three
quarters of the exchange. The meta backend recognized that pattern at exactly 23 hidden rows, so
every drafted step, which carries 23 sessions times its accepted draft tokens, fell back to the full
four-card reduction. The recognizer now reads the row count off the root tensor. Card to card with
the draft at depth 3: 372.8 tokens/s with the owner sum, 360.5 with `GGML_CUDA_AR_REDUCE_OWNERS=0`.

The periodic owner layout in that kernel, measured earlier the same day, gave 424.4 against 413.2.

## What the knobs are worth

Trials on the build before the two fixes above, one trial each, 23 sessions:

| Setting | tokens/s |
| --- | ---: |
| the profile as it launches, no draft | 469.7 |
| `--expert-row-limit=32` | 478.7 |
| `--expert-row-limit=8` | 467.6 |
| `GGML_TENSOR_N_SPLIT_ROWS=0` | 472.5 |
| `GGML_TENSOR_N_SPLIT_ROWS=64` | 468.1 |
| `--sort-expert-rows=off` | 465.9 |
| `--host-owner-rows=off` | 464.4 |
| the profile as it launches, draft at depth 3 | 361.1 |
| `LLAMA_SAMPLE_THREADS=1` | 302.9 |
| `LLAMA_SAMPLE_THREADS=8` | 373.4 |
| `LLAMA_SAMPLE_THREADS=16` | 386.0 |
| 16 threads and `--draft-depth=1` | 393.7 |
| 16 threads and `--draft-depth=2` | 389.2 |
| 16 threads and `--draft-depth=4` | 366.4 |
| 16 threads and `--draft=off` | 509.1 |

The sampling thread count is the largest single setting on the list. `sample_cpu_batch` samples a
step's rows on `min(--threads, LLAMA_SAMPLE_THREADS)` threads and the registry default was 4, so 23
rows went through in about six series at 1.37 ms a row over the 201,088-token vocabulary. Raising it
is worth about 8% with the draft and without it.

The DSpark draft as it stands costs about a quarter of the throughput at this batch: 386.0 with it
against 509.1 without, both at 16 sampling threads. Shallower drafts are better than the profile's
depth of 3. That is the drafter's domain, not this chapter's.

## Timelines

Window rate per session at a 16,384-token context, 23 sessions, in the order they were captured:

| Capture | tokens/s a session |
| --- | ---: |
| host route, 256-cell pad | 18.17 |
| host route, no CUDA graphs | 18.5 |
| host route, wide pad | 20.0 |
| card-to-card route, 256-cell pad | 16.83 |
| host route, 256-cell pad, draft at depth 3 | 12.67 |

CUDA graphs are worth 9% over no graphs at all once the pool stops changing shape, because the host
route still cuts a step into graph segments with the reductions running live between them. The
wide-pad capture counted 93,884 graph spans on four cards in 6.045 s, 196 a card a step, and only
three kernels ran outside a graph: the owner gather, the add and the BF16 convert. A graph-trace run
counted 18,160 graph launches a card in a 6-second window at 2.2 to 3.4 us of host time each, and
the earlier window showed `cudaEventSynchronize` 43,020 times for 1,806.6 ms, four to a reduction.

Counting graph spans as card time, that capture's 50.4 ms step is 33.5 ms of card work and 16.8 ms
of idle: 8.24 ms in one gap a step, which is the sampling window, 4.0 ms in 33 gaps between 0.1 and
1 ms, which are the waits between graph segments, and 4.3 ms in about 860 shorter gaps. The route's
copies are not the constraint: 38.8 MB leaves each card and 18.5 MB arrives per step, and the cards
spend 1.6 ms a step on them.

## What is left, in the order the data puts it

1. The expert projections, 14.2 ms of a 54.6 ms step on every card. The ring kernel in
   `mmvq-moe.cu` already reads each distinct expert exactly once and streams its row tiles through
   shared memory with bulk copies, but it admits only IQ3_XXS and Q8_0 and at most 64 token-expert
   pairs, where 23 sessions name 184. Extending it to the K-quants and to that many pairs is the
   largest piece of work the measurements point to.
2. CPU sampling, 7.28 ms of a step with the old default and about 2 ms at 16 threads. A one-pass
   vector form of the sample chain measured 0.175 ms a row against 1.373 ms on the box's CPU and
   picked the same token on 576 of 576 rows; it would make the stall about 0.25 ms. Sampling on the
   card would remove it from the step altogether.
3. The four-card gather, 4.10 ms a step a card at 36.1 us a call, which includes waiting for the
   slowest card. The host route's event synchronizes and its two-slot pool are in the same budget.
4. Relative attention, 5.26 ms a step a card across its two kernels.
