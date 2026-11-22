# What to measure to know where Inkling 2 and GLM-5.3 Flash can get faster

Both models serve 23 agent sessions on four RTX PRO 6000 cards with no NVLink. Each list below is a thing
to measure, what it would tell us, and whether we have it today. The point of the whole list is that a
cost can no longer hide: every millisecond of a step and every byte of a card has a named owner, on the
traffic we serve, for both models, in one format.

**Have** means: `yes` a tool in the repo produces it on demand; `partly` it was read once by hand, or for
one model only, or needs a profiler window; `no` nothing produces it.

**Models** says where a number means something, not who has to build it. Five rows are about one
architecture (26, 27, 32, 33 and the vision tower in 16). Every other row is the same reading for any model,
and the engine takes those itself:

## What the engine now keeps by itself

The ledger (`ggml/include/ggml-ledger.h`) is on by default in every process, for any model. Code names what
it counts or times; nothing in it knows an architecture. A running server gives it at `GET /ledger`, a process
writes it at exit with `GGML_LEDGER_FILE`, and `scripts/ledger.py` prints the difference between two
snapshots a step. `scripts/inkling2/search.py` now keeps it with every trial.

| Row | What the ledger holds | Slots |
| ---: | --- | --- |
| 4 | Rows and sequences of every decode and ubatch, as a histogram | `decode.rows`, `decode.ubatch.*` |
| 8 | Host time of every decode phase and every server stage | `decode.*`, `stage.*` |
| 8, 10 | On one decode in eight, how long each device had the step's work, against the step's wall time | `cuda.span_us[d]` |
| 12 | Graphs built, allocated, captured, instantiated and evicted, and each kept graph turned down with its reason | `graph.kept.reject.*`, `cuda.graph.*`, `meta.capture.*` |
| 14 | The spread of every one of these by power of two, so a rare slow step shows beside the median | every slot |
| 16 | Memory a context holds on each device for weights, its own state and compute, and the device's used memory | `mem.ctx*`, `cuda.mem.used_mib[d]` |
| 22 | Host time of every reduction and gather between devices, and the ones the fast route refused | `meta.*` |
| 39 | Every graph a device ran kernel by kernel, with why it was not replayed | `cuda.eval.direct.*[d]` |

What it does not hold yet: time by kernel family (9), hardware counters (18), expert routing and attention
statistics (24 to 27), and traffic (34 to 38).

The two models, for reading the lists:

| | Inkling 2 Small | GLM-5.3 Flash |
| --- | --- | --- |
| Layers | 48, all MoE, 40 sliding-window and 8 global | 45 plus one MTP block, 34 recurrent (KDA) and 11 latent attention (MLA) |
| Experts | 384 a layer, top 8, Q3_K/Q4_K/Q5_K | 288 a layer, top 8, IQ3_XXS, plus shared experts and dense Q8 |
| Attention | relative position, sinks, gated output, short convolutions with state | sparse: an indexer picks 2,048 cells from pools of four tokens |
| Residual | one sum per six-layer block, 96 depth mixes a pass | hyper-connections |
| Context state | F16 global KV split by head, 8,192 bytes a token a card | 4-bit latent cache, 8,352 bytes a token mirrored on every card, plus recurrent state |
| Speculation | DSpark drafter, measured -23% | MTP head, about 2.15 tokens a step |
| Weights a card (the pack divided by four, not read on a card) | about 66 GiB | about 31 GiB |
| Vocabulary | 201,024 rows | 154,880 |

## 1. The number we steer by

Today each model is steered by a steady synthetic read. Inkling read 411.1 tok/s on it and 259.4 on real
coding requests; GLM read 42.8 tok/s a session steady and 18.8 on agent-shaped traffic.

| # | Measure | What it tells us | Have | Models |
| ---: | --- | --- | --- | --- |
| 1 | Replay of recorded agent sessions as the one benchmark: real prompt and reply sizes, tool turns, pauses, real depth | The rate people get. Every other number is judged by whether it moves this one | partly (`replay_*.py`, `coding_probe.py`; GLM's `agents.py` is not in the repo) | both |
| 2 | Where a session's wall clock goes: reading prompts, decoding, waiting behind another session's prompt, checkpoints, restores | Whether decode is even the right target. Inkling's prefill was 14 of the workload's 15 minutes | partly | both |
| 3 | Share of traffic spent in steady state, and how long a server takes to reach it | GLM passes rose 24.8 to 45 tok/s over six passes while shapes were captured; only 43% of a read's time was in full steps | partly (GLM, by hand) | both |
| 4 | Census of step shapes: rows a step, sequences a step, prompt rows mixed in, and time by shape | Which shapes deserve a fast path. One mixed step ran up to 19 forward passes | partly (GLM, by hand) | both |
| 5 | Latency a session feels: time to first token, time between tokens at p50 and p99, time to next turn | Stalls that an average rate hides | partly | both |
| 6 | Output check beside every speed number: KLD and top-token agreement against a reference, coding probe pass rate | The price of every change that alters output (KV format, wire format, quant mix) | partly (Inkling's parity capture is pending) | both |
| 7 | Noise floor: the same configuration run twice, on the same box and on another | How large a difference has to be before we believe it. GLM's steady read stopped at 45 tok/s on one host where 47.0 to 49.5 were predicted from another | no | both |

## 2. Where a step's time goes

| # | Measure | What it tells us | Have | Models |
| ---: | --- | --- | --- | --- |
| 8 | A step ledger the engine writes itself, cheap enough to leave on: host phases (build, allocate, inputs, compute, sample, per-slot server work, draft feed) and per card busy and idle by cause | The whole step with no profiler. Today this needs an Nsight window and a reader script | partly (`LLAMA_TIME_DECODE`, `LLAMA_DECODE_TRACE`, `summary.py`) | both |
| 9 | Kernel families per card per step: ms, launches, bytes read | Which kernel to work on. One reader for both models instead of two | partly (`decode_report.py` for GLM, `summary.py` for Inkling) | both |
| 10 | The critical path across cards: which card finishes last before each reduction, and the slack on the others | Work on a card that is not last buys nothing. Inkling's gather time includes waiting for the slowest card | no | both |
| 11 | Launches and gaps: kernels a step, gaps by size, waits between graph segments, host event waits | Launch overhead. A GLM decode is 3,458 kernels; Inkling spent 1,806 ms in event waits in a six-second window | partly | both |
| 12 | Graph lifecycle: builds, allocations, captures, evictions and their reason, reuse refused and why, hit rate by shape | The cost of every shape change. This is where the pool-padding stall and the kept-graph evictions were | partly (`GGML_GRAPH_TRACE`) | both |
| 13 | Host threads: time on core a thread, which one is pinned at 100%, lock waits, page faults, NUMA placement | GLM's 150 ms step was one host thread; a checkpoint was 63 ms of page faults | no | both |
| 14 | Rare steps, by histogram and cause instead of a mean | A GLM pool rebuild is one 316 ms step; a mean spread it as 2.5 ms a step | partly (by hand) | both |
| 15 | The same ledger for reading prompts | Prefill has no trace for Inkling | no | both |

## 3. Where each card's resources go

| # | Measure | What it tells us | Have | Models |
| ---: | --- | --- | --- | --- |
| 16 | Memory by owner on each card, over time and at its peak: weights, KV pool, sliding caches, recurrent state, compute buffers, kept graphs, masks, vision tower, draft, free | What a larger pool or more kept graphs takes from. GLM with 128 kept graphs filled 97,107 of 97,887 MiB | partly (`GGML_CUDA_GRAPH_MEM_STATS`, free memory after load) | both |
| 17 | Bytes read a step per kernel against the card's 1,597 GB/s, and how many times the same bytes are read | How close a kernel is to the memory limit. GLM's experts ran at about 70% of nominal; Inkling re-reads 49.4 GB a step of shared cells | partly (by arithmetic) | both |
| 18 | Hardware counters per kernel at the serving shape: L2 hit rate, DRAM throughput, active warps a SM, tensor-core use, register and shared memory use | What limits a kernel: memory, SMs, or the launch shape. This is the answer to SMs, warps and L2 | no | both |
| 19 | Each kernel alone at the serving shape and real routing, against its own limit | A kernel's headroom without a server (`test-backend-ops perf`) | partly | both |
| 20 | Clocks, power and temperature a card across a long run | Whether a card is throttled. Max-Q and Server cards differ by 12% in memory rate | no | both |
| 21 | The box: host bridges, which pairs of cards have peer access, NUMA node a card, and each route's rate under load | Which reduction route is right on this host. A route picked by timing one store ran three times slower | partly (`test-backend-meta-allreduce-copy --bench`) | both |
| 22 | Reductions: how many a step, bytes each, wire format, and the time of each phase | GLM has about 90 a decode at 91 us each; Inkling's gathers are 4.1 ms a step a card | partly | both |
| 23 | Balance between cards: busy time, memory, expert load | Inkling's expert rows split 3/2/2/2, which costs 3.74 ms on the critical path; its first card has 700 MiB less free | partly | both |

## 4. What the model does at run time

These decide which kernel shapes are worth building. They are properties of the model on our traffic, so
they need no four-card box.

| # | Measure | What it tells us | Have | Models |
| ---: | --- | --- | --- | --- |
| 24 | Expert routing: distinct experts a step a layer, overlap between sessions and between a token and its drafts, repeats from one step to the next, hot experts | Expert bytes a step and whether a cache or placement by heat would pay. GLM touches 177.5 of 288 at 69 rows; Inkling about 147 of 384 at 23 | partly (`GGML_CUDA_MMQ_ROUTE_LOG`, `GGML_CUDA_DP_EXPERT_STATS`) | both |
| 25 | Routing weights: how small the smallest of the eight are | Whether skipping the weakest experts is cheap in output | no | both |
| 26 | Attention reach: share of cells sessions hold in common, cells a row reads, where the attention mass sits by distance, how often a head attends to nothing (the sink) | The size of a shared-prefix kernel's win and whether far cells can be read coarser | partly | Inkling |
| 27 | Indexer behaviour: how many of the 2,048 selected cells change from one step to the next and between a token and its drafts, and what triggers a pool rebuild | Whether selection can be updated instead of redone | no | GLM |
| 28 | Stored values: range and spread of K, V, latent cache and recurrent state | Which storage format each can take (`kv_stats.py`) | partly (Inkling) | both |
| 29 | Drafts: acceptance by position, by session and by kind of text, cost of a drafted row against a plain one, the break-even depth | Whether speculation pays at 23 sessions and at what depth | partly (`rate.py`, the spec-bound report) | both |
| 30 | Sampling: candidates a row really needs, how often a request's settings force rows back to the host | A repeat penalty took every GLM slot off the card's top-k and cost 22.9 ms an iteration against 2.3 | partly | both |
| 31 | Output error per tensor class against its bytes | Where bits can be taken out of the pack to buy pool or kept graphs | no | both |
| 32 | Depth-mix weights and gate values: sources and gates that are near zero | Reads that could be skipped in Inkling's 96 mixes and gated outputs | no | Inkling |
| 33 | Recurrent state traffic and rollback cost under drafts | The state is read once and written once per drafted row | partly (by model) | GLM |

## 5. What the traffic looks like

| # | Measure | What it tells us | Have | Models |
| ---: | --- | --- | --- | --- |
| 34 | Real sessions: prompt and reply lengths, gaps between turns, how many sessions decode at once, depth over a session's life | The load to tune for. 23 sessions decoding at once may be rare | no | both |
| 35 | Shared prefixes between sessions and how often a prompt is already cached | The value of prefix copy and of the host cache | no | both |
| 36 | Slot churn: saves, restores, evictions, their time and size, snapshots that failed and forced a re-read | The cost of 23 slots serving more than 23 conversations | partly | both |
| 37 | Interference: a reply's rate while another session's prompt is read | GLM replies fell to 5.7 tok/s; the prompt-step settings trade this against time to first token | partly | both |
| 38 | Request parameters clients send | Which settings knock a request off a fast path | no | both |

## 6. Fast paths that silently decline

Most of the large losses found so far were a step falling off its intended path with no error: a drafted
step of 92 rows left three Inkling fast paths that took exactly 23, ten layers a step declined the fused
router, a quarter of GLM's subgraphs launched kernel by kernel.

| # | Measure | What it tells us | Have | Models |
| ---: | --- | --- | --- | --- |
| 39 | A count at every dispatch decision with the reason the faster path declined, and the share of step time on the intended path | Every silent fallback, in one table | partly (`GGML_CUDA_ROUTE_STATS`) | both |
| 40 | Each kernel's admission conditions against the shapes traffic produces | The shapes with no fast path, weighted by time | no | both |
| 41 | What a run really launched with: every variable read, its source, and the difference from the profile | Settings nobody chose. `--keep-graphs` sets six variables and the Inkling profile sets none of them | partly (`GGML_TUNABLES_DUMP`) | both |

## 7. How far from possible

| # | Measure | What it tells us | Have | Models |
| ---: | --- | --- | --- | --- |
| 42 | One living bound per model and phase, fed by measured bytes, with the gap per line | Where the headroom is. GLM's decode has one; Inkling has one for prefill only | partly | both |
| 43 | Sensitivity by injected delay: slow one kernel family or one route by a known amount and read the step | What a speedup there would be worth, before writing the kernel. A step is not a sum: kernels overlap and cards wait | no | both |
| 44 | Curves: rate against sessions, depth, draft depth and number of cards | The slope names the limit. 5.5 ms a session in a straight line was host work | partly (GLM) | both |
| 45 | The same models on other engines | An outside mark. TensorFold reports 70 tok/s at 16 concurrent for GLM | partly | both |

## 8. So that what we learn is kept

| # | Measure | What it tells us | Have | Models |
| ---: | --- | --- | --- | --- |
| 46 | One record a run for both models: build, box, configuration, workload, ledgers, output hash | A history that can be asked questions. The box-side readers (`idle-census.py`, `passcount.py`, `evalwhy.py` and others) are not in the repo | partly (MLflow, Inkling search only) | both |
| 47 | What the local fixture predicts against four cards | Which questions need a rented box and which do not | no | both |
| 48 | The same ledger on every commit, on the fixture | A regression found the day it lands | no | both |
| 49 | A long soak: memory growth, kept-graph population, rate and temperature over hours | What only appears after a day of serving | no | both |
| 50 | Behaviour under memory pressure: refused captures, forced saves, short-of-memory warnings | The edge of the budget before a session hits it | partly | both |

## What to build first

Thirteen of the fifty carry most of the rest, in this order:

1. **Replay of real sessions (1)**, with the noise floor (7). Without it every other number is judged against the wrong load.
2. **The engine's own step ledger (8)** with graph lifecycle (12) and memory by owner (16). One format, both models, on in every run.
3. **The fallback count (39)** and the launch dump (41). The cheapest way to find the next large loss.
4. **The critical path across cards (10)**. It decides which of the other findings matter.
5. **Routing and attention statistics (24, 26, 27)**. They size the two largest kernels of each model and need no rented box.
6. **Hardware counters per kernel (18)**. They turn SMs, warps and L2 from guesses into readings.
7. **Injected delay (43)**. It prices a kernel before anyone writes it.
