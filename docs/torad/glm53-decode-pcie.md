# GLM-5.3 Flash decode with PCIe costs

## Goal and corrected answer

Locate the PCIe transfers in four-card DP decode, put their costs at the dependency that needs their result, and update the 400k-context IQ3_XXS/Q8 throughput estimate. Distinguish current code from possible optimizations and assumptions from measurements.

**At 40 active sessions, an optimized DP4 design gives about 98-119 generated tokens/s per session under the reference assumptions below.** All drafts are accepted, rated DRAM bandwidth is fully used, and active attention data is GPU-resident. This is a conditional model result, not a measured rate or a universal hardware ceiling.

The earlier **106-131** figure was a **DRAM-only** estimate. PCIe was discussed but not included in its rate equation. The earlier **129-167** shared-weight-layout figure is not corrected here: that hypothetical architecture needs its own communication graph, and DP4's communication bill cannot be applied to it.

| Forty active sessions, each at 400k | Modeled tokens/s/session |
| --- | ---: |
| Previous DP4 DRAM-only baseline | 106-131 |
| Optimized transfers, 63 GB/s links, zero phase latency | 101-123 |
| Optimized transfers, reference PCIe assumptions | **98-119** |
| Same, plus proposed overlap with shared-expert work | 101-122 |
| Optimized transfers, contended-fabric sensitivity | 80-94 |
| Optimized transfers, host KV, identical selected rows reused within verification | 85-100 |
| Optimized transfers, host KV, selected rows fetched separately for each query | **69-79** |
| Current transfer protocol, otherwise optimized resident design | 77-90 |
| Current transfer protocol and full catch-up, resident KV | 74-86 |
| Current transfer protocol and full catch-up, per-query host KV | 55-62 |

Every row still uses optimistic weight/compute assumptions. The last rows are not forecasts of today's server: graph churn, kernel inefficiency, load imbalance and other costs remain outside the model. Host-per-query is a no-reuse payload scenario, not a measurement of physical PCIe bytes. Host/L2 caching can reduce it; inefficient mapped reads can increase its time.

The ranges compare independent expert choices for the three verification rows with identical choices within a sequence. Different sessions still route independently. They are not confidence intervals.

## Transfer locations in the code

Source basis: core GLM/meta/CUDA/speculative files at `d54ea817b`, unchanged in the inspected `002e02e92` checkout. No C++ inference code was changed for this estimate.

| Location | Transfer and dependency | Treatment |
| --- | --- | --- |
| `src/models/glm5next.cpp:612-638` | Owner-local activation -> gather on all expert ranks -> routed experts -> reduction -> owner result | Two sequential communication boundaries per MoE layer |
| `ggml/src/ggml-backend-meta.cpp:57-73` | Scatter is a reshape/layout after reduction | **No third PCIe collective** |
| `ggml/src/ggml-backend-meta.cpp:3583-3638` | Each local subgraph is followed by its gather/reduction before the dependent subgraph | Cannot hide all communication behind the whole target pass |
| `src/models/glm5next.cpp:641-655` | Shared expert reads local input, independent of remote routed results | Possible overlap with that layer's communication only; not claimed implemented |
| `ggml/src/ggml-cuda/allreduce.cu:421-493,1626-1699` | Current numeric reduction publishes to mapped host staging, waits for peers, then reads their values in its middle size band | Host write and read legs priced separately |
| `ggml/src/ggml-cuda/fattn.cu:1187-1221` | GPU-selected indices -> mapped host latent rows -> GPU scratch -> attention | Required host-to-GPU dependency when latent KV is host-resident |
| `src/llama-graph.cpp:120-140`; `common/speculative.cpp:2015-2161` | Proposal hidden input is CPU-provided; a nonterminal proposal reads hidden output to CPU for the next proposal | Two proposal uploads and one proposal readback for MTP2 |
| `src/llama-graph.cpp:173-195` | Device-H catch-up still uploads CPU carry rows to their owners | One row/sequence above the device-H threshold; otherwise all shifted verification rows |
| `src/llama-context.cpp:3279-3306`; `common/speculative.cpp:1950-1993` | Verification hidden rows are read back before acceptance/retained-state update | Three rows/sequence in this full-depth scenario, even with device-H |
| `src/llama-graph.cpp:95-107,203-215`; `src/llama-context.cpp:2119-2149` | Tokens, positions, metadata, sampled results and scheduling cross the host boundary | Explicit small-transfer budget and host-boundary latency, not silently zero |
| `src/llama-context.cpp:2435-2447` | Raw target logits may return to CPU when a backend sampler is unavailable | Optional `--raw-logits`; 619,520 bytes/output row |

The 45 target layers contain 42 MoE layers and three leading dense FFNs. Thus target verification has **84 collective boundaries**. Two proposal calls add four. Today's full catch-up adds two more: **90 total**. The hypothetical cache-writing-only catch-up used in the original best-world model removes that final MoE and attention, leaving **88**.

The current catch-up executes before acceptance, including rejected verification rows (`common/speculative.cpp:1875-1993`). Full head attention and FFN are built at `src/models/glm5next.cpp:875-915`. The full-catch-up estimate includes those weights and context reads but no shortlist projection for its zero requested output rows.

### What does not cross PCIe every layer

- Resident latent KV, indexer keys, KDA state, shared experts and owner-partitioned selection data stay on the sequence owner.
- Routed expert weights are already on their owning GPU. The repeated exchanges carry **activations and partial outputs**, not the entire model's weights.
- Owner-local backend sampling does not require the old whole-vocabulary peer gather.
- A 400k history is not copied every token. At long-context sparse attention, selected latent cells number at most 2,051 per query, or 1,115,744 Q8 bytes per attention layer. The pooled-key scan over the history stays GPU-local.
- Model loading and derived-weight uploads are startup costs.
- Prompt snapshot save/restore, slot migration, full checkpoint restore and replay are admission/recovery costs. They matter for mixed workloads but are not steady-decode events in these tables.
- Prefill concurrent with decode is excluded, not treated as free.
- Host KV also writes each new 544-byte latent row back to host memory; these writes are included.

## PCIe assumptions, not measurements

Bandwidths are decimal GB/s. A peer fabric byte counts a send once, not both its send and receive endpoints.

| Parameter | Encoding-limit case | Reference scenario | Contended sensitivity |
| --- | ---: | ---: | ---: |
| Peer bandwidth per GPU, per direction | 63 | 54 | 27 |
| Shared peer fabric capacity, sum of sends | 252 | 216 | 54 |
| Host bandwidth per GPU, per direction | 63 | 40 | 20 |
| Shared host path capacity | 252 | 128 | 64 |
| Added latency per complete collective | 0 us | 5 us | 20 us |
| Added latency per host staging/handoff event | 0 us | 5 us | 10 us |
| CPU/graph boundary budget per graph | 0 us | 20 us | 50 us |

The 54 GB/s reference is near historical **pairwise bulk** measurements, not a measurement of simultaneous four-way traffic. The 216 GB/s fabric assumption requires all four endpoints to sustain that rate together. NODE topology and a shared NUMA node do not prove it. The host rates and all nonzero latency values above are sensitivity inputs.

Small inputs/control/results receive a budget of **64 KiB per GPU, per direction, per graph**. Four graph calls occur in a full MTP2 cycle. It is deliberately visible and adjustable with `--control-bytes`; it is not an exact byte count of every sampler and metadata configuration. Actual per-copy API overhead, transfer granularity and graph retention can exceed the host-boundary budget.

Host latency is a budget per complete batched event, not a measured cost per CUDA API call. In particular, the current device-H carry path uploads rows individually in a sequence loop; its total overhead can exceed the modeled batch budget. The fabric parameter is an effective shared service rate, not a reconstructed switch/root-complex topology. It must respect the real machine's bottleneck links and contention.

The zero-latency case is only a payload floor. PCIe protocol headers, read completions, polling, chunk scheduling and launch delays are not physically zero. The reference uses effective bandwidth and additive latency to represent them parametrically, not to assert that they have been measured.

## Bytes and schedule

Let `R=N(d+1)`, `B=R*4096*4`, and `W=B/2` for BF16 reduction. Let `r_i` be rows owned by rank i. Current defaults are BF16 numeric wire transport, peer LL through 128 KiB, and copy-engine reduction starting at 1 MiB.

### Current protocol

- Gather sends a zero-padded full B-byte buffer from each rank to three peers: **12B sent across the box**.
- For small LL messages, data-plus-flag packets double that payload.
- Mid-sized numeric reduction publishes W bytes/rank to host, then every rank reads three other W-byte buffers: **4W GPU-to-host + 12W host-to-GPU**.
- Direct numeric routes send 12W across peers, with LL packet overhead when applicable.
- At MTP2, target BF16 reduction enters the host-staged band at **six active sessions** and remains there through the launcher's 40 resident-slot limit. At 40 sessions, W is **960 KiB**, just below the default 1 MiB copy-engine threshold.
- P2P capability alone does not remove this software route. Peer probing/allocation failures can cause further fallbacks, not modeled here.

### Proposed owner-slice protocol

- Gather: rank i sends `3*r_i*16384` bytes; receives `(R-r_i)*16384`.
- Destination-only reduction: rank i sends `(R-r_i)*8192`; its owned results receive `3*r_i*8192`.
- Box totals: **3B gather + 3W reduction**, not a redundant result on every GPU.
- Balanced ranks send `1.125B` per rank for the pair, with the same receive volume.
- A single active owner cannot borrow four links: its gather still sends three copies from that one endpoint. The estimator uses actual consecutive-ID owner counts, not `N/4` fractional owners.

These protocols are proposals, not configuration flags. The BF16 reduction already entails a precision choice; changing reduction order and transport must still be numerically qualified.

At 40 sessions, target plus two proposals sends **94.37 MB/GPU/cycle** with optimized communication. Current transfer logic, holding cache-only catch-up fixed, instead sends **251.66 MB/GPU of peer gather data**, plus **41.94 MB/GPU to host staging** and **125.83 MB/GPU back from host staging**. The current host-staging endpoints total **671.09 MB across the box** before any host KV.

### Timing equation

For a peer event:

`t_payload = max(max_i(send_i)/b_peer, max_i(receive_i)/b_peer, sum_i(send_i)/b_fabric)`.

Send and receive may proceed full duplex; they are not added. Shared-fabric capacity can still be the limiting term.

For a host publish/consume event:

`t_payload = max(max_i(send_i)/b_host, sum_i(send_i)/b_host_shared) + max(max_i(receive_i)/b_host, sum_i(receive_i)/b_host_shared)`.

The two host legs are dependent. This serialized schedule does not assume chunk pipelining or overlapping mapped-host reads with earlier writes. A different schedule needs its own model.

Each event also pays its specified phase latency. Each MoE is:

`local producer -> gather -> routed experts -> reduction -> local join -> next layer`.

The target's 42 repetitions remain 42 dependency pairs. They are not combined into one large transfer for latency accounting. Each host-KV stage is placed after index selection and before its own attention. Proposal steps and verification/catch-up remain dependent.

Without speculative overlap:

`t_cycle = t_DRAM + sum(t_transfer_event + t_event_latency) + t_host_boundaries`.

With proposed shared-expert overlap, subtract at most, **per layer**:

`min(t_gather + t_reduce, shared_weight_bytes / DRAM_bandwidth)`.

The shared branch may be scheduled/chunked across those two communication windows; no credit is taken for hiding transfers behind unrelated layers, routed-expert work or the full target pass. This idealized overlap also assumes communication does not materially slow the shared branch through GPU memory/SM contention. Shared weights are 26,738,688 bytes per layer; all 44 target/proposal MoE calls offer only **0.737 ms** at rated bandwidth. They cannot hide an arbitrarily long PCIe delay.

Every extra 10 us per collective adds **0.88 ms per MTP2 cycle** before that bounded overlap. An average kernel duration from an old trace cannot simply replace this latency: it can already include payload time, rank waiting and upstream compute delays.

### Forty-session reference breakdown

| Cost | Independent verification routing | Same experts within each sequence |
| --- | ---: | ---: |
| Original DRAM work | 28.220 ms | 22.929 ms |
| Peer MoE payload | 1.748 ms | 1.748 ms |
| 88 collective latency budgets | 0.440 ms | 0.440 ms |
| Control/host boundaries and hidden handoffs | 0.157 ms | 0.157 ms |
| Cycle, no shared overlap | **30.565 ms** | **25.274 ms** |
| Three delivered tokens / cycle | **98.15 tok/s** | **118.70 tok/s** |

This corresponds to about **3,926-4,748 generated tokens/s across the box**. It does not include prompt processing or HTTP/client delay.

## Host KV changes the result more than optimized expert payload

With cache-only catch-up, target verification plus two proposals issues 1,400 attention queries across the box per MTP2 cycle. At the maximum valid sparse selection:

- No cross-query reuse: `1400*2051*544 = 1.562 GB` from host per cycle.
- Identical selections reused among each sequence's three verification rows: `520*2051*544 = 580.19 MB`.
- Current full catch-up adds another 120 attention queries, raising the no-reuse total to **1.696 GB**.

The four cards share the assumed host path. At 128 GB/s aggregate, 1.562 GB alone costs **12.20 ms**, not 3.05 ms obtained by dividing that aggregate demand by four again. Additional GPU scratch writes and attention reads of duplicate staging are included in the host scenarios' DRAM term.

The current staging kernel is query-indexed and has no explicit persistent hot-row cache. Identical selections, deduplicated staging and a persistent exact working set could save transfers, but their hit rates are not known. The resident and reuse cases therefore cannot be substituted for host-per-query without a capacity/hit-rate argument.

## Concurrency and acceptance

Reference PCIe assumptions, optimized transfers, resident KV and perfect MTP2:

| Active sessions | Tokens/s/session |
| ---: | ---: |
| 1 | 330-369 |
| 4 | 230-308 |
| 8 | 172-247 |
| 12 | 145-210 |
| 24 | 112-152 |
| 36 | 101-125 |
| 40 | 98-119 |

Small batches make the ideal kernel/reuse assumptions especially loose. Owner or expert imbalance, finite arithmetic throughput and individual transfers can dominate.

At 40 sessions:

- No MTP: about **48 tok/s/session**.
- One of two drafts accepted on average: **65-79 tok/s/session**.
- That acceptance and 70% of rated DRAM bandwidth: **47-57 tok/s/session**.

The last line is a sensitivity, not a calibrated expectation. Optimization does not guarantee draft acceptance. Fixed full-depth work remains charged even when drafts are rejected; EOS, variable depth and replay need a trace-driven workload model.

## Cross-check distinctions

- Count **42 MoE layers** for target collectives, but **11 MLA/DSA layers** for target host-KV staging. Reusing the MoE layer count for attention greatly overstates host-KV demand.
- Working P2P does not imply every collective avoids host memory. Large integer gathers and mid-sized numeric reductions use different branches in the current CUDA implementation.
- MTP2 retains all three verification hidden rows. Its proposal loop reads the hidden result only when another proposal will consume it; the terminal hidden getter is skipped.
- Endpoint byte counts establish resource floors, not measured concurrent-fabric rates. Synchronization, mapped-read efficiency and contention remain explicit assumptions.

## Reproduction and limits

- Estimator: `scripts/glm53/estimate_decode_pcie.py`.
- Tests: `scripts/glm53/test_estimate_decode_pcie.py`.
- All 41 preset results, parameters and event ledgers: [glm53-decode-pcie.json](glm53-decode-pcie.json).
- Exact tensor-derived inputs and the original DRAM calculation: [glm53-decode-roofline.md](glm53-decode-roofline.md), [glm53-decode-roofline.json](glm53-decode-roofline.json).
- Full configuration/cache map: [glm53-four-gpu-simulation.md](glm53-four-gpu-simulation.md).

```sh
python3 scripts/glm53/estimate_decode_pcie.py --sessions 40
python3 scripts/glm53/estimate_decode_pcie.py --sessions 40 --kv host-per-query
python3 scripts/glm53/estimate_decode_pcie.py --protocol current --catchup full --kv host-per-query
python3 scripts/glm53/estimate_decode_pcie.py --peer-gbps 27 --fabric-gbps 54 --collective-us 20
python3 scripts/glm53/estimate_decode_pcie.py --sweep > /tmp/glm53-pcie-results.json
python3 -m unittest discover -s scripts/glm53 -p 'test_estimate_decode_pcie.py'
```

Tests were run failing for the missing behavior before implementation. All **16 PCIe-estimator tests passed**, followed by **121/121 tests in the complete GLM Python tooling suite**. They cover owner placement, transfer conservation, no phantom scatter, current threshold bands, LL flags, full duplex versus shared-fabric limits, 84/88/90 boundaries, host-KV reuse, catch-up costs, bounded overlap, acceptance, generated sensitivity cases, prior DRAM arithmetic and the real CLI. Full-suite log: `/tmp/glm53-pcie-suite.log`. These validate the accounting and CLI, not GPU performance.

No four-GPU benchmark, numerical qualification, capacity qualification or paid hardware run was performed. The remaining hardware gaps are **simultaneous collective bandwidth/latency**, **mapped-host sparse-read bandwidth under four-GPU contention**, and **actual routing/acceptance at 400k**. Finite compute/dequantization, temporary traffic, graph rebuilds, rollback/replay and serving overhead can lower every optimistic result above.
