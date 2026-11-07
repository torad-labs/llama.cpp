# GLM-5.3 Flash DRAM-only decode baseline

> PCIe time was not included in the rate equation below. The corrected dependency-aware analysis is [GLM-5.3 Flash decode with PCIe costs](glm53-decode-pcie.md). Its reference 40-session result is **98-119 tok/s/session** with optimized DP4 transfers and resident KV, or **69-79** with per-query host KV, still under ideal compute and full MTP2 acceptance. The shared-weight redesign below has no communication model and is not a PCIe-aware estimate.

## Original goal and DRAM-only answer

Estimate the best-world steady decode rate for the usual BoldingBuilds IQ3_XXS/Q8 pack on four RTX PRO 6000 Blackwell Server Edition cards, with independent sessions each holding 400,000 context tokens. Separate a hardware-resource ceiling from a measured or promised serving rate.

For the current sequence-owned DP4 architecture, the DRAM-only model (excluding PCIe time) gives **about 106-131 generated tokens/s per session at 40 active sessions**, with MTP depth two and every draft accepted. That is about 4,250-5,230 tokens/s across the box. Without speculation, the same model gives about **51 tokens/s per session**.

A more ambitious architecture that also shares dense-weight traffic across the four GPUs, instead of reading a dense replica on each owner, raises the DRAM-only 40-session envelope to **129-167 tokens/s per session** with ideal MTP2. That is not a current launcher setting or an established implementation.

These are conditional rooflines, not universal maxima. Routing concentration, speculative depth, acceptance, cache reuse and the chosen parallel architecture change the ceiling. The ranges below are two expert-reuse scenarios, not statistical confidence intervals. Exceptionally correlated routing across sessions can exceed this model; imbalanced routing can fall below it.

## Inputs from the actual files

The three pinned GGUF shard headers were read with bounded HTTP Range requests. All **1,412 tensor records** were counted, and each shard's final tensor extent matched its published file size. No full-model download or GPU run was needed.

Pack: [BoldingBuilds/orcarouter_GLM-5.3-Flash-Uncensored-GGUF](https://huggingface.co/BoldingBuilds/orcarouter_GLM-5.3-Flash-Uncensored-GGUF/tree/1bc0afadd92f4ff9fc2b1281a95a7921b058d35d/IQ3_XXS), revision `1bc0afadd92f4ff9fc2b1281a95a7921b058d35d`.

| Quantity | Exact bytes | Decimal GB |
| --- | ---: | ---: |
| All tensor payloads, including draft | 134,296,001,912 | 134.296 |
| Target non-routed tensors inside blocks | 8,509,631,736 | 8.510 |
| Full target output matrix | 674,037,760 | 0.674 |
| Target output norm | 16,384 | 0.000016 |
| Target dense read set, including output | 9,183,685,880 | 9.184 |
| All 42 target routed-expert banks | 116,530,348,032 | 116.530 |
| Eight selected experts per target layer, one token | 3,236,954,112 | 3.237 |
| All routed experts in the single draft block | 7,700,742,144 | 7.701 |
| Draft non-routed tensors | 207,188,096 | 0.207 |
| Derived 98,304-row draft shortlist matrix | 427,819,008 | 0.428 |

The token embedding table is another 674,037,760 bytes, but inference looks up token rows rather than reading the entire table each step. Its full size is therefore excluded from the per-step dense read set. Small embedding reads, temporary activation traffic and some scalar work are omitted from this optimistic model.

The important correction is that this is not an 18B-active model with every active parameter stored at three bits. Target routed experts are IQ3_XXS; attention, shared experts, leading dense FFNs, output and the entire MTP block are Q8_0. Control/indexer tensors also include BF16 and F32. A target pass touches roughly **12.42 GB of weight data for one token**, before cache traffic, rather than the approximately 6.75 GB obtained by multiplying 18B by three bits.

The actual Bolding metadata also confirms the dimensions previously inferred from another export: 45 target blocks plus one NextN block, 34 KDA and 11 MLA blocks, hidden width 4,096, 64 attention/KDA heads, KDA width 128, 288 experts/top-8, and four-token pools with 2,048 selected token cells.

## Hardware basis

[NVIDIA's Server Edition specification](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/) lists:

- 96 GB GDDR7 per GPU.
- **1,597 GB/s memory bandwidth per GPU**, or 6.388 TB/s across four only when work/data placement can use all four.
- PCIe 5.0 x16; no NVLink, confirmed in the [NVIDIA product specification](https://dam-cdn.nvd.orangelogic.com/AssetLink/3km2720jiy76r06ctf8mxg21743p1058.pdf).
- Up to 600 W; the local qualification records used the Server Edition at that power limit.

The 1,792 GB/s figure belongs to the Workstation/Max-Q specifications, not this Server Edition. If the final cards truly provide that memory rate, bandwidth-dominated numbers increase by about 12.2%, before any power/clock/compute differences.

Do not use the advertised FP4 sparse PFLOPS as this model's throughput. IQ3_XXS is not native FP4, and selecting eight MoE experts is not 2:4 structured sparsity.

## DRAM-only per-session results (exclude PCIe time)

Every row is steady generation with the listed number of active, independent, equal-length 400k histories. Idle configured slots do not count as active sessions. Thinking/reasoning tokens count as generated tokens.

| Active sessions | Optimized DP4, no MTP | Optimized DP4, perfect MTP2 | Redesigned shared-weight execution, perfect MTP2 |
| ---: | ---: | ---: | ---: |
| 1 | 156 | 359-406 | 924-1,318 |
| 4 | 127 | 244-333 | 407-735 |
| 8 | 103 | 182-267 | 259-477 |
| 12 | 87 | 153-227 | 204-362 |
| 24 | 64 | 120-165 | 149-227 |
| 36 | 53 | 109-137 | 132-177 |
| 40 | 51 | 106-131 | 129-167 |

Units: generated tokens/s **per session**. Round values when using them in planning; the displayed precision is not a performance guarantee.

The redesigned column is an aggregate-bandwidth thought experiment that divides dense weights and sequence-state work evenly across GPUs. It would require distributed dense execution and a compatible non-replicated context/state layout. Merely changing `-sm dp` to `-sm tensor` does not establish that forty 400k histories fit or attain those rates. Particularly at one session, serial inter-GPU latency and small-matrix efficiency will make this column loose.

The lower MTP number assumes the three verified token rows select experts independently. The upper number assumes those three rows use the same experts within each sequence, while different sessions still route independently. Both assume all proposals are accepted and ideal within-batch cache reuse. Actual routing must be measured to choose between or outside these scenarios.

## Derivation

Let N be active sessions, d the fixed draft depth (0 or 2), and B = N(d+1) target verification rows. With eight of 288 experts selected per row, independent uniform routing touches this fraction of an expert bank:

`p(B) = 1 - (1 - 8/288)^B`.

This models reuse: loading one expert's weights can serve multiple rows in the batch. Treating every row as eight entirely new experts overcounts traffic; assuming all rows share the same experts undercounts diverse sessions.

For DP4, the busiest owner has `ceil(N/4)` sequences and reads one full dense target read set. With perfectly balanced expert work, its target weight traffic is:

`9.183685880 GB + 116.530348032 GB * p(B) / 4`.

For the within-sequence-reuse scenario, replace p(B) with p(N) in the target term. For the redesigned shared-weight case, divide the dense term by four too and distribute context work evenly.

The main 400k-context terms are:

- Pooled index keys: `100000 pools * 136 bytes * 11 layers = 149.6 MB` per sequence per scan. The model reuses these keys ideally across the verification queries; arithmetic is still performed for every query.
- Selected latent rows: `2051 * 544 bytes * 11 layers = 12.273184 MB` per sequence, with ideal reuse across the verified rows.
- Current KDA matrix plus convolution state: `47.906816 MB` per sequence across 34 layers.
- Recurrent traffic: one current-state read plus d+1 resulting state writes, or `(d+2) * 47.906816 MB`, retaining rollback state rather than pretending speculative state is free.

This cache accounting is optimistic and approximate. It excludes score/top-k scratch, most intermediate activations and allocator traffic. Hardware cache residency and actual selection overlap can alter DRAM traffic.

MTP adds d sequential proposal passes. Each proposal reads its dense/shortlist set, the selected Q8 experts and its attention/indexer data. Catch-up is assumed optimized down to the embedding/hidden merge and cache-writing projections, rather than the current full head evaluation. Its projection read set is approximately 40.03 MB per dense replica. That pruning is a best-world design assumption, not work completed by this calculation.

For each dependent phase, divide modeled bytes on the busiest GPU by 1,597 GB/s, then sum the phase floors. Delivered tokens per sequence per full-acceptance cycle are d+1.

### Forty-session example

For the spread-routing MTP2 case:

- Target work: about 40.86 GB/GPU/cycle.
- Two draft passes: about 4.17 GB/GPU/cycle.
- Optimized catch-up: about 0.04 GB/GPU/cycle.
- Total: about 45.07 GB/GPU/cycle.
- Ideal cycle time: `45.07 / 1597 = 0.02822 seconds`.
- Three accepted output tokens: `3 / 0.02822 = 106.3 tokens/s/session`.

With within-sequence expert reuse, the ideal cycle is about 22.93 ms and the rate is 130.8 tokens/s/session.

## Compute and PCIe checks

The target matrix, pool-scoring and selected-attention work totals approximately 45.45 GFLOP per evaluated row, excluding smaller elementwise operations and IQ3 decoding instructions. A 120-row verification is approximately 5.45 TFLOP across the box. At the modeled 22.9-28.2 ms cycle this is about 193-238 TFLOP/s across four GPUs, before the smaller draft work. It does not immediately contradict even the four cards' advertised 480 TFLOP/s aggregate FP32 peak, but it does not prove the actual mixed-quant kernels can execute efficiently. This calculation never assumes native FP4 execution of IQ3_XXS.

With ideal owner-slice gathering and destination-only reduction, a 40-session MTP2 cycle needs roughly 94.37 MB of MoE peer payload sent per GPU, with a similar receive volume. At the PCIe 5.0 x16 encoding-limit rate of roughly 63 GB/s, that is a 1.50 ms transmission floor. Protocol overhead, switch/root-complex contention, ordering, latency and topology can increase it. The recorded 53.7-55.3 GB/s pairwise bulk measurements do not establish concurrent four-way bandwidth.

This shows that optimized bulk peer byte volume need not dominate a 23-28 ms cycle. The current host-staging band, redundant full-buffer gathering, synchronization and graph churn can still dominate current execution. The roofline gives those costs no penalty; it does not assume their removal has already happened.

## Why perfect optimization is not perfect MTP acceptance

Software can remove avoidable work. It cannot make every proposed model token correct without changing the prediction problem. If a full-depth MTP2 cycle emits two tokens on average instead of three, throughput is two thirds of the perfect-acceptance column at the same modeled cycle cost.

At 40 DP4 sessions:

| Assumptions | Calculated tokens/s/session |
| --- | ---: |
| Both drafts accepted; 100% rated bandwidth | 106-131 |
| One of two drafts accepted on average; 100% rated bandwidth | 71-87 |
| One of two drafts accepted on average; 70% rated bandwidth | 50-61 |

The 70% row is a sensitivity input, not a measured efficiency prediction. Other omitted costs would reduce it further. Longer speculation or different routing statistics can change the ceiling; this calculation keeps the intended MTP2 design.

## What these numbers do not include

- Cold prefill or another request prefilling alongside generation.
- Queueing, prompt snapshot restore, HTTP/client latency, or final-answer-only token counting.
- A verified capacity layout for forty fully occupied 400k histories. All active sparse data is assumed GPU-resident or served from an equally cheap exact working set, which requires the proposed memory improvements.
- Dequantization overhead, launch/synchronization latency, graph rebuilds, bandwidth inefficiency, thermal/power throttling, or expert/rank imbalance.
- A proof that the proposed peer protocols, pooled cache layout, catch-up pruning or redesigned dense execution preserve the required numerical behavior.
- A universal maximum over arbitrary routing, prompt sharing or speculative algorithms. Identical sessions and perfect-oracle drafts are not the workload being modeled.

## Evidence and reproduction

- Machine-readable inputs and all scenarios: [glm53-decode-roofline.json](glm53-decode-roofline.json).
- Exact raw header captures, complete tensor manifest and calculation scripts: `.Codex/analysis/glm53-ceiling-20261104.RgdyZj/`.
- Recompute without network or model loading:

```sh
python3 -I .Codex/analysis/glm53-ceiling-20261104.RgdyZj/calculation/calculate.py \
  .Codex/analysis/glm53-ceiling-20261104.RgdyZj/headers/manifest.json \
  /tmp/glm53-roofline-recomputed.json
```

The three full GGUF files were not rehashed here; only their pinned headers and tensor extents were inspected. Full-file SHA256 provenance remains in the serving recipe and earlier qualification record.

### Three-pass check

1. Weight accounting: rejected the pure-three-bit active-parameter shortcut; used all 1,412 actual tensor headers and explicit draft/dense separation.
2. Hardware/placement: rejected pooling four bandwidth figures while leaving dense execution on one owner; used the Server Edition's 1,597 GB/s and reported a separate redesigned-layout case.
3. Workload: rejected perfect MTP acceptance as a consequence of software optimization; exposed expert-reuse and acceptance assumptions, plus compute/PCIe and capacity limits.

This baseline also lacks simultaneous-fabric bandwidth, collective latency and host-KV staging costs. Those are now explicit sensitivity parameters in the [PCIe-aware companion](glm53-decode-pcie.md), alongside routing and acceptance. No new GPU run or money was used for either calculation.
