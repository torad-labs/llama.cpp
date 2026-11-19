# Inkling 2 Small on four PCIe Blackwell GPUs

The final grouped qualification selected the deployment settings and measured 23 coding streams, a full-length request, peer latency and session saves. [Final four-card measurements](13-batched-serving-results.md)

## What the optimization campaign predicted

The five kernel items cover the model's relative-position attention, residual mixing, short convolutions, routed experts and KV precision. The local code and hardware checks support keeping the fused attention and convolution implementations. The production decision also depends on the number of experts real requests touch, pauses for a new 262,144-token prompt, and whether a coding session resumes correctly after eviction. Those measures can select different settings from a single-stream decode test. See the [campaign assessment](10-serving-campaign-review.md) and [Chinese-language sources and TensorFold comparison](11-pcie-blackwell-throughput.md).

## Physical host and model provenance

Rig rented Vast instance 61306114 in California at USD 7.14074 per hour. It has four 96 GB NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs, 188 SMs per GPU, and PCIe 5.0 x16 links. All 12 directed peer pairs support reads and writes. The median single-link copy-engine read measured 53.47 GB/s within a CPU socket and 38.75 GB/s across sockets. Host-to-device transfers measured about 57 GB/s per card. These figures describe individual transfers. Simultaneous traffic can contend for the host links.

The abliterated source is `benthecarman/Inkling-2-Small-abliterated` revision `e4853745aa9cb4587e8959d3c80b92e1bef99a4a`. The converter read all 24 verified shards. The Q3_K_M recipe uses Bartowski's original-model importance matrix from revision `75970601c5da75212b2ffb914344e1b7dae2008d`. Twelve of its 36,864 expert projection observations were empty and used unit weights. The independently converted F16 vision projector has SHA-256 `06e2b478ca029ab3aea9ed041032079401e168274c248d6eba39081c375ff031`. [Weight source](https://huggingface.co/benthecarman/Inkling-2-Small-abliterated/tree/e4853745aa9cb4587e8959d3c80b92e1bef99a4a)

OrcaRouter's public catalog contained 41 models on November 18 and none of their names, declared parents or architecture metadata identified Inkling. OpenUncensored published a separate Inkling 2 Small derivative with MXFP8 experts and BF16 weights, totaling 623 GB. Exa search and the linked primary file listings yielded no public 3-bit GGUF for either derivative. The first release remains the selected deployment target. [OrcaRouter catalog](https://huggingface.co/orcarouter) · [OpenUncensored model and weights](https://huggingface.co/OpenUncensored/Inkling-2-Small-Uncensored)

The finished model contains 1,092 tensors in 283,538,490,336 bytes (264.07 GiB), SHA-256 `afb72c67dad0c1c34424de631a9290595b6d1db55e346c001fc38db3efc59e2d`. The GGUF records the derivative revision and original importance-matrix digest. All 48 fused expert gate/up tensors are Q3_K. The 48 expert down projections comprise 45 Q4_K and three Q5_K tensors.

On November 17 at 10:25 p.m. CT, the publisher replaced all 24 model shards on `main` with revision `8574652a49e1f39286b25cb943e57e08ec931b17`. The first-release tests in this report use the pinned `e4853745` weights. Both revisions publish Safetensors, and the Hugging Face index returned no declared quantization of this derivative on November 18. Bartowski's existing Q3_K_M declares the original Thinking Machines checkpoint as its parent. [Publisher history](https://huggingface.co/benthecarman/Inkling-2-Small-abliterated/commits/main) · [Stock quantization](https://huggingface.co/bartowski/Inkling-2-Small-GGUF)

## Isolated hardware measurements

- The SM120 global-attention comparisons passed 54 numerical checks. For 23 readers with 196,608 shared and 65,536 private tokens each (1,703,936 physical cells), grouped query heads took 4.67 ms per measured operator and the dense path took 10.26 ms. With one 262,144-token reader and 22 short readers, the measurements were 0.834 and 2.089 ms. Dense attention remained useful when mask density was high.
- Q3_K gate/up at 23 rows changed its winner with routing concentration. At 61 distinct experts the grouped kernel took 100.0 microseconds and the row kernel 144.0. At 150 distinct experts they took 273.5 and 219.1 microseconds. The published base control touched a median of 59 experts per layer on synthetic decode and 102 on 23 different coding requests.
- The host-staged BF16 reduction now has an optional path that reads each GPU's 1,152 owned convolution channels per token. Four physical cards matched the full-reduction results in the operator checks, including fallbacks for extra readers, different geometry and Q8. The local trace on four logical ranks measured a drop from 211,968 to 52,992 bytes per host-to-device peer receive. The device-to-host writes are unchanged. At 96 eligible sites this would save 174.66 MiB per 23-row decode step.
- The native NCCL gather recognizes equal contiguous rank ranges, transmits each owned range and reconstructs the full tensor in place. Inkling can lay out its convolution and logits shards with channels outermost for that operation. On four physical GPUs, three positive and fallback graph shapes reproduced every expected value on six successive inputs. CUDA graph replay worked, and the test observed eight optimized collectives. The model fixture also passed prefill, decode, separate prompt chunks, sequence copy, serialized state and rollback on four physical cards. [NCCL in-place collective semantics](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/inplace.html)

## Published base-model controls

These early measurements used the public Bartowski Q3_K_M checkpoint while the requested derivative was being quantized. Its expert down-projection types also differ. Short balanced decode used 23 streams with 128 prompt and 48 generated tokens each. The 4,096-token prompt throughput is measured separately.

| PCIe route | Shared F16 pool | Solo decode (tok/s) | 23-stream aggregate decode (tok/s) | 4,096-token prefill (tok/s) |
| --- | ---: | ---: | ---: | ---: |
| Host-staged | 2,359,296 | 53.2 | 209.0 | 641 |
| Host-staged sliced gather | 2,359,296 | 51.3 | 291.4 | 1,239 |
| Engine / NCCL | 3,014,656 | 58.6 | 247.9 | 1,709 |

On the engine route with a 2,359,296-token pool, one cold 262,144-token retrieval took 394.66 s to first token while 22 clients continued generating. Those peers delivered a median 0.659 tok/s each and a pooled p95 delivery gap of 1.810 s. Both retrieval identifiers were correct. In a separate run a 262,144-token coding request yielded a patch passing all 46 executable checks. The request omitted an output cap and stopped naturally after 4,020 generated tokens.

The base RAM test saved 9,115,904,564 bytes and reused the full 262,144-token prefix in another slot. The restored and independently resident prefixes chose the same greedy next token but differed on their top-32 distribution (30 common IDs, maximum common log-probability difference 2.686). Two independent cold prefills had already diverged in 88 of 96 expert sites in the first 2,048-token chunk. The small-model same-slot blocking-file control reproduced the asynchronous RAM result exactly. The physical derivative run is needed to distinguish transfer fidelity from GPU accumulation order after reallocation.

A delayed-arrival base-model run put 23 independent requests in 23 live slots. Each request asked for 262,144 tokens, with 196,608 common tokens and 65,536 private tokens. Twenty of the 22 followers reused the full common prefix. Two followers read it cold, taking the total from the ideal 1,703,936 evaluated tokens to 2,097,152. All 23 streams delivered output after 1,924 seconds. The first donor had already advanced beyond the fork and the default periodic checkpoint interval was zero. The candidate enables active-prefix reuse in a slot whose former prompt was safely saved, and the delayed-arrival test supplies periodic rollback checkpoints.

The base throughput probes recorded routed-expert histograms. That observer runs an extra GPU kernel at each routed MMQ site. Derivative throughput probes disable it and record their PCIe traffic separately. The full 23-way coding comparison measures delivered tokens during the interval in which all streams were decoding.

## Abliterated first-release serving results

All three routes loaded all 49 offloadable layers from the verified Q3_K_M. With 23 simultaneous streams, 128 prompt and 128 generated tokens each, a 2,359,296-token unified F16 pool and 256 GiB of RAM cache, the client measured:

| Transfer route | Aggregate decode while all 23 were generating (tok/s) | Median time to first token (s) |
| --- | ---: | ---: |
| Host staging with sliced gathers | 376.2 | 2.63 |
| Host staging | 354.4 | 2.97 |
| Engine / NCCL | 340.5 | 2.29 |

Slot prompts and greedy settings were identical. The host-sliced path first diverged from the host-generated token sequence on one of 23 prompts, and NCCL on two. These short-prompt rates measure batched generation. The work-sized pool and coding traffic are separate tests.

The host-sliced route also loaded the GLM-sized 3,014,656-token F16 pool, a 294,912-token per-session window and 256 GiB of async host snapshot capacity. Free VRAM after load on ranks 0 through 3 was 1,691, 2,421, 2,475 and 2,421 MiB. Both output-limit properties were `-1`. A single stream delivered 68.8 tok/s, a 4,096-token prompt processed at 1,953 tok/s, and the 23-stream common interval measured 411.1 tok/s on synthetic prompts and 259.4 tok/s on 23 different coding requests. The two workloads use the same model, server and pool.

The long reader started cold and retrieved both identifiers from 262,144 input tokens. Its first token arrived after 162.29 s. All 22 peers continued generating, at a median 1.61 tok/s each. Their pooled p95 inter-delivery gap was 741 ms and the maximum gap was 8.46 s. Reader TTFT and peer gaps are client-side measurements.

The uncapped high-effort coding request read 262,144 tokens at 1,898 tok/s and stopped naturally after 3,109 output tokens at 58.8 tok/s. Its fenced diff contained a bare `@@` marker and failed Git's unified-diff format. Replacing only that header with the line numbers and lengths from its own hunk let the untouched implementation pass 46/46 isolated tests. The original request remains a strict failure.

The service passed all six effort-disabled/high answer checks across OpenAI Chat Completions, OpenAI Responses and Anthropic Messages, 24 effort-routing checks, eight invalid-setting rejections and both tool round trips. OpenAI Chat and Anthropic Messages each returned the expected image colors.

The unmodified fork revision `35a2ce34d6061aee4fec73f5124844bc1a7cdf0b` loaded the same weights and allocated both KV caches, then aborted in `ggml_backend_sched_split_graph` during graph reservation because a node had no assigned backend. It produced no comparable throughput under TP4. The optimization comparisons below use runnable control builds with the same weights and pool.

## PCIe transport on the same weights and work-sized pool

Each row uses the candidate runtime, the first-release GGUF and the same 23 coding requests. The client counts output tokens only in the interval all 23 clients were decoding. Traffic is the sum of the four PCIe endpoints, sampled once per second and divided by client output tokens/s. A PCIe transfer between GPUs can contribute at both endpoints. The coding runs had nine to ten complete one-second samples each.

| Transport | Solo decode (tok/s) | 4,096-token prefill (tok/s) | Synthetic 23 streams (tok/s) | Coding 23 streams (tok/s) | PCIe RX MB/output token | PCIe TX MB/output token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Engine / NCCL | 64.9 | 2,244 | 368.9 | 242.5 | 27.88 | 33.44 |
| Engine, native NCCL gather with transposed shards | 71.3 | 2,731 | 377.9 | 247.6 | 21.12 | 28.06 |
| Host, sliced gathers | 67.5 | 2,411 | 420.9 | 263.9 | 20.72 | 11.54 |
| Host, sliced gathers and BF16 owner rows | 68.9 | 2,400 | 431.6 | 265.1 | 11.24 | 10.03 |

The owner-row path reduced endpoint receive traffic per delivered coding token by 45.7% against the corresponding host control. The measured 0.5% coding-throughput difference is too small to distinguish from ordinary run variation here. On the NCCL route, native gathers reduced receive traffic per token by 24.3% and increased 4,096-token prefill throughput by 21.7% in this comparison.

Forcing the routed Q3_K/Q4_K/Q5_K matrix-vector operator through 23 rows on the host owner-row configuration delivered 265.9 coding tok/s and 418.9 synthetic tok/s. The earlier owner-row run with ordinary expert dispatch delivered 265.1 and 431.6, respectively. The coding difference is small. With the 23-row setting, the fully uncapped repeat of the identical 262,144-token coding prompt passed all 46 checks exactly as returned. The model stopped after 5,775 generated tokens at 59.5 tok/s. Prefill took 116.0 s. In the mixed reader run, TTFT was 141.23 s, both identifiers were correct, the 22 peers delivered 40.50 tok/s together and their p95 gap was 643 ms. These two trials do not establish a general coding accuracy rate.

## Shared 256 Ki-token sessions

With periodic checkpoints every 32,768 tokens, one request read 262,144 tokens and started generating. Then 22 independent clients arrived with the same 196,608-token prefix and distinct 65,536-token tails. The donor had already read its private tail. All 22 recovered the earlier common prefix, all 23 generated, and the client verified per-request counts on 23 distinct live slots. Nominal input was 6,029,312 tokens, cached work was 4,325,376, and evaluated prompt work was exactly 1,703,936 tokens. Elapsed time to first output on every stream was 1,239.24 s. A 23-way burst of large private tails still incurs their prefill cost. The pool is dynamic. It cannot simultaneously hold six million unrelated F16 tokens.

## RAM restore and numerical reproducibility

On a cold 262,144-token prefix, the async RAM handoff saved and restored 9,115,904,564 bytes in 96 copy ranges. The destination reused all 262,144 tokens and evaluated only the 18-token suffix. A blocking file snapshot taken from the same original prefill was restored into the **same destination slot** and continued with the identical suffix. All 32 returned candidate tokens and log probabilities matched the RAM path exactly, including the greedy token.

An independent prefill and resident continuation in a different layout had the same greedy token but only 26 of 32 candidates in common with RAM restore, with a maximum common log-probability difference of 1.0833. The independent prefixes already differed before continuation. The strict cross-layout reproducibility check failed. At two, 2,048 and 8,192 tokens, cold same-slot repeats, a different empty slot, and changing the unrelated neighbour's token content each had zero observed difference. Adding the neighbour changed the top probabilities at all three lengths. Those observations identify sensitivity to occupied KV layout while isolating the RAM transfer from that effect.

## A cold 256 Ki-token request alongside 22 generating peers

Each row is one client-side measurement on the same abliterated pack and 3,014,656-cell pool. The cold reader retrieved both identifiers in every run. The peer gap includes time waiting for a delivery across the reader's arrival. The maximum is a single observed gap.

| Transport and prefill scheduling | Reader time to first token | Aggregate peer tok/s | Median tok/s per peer | Peer p95 gap | Longest peer gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| Host owner rows, fixed 1,024-token steps | 141.23 s | 40.50 | 1.84 | 643 ms | 8.59 s |
| Host owner rows, 200 ms adaptive target | 414.92 s | 105.04 | 4.77 | 252 ms | 0.83 s |
| NCCL sliced gather, fixed 1,024-token steps | 145.33 s | 39.51 | 1.80 | 699 ms | 0.91 s |

The adaptive target trades cold-reader throughput for more frequent peer delivery. Set `--prompt-step-ms 200` when serving interactive peers during a long read is the priority. The largest gaps also include the cost of displacing the previous session. The fixed host run displaced 267,918 tokens and spent 8.485 s saving it. The adaptive and NCCL rows each displaced only 79 tokens and spent about 0.03 s saving it. Their maxima cannot isolate the scheduling effect.

## Long-prefill graph reuse

On the portable x86-64-v3 and SM120a release build, rounding the global attention span to buckets of 8,192 cells reduced same-size graph reallocations from 128 to 42 during one cold 262,144-token prefill. The server read the identical input in 113.59 s at 2,308 tok/s, compared with 118.00 s at 2,222 tok/s using the original spans. Both generated token IDs and every top-32 log probability at both positions matched exactly. This was a repeated-text throughput probe. The retrieval and executable coding probes above used their own prompts.

The `tensor-4` profile enables this change for single-reader F16 prefill above 32k. Use `--no-bucket-prefill-kv` to select the original spans. Keeping the one-bit attention mask also matters for capacity. Disabling it at the full pool made rank 0 request a 12,961 MiB compute buffer and startup failed.

## Full-model GPU profiles

Nsight Systems captured the complete model on all four physical GPUs. Every client sent a six-token prompt and received 16 tokens, for 368 output tokens across 23 slots. The host-sliced run took 2.365 s under the profiler. Native NCCL took 2.627 s. These captures include replacing 23 prior short prompts. They use different prompts and much shorter generation than the coding-throughput measurements.

On host-sliced transport, quantized routed expert matrix-vector kernels consumed 792 ms summed across four GPUs, 40.0% of summed kernel time. Their intervals covered 272 ms of client wall time. The native route executed 1,647 four-rank AllGather calls and 1,632 four-rank AllReduce calls. The collective kernels consumed 300 and 791 ms of summed GPU time, respectively.

Both routes issued 43,776 device-to-host copies and 44,064 stream synchronizations before their first GPU kernel. Saving the short histories took 661 ms on host-sliced and 721 ms on native. A separate workload exposed the cost of displacing a long history: saving a previous 266k- to 268k-token session accounted for 8.485 s of one 8.588 s peer gap and 12.844 s of one 12.899 s gap.

The first paid window ran for 4.27 h. Rig reported USD 30.49 at shutdown. Both CUDA profiles and 16,260,285 original Prometheus samples were copied off the machine. Local Grafana replay reproduced the four-card scrape, the 266.10 tok/s coding measurement and both Loki log queries. The retained disk preserves the original quantized model between paid compute windows.
