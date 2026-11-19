# Serving the original abliterated Inkling 2 Small on four PCIe Blackwells

The selected `tensor-4` profile uses the corrected SM120a runtime, ordinary fast prefill, sorted expert rows at 23, batched session saves, bucketed long-prefill graphs and host-sliced BF16 tensor-parallel reductions. The CUDA library SHA-256 is `505015a0193984c1403811185790733d2958a8a59647a59c7cd636c09fdd7656`. The release contains the exact four-GPU process binaries and effective source.

## Configuration

The first abliterated revision is `e4853745aa9cb4587e8959d3c80b92e1bef99a4a`. Its imatrix Q3_K_M pack is 283,538,490,336 bytes and averages 3.82 bits per weight. The F16 image projector is included. The shared F16 KV pool has 3,014,656 cells, 23 slots and a 294,912-token ceiling per slot. A 262,144-token input leaves 32,768 tokens for the reply and reasoning. The server has no independent output-token cap. Inactive sessions use up to 256 GiB of host RAM. OpenAI Chat Completions, OpenAI Responses and Anthropic Messages use the same model, template and session pool.

## Kernel and cache assessment

The campaign identified genuine hot kernels. The priorities changed under the work workload. PCIe transport and saving old sessions caused delays visible to all users. The 23-client coding replay uses distinct prompts, a pinned request manifest and 512 generated tokens per stream. It measures the common interval while all 23 clients are decoding. Single-stream rates, first-token latency and profiled six-token prompts are reported separately.

| Four-card workload | Earlier placement build | Batched saves, ordinary expert order | Combined FP32 prefill and sorted experts | Selected fast prefill and sorted experts |
| --- | ---: | ---: | ---: | ---: |
| Simultaneous coding decode, aggregate tok/s | 266.10 | 271.41 | 273.68 | 278.44 |
| Cold 262,144-token reader alongside 22 peers, seconds to first token | 160.82 | 148.07 | 191.37 | 152.12 |
| Output from the 22 peers while the reader prefills, aggregate tok/s | 35.70 | 56.31 | 43.57 | 54.96 |
| Longest interval between peer tokens, seconds | 12.899 | 0.874 | 1.233 | 0.908 |
| Uncapped 262,144-token coding task, prompt tok/s | 2,162 | 2,249 | 1,668 | 2,174 |

The original NCCL control measured 242.5 coding tok/s. Relative to the matching host-transport control, computing only the owner’s output rows cut PCIe receive bytes per delivered token by 45.7%. The decoded throughput difference for that individual change was about 0.5% in that test. Results for the early floating-point A/B that wrote past the allocated scratch are excluded.

The batched saver moved recurring reads off the server thread while preserving the bytes. Saving a 266,267-token session occupied that thread for 12.844 s on the placement build. Saving a 265,775-token session on the batched build took 19.87 ms on the thread and 5.119 s through background completion. The longest observed peer interruption fell from 12.899 s to 0.874 s with 22 active peers and a cold 262,144-token reader. Separate Nsight captures of 23 occupied slots showed 43,776 device-to-host transfers before the first GPU kernel in the control versus zero with the batched saver. Across the entire profiled request interval, both transferred the same 2,341,436,800 bytes to host memory.

The optional FP32 kernel exposed a four-rank allocator defect. A rank needed 7,177,088 bytes for a 128-query case where the unsplit graph reserved 3,145,728. The corrected binary passed local allocation guards, then repeated identical-placement cold prefills exactly on the four cards at 2, 2,048 and 8,192 tokens. Changing the occupied layout still changed some scores. The grouped FP32 setting delivered 273.68 coding tok/s and passed 46 of 46 uncapped coding checks, but reduced that task’s input throughput to 1,668 tok/s and increased the mixed reader’s first-token time to 191.37 s. An experimental later local dispatch is outside the tested release.

## Fidelity and shared context

The blocking file and RAM paths used the same saved 262,144-token sequence, restored into the same destination and continued with the same 18-token suffix. All 32 candidate IDs and their log probabilities matched exactly on both the control and corrected combined runtime. They reused all 262,144 prefix tokens. Independent cold and resident prefills in different layouts gave different distributions. With the combined FP32 setting, the restored and resident continuations shared 31 of 32 candidates, agreed on the greedy token and differed by up to 0.228 in common log probability. On ordinary attention the corresponding earlier result shared 27 candidates and differed by up to 1.937. This deployment cannot promise identical output across independent placements.

Twenty-three simultaneous 262,144-token requests were verified on this pack with 196,608 common tokens and 65,536 private tokens per client. All 22 followers reused the full common prefix. That required 1,703,936 evaluated tokens for 6,029,312 nominal inputs. The batched-save run repeated the handoff with 65,536 shared and 4,096 private tokens. The corrected combined run repeated it with 32,768 shared and 128 private tokens, reusing the full prefix on all 22 followers in both runs. Twenty-three entirely private 262,144-token prompts exceed the pool.

The tested effort controls are `none`, `minimal`, `low`, `medium`, `high`, `xhigh` and `max`. The template defaults to `high` and maps the last two names to the same highest native value. The API qualification includes rendered effort controls, invalid settings, tool round trips and reasoning delivery. The supplied projector was exercised through Chat Completions and Messages.

## Final grouped run

The 23 coding streams each generated 512 tokens. During their 41.94 seconds of simultaneous decoding, clients received 11,677 tokens at 278.44 tok/s, about 12.11 tok/s each. Counting the whole request window, including starts and finishes, they received 256.76 tok/s with a 3.05 s median time to first token. The single-stream short decode ran at 64.82 tok/s. The concurrent throughput was 14.8% above the original NCCL control on the same pack and workload. Each table entry is one completed run.

With ordinary prefill and sorted expert rows, the uncapped coding request evaluated 262,144 fresh tokens at 2,173.86 tok/s, generated 5,196 tokens at 55.65 tok/s and stopped naturally. Its patch passed all 46 executable checks. In the mixed workload both retrieval identifiers were correct, and all 22 peer streams remained active throughout the reader. Their 8,382 overlapping token-delivery intervals had a p95 of 702 ms and a maximum of 908 ms. Saving the displaced 267,339-token session held the server thread for 21.15 ms and completed in 5.147 s. The earlier batched configuration with the original expert order had the shortest observed cold-reader latency. The selected configuration had the highest sustained 23-client coding throughput.

All three APIs returned the expected answer at both sampled effort settings. They passed 24 effort-routing cases including defaults and rejected all eight invalid requests. The model emitted reasoning at both sampled settings, including native level zero. High-effort reasoning reached the client on all three APIs. Complete tool exchanges and still-image answers passed through Chat Completions and Messages. The final grouped run did not repeat the separate file/RAM identity and shared-prefix workloads described above.

## Rental and instrumentation

Rig reported four GPU windows of 4.27, 0.46, 0.55 and 0.19 hours. Their estimated compute charges were $30.49, $3.28, $3.93 and $1.36, totaling $39.06. The last three windows totaled $8.57. Storage between windows and provider-side charge adjustments are separate. The first window included model conversion. The subsequent windows reused the verified model. Local development, telemetry replay and release packaging ran with the four GPUs stopped.

The full monitoring archive replayed locally in Grafana, Prometheus and Loki. It contains 18,137,971 original Prometheus samples across the qualification windows. A single scrape recorded all four cards, the replay matched the selected 23-client result and both provisioned dashboards returned real log entries. GPU, host, cgroup and server counters and the underlying log databases are preserved.
