# GLM-5.3-Flash with data-parallel attention

Experimental DP from [PR #110](https://github.com/torad-labs/llama.cpp/pull/110), combined with main for the November 2 baseline and extended by the November 3 completion work below. Four RTX PRO 6000 96 GB cards and 40 long-context slots are the target. The selected model is the BoldingBuilds uncensored pack. Capacity and numerical qualification remain separate requirements; a successful boot or component test does not qualify concurrent 400K sessions.

Latest four-card test: the November 5 candidate `136589f16` passes all 84 focused expert-kernel cases under Compute Sanitizer, but still crashes during full-model four-session decode. With 36 reserved private slots, resident Q8 caches/state and adaptive MTP maximum depth 2, one uninstrumented active session delivered a cached continuation at 71.203 tokens/s after a 512-token prompt. Nsight and CUDA-GDB reproduce the four-session failure; the debugger and matching binary instructions point to CPU stack exhaustion in recursive Meta split-state analysis when the active batch changes. Twelve/36 active sessions and populated 400K capacity remain unqualified. The earlier `e2435726a` attempt measured 73.216 tokens/s for one active session and rejected 40 reserved slots for breaching the 4,096 MiB free-memory requirement. See the [GPU results, failure evidence and TensorFold comparison](glm53-dp-kernel-audit-20261105.md). The temporary rentals were stopped promptly and destroyed after collection; production pins and placed profiles are unchanged.

The November 4 host-KV measurements below used a different configuration: 40 private slots and MTP2 on runtime `82ad56cb8`. One active session produced 128-token replies at 45.3 tokens/s with a 512-token prompt, 35.2 at 4,096 and 31.7 at 24,576. The shorter concurrent probes returned about 28 tokens/s per session at two sessions, 12-16 at four and 6 at eight. These are historical host-KV results, not the resident-cache baseline.

The earlier 72-slot capacity probe allocated private 442,368-cell slots and passed short concurrent decode at a median 0.587 tokens/s per request. That historical result is not the current throughput baseline. Long-context capacity testing remains deferred until concurrent throughput is usable.

## Follow-ups after PR #116

[PR #116](https://github.com/torad-labs/llama.cpp/pull/116) merged as `de18d20cb`. It includes optional CUDA graph memory diagnostics, activity-based graph aging, prompt-prefix/cache-accounting cleanup and the DP backend-sampling candidate. The following source changes address separate items from the throughput plan:

- `633d76232`: both sampling entry points obtain one borrowed output-row view after one context synchronization. Existing individual getters retain their synchronization contract; grammar resampling reuses the same view.
- `42ff59502`: an idle slot remembers an unchanged `too_large` or `no_space` RAM-cache refusal. Slot mutation, restore, cache insertion/removal and budget changes invalidate the refusal; transient failures are retried. This affects admission work, not a measured steady-decode rate.
- `01eaef311`: a GLM MTP drafter registers generation/catch-up shapes up to its configured depth and microbatch size. The existing three extra graph slots can retain 40-row generation and 120-row MTP2 catch-up shapes. Each keeps its own scheduler and inputs, and the sampler-binding guard remains. Reservation measures the larger shapes, including bound nextn storage; a shape exceeding that per-backend allowance falls back to the main scheduler. Arbitrary prefill shapes and scheduler input boundaries are unchanged.
- `65fc151e1`: DP index capture reads the selection matrix once, then distributes its rows on the host. Hidden-state reads coalesce physically adjacent rows on each owner and enqueue them before one context completion. Meta async reads now support partial intervals of an outermost split, including empty owners. All requested indices are validated before queuing writes into a caller's buffer.

The transfer change preserves host bytes, rollback and checkpoint behavior. For a three-row verification segment on one owner it requests one transfer instead of three; a selection matrix uses one Meta read instead of one read per sequence. A debugger count in the CPU DP fixture observes 32 hidden-state reads, each returning five rows (2,560 bytes) through two async requests and one scheduler synchronization. The sampling-row fixture observes exactly one scheduler synchronization in each of its 12 bundled reads. These count API calls, not GPU time. The change does not retain all selections or carry rows on the GPUs. That larger change requires owned snapshot storage and graph bindings that survive buffer reuse.

CPU validation passes: both sampling suites; all 13 server cache/restore cases and the cache-accounting unit target; all 15 MTP groups; logical two/four-rank MTP with default and Q8 caches; logical two/four-rank GLM including forks, restore and nextn row ordering; the k-pool suite; and the full Meta split suite. The alternating-shape regression fails at 33 rows before the retention change and passes at 32, 33 and 40 sequences afterward, with logits identical to graph reuse disabled. The partial-read regression aborts before the Meta change and passes afterward. The consumed-restore regression fails before the cache revision fix and passes afterward: freeing a RAM snapshot invalidates another idle slot's cached `no_space` refusal. These use small synthetic models or stories260K, not the full model pack. The pre-existing Meta compute-buffer expectation warning remains.

The CUDA server and focused test targets compile for SM120a with `GGML_NATIVE=OFF`. GPU-only test cases were not exercised, and no GPUs were started or used for these checks. This is compilation coverage, not a qualified remote runtime package. Physical CUDA replay, active memory and short 1/4/12/40-stream throughput must be measured before any speed or capacity claim. The existing full-model DP-versus-tensor KL discrepancy remains unresolved. Full device retention, constant-weight preparation and incremental CUDA graph occupancy accounting remain separate work; cache-budget tuning and persistent hot-KV staging remain dependent on profiling.

## November 3 completion candidate

Runtime source `8fae0c4009cf0b9f49cb62e521091c6a647b18be` adds three paths:

- Partitioned experts estimate work from the global expert count. Small batches can use the existing vector and fused gate/up/clamp kernels, with non-local expert IDs masked before dispatch. Fusion retains the existing shape, stride and memory-alias checks.
- Large gathers push the original words into peer inboxes and read them locally. The parity protocol supports captured replay; hosts without the required peer mappings retain the staged fallback. The initialization line names the selected large-gather route.
- Resident k-pool metadata uses a cache-owned bank split by sequence owner. Graphs select the active sequences' local tables and upload journal patches only to the owning device. This changes metadata storage and transfer, not attention or recurrent arithmetic.

The CPU tests cover logical two/four-rank GLM execution, tensor-mode regressions, and resident metadata with 40/41 slots, including rollback, restore and direct sparse indices. The portable CUDA build targets SM120a. On four RTX PRO 6000 Max-Q cards (Vast box 58990727, driver 615.71.09), the corrected artifact passes four/two-card collective and captured-replay checks, 124 expert cases, 80 resident-metadata cases, synthetic GLM/MTP checks, and all 12 fused expert cases under Compute Sanitizer with zero memory errors. No measured serving profile is established by these component checks.

The second physical run passed four/two-card direct collectives and captured gather replay. An artifact timestamp mistake omitted four added fused expert cases and the final equivalent metadata-upload refactor: its 120 expert cases and eight fused sanitizer cases must not be described as the final 124/12-case coverage. The corrected artifact is `c284cc188078bcfd5f52983298f6af5c0a3e24bdb4f85b084bf10d9de1ae8941`.

At 2K, the corrected artifact matches the prior `cd47dc9` reference within the native reference format's precision in both layouts: tensor KL 0.000000, DP KL -0.000001, and 100% top-token agreement for each. The small negative value is the tool's quantized-reference floor, not a negative mathematical divergence. Four-card DP versus tensor remains 0.013567. The earlier artifact's controls give 0.013251 for changing only tensor microbatch 512 to 1024, and 0.010915 for four-card versus two-card DP. Cross-layout results do not meet the requested absolute 1e-6 bar. A batch-shape control explains why final KL alone cannot identify a faulty operator; it does not waive that bar.

Bounded captures of the first 512-row evaluation locate the first observed arithmetic differences. Two-card tensor and DP have identical input, hyper-connection and attention-normalization tensors. Their first QKV projection differs by at most 2.38418579e-6 after aligning the layouts: tensor stores `[Q0,K0,V0,Q1,K1,V1]`, while DP stores `[Q0,Q1,K0,K1,V0,V1]`. Comparing the raw buffers without that permutation would report a packing difference as a numerical error. Two-card and four-card DP are bit-identical through the first three dense blocks and the first MoE input. The first MoE block's output differs by at most 2.38418579e-7. These captures support arithmetic-order sensitivity; they do not establish that every final-logit difference is harmless or satisfy the absolute KL limit. All 219 capture files/manifests were verified by SHA-256 against the remote originals.

Expert counters work with `-lv 4`. The four-card diagnostic prompt recorded 44,759 / 41,283 / 40,569 / 48,109 hits, totaling 174,720 for 520 rows, eight selected experts and 42 routed layers. These instrumented prompt counts establish work on every rank, not balance or throughput under 36/40 concurrent sessions.

### Capacity attempt and restore repair

The same artifact ran the selected BoldingBuilds pack with its projector, non-unified Q8 caches and recurrent state, MTP depth 2, batch 2048 and microbatch/prompt-step 512. Forty slots loaded, but card 0 had only 1,544 MiB free (95,798 MiB used), below the required 4,096 MiB reserve. The harness rejected that configuration before active work; this was not an observed OOM.

Thirty-six slots loaded with 6,832 MiB free on card 0. All 36 independent 30,720-token cold prompts completed with 64 output tokens each. This phase took 1,617.575 s and peaked at 92,468 MiB used on card 0, leaving 4,874 MiB. Native prefill rates were 19.74-22.31 tokens/s per request; decode was 0.617-1.697 tokens/s while other requests were still prefilling. These are not steady 36-way decode rates. A separate real tool call retained 4,316 cached tokens and processed 14 new tokens on the result turn.

Four separate 409,088-token seeds completed at 2,111-2,159 prefill tokens/s each. Restoring the first seed into slot 4 retained its prefix, but the extension aborted in `ggml_backend_meta_set_tensor_async`: the outer-range byte sum did not equal the view's visible byte length. No 36-way 400K decode or second-turn result was obtained. Results were collected, stop was requested 4.27 s after compute ended, and the provider confirmed the box stopped at 19:10:57 UTC on November 3.

Commit `dfc9ce76cdf9cf46fd916ff5ed66b0c7f630c2b5` fixes that byte calculation. A contiguous view containing one outer plane can retain an unused stride from its larger parent; its visible bytes, not that stride, determine the transfer interval. The existing stream-view test reproduces the same assertion on CPU before the fix. F32 and Q8 cases pass afterward, including a non-first local stream, asynchronous transfers, partial memset and checks that other streams remain unchanged. The full CPU meta-split and k-pool suites pass. The portable CUDA build passes; artifact SHA-256 is `cdc9cded2b5a0c8407dd8e12fab39ccb5095a096e30c65773efdd2aeee175b98`. Later host-KV runs below verify the repair on physical GPUs, including 54 additional full-prefix restores.

The slow cold workload also exposes a scheduling constraint: a balanced 512-row microbatch over 36 sequences supplies 14 tokens per sequence. KDA's chunked path requires at least 128, and its decay-precompute path at least 32. This identifies unavailable fast paths from source, not their share of measured runtime. No scheduling policy or KDA arithmetic was changed. Concurrent capacity, throughput and the strict numerical requirement remain unqualified; no four-card serving profile is promoted.

## Convergence baseline validation

Runtime source `f089c9cb4bc285a11e510e1c4336b95aca449b08` passed CPU and CUDA builds, 35 focused CPU CTests, the three tensor-mode regressions on CPU/CUDA, and 108 expert-partition cases on CPU and each local RTX 5080 / RTX 5070 Ti. The portable `engine-f089c9c` package passed Rig's Ubuntu 22.04 driver-only installation/decode gate with the small stories260K-f32 test model. Later baseline commits change documentation only.

The full CUDA GLM architecture command still reproduces the known unsplit-prefix discrepancy, max_abs `2.96160579e-06`. The Meta teardown warning and four-card host gather bottleneck remain. These component checks do not qualify the full model, forty-slot capacity, or two concurrent 400K sessions with the six-slot 1M tensor configuration.

## What `-sm dp` does

Each sequence belongs to device `seq_id % device_count`. Its attention cache, recurrent state, k-pool and MTP hidden-state storage stay on that device. Dense weights are replicated. Routed experts are divided by expert index; a gather before each expert block and a reduction after it combine their work. A single stream can be slower than tensor parallelism because its dense path runs on one card.

The first implementation requires GLM5NEXT, equal device shares, all layers on GPU, flash attention and non-unified KV. CPU layer offload, unequal shares, unified KV and a hybrid tensor/data split are unsupported. Use `-fit off` and specify capacity explicitly.

## Build

Use main, a CUDA toolkit supporting the target cards, and the same pinned build on every comparison:

```sh
git fetch origin main
git switch main
git merge --ff-only origin/main
cmake -S . -B build-dp -DGGML_CUDA=ON -DGGML_CUDA_NCCL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-dp --parallel 6 --target llama-server llama-perplexity test-llama-archs test-speculative-mtp test-backend-ops test-backend-meta-split test-backend-meta-allreduce-copy
build-dp/bin/llama-server --list-devices
```

Record `git rev-parse HEAD`, the GGUF hashes, driver/toolkit versions and `nvidia-smi topo -m` with each result. `GGML_CUDA_ALLREDUCE=internal` selects the fork's collective path. Check the initialization log for the actual path; the host-staged fallback is functional but does not establish PCIe peer-to-peer performance.

## Model pack

The current qualification uses [BoldingBuilds/orcarouter_GLM-5.3-Flash-Uncensored-GGUF](https://huggingface.co/BoldingBuilds/orcarouter_GLM-5.3-Flash-Uncensored-GGUF/tree/1bc0afadd92f4ff9fc2b1281a95a7921b058d35d/IQ3_XXS), revision `1bc0afadd92f4ff9fc2b1281a95a7921b058d35d`: three IQ3_XXS shards totaling 134,305,522,848 bytes. Use the matching 1,128,047,744-byte F16 projector when testing the served configuration. Rig's head file pins all four SHA-256 hashes. This is the same pack as the two-card baseline.

The original brief's [Unsloth UD-IQ3_XXS pack](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF/tree/a38483c8cd5df544f53d70fb281afe97369d5ab6/UD-IQ3_XXS) at revision `a38483c8cd5df544f53d70fb281afe97369d5ab6` has four shards, totaling about 120.37 GB. Its first shard contains metadata only. Download all four into the same directory. The fork accepts its `glm5-next` architecture name and metadata prefix as well as the existing `glm5next` spelling. Its weight budget does not establish capacity for the selected larger pack.

The Unsloth pack label does not describe every tensor: its target experts use IQ2_S, IQ3_S and IQ4_XS; its MTP experts use Q2_K/Q3_K. These formats are supported by the CUDA DP expert path. They differ from the Q8_0 MTP weights assumed in the original capacity estimate and from Rig's selected BoldingBuilds uncensored IQ3_XXS pack. Keep each pack's quality, acceptance and performance evidence separate.

The Unsloth tensor headers imply approximately 25.58 GiB of target experts, 0.61 GiB of MTP experts and 7.33 GiB of replicated weights per card under four-way DP, before buffers, caches and runtime allocations. These are weight-byte calculations, not measured VRAM peaks. Local metadata-only loading validates all 1,412 tensor names/shapes including MTP; it does not validate the weight payloads or model outputs.

## Candidate serving command

Select the launcher's `dp-4` recipe explicitly with `--profile dp-4`. It starts one `llama-server` process with `CUDA_VISIBLE_DEVICES=0,1,2,3` and `-sm dp -ts 1,1,1,1`: all four cards form one group, with experts partitioned across that group. It does not create two independent pairs or use a 2x2 tensor/data split. `dp-4` is a launcher profile name; the engine's split-mode flag is `-sm dp`. The startup log identifies the group as `Meta(CUDA0,CUDA1,CUDA2,CUDA3)`.

Run it on the remote GPU host from a checkout containing the completion candidate and singleton-restore repair above. Omitting `--profile` selects the separate `tensor-2` recipe. Resident `dp-4` defaults to 36 slots and accepts 1 through 40. Each slot receives 442,368 private cells on its sequence owner, with dense weights mirrored and routed experts partitioned. The launcher sets four devices, internal collectives, graph capture, Q8 target/indexer/recurrent caches, batch 2048, microbatch 512, prompt step 512, and disables allocation fitting (`-fit off`, required for the capacity math below) together with `--backend-sampling`. With MTP enabled it also sets Q8 draft caches and draft microbatch 512. It disables idle-slot caching (`--no-cache-idle-slots`): each slot's state stays in VRAM, and a slot taken by another conversation still saves its state to the RAM cache first, so prefix restoration keeps working. Copying every idle slot into RAM whenever a request starts held the others back: at 36 sessions the next turn's first token took 5.5-14.5 s with it and 1.3-1.6 s without it (rented four-card box, November 6).

```sh
# Inspect without starting a model or touching a GPU.
python3 scripts/glm53/serve.py --profile dp-4 --slots 4 \
  --server build-dp/bin/llama-server --model /path/to/first-shard.gguf \
  --mmproj /path/to/matching-projector.gguf --dry-run

# Resident capacity candidate with MTP retained.
python3 scripts/glm53/serve.py --profile dp-4 --slots 36 --mtp-depth 2 \
  --server build-dp/bin/llama-server --model /path/to/first-shard.gguf \
  --mmproj /path/to/matching-projector.gguf
```

The second command is unqualified for 36 concurrent 400K sessions. The team configuration keeps MTP enabled. The previous 40-slot MTP-off proposal is withdrawn; its projected savings were not measured. The launcher retains its explicit draft-off option for independent experiments, but neither the capacity calculator nor the host-KV recipe relies on it.

The DP launcher disables idle graph expiry with `GGML_CUDA_GRAPH_IDLE_MS=0` while retaining the finite 32,768-node budget. It does not disable node-based eviction or establish memory headroom under concurrent work. `--trace-decode` enables verbosity 4 alongside graph and phase diagnostics; omit it for uninstrumented throughput measurements.

The launcher removes inherited engine overrides for reproducibility. Use the raw command below for controlled wire-format and expert-counter diagnostics instead of expecting inherited environment variables to override the launcher.

Start with four slots for qualification. Increase `GLM_SLOTS` only after checking loaded and active memory peaks. Forty slots give `-c 17694720`; each slot has 442,368 cells. Advertise 409,600 tokens in the client/provider configuration and reserve 32,768 for the reply.

```sh
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export GGML_CUDA_ALLREDUCE=internal
export GGML_CUDA_GRAPH_MAX=0
export LLAMA_MTP_DEVICE_H=1
export LLAMA_TOK_EMBD_GPU=1

# Start quality comparisons with F32 collective wires.
export GGML_CUDA_AR_BF16_THRESHOLD=0
# Optional qualification counter; prints per-layer/per-device expert hits at clean shutdown.
export GGML_CUDA_DP_EXPERT_STATS=1

GLM_MODEL=/path/to/GLM-5.3-Flash-Uncensored-IQ3_XXS-00001-of-00003.gguf
GLM_SLOTS=4
GLM_CTX=$((GLM_SLOTS * 442368))

build-dp/bin/llama-server -m "$GLM_MODEL" \
  -ngl all -fit off -sm dp -ts 1,1,1,1 \
  -np "$GLM_SLOTS" -c "$GLM_CTX" -ub 512 -b 2048 --prompt-step 512 \
  -ctk q8_0 -ctv q8_0 -ctki q8_0 -cts q8_0 \
  --spec-type draft-mtp --spec-draft-n-max 2 -ubd 512 -ctkd q8_0 -ctvd q8_0 \
  -fa on --jinja --chat-template-file models/templates/GLM-5.3-Flash.jinja \
  -lv 4 --metrics --host 127.0.0.1 --port 8080
```

Set `GLM_MODEL` to the actual first shard; the filename above is a placeholder. Keep the remaining shards beside it. The pack must contain the MTP block. Add the pack's matching `--mmproj` when serving image inputs. Do not add `--kv-unified`.

Draft Q8 and recurrent Q8 are capacity candidates, not a measured quality recommendation. Qualify them against F16 draft cache and F32 recurrent state; the existing two-card Rig recipe uses F16 draft cache after a prior Q8 reply divergence. Changing either format changes the memory budget. Preserve the same template, tool definitions, message history and sampling settings in A/B runs.

After the F32-wire run passes, unset `GGML_CUDA_AR_BF16_THRESHOLD` and repeat quality and performance measurements. Disable expert counters for timing runs. Counter output requires `-lv 4` because the common logger maps backend information messages to trace verbosity. Counters accumulate one hit per selected routed expert at the up projection, including graph replay; `first` identifies the first expert on that device, and `counts` follows expert-index order. Counts include warmup and prompt/decode work until clean shutdown.

## Experimental host KV with MTP retained

`--kv-host` enables `LLAMA_KV_HOST=1` for the four-card MTP-2 recipe. Each owner's target and draft Q8 MLA latent history is stored in pinned, mapped host RAM. The indexer, KDA state and resident k-pool selection tables remain on the owning GPU. CUDA copies the selected canonical Q8 rows into bounded GPU scratch before attention, preserving selection order and the existing attention arithmetic. The small dense span copies its visible cache. GPU writes, rollback and checkpoint transfers retain the existing cache layout. The target and draft each reserve their own scratch.

This is fixed host storage with per-operation staging. It does not implement a persistent GPU hot-row cache, pressure-triggered eviction, overlap with indexing, physical prefix sharing, or PCP/DCP. Every sparse attention call transfers its selected rows; PCIe cost and MTP acceptance need remote measurement. CUDA capture and 400K attention checks are provided in `test-backend-meta-split --host-kv-four`, which requires four GPU devices and prints their identities. `--host-kv` uses four logical ranks on one GPU and is not four-card qualification. The CPU suite covers ownership, writes, checkpoint transfers and attention outputs with changing selections. Unsupported backends return no mapped buffer type; direct sparse indices and Q8 pooled MLA are required.

The launcher requires a host-memory budget, rejects insufficient available RAM before starting, and permits up to the engine's 256 slots in this experimental mode. The budget excludes the model, 32 GiB RAM prompt cache and system allocations. The slot limit is an input bound, not a capacity claim. Example for inspecting the 72-slot estimate without starting the server:

```sh
python3 scripts/glm53/serve.py --profile dp-4 --kv-host --slots 72 \
  --host-budget-mib 200000 --mtp-depth 2 \
  --server build-dp/bin/llama-server --model /path/to/first-shard.gguf \
  --mmproj /path/to/matching-projector.gguf --dry-run
```

`scripts/glm53/capacity.py` fits fixed and per-owner-slot allocation from two measured MTP-2 boots of the same pack. At 442,368 cells, target plus draft latent history is 2,754 MiB per slot; the resident indexer is another 2,065.5 MiB. The observed increase from 36 to 40 slots is 5,288 MiB per card, so the estimate retains 468.5 MiB per owned slot for recurrent state and other incremental allocations. It subtracts only the offloaded latent bytes and adds 1,321.836 MiB/card for staging: target and draft main graphs plus each context's three retained 32-row graph slots. It checks the busiest card, host budget and explicit margins.

For box 58990727's 36/40-slot boots, a 4,096 MiB reserve plus 2,048 MiB growth allowance gives 36 resident slots or 72 host-KV slots. Seventy-two require 198,288 MiB (193.64 GiB) of RAM just for latent history. Increasing the growth allowance to 4,096 MiB gives 68 host-KV slots. These are linear projections, not measured full-context capacity or decode rates. Growth at 400K, allocator reuse, graph captures and host metadata can change the limit. The calculator marks both capacity and quality unqualified.

```sh
python3 scripts/glm53/capacity.py \
  --lower-slots 36 --lower-memory cold-36-loaded.memory.json \
  --upper-slots 40 --upper-memory cold-40-loaded.memory.json \
  --host-kv --host-budget-mib 200000 --peak-extra-mib 4096
```

The strict cross-layout KL requirement remains unmet. Keep this candidate out of Rig's placed profile and engine pin until component, MTP/restore, quality and active-memory checks pass remotely.

### November 3 host-KV measurements

Vast box 59024660 has four RTX PRO 6000 Max-Q cards with 97,342 MiB each. All twelve directed peer-read checks pass. These runs retain the BoldingBuilds pack and projector, non-unified caches, MTP depth 2 and the serving flags above. Public model files were verified by SHA-256. The box is stopped between runs and immediately after essential result collection.

The first full-model host-KV preflight found a greedy-token difference against resident KV at 8K. Commit `c55a759f9a34dce1d3866ec7249cebb2a26ae31c` restores the resident path's bounded early attention-kernel selection: stage the small cache below the sparse-path cutoff, then stage selected rows above it. A CPU graph-construction regression fails before the fix and passes afterward. On four physical GPUs, masked and direct sparse attention tests pass under captured replay and Compute Sanitizer reports zero errors. Resident and host KV then generate the same 64 greedy tokens at 8K with MTP enabled. Non-first local stream restoration and a real tool-call/result cache continuation also pass. These are limited correctness checks, not full-model KL qualification.

That build boots forty slots with 442,368 cells each. All forty short requests complete and are observed decoding concurrently. Card 0 has 28,462 MiB free after load and 27,056 MiB after the short decode. This establishes allocation headroom, not forty fully occupied 400K contexts. The 128-token output round takes 182.437 s including admission and prefill: 28.064 aggregate output tokens/s, with median native per-response decode of 0.922 tokens/s. The 64-slot follow-up was interrupted to investigate this slowdown; it was not an OOM result.

A separate Nsight Systems capture contains a 34.745 s forty-stream interval with 8,943 graph instantiations and 17,886 CUDA memory queries. Their summed API durations are 2.040 s and 4.379 s across host threads; these are not wall-time shares. Mapped-host row staging totals 0.120 s of GPU kernel time across devices. Every scheduled model segment uses the Meta GPU backend; no CPU model fallback was observed.

An attempt to merge the input-limited scheduler segments (`75d5bd454`) passed small CPU/GPU replay, sanitizer, 8K generation, restore and tool-cache checks, but failed the forty-slot MTP reservation. Keeping all copied inputs alive together required a 32,227.06 MiB allocation on GPU 0, which failed with OOM. Commit `a3ea9b8ce73c64271931dc8720f5215f01ecbd12` reverts that attempt. The split limit bounds staging memory even though its input array can grow; it cannot simply be removed. The box was stopped after collection. The resumed capacity run uses the previously validated `c55a759f9` binary. Graph-cache churn remains a separate performance issue, with no claimed fix or speedup.

The resumed run completes short concurrent rounds at all four allocation sizes. Each slot reserves 442,368 cells and starts with a distinct 128-token prompt. All requested slots are observed decoding concurrently. The forty-slot round generates 128 tokens per request; the other rounds generate 32. Native decode rates are per-response intervals, not aggregate wall throughput or fully occupied 400K rates.

| Slots | Least free after load, MiB | Least free after short decode, MiB | Median native decode, tokens/s per request |
| --- | ---: | ---: | ---: |
| 40 | 28,462 | 26,952 | 0.897 |
| 64 | 13,168 | 11,852 | 0.571 |
| 68 | 10,534 | 9,022 | 0.512 |
| 72 | 7,904 | 6,464 | 0.587 |

With seventy-two slots allocated, a fresh 409,088-token seed completes in 644.170 s, or 635.063 prompt tokens/s. Fifty-four additional slots successfully restore 409,088 cached tokens and process their distinct tails. The partial deep run peaks at 90,056 MiB on card 0, leaving 7,286 MiB. Sequential restore latency grows to about 17 s per request by slot 37. On each new task the server visits idle slots to publish RAM-cache snapshots; even a rejected no-space save computes target/draft state sizes before its admission check. This is a separate source-level scaling concern from CUDA graph reuse.

That run does not complete concurrent 400K decode. A replacement test driver uses a ten-second status timeout during heavy cache bookkeeping, times out, and triggers controlled shutdown. This is a harness failure, not a CUDA failure or a completed capacity qualification.

A final bounded attempt starts with the same runtime and flags, a longer status timeout, and concurrent admission while keeping the seed slot idle. The user stops it before qualification completes because short-context throughput is already unusable. The workload is interrupted and the provider confirms the box exited with both current and intended state stopped. No completed full-context result is claimed, and the box is not restarted to retrieve incomplete data. Existing closed results and the Nsight capture are retained locally. Further GPU work must establish useful concurrent throughput with short prompts before spending on long-context capacity.

## Further capacity approaches checked on November 3

Chinese-language searches covered Zhihu and Linux.do. Exa's public MCP endpoint returned HTTP 403, so these results came from the available web search, with implementation claims checked against original sources. Relevant leads were the [Tair team's hierarchical-cache article](https://zhuanlan.zhihu.com/p/2048808646793023819), the [Chinese ECHO discussion](https://zhuanlan.zhihu.com/p/2065819907154093188), and [GLM deployment reports](https://linux.do/t/topic/2878743). They are leads, not measurements of this fork or the selected pack.

- **Draft-off is not the selected strategy.** SGLang's [Flash cookbook](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/GLM/GLM-5.3-Flash.mdx) separates low-latency MTP and high-throughput recipes with speculation off. The team requirement here explicitly keeps MTP; its caches and scratch remain in our estimate.
- **Active sparse-KV offload is a new implementation direction.** [HiSparse](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/hisparse_guide.mdx) retains full sparse-attention history in pinned CPU RAM and retrieves selected rows into a small GPU working set. Its current SGLang route requires prefill/decode disaggregation; shared-index prefetch has additional restrictions, including no speculation. [ECHO](https://www.usenix.org/conference/osdi26/presentation/liu-guangda) provides another primary design: GPU-graph-compatible eviction/recall and lossless prefetch overlapped with indexing. Neither is an existing llama.cpp flag or a validated port to this hybrid model.
- **Pressure-triggered offload preserves the ordinary fast path.** [vLLM's Hybrid HiSparse](https://vllm-project.github.io/2026/09/08/glm53-part1-hybrid-sparse-offloading.html) leaves KV resident while there is room, then releases older pages into a host-backed sparse working set. Active indexer history still consumes GPU memory. That implementation supports speculative decoding through replayable resolver plans. It is demonstrated on GLM-5.3, not this Flash GGUF pack. A fork implementation would need exact Q8 row transfers, device-owned indexer/KDA state, bounded host RAM, capture-safe miss handling, and correct pool compression, rollback and checkpoint restoration. The likely capacity gain is an inference; PCIe traffic and miss rates must be measured.
- **Context parallelism offers a different way to use the cards for a single long prompt.** [vLLM's PCP/DCP design](https://vllm-project.github.io/2026/10/23/glm53-part2-pcp-dcp-hisparse.html) distributes prompt work and persistent KV by token. It adds communication and temporary gathered KV during prefill. Relative to our existing DP ownership, sharding a cache does not remove another four copies: DP already stores one copy across the box. The potential benefit is spreading an unusually long prompt's work and storage across owners. Their GLM-5.3 H200 measurements do not establish the cost on four PCIe cards or the KDA integration needed here.

The original brief already deferred per-device shared pools, physical prefix sharing, native all-gather/reduce-scatter and expert rebalancing. They remain useful but are not new findings. Per-device paging can reduce reserved-but-unused space, and identical team prefixes can share physical pages with copy-on-write; neither reduces the bytes needed by forty unrelated, fully occupied contexts. The current qualification intentionally materializes independent caches even when its prompts share a seed.

## Qualification and capacity

Local development used an RTX 5080 and RTX 5070 Ti without peer-to-peer access. Four logical ranks on those two cards validate partition mechanics; they do not validate four-card topology, capacity or speed.

On the four-card host, run the component and synthetic-model checks before loading the full pack:

```sh
build-dp/bin/test-backend-meta-allreduce-copy --direct 4 --require-p2p
build-dp/bin/test-backend-ops test -o MUL_MAT_ID_PARTITION
build-dp/bin/test-llama-archs --glm-arch-alias -s 1234
GGML_CUDA_AR_BF16_THRESHOLD=0 GGML_CUDA_ALLREDUCE=internal \
  build-dp/bin/test-llama-archs -a glm5next -s 1234
GGML_CUDA_AR_BF16_THRESHOLD=0 GGML_CUDA_ALLREDUCE=internal \
  build-dp/bin/test-llama-archs --glm-dp-device CUDA0,CUDA1,CUDA2,CUDA3 -s 1234
GGML_CUDA_AR_BF16_THRESHOLD=0 GGML_CUDA_ALLREDUCE=internal \
  build-dp/bin/test-speculative-mtp --dp-device CUDA0,CUDA1,CUDA2,CUDA3
```

The direct collective test covers host staging and peer paths on F32/BF16 wires, including tails and transfers larger than 32 MiB. `--require-p2p` fails if a requested peer path falls back to host memory. Without it, a fallback is labeled in the output. For three/four ranks it also runs an actual two-rank reference with the same two contributions and zero-valued extra ranks; this is bit-identical. Arbitrarily regrouping nonzero floating-point contributions is not guaranteed to preserve bits.

The standard `-a glm5next` command includes DP 2 and DP 4 with default and Q8 caches. With fewer than four CUDA devices, repeated devices are labeled logical ranks; a CPU-only build uses logical CPU ranks. The full CPU command passes locally. On the local CUDA cards the DP configurations pass, then the existing unsplit generated-prefix rollback comparison fails; the executable from before this test integration reproduces that failure. The focused `--glm-dp-device` command isolates DP results but does not waive the unsplit failure.

The expert-partition suite passes 108 cases on CPU and each local CUDA card, including all five expert formats in the pinned Unsloth pack and rows with no local experts. The architecture-alias regression compares identical CPU logits through user-supplied metadata and file loading.

The MTP test compares every draft and greedy verification result with an unsplit model, and exercises forced acceptance of zero, one and two drafts after slot reuse. The untrained fixture has no naturally accepted greedy drafts; these are correctness checks, not an acceptance-rate benchmark. Run it under Compute Sanitizer as well. The backend handles failed CUDA graph updates by re-instantiating the graph; use `--report-api-errors no` to check device memory without counting that handled update result as a memory failure.

- Synthetic GLM tests cover two/four ranks, mixed streams, microbatch boundaries, cross-owner forks, per-sequence/full checkpoints, MTP hidden-store/head behavior and rollback against an unsplit model. CPU tests also check rollback against the kept-prefix oracle.
- CUDA dense kernels change arithmetic with row count. Single-device 16-row and 64-row controls themselves change one near-tied top token in the synthetic fixture. DP comparisons retain the KL limit of 1e-6 against both controls and require top-token agreement with one entire unsplit control. This is not a claim of byte-identical logits across batch shapes.
- Before promotion, run the actual model at 2K, 4K and 16K, including multi-turn coding/tool workloads and MTP draft/acceptance comparisons. Repeat with BF16 collective wires enabled.
- On four physical cards, check the per-device KV allocation is one quarter of the total, then exercise 40 concurrent approximately 30K prompts. Record peak VRAM during load, prefill, decode and graph capture; every card must remain under the requested 93 GB ceiling.
- Measure 1/12/40 decoding streams and one/four 45K prefills. Report aggregate throughput, per-stream throughput and inter-token latency while another request prefills. Use ABBA against two-card tensor mode on two of the same cards.
- The fallback candidate is 36 slots at `-ub 1024` (`-c 15925248`). It needs its own measurement. Smaller microbatches can improve responsiveness but do not guarantee the highest prefill rate.

The requested 40-slot Rig profile must use the measured peak plus 1 GiB as `min_vram_mib`. Its `draft_cache = { k = "q8_0", v = "q8_0" }` overrides the draft formats only for that profile. The draft microbatch inherits the target's 512 rows when `-ubd` is absent. Until the hardware measurements exist, no measured VRAM threshold or production profile is claimed.

## Four-card measurements (Nov 2, box 58861766)

Four RTX PRO 6000 Blackwell Server Edition (97,887 MiB each by nvidia-smi, all four on one NUMA node over PCIe, no NVLink), EPYC 9554, driver 580.105.08, CUDA 13.0. The fork at 9018dec92 (train/glm53-dp), built with `-DGGML_CUDA=ON -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120 -DLLAMA_CURL=OFF`. The pack measured is the one rig serves and pins by hash, BoldingBuilds/orcarouter_GLM-5.3-Flash-Uncensored-GGUF at 1bc0afad, IQ3_XXS in three shards (sha256 verified against `heads/glm-5.3-flash/head.toml`): the orcarouter abliteration of zai-org/GLM-5.3-Flash, quantized by BoldingBuilds. The Unsloth pack above was not measured. The runbook that drove the box is `scripts/glm53/dp-qualification.sh`'s ancestor kept with the session; every number below is in its logs.

Component checks on the four cards: `test-backend-meta-allreduce-copy --direct 4 --require-p2p` 66 ok (peer copy, host linked-list, host chunked; F32 and BF16 wires up to 16,777,219 elements); `test-llama-archs --glm-dp-device CUDA0,CUDA1,CUDA2,CUDA3 -s 1234` exit 0, "GLM dp 4: mean KL 0, 0 differing top tokens"; `test-speculative-mtp --dp-device CUDA0,CUDA1,CUDA2,CUDA3` exit 0; `test-llama-archs -a glm5next -s 1234` fails on the unsplit generated-prefix comparison recorded above as pre-existing (max_abs 2.96e-06), after its DP configurations pass.

### Two crashes at load, fixed in 9018dec92

The 40-slot server of the candidate command did not load on 9bd1f9c61. Each crash has its test in `tests/test-llama-archs.cpp`, written first and failing before the fix, and both reproduce on a CPU build (the DP ranks run on the CPU device there):

- `glm5next.cpp:420: GGML_ASSERT(n_tokens == n_tps*n_stream)` in the slot-graph reserve (`llama-context.cpp`): the reserve measured 32 rows over min(n_seqs, 32) sequences while the full memory context views every stream of the cache, so with more than 32 slots the indexer scoring found 32 rows over 40 streams. The reserve now keeps every sequence (test_glm_many_slots). Any server over 32 slots crashed this way, DP or not.
- `ggml-backend-meta.cpp:2835: GGML_ASSERT(size % chunk_size_full == 0)` on the server's two-token probe at load (common_context_can_seq_rm): the MTP head graph set `t_h_nextn` to every row and the masked read took the first n_outputs of them, a partial read of a split tensor under DP, and the wrong rows on one device whenever a step has fewer outputs than rows. The head now hands the masked read the output rows (test_glm_nextn_masked_rows).

### Memory at 40 and 36 slots

Per card at 40 slots (`-c 17694720`, 442,368 cells a slot, q8_0 caches, ub 512): model 39,218 MiB, attention cache 25,245, indexer cache 18,934, recurrent state 1,371, the MTP head's caches 2,295 and 1,721, compute 2,206 plus the head's 1,529, three graph slots up to 278 each. Load peak 93,861 MiB on the busiest card, 95,685 MiB resident after load: 2.2 GB free. That is not enough: the graphs grow past the reserve at depth under DP, and the 12-stream run below ended the server with `cudaMalloc failed: out of memory` allocating 603.83 MiB on device 0 in `ggml_backend_sched_alloc_graph` (process_ubatch). The 40-slot profile the PR asks for cannot be served on the 96 GB class.

At 36 slots (`-c 15925248`): load peak 90,459 MiB, 90,473 resident after load, 90,819 after a one-stream 400k session, 94,649 during twelve concurrent 400k prefills at ub 512 (the runtime growth is about 4 GB a card under that load). At ub 2048 the same server loads at 95,459 MiB and reaches 96,143 under the twelve prefills.

### 400k sessions, Claude Code's shape

`scripts/glm53/dp_longctx.py` builds a history the way Claude Code sends one: a system prompt, twelve tool definitions, a user request, then Read calls and their results from a public corpus (the fork's `src/`) until the server's own count (`/apply-template` and `/tokenize`) reaches the target; each later turn appends the reply, a question and one more Read, so the prefix stays byte for byte and the cache can hold. Sessions here measured 395k to 400k tokens (the builder's first version let two sessions through at 443k and 458k, over the slot; the server answered 400 and the builder was fixed in 9018dec92: the count decides, never the estimate).

One stream alone (36 slots, ub 512): a 395,395-token history read in 235.2 s, 1,681 tokens a second on its one card; the reply decoded at 37.9 tokens a second; turn 2 found the whole history in the cache (646 new tokens over 395,391 cached, 0.88 s) and decoded at 42.1; turn 3 at 46.9. No cache lost. (At 40 slots, before the memory ran out: 419,337 tokens in 252.1 s, 1,664 a second, decode 34.9 / 55.3 / 37.9.)

Twelve streams at once, each a 400k history from empty: the per-stream prefill rate is 100 to 136 tokens a second whatever the step or the microbatch, measured at ub 512 with the 512 step (130 a second a stream, 1,560 aggregate), at ub 512 with the 2048 step (115 to 136), and at ub 2048 with the 2048 step (101 to 103). Four cards together read twelve prompts no faster than one card reads one: each card's throughput falls from 1,681 tokens a second with one stream to about 300 to 390 with three. A stream decoding beside eleven such prefills produced 166 tokens in 84.0 s, 2.0 tokens a second, and its 646-token turn took 17.1 s to read. A 400k session opened while eleven others open theirs is ready after about an hour, against four minutes alone. The twelve-stream run was not taken to completion: turns 2 and 3 under concurrency are not measured.

The same history through `/v1/messages`, the way Claude Code sends it (system apart, tools as input_schema, Read calls as tool_use blocks, results as tool_result blocks), 392,251 input tokens on the 36-slot server: answered in 233.4 s with a tool_use block (a Read of `src/llama-quant.cpp`), stop_reason tool_use. The Anthropic path renders and serves a 400k agentic history.

### Why twelve streams read no faster than one

The branch's `-sm dp` is sequence parallel for attention and expert parallel for the MoE: each card keeps its own sequences' caches and the dense weights, and holds a quarter of the 288 routed experts. Every MoE layer (`src/models/glm5next.cpp`, `ggml_backend_meta_gather` before `build_moe_ffn`, `ggml_backend_meta_reduce` and `ggml_backend_meta_scatter` after it; the shared expert runs on each card's own rows) gathers the whole microbatch's rows onto every card, runs the card's 72 experts over all of them, and sums the four partials. Both collectives move the whole batch's rows, n_embd 4096 in F32, 16 KB a row: 8 MB a layer for a 512-row microbatch, 32 MB for 2048 rows, over 45 layers, and the four cards are on PCIe through the host, no NVLink.

The two collectives take different paths in `ggml/src/ggml-cuda/allreduce.cu`. The reduction over 1 MB goes by copy engine card to card (`ggml_cuda_ar_pipeline_init`: "copy-engine receives card to card, 96 MB receive scratch per GPU" in the server's log, the probe having found peer stores work on these cards): each card copies its 32 MB partial into three peers' receive buffers and adds what it received, 96 MB out and 96 MB in per card per layer, about 4.3 GB each way a 2048-row microbatch. The gather never takes the copy engines (`ggml_cuda_ar_allgather`: "the words go through the kernels whatever their size, so a capture holds the gather"): above the 128 KB peer-kernel limit it runs `ggml_cuda_ar_kernel_n` through pinned host memory in 1 MB chunks (`host_buf`, the "1024 KB chunked kernel staging"), each chunk a kernel store to the host, an arrival counter all four cards wait on, then every card's kernel reading the four chunks back from the host: 32 chunks a layer at 2048 rows, about 1,400 host round trips with a four-card barrier in each per microbatch, 360 at 512 rows. The per-microbatch time follows the rows: one stream (512 rows, 305 ms a microbatch, 1,681 tokens a second) and twelve streams (2,048 rows, about 1.7 s, 1,220 a second) cost the same per row, and the rows of twelve streams are no cheaper than the rows of one. With 8 of 288 experts a token, a token reaches a given card's experts with probability about 1 - (3/4)^8 = 0.90, so a top-k dispatch would carry nearly the same rows as the gather: the volume is in the model's shape, the path is the code's choice.

NCCL on the same tree (`-DGGML_CUDA_NCCL=ON` against the image's libnccl 2.28.3, GGML_CUDA_ALLREDUCE unset; its gather is an `ncclAllReduce` of the words, its reduction an `ncclAllReduce` in F32): twelve 400k streams at ub 512, step 2048, read at 119 to 129 tokens a second a stream after 90 s and 128 to 142 after seven minutes, against 115 to 136 on the internal collectives. Not the fix.

The server itself under nsys (`nsys profile --trace cuda,nvtx --cuda-graph-trace node`, twelve streams at 32k): a 66.5 s window of twelve concurrent prefills, then the server died under the profiler, `CUDA error: unspecified launch failure` 78 s in (the three twelve-stream runs without the profiler, five to seven minutes each, did not; what the profiler changes is not established). In the window each card is busy 50 to 52% of the time, and for 35% of it no card is busy at all, the host between launches. Of the kernel time summed over the four cards, 34.2% is the gather kernel, `ggml_cuda_ar_kernel_n<uint32>`: 39.7 s in 186,165 launches, median 89 µs, the longest 7.05 ms (a chunk waiting at its barrier for the slowest card). The two main expert matmul kernels (`mul_mat_q` IQ3_XXS at 16 and 80 rows) take 32.8 s together, the card-to-card copies of the reduction 11.5 s, host-to-device copies 6.6 s, the F32 reduction kernel for pieces under the copy-engine threshold 6.2 s. The cards spend more time in the gather than in the expert matmuls, and a third of the window idle.

This is the known shape of expert parallelism on PCIe hosts (ThunderEP, arXiv 2609.40093: "all inter-GPU transfers traverse CPU memory"; vLLM's EP guide requires NVLink or RDMA for its all-to-all backends and names the all-gather/reduce-scatter fallback as the only path without P2P), and the reason inference stacks serve such models on PCIe boxes as tensor parallel, where the per-layer all-reduce carries one row set instead of every card's. The branch has the pieces to do better than it does: the copy-engine peer path the reduction already takes, a reduce-scatter plus all-gather in place of four full partials (half the bytes), and a BF16 wire (`GGML_CUDA_AR_BF16_THRESHOLD`) for the rest.

### Two-card tensor mode on this tree

Rig's served profile (prefill-512k: `-sm tensor -ts 1,1`, three slots on one 524,288-cell unified pool, batch 2048/2048 step 1024, q8_0 caches, draft-mtp n_max 2 with the public vocab, GGML_CUDA_ALLREDUCE=internal) on cards 0 and 1, the same 395,395-token session on each build, main's first:

- main (28b26d62d, built the same way): the history read in 146.4 s, 2,700 tokens a second; the reply decoded at 51.8 tokens a second, turns 2 and 3 at 83.6 and 85.1 with the history in the cache. No cache lost. 87,155 MiB a card after the session.
- this tree (9018dec92): the read ran at 3,450 to 3,590 tokens a second until, 122,686 tokens in, the server died: `CUDA error: an illegal memory access was encountered` in `ggml_cuda_ar_acquire_slot` (`allreduce.cu:751`, `cudaEventSynchronize(p->ev_pool[i][slot].ker)`), at the same point on each of three runs (one at 419k, two at 395k).

So this tree breaks the two-card tensor profile rig serves, and main does not. The DP branch's 20 commits are the difference (the five PR #92 fixes are on both, as different SHAs). The same recipe at the branch's commits, one 250k session each, on the same cards (`llama-server` built at each commit, CUDA, arch 120):

- 974b2c611 (cuda: extend internal collectives to multiple peers): good, the 250k session read and three turns answered, cache kept.
- c875f8ff7 (llama: group ubatch rows by device and sequence): the 250k read completes (249,782 tokens), then turn 2, the cached prefix plus a short read, dies in `ggml-backend-meta.cpp:1357: GGML_ASSERT(split_state.ne[j]*split_state.nr[0] * tensor->src[i]->ne[src_ss[i].axis] == sum * tensor->ne[split_state.axis])`. 0d401ff8c (ggml: add explicit row layouts and meta gather/scatter), the commit before it, is good; 32eef41f3 (ggml: preserve ownership in slices of split cache buffers) is the first with this failure, and every later commit up to f78379aaa has it: `ggml_backend_meta_slice_view` now answers for every `GGML_OP_VIEW` of a split buffer in `ggml_backend_meta_get_split_state`, the tensor-split cache's views included.
- da8a76e5a (ggml: preserve device ownership through GLM ops and cache access), 9d77c323e, 7c07a29ac and the head: the illegal memory access at 122,686 tokens of turn 1, in `ggml_cuda_ar_acquire_slot`. f78379aaa (cuda: execute indirect matmuls over local expert ranges), the commit before da8a76e5a, reads the whole session and fails only at turn 2 as above, so the illegal access is da8a76e5a's.

Two regressions, then, both in tensor mode on two cards with no DP flag set: one in the second turn from the row grouping, one in the first read from the device-ownership work that followed it. Neither is covered by a test: `test-llama-archs` compares DP against batch-shape controls, and nothing on the branch runs the tensor recipe rig serves. The tests for both go in before the fixes, as the tests for the two load crashes above did.

### What this says

- The two load fixes and `dp_longctx.py` stand on their own: the slot-graph reserve bug hits any server over 32 slots, DP or not, and the session builder drives any llama-server.
- The 40-slot, 400k profile does not fit four 96 GB cards; 36 slots do, with 1 to 2 GB to spare under load.
- Alone, a 400k session is served well: 1,681 tokens a second to read, 38 to 47 to decode, the cache kept across turns. Beside eleven others it is not: a 400k read takes about an hour instead of four minutes, and decode beside the reads runs at 2 tokens a second. The rig profile this was for, many Claude Code sessions at 400k, is not served by this tree as it stands.
- The tree breaks the two-card tensor profile rig serves today, in two places. It cannot merge until both are fixed, with their tests.
- The path forward for DP prefill is in the gather: card to card by copy engine, as the reduction already goes, then reduce-scatter and a BF16 wire. Expert parallelism over PCIe will not read twelve prompts at four cards' speed, but it need not read them at one card's.


### Measurement schema 2

`dp_longctx.py` takes each request's decode rate from the server's `predicted_per_second`, including a literal zero. For old responses without that field it uses `max(predicted_n - 1, 0) / predicted_ms * 1000`; missing or nonpositive durations produce an unavailable rate. The first prompt-logit output is excluded from steady decode work.

Every raw row and partial JSONL row stores `request_start_s` and `request_end_s` on the run's shared monotonic clock, relative to run start. Per-turn `output_tps_aggregate` divides delivered output tokens by `max(request_end_s) - min(request_start_s)`. It includes prefill, queueing, HTTP and gaps between staggered requests. It is not server decode throughput. The old `decode_tps_aggregate` sum of request rates was removed. Historical rows without endpoints have unavailable common-interval throughput. Report `elapsed_s` also includes history construction and sampler shutdown, and is not the throughput denominator.

Raw producer timings and finish reasons are retained. Summary diagnostics sum draft, accepted, delivered, round, verify and replay counters and each verify-row histogram bucket. Acceptance and delivered-per-round use summed paired counts; missing fields remain unavailable rather than observed zeros. These reporting changes do not establish any speedup.

### November 4: 40-slot startup with backend sampling

Four RTX PRO 6000 Blackwell Server Edition cards (97,887 MiB each, 600 W, driver 615.71.09) on Vast machine 156124. These are different cards from the earlier Max-Q measurements. The pack is BoldingBuilds/orcarouter_GLM-5.3-Flash-Uncensored-GGUF IQ3_XXS at revision `1bc0afadd92f4ff9fc2b1281a95a7921b058d35d`; all three shards and the projector passed SHA256 verification.

The tracked `dp-4` recipe used 40 private slots, `-c 17694720`, host latent KV, Q8 caches and recurrent state, MTP2, backend sampling, batch 2048 and ubatch 512. Main `7728616ce` aborted during startup: the fused-GDN probe requested 640 output rows, but the sampler limit marked only 120, so DP ownership counts disagreed with the output-ID tensor. Commit `82ad56cb8` makes reservation use the marked count. The new regression fails before the fix and passes after it on CPU and physical two/four-GPU DP, including fully and partly sampled 40-sequence batches and exact greedy verification tokens against an unsampled context. The graph-key, graph-aging, disabled-memory-diagnostics, four-device capture, host-KV ownership, sampling-row, Meta sampler and MTP checks passed on the cards.

With the fix, the full server loaded 40 slots of 442,368 cells each with MTP enabled. Sampled peak memory was 69,235 MiB on card 0 and 67,595 MiB on each other card. This verifies startup allocation with host KV; the slots were not filled with 400k histories. Two identical 128-token prompts produced matching eight-token greedy replies. Their server-reported prefill rates were 23.65 tokens/s on the first execution and 273.11 on the repeat; decode was 12.79 and 11.79 tokens/s. These short probes include insufficient warmup to establish steady throughput.

The timed sweep failed before its first token: the HTTP stream ended before the common interval began. The collected log has no CUDA error or memory-headroom failure; the cause of the stream closure is unresolved. The client works against a local CPU server, including the same greedy/prime/stream sequence with 40 slots and 128/1200-token prompts. Its next version records HTTP headers and unexpected response data so the failure is distinguishable from an inference error. There is no valid 1/4/12/40-stream throughput result, tool-cache continuation result, profiler capture or 400k capacity result from this window.

All temporary instances were stopped and deleted after collection. The original cached-model instance remains stopped because its host would not restart it. Local receipts are under `.Codex/tmp/dp-gpu-20261104/`: `receipts/results` records the main startup failure; `retry/receipts/results` records the fixed CUDA checks and a download timeout; `final-retry/receipts/results` records the successful full-model startup, short probes, memory samples and incomplete sweep.

### November 4: single-session profiling and scaling controls

Runtime `82ad56cb8b67c83c0977ae9e4d23da9b5779d469`, built for SM120a, on Vast machine 156135: four RTX PRO 6000 Blackwell Server Edition cards, 97,887 MiB and 600 W each, driver 615.71.09. This is another machine, not the earlier 156124 or Max-Q host. The same BoldingBuilds pack and revision above passed full shard/projector SHA256 verification. The [machine-readable measurements](glm53-dp-profile-20261104.json) retain each response's native timing, MTP counters and the trace summaries.

The candidate uses `scripts/glm53/serve.py --profile dp-4`, `-sm dp -ts 1,1,1,1`, private 442,368-cell slots, Q8 caches/state, backend sampling, MTP2, batch 2048, ubatch/prompt step 512 and 8/16 CPU threads. Host-KV legs set `LLAMA_KV_HOST=1`; every leg enables the existing `LLAMA_TIME_DECODE` timers. CUDA graph count is unlimited, the node budget retains its default 32,768, and graph memory diagnostics are off. No production recipe was changed.

One active session, native decode tokens/s:

| Configured slots | Latent KV | MTP | 512-token prompt | 24,576-token prompt | 24,576-token prefill tokens/s |
| --- | --- | --- | ---: | ---: | ---: |
| 40 | Host | 2 | 45.3 | 31.7 | 604 |
| 1 | Host | 2 | 46.3 | 36.7 | 608 |
| 1 | GPU | 2 | 45.6 | 36.8 | 698 |
| 1 | GPU | Off, diagnostic | 62.4 | 25.5 | 720 |

Each reply contains 128 generated tokens. The 512 column averages two post-warmup requests; the longer-context entries are single measurements. The 40-slot host/MTP2 leg also measured 35.2 decode tokens/s and 649 prefill tokens/s at 4,096 context. Returning to 512 after the longer prompts gave 42.1 decode tokens/s. Cold first-use prefill was slower and is retained separately. These are bounded warmed requests, not a long-running stationarity test.

All MTP-on controls produced the same 128 token IDs for the corresponding prompt and seed. The MTP-off output differs, so its rate is a same-request diagnostic with a different expert-routing/token path, not an exact fixed-token ablation. The observation supports retaining MTP for longer contexts and investigating its short-context cost; it does not support disabling MTP globally. Configured slot count does not explain the short-context floor, and host KV has almost no decode penalty in the one-slot 24K comparison. The 40-slot 24K result is about 14% below the one-slot result; there was no randomized repeated-order experiment to isolate that difference.

Concurrent requests use distinct slot prefixes, 512-token prompts and 64 generated tokens per session:

| Active sessions | Native decode tokens/s per session | Delivered output tokens/s over the common HTTP interval, including prefill |
| ---: | ---: | ---: |
| 2 | 28.1-28.7 | 33.8 |
| 4 | 12.3-15.6 | 30.9 |
| 8 | 5.85-6.57 | 30.3 |

The concurrent reply length differs from the 128-token single-session controls. The common-interval column includes prefill, queueing and HTTP; it is not aggregate steady decode throughput. Per-session rates are not summed into a claimed aggregate rate. The equal-length 2/4/8-session comparison already shows poor scaling. A bounded 16-token SSE request also completed; this run did not reproduce or explain the earlier stream closure.

Nsight Systems 2026.1.3 captured separate 48-token, single-session requests at 512 and 4,096 context after warmup. CUDA/NVTX/OS-runtime traces were analyzed locally after collection. Profiled rates are excluded from the baseline table.

| Trace observation | 512 context | 4,096 context |
| --- | ---: | ---: |
| Span from first to last GPU kernel | 1,552.9 ms | 1,714.1 ms |
| Collective share of summed kernel durations across four GPUs | 74.6% | 73.2% |
| `cudaGraphInstantiate` calls | 368 | 1,076 |
| Summed host duration of those calls | 224.4 ms | 275.5 ms |
| `cudaStreamSynchronize` calls | 1,296 | 1,424 |

The main finding is coordination overhead in the DP execution path. Collective durations include waiting for other ranks, and summing durations across GPUs does not measure critical-path wall time. On GPU0, collectives account for about 38-40% of summed kernel duration; on the other cards they account for about 87-89%. Each card has no active kernel for roughly 46-51% of the captured kernel interval; copy-engine activity is excluded from that calculation. This is not evidence that the cards sustain their memory bandwidth or that all collective time could be removed.

One specific extra collective is outside the intended MoE gather/reduction pair. `llm_graph_context::build_sampling` in `src/llama-graph.cpp:4773` adds a dummy logits row with PAD. Under DP, logits are split by owning rows, and `ggml_backend_meta_is_gather` in `ggml/src/ggml-backend-meta.cpp:1784` treats padding on that split axis as a gather. The first target output in the NVTX trace has `PAD ... r=619520 w=1239040` on its owner and zero input bytes on the other ranks, followed by the large unsigned gather kernels. Full vocabulary rows are therefore gathered onto all four cards before sampling even for one owning stream. The large peer-push gather kernels alone account for about 21-22% of summed GPU kernel duration in these traces. That kernel-family percentage also includes captured executions whose tensor identity was not separately logged; it is not a projected speedup from fixing sampling. Owner-local DP sampling is a concrete optimization candidate; no implementation or measured speedup is claimed here.

Graph instantiation remains another measured cost. Most small instantiations in the 512 trace occur near its beginning, but further expensive instantiations occur throughout the request; the 4K trace has another large burst mid-request. The counts alone do not distinguish new shapes, Meta capture replacement and CUDA LRU eviction. The existing host timers place 85-95% of cumulative decode-call time in `graph_compute`; those aggregates mix target/draft and prefill/decode calls, so they are not a separate steady-decode CPU profile. Cache-lifetime and graph-identity attribution should precede changes to cache caps.

Bulk peer-copy measurements over 64 MiB transfers were 53.7-55.3 GB/s across all twelve directed GPU pairs, with P2P available everywhere. This rules out absent P2P on this host, not small-collective latency or rank-arrival imbalance. Compute Sanitizer 2026.2.1 passed `test-backend-meta-split --host-kv-four` with zero errors, including mapped Q8 attention and changing selections. GPU memory/utilization/power/clocks and PCIe telemetry were collected; occasional NVML query timeouts leave gaps. Sampled memory during the 40-slot requests peaked at 69,545 MiB on GPU0 and about 67,933 MiB on the other cards. The one-slot host run briefly overlapped the previous profiler process's CUDA cleanup during startup; its own server became ready and served requests after that old allocation was released. Transition peaks are identified separately in the JSON.

Nsight Compute 2026.2.1 was attempted remotely before downloading the model and refused access with `ERR_NVGPUCTRPERM`. CPU stack sampling also failed the host's `perf_event_open` capability check (paranoid level 4). Thus there are no hardware-counter rooflines or sampled CPU stacks from this run. The CUDA traces, engine timers, transport measurements and sanitizer results remain valid evidence. Running the profiler frontend elsewhere would still require permission to collect counters on the target.

The next work supported by this evidence is owner-local DP sampling, attribution of repeated graph capture/instantiation, and the collective/MTP scheduling path under concurrency. Arithmetic kernels, host-KV capacity and raw PCIe bandwidth are not established as the main cause. No 400K capacity, perplexity/KL, tool-call cache-continuity or 40-active-session qualification was attempted in this window.

Receipts, launch commands, raw responses, model hashes, telemetry and `.nsys-rep` files are in `.Codex/tmp/dp-profile-20261104/receipts/results`; `summary.json` and `early/` hold the local analysis. The raw archive SHA256 is retained in the tracked JSON. The first empty rental failed its direct-port mapping and was deleted. The replacement used proxy SSH, completed the measurements, and was stopped/deleted after collection at about 16 minutes into the successful controller run. Credit decreased about $1.44 across this profiling effort, including startup attempts and transfer. Only the original cached-model instance 59024660 remains, stopped and still incurring its existing disk charge.
