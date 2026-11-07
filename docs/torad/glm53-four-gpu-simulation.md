# GLM-5.3 Flash four-GPU decode simulation

## Goal

Explain how this fork serves GLM-5.3 Flash on four RTX PRO 6000 cards, and give the serving team a reproducible, CPU-only model of the memory, sequence ownership, inference phases and communication required for multiple independent 400,000-token histories. Also show 409,600 tokens, the documented client budget. The simulation must distinguish reserved slots from active sessions, latent attention history from indexer and recurrent state, resident KV from mapped host KV, and speculative proposals from delivered tokens.

Success means the model conserves tokens and owner-local storage, exposes its assumptions and source references, and identifies optimization candidates without presenting analytical byte counts as measured latency or throughput. It must not allocate model weights, start a server, contact a GPU, change the inference engine, or promise numerical/quality equivalence. Current four-card performance and full-context qualification remain separate from this simulation.

## Findings first

1. **The default four-rank reduction falls back to host staging at 6..40 full-depth MTP2 sessions**, despite working P2P. This is a source-proven dispatch band, not a measured slowdown attribution.
2. **The gather exchanges zero-padded full buffers.** An owner-slice implementation can reduce its logical four-card payload by 75%, without pretending expert routing touches only a few cards.
3. **The indexer reserves mostly unused pooled-key positions.** Keeping raw keys/gates but compacting the pooled plane can save about 5.043 GiB/card at 40 slots, before new mapping overhead.
4. **Two graph/hidden-state thresholds leave 11..21 MTP2 sessions in a distinct middle range.** Test that range explicitly instead of assuming smooth scaling with concurrency.
5. The code already has owner-local sampling, constant-layout preparation and retained DP selections. Do not count them again as new work.

The simulator and all 105 tests in the GLM Python tooling suite pass. No engine change, model load, paid machine, GPU benchmark or production configuration change was made. There is no new 400k tokens/s or quality claim.

## Configuration actually selected

The entry point is `scripts/glm53/serve.py --profile dp-4`, not `--glm53-profile sm120-balanced`. The latter is a different experimental kernel bundle and rejects non-F32 recurrent state (`common/glm53-profile.cpp:94`), whereas this DP capacity recipe explicitly uses Q8 state.

| Setting | Four-card DP recipe | Two-card default |
| --- | --- | --- |
| Placement | One four-device Meta backend; `-sm dp -ts 1,1,1,1` | `-sm tensor -ts 1,1` |
| Slots | 36 by default; resident launcher accepts 1..40 | 3 |
| Context | 442,368 private cells per slot; 15,925,248 total at 36 slots | One 524,288-cell unified pool |
| Main batch / microbatch | 2,048 / 512 | 2,048 / 2,048 |
| Prompt step beside generation | 512 | 1,024 |
| Target K/V and indexer | Q8_0 | Q8_0 |
| Recurrent state | Explicit Q8_0 | Default F32 |
| MTP | Depth 2 default; 98,304-token draft shortlist; adaptive prior 0.18 | Depth 2 default |
| Draft | Microbatch 512 and explicit Q8 K/V | Separate defaults |
| Sampler | Backend sampling explicitly enabled | Not explicitly enabled by this launcher |
| Allocation fitting | Off | Not overridden |
| Idle-slot RAM snapshots | Disabled | Disabled |
| Host prompt-cache budget | 32,768 MiB | 32,768 MiB |
| CUDA graphs | Count cap 0 means unlimited count, not disabled graphs | Same |
| Collectives | Internal | Same |
| Indexer query rounding | Q8 by default, lossy | Same |
| Direct sparse indices | Engine default | Explicitly disabled |
| MMQ overrides | Engine defaults | `MOE_NCOLS=0`, `L2_PF=1` |

Sources: `scripts/glm53/serve.py:48-105`, `docs/backend/GLM53-SERVING.md:79-135`. The launcher removes inherited LLAMA/GGML/CUDA/NVIDIA overrides except named server access-control settings, installs the shown environment, and sets the selected binary's library directory. Use its dry run rather than combining this recipe with an unrelated shell profile.

The engine requires GLM5NEXT, equal device shares, every layer and output on GPU, non-unified KV, KV offload and flash attention. It rejects tensor buffer overrides. See `src/llama.cpp:158-189`, `src/llama-model.cpp:1654-1689`, `src/llama-context.cpp:509-549`. Host KV is a supported experimental mapped-buffer path under this configuration, not the ordinary CPU-attention fallback.

The exact documented pack is BoldingBuilds/orcarouter_GLM-5.3-Flash-Uncensored-GGUF, revision `1bc0afadd92f4ff9fc2b1281a95a7921b058d35d`: three shards totaling 134,305,522,848 bytes and an optional 1,128,047,744-byte projector. Do not substitute the smaller Unsloth pack in its memory budget. The model files and hashes are listed in `docs/backend/GLM53-SERVING.md:41-50`.

The retained full architecture metadata is `.Codex/tmp/glm53-dp-20261102/requested-pack-model-metadata.json`, from the alternative Unsloth export: 45 target blocks plus one NextN block, hidden width 4,096, 64 attention/KDA heads, 128-wide KDA heads, 288 routed experts with eight selected, one 2,048-wide shared expert, vocabulary 154,880 and four hyper-connection streams. The Bolding run records independently establish the 4,096-wide expert rows, 288 experts, 42 target routed layers, vocabulary row width and cache geometry. This analysis did not re-download or enumerate the Bolding tensors, so the remaining shared architecture dimensions are explicit model assumptions, not a claim that the two quantization packs have identical weights or allocations.

The documented client budget is 409,600 input/history tokens plus 32,768 output tokens. This report also models decimal 400,000 explicitly; the two are not interchangeable. Retained history, tool schemas, reasoning, image tokens and new input all consume the context budget.

## What each GPU owns

For sequence ID `s`, the owner is `s % 4`. Nine of 36 slots belong to each GPU. IDs 0, 4, 8 and 12 all belong to GPU0, even though four sessions are active.

- Dense weights, embeddings, routers, attention projections and shared-expert weights are replicated.
- Each GPU owns 72 of the 288 routed experts in each routed layer.
- A sequence's attention history, indexer, recurrent state and selection metadata stay with its owner.
- Every routed-expert block still involves all four ranks: gather rows, evaluate local experts, combine partial results and keep the rows on their owners.
- A single stream therefore uses one card for its dense path while all four participate in its routed experts. This is neither four complete independent model replicas nor a 2x2 tensor/data layout.

Sources: `docs/torad/glm53-dp.md:60-64,277-281`. Under equal per-row dense work, four sessions on one owner have a 25% dense-phase utilization ceiling; four evenly placed sessions have a 100% ceiling. These are placement bounds, not observed GPU utilization or throughput.

## One request through the inference path

1. **Render and admit.** The server renders the chat/tool history, tokenizes it, chooses a slot and checks its private context budget. Stable prefixes may remain in that slot or be restored from a RAM snapshot. Image/audio embeddings are a separate prefill path through the optional projector; they are not free text tokens. The MTP implementation pairs media rows with target hidden state using its media token.

2. **Prefill missing rows.** The server prioritizes generating rows, then caps additional prompt work with prompt-step when generation is present. The 2,048-row logical batch can be split into 512-row physical microbatches. Equal-sequence splitting, owner placement and tokens per sequence determine the actual kernel opportunities. A populated cache and an allocated cache are different states.

3. **Run the target trunk on owners.** Embeddings initialize four mHC streams. Each of 45 blocks reduces/mixes those streams around attention and feed-forward work. The first three feed-forward blocks are dense; the remaining 42 have routed experts plus one shared expert.
   - **34 KDA blocks:** project Q/K/V, update the short convolution history, derive gates and update the matrix-valued recurrent state. They do not read all 400k old tokens.
   - **11 DSA/MLA blocks:** write the shared 512-value latent K/V row, write raw index key and independent compressor gate, finish new four-token pools, score pooled keys, select 512 pools and expand them into up to 2,048 token cells plus the tail. The absorbed query attends against the shared latent history; value expansion and output projection follow.
   - **Routed feed-forward work:** gather normalized 4,096-wide rows, evaluate each rank's 72 experts, reduce partial outputs and scatter to owners. Shared-expert evaluation uses owner-local rows. mHC then returns the result to the four-stream residual.
   - **Target head:** mean the mHC streams, normalize, expose the 4,096-value hidden row and project to the full 154,880-token vocabulary. Owner-local backend sampling avoids the old PAD gather.

4. **Generate MTP proposals.** The single NextN head uses token embedding plus the preceding target/kept hidden row. It is an ordinary DSA/MoE block with a plain residual, no mHC mixer and no KDA cache. Depth two requires up to two batched proposal decodes; the 98,304-token shortlist applies to the draft projection, not to the target vocabulary.

5. **Verify and catch up.** A fresh full-depth cycle verifies `[last sampled token, draft 1, draft 2]` in one target decode. Before deciding acceptance, the server also decodes **all those rows through the MTP head** to catch it up. Rejected proposal rows therefore consume catch-up work too. For four sessions: eight proposal rows, twelve target rows and twelve catch-up rows; the head is not simply run twice and then forgotten.

6. **Accept, roll back and deliver.** Target sampling accepts a matching prefix plus one target token. Ordinary MTP2 requests two recurrent rollback snapshots. Tail rollback restores the accepted recurrent/KV boundary; contexts requiring full restore can restore target, draft, MTP, prompt and sampler checkpoints and replay. Grammar or reasoning-budget activation can detach backend sampling and stop verification early. The client then receives accepted tokens, subject to EOS/output limits.

7. **Continue or cache.** The kept hidden row and valid draft selection can carry into the next cycle. A selection is reusable only within its active round, with compatible extent/width, consecutive positions, unique sequences and no completed-pool-boundary crossing. Slot mutation, restore and regrouping invalidate the appropriate state. On idle/admission, prompt snapshots may preserve a history for a later turn.

Sources: `tools/server/server-context.cpp:3288-3723,4292-4719`, `common/speculative.cpp:1597-1672,1783-2293`, `common/sampling.cpp:617-936`, `src/models/glm5next.cpp:262-655,657-953`, `src/llama-kv-cache-kpool.cpp:1569-1624`.

### MTP cost that a token-only model misses

For active count A, fixed draft depth D and accepted count a:

- Proposal rows: `A * D`.
- Target verification rows: `A * (D + 1)`.
- MTP catch-up rows: `A * (D + 1)`, if MTP is enabled.
- Delivered tokens: `A * (a + 1)`, before EOS/budget truncation.
- Logical target MoE collective phases: 84; proposal phases: `2D`; catch-up phases: 2. Full MTP2 gives 90 phases, not 90 CUDA kernel launches.
- Decode time per delivered token is `cycle time / (a + 1)` for each lockstep session. Cycle time includes proposal, target, catch-up, sampling, rollback and host work.

The device hidden-state store is double buffered, owner-aware, and normally engaged at 64 logical target rows. It does not eliminate all host reads: before acceptance, the current code retains every one-to-three verification hidden rows. Each row is 4,096 F32 values, or 16 KiB. Deferred accepted-row extraction is a smaller and more concrete target than treating all MTP state as already GPU-only.

The simulator advances logical delivered history. It does not assert that the final sampled token has already been evaluated into every cache; that token can be the input to the next decode. It does not predict token IDs, expert IDs, acceptance, mixed-depth batching, EOS, grammar activation or replay frequency.

### Short-context paths must not be extrapolated silently

Indexer scoring is skipped while the effective KV extent is at most 2,051 cells. Direct sparse indices have an additional cutoff of `max(4096, min(query_rows_per_sequence, 8) * padded_selected_width)`. The normal 2,051-cell bound pads to 2,080. Host KV preserves the early dense/masked kernel path by staging the visible cache below that cutoff, rather than prematurely switching attention arithmetic. At 400k, the long sparse path is eligible; at 512 tokens it is not the same operation.

The simulation deliberately leaves its selected-host-byte envelope null below 16,640 context tokens, covering all those short-path cutoffs rather than returning a misleading sparse figure. For long contexts it includes target, proposals and catch-up, counts a conservative 2,080 rows per query, and excludes padding writes and current-cache writes. This is a transfer envelope, not actual selection overlap or staging-buffer allocation.

Sources: `src/llama-kv-cache-kpool.cpp:18-28,121-123`, `src/llama-graph.cpp:4373-4394,4517-4525`, `ggml/src/ggml-cuda/fattn.cu:1150-1221`.

## Cache and memory ledger

Q8_0 uses 34 bytes for 32 values. The existing capacity model gives 544 bytes per latent row and 408 bytes per indexer allocation row, across eleven target attention layers plus one MTP attention layer. MLA shares its latent representation between attention K and V; counting two independent 512-value histories would double-count it.

At the full 442,368-cell reservation:

| Per private slot, with MTP | MiB | Placement |
| --- | ---: | --- |
| Target plus draft latent history | 2,754 | Owner GPU, or pinned mapped RAM under host KV |
| Target plus draft indexer allocation | 2,065.5 | Owner GPU even with host KV |
| Combined history reservation | 4,819.5 | Before recurrent state, weights, metadata or scratch |

Sources: `scripts/glm53/capacity.py:8-13,33-37`, `docs/torad/glm53-dp.md:147-164`.

Consequences:

- 36 resident slots reserve **42.359 GiB per card** for these two histories alone, whether one or 36 slots are active.
- 40 resident slots reserve **47.065 GiB per card** for histories alone.
- 40 host-KV slots leave **20.171 GiB per card** of indexer history and reserve **107.578 GiB of host RAM** for latent history.
- These totals exclude model weights, recurrent/rollback state, convolution history, graph buffers, metadata, staging, the projector and driver allocations. They are not a fit verdict. The simulator's separate active-history subtotal counts only the selected active sessions; idle histories can also be occupied and are not assumed to disappear.
- The older full-pack record reports about 39,218 MiB of model weights per card. That is a historical allocation, not a current re-enumeration of every tensor.
- A 32 GiB prompt snapshot cache is not extra active GPU context. It stores recoverable idle histories; restoring them costs host/device transfer and state reconstruction.
- Non-unified private capacity cannot be lent freely between active sequences. Paging would improve utilization of partially filled slots, but cannot shrink forty unrelated full histories.

The existing capacity estimator fits two measured boots, keeps an empirical per-slot residual, adds staging and reserves headroom. Its 36-resident/72-host projections are not active 400k qualification. Its retained-graph staging term assumes three extra 32-row shapes; source changes retaining larger MTP shapes need actual reservation/peak evidence instead of treating that number as timeless.

Host KV currently stages the selected canonical Q8 rows for each attention operation. It is not an implemented persistent hot-row cache, pressure-triggered pager, physical prefix-sharing system or context-parallel attention engine.

### Recurrent state, convolution and rollback

With the retained 64 heads of width 128 and convolution width four, each KDA layer has:

- Matrix state: `64 * 128 * 128` Q8 values = 1.0625 MiB.
- Convolution history: `3 * (4 - 1) * 64 * 128` F32 values = 0.28125 MiB.

Across 34 layers this is 45.6875 MiB for one current state. Configured MTP2 allocates current plus two rollback planes: **137.0625 MiB per slot**, independent of whether its history is 512 or 400,000 tokens. Nine owned slots require 1,233.5625 MiB; ten require 1,370.625 MiB, consistent with the older approximately 1,371 MiB/card record. Allocator alignment and extra workspaces remain excluded.

The draft head has no KDA state. Changing runtime adaptive depth from two to one does not shrink the configured rollback allocation; changing the launcher's maximum draft depth is a different experiment.

Sources: `src/llama-hparams.cpp:183-218`, `src/llama-memory-recurrent.cpp:32-35,114-127`, `src/llama-model.cpp:2998-3074`, `common/common.h:418-423`.

### Cache is several different mechanisms

| Mechanism | What it saves | What it does not save |
| --- | --- | --- |
| Latent attention history | Re-projecting old tokens | Index scoring and selected attention for the next query |
| Indexer history and compressed pools | Rebuilding old index keys | Scanning/scoring the candidate pool set |
| KDA recurrent state and convolution history | Replaying the whole recurrent prefix | Current recurrent updates and rollback bookkeeping |
| Resident pool metadata and MTP selection bank | Re-uploading metadata or downloading reusable selections | Validity checks after mutation, restore or a pool boundary |
| Idle prompt snapshots in host RAM | Re-prefilling a retained history | Snapshot serialization, restore traffic and admission latency |
| Retained llama/Meta/CUDA graphs | Rebuilding, allocating and launching equivalent work | The kernels and collectives in the replayed graph |
| GPU L2 | Some repeated state/weight/cache accesses | Guaranteed residency across many unrelated sequences |
| Host-KV staging scratch | Bounded GPU storage for an attention call | Repeated selected-row transfers between calls |

The expensive operation is different in each row. Increasing one cache cannot substitute for fixing another. Stable client prefixes matter: preserve prior reasoning, tool-call IDs, tool results and template settings. A change near the beginning of a 400k prompt can turn a warm decode request into a large prefill. The launcher does not repair a rewritten client history.

## Evidence boundary

The local machine exposes an RTX 5080 and RTX 5070 Ti, not the requested four-card host. No new GPU run was started.

The November 4 record used four 600 W RTX PRO 6000 Blackwell Server Edition cards, 97,887 MiB each. All GPU pairs were on one NUMA node through PCIe; the topology log reports NODE rather than NVLink. Measured directed 64 MiB peer transfers were 53.7-55.3 GB/s. This is not the small-message latency, aggregate four-card bisection bandwidth, or mapped-host bandwidth.

On runtime `82ad56cb8`, 40 allocated host-KV slots with MTP2 delivered about 45.3 native decode tokens/s for one 512-token request, 35.2 at 4,096 and 31.7 at 24,576. With 512-token prompts and 64-token responses, two/four/eight active sessions measured about 28 / 12-16 / 6 tokens/s per request. The common HTTP-interval rates were 33.8 / 30.9 / 30.3 tokens/s including prefill. These are different denominators and must not be added together.

Collectives took 73-75% of summed kernel duration across the GPUs in two short traces, but only about 38-40% on GPU0. Much of the other ranks' collective time was waiting. This is not an Amdahl fraction or a claim that removing collectives produces a 4x speedup. Graph instantiation and synchronization were also observed. Source changes after that runtime must be measured before assigning them a speedup.

Historical single-session 395k decode was 37.9 tokens/s, with cached follow-up turns at 42.1 and 46.9. The twelve-session cold 400k attempt did not complete a steady concurrent decode qualification, and a response alongside eleven prefills was about 2 tokens/s. Do not use the single-session result as a concurrent forecast.

Sources: `docs/torad/glm53-dp.md:261-275,333-384`, `docs/torad/glm53-dp-profile-20261104.json`, and the retained `topology.log` under `.Codex/tmp/dp-profile-20261104/receipts/results/`.

## Offline model

The executable is `scripts/glm53/simulate_decode.py`. It does byte/row/ownership accounting, not transformer arithmetic or CUDA cycle simulation. It deliberately returns a null tokens/s forecast and false qualification fields. Latency requires current per-phase traces, acceptance distributions and topology-specific measurements.

```sh
python3 scripts/glm53/simulate_decode.py --slots 36 --active 4 --context 400000
python3 scripts/glm53/simulate_decode.py --slots 40 --active 40 --context 409600 --host-kv
python3 scripts/glm53/simulate_decode.py --slots 36 --active 4 --prefill-sequences 36
python3 scripts/glm53/simulate_decode.py --slots 4 --active 4 --acceptance-pattern 0,2,1
python3 -m unittest discover -s scripts/glm53 -p test_simulate_decode.py
```

Its tests start from owner, byte and token conservation, include generated input combinations, and exercise the actual CLI. `--sequence-ids 0,4,8,12` models four sessions concentrated on GPU0; `--active 4` models IDs 0..3 spread across the cards. `--depth 0` is a target-only accounting control, not a recommendation to remove the team's MTP requirement.

The acceptance-pattern option advances histories through successive fresh cycles. For four sessions and acceptance counts 0,2,1, it delivers 24 tokens across the box and extends each history by six tokens, while issuing 36 target rows and 24 proposal rows. Acceptance is an input, not a forecast. Reusing a previously rejected draft or hitting EOS can change actual work; those paths are described separately, not silently treated as fresh full-width cycles.

The model uses one equal-sequence verification microbatch. It rejects configurations requiring that microbatch to be split, rather than guessing the batch allocator's ordering. Its prefill subsection is a separate equal-sequence thought experiment, not a second set of live requests automatically scheduled beside the generation subsection.

### Work at 400k, MTP depth two

One target verification evaluates three rows per active session, not three delivered tokens. With one accepted proposal, the cycle delivers two tokens; with none, one; with both, three. Draft generation, catch-up, rejection and rollback still cost work.

| Active sessions | Target rows | Rows per owner when balanced | Target MoE input payload over 42 layers, MiB | Extra target-shape slot eligible |
| ---: | ---: | --- | ---: | --- |
| 1 | 3 | 3 / 0 / 0 / 0 | 1.969 | Yes |
| 4 | 12 | 3 / 3 / 3 / 3 | 7.875 | Yes |
| 8 | 24 | 6 / 6 / 6 / 6 | 15.750 | Yes |
| 12 | 36 | 9 / 9 / 9 / 9 | 23.625 | No |
| 36 | 108 | 27 / 27 / 27 / 27 | 70.875 | No |
| 40 | 120 | 30 / 30 / 30 / 30 | 78.750 | No |

The payload column is `42 * rows * 4096 * sizeof(float)`: one logical input row set per target MoE layer. It is not the total PCIe traffic, and excludes returned expert partials, draft passes, protocol scratch and synchronization. There are 42 target routed layers, not 45; the first three feed-forward blocks are dense.

A balanced four-session batch puts three verification rows on each owner, but those rows remain distinct sequence chains. A kernel threshold based on tokens per sequence is not satisfied merely because twelve rows exist globally.

### Why row dispatch alone is not a large traffic reduction

With 288 experts divided into four groups of 72 and eight distinct uniformly chosen experts per token, the probability that a rank is needed is

`1 - choose(216, 8) / choose(288, 8) = 0.90314456`.

A token touches 3.613 ranks on average, or 2.709 remote ranks relative to its sequence owner. Sending a row only to ranks with selected experts therefore saves about 9.7% of the three remote row deliveries in this uniform model, not 97.2% from the ratio 8/288. Actual expert skew may change the result. Native collectives, destination-only reduction, early local work and overlap can still help; this calculation rejects only the claim that sparse expert selection automatically means sparse device traffic.

### Graph retention and prefill are discontinuous

`src/llama-context.cpp:1757-1806` retains up to three extra eligible graph shapes, each with its own scheduler/input lifetime. Target eligibility normally stops at 32 rows; registered MTP draft shapes are a separate exception with allocation guards. At depth two the target boundary is between ten and eleven full-width active sequences. The main graph can still reuse one stable larger shape. Graph reuse also depends on topology, sampler bindings and owner distribution, not just the total row count.

At a 512-row prefill budget, equal splitting gives:

| Simultaneously batched prompt sequences | Tokens per sequence | KDA dispatch opportunity |
| ---: | ---: | --- |
| 4 | 128 | Chunked |
| 8 | 64 | Decay precompute |
| 16 | 32 | Decay precompute |
| 36 | 14 | Recurrent |
| 40 | 12 | Recurrent |

The path labels are dispatch eligibility, not benchmark results. A candidate is to admit prompt work in small owner-balanced cohorts while preserving an inter-token deadline for already-generating sessions. That may improve prefill efficiency but can increase queueing/TTFT for the prompts outside the cohort. Increasing the global microbatch instead can also increase graph/staging memory; a prior unbounded split-merging attempt failed with a 32,227 MiB allocation.

## Three stronger findings from the final source pass

### 1. There is a host-staged reduction band even on a fully P2P-capable box

This is the most concrete decode-path candidate.

The default F32 expert partials use BF16 wire data. The internal defaults are 128 KiB for peer LL, 16 KiB for host LL and 1 MiB for copy-engine reductions. Above peer LL but below the copy-engine threshold, the four-rank numeric reduction uses mapped **host** staging. The large peer-inbox path is restricted to the unsigned gather operation; it does not handle those numeric reductions.

For full-depth MTP2, one target MoE reduction has `R = 3 * active` rows and `R * 4096 * 2` BF16 wire bytes:

| Active sessions | Target rows | Reduction wire bytes/rank | Current default route, peer probes successful |
| ---: | ---: | ---: | --- |
| 4 | 12 | 96 KiB | Peer LL |
| 5 | 15 | 120 KiB | Peer LL |
| 6 | 18 | 144 KiB | Host-staged chunked kernel |
| 8 | 24 | 192 KiB | Host-staged chunked kernel |
| 12 | 36 | 288 KiB | Host-staged chunked kernel |
| 36 | 108 | 864 KiB | Host-staged chunked kernel |
| 40 | 120 | 960 KiB | Host-staged chunked kernel |
| 43 | 129 | 1,032 KiB | Peer copy engine, if allocated; host fallback otherwise |

The last row requires the host-KV launcher path's larger slot allowance. Copy-engine reductions are not CUDA-graph capturable, so simply lowering their threshold trades host-staged traffic for a different launch/event path. Raising the peer-LL threshold may be a useful controlled experiment, but one CTA, doubled LL packet traffic and larger inbox rings can offset its benefit. A peer-resident numeric mid-size path preserving existing wire rounding and accumulation order is the more direct implementation candidate.

This mechanism can hurt concurrency even when bulk P2P bandwidth is excellent. It is not yet a measured attribution of the historical slowdown. An F32-wire quality control changes both precision and the dispatch threshold in rows, so its timing is not a pure precision ablation.

Sources: `ggml/src/ggml-cuda/allreduce.cu:608-620,794-802,1046-1056,1626-1696,1813-1844`.

### 2. The gather sends zero-padded full buffers, not just the owned slices

For a global F32 hidden matrix of size `B`, each of four ranks holds a full buffer containing its own rows and zero elsewhere. The gather combines these with bitwise OR. The direct peer implementation exchanges each full buffer with the other three ranks: **12B logical payload**, before LL flag overhead. A native owner-slice all-gather requires **3B**, a 75% logical-payload reduction for this part of the layer.

At 40 MTP2 sessions, `B = 120 * 4096 * 4 = 1.875 MiB`. Across 42 target MoE layers:

- Current full-buffer peer gather: **945 MiB** of logical peer payload per target verification.
- Owner-slice gather: **236.25 MiB**.
- Difference: **708.75 MiB per target verification**, before draft/catch-up work.

This is a different opportunity from routing each token to fewer expert ranks. The uniform-routing argument against a huge sparse-dispatch win does not apply to eliminating zero padding. The exact gather can preserve F32 words without arithmetic; unequal/empty owners, graph-captured replay and buffer lifetime still need correct handling.

The reductions still produce a full result on each rank before local scatter. Destination-only reduction is another possible saving, but changing reduction grouping can change floating-point results. Do not count its benefit without a design and quality check.

Sources: `ggml/src/ggml-cuda/allreduce.cuh:31-37`, `ggml/src/ggml-cuda/allreduce.cu:421-575,1626-1696,1975-1994`, `src/models/glm5next.cpp:612-655`.

### 3. Compact the pooled-key plane before offloading more latent KV

The 408-byte indexer row is **three 128-value Q8 heads**, not a 384-value scoring key:

1. Raw normalized index key.
2. An independently projected compressor gate.
3. A pooled key, written at a pool representative.

The scorer reads only the third 136-byte head at each representative. A pool covers four tokens. Keeping raw keys and gates for every token but allocating pooled keys densely per pool changes storage from `3N` heads to approximately `2N + N/4` heads. That saves **25% of the indexer allocation**, not 25% of the whole model.

With twelve target-plus-draft indexer layers, this is **516.375 MiB per reserved slot**, or **5.043 GiB per GPU at 40 balanced slots**. It also creates a contiguous pool-scoring layout instead of fetching representatives through a three-head token-cell stride. Improved effective bandwidth is a hypothesis; the storage delta is arithmetic.

This is not a drop-in tensor resize. The physical-token-to-pool mapping, compression writes, rollback, state serialization, restore and owner-local graph inputs all need the new representation. Keep raw keys and gates: `src/llama-graph.cpp:4361-4366` explicitly rebuilds all affected pooled keys after position mutation. An aggressive design that discards old raw compressor inputs is unsafe without a valid cold archive/rebuild mechanism.

At 400,000 tokens there are about 100,000 completed pools. Their unique pooled keys occupy **13.6 MB per attention layer**, before padding and metadata. Eleven target layers times three verification queries correspond to **448.8 MB of logical key/query data per session per cycle**. This is not measured DRAM traffic: heads, tiling, cross-query reuse, cache lines and repeated loads change physical reads. It is also one third of a calculation that incorrectly scans all three stored heads for each pool.

Sources: `src/models/glm5next.cpp:382-440,450-491`, `src/llama-model.cpp:3014-3028`, `src/llama-graph.cpp:4361-4366`.

## Optimization candidates to distinguish from work already done

### Already in this checkout; measure before crediting a gain

- Owner-local DP sampling and retained sampler graphs were merged in PR 119. The historical full-vocabulary PAD gather is not an unfixed recommendation.
- PR 120 removes terminal hidden getters, avoids unnecessary output permutations, prepares constant KDA convolution/indexer-APE layouts and retains DP selections in owner-partitioned storage.
- Earlier work already added resident metadata patches, direct sparse indices, refusal caching for idle prompt snapshots, coalesced hidden reads and larger registered draft graph shapes.
- None of those source changes is a measured concurrent 400k result. The old trace percentages are not their measured savings.

Sources: `docs/torad/glm53-dp-optimization-plan.md:82-130`, `docs/torad/glm53-dp.md:9-22`; the shared checkout advanced to merge `d54ea817b` while this analysis ran.

### Other candidates supported by the model

1. **Schedule by sequence-owner work, not only active-session count.** An early-arriving rank spends time inside a collective while a loaded owner finishes attention. Put new cold work on less-loaded owners and test owner-balanced prefill cohorts. Preserve cache affinity for warm requests; moving an existing 400k state just to balance a step can cost more than the imbalance. Measure per-rank arrival skew before changing the transport.

2. **Choose prefill cohorts around KDA's per-sequence thresholds.** Four 128-token chunks can enable a different algorithm from 36 fourteen-token chunks at the same total row budget. This is more specific than increasing batch size. Pair it with a generation deadline so an efficient prompt chunk does not starve ongoing streams. Compare TTFT and inter-token latency together.

3. **Treat MTP depth, batch shape and graph reuse as one scheduling decision.** The extra target graph limit creates a boundary at eleven MTP2 sessions. Retaining bounded target verification shapes, or making shape selection more stable, could avoid rebuilding around that boundary. Do not remove scheduler input-lifetime limits or set unlimited memory budgets: those limits previously prevented an actual OOM. Same-shape reuse above 32 already exists.

4. **Keep accepted hidden state on the GPU across small MTP cycles.** Removing the final unused getter is not the same as removing eager downloads and subsequent uploads of accepted rows. The design needs owned storage, row/sequence identity, stream ordering, and acceptance/rollback-aware invalidation. Retaining a pointer into reusable graph scratch is not sufficient. The small byte count does not rule out a synchronization benefit.

5. **Reuse exact staged latent rows, not approximate attention selections.** A persistent host-KV working set can avoid copying rows selected repeatedly in target verification, MTP and successive decode steps. Cache canonical Q8 bytes by sequence, layer, physical row and write generation; overwrite, restore, rollback, slot reuse and pool mutation must invalidate stale content. Measure the actual overlap/miss curve before choosing cache size or prefetching. Sparse attention bounds selected attention work, but it does not eliminate the long-history index scan.

6. **Make reduction destination-aware and overlap useful local work.** The final expert result is only needed by the owning rows. A real reduce-scatter and owner-local shared-expert work may reduce replicated intermediate traffic or waiting. Separate launch/barrier latency, rank arrival skew and payload cost. Bulk P2P bandwidth and collective kernel duration alone do not tell which one to optimize.

7. **Spend memory on active work rather than empty reservations.** Per-owner paged pools or immutable prefix pages could release unused private capacity. Prefix sharing must include exact token/template identity and the recurrent-state boundary; divergence needs copy-on-write and separate recurrent state. This improves partially filled or genuinely shared workloads, not forty independent full contexts. It is an architectural change, not an existing launch flag.

There is a second shape boundary at 64 target rows for device-resident hidden-state handoff. Full-depth MTP2 has 33..63 rows at **11..21 active sessions**: above the extra target-graph limit but below the device-hidden threshold. This does not establish a throughput valley, but it gives a precise reason to measure around 10/11 and 21/22, rather than only powers of two. The concurrent MTP policy also uses a configured cost prior: service-cycle cost calibration is restricted to isolated one-sequence work (`common/speculative.cpp:1662-1663,1783-1792`). Do not assume adaptive 0.18 automatically learns saturated DP contention.

The first three are scheduling/representation opportunities. The next three need explicit device-buffer lifetimes and traces. The last is primarily capacity and admission work. None is promoted as a measured speedup or implemented engine change in this task.

### Ideas that did not survive the evidence

- Splitting the box into four independent full-model replicas: the selected pack does not fit a single 96 GB card.
- Treating DP as two copies of two-card tensor parallelism: this engine's DP mode does not implement that layout.
- Assuming expert all-to-all removes almost all traffic: eight experts usually touch nearly every rank in the uniform placement model.
- Treating all collective kernel time as network transfer: waiting for dense work on another owner is included.
- Turning MTP off everywhere: the selected team configuration retains MTP, and historical long-context behavior differs from short-context behavior.
- Assuming a 400k capacity allocation proves 400k serving speed or quality: it proves neither.
- Promoting the older SM120 profile bundle into the Q8-state recipe: they have different contracts and validation scopes.

## Read coverage, gaps and verification

Source mapping covered the relevant text-inference path, not every architecture or CUDA template specialization in llama.cpp:

| Area | Source read |
| --- | --- |
| Launcher, cache sizing, experimental profiles | `scripts/glm53/serve.py`, `capacity.py`, `common/glm53-profile.cpp` |
| Architecture metadata and device configuration | `conversion/glm5next.py`, `src/llama.cpp`, `src/llama-model.cpp`, `src/llama-hparams.cpp` |
| Target/NextN graph, mHC, KDA, MLA/DSA, expert ownership | `src/models/glm5next.cpp`, relevant mHC helpers in `src/models/deepseek4.cpp`, `src/llama-graph.cpp` |
| Sequence allocation and persistent state | `src/llama-batch.cpp`, `src/llama-kv-cache.cpp`, `src/llama-kv-cache-kpool.cpp`, `src/llama-memory-hybrid.cpp`, `src/llama-memory-recurrent.cpp` |
| Decode, graph reuse, output and hidden stores | Relevant construction/reserve/decode/sampling/NextN paths in `src/llama-context.cpp` and its header |
| Server scheduling, admission, prompt snapshots and restore | `tools/server/server-context.cpp`, `server-task.cpp/.h`, `server-queue.cpp/.h` |
| Draft, catch-up, acceptance, replay and sampler constraints | `common/speculative.cpp/.h`, `common/sampling.cpp`, speculative parameter conversion in `common/common.cpp/.h` |
| Multi-GPU execution and kernels | Relevant Meta split/gather/reduce/scatter/sampling paths in `ggml/src/ggml-backend-meta.cpp`; CUDA expert dispatch in `ggml-cuda.cu`; `allreduce.cu/.cuh`, sparse dispatch and host staging in `fattn.cu`, KDA/GDN and mHC dispatch/fusion paths |
| Evidence | Tracked GLM serving/decode/DP documents, the November 4 profile JSON, retained topology/startup records and architecture metadata |

The source window began at `c75d0e787` with other sessions' pending changes and advanced to PR120 merge `d54ea817b`. The current recommendations account for that merge; the historical performance data is from `82ad56cb8` or explicitly labeled older runs.

Still missing for a performance guarantee:

- A current four-card run of the merged engine, with actual occupied 400k histories, not merely large allocations.
- Current per-rank critical-path timings and graph hit/miss attribution across the relevant row thresholds.
- Measured selection overlap, pool-scoring memory traffic, expert skew, acceptance and mixed-depth distributions.
- A current complete tensor/allocator/workspace enumeration for the selected Bolding pack.
- Full current long-context numerical/task qualification. The recorded cross-layout KL discrepancy does not meet the requested absolute 1e-6 bar.
- Projector kernel cost for real image shapes; this analysis models text decode after prefill, not a vision workload benchmark.

The CPU simulator is intentionally conditional: equal-length independent active histories, fixed full proposal depth, supplied acceptance, one verification microbatch, successful peer setup for its peer-route table, and no predicted token contents. Short dense/masked staging is identified but not assigned a sparse transfer number. Graph padding, allocator behavior, kernel fusion, cross-query reuse and CPU/network scheduling prevent its byte counts from becoming a wall-clock model.

Tests were written before each modeled behavior. The initial owner/memory/cycle contract failed with the unimplemented model. Later source-derived recurrent, pool, catch-up, transport and threshold checks failed before those fields/behaviors existed; the short-host case exposed the incorrect sparse envelope and was fixed. The final focused suite passed **23 tests**, including generated ownership/history cases and real CLI calls. The full documented Python command then passed **105 tests**:

```sh
python3 -m unittest discover -s scripts/glm53 -p 'test_*.py'
```

This checks the offline tool and existing Python tooling, not the C++ inference engine. No C++ suite or GPU qualification was rerun. The existing C++ architecture-suite failures documented in `docs/torad/glm53-dp-optimization-plan.md:118-124` are not represented as resolved by this task.

## Three-pass synthesis and next experiment order

**Survived:** sequence-owned DP, separate latent/indexer/recurrent scaling, mandatory pre-accept catch-up, implemented sampling/selection locality, the mid-sized host-staging band, full-buffer gather overhead, and the compact pooled-plane storage opportunity.

**Rejected or corrected:** four-way per-sequence tensor splitting; treating 400k as a session count; scanning all three indexer heads during scoring; counting MTP cache twice; treating every short host-KV call as sparse; treating old sampling-gather traces as an unfixed current path; converting summed kernel percentages into a speedup.

**Highest risk:** optimizing the short-prompt coordination trace and then assuming it predicts occupied 400k behavior. Long-history pool scoring and active cache/graph growth can change the limiting resource, while the existing numerical qualification gap remains open.

Confidence in the configuration/source accounting progressed from 85% to 90% to 95% as the source contradictions and lifecycle omissions were resolved. This is a qualitative assessment of the map, not statistical confidence or confidence in a throughput forecast.

A bounded next engineering sequence would be:

1. Establish a current same-pack baseline and expose which collective route each target/proposal/catch-up shape selects. Preserve F32-wire quality controls separately from the ordinary BF16 recipe.
2. Compare a capture-safe peer numeric path for the 128 KiB..1 MiB reduction band. Preserve wire precision and accumulation order so transport is the tested variable.
3. Replace zero-padded full-buffer gathering with owner-slice exchange, including uneven and empty ranks and captured replay.
4. Prototype compact pooled-key storage with raw key/gate preservation. Test append, pool completion, rollback, fork, restore and position mutation before crediting the memory saving.
5. Measure target shape retention, accepted hidden-row lifetime and KDA-aware prefill cohorts together at the relevant concurrency boundaries.
6. Only after useful short-context scaling, repeat with genuinely occupied 400k histories, warm follow-up turns, concurrent cold admission, and long-context quality checks.

No items in this list were applied to the engine or authorized as a paid hardware run.
