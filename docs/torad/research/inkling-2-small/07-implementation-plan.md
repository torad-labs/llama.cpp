# 07. Implementation and qualification plan

This is a large feature by the project's standards and should be run as a campaign: items
with specs and checks that fail today, built in order, each phase green before the next
starts. Time ranges below are planning figures for one focused seat, given that the
upstream implementation exists to copy from. They are estimates, not commitments, and the
box rental in Phase 4 is a spending decision for Marcos.

## Phase 0. Pin the inputs (half a day)

- Pin upstream at `5c008e000` plus `dd099a6bf`. Record any later upstream Inkling fixes at
  branch start and re-check before the box read.
- Pin the weight source at `benthecarman/Inkling-2-Small-abliterated` revision
  `e4853745aa9cb4587e8959d3c80b92e1bef99a4a`, and record SHA-256 hashes for every file at
  download. Stage stock Unsloth `UD-Q3_K_XL` only as an engine bring-up and comparison
  input. The served packs are locally converted Q3-class and IQ3-class artifacts from the
  abliterated source; see [09-abliterated-target.md](09-abliterated-target.md).
- Checks that fail today and define done for the campaign: the converter rejects the
  checkpoint config, the engine rejects the `inkling2` arch, and the server cannot render
  the model's chat template with an image.

## Phase 1. Correct baseline port (3 to 6 days)

Port in dependency order, keeping upstream's structure and comments where they carry
reasons:

1. GGUF constants, tensor mapping, writer keys; arch registration; hparams fields
   including the two generic router controls; model class and tensor pointers; vocab
   pre-tokenizer; quantize exceptions; model saver.
2. KV cache relative-position setter; `build_attn` explicit mask and per-query sink
   handling; router precision in `build_moe_ffn`; graph node budget for the arch.
3. `src/models/inkling2.cpp` and its declarations.
4. CUDA per-head mask support across vec, tile and MMA paths, re-applied onto this fork's
   flash-attention code by reading both sides. The shared-mask optimizations must decline
   exactly where upstream declines them.
5. Converter and mmproj converter; mtmd projector, preprocessor and subcontext positions;
   chat media-block split.

Acceptance:

- Upstream's new tests pass in this tree: `tests/test_conversion_fp8.py` Inkling classes,
  the `test-llama-archs` fixture, `test-mtmd-impl` preprocessor cases, `test-chat` media
  blocks, and the `test-backend-ops` per-head mask family on CUDA.
- A logits oracle: the synthetic arch fixture, run in this fork and in a pinned upstream
  build on identical random weights, agrees on prefill and decode logits within upstream's
  own fixture tolerance and on every argmax.
- The project's full check is green. Nothing in this phase touches GLM behavior; the GLM
  served-shape tests are the regression net.

## Phase 2. Tensor split and state correctness (2 to 4 days)

- Walk every tensor in the arch's tensor list through
  `llama_meta_device_get_split_state` and write the split decision down in code, with
  special rules where the standard table has no analog: relative projection and bias,
  clip scales, sink projection, dynamic convolution projections, convolution kernels,
  residual queries, the recurrent state, and both KV caches.
- Assert the decisions that capacity depends on: global and sliding KV split by heads,
  never mirrored; convolution state split consistently with its streams.

Acceptance:

- New cases in the meta split tests cover the Inkling tensor names and cache names.
- A two-rank synthetic run matches one-rank logits within tolerance for prefill, decode,
  chunked prefill across at least two chunk boundaries, sequence copy, state save and
  restore, and rollback after a rejected speculative-style step. The convolution state is
  the thing these cases exist to catch.
- Graph reuse holds at fixed row and sequence counts while the pool grows, or the failure
  is understood and scheduled into the Step 1 kernel work of
  [06-optimization-roadmap.md](06-optimization-roadmap.md) before the box read.

## Phase 3. Abliterated packs, server and launcher parity (2 to 3 days)

- Convert the pinned abliterated checkpoint and its vision projector, generate the
  importance matrix from that reference, and produce the Q3-class default and IQ3-class
  capacity packs. Record source revision, calibration corpus, recipe and artifact hashes
  with every pack. Stock Unsloth artifacts never become a launcher default.
- `scripts/inkling2/serve.py` with the profiles from
  [05-capacity-and-tensor4.md](05-capacity-and-tensor4.md) and pinned dry-run tests in the
  style of `scripts/glm53/test_qualification.py`.
- Chat qualification against the official template: text turns, system prompts, reasoning
  effort values including an invalid one returning a client error, tool declarations,
  tool calls with nested arguments, tool results, thinking content round-trip, and image
  parts before, between and after text.
- Prompt cache with hybrid state: save and restore mid-conversation, restore under a
  different slot, and prefix reuse that must refuse or snapshot correctly where
  convolution state is involved.
- Sampling defaults and the unpadded vocabulary: no sampled id at or above 200,058, EOS
  behavior at `<|content_model_end_sampling|>` and token 200,006 verified in server tests.

Acceptance: server unit tests for the above, dry-run plans pinned, and a CPU or
single-card smoke of the synthetic fixture through `llama-server` including one image
request against a synthetic mmproj. Pack qualification also compares stock, the abliterated
reference and each quantized abliterated pack on one pinned behavior set, so conversion and
quantization are shown to preserve the selected refusal profile without breaking coding or
tool responses.

## Phase 4. The four-card box read (one rental)

Hardware gate first: four RTX PRO 6000 with peer access between cards. Check
`nvidia-smi topo -m` before anything else. A host whose cards cannot map each other's
memory reads a different engine, as the GLM reads established. Stop the rental the moment
a leg disproves its layout rather than tuning on rented time.

1. **Route.** Backend meta all-reduce bench at 4 ranks for 23-row and 69-row reductions
   at width 4,608. This picks the reduction route default for the read.
2. **Load.** Tensor-4 profile dry run, then launch. Record free MiB per card. Under
   4,096 MiB free, drop one pool step, then the pack, recording each miss before reading
   any rate.
3. **Rate.** Sessions at 1, 2, 4, 8, 16 and 23, short history, then at about 4,000 tokens
   of history. Per-session tokens per second is the number.
4. **Short turns.** Repeat the sweep with `--session-step` and `--prompt-shapes` in the
   GLM pattern, counting graphs built and replayed.
5. **A read beside a stream.** One long reply streaming while another session reads about
   200,000 tokens. Read the streamer's longest chunk wait, then again with
   `--prompt-step-ms` set near twice the measured step time.
6. **Depth.** Fill the pool with sessions at the slot limit at once. No cache lost, peak
   VRAM per card recorded. This leg is the capacity claim; rates without it are not the
   goal read.
7. **Images.** Interleaved image and text requests while sessions decode: encode latency,
   correctness on a screenshot and a chart prompt, multi-image ordering, and an image at
   the token limit.
8. **Quality.** Greedy continuations on a fixed prompt set under the chosen abliterated
   pack against the abliterated quality-profile pack at a small pool; a needle probe at the
   slot limit; long-document completion at 2,048-token and 8,192-token reply budgets, with
   unfinished answers counted separately; the reduction wire leg (BF16 against 8-bit) and
   the Q8_0 KV leg if the clock allows, each reported as agreement rates, no more.

Write every leg's result into the launcher docs with the box's card variant, host CPU and
topology, then destroy the box.

## Phase 5. Optimization (1 to 3 weeks, measured order)

Follow [06-optimization-roadmap.md](06-optimization-roadmap.md): census, relative-attention
kernel, prologue fusions, convolution kernel, MoE gates at real shapes, graph shapes, KV
precision. Each item lands with its backend-ops or arch-fixture coverage and a before and
after step budget at the row counts from Phase 4. Items that do not move a served row
count are dropped, however elegant.

## Phase 6. DSpark (1 to 2 weeks, optional and last)

- Hash the abliterated repository's `mtp.safetensors` against the stock file, then add
  converter support for the pinned file into a draft GGUF, with its own arch or a clearly
  separated sub-model representation.
- Draft graph: five layers with RoPE, one global layer, dense MLPs, input from target
  hidden states at layers 5, 11, 17, 23, 29, 35, 41 and 48, and the Markov RNN head.
- Target graph taps at those depths without perturbing the non-speculative path's shapes.
- Server coupling: draft 7, verify 8, rollback of target convolution state, draft KV and
  Markov state on rejection; acceptance-length telemetry per session; an adaptive depth
  control by analogy with GLM's, driven by measured acceptance.
- Acceptance: acceptance length and per-session rate at 1, 8 and 23 sessions against the
  Phase 5 baseline, with speculation off remaining a first-class profile. If acceptance
  does not pay at served concurrency on this interconnect, DSpark stays off by default
  and the phase still counts as answered.

## What done looks like

Parity with GLM Tensor 4, concretely: one launcher command brings up 23 reserved agent
sessions on four cards from a pinned pack, with images, tools, reasoning effort, prompt
cache, chunked prompt reads that do not stall live replies, and a recorded box read behind
every default. Anything less is a named phase above, not a surprise.
