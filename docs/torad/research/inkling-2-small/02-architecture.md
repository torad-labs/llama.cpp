# 02. Architecture

This file describes the computation the engine has to perform. It follows the published
`config.json`, the Hugging Face architecture write-up, NVIDIA's NeMo model reference, and
upstream llama.cpp's graph in `src/models/inkling2.cpp` at `upstream/master`. Where the
prose sources and the code could differ, the code wins and is cited.

## Text backbone at a glance

- 48 decoder layers, hidden size 4,608.
- Every layer is MoE. There are no dense feed-forward layers and no shared experts.
- Layers are grouped into 8 residual blocks of 6 layers. The attention pattern lines up
  with the blocks: five sliding-window layers and one global layer per block. The global
  layers are 5, 11, 17, 23, 29, 35, 41 and 47.
- Token embeddings are RMS-normalized before layer 0. Raw embeddings that arrive from the
  vision encoder skip that norm, which upstream encodes explicitly in the graph
  (`src/models/inkling2.cpp:213-217` at upstream).
- The output head has 201,024 rows. Logits pass a tanh softcap at 30 and are then sliced
  to the 200,058 usable token ids, so padded rows can never be sampled.

## Block attention residuals

There is no single residual stream. The network keeps one running sum per six-layer block:

- The normalized token embeddings are the first block summary.
- Inside a block, attention and MoE outputs add into the current partial sum as usual.
- At the first layer of the next block, the finished sum is frozen as a summary and a new
  partial sum starts from that layer's attention output.

No sublayer reads a sum directly. Before each attention and each MoE, the model mixes over
depth: it RMS-normalizes every finished summary plus the current partial sum, scores them
against one learned query vector for that sublayer, softmaxes over at most nine sources,
and takes the weighted mix. Sliding-window layers learn one query shared by their Q, K and
V inputs. Global layers learn three queries, so Q, K and V can each read a different mix.
The final norm reads a mix the same way, through its own query.

Upstream implements the mix directly with reshape, concat, RMS norm, a small matmul, a
softmax and a second matmul (`mix` at upstream `src/models/inkling2.cpp:154-164`). Cost per
mix is small next to the projections, but there are 96 mixes in a forward pass and each one
reads up to nine hidden-width tensors per token. The summaries also stay live for the whole
graph, which the compute-buffer sizing has to allow for.

## Attention

Common shape in all layers: 48 query heads, 8 KV heads, head dimension 128, grouped-query
ratio 6. Five details make this attention unlike the fork's current models.

1. **Relative position instead of RoPE.** A fourth projection produces a 16-wide relative
   feature R per query head per token. A learned table per layer maps distance to a
   16-vector, for distances 0 to 1,023, with one extra slot for anything farther or
   invalid. The attention logit gains the dot product of R with the table row for the
   query-key distance. Both sliding and global layers use extent 1,024, so a global layer
   sees exact relative position only within the nearest 1,024 tokens and a single learned
   value beyond that.
2. **L2 clipping instead of QK normalization.** After an optional learned per-head scale on
   Q and on R, any head vector whose L2 norm exceeds a threshold is scaled down to it.
   The QK threshold is 22.6274 and the relative threshold is 8.0. Values are RMS-normalized
   per head with no learned gain.
3. **Per-token attention sink.** Each token predicts one extra logit per head. The sink
   joins the softmax denominator and contributes no output, so a head can attend to
   nothing. In the flash-attention path upstream folds the sink into the bias algebraically
   and passes a zero sink vector to the kernel (`src/models/inkling2.cpp:265-279`).
4. **Gated output.** The attention result is multiplied elementwise by a sigmoid gate
   projected from the attention input, before the output projection.
5. **Log-length scaling in global layers.** Q and R are scaled by a factor that grows with
   position: 1 below 128,000 tokens, then 1 + 0.1 * ln(position / 128,000). At the full
   1M context the factor is about 1.21.

Sliding-window layers attend causally within 1,024 tokens and share one input norm. Global
layers attend to the whole sequence and use separate input norms for Q, K and V.

## Short convolutions and their state

Each layer runs four causal depthwise convolutions with kernel size 4: on K, on V, on the
attention output after its projection, and on the MoE output. Each output mixes the current
position with the three before it, adds the pre-convolution value back, and rescales by
1/sqrt(5), or 1/sqrt(9) where a dynamic term is present.

In global layers the K and V convolutions are dynamic. A small projection predicts four
extra taps per KV head per token, added to the static kernel. Sliding layers use the
static kernel only.

The convolutions make every layer stateful across steps. The state per layer is the last
three positions of four streams: 1,024 K channels, 1,024 V channels and two 4,608-channel
hidden streams, 11,264 channels in all. At F32 that is 6.19 MiB per sequence across 48
layers, small next to the KV cache but semantically part of the sequence. It must be
saved, restored, copied and rolled back exactly like recurrent state, including on
speculative rejection and prompt-cache restore. Upstream stores all four streams in one
recurrent state tensor per layer at fixed channel offsets
(`convolve` at upstream `src/models/inkling2.cpp:166-191`).

## Mixture of experts

- 384 routed experts per layer, top-8 by sigmoid router score plus a learned per-expert
  bias used for selection.
- Router logits are computed in F32. Selected weights are normalized by their sum plus
  1e-20, then scaled by `route_scale` 8.0.
- Expert intermediate size is 2,304. The activation is a softcapped SwiGLU, exposed in
  llama.cpp as `LLM_FFN_SITU` with gate softcap 4.0 and up softcap 25.0.
- Packed gate and up weights are stored interleaved in the checkpoint and de-interleaved
  during conversion.
- Expert parameters dominate the model: 12.23B per layer, 587.1B in total, against roughly
  6.6B for everything else in the text model. Any fit or speed question is mostly a
  question about these tensors.

## Numerics of the native checkpoint

Only the routed experts are quantized in the released checkpoints. Layers 0-39 store NVFP4:
4-bit values, an E4M3 block scale per 16 values, and a per-expert outer scale given as an
amax over a shared denominator. Layers 40-47 store MXFP8: E4M3 values with a power-of-two
UE8M0 scale per 32 values. The all-MXFP8 repository stores every expert the second way.
Attention, embeddings, norms, router and the small Inkling-specific tensors stay in
floating point. vLLM's implementation notes also record reference activation quantization
inside the MoE path, including per-token NVFP4 activation scales, which matters only if we
later pursue a native-precision path instead of a GGUF pack.

## Vision tower

- 23-layer ViT, hidden size 1,024, 16 heads, MLP ratio 4, about 292M parameters.
- Images are split into 20x20 patches. Learned position embeddings on a 32x32 grid are
  bilinearly resized to the image, and 2D axial RoPE with theta 10,000 is applied inside
  the tower. A class token with its own position embedding is prepended, run through the
  encoder, and discarded.
- Neighboring 2x2 patch features are merged, so the language model receives one embedding
  per 40x40 pixel region. A two-layer GELU MLP projects to the 4,608 hidden size.
- Preprocessing converts to RGB, rescales by 1/255 and normalizes with mean and std 0.5.
  Images are padded to a multiple of 40 pixels rather than resized, unless they exceed the
  configured token budget. The model card gives 40 to 4,096 pixels per dimension as the
  optimal range.
- The config carries `temporal_patch_size: 2`. Nothing in the published inference paths
  found for this study exposes video input, and upstream llama.cpp implements still images
  only. Treat video as out of scope until a reference path says otherwise.

## DSpark draft head

The checkpoint bundles a separate draft model in `mtp.safetensors` at 3.42 GB. Its config
describes a different network from the target:

- 5 layers, hidden 4,608, 48 query heads, 8 KV heads, head dimension 128.
- Layers 0, 1, 3 and 4 are sliding-window at 1,024; layer 2 is global.
- Dense MLP with intermediate size 18,432. No experts.
- RoPE with theta 1,000,000, unlike the target's relative bias.
- It reads target hidden states from layers 5, 11, 17, 23, 29, 35, 41 and 48, which are
  exactly the global layers plus the final output.
- `block_size` 8 and a Markov head of type `rnn_hidden` with rank 256. Frameworks draft 7
  tokens per step and verify 8 rows including the bonus position. vLLM reports an average
  acceptance length of 3.5 and supports probabilistic drafting.

Nothing in the draft head resembles the GLM MTP head this fork serves today. It needs its
own converter output, its own graph, a path for target hidden states at eight depths, its
own KV handling, and rollback of both the target convolution state and the draft's Markov
state on rejection.
