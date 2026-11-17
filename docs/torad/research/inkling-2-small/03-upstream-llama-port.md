# 03. The upstream llama.cpp port

Upstream merged Inkling 2 support in `5c008e00085f4d3e394f9a1be649284999de1326`
(PR #32217), authored 2026-11-15 00:51 +0100, which is 2026-11-14 17:51 CT. A follow-up,
`dd099a6bf` (PR #32294), fixed image token placement in chat. Local `upstream/master` was
at `a946eb541` when checked on 2026-11-16. This file maps what those commits contain so the
port into this fork can be planned as re-application rather than rediscovery.

The headline commit changes 45 files with 1,667 insertions and 30 deletions. Of those 45
files, 33 also differ between the commit's parent and this fork's HEAD. The new files should
land nearly as-is. The shared files, especially the CUDA flash-attention set, need hand
work.

## What upstream supports after the two commits

- Text inference for the `inkling2` arch, including flash attention.
- GGUF conversion from the Hugging Face checkpoint for the text backbone, including the
  packed NVFP4 and MXFP8 experts.
- A separate mmproj conversion and a still-image encoder in mtmd.
- Chat rendering through the embedded Jinja template, with a fix that gives each media
  marker its own message block.
- Not supported: DSpark in any form, audio, video, LoRA specifics beyond the generic
  `build_lora_mm` call sites the graph already uses.

## Conversion

New file `conversion/inkling2.py` (327 lines), registered for both
`Inkling2ForConditionalGeneration` and `Inkling2ForCausalLM`.

Facts that matter for us:

- It logs a warning that it converts the text backbone only. Vision and MTP tensors are
  skipped by the text converter and handled, for vision, by a separate mmproj class in the
  same file. MTP tensors are dropped entirely.
- Packed experts are dequantized during conversion, streamed 128 rows per expert through
  `LazyChunkedTensor`. The real gate/up tensor of one layer is 384 x 4,608 x 4,608, which
  would expand to 30.375 GiB in F32; the chunked path needs about 16 MiB per projection
  plus roughly 370 MiB of chunk metadata for the whole model.
- The dequantizer validates companions strictly: NVFP4 requires scale, amax and outer
  scale denominator tensors with exact dtypes and shapes; MXFP8 requires the scale only
  and rejects the NVFP4 companions. NaN codes in either scheme are errors.
- The declared MXFP8 block shape in `hf_quant_config.json` conflicts with the stored
  shapes. The converter warns and trusts the stored shapes, giving an effective block of
  1 x 32 for MXFP8 and 1 x 16 for NVFP4.
- Interleaved gate/up rows are de-interleaved by slicing weights and their scale rows
  together, for packed and floating-point inputs alike.
- Small numerically sensitive tensors are forced to F32 in the GGUF: convolution kernels,
  clip scales, residual-mix queries and the relative bias projection. The same rule exists
  on the requantize side in `src/llama-quant.cpp`.
- The tokenizer is written as a GPT-2 style BPE with a dedicated `inkling2` pre-tokenizer,
  sliced to the 200,058 unpadded tokens. EOS and EOT are both token 200,006.
- The commit message is explicit about validation limits: synthetic packed weights match a
  pinned vLLM decode bit-for-bit, and CPU prefill and decode logits agree within 3.1e-6
  with full argmax agreement on fixtures, but conversion of the real 393 GB checkpoint was
  not tested upstream. Unsloth's published GGUFs are evidence a real conversion works, but
  they were produced with their own toolchain. Our first conversion run is still a
  verification event, not a formality.

Because conversion dequantizes to floating point and then applies normal GGUF output type
selection, upstream's converter does not preserve the native NVFP4 and MXFP8 encodings in
the GGUF. There is also no MXFP8 tensor type in ggml today. Any native-fidelity goal would
need new type and kernel work; the pack strategy in
[05-capacity-and-tensor4.md](05-capacity-and-tensor4.md) assumes requantization instead.

## GGUF and architecture registration

- `gguf-py/gguf/constants.py` gains the `inkling2` arch, its tensor list, and the
  `inkling2v` vision projector type.
- `gguf-py/gguf/tensor_mapping.py` gains checkpoint and Transformers names for every text
  and vision tensor, including the relative, sink, clip, convolution and residual-query
  tensors.
- `gguf-py/gguf/gguf_writer.py` gains writers for the new hyperparameters: padded and
  unpadded vocab sizes, relative length and extent, QK and relative clip thresholds,
  log-scale floor and alpha, attention-residual block size, and the SITU softcaps through
  existing activation keys.
- `src/llama-arch.h` and `src/llama-arch.cpp` register `LLM_ARCH_INKLING2`, its keys and
  its tensor names. `src/llama-model.cpp` maps the arch to the model class, marks its
  RoPE type as none, and creates memory with every layer in both the attention and the
  recurrent filters.
- `src/llama-hparams.h` gains the Inkling fields plus two generic router controls,
  `expert_router_f32` and `expert_weights_norm_eps`. `src/llama-hparams.cpp` teaches
  `n_embd_r()` to size the convolution state from the channel count.
- `src/llama-model.h` gains 16 per-layer tensor pointers and the final residual query.
- `src/llama-model-saver.cpp`, `src/llama-quant.cpp` and `src/llama-vocab.cpp` gain the
  matching save, quantize-exception and pre-tokenizer handling.

## The model graph

New file `src/models/inkling2.cpp` (315 lines) plus its class declarations in
`src/models/models.h`. Load-time validation is strict about dimensions, finite positive
scales, the sliding pattern length and the tokenizer size. The graph itself uses existing
building blocks throughout: hybrid ISWA memory input, recurrent state read and store,
`ggml_ssm_conv` for the static convolution, standard `build_moe_ffn`, and `build_attn`.

Three shared-code changes the graph depends on:

1. **Relative position input.** A new graph input holds an I32 distance matrix per cache,
   sized KV cells by tokens, filled on the host by a new
   `llama_kv_cache::set_input_relative_pos`. Distances outside 0..extent-1, and empty
   cells, map to the sentinel index `extent`. The graph projects R against the bias table
   first, then gathers by distance with `ggml_get_rows`, so only the gathered result is
   KV-sized. Graph reuse requires the KV count and token count to match exactly.
2. **Per-query sinks and an explicit mask.** `build_attn` gains an optional mask argument.
   With flash attention, the gathered bias and the causal mask are added and cast to F16
   as a per-head mask, and the sink is folded into the bias before the gather. Without
   flash attention, the sink is concatenated as an extra softmax column and sliced off.
3. **Router precision.** `build_moe_ffn` gains the F32 router accumulation and the
   sum-plus-epsilon weight normalization behind the two new hparams flags.

`src/llama-context.cpp` also raises the graph node budget for the arch to
`max(1024, 32 * n_tensors)`. The graph is node-heavy: 96 residual mixes, 192 convolutions
with state stores, three clips and a value norm per layer, and per-layer bias projection and
gather. Node and launch counts are a first-class concern for kept-graph capture in this
fork, where `GGML_CUDA_GRAPH_NODES` is already raised for GLM serving.

## CUDA flash-attention changes

The per-head mask is the invasive part. Five files change:

- `fattn.cu`: a mask with a head dimension is no longer rejected. Kernel selection guards
  the GQA optimizations, mask scanning and `kv_rows` so they only apply when the mask is
  shared across heads.
- `fattn-common.cuh`, `fattn-vec.cuh`, `fattn-tile.cuh`, `fattn-mma-f16.cuh`: the mask
  head stride widens to 64-bit and every kernel indexes the mask by the query head it is
  computing.
- `tests/test-backend-ops.cpp`: the flash-attention test gains a `mask_heads` parameter
  and a family of per-head mask cases across head sizes, query counts and KV types.

This fork's flash-attention code has its own sparse and quantized-latent paths for GLM and
diverges from upstream here. These hunks need to be re-applied by reading, not pasted, and
the per-head cases must come with them.

## Vision in mtmd

- New graph file `tools/mtmd/models/inkling2v.cpp` (33 lines) built on the generic ViT
  builder: patch embedding, resized learned positions, class token, 2D RoPE positions as
  explicit inputs, class token removal, 2x2 patch merge, two-layer GELU projector.
- `tools/mtmd/mtmd-model.cpp` registers the `inkling2v` projector, loads the class position
  embedding and projector tensors, sets an image token limit of 4,096 and a warmup of
  1,024 tokens.
- `tools/mtmd/mtmd-image.cpp` adds the preprocessor: pad to a multiple of 40 pixels with
  negative padding after normalization, resize only when over the pixel budget.
- `tools/mtmd/mtmd-subcontext.cpp` supplies the 1-indexed patch coordinates with the class
  token at 0.
- `tools/mtmd/mtmd.cpp` wires the image content token. The follow-up commit simplified
  this once the chat layer learned to split media into its own message blocks.

## Tests upstream added

- `tests/test_conversion_fp8.py`: packed expert dequantization against reference values,
  companion validation failures, and both checkpoint and Transformers tensor names.
- `tests/test-llama-archs.cpp`: a synthetic Inkling2 arch fixture for load and graph runs.
- `tests/test-mtmd-impl.cpp`: the image preprocessor.
- `tests/fusion/MTL.csv`: Metal fusion baselines, not relevant to CUDA serving.
- `tests/test-chat.cpp` in the follow-up: media block splitting cases.

These tests are the port's acceptance floor. They run without the real checkpoint, which
is what makes a local port verifiable before any four-card rental.
