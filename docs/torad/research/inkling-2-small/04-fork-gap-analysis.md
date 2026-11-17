# 04. Fork gap analysis

This file compares what Inkling 2 needs with what this fork at HEAD already has. Code
presence was checked with CodeGraph on 2026-11-16. `cg query inkling`, `Inkling` and
`inkling2` return no model symbols in this fork.

## What transfers directly

| Need | Fork status | Evidence |
|---|---|---|
| Hybrid sliding + global KV in one memory object | Present | `build_inp_mem_hybrid_iswa` at `src/llama-graph.cpp:5615`; `llama_memory_hybrid_iswa` combines `llama_kv_cache_iswa` and `llama_memory_recurrent` at `src/llama-memory-hybrid-iswa.cpp:11` |
| Sliding cache capped by window and sequence count | Present | `llama_kv_cache_iswa` sizes the SWA cache from `n_swa`, unified mode and `n_seq_max` at `src/llama-kv-cache-iswa.cpp:69-73` |
| Recurrent state read, store, save, restore, copy | Present | `build_rs` and `build_conv_state_store` at `src/llama-graph.cpp:4714` and `:4954`; hybrid ISWA implements `state_write`, `seq_cp` and `get_can_shift` |
| Causal short convolution op | Present | `ggml_ssm_conv` at `ggml/src/ggml.c:5693` |
| Softcapped SwiGLU activation | Present | `LLM_FFN_SITU` at `src/llama-graph.h:75` |
| NVFP4 tensor type and CUDA MMQ instances | Present | `GGML_TYPE_NVFP4` at `ggml/include/ggml.h:259`; NVFP4 quantize and MMQ load paths indexed in `ggml/src/ggml-cuda/` |
| Routed-expert decode and prefill kernels | Present, GLM-tuned | `ggml_cuda_mmvq_moe_usable` at `ggml/src/ggml-cuda/mmvq-moe.cu:866`; shared MMQ expert pair path at `ggml/src/ggml-cuda/mmq.cu:486` |
| Tensor split through the meta backend | Present, GLM-shaped | `llama_meta_device_get_split_state` at `src/llama-model.cpp:419`; pattern table for standard tensor names from `src/llama-model.cpp:445` |
| Prompt steps bounded by time beside replies | Present | `server_prompt_budget` at `tools/server/server-common.h:79`, wired by `--prompt-step-ms` |
| Kept CUDA graphs, unified KV pool, prompt cache | Present for GLM serving | The Tensor 4 launcher plan in `scripts/glm53/serve.py` and its pinned tests |

The memory shape is the single biggest transfer. GLM serving already forced this fork to
treat attention cache and recurrent state as one sequence state with rollback, save and
restore. Inkling's convolution state is smaller and simpler than GLM's KDA state, and its
KV is plain grouped-query cache rather than an MLA latent. The mechanisms carry; the
per-arch wiring does not exist yet.

## What is missing and must be ported or built

1. **The arch itself.** No `inkling2` registration, hparams, tensor map, model class,
   graph, converter, pre-tokenizer or mtmd projector. Upstream supplies all of these; see
   [03-upstream-llama-port.md](03-upstream-llama-port.md).
2. **Relative position plumbing.** No distance input and no
   `set_input_relative_pos` in the KV cache. The fork's `build_attn` also lacks upstream's
   explicit mask parameter and per-query sink handling.
3. **Per-head flash-attention masks.** Upstream had to teach the vec, tile and MMA kernels
   to index a mask by query head and to decline the shared-mask optimizations. This fork's
   flash-attention files carry GLM sparse and latent work and differ from upstream in the
   same places. Expect the most careful hand work of the port here.
4. **Router precision controls.** `expert_router_f32` and `expert_weights_norm_eps` do not
   exist in the fork's hparams or `build_moe_ffn`. Without them the router accumulates in
   the default precision and normalizes by a clamped sum, which can pick different experts
   on near-ties.
5. **An MXFP8 tensor type.** ggml has `GGML_TYPE_MXFP4` and `GGML_TYPE_NVFP4`; there is no
   MXFP8. The native checkpoint therefore cannot be represented bit-exact in GGUF. The
   study's pack strategy requantizes instead. A native path would be a separate type,
   quantize, dequantize and kernel project.
6. **Inkling split-state rules.** The pattern table in
   `llama_meta_device_get_split_state` knows standard names and has special sections for
   GLM and DeepSeek variants. Inkling adds tensors with no standard analog: the relative
   projection and bias table, per-head clip scales, the sink projection, dynamic
   convolution projections, four convolution kernels per layer, residual-mix queries, and
   a recurrent state whose channels are a concatenation of K, V and two hidden streams.
   Each needs an explicit split decision consistent with how Q, KV heads and hidden
   partials are split, and a test that would fail if the decision silently mirrored a
   cache. A wrong default here is not a slow path; a mirrored global KV cache at BF16
   would need about 92 GiB per card at the GLM pool size and simply not load.
7. **DSpark.** The fork's coupled speculation is gated to the GLM arch:
   `llama_context::mtp_draft_step` returns false unless the model is `LLM_ARCH_GLM5NEXT`
   with one nextn layer (`src/llama-context.cpp:6527-6535`). DSpark is a separate draft
   model with hidden-state taps at eight target depths and its own Markov state. None of
   that plumbing exists.
8. **Chat and server specifics.** The media-block split in `common/parsers/inkling2.cpp`
   and its test are upstream-only. Reasoning effort validation, Inkling tool-call parsing
   through the generic autoparser, and the server's behavior when the template raises on
   an unknown effort level all need qualification in this fork's server, whose chat code
   has its own history.

## Where GLM experience misleads if applied blindly

- **The reduction wire.** Tensor 4 GLM runs an 8-bit reduction wire with a quality read at
  four ranks. Inkling reductions feed short convolutions whose state persists across
  steps, so wire error is written into sequence state, not just into one step's logits.
  The wire needs its own quality leg for this model.
- **MoE fusion gates.** The fork's fused and shared expert paths carry conditions learned
  on GLM, including declining data-split experts in
  `ggml_cuda_should_fuse_mul_mat_q` (`ggml/src/ggml-cuda/ggml-cuda.cu:3971-3982`) and the
  shape rules in `ggml_cuda_can_share_mmq_id` (`ggml/src/ggml-cuda/ggml-cuda.cu:7127`).
  Inkling's SITU activation and its 384-expert, top-8, 2,304-wide shapes need their own
  pass through these gates with backend-ops coverage, not an assumption that GLM's choices
  are right.
- **Graph shapes.** GLM graph reuse work assumes inputs whose shapes are stable across
  steps at fixed row counts. Inkling's distance input is shaped by the cache's KV count,
  and upstream's reuse check requires an exact match. In a unified pool that grows and
  shifts, this input is a new source of graph churn until the relative bias moves into
  the kernel.
- **Prompt cache.** GLM prompt-cache restore already snapshots recurrent state, but
  Inkling prefix reuse is only valid where the convolution state at the restore point is
  also restored. The state is small; the correctness condition is the same one the fork's
  rollback tests already encode, and it needs Inkling cases.

## Development verifiability

The smallest published GGUF is 171.7 GB, so no real pack loads on a development machine's
cards. The port is nevertheless locally verifiable: upstream's conversion tests use
synthetic packed checkpoints, `test-llama-archs` uses a synthetic arch fixture, and the
per-head mask work lands in `test-backend-ops`. This fork adds its meta split tests under
`tests/test-backend-meta-split.cpp`. The plan in
[07-implementation-plan.md](07-implementation-plan.md) keeps every phase checkable this way
until the final box read.
