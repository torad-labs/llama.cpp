# Sources

All web sources were accessed on 2026-11-16. Each entry states what it was trusted for.
Vendor benchmark numbers were recorded as claims, never as measurements of this fork.

## Primary model sources

- Thinking Machines Lab model card: https://thinkingmachines.ai/model-card/inkling-2-small/
  License, release date, intended use, headline architecture, hardware requirements,
  checkpoint recommendation, evaluation tables.
- Hugging Face repository files for `thinkingmachines/Inkling-2-Small`:
  `config.json`, `chat_template.jinja`, `processor_config.json`,
  `special_tokens_map.json`, and the file listing. Dimensions, layer pattern, DSpark
  config, template behavior, special tokens, shard sizes.
- Hugging Face blog, "Welcome Inkling by Thinking Machines", Inkling 2 section:
  https://huggingface.co/blog/thinkingmachines-inkling
  Block attention residuals, L2 clipping, sinks, gated output, dynamic short
  convolutions, vision encoder description, generational comparison.

## Selected derivative

- Hugging Face repository for `benthecarman/Inkling-2-Small-abliterated`, checked at
  revision `e4853745aa9cb4587e8959d3c80b92e1bef99a4a` on 2026-11-16:
  https://huggingface.co/benthecarman/Inkling-2-Small-abliterated
  The README, `config.json`, `trial.json`, repository API record and file listing were
  trusted for the derivative's identity, format, architecture, storage and included files.
  Its evaluation table is author-reported and was recorded as a claim, not an independent
  measurement.
- Hugging Face model search for `Inkling-2-Small-abliterated`, accessed 2026-11-16:
  https://huggingface.co/api/models?search=Inkling-2-Small-abliterated&full=false&limit=100
  Trusted only for the narrower finding that no same-name GGUF repository appeared in that
  search when checked; the source repository was the only result.

## Framework implementations

- Upstream llama.cpp commit `5c008e00085f4d3e394f9a1be649284999de1326` (PR #32217) and
  follow-up `dd099a6bf` (PR #32294), read from the local `upstream/master` ref at
  `a946eb541`. The model graph, converter, per-head mask changes, mtmd support, tests,
  and the stated limit that real-checkpoint conversion was untested.
- vLLM PR #67325: https://github.com/vllm-project/vllm/pull/67325
  DSpark semantics, optimized kernel inventory, numerics notes, GB200 performance claims,
  SM100 limitation, TP4 context cap.
- vLLM blog and recipe: https://vllm.ai/blog/2026-11-12-inkling2-small and
  https://recipes.vllm.ai/thinkingmachines/Inkling-2-Small
  Architecture comparison table and the recommended speculative configuration.
- SGLang blog: https://www.sglang.io/blog/inkling-2-small-day0-support
  Feature matrix, MegaMoE and decode-attention kernel claims, benchmark methodology note
  that DSpark acceptance was simulated at 3.5.
- SGLang cookbook: https://docs.sglang.io/cookbook/autoregressive/ThinkingMachines/Inkling-2-Small
  Deployment recipes, TP and EP guidance including the NVFP4 sharding constraint,
  MegaMoE thresholds, parser names, encoder disaggregation behavior.
- NVIDIA NeMo AutoModel reference:
  https://docs.nvidia.com/nemo/automodel/nightly/model-coverage/vision-language-models/thinkingmachines/Inkling-2-Small
  Architecture table cross-check, checkpoint format notes, and the dequantization caveat
  for training loaders.

## GGUF builds

- Unsloth GGUF repository README: https://huggingface.co/unsloth/Inkling-2-Small-GGUF
  Build inventory, sizes, KLD and top-token agreement methodology, mmproj files, run
  commands. Its pointer to llama.cpp PR #32187 predates the upstream merge and was not
  followed.
- Unsloth guide: https://unsloth.ai/docs/models/inkling-2
  Recommended sampling, reasoning effort values, memory guidance.
- Community size accounting at https://vramneeded.com/models/inkling-2-small/ was used only
  to cross-check exact byte totals for the official repositories and GGUF builds, which
  were then recomputed independently in this study.

## Local sources

- This fork at HEAD, searched with CodeGraph (`cg`): absence of Inkling symbols; presence
  and locations of hybrid ISWA memory, recurrent state helpers, `ggml_ssm_conv`,
  `LLM_FFN_SITU`, NVFP4 and MXFP4 types, MoE kernel gates, split-state table, MTP gating
  and the prompt-step budget. File and line citations appear inline in
  [04-fork-gap-analysis.md](../04-fork-gap-analysis.md).
- `scripts/glm53/serve.py` and `docs/torad/glm53-two-gpu.md`: the Tensor 4 serving pattern,
  card capacity of 97,887 MiB, reserve-session model and the box-read order that
  [07-implementation-plan.md](../07-implementation-plan.md) adapts.
- Hosted pricing cross-check: https://www.together.ai/models/inkling-2-small
