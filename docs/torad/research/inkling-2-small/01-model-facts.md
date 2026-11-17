# 01. Model facts

Sources are listed in [data/sources.md](data/sources.md). Facts below come from Thinking
Machines Lab's model card, the Hugging Face repository files, and the framework teams'
support pages. Where sources disagree, the disagreement is noted.

## Release and license

- Provider: Thinking Machines Lab, Inc.
- Model announced 2026-11-05. Weights published on Hugging Face on 2026-11-12.
- License: Apache 2.0, with a separate Acceptable Use Policy linked from the model card.
- Repositories: `thinkingmachines/Inkling-2-Small` (mixed NVFP4 and MXFP8 experts) and
  `thinkingmachines/Inkling-2-Small-MXFP8` (all experts MXFP8).
- Modalities: text and image in, text out. No audio input, unlike the first Inkling.
- Context window: up to 1,048,576 tokens.
- Intended use, per the card: agentic and tool-use systems, coding assistants, chatbots,
  retrieval-augmented generation, research and fine-tuning.

## Headline properties

| Property | Value |
|---|---|
| Total parameters | 594B (593.72B in GGUF metadata) |
| Active parameters per token | 17B |
| Architecture | Decoder-only transformer, sparse MoE feed-forward |
| Layers | 48 |
| Hidden size | 4,608 |
| Experts | 384 routed per layer, top-8 per token, no shared experts |
| Attention | 48 query heads, 8 KV heads, head dimension 128 |
| Attention pattern | 5 sliding-window layers then 1 full layer, repeated 8 times |
| Sliding window | 1,024 tokens |
| Position encoding | Learned relative bias in attention logits, no RoPE in the text model |
| Vocabulary | 201,024 rows, 200,058 usable token ids |
| Native numerics | NVFP4 experts in layers 0-39, MXFP8 experts in layers 40-47 |
| Vision tower | 23-layer ViT, hidden 1,024, 16 heads, 20x20 patches, 2x2 merge |
| Speculator | DSpark draft head in `mtp.safetensors`, 7 drafted tokens per step |

Thinking Machines states the model was pretrained, midtrained and post-trained in NVFP4 and
recommends the mixed checkpoint for all use. They report the all-MXFP8 checkpoint gains one
to two percentage points on evaluations, which they do not consider meaningful in practice.

## Checkpoint files and sizes

The main repository holds 24 model shards plus `mtp.safetensors` at 3.42 GB, for
393,480,084,520 bytes total. The MXFP8 repository totals 622,803,514,224 bytes. The expert
quantization layout is recorded per tensor in `hf_quant_config.json`. One detail matters for
conversion: upstream llama.cpp's converter found that the declared MXFP8 block shape in that
file conflicts with the stored weight and scale shapes, and it trusts the stored shapes with
a warning. See [03-upstream-llama-port.md](03-upstream-llama-port.md).

Stated hardware from the model card:

- Mixed NVFP4 checkpoint: at least 500 GB aggregate VRAM, for example 2x B300 or 4x H200.
- MXFP8 checkpoint: at least 800 GB aggregate VRAM, for example 4x B300 or 8x H200.

Four RTX PRO 6000 cards provide 410.6 GB in total, below both figures. That gap is the
subject of [05-capacity-and-tensor4.md](05-capacity-and-tensor4.md).

## Selected abliterated derivative

Marcos selected `benthecarman/Inkling-2-Small-abliterated` as the serving target on
2026-11-16. The checked revision is `e4853745aa9cb4587e8959d3c80b92e1bef99a4a`. It is a
full checkpoint in the official mixed format, with 24 safetensors shards and an
`mtp.safetensors` file; Hugging Face reports 393,509,649,692 bytes of used storage. Its
config matches the architecture above, and its card states that the tokenizer and chat
template are unchanged.

The author's same-harness results put HumanEval at 97.0% against the stock model's 95.1%,
MBPP at 88.4% against 90.8%, and LongBench v2 at 48.5% against 54.1%. The author calls
every difference except LongBench v2 noise and attributes the long-document loss to longer
reasoning that more often exhausts a fixed reply budget. No GGUF of this derivative was
found in Hugging Face model search when checked. The target therefore has to be converted
and requantized locally; see [09-abliterated-target.md](09-abliterated-target.md).

## Community GGUF builds

Unsloth publishes dynamic, imatrix-calibrated GGUFs of the stock model today. Their BF16
build keeps the original NVFP4 experts bit-exact in layers 0-39 and stores every other
weight in BF16, which makes it larger than the native checkpoint because the last eight
expert layers grow from MXFP8 to BF16. Quality figures below are Unsloth's own measurements against that BF16
build on wikitext-2, and should be read as relative guidance, not agent-task quality.

| Build | Size (GB) | Mean KLD vs BF16 | Same top token |
|---|---:|---:|---:|
| UD-IQ1_S | 171.7 | 0.0922 | 89.7% |
| UD-IQ1_M | 178.5 | 0.0829 | 90.2% |
| UD-IQ2_XXS | 190.6 | 0.0577 | 91.8% |
| UD-Q2_K_XL | 205.0 | 0.0450 | 92.7% |
| UD-IQ3_XXS | 226.0 | 0.0364 | 93.3% |
| UD-Q3_K_XL | 285.1 | 0.0176 | 95.3% |
| UD-Q4_K_XL | 354.2 | 0.0029 | 98.0% |
| UD-Q8_K_XL | 386.5 | 0.0027 | 98.1% |
| BF16 | 484.4 | reference | - |

The vision projector ships separately as `mmproj-F16.gguf` at 0.66 GB, with F32 and BF16
variants. The chat template is embedded in every GGUF.

## Inference ecosystem status on 2026-11-16

- vLLM: day-0 support in an open pull request (#67325) plus a dedicated Docker image. The
  PR reports about 630 output tokens per second for one user at tensor parallel 4 on GB200
  with DSpark, and an acceptance length of 3.5 tokens per step. It is limited to datacenter
  Blackwell (SM100), and a 1M-token request needs tensor parallel 8.
- SGLang: day-0 support in development images named `dev-inkling-2-small`, with recipes for
  Blackwell, H200, AMD MI350X/MI355X and TPU7x. Their published decode numbers simulate
  DSpark acceptance at 3.5 tokens per step.
- llama.cpp upstream: text and still-image support merged 2026-11-14 CT as PR #32217. Some
  guides still point at the earlier draft PR #32187 because they were written before the
  merge. For this fork, upstream master is the source of truth.
- Transformers and NeMo AutoModel: support exists. NeMo's recipes dequantize experts at
  load for training and state that numerical equivalence with the quantized inference
  runtime is not established.
- MLX: community build behind an unmerged mlx-lm pull request, without the DSpark head.

## Serving behavior worth knowing before designing

- Recommended sampling from Thinking Machines: temperature 1.0, top-p 1.0, top-k 0, min-p 0.
- Reasoning effort is a per-request template input. The template maps none, minimal, low,
  medium, high, xhigh and max to numbers from 0.0 to 0.99 and injects a system message. It
  raises an exception on an unknown level, which the server must surface as a client error.
- Tool declarations and tool calls use Inkling-specific message and content tokens. vLLM and
  SGLang ship dedicated parsers. In llama.cpp the embedded template plus the generic
  autoparser carry this, with one Inkling-specific media-block fix upstream.
- Hosted reference pricing at Together AI is $0.30 per million input tokens and $1.00 per
  million output tokens, which sets the bar a self-hosted deployment is compared against.
