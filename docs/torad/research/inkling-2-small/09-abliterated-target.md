# 09. Selected abliterated target

Marcos selected the abliterated derivative as the serving target on 2026-11-16. The goal is
to keep that derivative's refusal-ablated behavior while reaching the Tensor 4 serving shape
designed in this study. This choice changes the weight source and qualification plan. It does
not change the model architecture, the upstream port, the tensor-split work, the capacity
arithmetic, or the optimization roadmap.

## Pinned source

| Field | Value |
|---|---|
| Repository | `benthecarman/Inkling-2-Small-abliterated` |
| Revision checked | `e4853745aa9cb4587e8959d3c80b92e1bef99a4a` |
| Base model | `thinkingmachines/Inkling-2-Small` |
| Hugging Face relation | Fine-tune of the base model |
| License on the card | Apache-2.0 |
| Format | The official mixed-checkpoint layout: NVFP4 experts in layers 0-39, MXFP8 experts in layers 40-47, BF16 elsewhere |
| Files | 24 model safetensors shards, `mtp.safetensors`, tokenizer files, processor files, chat template, and `trial.json` |
| Used storage reported by Hugging Face | 393,509,649,692 bytes |
| Repository state when checked | Created 2026-11-16 15:48 CT, last modified 16:05 CT, zero downloads and zero likes |
| Ready-made GGUF | None found in the Hugging Face model search on 2026-11-16; the only result was the source repository itself |

The repository's `config.json` matches the architecture recorded in
[02-architecture.md](02-architecture.md): 48 layers, hidden size 4,608, 384 routed experts,
top-8 routing, the same local and global layer pattern, the same vision configuration, and
the same five-layer DSpark configuration. The model card states that the config, tokenizer
and chat template are also unchanged. Those claims still need file and tensor checks during
Phase 0; the revision above is the input to pin, not a floating `main` branch.

## What is known about the derivative

The author reports same-harness, greedy comparisons against the stock model. HumanEval moved
from 95.1% to 97.0%, MBPP moved from 90.8% to 88.4%, and the author describes both changes
as noise. LongBench v2 moved from 54.1% to 48.5%. The stated cause is longer reasoning over
long documents: with a 2,048-token reply budget, the derivative ran out of tokens before
answering on 178 of 503 questions, against 131 for the stock model.

That long-context behavior is the main product risk for long coding sessions. It does not
block the choice, but it becomes a first-class qualification gate rather than a footnote.
The card says most refusals were removed, not that refusal behavior is uniformly absent.
`trial.json` records the steering recipe and trial parameters, but the repository does not
include a standalone reproduction program. All quality and refusal results currently come
from the author; no independent evaluation was found.

## Pack route

Unsloth's existing GGUFs are conversions of the stock model. They remain useful for engine
bring-up, size planning and stock-model comparisons, but they are not the served target.
The target packs must be produced from the pinned abliterated checkpoint:

1. Download the pinned revision and record SHA-256 hashes for every shard and small file.
2. Convert the text model and vision projector with the ported Inkling 2 converter.
3. Generate the importance matrix from the converted abliterated reference, using a pinned
   calibration mix that includes code, tool transcripts, repository text and long documents.
4. Produce a Q3-class pack as the default Tensor 4 candidate and an IQ3-class pack as the
   capacity candidate. Produce a Q4-class pack only as the quality experiment described in
   [05-capacity-and-tensor4.md](05-capacity-and-tensor4.md).
5. Name every artifact with the abliterated source revision and quantization recipe so it
   cannot be confused with a stock Unsloth build of the same nominal bit width.
6. Convert `mmproj` from the derivative repository, then compare its tensors with the stock
   projector. The card implies vision is unchanged; the comparison proves or disproves that
   for the pinned files.
7. Keep the derivative's `mtp.safetensors` with the pack. Hash it against the stock file
   before DSpark work assumes the draft head is unchanged.

The expected file sizes remain the stock Q3, IQ3 and Q4 size classes until local artifacts
exist. They are planning brackets, not measurements of the abliterated conversion.

## Qualification specific to this target

- **Source integrity:** tensor names, shapes, dtypes and sampled values must match the
  expected Inkling 2 layout. Compare small files and sampled tensors with the stock
  checkpoint to identify exactly what the derivative changed.
- **Conversion fidelity:** measure each quantized pack against a high-precision conversion of
  the abliterated source. Stock Unsloth fidelity numbers do not describe this target.
- **Behavior preservation:** run stock, abliterated reference and quantized abliterated packs
  over one pinned refusal-and-control set. Quantization must not silently restore the stock
  refusal profile or damage ordinary coding and tool behavior.
- **Long-session completion:** use long-document and repository tasks at high context depth
  with 2,048-token and 8,192-token answer budgets. Count unfinished answers separately from
  wrong answers, because that is the derivative's reported failure mode.
- **Agent work:** HumanEval and MBPP are only smoke checks. The deciding quality work is
  repository editing, nested tool calls, repeated tool results and multi-turn sessions under
  the server template.

If the Q3-class pack loses the selected behavior or the long-session completion rate is
unacceptable, the next reads are the Q4-class pack and a different imatrix or quantization
recipe from the same pinned source. Falling back to stock weights would be a new target
decision, not an implementation detail.
