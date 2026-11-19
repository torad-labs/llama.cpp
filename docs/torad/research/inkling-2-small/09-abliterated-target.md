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

## Build the pinned Q3_K_M text pack

The 24 source shards total 390,056,654,720 bytes. The bounded-disk builder streams the usual BF16 conversion into the native quantizer in tensor order. It needs space for the projected 264.05 GiB output and its small descriptor header. A sparse-file-capable filesystem is required. The source directory can contain symlinks to shards staged on different filesystems. The builder checks every text shard and the tokenizer and configuration files against the pinned source. [Source revision](https://huggingface.co/benthecarman/Inkling-2-Small-abliterated/tree/e4853745aa9cb4587e8959d3c80b92e1bef99a4a)

For an initial serving benchmark, the stock Bartowski matrix covers all 48 fused gate/up tensors and 48 down tensors with the expected widths and 384 experts. Its metadata records 590 calibration chunks of 512 tokens. Twelve of its 36,864 routed projection/expert entries have zero observations and the native quantizer uses unit weights for those entries. The builder records the matrix digest, calibration source and unobserved count in the GGUF. Derivative-specific calibration remains a separate quality comparison. [Pinned stock matrix](https://huggingface.co/bartowski/Inkling-2-Small-GGUF/blob/75970601c5da75212b2ffb914344e1b7dae2008d/Inkling-2-Small-imatrix.gguf)

Run these commands in the checkout with its converter dependencies installed. Stage the source into the desired disk and RAM layout first if both directories cannot fit on one filesystem.

```sh
hf download benthecarman/Inkling-2-Small-abliterated \
    --revision e4853745aa9cb4587e8959d3c80b92e1bef99a4a \
    --include 'model-*.safetensors' --include '*.json' --include 'chat_template.jinja' \
    --local-dir /workspace/inkling/source
hf download bartowski/Inkling-2-Small-GGUF Inkling-2-Small-imatrix.gguf \
    --revision 75970601c5da75212b2ffb914344e1b7dae2008d --local-dir /workspace/inkling
cmake --build build --target llama-quantize -j 4
python scripts/inkling2/pack.py /workspace/inkling/source \
    /workspace/inkling/Inkling-2-Small-abliterated-e4853745-Q3_K_M-imatrix.gguf \
    --quantizer build/bin/llama-quantize --imatrix /workspace/inkling/Inkling-2-Small-imatrix.gguf \
    --calibration stock-bartowski
```

For a derivative-calibrated build, supply the matrix collected on the derivative and set `--calibration` to the reference model revision, precision and corpus revision used. The builder always quantizes from the original pinned weights. The `stock-bartowski` label checks the published matrix SHA-256.

## Calibrate on the derivative

The first pass uses the running revision's Q3_K_M weights to collect activation statistics. This measures the quantized derivative's routing. A higher-precision reference measures how much routing the initial quantization changed. The full text descriptor dry-run projected 335.06 GiB for Q4_K_M, 392.66 GiB for Q5_K_M, 453.85 GiB for Q6_K, 587.75 GiB for Q8_0 and 1,106.06 GiB for the BF16 intermediate. Q4_K_M fits the aggregate memory of four 95.6 GiB cards at calibration context, subject to per-card allocation. Larger builds need CPU offload. A source snapshot, the existing Q3 pack and a Q4 reference together need roughly 962 GiB of storage. The 560 GiB disk plus 362 GiB shared-memory staging area cannot keep all three at once.

The public pilot combines repository code, long technical documentation and 25 tool-use transcripts from pinned revisions. The script checks the dataset revision and rejects truncated viewer responses. It interleaves excerpts in 8,192-character blocks. The generated file has 438,965 tokens with the pinned tokenizer and SHA-256 `96be9e7c14c431fd3cd88d06ffe5cf08e86ec39afb8b9fed721044e3e5f935f6`. The transcripts use their published tool declarations, messages and results, rendered with the pinned derivative's native chat template. [Repository source](https://github.com/ggml-org/llama.cpp/tree/bafa0420c1e57edf04d71587d32fe57d45df7b95) [Tool transcript source](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2/tree/fc6cc2103c066455aade5d7fbb346039ae36ca5e/SFT)

With the server stopped and the first Q3 pack present, collect eight 4,096-token chunks. Layer split puts whole layers on each GPU and avoids tensor-parallel collectives during observation. The collector copies intermediate activations to the host at each observed multiplication. Its expert input tensors alone account for about 135 GiB of device-to-host reads over this 32,768-token pilot.

```sh
python scripts/inkling2/calibrate.py corpus /workspace/inkling/calibration-public.txt
build/bin/llama-imatrix \
  -m /workspace/inkling/Inkling-2-Small-abliterated-e4853745-Q3_K_M-imatrix.gguf \
  --device CUDA0,CUDA1,CUDA2,CUDA3 -sm layer -ts 1,1,1,1 -ngl 99 --fit off \
  -f /workspace/inkling/calibration-public.txt -c 4096 -b 512 -ub 256 \
  --chunks 8 --no-ppl --parse-special -o /workspace/inkling/derivative-q3-c4096-8.gguf
python scripts/inkling2/calibrate.py coverage /workspace/inkling/derivative-q3-c4096-8.gguf
```

The report checks gate/up and down counts for each of 384 experts in all 48 layers. This pilot supplies 262,144 routing decisions per layer. The stock matrix's 590 shorter chunks left six of 18,432 layer/expert pairs unseen and 22 below 16 observations. For another 24 chunks from new positions, use `--from-chunk 8 --chunks 24` and a new output path. Merge the two files with `llama-imatrix --in-file first.gguf --in-file next.gguf -o combined.gguf` and run coverage again. Collection clears the KV cache between chunks. A 4,096-token pass does not sample positions near the derivative's 128,000-token scaling threshold. To probe that region, collect a separate matrix on a contiguous long document at a context above 128,000, with enough input for the tool's two-context minimum. The merge command requires equal chunk sizes.

Point the pack builder at the new matrix, choose a fresh output path and record the source in `--calibration`, for example `e4853745 / Q3_K_M stock-bartowski pilot / public corpus 96be9e7c / 8x4096`. Keep the first pack for a matched held-out comparison of next-token log probabilities, tool arguments and long-document retrieval. A change in coverage measures the routing sample. Quality needs the held-out comparison. With both Q3 packs plus the original shards on the rental, about 891 GiB are needed across disk and shared memory, before tool files and scratch space.
