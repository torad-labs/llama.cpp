# GLM-5.3 Flash serving recipes

Select the GPU layout explicitly with `scripts/glm53/serve.py --profile`:

| Launcher profile | Server layout | Engine flags and visible devices |
|---|---|---|
| `tensor-2` (default) | One server spans two cards with tensor parallelism | `-sm tensor -ts 1,1`, `CUDA_VISIBLE_DEVICES=0,1` |
| `dp-4` | One server spans all four cards; sequences have individual owners, dense weights are replicated, and routed experts are partitioned across all four | `-sm dp -ts 1,1,1,1`, `CUDA_VISIBLE_DEVICES=0,1,2,3` |

`dp-4` is the Python launcher's profile name, selected with `--profile dp-4`. It creates one four-card group, not two independent two-card servers or a 2x2 tensor/data split. The four-card launcher, host-KV option, measurements and current throughput blocker are documented in the [four-card DP recipe](../torad/glm53-dp.md#candidate-serving-command). That candidate keeps MTP enabled; the historical MTP-off comparison below is specific to `tensor-2`.

The remaining commands and measurements on this page describe `tensor-2`. Omitting `--profile` selects that two-card default.

This recipe reproduces the retained 524,288-token serving configuration for
GLM-5.3-Flash-Uncensored IQ3_XXS on two 96 GB RTX PRO 6000 Blackwell cards. It
uses this fork, the exact model files below, and the bundled chat template and
MTP shortlist. It does not select `--glm53-profile`: the earlier
[SM120 experimental bundle](GLM53-SM120.md) has a separate qualification scope.
Direct sparse-attention indices remain off.

## Build and local files

The portable qualification build used CUDA compiler **13.3.33** and a
**glibc 2.35** build environment, targeting SM120. Use a compatible NVIDIA
driver and matching server/shared libraries from one build. Compilation on a
newer glibc host does not establish compatibility with glibc 2.35.

Build this fork at the commit you intend to measure, for example:

```sh
cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120a \
  -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=ON
cmake --build build --target llama-server --parallel 4
python3 -m unittest discover -s scripts/glm53 -p 'test_*.py'
```

Select the intended CUDA compiler/toolchain when configuring CMake. Keep a record
of the commit, compiler, driver, build flags, and executable/library hashes for
each comparison. The launcher does not build, download files, or manage GPUs.

Obtain the public files from
[BoldingBuilds/orcarouter_GLM-5.3-Flash-Uncensored-GGUF at revision 1bc0afadd92f4ff9fc2b1281a95a7921b058d35d](https://huggingface.co/BoldingBuilds/orcarouter_GLM-5.3-Flash-Uncensored-GGUF/tree/1bc0afadd92f4ff9fc2b1281a95a7921b058d35d).
Keep all three GGUF shards together and pass the first shard to `--model`.

| File relative to the model repository | Bytes | SHA256 |
|---|---:|---|
| `IQ3_XXS/GLM-5.3-Flash-Uncensored-IQ3_XXS-00001-of-00003.gguf` | 45,672,911,040 | `9916f01db53f8a2bfd5db1d6eb8cd48f93ae2b732fe1dfce993450745334cae1` |
| `IQ3_XXS/GLM-5.3-Flash-Uncensored-IQ3_XXS-00002-of-00003.gguf` | 45,407,495,392 | `139f2ea5e2d1bb847cd21c5d5122e3a838fde7b5ae2895c6e2623ff318519a00` |
| `IQ3_XXS/GLM-5.3-Flash-Uncensored-IQ3_XXS-00003-of-00003.gguf` | 43,225,116,416 | `b94bcd70929d7f4445d3acb6490a801511452214c8cf633fa2d912bef89f2188` |
| `mmproj-GLM-5.3-Flash-Uncensored-F16.gguf` | 1,128,047,744 | `913d809cc01177fe8320c4d30ac078df60b444888ca0ed07b1b62b3c45d85b9e` |

Verify these hashes once after obtaining the files. The launcher checks file
presence at launch but does not rehash approximately 134 GB of model shards on
every startup. The optional projector is required for image requests; omitting
it gives a text-only variant of the recipe. See the
[asset provenance](../../scripts/glm53/assets/README.md) for the public-corpus
draft shortlist and pinned template hashes.

## Inspect and launch

First print the exact argument vector and environment without loading a model:

```sh
python3 scripts/glm53/serve.py --profile tensor-2 \
  --server ./build/bin/llama-server \
  --model /models/GLM-5.3/IQ3_XXS/GLM-5.3-Flash-Uncensored-IQ3_XXS-00001-of-00003.gguf \
  --mmproj /models/GLM-5.3/mmproj-GLM-5.3-Flash-Uncensored-F16.gguf \
  --mtp-depth off \
  --dry-run
```

Remove `--dry-run` to launch. The script uses Python's standard library and
replaces its process with `llama-server`, without a shell or background daemon.
Model/server paths may be placeholders during a dry run. Bundled asset hashes
are checked in both modes. By default it binds to `127.0.0.1:8102`; `--host` and
`--port` change that address. Exposing it beyond localhost requires your own
access controls.

| Setting | Recipe |
|---|---|
| GPU placement | `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `CUDA_VISIBLE_DEVICES=0,1`, `-ngl 99 -sm tensor -ts 1,1` |
| Context and slots | `-c 524288 -np 3 --kv-unified` |
| Attention/indexer caches | `--cache-type-k q8_0 --cache-type-v q8_0 -ctki q8_0` |
| Prompt batching | `-b 2048 -ub 2048 --prompt-step 1024` |
| RAM prompt cache | `--cache-ram 32768 --no-cache-idle-slots` |
| Chat and attention | `--jinja -fa on`, bundled `GLM-5.3-Flash.jinja` |
| MTP | `--spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-mtp-adaptive 0.18`, bundled 98,304-ID shortlist |
| CUDA graphs/collectives | `GGML_CUDA_GRAPH_MAX=0`, `GGML_CUDA_ALLREDUCE=internal` |
| Indexer query | `LLAMA_INDEXER_Q8_QUERY=1` |
| MTP rows/token embeddings | `LLAMA_MTP_DEVICE_H=1`, `LLAMA_TOK_EMBD_GPU=1` |
| Direct sparse indices | on, the engine's default: the launcher sets nothing (`LLAMA_KPOOL_DIRECT_INDICES_LEGACY=1` is the mask/compaction route) |
| MMQ scheduling | `GGML_CUDA_MMQ_MOE_NCOLS=0`, `GGML_CUDA_MMQ_L2_PF=1` |
| Diagnostics | `--metrics --log-timestamps`, `GGML_SCHED_DEBUG=0` |

`--mtp-depth 1`, `--mtp-depth 2`, and `--mtp-depth off` select the corresponding
comparison. Off removes the MTP-specific server options. `--prompt-step N`
limits prompt work beside an active response; zero removes that cap. These
choices affect workload-dependent throughput and response gaps. The launcher
leaves sampling flags unset, as in the retained plan; requests can override the
model's sampling defaults.

The launcher removes inherited `LLAMA_*`, `GGML_*`, `CUDA_*`, and `NVIDIA_*`
overrides, then installs the displayed settings. This prevents a forgotten
profile or diagnostic switch from changing the recipe. The server's own access
controls stay: `LLAMA_API_KEY`, `LLAMA_ARG_API_KEY_FILE`, `LLAMA_ARG_SSL_KEY_FILE`
and `LLAMA_ARG_SSL_CERT_FILE` reach the server, and the dry run lists the keys it
removes under `removed`. The `tensor-2` profile fixes device
visibility to cards 0 and 1; container device mapping must make the intended
cards available. `LD_LIBRARY_PATH` is set to the selected server's directory,
so place its matching shared libraries there. Run the displayed command
manually if you need a different configuration, and record those differences.

## Memory and concurrency limits

The 524,288 KV cells form **one shared pool**, not three independent 524,288-token
conversations. Concurrent prompts, retained histories, generated tokens, and
working memory compete for capacity. The original placement requirement was
at least 97,000 MiB on each card; nominal VRAM alone does not establish fit with
other GPU processes, different drivers/builds, projector workloads, or deeper
drafts. The 32,768 MiB RAM-cache budget is host memory in addition to model
mapping and other process memory. It is not another GPU-context allocation.

`--no-cache-idle-slots` keeps the served idle-slot policy explicit. Stable prompt
prefixes permit reuse, but a cache setting cannot repair history that the client
rewrites or omits. Large context allocation is a capacity setting, not evidence
that every context depth, three fully occupied histories, or every image shape
has been tested. This recipe does not enable the separate 1M-context profile.

Q8 indexer queries are a retained lossy setting: rounding can change pool
selection near score ties. Use `--indexer-query f32` to disable query rounding
for a comparison; `--indexer-query q8` is the default. This option changes query
scoring only, leaving the Q8 indexer/KV caches unchanged. Matched eight-token
probes on fresh servers diverged at token eight at 32k and token seven at 128k,
for both cold and warm requests. The checks do not establish coding-quality
equivalence or tell which continuation is better.

## API and history

OpenAI-compatible chat request:

```sh
curl http://127.0.0.1:8102/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"GLM-5.3-Flash-Uncensored-IQ3_XXS","messages":[{"role":"user","content":"Write a Python function that rejects duplicate JSON object keys."}],"max_tokens":32768,"stream":false}'
```

Native Anthropic-compatible request:

```sh
curl http://127.0.0.1:8102/v1/messages \
  -H 'Content-Type: application/json' -H 'anthropic-version: 2023-06-01' \
  -d '{"model":"GLM-5.3-Flash-Uncensored-IQ3_XXS","messages":[{"role":"user","content":"Write a Python function that rejects duplicate JSON object keys."}],"max_tokens":32768,"stream":false}'
```

These are examples for a server you have started, not checks the launcher runs. The output limit includes thinking tokens; the small coding fixture used here exhausted 2,048- and 8,192-token limits during reasoning. Set a deliberate output budget and check the stop reason before treating a response as complete.
For a tool loop, retain the complete assistant response in the next request:
OpenAI `reasoning_content`, content and `tool_calls`, or native `thinking`, text
and `tool_use` blocks. Return tool results with the original call IDs and retain
their order. Preserve `defer_loading` and `tool_reference` data during discovery;
the template places discovered definitions at their tool-result position. This
helps keep earlier prompt bytes stable when additional tools become available.

The template defaults `clear_thinking` to false, retaining supplied historical
reasoning. The startup suggestion to add `--reasoning-preserve` is a capability
notice: this template already preserves it without that flag. Explicit
`chat_template_kwargs.preserve_reasoning=false` or `clear_thinking=true` changes
that behavior. A server flag cannot recover reasoning omitted by the client.

Reasoning effort is rendered near the start of the prompt: `low` and `high` map
to those labels; the default and other values map to `max`. Changing effort
between turns changes the early prefix and can reduce cache reuse. For OpenAI
requests use `reasoning_effort`; for native requests the same template argument
can be passed as `chat_template_kwargs.reasoning_effort`. Keep the effective
setting stable within a session when comparing cache behavior. This template
always opens a thinking section for generation; this recipe makes no claim
that an effort setting disables reasoning.

## Measurement scope

The measurements below were collected before PR #92 was rebased onto
`ddf1d49b3` (main after PR #109). The combined branch includes PR #108's
device-resident MTP/token-embedding work and PR #109's peer-copy reductions.
Local builds and regression checks do not qualify that combined build on GPUs;
remeasure it before treating these rates as its current performance.

The exact build, request workload, raw measurements, and remaining limitations
belong with the associated [PR #92](https://github.com/torad-labs/llama.cpp/pull/92)
qualification record. The earlier SM120 bundle's 200 tok/s target and historical
rates are not results for this serving recipe. Short kernel tests, a large
context allocation, or a passing synthetic coding fixture do not establish a
GPU throughput ceiling or broad coding/reasoning quality. Direct sparse indices
and the prior bundled profiles are not promoted by this recipe.

The November 2 sweep measured warm raw-completion decoding at 32k and 128k
tokens, with three repetitions. Values below are median server decode tokens per
second **per request**, including the three-active-session columns.

| Prompt | MTP off, single | MTP 1, single | MTP 2, single | MTP off, three active | MTP 1, three active | MTP 2, three active |
|---|---:|---:|---:|---:|---:|---:|
| 32k | 131.56 | 182.74 | 138.27 | 67.84 | 57.09 | 32.45 |
| 128k | 125.73 | 134.73 | 162.38 | 51.88 | 37.41 | 29.54 |

For predominantly concurrent coding, start with `--mtp-depth off`, as in the
launch example. Compare `--mtp-depth 1` for one shorter session and
`--mtp-depth 2` for one longer session. Omitting the option retains the original
depth-two configuration; the launcher does not switch modes automatically.
These sequential configuration sweeps used repeated public code, 128 output
tokens, temperature zero, and a 512k allocation. They are not general coding
scores or guaranteed rates, and an opposite-order full-model repeat remains
outstanding. This recommendation applies to warm concurrent histories. In a
separate 128k cold-prefill overlap leg, the existing warm response fell to
3.43 tokens/s with MTP off, 4.72 at depth one, and 6.04 at depth two; the new
request's client time to first output was 38.33, 43.29, and 41.92 seconds,
respectively. A mode that wins with warm sessions can lose while another session
prefills. These overlap legs were single observations, not tail percentiles.

Depth one passed the staged native tool-loop fixture, and MTP off passed its
OpenAI-compatible counterpart. Both introduced a new duplicate-key requirement
on turn two, required another implementation change, and passed all eight
follow-up checks. These are bounded synthetic tests. Workload quality and cache
retention must be checked alongside throughput when changing serving settings.

A capacity-pressure check filled three separate 163,840-token histories, then
replaced one with a 262,144-token prompt. Both earlier histories returned with
163,836 cached tokens and only four prompt tokens evaluated. The run exercised
an allocation-related restore failure: the host snapshot remained recoverable,
an idle slot was saved and cleared, and the retry succeeded. Returning first
output took 5.40 and 3.55 seconds including client transfer and restore work;
cache reuse does not make host-to-device restoration free. This check used MTP
off and the 32 GiB host-cache budget; smaller budgets can require intentional
eviction.

One target-only 128k overlap comparison changed `--prompt-step` from 1024 to
256. The existing warm response improved from 3.43 to 5.63 server tokens/s,
but the cold request's client time to first output increased from 38.33 to
48.67 seconds. The recipe retains 1024; compare 256 when protecting an existing
stream matters more than admitting a new long prompt. This single comparison
does not establish a latency percentile or a universal optimum.

The real-model recurrent rollback test also passed all four cases on the two
GPUs: three rows removed from a nine-row batch, and one, two, or three rows
removed from a four-row verification batch. It used a 512-token test context,
Q8 caches, and a rounding-aware error bound with one-token-early/late controls;
it is not a full long-context distribution test.
