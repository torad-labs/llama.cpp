# torad-labs/llama.cpp

Torad's public fork of llama.cpp. The CUDA kernels and server features were measured on
[Ternary Bonsai 2 27B](docs/torad/bonsai-27b/README.md): PrismML's ternary Qwen3.8-27B, packed
as PQ2_0 at 2.13 bits per weight, 7.66 GB with its MTP draft head. It fits a 16 GB GeForce
card with a 262,144-token window. The commits are general CUDA and server changes.
Ternary Bonsai 2 27B is the model the numbers below were measured on.

The base is [PrismML's llama.cpp](https://github.com/PrismML-Eng/llama.cpp), which tracks
[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) and adds the ternary formats.
PrismML's note on which GGUF to load, and the upstream project README, are in
[docs/upstream-readme.md](docs/upstream-readme.md). Every change on this fork, with its
measurement and its off switch, is in [TORAD.md](TORAD.md).

## Headline

One RTX 5070 Ti (16 GB), the public pack, PrismML
[`prism-b10709-9a9394a`](https://github.com/PrismML-Eng/llama.cpp/releases/tag/prism-b10709-9a9394a)
against this fork at [`c008fe8`][c008fe8]. Both binaries were built with the flags in
[Build](#1-build). rig 0.1.2 to 0.1.6 was installed. The host had been quiet for two minutes
(1-minute load under 6; at most 4.4 during the run,
[`quiet-rerun.sh`](docs/torad/bonsai-27b/runs/2026-09-24-headline/quiet-rerun.sh)).
[`bench-matrix.sh`](docs/torad/bonsai-27b/bench-matrix.sh): llama-bench, q4_0 K/V, flash
attention on, no draft head on either side, 3 repetitions.

| context | prefill pp512, Prism -> fork | decode tg128, Prism -> fork |
|---|---|---|
| 0 | 1,888 -> 2,164 tok/s | 85.8 -> 85.8 tok/s |
| 16,384 | 1,626 -> 1,994 | 71.2 -> 81.9 |
| 65,536 | 874 -> 1,479 | 50.3 -> 75.0 |
| 131,072 | 547 -> 1,104 | **35.5 -> 67.3** |

Depth-0 decode in this table is 85.8 tok/s on both sides. A 5-repetition A-B-A-B on the
[Bonsai page](docs/torad/bonsai-27b/README.md) gives Prism 88.4 / 88.2 and the fork 87.1 / 87.1
tok/s. With every off switch set, the fork does 87.4 tok/s, so the gap sits outside those switches.

Served rows are the 2026-09-22 run, fork at [`c54ace8`][c54ace8]. One conversation, a
65,536-token window, six prompts of 512 tokens, greedy, two runs each, with Prism measured in
the same session. The 2026-09-24 served rows ran at host load 20-50. That load moves decode by
8-12 tok/s, so the table uses the earlier run.

| server | decode | against Prism, same session |
|---|---|---|
| fork, MTP draft head | **104.8 tok/s** (85.8 on prose, 120.8 on step-by-step reasoning) | **1.40x** (Prism 75.0) |
| fork, no draft head, recurrent state in f32, as Prism keeps it | 77.0 tok/s | -1.5%, within the drift (Prism 78.2) |
| fork, no draft head, recurrent state in q8_0 (`-cts q8_0`, 3.4x less state VRAM) | 71.3 tok/s | -4.9% (Prism 75.0) |

Prism b10709 refuses this pack's draft head at load, so it has no MTP row. On the fork, 47% of
drafted tokens were accepted on prose and 85% on arithmetic reasoning. The command for that
row is in [Benchmark](#4-benchmark). Per-prompt rows, logs, and the rest of the method are on
the [Bonsai page](docs/torad/bonsai-27b/README.md).

## Quick start

CUDA 12.8 or newer, architecture sm_120 (RTX 50-series and RTX PRO Blackwell).

### rig

On Linux, [rig](https://github.com/torad-labs/rig) checks the card, installs a build, fetches
the pack, and serves it. The machine needs the NVIDIA driver (CUDA 13 or newer) and glibc 2.35
or newer (Ubuntu 22.04+, Debian 12+, Fedora 36+):

```bash
curl -fsSL https://github.com/torad-labs/rig/releases/latest/download/install.sh | sh
rig up bonsai-2-27b
```

`rig up` installs the engine and draft head that rig currently publishes, and listens on
`127.0.0.1:8099`. The tables above were measured from a source build at the commits named
there, with the pack's own draft head. Steps 1 to 3 below are that build. What rig pins is
on the [Bonsai page](docs/torad/bonsai-27b/README.md).

### 1. Build

```bash
git clone https://github.com/torad-labs/llama.cpp && cd llama.cpp
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120a -DGGML_NATIVE=ON \
  -DGGML_CUDA_FA=ON -DGGML_CUDA_FA_ALL_QUANTS=OFF -DGGML_CUDA_GRAPHS=ON \
  -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF
cmake --build build --target llama-server llama-bench -j
```

### 2. Fetch the model

Apache-2.0. The file is ProCreations'
[Ternary-Bonsai-2-27B-MTP](https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-MTP):
PrismML's 851 tensors byte for byte, plus a 15-tensor MTP draft head in Q8_0.

```bash
hf download ProCreations/Ternary-Bonsai-2-27B-MTP Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf \
  --revision efffdea64c1f9e93cc7fa6bb24f72ae9d66ecf51 --local-dir models
echo "3cb3f0056d2e34ee44245a64396004a21f8492573d6ce1266ec4b7222c131dd4  models/Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf" | sha256sum -c
```

### 3. Serve

For a 16 GB card, four conversations share one 294,912-token pool. Each may use the model's
full 262,144-token window.

```bash
build/bin/llama-server -m models/Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf --alias bonsai-2-27b \
  -ngl 99 -fa on -c 294912 -np 4 --kv-unified --jinja \
  --cache-type-k q4_0 --cache-type-v q4_0 -cts q8_0 --attn-mask-bits \
  --chat-template-file docs/torad/bonsai-27b/chat-template.jinja --reasoning-format deepseek \
  --spec-type draft-mtp --spec-draft-n-max 2 -ctkd q4_0 -ctvd q4_0 \
  --checkpoint-every 16384 --cache-ram 8192 \
  --temp 1.0 --top-p 0.95 --top-k 20 --host 127.0.0.1 --port 8080
```

`chat-template.jinja` is the model's own template with one change: a system message after the
first turn renders as a system turn. Claude Code sends those.

On a bigger card, keep each conversation's window at 262,144 and add conversations to the
shared pool. The 32 GB and 96 GB rows are computed by rig from VRAM constants measured on an
RTX 5080:

| card | `-c` | `-np` |
|---|---|---|
| 16 GB (RTX 5070 Ti, 5080) | 294912 | 4 |
| 32 GB (RTX 5090) | 786432 | 8 |
| 96 GB (RTX PRO 6000 Blackwell) | 3538944 | 16 |

To put the server behind Claude Code, use [splice](https://github.com/torad-labs/splice). The
Bonsai page has the `splice.toml` and the three settings that were measured.

### 4. Benchmark

The depth table, on the binary from step 1:

```bash
docs/torad/bonsai-27b/bench-matrix.sh build/bin models/Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf
```

That runs `llama-bench -ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -p 512 -n 128` at depths 0, 16384,
65536, and 131072, three repetitions, no draft head.

The 104.8 tok/s row is [`bench-server.sh`](docs/torad/bonsai-27b/bench-server.sh) against the
fork-mtp leg of
[`served.sh`](docs/torad/bonsai-27b/runs/2026-09-22-headline/served.sh). Flags that row depends
on: `-ngl 99`, `-fa on`, `--cache-type-k q4_0`, `--cache-type-v q4_0`, `-cts q8_0`,
`--attn-mask-bits`, `--spec-type draft-mtp`, `--spec-draft-n-max 2`, `-ctkd q4_0`,
`-ctvd q4_0`, and `-c 65536 -np 1`. The client sends temperature 0 and top_k 1, so the row is
greedy. The recorded script used port 18090. Start the server, then run the client in another
shell:

```bash
build/bin/llama-server -m models/Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf --alias bonsai-2-27b \
  -ngl 99 -fa on -c 65536 -np 1 --jinja \
  --cache-type-k q4_0 --cache-type-v q4_0 -cts q8_0 --attn-mask-bits \
  --chat-template-file docs/torad/bonsai-27b/chat-template.jinja --reasoning-format deepseek \
  --spec-type draft-mtp --spec-draft-n-max 2 -ctkd q4_0 -ctvd q4_0 \
  --temp 1.0 --top-p 0.95 --top-k 20 --host 127.0.0.1 --port 8080
```

```bash
docs/torad/bonsai-27b/bench-server.sh 8080 fork-mtp
```

Step 3 is the command to leave running on a 16 GB card. It uses `-c 294912 -np 4` and also
passes `--kv-unified`, `--checkpoint-every 16384`, and `--cache-ram 8192`, which keep several
conversations resident. The server command above is the one the 104.8 tok/s row measured.

Both commands leave `--kv-mean-center` unset, and so did the headline runs. When to set it is
the next section.

## q4_0 cache

The message on [`60feea0`][60feea0] says the native q4_0 flash-attention read needs
`--kv-mean-center` for a usable 4-bit K cache at long context. The kernel selects that read in
[`ggml_cuda_fattn_mma_raw`](ggml/src/ggml-cuda/fattn.cu) from the cache type (q4_0 K and V),
head size 256, and GQA above 4. `GGML_CUDA_FATTN_Q4_0_LEGACY=1` is the off switch.

`--kv-mean-center FNAME` is a separate option. The argument is a bias file from
[`tools/kv-mean-center`](tools/kv-mean-center/README.md). After the file is loaded, `cpy_k()`
subtracts that bias before the q4_0 store. Context creation accepts the flag when
`--cache-type-k` is q4_0. Set it when you have a bias calibrated with that same cache type
([docs/kv-mean-center.md](docs/kv-mean-center.md)).

A quantized K cache whose head size is a multiple of 64 is Hadamard-rotated unless
`LLAMA_ATTN_ROT_DISABLE=1`. This model's head size is 256, and the scripts in
[`docs/torad/bonsai-27b/runs/`](docs/torad/bonsai-27b/runs/) leave that variable unset, so the
headline runs rotated K. On a hybrid-attention model, the tool's README reports mean KLD
against an f16 K cache of 0.00144 with rotation alone and 0.00111 with rotation plus a bias
calibrated with `-ctk q4_0`. A bias recorded in the other basis is rejected at load.

## What's different from upstream

Each row is one commit. The Bonsai page has the measurement on this model.

| change | commit | off switch |
|---|---|---|
| Tensor-core flash attention reads a q4_0 K/V cache directly, with int8 QK | [60feea0][60feea0] | `GGML_CUDA_FATTN_Q4_0_LEGACY=1` |
| Chunked tensor-core Gated DeltaNet prefill, a port of [ggml-org#26001](https://github.com/ggml-org/llama.cpp/pull/26001) | [b120f63][b120f63] | `GGML_CUDA_GDN_CHUNKED=0` |
| The MTP draft head reads the pack's Hadamard-latent token table through the inverse rotation | [514c53c][514c53c] | omit `--spec-type draft-mtp` |
| Recurrent state cache in a narrower type, `-cts q8_0`. The update math stays f32 | [a941add][a941add] | omit `-cts` (f32) |
| Attention mask as one bit per KV cell, `--attn-mask-bits` | [9e358e6][9e358e6] | omit the flag |
| A small float matrix the tiled kernel cannot take goes to the vector kernel | [55b7db6][55b7db6] | `GGML_CUDA_MMVF_UNTILED_LEGACY=1` |
| GeForce Blackwell takes the transposed conv-state concat and the float4 scale | [d371dff][d371dff] | `GGML_CUDA_CONCAT_TRANSPOSE_SM120_LEGACY=1`, `GGML_CUDA_SCALE_VEC4_SM120_LEGACY=1` |
| A pinned context checkpoint every N prompt tokens, and one at the token a prompt forked, `--checkpoint-every` | [aa8ef36][aa8ef36] | omit the flag |

[aa8ef36][aa8ef36] is the parent of [60feea0][60feea0]. The Bonsai page lists the checkpoint
with `60feea0`, which is where both landed on this branch.

The full list is [TORAD.md](TORAD.md).

## Contributing

Issues and pull requests for this fork:
[issues](https://github.com/torad-labs/llama.cpp/issues),
[pull requests](https://github.com/torad-labs/llama.cpp/pulls).

A change here is one commit with its own measurement and its own off switch, so it can be
rebased, dropped, or sent upstream on its own. [TORAD.md](TORAD.md) is that list.
[CONTRIBUTING.md](CONTRIBUTING.md) is the upstream llama.cpp rule set, including the note on
disclosing AI-written changes.

## License

[MIT](LICENSE). Copyright (c) 2023-2026 The ggml authors. Same license as
[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp). Third-party notices are in the
[upstream README](docs/upstream-readme.md#acknowledgements).

## Upstream

[PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp) is the base. This fork merges
that line when it takes a newer base (`prism` remote). PrismML tracks
[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp).

A fix that belongs upstream can go as its own commit. Opening it there is a separate pull
request. The maintained description is the last section of [TORAD.md](TORAD.md).

## Documentation

- [Ternary Bonsai 2 27B](docs/torad/bonsai-27b/README.md): the runs, the method, splice
- [Build](docs/build.md)
- [llama-server](tools/server/README.md)
- [Docker](docs/docker.md)
- [Upstream README](docs/upstream-readme.md): what llama.cpp is, backends, tools

[c008fe8]: https://github.com/torad-labs/llama.cpp/commit/c008fe8cf5d8086586d3527c41868d0da8781f39
[c54ace8]: https://github.com/torad-labs/llama.cpp/commit/c54ace84bb817d88031c2181e6e3689e1733b8de
[60feea0]: https://github.com/torad-labs/llama.cpp/commit/60feea00bdfe4bc47a2ba16ab153968c34084ea8
[b120f63]: https://github.com/torad-labs/llama.cpp/commit/b120f63c003649a4dc9939af4df2e1134d606d1a
[514c53c]: https://github.com/torad-labs/llama.cpp/commit/514c53c14bf6c4aa8eed8805a34a823938a0247b
[a941add]: https://github.com/torad-labs/llama.cpp/commit/a941add71eca6c52851422eb30d6f8e275b9cc26
[9e358e6]: https://github.com/torad-labs/llama.cpp/commit/9e358e628c307255392b5b68287c57d60eaf735d
[55b7db6]: https://github.com/torad-labs/llama.cpp/commit/55b7db6e20d67ea0ec1b635c2980f37836df7e88
[d371dff]: https://github.com/torad-labs/llama.cpp/commit/d371dffe4b84c62576a575fd9455e8beb95fc599
[aa8ef36]: https://github.com/torad-labs/llama.cpp/commit/aa8ef36b689de6dbeeeebe40bfd4990670738034
