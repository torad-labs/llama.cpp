# Ternary Bonsai 2 27B on a Blackwell GPU, driven from Claude Code

Ternary Bonsai 2 27B is PrismML's ternary Qwen3.8-27B: 64 layers (48 Gated DeltaNet + 16 full
attention), packed as PQ2_0 at 2.13 bits per weight, 7.66 GB with its MTP draft head. It fits a
16 GB GeForce card with a 262,144-token window. This fork (torad-labs/llama.cpp) is PrismML's llama.cpp plus kernels and server features
measured on that model on RTX 50-series cards. This page has the numbers and every command needed
to reproduce them, then shows how to put the server behind Claude Code with
[splice](https://github.com/torad-labs/splice).

## Headline

One RTX 5070 Ti (16 GB), the public pack, and PrismML's latest release (`prism-b10709-9a9394a`)
against this fork at `3520147`. Both were built with the flags in step 1.

**Long context is where the fork pays.** At 131,072 tokens of context it decodes 1.83× and
prefills 2.0× faster, mostly because the attention reads the q4_0 cache directly instead of
converting it to fp16 first. Prefill also carries the chunked Gated DeltaNet kernel, which is
worth +9 % at 32K on its own (row `8ea0ee2` below). From [`bench-matrix.sh`](bench-matrix.sh) (llama-bench, q4_0 K/V, flash attention
on, no draft head on either side, 3 repetitions):

| context depth | prefill (pp512), Prism → fork | decode (tg128), Prism → fork |
|---|---|---|
| 0 | 1,736 → 1,937 tok/s | 78.8 → 76.3 tok/s |
| 16,384 | 1,475 → 1,742 | 67.5 → 73.9 |
| 65,536 | 804 → 1,323 | 46.4 → 66.1 |
| 131,072 | 492 → 985 | 32.1 → 58.6 |

At depth 0 the decode rows are within the card's drift. A-B-A-B with 5 repetitions each gives
Prism 78.3 / 77.2 and the fork 77.6 / 75.8, and setting every fork off switch leaves 75.7
([`d0-decode.tsv`](runs/2026-09-22-headline/d0-decode.tsv)).

**Served, the MTP draft head is the fork's largest single gain.** From
[`bench-server.sh`](bench-server.sh): one conversation, 65,536-token window, six prompts of 512
tokens each, greedy. Each row is two runs, compared with Prism measured in the same session
between them, because the card's speed drifts by a few percent from session to session:

| server | decode | against Prism in the same session |
|---|---|---|
| fork with the MTP draft head (the serve command below) | **104.8 tok/s**: 85.8 on prose, 120.8 on step-by-step reasoning | **1.40×** (Prism 75.0) |
| fork, no draft head, recurrent state in f32 as Prism keeps it | 77.0 tok/s | −1.5 %, within the drift (Prism 78.2) |
| fork, no draft head, recurrent state in q8_0 (`-cts q8_0`, 3.4× less state VRAM) | 71.3 tok/s | −4.9 % (Prism 75.0) |

Prism b10709 has no MTP row because it refuses the pack's draft head at load.
[`served.sh`](runs/2026-09-22-headline/served.sh) records that refusal as its last leg, and
[`served-state-f32.sh`](runs/2026-09-22-headline/served-state-f32.sh) is the f32-state session.
The q8_0 state's cost is one extra pass per layer that quantizes it back into the cache, and the
fork's next change moves that into the Gated DeltaNet kernel.

The draft head's gain depends on the text: 47 % of drafted tokens were accepted on prose and 85 %
on arithmetic reasoning. Per-prompt rows, the servers' logs and the build hashes are in
[`runs/2026-09-22-headline/`](runs/2026-09-22-headline/).

## What the fork changes

Each change is one commit with its own measurement and an environment switch that turns it off.
[TORAD.md](../../../TORAD.md) is the maintained list; the rows that matter for this model:

| change | commit | measured on this model | off switch |
|---|---|---|---|
| Tensor-core flash attention reads the q4_0 K/V cache natively, with int8 Q·K, instead of converting the whole context to fp16 before every attention op | `da69dc5` | RTX 5080: prefill at 131K 930 → 1,382 tok/s, decode at 131K 44.5 → 73.3 tok/s | `GGML_CUDA_FATTN_Q4_0_LEGACY=1` |
| The MTP draft head reads the pack's Hadamard-latent token table through the inverse rotation, the way the trunk does | `a0af7ec` | Stock PrismML refuses the head at load; the MTP rows of the headline are what it enables | omit `--spec-type draft-mtp` |
| The recurrent (Gated DeltaNet) state cache in a narrower type, `-cts q8_0`; the update math stays f32 in the kernel | `ca9e071` | RTX 5080, 4 conversations with 2 rollback snapshots each: 1,795.5 → 526.5 MiB of VRAM | omit `-cts` (f32) |
| The attention mask as one bit per KV cell, `--attn-mask-bits` | `3d68483` | At 294,912 cells × 512: mask 288 → 18 MB (target) and 64 → 4 MB (draft); the same greedy tokens with bit-identical top-5 logprobs | omit the flag |
| A pinned context checkpoint every N prompt tokens and at the token a prompt forked, `--checkpoint-every`, so a compacted or new Claude Code session resumes from a checkpoint instead of re-reading its whole prompt | in `da69dc5` | — | omit the flag |
| A small float matrix the tiled kernel cannot take goes to the vector kernel, not a ~26 µs cuBLAS GEMM: the 48-row gate projections of every Gated DeltaNet layer at each MTP verify | `8498fa7` | RTX 5070 Ti, served with MTP: 83.3 → 90.1 / 90.8 tok/s (A-B-A) | `GGML_CUDA_MMVF_UNTILED_LEGACY=1` |
| Chunked tensor-core Gated DeltaNet prefill (a port of ggml-org/llama.cpp#26001) | `8ea0ee2` | RTX 5070 Ti: Gated DeltaNet per 512-token ubatch 53.3 → 15.6 ms, prefill at 32K 1,330 → 1,446 tok/s; KL 0.00148 against the recurrent kernel | `GGML_CUDA_GDN_CHUNKED=0` |
| GeForce Blackwell takes the transposed conv-state concat and the float4 scale, which were gated to the DGX Spark | `b4c2d66` | RTX 5070 Ti, MTP verify step: GPU time 20.0 → 19.4 ms; identical logits | `GGML_CUDA_CONCAT_TRANSPOSE_SM120_LEGACY=1`, `GGML_CUDA_SCALE_VEC4_SM120_LEGACY=1` |

`da69dc5` is this branch's first commit: PrismML's tree with the q4_0 flash attention and
`--checkpoint-every` folded in.

## Tried, not shipped

Measured on the RTX 5070 Ti and kept out of `main`, each for the reason given:

- **PTQ1_0, a lossless 1.75 bpw repack of the same ternary weights (6.40 GB instead of 7.65).**
  - The repack is exact: 5 base-3 digits per byte, dequantized bit-identical to PQ2_0.
  - One fix worked: splitting each 128-weight block across four lanes raised achieved occupancy
    from 25-33 % to 72 %.
  - Decode still loses: 61.1 / 60.5 tok/s against PQ2_0's 73.6 / 73.2 (tg128, same binary, A-B-A).
  - The kernel is issue-bound, not memory-bound. It executes 1.75-2.2× PQ2_0's instructions at
    43-60 % of DRAM bandwidth, because the base-3 digit walk costs about 10 SASS ops per `dp4a`
    against PQ2_0's 4.3. Reading 18 % fewer bytes does not pay for that on this card.
- **PQ2_0 batches of 7-8 rows on the tensor-core path (MMQ).** 7-row prompts 199 → 245 tok/s,
  8-row 214 → 267, but KL against the vector path is 0.00204 (bar 0.00150) and same top-p 98.1 %
  (bar 98.3 %).
- **Decoding each weight block once for several columns in the vector kernel.** No measurable
  change at 2-6 columns; the kernel was already at 82-94 % of DRAM bandwidth.

## Run it yourself

### 1. Build (CUDA 12.8 or newer, sm_120: RTX 50-series and RTX PRO Blackwell)

```bash
git clone https://github.com/torad-labs/llama.cpp && cd llama.cpp
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120a -DGGML_NATIVE=ON \
  -DGGML_CUDA_FA=ON -DGGML_CUDA_FA_ALL_QUANTS=OFF -DGGML_CUDA_GRAPHS=ON \
  -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF
cmake --build build --target llama-server llama-bench -j
```

### 2. Fetch the model (Apache-2.0)

The pack is ProCreations' PQ2_0 file: PrismML's 851 tensors byte for byte, plus a 15-tensor
multi-token-prediction (MTP) draft head in Q8_0 that the server uses for speculative decoding.

```bash
hf download ProCreations/Ternary-Bonsai-2-27B-MTP Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf \
  --revision efffdea64c1f9e93cc7fa6bb24f72ae9d66ecf51 --local-dir models
echo "3cb3f0056d2e34ee44245a64396004a21f8492573d6ce1266ec4b7222c131dd4  models/Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf" | sha256sum -c
```

### 3. Serve

For a 16 GB card: four conversations share one 294,912-token pool, and each may use the model's full
262,144-token window.

```bash
build/bin/llama-server -m models/Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf --alias bonsai-2-27b \
  -ngl 99 -fa on -c 294912 -np 4 --kv-unified --jinja \
  --cache-type-k q4_0 --cache-type-v q4_0 -cts q8_0 --attn-mask-bits \
  --chat-template-file docs/torad/bonsai-27b/chat-template.jinja --reasoning-format deepseek \
  --spec-type draft-mtp --spec-draft-n-max 2 -ctkd q4_0 -ctvd q4_0 \
  --checkpoint-every 16384 --cache-ram 8192 \
  --temp 1.0 --top-p 0.95 --top-k 20 --host 127.0.0.1 --port 8080
```

The fork's own flags (`-cts`, `--attn-mask-bits`, `--checkpoint-every`) and the MTP draft head are
rows in the table above, each with its number; the rest are stock llama-server flags. This command,
as written, loads on an RTX 5070 Ti in 13,586 MiB and answers a streamed tool call with its usage
frame ([`readme-command.sh`](runs/2026-09-22-headline/readme-command.sh),
[`readme-command.txt`](runs/2026-09-22-headline/readme-command.txt)). On a bigger card, keep each
conversation's window at 262,144 and add conversations to the shared pool. The larger rows are
computed by [rig](https://github.com/torad-labs/rig) from VRAM constants measured on the RTX 5080,
not measured on those cards:

| card | `-c` | `-np` |
|---|---|---|
| 16 GB (RTX 5070 Ti, 5080) | 294912 | 4 |
| 32 GB (RTX 5090) | 786432 | 8 |
| 96 GB (RTX PRO 6000 Blackwell) | 3538944 | 16 |

`chat-template.jinja` in this directory is the model's own template with one change. A system
message after the first turn renders as a system turn instead of raising. Claude Code sends those
mid-conversation.

### 4. Claude Code, through splice

[splice](https://github.com/torad-labs/splice) is a local gateway that speaks Anthropic's Messages API
to Claude Code and the OpenAI chat dialect to llama-server. Install it (Java 21+, with Claude Code
on PATH), register the heads, and restart the daemon:

```bash
curl -fsSL https://github.com/torad-labs/splice/releases/latest/download/install.sh | bash
splice install --all
```

Then add the local server to `splice.toml`:

```toml
[providers.bonsai]
dialect = "openai-chat"
base_url = "http://127.0.0.1:8080/v1"
auth = { kind = "api-key", env = "BONSAI_API_KEY" }   # llama-server ignores it; any value works

[providers.bonsai.quirks]
reasoning_effort = false   # see below
slot_affinity = true       # splice v0.4.0 and later; drop this line on v0.3.x

[[providers.bonsai.models]]
id = "bonsai-2-27b"                 # exactly what GET /v1/models lists (the --alias above)
label = "Ternary Bonsai 2 27B (local)"
context_window = 262144             # at or under the server's per-conversation window

[heads.bonsai]
provider = "bonsai"
port = 3106
discovery_prefix = "claude-bonsai--"
pinned_model = "bonsai-2-27b"

[heads.bonsai.claude]
command = "claude-bonsai"
```

Launch it with `claude-bonsai`. Check it with `splice doctor --live`, which sends one small
streamed tool call to the model, so tool support is shown rather than assumed.

Three settings carry measurements:

- **`reasoning_effort = false`.** The model's template accepts `low`, `medium` and `xhigh`, and
  raises on any other value. Against llama-server, `high` and `max` each return HTTP 500
  ("Unexpected reasoning effort"), and those are values Claude Code sends. With the quirk, splice
  omits the field and the template's default, `xhigh`, applies.
- **`slot_affinity`.** llama-server assigns a request to the slot whose cached prompt is most
  similar. Every Claude Code session opens with the same ~30K-token preamble, so a second session
  can take over a live conversation's slot, and that conversation then re-reads its whole prompt.
  splice sends `id_slot` per conversation to prevent it. The quirk ships in splice v0.4.0; on
  v0.3.x, leave the line out and a second session can still cost a re-read.
- **Streamed tool calls and usage.** On this server, with the template above, a streamed tool
  call arrives as `tool_calls` deltas with `finish_reason: tool_calls`, followed by a usage frame.
  That includes a conversation with a system message after the first turn. Claude Code needs the
  usage frame to count tokens and compact on time.

## Method

- **Card and build:** RTX 5070 Ti 16 GB (driver 610.43.02), shared with a desktop session. Both
  runtimes were built with the flags in step 1, and each run records the sha256 of the build's
  `libggml-cuda.so` beside its numbers.
- **Baseline:** PrismML's llama.cpp at its latest release tag, `prism-b10709-9a9394a`, built
  unmodified. It has no `-cts`, `--attn-mask-bits` or `--checkpoint-every`, so its served runs use
  the same command without those flags. It also refuses this pack's MTP draft head at load
  ([`server-prism-mtp.log`](runs/2026-09-22-headline/server-prism-mtp.log)): the head reads the
  pack's Hadamard-latent token table without the inverse rotation, and the fork's `a0af7ec` adds it.
  So the engine comparison runs with MTP off on both sides, and the MTP gain is measured on the
  fork alone, on top of that.
- **Throughput:**
  - `bench-matrix.sh` (llama-bench): prompt processing and decode at depth, 3 repetitions per cell.
  - `bench-server.sh`: decode as a client sees it, from the server's own timings. Greedy, prompt
    cache off, one discarded warm-up, four prompts with thinking off and two with it on.
  - Runtimes are measured A-B-A where the numbers are close, with no other compute job on the card.
  - The scripts in [`runs/2026-09-22-headline/`](runs/2026-09-22-headline/) take `PRISM_BIN`,
    `FORK_BIN`, `MODEL` and `GPU` from the environment.
- **Kernel changes:**
  - Measured with Nsight Systems as GPU time per step (`--cuda-graph-trace=node`), and with Nsight
    Compute for occupancy and DRAM throughput.
  - Checked against the CPU backend with `test-backend-ops`, with the change on and with its off
    switch.
- **Quality bar for a change that alters numerics:** KL divergence
  (`llama-perplexity --kl-divergence`, 4,096 scored tokens) against the path it replaces, at or
  under the KL of the int8 Q·K flash attention against its f16 path: mean 0.00150, same top-p
  ≥ 98.3 %. A change that does not alter numerics shows KL at the protocol's noise floor (99 %
  KLD 0.000035, from the base file's rounding).

Raw outputs of every run named here are in [`runs/`](runs/).
