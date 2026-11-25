# Qualifying Inkling on four RTX PRO 6000 Blackwell GPUs

## Serve the qualified configuration

For the original abliterated release `e4853745`, place the Q3_K_M GGUF and `mmproj-F16.gguf` together in the pack directory. The `tensor-4` profile defaults to a 3,014,656-token shared F16 pool, 23 slots and 256 GiB of host session cache.

```sh
python3 scripts/inkling2/serve.py --profile tensor-4 \
    --packs /srv/models/inkling-2-small-abliterated-q3km \
    --server "$PWD/build/bin/llama-server"
```

The per-session window is 294,912 tokens. A 262,144-token input leaves 32,768 tokens for reasoning and output. The server has no additional reply cap. Automatic slot selection allows active text sessions to share a common prefix and retain distinct continuations. The server takes a rollback checkpoint every 32,768 prompt tokens and uses sliced host transfers with BF16 reductions. See the [physical results](../../docs/torad/research/inkling-2-small/12-four-card-serving.md) for independent versus shared prompts, cold-reader latency, and numerical behavior after eviction.

### The routed experts over four cards

Under `-sm tensor` each card holds a share of every routed expert's hidden width of 2,304, which is the down projection's K. Q3_K_M's Q4_K and Q5_K down projections come in blocks of 256, which four cards can only share 512/512/512/768, and every layer waits for the card with 768: it reads 1.33x the average card's expert bytes. The engine splits such a pack by expert instead, 96 of the 384 experts whole on each card (`LLAMA_INKLING2_EXPERT_SPLIT_LEGACY=1` keeps the uneven rows). How even the cards are then follows each step's routing: by the stock imatrix's expert counts, the busiest card reads about 1.15x the average at 23 rows and 1.07x at a drafted step's 92.

A down projection in blocks of 32 is split evenly, 576 of the width a card, at every row count. The pack's down type is a choice of quality against size; `pack.py` keeps Q3_K_M's (`--expert-down k-quant`, the default) unless told otherwise:

| down projections | how | quality: KL from BF16 | size | KV pool | one card's experts, 23 / 92 rows |
|---|---|---|---|---|---|
| Q4_K, Q5_K in layers 0-2 (served) | `pack.py`, default | 0.0342 | as served | as served | 1,844 / 5,108 us (by expert: about 1,800 / 4,100) |
| IQ4_NL, Q5_0 in layers 0-2 | `pack.py --expert-down iq4_nl` | 0.0389 | as served | as served | 1,450 / 4,312 us |
| Q4_1, Q5_1 in layers 0-2, from BF16 | `pack.py --expert-down q4_1` | 0.0320 | +12.2 GB, 3.06 GB a card | about 370,000 fewer tokens (12%) | 1,467 / 4,313 us |
| Q4_1, Q5_1 from the served pack | `even_down.py PACK OUT` | as served (each scale rounded once to f16) | +12.2 GB, 3.06 GB a card | about 370,000 fewer tokens (12%) | 1,467 / 4,313 us |
| Q4_0, Q5_0 in layers 0-2 | `pack.py --expert-down q4_0` | 0.0541 | as served | as served | 1,418 / 4,251 us |

The quality column is the down projections' KL divergence from BF16 on Gemma 4 E2B with every other weight Q8_0 and an imatrix, on text the imatrix did not see: a stand-in, as no Inkling weights were at hand. The KV pool column is the tensor-4 profile's 3,014,656-token F16 pool, which has to give up the extra weights' memory. The times are one RTX 5060 Ti's gate/up and down products for one layer on one card, the busiest card for the served pack. `even_down.py` needs no source checkpoint; it reads the served pack and writes a new one.

## Run qualification

Run from the checkout on the test host. Use the server and adjacent CUDA libraries built from this fork, the calibrated Q3_K_M GGUF for the pinned abliterated revision, and its F16 projector. For a split GGUF, use `--model /path/to/first-shard.gguf`. Use `--mmproj` when the projector has a different filename.

At startup `tensor-4` allocates 92 GiB for the eight global KV layers and 3.91 GiB for the 40 sliding-window layers across the four GPUs. Each slot also has about 6.19 MiB of convolution state. The 23 sessions draw from the same global pool. At the 294,912-token ceiling, it holds ten independent full-length histories.

### Native draft

Convert the pinned Inkling `mtp.safetensors` alongside its tokenizer and configuration, then give the draft its own copies of the target's vocabulary weights. The conversion reads only the draft weights. The second command copies the embedding, embedding norm and output tensors from the exact target GGUF in the pack.

```sh
python3 convert_hf_to_gguf.py /path/to/pinned-inkling-config-and-mtp --dspark --outtype bf16 --outfile draft-bf16.gguf
python3 scripts/glm53/dflash_own_head.py \
    --draft draft-bf16.gguf \
    --target "$PACKS/Inkling-2-Small-abliterated-e4853745-Q3_K_M-imatrix.gguf" \
    --out "$PACKS/Inkling-2-Small-abliterated-e4853745-DSpark-BF16-own-head.gguf"
```

Pass `--draft` to load the paired DSpark draft. The flag also applies the draft layout: Q8_0 global KV, Q8_0 sliding KV and a 256-token physical microbatch. By default the draft is split across `CUDA0,CUDA1,CUDA2,CUDA3` and the target keeps the full 3,014,656-token pool. `--draft-device` takes `all`, a comma list of distinct cards, or one card; a draft parked on one card shrinks the pool to 2,097,152 tokens with Q8_0 KV or 1,310,720 with F16 KV, because the draft's own KV grows with the pool and the full pool beside the whole draft filled CUDA3 (97,258 of 97,887 MiB) and evicted CUDA graphs. `--context` overrides either. The draft uses Q8_0 KV and proposes up to three tokens; the server verifies each proposal with the target's sampling settings. `--draft-layout` applies the same layout and pool without a draft, as the matched control. `tensor-4` without the flag is the qualified serving configuration.

`--draft-depth` and `--draft-device` select other configurations. Use `--draft-model` when supplying a different target path. `--no-draft` is the plain serving control.

For repeated text prefixes, let the server choose each request's slot. When the server can fork a matching active request, the sessions share its KV cells and evaluate their own suffixes. The prefix must cover at least 256 tokens and exceed the configured similarity threshold (10% by default). Text chat can share with the vision projector loaded. The active-prefix path skips requests containing media. The pinned-slot qualification phases exercise independent sessions. Use `--no-prefix-copy` to disable prefix sharing.

The host prompt cache allocates serialized states as sessions enter it. Allow host memory beyond the budget for live rollback checkpoints, spare checkpoint buffers, the process, model mapping and the operating system. An idle session can stay on its GPU slot until another request reuses the slot or the device pool comes under pressure.

```sh
python3 scripts/inkling2/qualify.py \
    --profile tensor-4 \
    --server "$PWD/build/bin/llama-server" \
    --packs /models/Inkling-2-Small-abliterated-e4853745 \
    --sha256-manifest /models/Inkling-2-Small-abliterated-e4853745/SHA256SUMS \
    --out /results/inkling-tp4-01 --backup-dir /persistent/results \
    --seconds 14400
```

For an exported checkout without Git metadata, add `--source-head FULL_GIT_REVISION --source-bundle-sha256 DEPLOYED_TAR_SHA256`. Passing `--source-bundle /path/to/source.tar.gz` instead computes the digest on the host. With both options, the runner compares the digests. Use `--server-log-link /workspace/inkling/results/server.log` for a monitor following the filename across route restarts.

Keep the checksum manifest with the conversion record that identifies the source revision and imatrix. Its paths are relative to the manifest. Omitting `--sha256-manifest` records model paths, sizes and modification times. A local backup survives removal of the test machine only when the destination is an independently persistent mount. The runner owns its localhost server, releases it at exit and leaves instance billing to the operator.

Use `--dry-run` to inspect the route commands before loading any weights. For a short first pass, add `--phases chat balanced`. Reuse the same command and result folder with `--resume --phases long restore` for the remaining work. Completed stages are reused only with matching host, software, model-file metadata and workload settings. A fresh result folder keeps an independent repetition. Reversing `--routes engine host-sliced host` changes the measurement order.

| Phase | Checks and measurements |
| --- | --- |
| chat | Renders the seven native effort settings. Scores a Python aliasing problem at disabled and high effort through OpenAI Chat Completions, OpenAI Responses and Anthropic Messages. High effort must produce separated reasoning and the correct final answer. Checks a model-initiated tool call, its argument and the follow-up answer through both chat APIs. |
| balanced | Pins 23 simultaneous streams to distinct slots. Each receives 128 generated tokens by default. Saves each client's token IDs, delivery timestamps, TTFT and timings. Computes per-client and aggregate delivery rates within the interval shared by all 23 streams. |
| long | Inserts two distinct approval identifiers at 17% and 73% of a 262,144-token repository-audit prompt. Requires a cold read and retrieval of both identifiers in order while 22 other streams span the reader's first token and continue receiving tokens. Records the reader's TTFT, engine timings, each peer's delivery rate and gaps. The default requests 64 reader output tokens. Use `--reader-output-tokens 32768` in a fresh result folder to fill the whole slot. This fixed-length load deliberately continues generation past EOS. |
| restore | Uses the same 262,144-token prefix plus an 18-token suffix in independent cold, resident and RAM-restored continuations. Checks the full cached length, the save and restore log entries for the right slots, and restored versus resident top-32 log probabilities. Records differences against cold separately. |

The restore phase also records the original and independently computed resident prefix distributions before either continuation. To compare the RAM transfer against a blocking file restore, add `--restore-reference-dir /path/to/local/scratch`. The 262,144-token F16 snapshot needs about 9 GiB before periodic checkpoints. Allow at least 12 GiB of temporary disk space with the default 32,768-token checkpoint interval. The runner snapshots the original prefix, displaces it, continues the RAM-restored session, then reloads the snapshot into the same destination slot and repeats the suffix with the other sessions untouched. It removes the reference file after use. Both the original cross-slot comparison and the file control must pass the existing tolerance.

`identity.json` and `topology.txt` describe the assets, executable, loaded-library candidates, four GPUs and peer-access matrices. Every route/attempt retains its startup log, GPU memory snapshot, raw client evidence and per-phase PCIe RX/TX samples from `nvidia-smi dmon`. Inspect the samples for driver support. `comparison.json` tracks delivery and TTFT ratios against the host baseline, per-slot greedy divergence and cold/resident/restored top-32 differences. A failing phase retains its evidence and exits with a nonzero status. Rerunning with `--resume` creates a fresh server for unfinished phases.

## Decode settings search

`search.py` looks for the settings a profile decodes fastest with. A setting is a launcher option (`--draft-depth`, `--no-draft`) or a variable of `tunables.toml` (`LLAMA_SAMPLE_THREADS`, the relative decode's stream counts). Every one of them is read when the server starts, so a trial is one server start.

```sh
python3 scripts/inkling2/search.py run \
    --profile tensor-4 \
    --server "$PWD/build/bin/llama-server" \
    --packs /models/Inkling-2-Small-abliterated-e4853745 \
    --build "$(git rev-parse --short HEAD)" \
    --out /results/decode-search-01
```

A trial pins one coding request to each of the profile's sessions and lets every session generate the same number of tokens with the server's own sampling. Its rate is the tokens a second delivered while all sessions decode, the median of the measured rounds after a warm round. The first trial is the profile as it launches. The search then walks one setting at a time and keeps a value only when it beats the best so far by `--gain` (2%). `--context N` puts about N tokens of this repository's scripts before each request, for the settings whose effect grows with the context a session decodes over; a search folder holds the trials of one context. `--knob NAME=V1,V2` replaces the default space (`--knob=--draft-depth=2,4` for an option) and `--baseline` measures the profile alone.

Each trial is a line of `trials.jsonl` with its settings, rate, the share of drafted tokens the target kept, and the server's error when it did not start. A trial folder keeps the launch plan and the server log. Rerunning the same command skips finished trials. `best.json` holds the settings kept so far.

`search.py log --out DIR` sends each trial to MLflow once, as a run with its settings as parameters, from the machine that holds `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD`. `search.py apply --out DIR` writes the kept settings into `tuned.json` beside `serve.py`, which then launches the profile with them. An option on the command line still wins, and `--tuned /dev/null` launches the profile without them. `search.py load --url URL` sends the same load to a server that already runs, for a profiler to watch.
