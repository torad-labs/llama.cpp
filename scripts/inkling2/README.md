# Qualifying Inkling on four RTX PRO 6000 Blackwell GPUs

## Serve the qualified configuration

For the original abliterated release `e4853745`, place the Q3_K_M GGUF and `mmproj-F16.gguf` together in the pack directory. The `tensor-4` profile defaults to a 3,014,656-token shared F16 pool, 23 slots and 256 GiB of host session cache.

```sh
python3 scripts/inkling2/serve.py --profile tensor-4 \
    --packs /srv/models/inkling-2-small-abliterated-q3km \
    --server "$PWD/build/bin/llama-server"
```

The per-session window is 294,912 tokens. A 262,144-token input leaves 32,768 tokens for reasoning and output. The server has no additional reply cap. Automatic slot selection allows active text sessions to share a common prefix and retain distinct continuations. The server takes a rollback checkpoint every 32,768 prompt tokens and uses sliced host transfers with BF16 reductions. See the [physical results](../../docs/torad/research/inkling-2-small/12-four-card-serving.md) for independent versus shared prompts, cold-reader latency, and numerical behavior after eviction.

## Run qualification

Run from the checkout on the test host. Use the server and adjacent CUDA libraries built from this fork, the calibrated Q3_K_M GGUF for the pinned abliterated revision, and its F16 projector. For a split GGUF, use `--model /path/to/first-shard.gguf`. Use `--mmproj` when the projector has a different filename.

At startup `tensor-4` allocates 92 GiB for the eight global KV layers and 3.91 GiB for the 40 sliding-window layers across the four GPUs. Each slot also has about 6.19 MiB of convolution state. The 23 sessions draw from the same global pool. At the 294,912-token ceiling, it holds ten independent full-length histories.

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
