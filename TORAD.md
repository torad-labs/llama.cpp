# torad-labs/llama.cpp

`main` is PrismML's llama.cpp at tag `prism-b10685-7dffb15` (itself ggml-org/llama.cpp b10685
plus the ternary PQ/PTQ kernels) with the commits below on top. Each is one self-contained change
with its own measurement and its own off switch, so any of them can be rebased, dropped or sent
upstream on its own. Nothing here is model-specific: the examples are Ternary Bonsai 2 27B because
that is the model these were measured on.

This repo carries kernels and server features only. How a build is produced per GPU, how a model
is fetched, derived, gated and served, lives in [torad-labs/rig](https://github.com/torad-labs/rig),
which pins a commit of this branch as a submodule.

| commit | change | off switch |
|---|---|---|
| `4651ce7` | `--checkpoint-every N`: a pinned context checkpoint every N prompt tokens and one at the exact token a prompt forked, so a compaction or a new session on a recurrent/hybrid model resumes from a checkpoint instead of re-prefilling from zero | omit the flag |
| `da69dc5` | tensor-core flash attention reads a q4_0 K/V cache natively (head size 256, GQA > 4) with int8 Q·K; decode uses it from 4,096 tokens on. RTX 5080: prefill at 131K 930 → 1,382 tok/s, decode at 131K 44.5 → 73.3 tok/s | `GGML_CUDA_FATTN_Q4_0_LEGACY=1` |
| `f0d83e0` | a rank-1 LoRA fused into one decode launch per adapted weight (dot + scaled outer add) instead of four; `test-backend-ops LORA_RANK1` checks the graph against CPU | `GGML_CUDA_LORA_RANK1_FUSE=0` |
| `8498fa7` | a small float matrix mmf cannot tile (rows not a multiple of 32) takes the vector kernel instead of cuBLAS's ~26 µs 4-block GEMM while rows × cols ≤ 512: the 48-row `ssm_alpha`/`ssm_beta` of every Gated DeltaNet layer at each MTP verify. RTX 5070 Ti, served head with MTP: 83.3 → 90.1 / 90.8 tok/s (A-B-A) | `GGML_CUDA_MMVF_UNTILED_LEGACY=1` |

Remotes for maintenance: `prism` → PrismML-Eng/llama.cpp (the base), merged in when wanted. The
83 PrismML branches this repo was first pushed with were removed on 2026-09-20 — every one was at
the same commit as PrismML's copy; they are one `git fetch prism` away.

## CI in this repo

Upstream's `.github/workflows/` (52 workflows) is deleted on `main`: this fork does not run
upstream's build matrix, and 52 third-party workflows are a supply-chain surface nobody here
audits. The org's required checks (secret scan, workflow lint, PR title) run from the org, not
from files here. When a merge from `prism` brings workflows back, delete them again in the
merge commit. Builds of this fork are made and verified by rig, per GPU.

`.gitleaks.toml` narrows three false-positive classes found in the full 10,688-commit upstream
history (Actions cache-key names, xxhash's `data_key_lo/hi`, a README placeholder), each opened
at its commit on 2026-09-20 and none live. Proven on gitleaks v8.21.2 with canaries: control 15
(12 findings + 3 canaries, no config), red 3 of 3 canaries still firing with the config — two of
them on the same line as an allowlisted value — and green 0 on `main`.
