# torad-labs/llama.cpp

`main` is PrismML's llama.cpp at tag `prism-b10685-7dffb15` (itself ggml-org/llama.cpp b10685
plus the ternary PQ/PTQ kernels) with the commits below on top. Each is one self-contained change
with its own measurement and its own off switch, so any of them can be rebased, dropped or sent
upstream on its own. Nothing here is model-specific: the examples are Ternary Bonsai 2 27B because
that is the model these were measured on. [docs/torad/bonsai-27b](docs/torad/bonsai-27b/README.md)
has that model against PrismML's latest release, with every command to reproduce it and to serve
it to Claude Code.

This repo carries kernels and server features only. How a build is produced per GPU, how a model
is fetched, derived, gated and served, lives in [torad-labs/rig](https://github.com/torad-labs/rig),
which pins a commit of this branch as a submodule.

| commit | change | off switch |
|---|---|---|
| in `da69dc5` | `--checkpoint-every N`: a pinned context checkpoint every N prompt tokens and one at the exact token a prompt forked, so a compaction or a new session on a recurrent/hybrid model resumes from a checkpoint instead of re-prefilling from zero | omit the flag |
| `da69dc5` | tensor-core flash attention reads a q4_0 K/V cache natively (head size 256, GQA > 4) with int8 Q·K; decode uses it from 4,096 tokens on. RTX 5080: prefill at 131K 930 → 1,382 tok/s, decode at 131K 44.5 → 73.3 tok/s | `GGML_CUDA_FATTN_Q4_0_LEGACY=1` |
| `f0d83e0` | a rank-1 LoRA fused into one decode launch per adapted weight (dot + scaled outer add) instead of four; `test-backend-ops LORA_RANK1` checks the graph against CPU | `GGML_CUDA_LORA_RANK1_FUSE=0` |
| `8498fa7` | a small float matrix mmf cannot tile (rows not a multiple of 32) takes the vector kernel instead of cuBLAS's ~26 µs 4-block GEMM while rows × cols ≤ 512: the 48-row `ssm_alpha`/`ssm_beta` of every Gated DeltaNet layer at each MTP verify. RTX 5070 Ti, served head with MTP: 83.3 → 90.1 / 90.8 tok/s (A-B-A) | `GGML_CUDA_MMVF_UNTILED_LEGACY=1` |
| `8ea0ee2` | chunked tensor-core Gated DeltaNet prefill, a port of ggml-org/llama.cpp#26001 (16-token chunks: f32 forward substitution, fp16 WMMA Q·Kᵀ and state/output with f32 accumulation, f32 state); reads qwen35's q/k/v views and raw gates in place and leaves the last K-1 tokens of a ubatch on the recurrent kernel, so MTP rollback snapshots are unchanged. RTX 5070 Ti: GDN per 512-token ubatch 53.3 → 15.6 ms, prefill pp512 at 32,768 1,330 → 1,446 tok/s; KL against the recurrent kernel 0.00148 (the int8 Q·K bar: 0.00150) | `GGML_CUDA_GDN_CHUNKED=0` |
| `b4c2d66` | GeForce Blackwell (sm_120) takes two paths that were gated to the DGX Spark: the transposed dim-0 concat for a Gated DeltaNet layer's conv state (3 columns) with the new rows, 11.50 → 1.77 µs per layer at an MTP verify, and the float4 scale, 6.10 → 5.14 µs. RTX 5070 Ti, 3-row verify step: GPU time 20.0 → 19.4 ms (−2.8%), logits identical (KL at the protocol's noise floor, same top p 100 %) | `GGML_CUDA_CONCAT_TRANSPOSE_SM120_LEGACY=1`, `GGML_CUDA_SCALE_VEC4_SM120_LEGACY=1` |
| `8f64f83` | a q8_0 recurrent-state cache (`-cts q8_0`) takes the fused snapshot write: the Gated DeltaNet kernel quantizes each warp-wide slice of a state column to one q8_0 block as it stores it (cpy's formula, so the cache holds the same bytes) and the per-layer f32 → q8_0 cpy (~14 µs) is skipped at decode and MTP verify; the chunked prefill keeps its cpy. RTX 5070 Ti, GPU time per step with one state written: decode 14.15 → 13.40 ms (−5.3 %), a 3-token ubatch 16.94 → 16.57 ms; served with the MTP draft head, where a verify writes K = 3 snapshots per layer (three cpys before this), legacy / fused A-B-A-B 104.8 / 118.1 / 110.1 / 120.2 tok/s (+9-10 %) with the same draft acceptance and the same text byte for byte; logits identical (KL at the protocol's noise floor, same top p 100 %) | `GGML_CUDA_GDN_Q8_CACHE_LEGACY=1` |

Every switch in the last column is read once per process and parses as an integer: a `*_LEGACY` switch set to `0`
is the same as unset (the change stays on), and `=0` turns off `GGML_CUDA_LORA_RANK1_FUSE` and
`GGML_CUDA_GDN_CHUNKED`.

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
