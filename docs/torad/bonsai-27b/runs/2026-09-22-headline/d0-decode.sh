#!/usr/bin/env bash
# Decode at depth 0 (tg128), where the matrix shows the fork ~3 % behind Prism b10709: A-B-A-B with five
# repetitions each, then the fork with every one of its off switches set, to attribute the gap.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
# PRISM_BIN, FORK_BIN: the two builds' bin dirs; MODEL: the public pack (README step 2); GPU: the card's index
# (0 by default; these runs used 1, an RTX 5070 Ti)
PR=${PRISM_BIN:?set PRISM_BIN to a prism-b10709-9a9394a build/bin}; FK=${FORK_BIN:?set FORK_BIN to a torad-labs/llama.cpp build/bin}
M=${MODEL:?set MODEL to Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf}
tg() { "$1/llama-bench" -m "$M" -ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -p 0 -n 128 -r 5 -o csv 2> /dev/null |
       tail -1 | awk -F, -v l="$2" '{gsub(/"/,""); print l "\t" $(NF-1) "\t±" $NF}'; }
{
  tg "$PR" prism-1; tg "$FK" fork-1; tg "$PR" prism-2; tg "$FK" fork-2
  GGML_CUDA_FATTN_Q4_0_LEGACY=1 GGML_CUDA_MMVF_UNTILED_LEGACY=1 GGML_CUDA_GDN_CHUNKED=0 GGML_CUDA_LORA_RANK1_FUSE=0 \
    tg "$FK" fork-all-switches-off
} | tee "$here/d0-decode.tsv"
