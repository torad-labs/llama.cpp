#!/usr/bin/env bash
# Prompt processing and decode at increasing context depth, with the KV cache the served head uses
# (q4_0 K and V, flash attention on). Three repetitions per cell; llama-bench prints mean and stddev.
#
#   ./bench-matrix.sh <build/bin> <model.gguf> [depths]     # depths default: 0,16384,65536,131072
set -euo pipefail

bin=${1:?usage: bench-matrix.sh <build/bin> <model.gguf> [depths]}
model=${2:?usage: bench-matrix.sh <build/bin> <model.gguf> [depths]}
depths=${3:-0,16384,65536,131072}

"$bin/llama-bench" -m "$model" -ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 \
  -p 512 -n 128 -d "$depths" -r 3 -o md
