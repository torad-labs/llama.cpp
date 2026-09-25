#!/usr/bin/env bash
# Depth matrix, PrismML prism-b10709-9a9394a vs torad-labs/llama.cpp c008fe8, public pack, RTX 5070 Ti.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
# PRISM_BIN, FORK_BIN: the two builds' bin dirs; MODEL: the public pack (README step 2); GPU: the card's index
# (0 by default; these runs used 1, an RTX 5070 Ti)
PR=${PRISM_BIN:?set PRISM_BIN to a prism-b10709-9a9394a build/bin}; FK=${FORK_BIN:?set FORK_BIN to a torad-labs/llama.cpp build/bin}
M=${MODEL:?set MODEL to Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf}
{ echo "prism libggml-cuda $(sha256sum $PR/libggml-cuda.so | cut -c1-16)"; echo "fork  libggml-cuda $(sha256sum $FK/libggml-cuda.so | cut -c1-16)"
  echo "pack  $(sha256sum $M | cut -c1-16)"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader -i ${GPU:-0}; } > "$here/artifacts.txt"
for leg in prism fork; do B=$PR; [ $leg = fork ] && B=$FK
  CUDA_VISIBLE_DEVICES=${GPU:-0} bash "$here/../../bench-matrix.sh" "$B" "$M" > "$here/matrix-$leg.md" 2> "$here/matrix-$leg.err"
done
