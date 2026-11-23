#!/usr/bin/env bash
# Runs one binary's MMQ case matrix: tag names the build (base|cand|...).
# Usage: bench.sh <tag> <perf|ref|all>
set -u
cd "$HOME/src" || exit 1
tag=$1; what=${2:-all}
BIN=./build/bin/mmq-microperf
out="$HOME/work/results-$tag.txt"
: > "$out"

say() { printf '%s\n' "$*" | tee -a "$out"; }

run() {
    say "== $*"
    "$BIN" "$@" 2>&1 | tee -a "$out"
}

if [ "$what" = perf ] || [ "$what" = all ]; then
    # natural widths at the served row counts (J picked by the spread rule: 128 rows J8, 256 J16, 512 J24, 1024 J40, 2048 J64)
    for tok in 128 256 512 1024 2048; do
        run q3_K 2304 4608 "$tok" 1 perf
        run q4_K 4608 2304 "$tok" 0 perf
    done
    for shape in "q5_K 2304 4608 1024 1" "q5_K 4608 2304 1024 0"; do
        run $shape perf
    done
    # forced widths at 1024 rows, the served prefill
    for J in 8 16 24 32 40 48 56 64 80 96 112 128; do
        for shape in "q3_K 2304 4608 1024 1" "q4_K 4608 2304 1024 0" \
                     "q5_K 2304 4608 1024 1" "q5_K 4608 2304 1024 0"; do
            GGML_CUDA_MMQ_MOE_NCOLS=$J run $shape perf
        done
    done
fi

if [ "$what" = ref ] || [ "$what" = all ]; then
    # numeric check against the CPU backend at the served J and at every width the table can pick
    for shape in "q3_K 256 1024 1024 1" "q4_K 512 1024 1024 0" \
                 "q5_K 256 1024 1024 1" "q5_K 512 1024 1024 0"; do
        run $shape ref
    done
    for J in 8 16 24 48 64 128; do
        for shape in "q3_K 256 1024 1024 1" "q4_K 512 1024 1024 0"; do
            GGML_CUDA_MMQ_MOE_NCOLS=$J run $shape ref
        done
    done
fi
say "DONE $tag"
