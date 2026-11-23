#!/bin/sh
# kv1b attribution runner for the CloudRift box. Invoked through on-box as separate
# whitespace-free tokens, e.g.:  on-box sh scripts/inkling2/kv1b_run.sh dirty
# The helper fuses any argument that holds spaces into one command name, so every
# token here stays bare; per-variant file sets arrive via $2/$3 below.
#
# usage: sh scripts/inkling2/kv1b_run.sh TAG [BASE_REF] [RESTORE_COMMA_LIST]
#   TAG              label for /tmp logs (a-z0-9- only)
#   BASE_REF         ref to take restored files from (default ad51d800a86a5a8e31762cee70232cb057dc73c3)
#   RESTORE_LIST     comma-separated repo-relative paths written back to their BASE_REF content before
#                    building; empty entries are an error, a failed write stops the run, and the index is
#                    never touched (the tree this runs in is shared and dirty by design)
set -u
export PATH=/usr/local/cuda-13.3/bin:$PATH
TAG=${1:-run}
BASE=${2:-ad51d800a86a5a8e31762cee70232cb057dc73c3}
RESTORE=${3:-}

if [ -n "${RESTORE:-}" ]; then
    echo "=== $TAG restoring paths to $BASE"
    oldIFS=$IFS
    IFS=,
    for p in $RESTORE; do
        if [ -z "$p" ]; then
            echo "empty entry in the restore list (check the commas): refusing to touch more than was asked" >&2
            exit 13
        fi
        mkdir -p "$(dirname "$p")"
        git show "$BASE:$p" > "$p" || {
            echo "restore of '$p' from $BASE failed: stopping instead of testing the unrestored tree" >&2
            exit 14
        }
    done
    IFS=$oldIFS
fi

echo "=== $TAG window-base $(git rev-parse HEAD) $(date -u +%H:%M:%S)"  # the known-committed anchor every arm names in its header
echo "=== $TAG CONFIGURE/BUILD $(date -u +%H:%M:%S)"
cmake -S . -B build-cuda -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF >/dev/null 2>/tmp/cfg-$TAG.err || {
    cat /tmp/cfg-$TAG.err
    exit 11
}
cmake --build build-cuda --target test-llama-archs -j 14 >/dev/null 2>/tmp/build-$TAG.err || {
    tail -40 /tmp/build-$TAG.err
    exit 12
}
cmake --build build-cuda --target test-backend-ops -j 14 >/dev/null 2>/tmp/bops-$TAG.err || {
    tail -20 /tmp/bops-$TAG.err
    exit 13
}
echo "=== $TAG KV1B CTEST $(date -u +%H:%M:%S)"
cd build-cuda && ctest -R '^test-inkling-kv1b-' --output-on-failure
