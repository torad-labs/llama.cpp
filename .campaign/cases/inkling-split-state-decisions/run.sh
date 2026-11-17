#!/bin/sh
# Cases for inkling-split-state-decisions: every inkling2 weight, cache and the convolution
# state has an explicit tensor-split decision in the meta backend. The meta split tests cover
# the Inkling tensor and cache names, and they pass.
# GPU runs use card 1 only, under /run/user/1000/gpu1.lock; card 0 serves another seat.
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$ROOT" || { echo "run.sh: cannot cd to $ROOT" >&2; exit 2; }

U=d3c3e242c16b8cbf51400fad410cd69b22d64f35
B=build-campaign
T=$(mktemp -d /tmp/inkling-split-state-decisions.XXXXXX)

fail() { printf '%s\n' "$*" > "$REASON"; exit 1; }

require_merged() {
    git cat-file -e "$U^{commit}" 2>/dev/null || fail "upstream commit $U is not in this repository"
    git merge-base --is-ancestor "$U" HEAD || fail "upstream $U is not merged into HEAD"
}

build_target() {
    require_merged
    cmake -S . -B "$B" -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120a \
        -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_EXAMPLES=OFF \
        -DGGML_CCACHE=OFF > "$T/configure.log" 2>&1 \
        || fail "cmake configure with GGML_CUDA=ON failed: $(tail -n 1 "$T/configure.log")"
    cmake --build "$B" -j "$(nproc)" --target "$1" > "$T/build-$1.log" 2>&1 \
        || fail "building $1 failed: $(grep -m1 -E 'error' "$T/build-$1.log" || tail -n 1 "$T/build-$1.log")"
    [ -x "$B/bin/$1" ] || fail "$B/bin/$1 was not produced by the build"
}

gpu1() { flock /run/user/1000/gpu1.lock env CUDA_VISIBLE_DEVICES=1 "$@"; }

# The meta split tests, run once on card 1 (two ranks on the one card); both cases read this run.
META_RC=
meta_split_run() {
    if [ -z "$META_RC" ] && [ ! -f "$T/meta.rc" ]; then
        build_target test-backend-meta-split
        gpu1 "./$B/bin/test-backend-meta-split" > "$T/meta.out" 2>&1
        echo $? > "$T/meta.rc"
    fi
    [ -f "$T/meta.rc" ] || fail "the meta split tests did not run"
    META_RC=$(cat "$T/meta.rc")
}

case_meta_split_tests_pass() {
    meta_split_run
    [ "$META_RC" -eq 0 ] || fail "test-backend-meta-split exited $META_RC: $(grep -m1 '^FAIL' "$T/meta.out" || tail -n 1 "$T/meta.out")"
    ! grep -q 'no GPU for the meta backend, skipped' "$T/meta.out" || fail "test-backend-meta-split found no GPU on card 1 and skipped"
}

case_meta_split_tests_cover_inkling() {
    meta_split_run
    grep -qi inkling "$T/meta.out" || fail "test-backend-meta-split output names no Inkling tensor or cache (exit $META_RC)"
}

run_case() {
    name=$1; fn=$2
    REASON="$T/$name.reason"; : > "$REASON"
    ( set +e; "$fn" ) > "$T/$name.log" 2>&1
    rc=$?
    if [ "$rc" -eq 0 ]; then
        echo "PASS $name"
    else
        why=$(head -n 1 "$REASON" | tr -d '\r')
        [ -n "$why" ] || why="exited $rc: $(tail -n 1 "$T/$name.log" | tr -d '\r')"
        echo "FAIL $name: $why"
    fi
}

run_case meta-split-tests-pass            case_meta_split_tests_pass
run_case meta-split-tests-cover-inkling   case_meta_split_tests_cover_inkling

rm -rf "$T"
exit 0
