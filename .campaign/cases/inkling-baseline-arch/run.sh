#!/bin/sh
# Cases for inkling-baseline-arch: the Inkling 2 architecture is live. Its synthetic
# fixture agrees in the architecture tests, its per-head attention mask cases pass on
# CUDA against the CPU reference, its chat media handling passes, and the image
# preprocessor cases pass.
# GPU runs use card 1 only, under /run/user/1000/gpu1.lock; card 0 serves another seat.
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$ROOT" || { echo "run.sh: cannot cd to $ROOT" >&2; exit 2; }

U=d3c3e242c16b8cbf51400fad410cd69b22d64f35
B=build-campaign
T=$(mktemp -d /tmp/inkling-baseline-arch.XXXXXX)

fail() { printf '%s\n' "$*" > "$REASON"; exit 1; }

require_merged() {
    git cat-file -e "$U^{commit}" 2>/dev/null || fail "upstream commit $U is not in this repository"
    git merge-base --is-ancestor "$U" HEAD || fail "upstream $U is not merged into HEAD"
}

build_target() {
    require_merged
    cmake -S . -B "$B" -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120a \
        -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_EXAMPLES=OFF \
        > "$T/configure.log" 2>&1 || fail "cmake configure with GGML_CUDA=ON failed: $(tail -n 1 "$T/configure.log")"
    cmake --build "$B" -j "$(nproc)" --target "$1" > "$T/build-$1.log" 2>&1 \
        || fail "building $1 failed: $(grep -m1 -E 'error' "$T/build-$1.log" || tail -n 1 "$T/build-$1.log")"
    [ -x "$B/bin/$1" ] || fail "$B/bin/$1 was not produced by the build"
}

gpu1() { flock /run/user/1000/gpu1.lock env CUDA_VISIBLE_DEVICES=1 "$@"; }

case_inkling2_fixture_agrees() {
    build_target test-llama-archs
    gpu1 "./$B/bin/test-llama-archs" -a inkling2 > "$T/archs.out" 2>&1
    rc=$?
    summary=$(grep 'Summary:' "$T/archs.out" | tail -n 1 | sed 's/^.*Summary:/Summary:/')
    [ "$rc" -eq 0 ] || fail "test-llama-archs -a inkling2 exited $rc; ${summary:-no summary}"
    echo "$summary" | grep -q -E '^Summary: all [1-9][0-9]* test\(s\) passed' \
        || fail "test-llama-archs -a inkling2 did not pass any test: ${summary:-no summary}"
}

case_per_head_mask_cases_pass_on_cuda() {
    build_target test-backend-ops
    gpu1 "./$B/bin/test-backend-ops" test -b CUDA0 -o FLASH_ATTN_EXT -p 'mask_heads=([2-9]|[1-9][0-9]+)' \
        > "$T/fa.out" 2>&1
    rc=$?
    grep 'mask_heads=' "$T/fa.out" > "$T/fa.cases"
    n_ok=$(grep -c 'OK' "$T/fa.cases")
    n_fail=$(grep -c 'FAIL' "$T/fa.cases")
    [ "$rc" -eq 0 ] || fail "test-backend-ops FLASH_ATTN_EXT per-head mask cases exited $rc ($n_fail failing, $n_ok passing)"
    [ "$n_fail" -eq 0 ] || fail "$n_fail per-head mask cases failed on CUDA0"
    [ "$n_ok" -gt 0 ] || fail "no per-head mask (mask_heads > 1) FLASH_ATTN_EXT case passed on CUDA0"
}

case_chat_media_handling_passes() {
    build_target test-chat
    env CUDA_VISIBLE_DEVICES=1 "./$B/bin/test-chat" > "$T/chat.out" 2>&1
    rc=$?
    [ "$rc" -eq 0 ] || fail "test-chat exited $rc: $(tail -n 1 "$T/chat.out")"
    grep -q 'All tests passed' "$T/chat.out" || fail "test-chat did not report all tests passed"
}

mtmd_image_run() {
    build_target test-mtmd-impl
    env CUDA_VISIBLE_DEVICES=1 "./$B/bin/test-mtmd-impl" 'test_image_preprocessor.*' > "$T/mtmd.out" 2>&1
    MTMD_RC=$?
}

case_image_preprocessor_cases_pass() {
    mtmd_image_run
    [ "$MTMD_RC" -eq 0 ] || fail "test-mtmd-impl image preprocessor cases exited $MTMD_RC: $(grep -m1 '\[FAIL\]' "$T/mtmd.out" || tail -n 1 "$T/mtmd.out")"
    ! grep -q '\[FAIL\]' "$T/mtmd.out" || fail "image preprocessor case failed: $(grep -m1 '\[FAIL\]' "$T/mtmd.out")"
    grep -q '\[PASS\]' "$T/mtmd.out" || fail "no image preprocessor case ran"
}

case_inkling2v_image_preprocessor_passes() {
    mtmd_image_run
    grep -q -E '^ *test_image_preprocessor_inkling2v .*\[PASS\]' "$T/mtmd.out" \
        || fail "the inkling2v image preprocessor case did not pass: $(grep -m1 'inkling2v' "$T/mtmd.out" || echo 'not run')"
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

run_case inkling2-fixture-agrees                 case_inkling2_fixture_agrees
run_case inkling2-per-head-mask-cases-pass-on-cuda case_per_head_mask_cases_pass_on_cuda
run_case chat-media-handling-passes              case_chat_media_handling_passes
run_case image-preprocessor-cases-pass           case_image_preprocessor_cases_pass
run_case inkling2v-image-preprocessor-passes     case_inkling2v_image_preprocessor_passes

rm -rf "$T"
exit 0
