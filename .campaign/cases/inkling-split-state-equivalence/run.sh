#!/bin/sh
# Cases for inkling-split-state-equivalence: the synthetic Inkling fixture under a two-rank
# tensor split matches one rank, sequence state included. The comparisons run as ctest cases
# whose names carry inkling; they are registered, they run here rather than skip, and they pass.
# The fixture written out by test-llama-archs also runs prefill across chunk boundaries and
# decode under a two-rank tensor split.
# GPU runs use card 1 only, under /run/user/1000/gpu1.lock; card 0 serves another seat. Two
# ranks share that one card (CUDA0 twice under CUDA_VISIBLE_DEVICES=1).
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$ROOT" || { echo "run.sh: cannot cd to $ROOT" >&2; exit 2; }

U=d3c3e242c16b8cbf51400fad410cd69b22d64f35
B=build-campaign
T=$(mktemp -d /tmp/inkling-split-state-equivalence.XXXXXX)

fail() { printf '%s\n' "$*" > "$REASON"; exit 1; }

require_merged() {
    git cat-file -e "$U^{commit}" 2>/dev/null || fail "upstream commit $U is not in this repository"
    git merge-base --is-ancestor "$U" HEAD || fail "upstream $U is not merged into HEAD"
}

# The whole tree: the ctest cases may run any test program.
build_all() {
    require_merged
    if [ ! -f "$T/built" ]; then
        cmake -S . -B "$B" -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120a \
            -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_EXAMPLES=OFF \
            -DGGML_CCACHE=OFF > "$T/configure.log" 2>&1 \
            || fail "cmake configure with GGML_CUDA=ON failed: $(tail -n 1 "$T/configure.log")"
        cmake --build "$B" -j "$(nproc)" > "$T/build.log" 2>&1 \
            || fail "building the tree failed: $(grep -m1 -E 'error' "$T/build.log" || tail -n 1 "$T/build.log")"
        : > "$T/built"
    fi
}

gpu1() { flock /run/user/1000/gpu1.lock env CUDA_VISIBLE_DEVICES=1 "$@"; }

# ctest -R inkling, run once on card 1; the cases below read this run.
ctest_run() {
    build_all
    if [ ! -f "$T/ctest.rc" ]; then
        gpu1 ctest --test-dir "$B" -R inkling --no-tests=error --output-on-failure > "$T/ctest.out" 2>&1
        echo $? > "$T/ctest.rc"
    fi
    CTEST_RC=$(cat "$T/ctest.rc")
}

case_inkling_ctest_cases_registered() {
    build_all
    ctest --test-dir "$B" -N -R inkling > "$T/list.out" 2>&1 || fail "ctest -N -R inkling failed: $(tail -n 1 "$T/list.out")"
    n=$(sed -n -E 's/^Total Tests: ([0-9]+).*/\1/p' "$T/list.out" | tail -n 1)
    [ -n "$n" ] || fail "ctest -N -R inkling printed no test count: $(tail -n 1 "$T/list.out")"
    [ "$n" -gt 0 ] || fail "no ctest case has inkling in its name"
}

case_inkling_ctest_cases_pass_on_card_1() {
    ctest_run
    [ "$CTEST_RC" -eq 0 ] || fail "ctest -R inkling exited $CTEST_RC: $(grep -m1 -E 'tests passed|No tests were found' "$T/ctest.out" || tail -n 1 "$T/ctest.out")$(grep -E '^[[:space:]]*[0-9]+ - .*\((Failed|Timeout|SEGFAULT|Subprocess aborted|Exception|Child aborted)\)' "$T/ctest.out" | head -n 3 | tr -s ' ' | tr '\n' ';' | sed 's/^/; /')"
}

case_inkling_ctest_cases_all_run() {
    ctest_run
    grep -q -E 'tests passed' "$T/ctest.out" || fail "ctest -R inkling reported no results (exit $CTEST_RC): $(tail -n 1 "$T/ctest.out")"
    skipped=$(grep -E '\*\*\*Skipped|\*\*\*Not Run|Not Run|\*\*\*Disabled' "$T/ctest.out" | head -n 3 | tr -s ' ' | tr '\n' ';')
    [ -z "$skipped" ] || fail "inkling ctest cases did not run with card 1 visible: $skipped"
}

case_inkling2_fixture_runs_under_two_rank_tensor_split() {
    build_all
    mkdir -p "$T/models"
    env CUDA_VISIBLE_DEVICES=1 "./$B/bin/test-llama-archs" -a inkling2 -o "$T/models" > "$T/save.out" 2>&1 \
        || fail "test-llama-archs -a inkling2 -o did not write the fixture: $(tail -n 1 "$T/save.out")"
    set -- "$T"/models/inkling2*.gguf
    [ -f "$1" ] || fail "test-llama-archs -a inkling2 -o wrote no inkling2 model"
    for m in "$@"; do
        base=$(basename "$m")
        # 64 prompt tokens in ubatches of 16 cross three chunk boundaries; then 8 decoded tokens
        gpu1 "./$B/bin/llama-bench" -m "$m" -sm tensor -dev CUDA0/CUDA0 -ngl 99 -p 64 -n 8 -b 16 -ub 16 -r 1 -o csv \
            > "$T/$base.csv" 2> "$T/$base.err"
        rc=$?
        [ "$rc" -eq 0 ] || fail "llama-bench on $base under a two-rank tensor split exited $rc: $(grep -m1 -E 'not implemented|GGML_ASSERT' "$T/$base.err" || grep -m1 -i 'error' "$T/$base.err" || tail -n 1 "$T/$base.err")"
        rows=$(grep -c '^"' "$T/$base.csv")
        [ "$rows" -eq 2 ] || fail "llama-bench on $base under a two-rank tensor split gave $rows of 2 results (prefill, decode)"
    done
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

run_case inkling-ctest-cases-registered                    case_inkling_ctest_cases_registered
run_case inkling-ctest-cases-pass-on-card-1                case_inkling_ctest_cases_pass_on_card_1
run_case inkling-ctest-cases-all-run                       case_inkling_ctest_cases_all_run
run_case inkling2-fixture-runs-under-two-rank-tensor-split case_inkling2_fixture_runs_under_two_rank_tensor_split

rm -rf "$T"
exit 0
