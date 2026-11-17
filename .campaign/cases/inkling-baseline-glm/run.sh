#!/bin/sh
# Cases for inkling-baseline-glm: on the merged tree, GLM 5.3 Flash serving behaves as
# before: the architecture fixtures, the tensor split decisions, the prompt step budget
# logic and the pinned launcher qualification dry runs pass.
# GPU runs use card 1 only, under /run/user/1000/gpu1.lock; card 0 serves another seat.
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$ROOT" || { echo "run.sh: cannot cd to $ROOT" >&2; exit 2; }

U=d3c3e242c16b8cbf51400fad410cd69b22d64f35
B=build-campaign
T=$(mktemp -d /tmp/inkling-baseline-glm.XXXXXX)

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

archs_pass() {
    # $1: log name; rest: test-llama-archs arguments
    log="$T/$1.out"; shift
    build_target test-llama-archs
    gpu1 "./$B/bin/test-llama-archs" "$@" > "$log" 2>&1
    rc=$?
    summary=$(grep 'Summary:' "$log" | tail -n 1 | sed 's/^.*Summary:/Summary:/')
    [ "$rc" -eq 0 ] || fail "test-llama-archs $* exited $rc; ${summary:-no summary}"
    echo "$summary" | grep -q -E '^Summary: all [1-9][0-9]* test\(s\) passed' \
        || fail "test-llama-archs $* did not pass any test: ${summary:-no summary}"
}

case_architecture_fixtures_pass() { archs_pass all; }
case_glm5next_fixture_passes()    { archs_pass glm5next -a glm5next; }

case_tensor_split_decisions_pass() {
    build_target test-backend-meta-split
    gpu1 "./$B/bin/test-backend-meta-split" > "$T/split.out" 2>&1
    rc=$?
    [ "$rc" -eq 0 ] || fail "test-backend-meta-split exited $rc: $(grep -m1 -i 'fail' "$T/split.out" || tail -n 1 "$T/split.out")"
}

case_prompt_step_budget_passes() {
    build_target test-server-response
    env CUDA_VISIBLE_DEVICES=1 "./$B/bin/test-server-response" > "$T/resp.out" 2>&1
    rc=$?
    [ "$rc" -eq 0 ] || fail "test-server-response exited $rc: $(grep -m1 -i 'budget' "$T/resp.out" || tail -n 1 "$T/resp.out")"
    grep -q -x 'OK' "$T/resp.out" || fail "test-server-response did not print OK"
}

case_launcher_qualification_dry_runs_pass() {
    require_merged
    ( cd scripts/glm53 && python3 -m unittest test_qualification ) > "$T/qual.out" 2>&1
    rc=$?
    [ "$rc" -eq 0 ] || fail "test_qualification exited $rc: $(tail -n 1 "$T/qual.out")"
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

run_case glm-architecture-fixtures-pass          case_architecture_fixtures_pass
run_case glm5next-fixture-passes                 case_glm5next_fixture_passes
run_case tensor-split-decisions-pass             case_tensor_split_decisions_pass
run_case prompt-step-budget-passes               case_prompt_step_budget_passes
run_case launcher-qualification-dry-runs-pass    case_launcher_qualification_dry_runs_pass

rm -rf "$T"
exit 0
