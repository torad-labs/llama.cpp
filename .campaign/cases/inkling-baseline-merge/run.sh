#!/bin/sh
# Cases for inkling-baseline-merge: upstream d3c3e242c is merged, the fork's deleted
# workflows stay deleted, conflicts are resolved, and the merged tree configures and
# builds the server and its test programs with CUDA enabled.
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$ROOT" || { echo "run.sh: cannot cd to $ROOT" >&2; exit 2; }

U=d3c3e242c16b8cbf51400fad410cd69b22d64f35
B=build-campaign
T=$(mktemp -d /tmp/inkling-baseline-merge.XXXXXX)

fail() { printf '%s\n' "$*" > "$REASON"; exit 1; }

require_merged() {
    git cat-file -e "$U^{commit}" 2>/dev/null || fail "upstream commit $U is not in this repository"
    git merge-base --is-ancestor "$U" HEAD || fail "upstream $U is not merged into HEAD"
}

configure() {
    cmake -S . -B "$B" -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120a \
        -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_EXAMPLES=OFF \
        > "$T/configure.log" 2>&1 || fail "cmake configure with GGML_CUDA=ON failed: $(tail -n 1 "$T/configure.log")"
}

build_target() {
    require_merged
    configure
    cmake --build "$B" -j "$(nproc)" --target "$1" > "$T/build-$1.log" 2>&1 \
        || fail "building $1 failed: $(grep -m1 -E 'error' "$T/build-$1.log" || tail -n 1 "$T/build-$1.log")"
    [ -x "$B/bin/$1" ] || fail "$B/bin/$1 was not produced by the build"
}

case_upstream_commit_is_merged() {
    require_merged
}

case_upstream_workflows_stay_deleted() {
    require_merged
    git ls-tree -r --name-only "$U" .github/workflows > "$T/wf" || fail "cannot list upstream workflows"
    [ -s "$T/wf" ] || fail "upstream $U lists no workflows to compare"
    back=""
    while IFS= read -r p; do
        if git cat-file -e "HEAD:$p" 2>/dev/null; then back="$back $p"; fi
    done < "$T/wf"
    [ -z "$back" ] || fail "upstream workflows are back in HEAD:$back"
}

case_no_conflict_markers_in_tree() {
    require_merged
    [ -z "$(git ls-files -u)" ] || fail "the index still has unmerged paths"
    hits=$(git grep -l -E '^(<<<<<<<|>>>>>>>)( |$)' HEAD -- 2>/dev/null | head -n 5 | tr '\n' ' ')
    [ -z "$hits" ] || fail "conflict markers left in: $hits"
}

case_cuda_configure_succeeds() {
    require_merged
    configure
    grep -q '^GGML_CUDA:BOOL=ON$' "$B/CMakeCache.txt" || fail "GGML_CUDA is not ON in $B/CMakeCache.txt"
}

case_llama_server_builds()            { build_target llama-server; }
case_test_llama_archs_builds()        { build_target test-llama-archs; }
case_test_chat_builds()               { build_target test-chat; }
case_test_mtmd_impl_builds()          { build_target test-mtmd-impl; }
case_test_backend_meta_split_builds() { build_target test-backend-meta-split; }
case_test_server_response_builds()    { build_target test-server-response; }
case_test_backend_ops_builds()        { build_target test-backend-ops; }

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

run_case upstream-commit-is-merged         case_upstream_commit_is_merged
run_case upstream-workflows-stay-deleted   case_upstream_workflows_stay_deleted
run_case no-conflict-markers-in-tree       case_no_conflict_markers_in_tree
run_case cuda-configure-succeeds           case_cuda_configure_succeeds
run_case llama-server-builds               case_llama_server_builds
run_case test-llama-archs-builds           case_test_llama_archs_builds
run_case test-chat-builds                  case_test_chat_builds
run_case test-mtmd-impl-builds             case_test_mtmd_impl_builds
run_case test-backend-meta-split-builds    case_test_backend_meta_split_builds
run_case test-server-response-builds       case_test_server_response_builds
run_case test-backend-ops-builds           case_test_backend_ops_builds

rm -rf "$T"
exit 0
