#!/bin/sh
# Cases for inkling-server-parity-state: saving an Inkling session mid-conversation and restoring it,
# into the same slot or another one, gives the same next logits as the uninterrupted run, and prefix
# reuse refuses or snapshots correctly where the convolution state is involved. The comparisons run
# as ctest cases whose names carry inkling-state; they are registered, they run rather than skip, and
# they pass.
# The server-level cases put the synthetic inkling2 model that test-llama-archs -o writes behind this
# fork's llama-server (unified cache, two slots, a slot save path and a RAM prompt cache) and compare
# the next-token distribution a client gets after a save and restore, a trip through the prompt
# cache, or a reused prefix with the one a fresh slot gives for the same tokens. Whether the server
# reuses the state or refuses and reads the prompt again, the distribution must be the same.
# GPU runs use card 1 only, under /run/user/1000/gpu1.lock; card 0 serves another seat.
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$ROOT" || { echo "run.sh: cannot cd to $ROOT" >&2; exit 2; }

U=d3c3e242c16b8cbf51400fad410cd69b22d64f35
B=build-campaign
LOCK=/run/user/1000/gpu1.lock
T=$(mktemp -d /tmp/inkling-server-parity-state.XXXXXX)
SERVER_PID=

fail() { printf '%s\n' "$*" > "$REASON"; exit 1; }

require_merged() {
    git cat-file -e "$U^{commit}" 2>/dev/null || fail "upstream commit $U is not in this repository"
    git merge-base --is-ancestor "$U" HEAD || fail "upstream $U is not merged into HEAD"
}

configure() {
    require_merged
    if [ ! -f "$T/configured" ]; then
        cmake -S . -B "$B" -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120a \
            -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_EXAMPLES=OFF \
            -DGGML_CCACHE=OFF > "$T/configure.log" 2>&1 \
            || fail "cmake configure with GGML_CUDA=ON failed: $(tail -n 1 "$T/configure.log")"
        : > "$T/configured"
    fi
}

# The whole tree: the ctest cases may run any test program.
build_all() {
    configure
    if [ ! -f "$T/built" ]; then
        cmake --build "$B" -j "$(nproc)" > "$T/build.log" 2>&1 \
            || fail "building the tree failed: $(grep -m1 -E 'error' "$T/build.log" || tail -n 1 "$T/build.log")"
        : > "$T/built"
    fi
}

gpu1() { flock "$LOCK" env CUDA_VISIBLE_DEVICES=1 "$@"; }

# ctest -R inkling-state, run once on card 1; the cases below read this run.
ctest_run() {
    build_all
    if [ ! -f "$T/ctest.rc" ]; then
        gpu1 ctest --test-dir "$B" -R inkling-state --no-tests=error --output-on-failure > "$T/ctest.out" 2>&1
        echo $? > "$T/ctest.rc"
    fi
    CTEST_RC=$(cat "$T/ctest.rc")
}

case_ctest_registered() {
    configure
    ctest --test-dir "$B" -N -R inkling-state > "$T/list.out" 2>&1 || fail "ctest -N -R inkling-state failed: $(tail -n 1 "$T/list.out")"
    n=$(sed -n -E 's/^Total Tests: ([0-9]+).*/\1/p' "$T/list.out" | tail -n 1)
    [ -n "$n" ] || fail "ctest -N -R inkling-state printed no test count: $(tail -n 1 "$T/list.out")"
    [ "$n" -gt 0 ] || fail "no ctest case has inkling-state in its name"
}

case_ctest_pass() {
    ctest_run
    [ "$CTEST_RC" -eq 0 ] || fail "ctest -R inkling-state exited $CTEST_RC: $(grep -m1 -E 'tests passed|No tests were found' "$T/ctest.out" || tail -n 1 "$T/ctest.out")$(grep -E '^[[:space:]]*[0-9]+ - .*\((Failed|Timeout|SEGFAULT|Subprocess aborted|Exception|Child aborted)\)' "$T/ctest.out" | head -n 3 | tr -s ' ' | tr '\n' ';' | sed 's/^/; /')"
}

case_ctest_all_run() {
    ctest_run
    grep -q -E 'tests passed' "$T/ctest.out" || fail "ctest -R inkling-state reported no results (exit $CTEST_RC): $(tail -n 1 "$T/ctest.out")"
    skipped=$(grep -E '\*\*\*Skipped|Not Run|\*\*\*Disabled' "$T/ctest.out" | head -n 3 | tr -s ' ' | tr '\n' ';')
    [ -z "$skipped" ] || fail "inkling-state ctest cases did not run with card 1 visible: $skipped"
}

cat > "$T/state.py" <<'PY'
import json, sys, urllib.request, urllib.error

BASE = sys.argv[2]
VOCAB = int(sys.argv[3])

def die(msg):
    print(msg)
    sys.exit(1)

def post(path, body, timeout=300):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except Exception:
            return e.code, raw.decode(errors="replace")

def short(x):
    return (json.dumps(x) if not isinstance(x, str) else x)[:300]

def tokens(seed, n):
    # ids 1..VOCAB-1 from a fixed generator, the same on every run
    out, x = [], seed
    for _ in range(n):
        x = (x * 1103515245 + 12345) % (1 << 31)
        out.append(1 + x % (VOCAB - 1))
    return out

def slot(i, action, filename=None):
    body = {"filename": filename} if filename else {}
    status, out = post(f"/slots/{i}?action={action}", body)
    if status != 200:
        die(f"/slots/{i}?action={action} answered {status}: {short(out)}")

def erase_all():
    slot(0, "erase")
    slot(1, "erase")

def complete(prompt, id_slot, cache, probs=False):
    body = {"prompt": prompt, "n_predict": 1, "id_slot": id_slot, "cache_prompt": cache, "temperature": 0.0,
            "return_tokens": True, "seed": 1}
    if probs:
        body["n_probs"] = VOCAB
    status, out = post("/completion", body)
    if status != 200 or not isinstance(out, dict):
        die(f"/completion on slot {id_slot} answered {status}: {short(out)}")
    toks = out.get("tokens") or []
    if not toks:
        die(f"/completion on slot {id_slot} returned no token")
    if not probs:
        return toks[0]
    try:
        alts = out["completion_probabilities"][0]["top_logprobs"]
        dist = {int(a["id"]): float(a["logprob"]) for a in alts}
    except Exception:
        die(f"/completion gave no next-token distribution: {short(out)}")
    if len(dist) < VOCAB // 2:
        die(f"/completion gave {len(dist)} of the {VOCAB} next-token probabilities asked for")
    return dist

def fresh(prompt):
    erase_all()
    return complete(prompt, 0, False, probs=True)

def same(got, want, what):
    # next-token log-probabilities, each less its mean over the ids both give: the logits up to a shift.
    # On the -d 0.1 fixture, changing one earlier token of A moves them by 0.012 or more; a fresh slot and
    # a restored one were read 0 apart (Nov 17, 2026, one RTX PRO 6000)
    ids = set(got) & set(want)
    if len(ids) < VOCAB // 2:
        die(f"{what}: the two runs share {len(ids)} of {VOCAB} next-token ids")
    mg = sum(got[i] for i in ids) / len(ids); mw = sum(want[i] for i in ids) / len(ids)
    diff = max(abs((got[i] - mg) - (want[i] - mw)) for i in ids)
    if diff > 2e-3:
        top_g = max(got, key=got.get); top_w = max(want, key=want.get)
        die(f"{what}: the next-token logits differ from the uninterrupted run's by up to {diff:.4g} "
            f"(top token {top_g} against {top_w})")

A = tokens(11, 120)     # the conversation so far (the synthetic model trains at 256 tokens)
Bt = tokens(23, 30)     # its next turn
C = tokens(37, 30)      # a different next turn
D = tokens(53, 50)      # another session's conversation

def main():
    case = sys.argv[1]
    if case in ("restore_same", "restore_other"):
        target = 0 if case == "restore_same" else 1
        erase_all()
        g = complete(A, 0, True)
        slot(0, "save", "inkling-mid.bin")
        erase_all()
        slot(target, "restore", "inkling-mid.bin")
        got = complete(A + [g] + Bt, target, True, probs=True)
        want = fresh(A + [g] + Bt)
        same(got, want, f"restored into slot {target}")
    elif case == "prompt_cache":
        erase_all()
        g = complete(A, 0, True)
        complete(D, 0, True)             # another session takes the slot; the first waits in the prompt cache
        got = complete(A + [g] + Bt, 1, True, probs=True)
        want = fresh(A + [g] + Bt)
        same(got, want, "back from the prompt cache")
    elif case == "prefix_diverge":
        erase_all()
        complete(A + Bt, 0, True)
        got = complete(A + C, 0, True, probs=True)
        want = fresh(A + C)
        same(got, want, "a turn that shares the prefix and then diverges")
    elif case == "prefix_shorter":
        erase_all()
        complete(A + Bt, 0, True)
        got = complete(A[:90], 0, True, probs=True)
        want = fresh(A[:90])
        same(got, want, "a turn that is a shorter prefix of the cached one")
    else:
        die("unknown case " + case)

main()
PY

setup_server() {
    build_all
    mkdir -p "$T/models" "$T/slots"
    # weights at a spread of 0.1, where the next logits follow every earlier token (at the default 0.01 they barely move)
    env CUDA_VISIBLE_DEVICES=1 "./$B/bin/test-llama-archs" -a inkling2 -d 0.1 -o "$T/models" > "$T/save.out" 2>&1 \
        || fail "test-llama-archs -a inkling2 -o did not write the synthetic model: $(tail -n 1 "$T/save.out")"
    m=$(ls "$T"/models/inkling2*.gguf 2>/dev/null | head -n 1)
    [ -n "$m" ] || fail "test-llama-archs -a inkling2 -o wrote no inkling2 model"
    echo "$m" > "$T/model.path"
    python3 -I -c '
import struct, sys
# the GGUF header read by hand: the key-value section, up to the key asked for
f = open(sys.argv[1], "rb")
def rd(fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]
def rstr():
    return f.read(rd("Q")).decode()
SIZES = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
def value(t):
    if t == 8:
        return rstr()
    if t == 9:
        et, n = rd("I"), rd("Q")
        return [value(et) for _ in range(n)]
    return rd(SIZES[t])
assert f.read(4) == b"GGUF", "not a GGUF file"
rd("I"); rd("Q"); n_kv = rd("Q")
for _ in range(n_kv):
    k = rstr(); v = value(rd("I"))
    if k == sys.argv[2]:
        print(int(v)); sys.exit(0)
sys.exit("no key " + sys.argv[2])' "$m" inkling2.vocab_size_unpadded > "$T/vocab" 2> "$T/vocab.err" \
        || fail "could not read inkling2.vocab_size_unpadded from the synthetic model: $(tail -n 1 "$T/vocab.err")"
    python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])' > "$T/port"
}

start_server() {
    REASON="$T/server.reason"; : > "$REASON"
    ( setup_server ) > "$T/setup.log" 2>&1 || return 0
    port=$(cat "$T/port")
    # fd 9 holds card 1's lock for as long as this script or the server keeps it open
    exec 9> "$LOCK"
    flock 9
    CUDA_VISIBLE_DEVICES=1 "./$B/bin/llama-server" -m "$(cat "$T/model.path")" -ngl 99 -c 4096 -np 2 --kv-unified \
        --slot-save-path "$T/slots" --cache-ram 1024 --host 127.0.0.1 --port "$port" > "$T/server.log" 2>&1 &
    SERVER_PID=$!
    i=0
    while [ $i -lt 240 ]; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            printf '%s\n' "llama-server with the synthetic inkling2 model exited before it was ready: $(grep -m1 -i -E 'error|exception|failed' "$T/server.log" || tail -n 1 "$T/server.log")" > "$REASON"
            SERVER_PID=
            return 0
        fi
        if python3 -c 'import sys,urllib.request; sys.exit(0 if urllib.request.urlopen(sys.argv[1], timeout=2).status == 200 else 1)' \
            "http://127.0.0.1:$port/health" 2>/dev/null; then
            echo "http://127.0.0.1:$port" > "$T/base"
            return 0
        fi
        i=$((i + 1))
        sleep 0.5
    done
    printf '%s\n' "llama-server with the synthetic inkling2 model was not ready after 120 s: $(tail -n 1 "$T/server.log")" > "$REASON"
}

stop_server() {
    if [ -n "$SERVER_PID" ]; then
        kill "$SERVER_PID" 2>/dev/null
        wait "$SERVER_PID" 2>/dev/null
        SERVER_PID=
    fi
    flock -u 9 2>/dev/null
    exec 9>&-
}
trap 'stop_server; rm -rf "$T"' EXIT
trap 'exit 1' INT TERM

served() {
    [ -f "$T/base" ] || fail "$(cat "$T/server.reason" 2>/dev/null | head -n 1)"
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        fail "llama-server is no longer running: $(grep -m1 -i -E 'GGML_ASSERT|error|exception|abort' "$T/server.log" || tail -n 1 "$T/server.log")"
    fi
    msg=$(python3 "$T/state.py" "$1" "$(cat "$T/base")" "$(cat "$T/vocab")" 2>&1) || fail "${msg:-the request crashed}"
}

run_case() {
    name=$1; fn=$2; shift 2
    REASON="$T/$name.reason"; : > "$REASON"
    ( set +e; "$fn" "$@" ) > "$T/$name.log" 2>&1
    rc=$?
    if [ "$rc" -eq 0 ]; then
        echo "PASS $name"
    else
        why=$(head -n 1 "$REASON" | tr -d '\r')
        [ -n "$why" ] || why="exited $rc: $(tail -n 1 "$T/$name.log" | tr -d '\r')"
        echo "FAIL $name: $why"
    fi
}

# the ctest cases take card 1's lock themselves, so they run before this script's server holds it
run_case inkling-state-ctest-cases-registered                      case_ctest_registered
run_case inkling-state-ctest-cases-pass-on-card-1                  case_ctest_pass
run_case inkling-state-ctest-cases-all-run                         case_ctest_all_run

start_server
run_case inkling-state-server-restore-same-slot-same-next-logits   served restore_same
run_case inkling-state-server-restore-other-slot-same-next-logits  served restore_other
run_case inkling-state-server-prompt-cache-return-same-next-logits served prompt_cache
run_case inkling-state-server-prefix-diverging-turn-same-next-logits served prefix_diverge
run_case inkling-state-server-shorter-prefix-turn-same-next-logits served prefix_shorter
stop_server

exit 0
