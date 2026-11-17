#!/bin/sh
# Cases for inkling-optimize-relative-attention: an Inkling layer's relative-position bias is added inside the
# attention kernel from the relative features and the distance table, distances derived on the device, so no bias
# tensor grows with the unified cache pool and graph reuse holds while sessions come and go; logits agree with the
# materialized path, distances beyond 1024 included. The comparisons the item names run as ctest cases whose names
# carry inkling-rel; they are registered, run rather than skip, and pass. The fixture still agrees.
# The served cases put the synthetic inkling2 model test-llama-archs -o writes behind this fork's llama-server: its
# next-token logits on card 1 past distance 1024, in a prompt and in decoding, against the same served on the CPU
# alone; its compute buffer at two pool sizes; and its graph reuse while sessions come and go.
# GPU runs use card 1 only, under /run/user/1000/gpu1.lock; card 0 serves another seat.
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$ROOT" || { echo "run.sh: cannot cd to $ROOT" >&2; exit 2; }

U=d3c3e242c16b8cbf51400fad410cd69b22d64f35
B=build-campaign
LOCK=/run/user/1000/gpu1.lock
T=$(mktemp -d /tmp/inkling-optimize-relative-attention.XXXXXX)
trap 'rm -rf "$T"' EXIT
trap 'exit 1' INT TERM

fail() { printf '%s\n' "$*" > "$REASON"; exit 1; }

require_merged() {
    git cat-file -e "$U^{commit}" 2>/dev/null || fail "upstream commit $U is not in this repository"
    git merge-base --is-ancestor "$U" HEAD || fail "upstream $U is not merged into HEAD"
}

# gated <log> <command...>: runs a configure or build, and while hostshield's build gate turns it away for a busy
# machine ("the batch service refused") waits a minute and asks again, for up to an hour: a busy gate is not
# a result
gated() {
    glog=$1; shift
    tries=0
    while :; do
        "$@" > "$glog" 2>&1 && return 0
        grep -q -E 'not admitted by the shared queue|could not obtain shared batch admission|batch service refused' "$glog" || return 1
        tries=$((tries + 1))
        [ "$tries" -lt 60 ] || return 1
        sleep 60
    done
}

# The configure and the build are tried once per run: a case after one that failed fails with the same reason
# rather than waiting for the build again
configure() {
    require_merged
    [ ! -f "$T/configure.failed" ] || fail "$(cat "$T/configure.failed")"
    if [ ! -f "$T/configured" ]; then
        if ! gated "$T/configure.log" cmake -S . -B "$B" -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120a \
            -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_EXAMPLES=OFF \
            -DGGML_CCACHE=OFF; then
            echo "cmake configure with GGML_CUDA=ON failed: $(grep -m1 -E 'CMake Error|refused|could not' "$T/configure.log" || tail -n 1 "$T/configure.log")" > "$T/configure.failed"
            fail "$(cat "$T/configure.failed")"
        fi
        : > "$T/configured"
    fi
}

# The whole tree: the ctest cases may run any test program.
build_all() {
    configure
    [ ! -f "$T/build.failed" ] || fail "$(cat "$T/build.failed")"
    if [ ! -f "$T/built" ]; then
        if ! gated "$T/build.log" cmake --build "$B" -j "$(nproc)"; then
            echo "building the tree failed: $(grep -m1 -E ' error|Error [0-9]|refused|could not|missing required' "$T/build.log" || tail -n 1 "$T/build.log")" > "$T/build.failed"
            fail "$(cat "$T/build.failed")"
        fi
        : > "$T/built"
    fi
}

gpu1() { flock "$LOCK" env CUDA_VISIBLE_DEVICES=1 "$@"; }

# ctest -R <prefix>, run once on card 1; the three ctest cases read this run
ctest_run() {
    build_all
    if [ ! -f "$T/ctest.rc" ]; then
        gpu1 ctest --test-dir "$B" -R "$PREFIX" --no-tests=error --output-on-failure > "$T/ctest.out" 2>&1
        echo $? > "$T/ctest.rc"
    fi
    CTEST_RC=$(cat "$T/ctest.rc")
}

case_ctest_registered() {
    configure
    ctest --test-dir "$B" -N -R "$PREFIX" > "$T/list.out" 2>&1 || fail "ctest -N -R $PREFIX failed: $(tail -n 1 "$T/list.out")"
    n=$(sed -n -E 's/^Total Tests: ([0-9]+).*/\1/p' "$T/list.out" | tail -n 1)
    [ -n "$n" ] || fail "ctest -N -R $PREFIX printed no test count: $(tail -n 1 "$T/list.out")"
    [ "$n" -gt 0 ] || fail "no ctest case has $PREFIX in its name"
}

case_ctest_pass() {
    ctest_run
    [ "$CTEST_RC" -eq 0 ] || fail "ctest -R $PREFIX exited $CTEST_RC: $(grep -m1 -E 'tests passed|No tests were found' "$T/ctest.out" || tail -n 1 "$T/ctest.out")$(grep -E '^[[:space:]]*[0-9]+ - .*\((Failed|Timeout|SEGFAULT|Subprocess aborted|Exception|Child aborted)\)' "$T/ctest.out" | head -n 3 | tr -s ' ' | tr '\n' ';' | sed 's/^/; /')"
}

case_ctest_all_run() {
    ctest_run
    grep -q -E 'tests passed' "$T/ctest.out" || fail "ctest -R $PREFIX reported no results (exit $CTEST_RC): $(tail -n 1 "$T/ctest.out")"
    skipped=$(grep -E '\*\*\*Skipped|Not Run|\*\*\*Disabled' "$T/ctest.out" | head -n 3 | tr -s ' ' | tr '\n' ';')
    [ -z "$skipped" ] || fail "$PREFIX ctest cases did not run with card 1 visible: $skipped"
}

# test-llama-archs -a inkling2: the synthetic fixture agrees across its backends
case_fixture_agrees() {
    build_all
    gpu1 "./$B/bin/test-llama-archs" -a inkling2 > "$T/archs.out" 2>&1
    rc=$?
    summary=$(grep 'Summary:' "$T/archs.out" | tail -n 1 | sed 's/^.*Summary:/Summary:/')
    [ "$rc" -eq 0 ] || fail "test-llama-archs -a inkling2 exited $rc; ${summary:-no summary}"
    echo "$summary" | grep -q -E '^Summary: all [1-9][0-9]* test\(s\) passed' \
        || fail "test-llama-archs -a inkling2 did not pass any test: ${summary:-no summary}"
}

cat > "$T/gguf.py" <<'PY'
import struct, sys
# reads the GGUF key-value section by hand and prints the integer value of each key asked for, one on a line
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
kv = {}
for _ in range(n_kv):
    k = rstr(); kv[k] = value(rd("I"))
for k in sys.argv[2:]:
    if k not in kv:
        sys.exit("no key " + k)
    print(int(kv[k]))
PY

# The synthetic inkling2 model test-llama-archs -o writes, once per run, at a weight spread of 0.1, where the next
# logits follow every earlier token (at the default 0.01 they barely move). Sets MODEL, VOCAB and NHEAD.
fixture() {
    build_all
    if [ ! -f "$T/model.path" ]; then
        mkdir -p "$T/models"
        gpu1 "./$B/bin/test-llama-archs" -a inkling2 -d 0.1 -o "$T/models" > "$T/save.out" 2>&1 \
            || fail "test-llama-archs -a inkling2 -o did not write the synthetic model: $(tail -n 1 "$T/save.out")"
        m=$(ls "$T"/models/inkling2*.gguf 2>/dev/null | head -n 1)
        [ -n "$m" ] || fail "test-llama-archs -a inkling2 -o wrote no inkling2 model"
        python3 -I "$T/gguf.py" "$m" inkling2.vocab_size_unpadded inkling2.attention.head_count > "$T/model.kv" 2> "$T/model.err" \
            || fail "could not read the synthetic model's vocabulary and head count: $(tail -n 1 "$T/model.err")"
        echo "$m" > "$T/model.path"
    fi
    MODEL=$(cat "$T/model.path")
    VOCAB=$(sed -n 1p "$T/model.kv")
    NHEAD=$(sed -n 2p "$T/model.kv")
}

cat > "$T/client.py" <<'PY'
import json, sys, threading, time, urllib.request, urllib.error

def die(msg):
    print(msg)
    sys.exit(1)

def tokens(vocab, seed, n):
    # ids 1..vocab-1 from a fixed generator, the same on every run
    out, x = [], seed
    for _ in range(n):
        x = (x * 1103515245 + 12345) % (1 << 31)
        out.append(1 + x % (vocab - 1))
    return out

def post(base, body):
    req = urllib.request.Request(base + "/completion", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")[:300]
    except Exception as e:
        return 0, str(e)[:300]

def complete(base, vocab, seed, n, steps):
    # a fresh prompt of n tokens; the next-token distribution of each of `steps` greedy steps, with its token
    body = {"prompt": tokens(vocab, seed, n), "n_predict": steps, "ignore_eos": True, "cache_prompt": False,
            "temperature": 0.0, "return_tokens": True, "seed": 1, "n_probs": vocab}
    status, out = post(base, body)
    if status != 200 or not isinstance(out, dict):
        die(f"/completion of a {n}-token prompt answered {status}: {out if isinstance(out, str) else json.dumps(out)[:300]}")
    try:
        res = []
        for step in out["completion_probabilities"]:
            dist = {int(a["id"]): float(a["logprob"]) for a in step["top_logprobs"]}
            res.append([int(step["id"]), dist])
    except Exception:
        die(f"/completion of a {n}-token prompt gave no next-token distributions: {json.dumps(out)[:300]}")
    if len(res) != steps:
        die(f"/completion of a {n}-token prompt gave {len(res)} of {steps} steps")
    for tok, dist in res:
        if len(dist) < vocab // 2:
            die(f"/completion gave {len(dist)} of the {vocab} next-token probabilities asked for")
        if any(v != v or v in (float("inf"),) for v in dist.values()):
            die(f"/completion of a {n}-token prompt gave a non-finite log-probability")
    return res

def run(base, vocab, out, steps, prompts, together):
    # prompts are (seed, n) pairs; together sends them at once, so the server batches them with each other
    results = [None] * len(prompts)
    errors = []
    def one(i):
        try:
            results[i] = complete(base, vocab, prompts[i][0], prompts[i][1], steps)
        except SystemExit as e:
            errors.append(str(e.code) if e.code not in (None, 1) else "a request failed")
    if together:
        ts = [threading.Thread(target=one, args=(i,)) for i in range(len(prompts))]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
    else:
        for i in range(len(prompts)):
            one(i)
    if errors or any(r is None for r in results):
        die(errors[0] if errors else "a request gave no result")
    json.dump(results, open(out, "w"))

def gap(a, b):
    # next-token log-probabilities, each less its mean over the ids both give: the logits up to a shift
    ids = set(a) & set(b)
    ma = sum(a[i] for i in ids) / len(ids); mb = sum(b[i] for i in ids) / len(ids)
    return len(ids), max(abs((a[i] - ma) - (b[i] - mb)) for i in ids)

def compare(what, fa, fb, tol, vocab):
    A = json.load(open(fa)); B = json.load(open(fb))
    if len(A) != len(B):
        die(f"{what}: {len(A)} results against {len(B)}")
    for p, (ra, rb) in enumerate(zip(A, B)):
        for s, ((ta, da), (tb, db)) in enumerate(zip(ra, rb)):
            da = {int(k): v for k, v in da.items()}; db = {int(k): v for k, v in db.items()}
            n, d = gap(da, db)
            if n < vocab // 2:
                die(f"{what}: prompt {p} step {s}: the two runs share {n} of {vocab} next-token ids")
            if d > tol:
                die(f"{what}: prompt {p} step {s}: the next-token logits differ by up to {d:.4g} (bar {tol:g}; "
                    f"top token {max(da, key=da.get)} against {max(db, key=db.get)})")
            if ta != tb:
                break  # a near tie went the other way; the steps after it continue different texts

def comego(base, vocab, out):
    # three sessions start together and leave one by one (60, 100 and 140 generated tokens), then a fourth comes
    lengths = [60, 100, 140]
    errors = []
    def one(seed, n, steps):
        body = {"prompt": tokens(vocab, seed, n), "n_predict": steps, "ignore_eos": True, "cache_prompt": False, "temperature": 0.0}
        status, o = post(base, body)
        if status != 200:
            errors.append(f"/completion answered {status}: {o if isinstance(o, str) else json.dumps(o)[:200]}")
    ts = [threading.Thread(target=one, args=(11 + i, 40 + 10 * i, lengths[i])) for i in range(3)]
    for t in ts:
        t.start(); time.sleep(0.02)
    for t in ts:
        t.join()
    one(91, 50, 60)
    if errors:
        die(errors[0])

cmd = sys.argv[1]
if cmd in ("run", "together"):
    base, vocab, out, steps = sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5])
    nums = [int(x) for x in sys.argv[6:]]
    run(base, vocab, out, steps, list(zip(nums[0::2], nums[1::2])), cmd == "together")
elif cmd == "compare":
    compare(sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5]), int(sys.argv[6]))
elif cmd == "comego":
    comego(sys.argv[2], int(sys.argv[3]), sys.argv[4] if len(sys.argv) > 4 else "")
else:
    die("unknown command " + cmd)
PY

# serve <name> <llama-server args...>: the synthetic model behind this fork's llama-server on card 1, its training
# context lifted so prompts can go past 1024 tokens; card 1's lock is held on fd 9 while any server of the case runs.
# Sets BASE. Every server a case starts is stopped when the case ends (stop_servers).
serve() {
    sname=$1; shift
    port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])')
    if [ -z "$LOCKED" ]; then
        exec 9> "$LOCK"
        flock 9
        LOCKED=1
    fi
    CUDA_VISIBLE_DEVICES=1 "./$B/bin/llama-server" -m "$MODEL" --override-kv inkling2.context_length=int:262144 \
        --host 127.0.0.1 --port "$port" "$@" > "$T/$sname.log" 2>&1 &
    spid=$!
    echo "$spid" >> "$T/pids"
    i=0
    while [ $i -lt 240 ]; do
        kill -0 "$spid" 2>/dev/null \
            || fail "llama-server $* with the synthetic inkling2 model exited before it was ready: $(grep -m1 -E 'GGML_ASSERT|error|exception|failed|invalid' "$T/$sname.log" || tail -n 1 "$T/$sname.log")"
        if python3 -c 'import sys,urllib.request; sys.exit(0 if urllib.request.urlopen(sys.argv[1], timeout=2).status == 200 else 1)' \
            "http://127.0.0.1:$port/health" 2>/dev/null; then
            BASE="http://127.0.0.1:$port"
            return 0
        fi
        i=$((i + 1))
        sleep 0.5
    done
    fail "llama-server $* with the synthetic inkling2 model was not ready after 120 s: $(tail -n 1 "$T/$sname.log")"
}

stop_servers() {
    if [ -f "$T/pids" ]; then
        for p in $(cat "$T/pids"); do
            kill "$p" 2>/dev/null
        done
        for p in $(cat "$T/pids"); do
            wait "$p" 2>/dev/null
        done
        rm -f "$T/pids"
    fi
    if [ -n "$LOCKED" ]; then
        flock -u 9 2>/dev/null
        exec 9>&-
        LOCKED=
    fi
}

# client <args...>: the client above; a failure fails the case with what it printed
client() {
    msg=$(python3 -I "$T/client.py" "$@" 2>&1) || fail "${msg:-the client crashed}"
}

# served logits are held to agree within 2e-3 (the bar the frozen inkling-server-parity-state cases use): on this
# fixture changing one earlier token of a prompt moves them by 0.012 or more, while on Nov 17, 2026 (one RTX PRO 6000,
# card 1, F16 cache) CUDA against the CPU read 1.1e-4 at 200 and at 1200 tokens, and a prompt read in 16-token chunks
# against one read whole 1.8e-4
TOL=2e-3

run_case() {
    name=$1; fn=$2; shift 2
    REASON="$T/$name.reason"; : > "$REASON"
    ( set +e; LOCKED=; trap stop_servers EXIT; "$fn" "$@" ) > "$T/$name.log" 2>&1
    rc=$?
    if [ "$rc" -eq 0 ]; then
        echo "PASS $name"
    else
        why=$(head -n 1 "$REASON" | tr -d '\r')
        [ -n "$why" ] || why="exited $rc: $(tail -n 1 "$T/$name.log" | tr -d '\r')"
        echo "FAIL $name: $why"
    fi
}

PREFIX=inkling-rel

# The next-token logits of a 1500-token prompt, whose last rows reach keys more than 1024 positions back, served on
# card 1 against the same prompt served on the CPU alone
case_prompt_past_1024_agrees_with_cpu() {
    fixture
    serve srv-cpu -dev none -ngl 0 -c 4096 -np 1
    client run "$BASE" "$VOCAB" "$T/p-cpu.json" 1 11 1500
    stop_servers
    serve srv-cuda -ngl 99 -fa on -c 4096 -np 1
    client run "$BASE" "$VOCAB" "$T/p-cuda.json" 1 11 1500
    stop_servers
    client compare "a 1500-token prompt on CUDA against the CPU" "$T/p-cuda.json" "$T/p-cpu.json" "$TOL" "$VOCAB"
}

# 24 greedy decode steps after a 1012-token prompt, so the decoded rows cross distance 1024 to the first keys,
# served on card 1 against the CPU alone, step by step
case_decode_past_1024_agrees_with_cpu() {
    fixture
    serve srv-cpu -dev none -ngl 0 -c 4096 -np 1
    client run "$BASE" "$VOCAB" "$T/d-cpu.json" 24 23 1012
    stop_servers
    serve srv-cuda -ngl 99 -fa on -c 4096 -np 1
    client run "$BASE" "$VOCAB" "$T/d-cuda.json" 24 23 1012
    stop_servers
    client compare "decoding past distance 1024 on CUDA against the CPU" "$T/d-cuda.json" "$T/d-cpu.json" "$TOL" "$VOCAB"
}

# The compute buffer a unified pool of 4096 cells and one of 65536 reserve for 512-row ubatches with flash attention.
# A bias over heads x cells x rows grows by n_head x 61440 x 512 F16 values between the two; the growth must stay
# under that one tensor. On Nov 17, 2026 the materialized bias grew card 1's compute buffer by 1140 MiB on a
# 4-head fixture, against 240 MiB for one such tensor.
compute_buffer() { # compute_buffer <cells>: the largest CUDA0 compute buffer the server reserved, in MiB
    serve "srv-buf-$1" -ngl 99 -fa on -c "$1" -np 4 --kv-unified -b 512 -ub 512 -v
    stop_servers
    v=$(sed -n -E 's/.*CUDA0 compute buffer size = *([0-9.]+) MiB.*/\1/p' "$T/srv-buf-$1.log" | sort -g | tail -n 1)
    [ -n "$v" ] || fail "llama-server -c $1 printed no CUDA0 compute buffer size"
    echo "$v" > "$T/buf-$1"
}

case_no_bias_grows_with_pool() {
    fixture
    compute_buffer 4096
    compute_buffer 65536
    small=$(cat "$T/buf-4096"); big=$(cat "$T/buf-65536")
    msg=$(python3 -c '
import sys
small, big, nh = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
one = nh * (65536 - 4096) * 512 * 2 / (1 << 20)
if big - small >= one:
    print(f"the CUDA0 compute buffer grows by {big - small:.0f} MiB from a 4096-cell pool to a 65536-cell one "
          f"({small:.0f} to {big:.0f} MiB); one bias over the {nh} heads, the added cells and 512 rows is {one:.0f} MiB")
    sys.exit(1)' "$small" "$big" "$NHEAD") || fail "$msg"
}

# Three sessions start together and leave one by one, then a fourth comes: the graphs-reused count each session
# reports when it ends must grow by at least half the decode steps since the last report (60, 40, 40 and 60 steps).
# On Nov 17, 2026 the fixture reused 56, then 40 and 39 more.
case_graph_reuse_holds() {
    fixture
    serve srv-reuse -ngl 99 -fa on -c 4096 -np 3 --kv-unified
    client comego "$BASE" "$VOCAB"
    stop_servers
    counts=$(sed -n -E 's/.*graphs reused = *([0-9]+).*/\1/p' "$T/srv-reuse.log" | tr '\n' ' ')
    msg=$(python3 -c '
import sys
c = [int(x) for x in sys.argv[1].split()]
if len(c) != 4:
    print(f"the server reported graphs reused {len(c)} times for 4 sessions: {c}"); sys.exit(1)
steps = [60, 40, 40, 60]
prev = 0
for i, (n, s) in enumerate(zip(c, steps)):
    if n - prev < s // 2:
        print(f"graphs reused went from {prev} to {n} over the {s} decode steps before session end {i + 1} ({c})"); sys.exit(1)
    prev = n' "$counts") || fail "$msg"
}

# the ctest cases take card 1's lock themselves, so they run before the servers
run_case inkling-rel-ctest-cases-registered                   case_ctest_registered
run_case inkling-rel-ctest-cases-pass-on-card-1               case_ctest_pass
run_case inkling-rel-ctest-cases-all-run                      case_ctest_all_run
run_case inkling2-fixture-agrees                              case_fixture_agrees
run_case inkling2-prompt-past-distance-1024-agrees-with-cpu   case_prompt_past_1024_agrees_with_cpu
run_case inkling2-decode-past-distance-1024-agrees-with-cpu   case_decode_past_1024_agrees_with_cpu
run_case inkling2-no-per-head-bias-grows-with-the-cache-pool  case_no_bias_grows_with_pool
run_case inkling2-graph-reuse-holds-while-sessions-come-and-go case_graph_reuse_holds

exit 0
