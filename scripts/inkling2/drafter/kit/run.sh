#!/usr/bin/env bash
# Driver of the DSpark recording kit: runs the stages named in stages.txt, in order, and stops at the first that
# fails. Everything that is slow to make again (the engine build, the datasets, the fixture, the recordings) lives in
# one folder beside rig's jobs, so a later job on the same box finds it.
set -euo pipefail
umask 077
KIT=$PWD
SHA=$(tr -d ' \n' < "$KIT/engine.sha")
MODE=$(tr -d ' \n' < "$KIT/mode.txt")
CACHE=${DSPARK_CACHE:-$(dirname "$(dirname "$KIT")")/dspark}
SRC=$CACHE/src-$SHA
BLD=$CACHE/build-$SHA
VENV=$CACHE/venv
HUB=$CACHE/hf/hub
DATA=$CACHE/data
FIX=$CACHE/fixture
PLANS=$CACHE/plans-$MODE
REC=$CACHE/rec-$MODE
LOGS=$CACHE/logs
PACKROOT=$CACHE/pack
PACKS=${DSPARK_PACKS:-$PACKROOT/model}
TOOL=$BLD/bin/dspark_tool
RES=$KIT/results
THREADS=$(nproc)
[ "$THREADS" -gt 32 ] && THREADS=32
mkdir -p "$CACHE" "$RES" "$LOGS"
export UV_CACHE_DIR=$CACHE/uv-cache UV_PYTHON_INSTALL_DIR=$CACHE/uv-python HF_HUB_DISABLE_TELEMETRY=1
PY=("$VENV/bin/python" -I)

say() { printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*"; }

includes() {
    printf -- '-I%s ' "$1" "$1/include" "$1/common" "$1/tools/server" "$1/tools/mtmd" "$1/vendor" "$1/ggml/include"
}

# the end of a long log, where its failure is
excerpt() { [ -f "$1" ] && tail -c "${2:-6000}" "$1"; return 0; }

stage_tools() {
    local need=0 p
    for p in cmake ninja g++ python3; do command -v "$p" > /dev/null || need=1; done
    if [ "$need" = 1 ]; then
        apt-get -qq update
        DEBIAN_FRONTEND=noninteractive apt-get -qq install -y cmake ninja-build build-essential python3 python3-venv > "$LOGS/apt.log" 2>&1 ||
            { excerpt "$LOGS/apt.log"; return 1; }
    fi
    if [ ! -x "$VENV/bin/python" ]; then
        chmod +x "$KIT/uv"
        "$KIT/uv" venv --quiet "$VENV"
        "$KIT/uv" pip install --quiet --python "$VENV/bin/python" 'numpy>=2' pyyaml tqdm 'huggingface_hub>=0.25' 'pyarrow>=15'
    fi
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv > "$RES/gpu.csv"
    {
        echo "cores $(nproc)"; free -g | sed -n 2p; df -h "$CACHE" | tail -1
        g++ --version | head -1; cmake --version | head -1; nvcc --version | tail -1; "$VENV/bin/python" --version
    } > "$RES/box.txt" 2>&1
    cat "$RES/gpu.csv" "$RES/box.txt"
}

# torch for the trainer, held to a sum on every card: a wheel that does not know the card fails here and not hours in
torch_works() {
    "${PY[@]}" - <<'PYEOF'
import torch
print("torch %s, cuda %s, %d cards" % (torch.__version__, torch.version.cuda, torch.cuda.device_count()))
assert torch.cuda.is_available() and torch.cuda.device_count() > 0
for i in range(torch.cuda.device_count()):
    x = torch.randn(1024, 1024, device="cuda:%d" % i)
    y = x.to(torch.bfloat16).float() @ x.half().float().T
    assert bool(torch.isfinite(y).all())
    free, total = torch.cuda.mem_get_info(i)
    print("card %d %s: %.1f of %.1f GB free" % (i, torch.cuda.get_device_name(i), free / 2**30, total / 2**30))
PYEOF
}

stage_torch() {
    local index
    if torch_works > "$RES/torch.txt" 2>&1; then cat "$RES/torch.txt"; return 0; fi
    chmod +x "$KIT/uv"
    for index in default https://download.pytorch.org/whl/cu128; do
        if [ "$index" = default ]; then
            "$KIT/uv" pip install --quiet --python "$VENV/bin/python" torch > "$LOGS/torch.log" 2>&1 || { excerpt "$LOGS/torch.log"; continue; }
        else
            "$KIT/uv" pip install --quiet --reinstall --python "$VENV/bin/python" torch --index-url "$index" > "$LOGS/torch.log" 2>&1 ||
                { excerpt "$LOGS/torch.log"; continue; }
        fi
        if torch_works > "$RES/torch.txt" 2>&1; then say "torch: from $index"; cat "$RES/torch.txt"; return 0; fi
        say "torch: the wheel from $index does not run on these cards"; excerpt "$RES/torch.txt" 3000
    done
    return 1
}

# compile the tool against the headers only, so its errors come back without waiting for the engine build
stage_syntax() {
    local s=$KIT/source waited=0
    if [ ! -d "$s" ]; then
        s=$SRC
        while [ ! -d "$s" ] && [ "$waited" -lt 600 ]; do sleep 5; waited=$((waited + 5)); done
    fi
    [ -d "$s" ] || { say "syntax: no engine source"; return 1; }
    # shellcheck disable=SC2046
    if g++ -std=c++17 -fsyntax-only -fmax-errors=80 -Wall -Wextra -Wno-unused-parameter -march=x86-64-v3 -pthread \
        $(includes "$s") "$KIT/dspark_tool.cpp" > "$RES/tool-syntax.log" 2>&1; then
        say "syntax: clean"; excerpt "$RES/tool-syntax.log" 3000
    else
        say "syntax: errors"; grep -n -E "error|Error" "$RES/tool-syntax.log" | head -60; return 1
    fi
}

stage_engine() {
    if [ -f "$BLD/.done" ]; then say "engine: built before"; return 0; fi
    if [ ! -d "$SRC" ]; then
        [ -d "$KIT/source" ] || { say "engine: the kit has no source/ (rig job run --source ... --commit ...)"; return 1; }
        rm -rf "$SRC.tmp"; cp -a "$KIT/source" "$SRC.tmp"; mv "$SRC.tmp" "$SRC"
    fi
    cmake -S "$SRC" -B "$BLD" -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON -DGGML_CCACHE=OFF -DGGML_NATIVE=OFF -DGGML_AVX512=OFF \
        -DCMAKE_C_FLAGS=-march=x86-64-v3 -DCMAKE_CXX_FLAGS=-march=x86-64-v3 \
        -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=ON \
        -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_UI=OFF \
        -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_CURL=OFF > "$LOGS/engine-configure.log" 2>&1 ||
        { excerpt "$LOGS/engine-configure.log"; return 1; }
    cmake --build "$BLD" --target llama-server --parallel "$(nproc)" > "$LOGS/engine-build.log" 2>&1 ||
        { grep -n -B2 -A12 -E "error|FAILED" "$LOGS/engine-build.log" | head -150; return 1; }
    touch "$BLD/.done"
    ls "$BLD/bin" > "$RES/engine-bin.txt"
    say "engine: built"
}

stage_tool() {
    local h
    h=$(sha256sum "$KIT/dspark_tool.cpp" | cut -c1-16)
    if [ -x "$TOOL" ] && [ "$(cat "$BLD/.tool-hash" 2> /dev/null)" = "$h" ]; then say "tool: up to date"; return 0; fi
    # shellcheck disable=SC2046
    g++ -std=c++17 -O2 -march=x86-64-v3 -pthread -fmax-errors=80 $(includes "$SRC") "$KIT/dspark_tool.cpp" -o "$TOOL.tmp" \
        -L"$BLD/bin" -lllama-server-impl -lllama-common -lllama -lmtmd -lggml -lggml-base -Wl,-rpath,"$BLD/bin" \
        > "$RES/tool-compile.log" 2>&1 || { grep -n -E "error|undefined" "$RES/tool-compile.log" | head -60; return 1; }
    mv "$TOOL.tmp" "$TOOL"
    echo "$h" > "$BLD/.tool-hash"
    say "tool: built"
}

# Large inputs ride in parts of 1 GiB, since rig reads every file of a kit whole and gives its packing ten minutes:
# each job's parts are moved beside the cache, so the release and the draft can arrive over as many jobs as it takes.
PARTS=$CACHE/parts
stage_parts() {
    local f n=0
    mkdir -p "$PARTS"
    for f in "$KIT"/*.part-[0-9]*; do
        [ -f "$f" ] || continue
        mv -- "$f" "$PARTS/"
        n=$((n + 1))
    done
    say "parts: $n arrived with this job; held now:"
    ls -l "$PARTS" | awk 'NR > 1 {print "    " $5, $9}'
}

# await_parts NAME: the parts of NAME are uploaded beside the cache while earlier stages run, so a stage that joins
# them first waits until they hold the bytes parts.sizes gives for NAME (an hour at most). A name the file does not
# list is joined as it stands.
await_parts() {
    local name=$1 want have n=0
    want=$(awk -v n="$name" '$1 == n {print $2}' "$KIT/parts.sizes" 2> /dev/null || true)
    [ -n "$want" ] || return 0
    while :; do
        have=$(find "$PARTS" -maxdepth 1 -name "$name.part-*" -printf '%s\n' 2> /dev/null | awk '{s += $1} END {print s + 0}')
        [ "$have" = "$want" ] && break
        if [ "$n" -ge 360 ]; then say "parts: $name holds $have of $want bytes after an hour"; return 1; fi
        [ $((n % 6)) = 0 ] && say "parts: waiting for $name, $have of $want bytes"
        n=$((n + 1))
        sleep 10
    done
    say "parts: $name is whole ($want bytes)"
}

# The pack's public sources (320 GB), fetched while the release is still arriving: the restore's own fetches under the
# restore's own lock, so the sealed restore that follows waits for them and then finds every source staged.
stage_prefetch() {
    if [ "$MODE" = fixture ]; then say "prefetch: the fixture has none"; return 0; fi
    if [ -s "$PACKROOT/model/SHA256SUMS" ]; then say "prefetch: the pack stands"; return 0; fi
    mkdir -p "$PACKROOT"
    chmod +x "$KIT/uv"
    HF_XET_HIGH_PERFORMANCE=1 "$KIT/uv" run --quiet --script "$KIT/prefetch.py" "$PACKROOT" "$KIT/prefetch" \
        > "$LOGS/prefetch.log" 2>&1 || { excerpt "$LOGS/prefetch.log" 8000; return 1; }
    grep -a -E "^(Verified|Staged|All pinned)" "$LOGS/prefetch.log" | tail -60
    du -sh "$PACKROOT"/.inkling-stock-* "$PACKROOT/auxiliary"
    df -h "$CACHE" | tail -1
}

# The served pack, in pack mode: the release's sealed restore from pinned public sources into $PACKROOT/model, held
# to the release's sums by the restore itself. The stock shards and auxiliary tensors it read (320 GB) go once the
# pack stands, since the recordings need the room and they can be fetched again.
stage_pack() {
    if [ "$MODE" = fixture ]; then say "pack: the fixture has none"; return 0; fi
    mkdir -p "$PACKROOT"
    if [ ! -s "$PACKROOT/model/SHA256SUMS" ]; then
        # the release's parts, joined here and held to its sum
        await_parts release.tar
        cat "$PARTS"/release.tar.part-* > "$PACKROOT/release.tar"
        (cd "$PACKROOT" && sha256sum -c -- "$KIT/release.sha256")
        rm -rf "$PACKROOT/release"
        mkdir -p "$PACKROOT/release"
        tar -xf "$PACKROOT/release.tar" -C "$PACKROOT/release"
        rm -f "$PACKROOT/release.tar"
        chmod +x "$KIT/uv"
        # the hub's client may use the whole line and every core: 320 GB of sources on a box billed by the hour
        HF_XET_HIGH_PERFORMANCE=1 "$KIT/uv" run --quiet --script "$KIT/restore-xet.py" "$PACKROOT" > "$LOGS/restore.log" 2>&1 ||
            { excerpt "$LOGS/restore.log" 8000; return 1; }
        test -s "$PACKROOT/model/SHA256SUMS"
        rm -f "$PARTS"/release.tar.part-*
    fi
    tail -5 "$LOGS/restore.log" 2> /dev/null || true
    rm -rf "$PACKROOT"/.inkling-stock-* "$PACKROOT/auxiliary"
    ls -l "$PACKROOT/model" | tee "$RES/pack.txt"
    df -h "$CACHE" | tail -1
}

# The stock draft the kit carries (the pack's own head, with the target's vocabulary tensors), held to its sum.
stage_draft() {
    local name
    if [ "$MODE" = fixture ]; then say "draft: the fixture writes its own"; return 0; fi
    name=$(awk '{print $2}' "$KIT/draft.sha256")
    case $name in "" | */*) say "draft: draft.sha256 names no file of the kit"; return 1 ;; esac
    if [ -f "$PACKS/$name" ] && (cd "$PACKS" && sha256sum -c -- "$KIT/draft.sha256" > /dev/null 2>&1); then
        say "draft: in place"
        return 0
    fi
    test -d "$PACKS"
    # in parts like the release, joined beside the pack and held to its sum before it takes its name
    await_parts "$name"
    mkdir -p "$PACKROOT/draft-in"
    cat "$PARTS/$name".part-* > "$PACKROOT/draft-in/$name"
    (cd "$PACKROOT/draft-in" && sha256sum -c -- "$KIT/draft.sha256")
    mv -- "$PACKROOT/draft-in/$name" "$PACKS/$name"
    rmdir "$PACKROOT/draft-in"
    rm -f "$PARTS/$name".part-*
    ls -l "$PACKS/$name"
}

stage_data() {
    if [ -f "$DATA/.ok" ]; then say "data: fetched and converted before"; return 0; fi
    mkdir -p "$DATA"
    "${PY[@]}" "$KIT/fetch_sources.py" "$HUB" > "$LOGS/fetch.log" 2>&1 || { excerpt "$LOGS/fetch.log"; return 1; }
    "${PY[@]}" "$KIT/build_requests_v5.py" --hub "$HUB" --out "$DATA" > "$LOGS/convert.log" 2>&1 ||
        { excerpt "$LOGS/convert.log"; return 1; }
    (cd "$DATA" && sha256sum -c "$KIT/requests.sha256")
    touch "$DATA/.ok"
    cp "$DATA/SOURCES.md" "$RES/"
}

# The draft is written again when it is not yet laid out as the served one (heads 128 wide, matrices in BF16). A target
# written before stays as it is: the recordings were made from it, and none of them read the draft.
stage_fixture() {
    local only=()
    if [ -f "$FIX/.ok" ] && [ -f "$FIX/.draft-bf16" ]; then say "fixture: written before"; return 0; fi
    [ -f "$FIX/.ok" ] && only=(--draft-only)
    "${PY[@]}" "$KIT/make_fixture.py" --gguf-py "$SRC/gguf-py" --target "$KIT/fixture-target.gguf" \
        --tokenizer "$KIT/tokenizer.gguf" --out "$FIX" "${only[@]}"
    touch "$FIX/.ok" "$FIX/.draft-bf16"
}

# the recording tool, started as the launcher would start the server
# launcher options a probe adds (a slot count, a changed engine variable); a recording to train on adds none
LAUNCH=()
tool() {
    if [ "$MODE" = fixture ]; then
        "${PY[@]}" "$KIT/launch.py" --source "$SRC" --tool "$TOOL" --model "$FIX/target.gguf" "${LAUNCH[@]}" "$@"
    else
        "${PY[@]}" "$KIT/launch.py" --source "$SRC" --tool "$TOOL" --packs "$PACKS" "${LAUNCH[@]}" "$@"
    fi
}

taps() { if [ -n "${TAPS:-}" ]; then echo "--taps $TAPS"; elif [ "$MODE" = fixture ]; then echo "--taps 0,3"; fi; }

REQUESTS=(--requests "$DATA/requests-heldout.jsonl" --requests "$DATA/requests-train.jsonl")

stage_prepare() {
    tool --show prepare > "$RES/launch.json"
    tool prepare "${REQUESTS[@]}" --out "$CACHE/prepared-$MODE.jsonl" --threads "$THREADS" > "$LOGS/prepare.log" 2>&1 ||
        { excerpt "$LOGS/prepare.log"; return 1; }
    excerpt "$LOGS/prepare.log" 2500
    cp "$CACHE/prepared-$MODE.jsonl" "$RES/prepared.jsonl"
}

stage_plan() {
    "${PY[@]}" "$KIT/make_plan.py" --prepared "$CACHE/prepared-$MODE.jsonl" --heldout "$DATA/requests-heldout.jsonl" \
        --train "$DATA/requests-train.jsonl" --out "$PLANS" | tee "$RES/plan.txt"
    cp "$PLANS"/*.json "$RES/" 2> /dev/null || true
}

# capture NAME PLAN [tool options]: one recording, its log kept beside the recordings and its end shown
capture() {
    local name=$1 plan=$2
    shift 2
    case $plan in /*) ;; *) plan=$PLANS/$plan ;; esac
    mkdir -p "$REC"
    # a recording is one engine's: the fixture's is made again by another, the pack's stops the job and stays
    if [ -d "$REC/$name" ] && [ "$(cat "$REC/$name.engine" 2> /dev/null)" != "$SHA" ]; then
        if [ "$MODE" != fixture ]; then
            say "capture: $REC/$name is another engine's recording ($(cat "$REC/$name.engine" 2> /dev/null || echo unnamed)); move it away first"
            return 1
        fi
        say "capture: $name was recorded by another engine; recording it again"
        rm -rf "${REC:?}/$name"
    fi
    echo "$SHA" > "$REC/$name.engine"
    # shellcheck disable=SC2046
    tool capture --plan "$plan" "${REQUESTS[@]}" --out "$REC/$name" --threads "$THREADS" $(taps) "$@" \
        > "$LOGS/capture-$name.log" 2>&1 || { excerpt "$LOGS/capture-$name.log" 12000; return 1; }
    grep -a -E "^[0-9:.]+ (dspark|rec)|dspark:" "$LOGS/capture-$name.log" | tail -25 || true
    cp "$REC/$name/summary.json" "$RES/summary-$name.json"
    cp "$REC/$name/meta.json" "$RES/meta-$name.json"
    head -c 400000 "$LOGS/capture-$name.log" > "$RES/capture-$name.head.log"
    tail -c 400000 "$LOGS/capture-$name.log" > "$RES/capture-$name.tail.log"
}

model_file() { if [ "$MODE" = fixture ]; then echo "$FIX/target.gguf"; else ls "$PACKS"/*Q3_K_M*.gguf | head -1; fi; }

# the draft the launcher serves: the fixture's, or the pack's with the target's vocabulary tensors in it
draft_file() { if [ "$MODE" = fixture ]; then echo "$FIX/draft.gguf"; else ls "$PACKS"/*DSpark*own-head*.gguf | head -1; fi; }

# words a stage adds to its command, from a file of the kit when the kit has one (train.args: a rate, a second card)
words() { if [ -f "$KIT/$1" ]; then tr '\n' ' ' < "$KIT/$1"; fi; }

TRAINED=$CACHE/train-$MODE

# The trainer's copy of the draft against the engine's own numbers, on a card: the fixture draft's dump always, and
# the real draft's when the kit carries it (parity/real-dump, made with the pack's draft).
stage_parity() {
    "${PY[@]}" "$KIT/selftest.py" --gguf-py "$SRC/gguf-py" --parity "$KIT/parity/fixture-dump" \
        --draft "$KIT/parity/fixture-draft.gguf" --device cuda | tee "$RES/parity-fixture.txt"
    if [ "$MODE" != fixture ] && [ -d "$KIT/parity/real-dump" ]; then
        "${PY[@]}" "$KIT/selftest.py" --gguf-py "$SRC/gguf-py" --parity "$KIT/parity/real-dump" \
            --draft "$(draft_file)" --device cuda | tee "$RES/parity-real.txt"
    fi
}

# what the served draft gets on the held-out recording, before any training
stage_baseline() {
    # shellcheck disable=SC2046
    "${PY[@]}" "$KIT/train_dspark.py" --gguf-py "$SRC/gguf-py" --draft "$(draft_file)" --target "$(model_file)" \
        --eval "$REC/heldout" --eval-only --out "$CACHE/baseline-$MODE" $(words baseline.args) 2>&1 | tee "$LOGS/baseline.log"
    cp "$CACHE/baseline-$MODE/eval-before.json" "$RES/eval-stock.json"
    # with --walk in baseline.args: what the server's rounds keep of the draft as served, and the most a draft can keep
    if [ -f "$CACHE/baseline-$MODE/walk-before.json" ]; then cp "$CACHE/baseline-$MODE/walk-before.json" "$RES/walk-stock.json"; fi
    tail -c 20000 "$LOGS/baseline.log" > "$RES/baseline.tail.log"
}
# the tuned draft's rounds, walked as the stock one's are
stage_walk_tuned() {
    # shellcheck disable=SC2046
    "${PY[@]}" "$KIT/train_dspark.py" --gguf-py "$SRC/gguf-py" --draft "$(tuned_file)" --target "$(model_file)" \
        --eval "$REC/heldout" --eval-only --out "$CACHE/walk-tuned-$MODE" $(words baseline.args) 2>&1 | tee "$LOGS/walk-tuned.log"
    cp "$CACHE/walk-tuned-$MODE/eval-before.json" "$RES/eval-tuned.json"
    cp "$CACHE/walk-tuned-$MODE/walk-before.json" "$RES/walk-tuned.json"
    tail -c 20000 "$LOGS/walk-tuned.log" > "$RES/walk-tuned.tail.log"
}

# the recordings a training reads: a, or those train.rec names (a b: both)
train_recs() {
    local name
    for name in $(if [ -f "$KIT/train.rec" ]; then cat "$KIT/train.rec"; else echo a; fi); do echo "$REC/$name"; done
}

# train on those recordings, evaluate on the held-out one before and after, and write the tuned draft as a GGUF the
# server loads
stage_train() {
    # shellcheck disable=SC2046
    "${PY[@]}" "$KIT/train_dspark.py" --gguf-py "$SRC/gguf-py" --draft "$(draft_file)" --target "$(model_file)" \
        --train $(train_recs) --eval "$REC/heldout" --out "$TRAINED" $(words train.args) 2>&1 | tee "$LOGS/train.log"
    cp "$TRAINED"/eval-before.json "$TRAINED"/eval-after.json "$TRAINED"/summary.json "$TRAINED"/train.jsonl "$RES/"
    tail -c 200000 "$LOGS/train.log" > "$RES/train.tail.log"
    ls -l "$TRAINED"
}

# where a sweep's trials go under the cache: sweep-MODE, or the name sweep.name gives a later sweep (full: full-MODE)
sweep_dir() { echo "$(if [ -f "$KIT/sweep.name" ]; then tr -d ' \n' < "$KIT/sweep.name"; else echo sweep; fi)-$MODE"; }

# The quick fine-tune at several settings at once, a card each, so one pass over a recording answers which rate to
# train at. sweep.args holds a trial a line: its name, then the trainer's words for it. Every trial is evaluated on
# the same held-out requests before and after; the tuned drafts stay beside the cache for a live check of the best.
stage_sweep() {
    local name words card=0 cards pids=() names=() rc=0 i out
    cards=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
    out=$CACHE/$(sweep_dir)
    mkdir -p "$out"
    while read -r name words; do
        case $name in "" | "#"*) continue ;; esac
        if [ "$card" -ge "$cards" ]; then say "sweep: more trials than the $cards cards"; rc=1; break; fi
        # shellcheck disable=SC2086,SC2046
        "${PY[@]}" "$KIT/train_dspark.py" --gguf-py "$SRC/gguf-py" --draft "$(draft_file)" --target "$(model_file)" \
            --train $(train_recs) --eval "$REC/heldout" --out "$out/$name" --device "cuda:$card" $words \
            > "$LOGS/sweep-$name.log" 2>&1 &
        pids+=($!)
        names+=("$name")
        say "sweep: $name on card $card: $words"
        card=$((card + 1))
    done < "$KIT/sweep.args"
    for i in "${!pids[@]}"; do
        name=${names[$i]}
        if wait "${pids[$i]}"; then
            cp "$out/$name/eval-before.json" "$RES/sweep-$name-eval-before.json"
            cp "$out/$name/eval-after.json" "$RES/sweep-$name-eval-after.json"
            cp "$out/$name/summary.json" "$RES/sweep-$name-summary.json"
            if [ -f "$out/$name/walk-after.json" ]; then cp "$out/$name/walk-after.json" "$RES/sweep-$name-walk-after.json"; fi
        else
            rc=1
            say "sweep: $name failed"
            excerpt "$LOGS/sweep-$name.log" 3000
        fi
        [ -f "$out/$name/train.jsonl" ] && cp "$out/$name/train.jsonl" "$RES/sweep-$name-train.jsonl"
        cp "$LOGS/sweep-$name.log" "$RES/sweep-$name.log"
    done
    for name in "${names[@]}"; do
        printf '%s: %s\n' "$name" "$(grep -a -h -E "trained [0-9]+ steps|accepted a round at depth" "$LOGS/sweep-$name.log" | cut -d' ' -f2- | tr '\n' ';')"
    done | tee "$RES/sweep.txt"
    return "$rc"
}

# live NAME DRAFT: the server as the launcher starts it, drafting with DRAFT; then the target's states for the same
# tokens, recorded as every recording is; then the trainer's copy of DRAFT asked for the tokens the server drafted.
live() {
    local name=$1 draft=$2 where
    if [ "$MODE" = fixture ]; then where=(--model "$FIX/target.gguf"); else where=(--packs "$PACKS"); fi
    rm -rf "${REC:?}/$name" "${PLANS:?}/$name"
    # shellcheck disable=SC2046
    "${PY[@]}" "$KIT/live_check.py" serve --source "$SRC" --server "$BLD/bin/llama-server" "${where[@]}" --draft "$draft" \
        --rec "$REC/heldout" --plan "$PLANS/plan-heldout.jsonl" --out "$PLANS/$name" --log "$LOGS/$name-server.log" \
        $(words live.args) || { excerpt "$LOGS/$name-server.log" 12000; return 1; }
    cp "$PLANS/$name/rounds.json" "$RES/$name-rounds.json"
    LAUNCH=(--slots 1)
    capture "$name" "$PLANS/$name/plan.jsonl" --no-cuts --requests "$PLANS/$name/requests.jsonl"
    LAUNCH=()
    # shellcheck disable=SC2046
    "${PY[@]}" "$KIT/live_check.py" compare --gguf-py "$SRC/gguf-py" --draft "$draft" --target "$(model_file)" \
        --rec "$REC/$name" --rounds "$PLANS/$name/rounds.json" --out "$RES/$name.json" $(words compare.args) 2>&1 |
        tee "$RES/$name.txt"
}

# the tuned draft a later stage takes: the full training's, or the one tuned.path names under the cache (a trial of
# the sweep: sweep-pack/NAME/draft-tuned.gguf)
tuned_file() {
    if [ -f "$KIT/tuned.path" ]; then echo "$CACHE/$(tr -d ' \n' < "$KIT/tuned.path")"; else echo "$TRAINED/draft-tuned.gguf"; fi
}

stage_live() { live live-stock "$(draft_file)"; }
stage_live_tuned() { live live-tuned "$(tuned_file)"; }

# bench NAME [DRAFT [WORD...]]: what DRAFT gets on the server as it is served while sessions are added, on sessions cut from the
# held-out recording, and how fast the server writes with it (no DRAFT: the same server without drafting)
bench() {
    local name=$1 draft=${2:-} where with=()
    shift
    [ $# -gt 0 ] && shift
    if [ "$MODE" = fixture ]; then where=(--model "$FIX/target.gguf"); else where=(--packs "$PACKS"); fi
    [ -n "$draft" ] && with=(--draft "$draft")
    # shellcheck disable=SC2046
    "${PY[@]}" "$KIT/live_bench.py" --source "$SRC" --server "$BLD/bin/llama-server" "${where[@]}" "${with[@]}" \
        --rec "$REC/heldout" --out "$RES/$name.json" --log "$LOGS/$name-server.log" $(words bench.args) "$@" 2>&1 |
        tee "$RES/$name.txt" || { excerpt "$LOGS/$name-server.log" 12000; return 1; }
    grep -a "draft acceptance" "$LOGS/$name-server.log" | tail -60 > "$RES/$name-server-acceptance.txt" || true
    # at log level 4 the draft code says, at each reply's end, the time it has spent so far drafting and accepting
    grep -a -E "statistics|acc per pos" "$LOGS/$name-server.log" | cut -c1-600 > "$RES/$name-server-draft-time.txt" || true
    [ -s "$RES/$name-server-draft-time.txt" ] || rm -f "$RES/$name-server-draft-time.txt"
    # with LLAMA_DECODE_TRACE=1 the engine marks each iteration's stages: where a round's time goes
    if grep -a -q "GLM53_TRACE" "$LOGS/$name-server.log"; then
        "${PY[@]}" "$KIT/trace_split.py" "$LOGS/$name-server.log" "$RES/$name-trace.json" 2>&1 | tee "$RES/$name-trace.txt" || true
    fi
}

stage_bench() { bench bench-stock "$(draft_file)"; }
stage_bench_tuned() { bench bench-tuned "$(tuned_file)"; }
stage_bench_none() { bench bench-none; }
# the stock draft asked for more tokens a round than the launcher asks (3): does a longer round write faster
stage_bench_d5() { bench bench-stock-d5 "$(draft_file)" --depth 5; }
stage_bench_d7() { bench bench-stock-d7 "$(draft_file)" --depth 7; }
# the stock draft again with the server logging at level 4: where a draft round's time goes as sessions are added
stage_bench_trace() {
    bench bench-stock-trace "$(draft_file)" --verbosity 4 --env LLAMA_DECODE_TRACE=1 --levels 23,16,8,4,2,1
}
# the same without a draft: what the target alone costs an iteration, by the sequences writing
stage_bench_trace_none() { bench bench-none-trace "" --env LLAMA_DECODE_TRACE=1 --levels 23,16,8,4,2,1; }

# replay NAME [DRAFT [WORD...]]: sessions cut from the held-out recording sent turn by turn with no slot named, the first
# alone and the others a few seconds apart, as coding agents send them: does the draft keep less in a session that began
# from another's prefix, in the sessions that started last, or in a session's later turns
replay() {
    local name=$1 draft=${2:-} where with=()
    shift
    [ $# -gt 0 ] && shift
    if [ "$MODE" = fixture ]; then where=(--model "$FIX/target.gguf"); else where=(--packs "$PACKS"); fi
    [ -n "$draft" ] && with=(--draft "$draft")
    # shellcheck disable=SC2046
    "${PY[@]}" "$KIT/live_replay.py" --source "$SRC" --server "$BLD/bin/llama-server" "${where[@]}" "${with[@]}" \
        --rec "$REC/heldout" --out "$RES/$name.json" --log "$LOGS/$name-server.log" $(words replay.args) "$@" 2>&1 |
        tee "$RES/$name.txt" || { excerpt "$LOGS/$name-server.log" 12000; return 1; }
    grep -a "draft acceptance" "$LOGS/$name-server.log" | tail -200 > "$RES/$name-server-acceptance.txt" || true
    # how the server chose each turn's slot, and what it copied or restored for it
    grep -a -i -E "selected slot|slot keeps|prefix|copied|copy|donor|restor|checkpoint|similarity|lcp|evict|purg|degrade|pos_max" "$LOGS/$name-server.log" |
        cut -c1-400 | tail -n 1500 > "$RES/$name-server-slots.txt" || true
}

stage_replay() { replay replay-stock "$(draft_file)"; }
stage_replay_tuned() { replay replay-tuned "$(tuned_file)"; }
# a probe beside it: the same sessions with the server reading every prompt itself, no prefix taken from another slot
stage_replay_nocopy() { replay replay-stock-nocopy "$(draft_file)" --env LLAMA_SLOT_PREFIX_COPY=0; }

# the tuned draft, to bring home
stage_pack_draft() {
    cp "$(tuned_file)" "$RES/draft-tuned.gguf"
    sha256sum "$RES/draft-tuned.gguf" | tee "$RES/draft-tuned.sha256"
}

# The smoke plan four ways, and the checks that hold a recording to what one session reading alone gets:
#   smoke-ref        one session, no continuation: what a right read is
#   smoke-ref-split  the same, each read ended at every cut: how far apart two right reads are
#   smoke            every session, with continuations, as the recordings to train on are made
#   smoke-replay     one session reading each continuation of smoke back as a prompt
stage_smoke() {
    rm -rf "$REC"/smoke* "$PLANS/replay-smoke"
    [ "$MODE" = fixture ] && TAPS=0,1,2,3
    LAUNCH=(--slots 1)
    capture smoke-ref plan-smoke.jsonl --no-cuts
    capture smoke-ref-split plan-smoke.jsonl --no-cuts --stops
    LAUNCH=()
    capture smoke plan-smoke.jsonl --trace
    "${PY[@]}" "$KIT/make_replay.py" --rec "$REC/smoke" --plan "$PLANS/plan-smoke.jsonl" --out "$PLANS/replay-smoke"
    LAUNCH=(--slots 1)
    capture smoke-replay "$PLANS/replay-smoke/plan.jsonl" --no-cuts --requests "$PLANS/replay-smoke/requests.jsonl"
    LAUNCH=()
    TAPS=
    if ! "${PY[@]}" "$KIT/check_record.py" --gguf-py "$SRC/gguf-py" --model "$(model_file)" --rec "$REC/smoke" \
        --ref "$REC/smoke-ref" --ref-split "$REC/smoke-ref-split" --replay "$REC/smoke-replay" | tee "$RES/check-smoke.txt"; then
        stage_diag || true
        return 1
    fi
}

# where the rows of the run with continuations stand apart from the one-session reads
stage_diag() {
    "${PY[@]}" "$KIT/diag_restore.py" --rec "$REC/smoke" --nocuts "$REC/smoke-ref" --stops "$REC/smoke-ref-split" |
        tee "$RES/diag-smoke.txt"
}

# One recording of the smoke plan for a probe: its name, the launcher options it adds, then tool options.
probe_run() {
    local name=$1 launch=$2
    shift 2
    # shellcheck disable=SC2206
    LAUNCH=($launch)
    say "probe: $name ($launch $*)"
    capture "$name" plan-smoke.jsonl --trace "$@" || say "probe: $name failed"
    LAUNCH=()
    if [ -f "$REC/$name/verify.txt" ]; then
        cp "$REC/$name/verify.txt" "$RES/verify-$name.txt"
        mkdir -p "$RES/verify-$name"
        cp "$REC/$name"/verify-*.bin "$RES/verify-$name/" 2> /dev/null || true
    fi
    cp "$REC/$name/trace.txt" "$RES/trace-$name.txt" 2> /dev/null || true
}

# Which rows a run of 23 sequences gets wrong, against one sequence read alone: the same plan with every layer
# recorded, read by one slot, by 23, by 23 with each sequence's state looked at around every decode, and by 23 with
# the engine keeping its recurrent rows another way.
stage_probe() {
    [ "$MODE" = fixture ] || { say "probe: a rehearsal stage"; return 1; }
    rm -rf "$REC"/p-*
    TAPS=0,1,2,3
    probe_run p-one "--slots 1" --no-cuts
    probe_run p-one-b "--slots 1" --no-cuts
    probe_run p-one-cuts "--slots 1"
    probe_run p-base ""
    probe_run p-nocuts "" --no-cuts
    probe_run p-verify "" --verify-states
    probe_run p-fixed "--set LLAMA_RS_FIXED_ROWS=1"
    probe_run p-any "--set LLAMA_GRAPH_ANY_SEQS=1"
    TAPS=
    stage_probe_compare
}

same_runs() {
    local a=$REC/$1 b=$REC/$2 same=0 differ=0 f
    for f in "$a"/req/*; do
        if cmp -s "$f" "$b/req/$(basename "$f")"; then same=$((same + 1)); else differ=$((differ + 1)); fi
    done
    echo "$1 and $2: $same request files are the same byte for byte, $differ differ"
}

# The same 23 sessions with each prompt decode in the server's shape (one micro-batch), then with two and with eight
# micro-batches a decode, against the one-session recording of stage probe.
stage_probe2() {
    [ "$MODE" = fixture ] || { say "probe2: a rehearsal stage"; return 1; }
    [ -d "$REC/p-one" ] || { say "probe2: no one-session recording to compare with (stage probe makes it)"; return 1; }
    rm -rf "$REC"/q-*
    TAPS=0,1,2,3
    probe_run q-one "--slots 1" --no-cuts
    probe_run q-view ""
    probe_run q-view-b ""
    probe_run q-view-nocuts "" --no-cuts
    probe_run q-view-verify "" --verify-states
    probe_run q-rows512 "" --step-rows 512
    probe_run q-rows2048 "" --step-rows 2048
    TAPS=
    {
        "${PY[@]}" "$KIT/probe_compare.py" --ref "$REC/p-one" --run "$REC/q-one" --run "$REC/q-view" --run "$REC/q-view-b" \
            --run "$REC/q-view-nocuts" --run "$REC/q-view-verify" --run "$REC/q-rows512" --run "$REC/q-rows2048"
        same_runs q-view q-view-b
    } | tee "$RES/probe2.txt"
}

stage_probe_compare() {
    "${PY[@]}" "$KIT/probe_compare.py" --ref "$REC/p-one" --run "$REC/p-one-b" --run "$REC/p-one-cuts" --run "$REC/p-base" \
        --run "$REC/p-nocuts" --run "$REC/p-verify" --run "$REC/p-fixed" --run "$REC/p-any" | tee "$RES/probe.txt"
}

# record.args: words every recording to train on or to judge by is made with (how often a way back is read again)
# shellcheck disable=SC2046
stage_heldout() { capture heldout plan-heldout.jsonl $(words record.args); }
# shellcheck disable=SC2046
stage_a() { capture a plan-a.jsonl $(words record.args); }
# shellcheck disable=SC2046
stage_b() { capture b plan-b.jsonl $(words record.args); }

check_rec() {
    "${PY[@]}" "$KIT/check_record.py" --gguf-py "$SRC/gguf-py" --model "$(model_file)" --rec "$REC/$1" |
        tee "$RES/check-$1.txt"
}

stage_check() {
    local name
    for name in heldout a b; do
        [ -f "$REC/$name/summary.json" ] || continue
        check_rec "$name"
    done
}
stage_check_a() { check_rec a; }
stage_check_b() { check_rec b; }

# the trainer's own checks on a made-up recording, on the processor: a change to the trainer is tried here before a
# card's hour is spent on it
stage_selftest() {
    "${PY[@]}" "$KIT/selftest.py" --gguf-py "$SRC/gguf-py" 2>&1 | tee "$RES/selftest.txt"
}

# a dozen steps of the trainer on the draft as served, from a recording already at hand: what a step takes and
# whether it fits a card, before a recording is made to train on
stage_trainsmoke() {
    "${PY[@]}" "$KIT/train_dspark.py" --gguf-py "$SRC/gguf-py" --draft "$(draft_file)" --target "$(model_file)" \
        --train "$REC/heldout" --eval "$REC/heldout" --out "$CACHE/trainsmoke-$MODE" --max-steps 12 --eval-requests 4 \
        --eval-every 6 --log-every 2 --lr 1e-5 2>&1 | tee "$LOGS/trainsmoke.log"
    tail -c 20000 "$LOGS/trainsmoke.log" > "$RES/trainsmoke.txt"
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tee -a "$RES/trainsmoke.txt"
    rm -f "$CACHE/trainsmoke-$MODE/draft-tuned.gguf"
}

# the sweep's trial the held-out requests accept most from becomes the tuned draft of the stages after it; with no
# trial above the draft as served, the job ends here
stage_pick() {
    "${PY[@]}" "$KIT/pick.py" "$CACHE" "$(sweep_dir)" | tee "$RES/pick.txt"
    tail -1 "$RES/pick.txt" > "$KIT/tuned.path"
}

# the engine's libraries and the tool, for a later box of the same cards
stage_pack_bin() {
    tar -C "$BLD/bin" -czf "$RES/engine-bin-$SHA.tar.gz" .
    sha256sum "$RES/engine-bin-$SHA.tar.gz"
}

stage_disk() { df -h "$CACHE" | tail -1; du -sh "$CACHE"/* 2> /dev/null | sort -h | tail -15; }

# A stage written try:NAME is run to learn from: when it fails, the stages after it still run, and the job fails at
# its end naming it. Any other stage that fails ends the job there.
failed=()
for stage in $(cat "$KIT/stages.txt"); do
    try=0
    case $stage in try:*) try=1; stage=${stage#try:} ;; esac
    say "== $stage"
    t0=$(date +%s)
    if [ "$try" = 1 ]; then
        set +e
        ( set -e; "stage_${stage//-/_}" )
        rc=$?
        set -e
        if [ "$rc" != 0 ]; then failed+=("$stage"); say "== $stage FAILED ($rc)"; fi
    else
        "stage_${stage//-/_}"
    fi
    say "== $stage done in $(( $(date +%s) - t0 ))s"
done
if [ "${#failed[@]}" != 0 ]; then say "failed: ${failed[*]}"; exit 1; fi
say "all stages done"
