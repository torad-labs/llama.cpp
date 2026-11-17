#!/bin/sh
# Cases for inkling-server-parity-chat: through this fork's llama-server and its chat tests, Inkling 2
# handles text turns and system prompts, every reasoning effort level (an unknown one refused as a
# client error), tool declarations, tool calls with nested arguments and their results, thinking
# content kept across turns, and images; sampling never emits an id at or above the unpadded
# vocabulary, and generation ends at the model end token. Server-level cases carry inkling in their
# test names (tools/server/tests, pytest -k inkling).
# The cases here also put the synthetic inkling2 model that test-llama-archs -o writes behind the
# server, with the official template (models/templates/thinkingmachines-Inkling-2-Small.jinja), and
# check what a client gets: the prompt the server renders (/apply-template), the status of a chat
# request, and the ids the server samples.
# GPU runs use card 1 only, under /run/user/1000/gpu1.lock; card 0 serves another seat.
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$ROOT" || { echo "run.sh: cannot cd to $ROOT" >&2; exit 2; }

U=d3c3e242c16b8cbf51400fad410cd69b22d64f35
B=build-campaign
LOCK=/run/user/1000/gpu1.lock
VENV=/tmp/inkling-server-tests-venv
TEMPLATE=models/templates/thinkingmachines-Inkling-2-Small.jinja
T=$(mktemp -d /tmp/inkling-server-parity-chat.XXXXXX)
SERVER_PID=

fail() { printf '%s\n' "$*" > "$REASON"; exit 1; }

require_merged() {
    git cat-file -e "$U^{commit}" 2>/dev/null || fail "upstream commit $U is not in this repository"
    git merge-base --is-ancestor "$U" HEAD || fail "upstream $U is not merged into HEAD"
}

build_targets() {
    require_merged
    if [ ! -f "$T/built" ]; then
        cmake -S . -B "$B" -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120a \
            -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_EXAMPLES=OFF \
            -DGGML_CCACHE=OFF > "$T/configure.log" 2>&1 \
            || fail "cmake configure with GGML_CUDA=ON failed: $(tail -n 1 "$T/configure.log")"
        cmake --build "$B" -j "$(nproc)" --target llama-server test-llama-archs > "$T/build.log" 2>&1 \
            || fail "building llama-server and test-llama-archs failed: $(grep -m1 -E 'error' "$T/build.log" || tail -n 1 "$T/build.log")"
        : > "$T/built"
    fi
}

gpu1() { flock "$LOCK" env CUDA_VISIBLE_DEVICES=1 "$@"; }

venv() {
    if [ ! -x "$VENV/bin/pytest" ]; then
        { python3 -m venv "$VENV" && "$VENV/bin/pip" install -q -r tools/server/tests/requirements.txt; } > "$T/venv.log" 2>&1 \
            || fail "could not make the server test venv at $VENV: $(tail -n 1 "$T/venv.log")"
    fi
}

case_server_tests_collected() {
    require_merged
    venv
    ( cd tools/server/tests && "$VENV/bin/pytest" unit -k inkling --collect-only -q ) > "$T/collect.out" 2>&1
    n=$(grep -c '::' "$T/collect.out")
    [ "$n" -gt 0 ] || fail "no server test in tools/server/tests/unit has inkling in its name: $(tail -n 1 "$T/collect.out")"
}

case_server_tests_pass() {
    build_targets
    venv
    ( cd tools/server/tests && LLAMA_SERVER_BIN_PATH="$ROOT/$B/bin/llama-server" \
        flock "$LOCK" env CUDA_VISIBLE_DEVICES=1 "$VENV/bin/pytest" unit -k inkling -q -rs ) > "$T/pytest.out" 2>&1
    rc=$?
    summary=$(grep -E '(passed|failed|error|skipped|no tests ran|deselected)' "$T/pytest.out" | tail -n 1)
    [ "$rc" -eq 0 ] || fail "pytest unit -k inkling exited $rc: ${summary:-$(tail -n 1 "$T/pytest.out")}"
    echo "$summary" | grep -q -E '[1-9][0-9]* passed' || fail "pytest unit -k inkling passed no test: $summary"
    ! echo "$summary" | grep -q -E 'skipped' || fail "inkling server tests were skipped, not run: $summary"
}

cat > "$T/chat.py" <<'PY'
import json, sys, urllib.request, urllib.error

BASE = sys.argv[2]
SYSTEM = "You answer briefly."

def die(msg):
    print(msg)
    sys.exit(1)

def post(path, body, timeout=120):
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

def short(body):
    s = json.dumps(body) if not isinstance(body, str) else body
    return s[:300]

def render(body):
    status, out = post("/apply-template", body)
    if status != 200 or not isinstance(out, dict) or not isinstance(out.get("prompt"), str):
        die(f"/apply-template answered {status}: {short(out)}")
    return out["prompt"]

def reply(body):
    # tool_choice none where tools are declared: the synthetic model has no end token, so a reply held to
    # a tool-call grammar could not end; the rendered prompts show the declarations and the history
    body = dict(body, max_tokens=4, temperature=1.0, seed=7)
    status, out = post("/v1/chat/completions", body)
    if status != 200:
        die(f"/v1/chat/completions answered {status}: {short(out)}")
    try:
        msg = out["choices"][0]["message"]
    except Exception:
        die(f"/v1/chat/completions gave no choice with a message: {short(out)}")
    if not isinstance(msg, dict) or msg.get("role") != "assistant":
        die(f"the reply is not an assistant message: {short(msg)}")

def contains(prompt, *parts):
    pos = 0
    for p in parts:
        i = prompt.find(p, pos)
        if i < 0:
            die(f"the rendered prompt lacks {p!r} (in order) : {prompt[:400]!r}")
        pos = i + len(p)

TURNS = [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": "Name a colour."},
    {"role": "assistant", "content": "Blue."},
    {"role": "user", "content": "Another one."},
]

TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Weather for a place",
    "parameters": {"type": "object", "properties": {
        "location": {"type": "object", "properties": {"city": {"type": "string"}, "country": {"type": "string"}}},
        "units": {"type": "array", "items": {"type": "string"}}},
        "required": ["location"]}}}]

NESTED = {"location": {"city": "Paris", "country": "FR"}, "units": ["celsius"]}
TOOL_HISTORY = [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": "Weather in Paris?"},
    {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function",
        "function": {"name": "get_weather", "arguments": json.dumps(NESTED)}}]},
    {"role": "tool", "tool_call_id": "call_1", "content": "21 C and sunny"},
    {"role": "user", "content": "Thanks. Tomorrow?"},
]

THINKING_HISTORY = [
    {"role": "user", "content": "First question."},
    {"role": "assistant", "reasoning_content": "PLAN-ALPHA think first", "content": "First answer."},
    {"role": "user", "content": "Second question."},
    {"role": "assistant", "reasoning_content": "PLAN-BETA think again", "content": "Second answer."},
    {"role": "user", "content": "Third question."},
]

EFFORT = {"none": "0", "minimal": "0.1", "low": "0.2", "medium": "0.7", "high": "0.9", "xhigh": "0.99", "max": "0.99"}

def main():
    case = sys.argv[1]
    if case == "text_render":
        p = render({"messages": TURNS})
        contains(p, "<|message_system|><|content_text|>" + SYSTEM + "<|end_message|>",
                 "<|message_user|><|content_text|>Name a colour.<|end_message|>",
                 "<|message_model|><|content_text|>Blue.<|end_message|>",
                 "<|message_user|><|content_text|>Another one.<|end_message|>")
    elif case == "text_reply":
        reply({"messages": TURNS})
    elif case == "effort_invalid":
        status, out = post("/v1/chat/completions", {"messages": TURNS, "reasoning_effort": "turbo", "max_tokens": 4})
        if not 400 <= status < 500:
            die(f"an unknown reasoning_effort answered {status}, not a client error: {short(out)}")
    elif case.startswith("effort_"):
        level = case[len("effort_"):]
        p = render({"messages": TURNS, "reasoning_effort": level})
        contains(p, "Thinking effort level: " + EFFORT[level] + "<|end_message|>")
        reply({"messages": TURNS, "reasoning_effort": level})
    elif case == "tools_render":
        p = render({"messages": TURNS, "tools": TOOLS})
        contains(p, "<|message_system|>tool_declare<|content_xml|>", '"name":"get_weather"', "<|end_message|>")
    elif case == "tools_reply":
        reply({"messages": TURNS, "tools": TOOLS, "tool_choice": "none"})
    elif case == "tool_call_render":
        p = render({"messages": TOOL_HISTORY, "tools": TOOLS})
        contains(p, '<|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":'
                    '{"location":{"city":"Paris","country":"FR"},"units":["celsius"]}}<|end_message|>')
    elif case == "tool_result_render":
        p = render({"messages": TOOL_HISTORY, "tools": TOOLS})
        contains(p, "<|content_invoke_tool_json|>", "<|message_tool|>get_weather<|content_text|>21 C and sunny<|end_message|>",
                 "<|message_user|><|content_text|>Thanks. Tomorrow?<|end_message|>")
    elif case == "tool_history_reply":
        reply({"messages": TOOL_HISTORY, "tools": TOOLS, "tool_choice": "none"})
    elif case == "thinking_render":
        p = render({"messages": THINKING_HISTORY})
        contains(p, "<|message_model|><|content_thinking|>PLAN-ALPHA think first<|end_message|>",
                 "<|message_model|><|content_text|>First answer.<|end_message|>",
                 "<|message_model|><|content_thinking|>PLAN-BETA think again<|end_message|>",
                 "<|message_user|><|content_text|>Third question.<|end_message|>")
    elif case == "thinking_reply":
        reply({"messages": THINKING_HISTORY})
    elif case == "vocab":
        limit = int(sys.argv[3])
        seen = 0
        for seed in range(1, 9):
            status, out = post("/completion", {"prompt": "Count with me: one two three", "n_predict": 48, "seed": seed,
                                               "temperature": 1.0, "top_k": 0, "top_p": 1.0, "min_p": 0.0,
                                               "ignore_eos": True, "return_tokens": True, "n_probs": 8,
                                               "cache_prompt": False})
            if status != 200 or not isinstance(out, dict):
                die(f"/completion answered {status}: {short(out)}")
            ids = list(out.get("tokens") or [])
            for step in out.get("completion_probabilities") or []:
                if "id" in step:
                    ids.append(step["id"])
                for alt in step.get("top_logprobs") or step.get("top_probs") or []:
                    if "id" in alt:
                        ids.append(alt["id"])
            bad = sorted({i for i in ids if not 0 <= i < limit})
            if bad:
                die(f"seed {seed}: the server emitted ids {bad[:8]} at or above the unpadded vocabulary of {limit}")
            seen += len(out.get("tokens") or [])
        if seen == 0:
            die("the server returned no sampled tokens to check")
    else:
        die("unknown case " + case)

main()
PY

# The synthetic model behind one server on card 1, started once for the cases below it
setup_server() {
    build_targets
    mkdir -p "$T/models"
    env CUDA_VISIBLE_DEVICES=1 "./$B/bin/test-llama-archs" -a inkling2 -o "$T/models" > "$T/save.out" 2>&1 \
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
    CUDA_VISIBLE_DEVICES=1 "./$B/bin/llama-server" -m "$(cat "$T/model.path")" --jinja --chat-template-file "$TEMPLATE" \
        -ngl 99 -c 4096 -np 2 --host 127.0.0.1 --port "$port" > "$T/server.log" 2>&1 &
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
    msg=$(python3 "$T/chat.py" "$1" "$(cat "$T/base")" "$(cat "$T/vocab")" 2>&1) || fail "${msg:-the request crashed}"
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

# the server tests take card 1's lock themselves, so they run before this script's server holds it
run_case inkling-server-tests-collected                          case_server_tests_collected
run_case inkling-server-tests-pass-on-card-1                     case_server_tests_pass

start_server
run_case inkling-chat-text-turns-and-system-prompt-rendered      served text_render
run_case inkling-chat-text-turns-get-a-reply                     served text_reply
for level in none minimal low medium high xhigh max; do
    run_case "inkling-chat-reasoning-effort-$level"              served "effort_$level"
done
run_case inkling-chat-unknown-reasoning-effort-is-client-error   served effort_invalid
run_case inkling-chat-tool-declarations-rendered                 served tools_render
run_case inkling-chat-tool-declarations-get-a-reply              served tools_reply
run_case inkling-chat-tool-call-nested-arguments-rendered        served tool_call_render
run_case inkling-chat-tool-result-rendered                       served tool_result_render
run_case inkling-chat-tool-call-history-gets-a-reply             served tool_history_reply
run_case inkling-chat-thinking-kept-across-turns                 served thinking_render
run_case inkling-chat-thinking-history-gets-a-reply              served thinking_reply
run_case inkling-sampling-stays-below-unpadded-vocabulary        served vocab
stop_server

exit 0
