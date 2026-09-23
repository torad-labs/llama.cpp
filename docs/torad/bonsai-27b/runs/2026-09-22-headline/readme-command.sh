#!/usr/bin/env bash
# The README's step-3 command, verbatim, from a clone's root: does it load on this card, what pool and
# slots does it report, how much VRAM does it hold, and does one streamed tool call come back whole.
# ./readme-command.sh <llama.cpp clone> <models dir> [port]     # needs curl and jq; CUDA_VISIBLE_DEVICES picks the card
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
clone=${1:?usage: readme-command.sh <llama.cpp clone> <models dir> [port]}; models=${2:?}; port=${3:-8080}
cmd=$(sed -n '/^### 3. Serve/,/^### 4/p' "$here/../../README.md" | sed -n '/^```bash$/,/^```$/p' | sed '1d;$d' | sed 's/\\$//' | tr '\n' ' ')
cmd=${cmd//models\//$models/}; cmd=${cmd//--port 8080/--port $port}
cd "$clone"
echo "$cmd" > "$here/readme-command.cmd"
bash -c "exec $cmd" > "$here/readme-command.log" 2>&1 &
pid=$!
for _ in $(seq 1 180); do
  curl -sf "http://127.0.0.1:$port/health" > /dev/null && break
  kill -0 $pid 2> /dev/null || break
  sleep 2
done
if ! kill -0 $pid 2> /dev/null; then
  echo "refused to start: $(grep -m1 ' E ' "$here/readme-command.log")" | tee "$here/readme-command.txt"; exit 1
fi
{
  echo "models: $(curl -s "http://127.0.0.1:$port/v1/models" | jq -c '[.data[].id]')"
  echo "props: $(curl -s "http://127.0.0.1:$port/props" | jq -c '{n_ctx: .default_generation_settings.n_ctx, total_slots}')"
  echo "vram: $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep "^$pid," | cut -d, -f2)"
  jq -n '{model: "bonsai-2-27b", stream: true, stream_options: {include_usage: true}, max_tokens: 256,
      messages: [{role: "user", content: "What is the weather in Lisbon right now? Use the tool."}],
      tools: [{type: "function", function: {name: "get_weather", description: "Current weather for a city",
        parameters: {type: "object", properties: {city: {type: "string"}}, required: ["city"]}}}],
      chat_template_kwargs: {enable_thinking: false}}' |
    curl -sN "http://127.0.0.1:$port/v1/chat/completions" -H 'Content-Type: application/json' -d @- |
    sed -n 's/^data: //p' | grep -v '^\[DONE\]' |
    jq -rs '"tool call: " + ([.[].choices[0]?.delta.tool_calls[0]?.function.name // empty] | join("")) +
            " args " + ([.[].choices[0]?.delta.tool_calls[0]?.function.arguments // empty] | join("")) +
            ", finish " + ([.[].choices[0]?.finish_reason // empty] | join(",")) +
            ", usage " + ([.[] | select(.usage) | .usage | "\(.prompt_tokens)+\(.completion_tokens)"] | join(","))'
} | tee "$here/readme-command.txt"
kill $pid; wait $pid 2> /dev/null
