#!/usr/bin/env bash
# Per-prompt decode speed of a running llama-server, read from the server's own timings: greedy, prompt
# cache off, one warm-up request discarded, then four prompts with thinking off and two with it on.
# With a draft head loaded (--spec-type draft-mtp) the draft counters are printed beside the speed.
#
#   ./bench-server.sh <port> <label> [max_tokens]     # needs curl and jq
set -euo pipefail

port=${1:?usage: bench-server.sh <port> <label> [max_tokens]}
label=${2:?usage: bench-server.sh <port> <label> [max_tokens]}
max_tokens=${3:-512}

declare -A prompts=(
  [code]="Write a Python module implementing an LRU cache with TTL expiry, thread-safe, with docstrings and a small test at the bottom."
  [sql]="Design a PostgreSQL schema for a multi-tenant invoicing system (tenants, customers, invoices, line items, payments) and write the DDL with indexes and constraints, then three analytical queries."
  [prose]="Write a reflective essay on why cities feel different at night, with concrete sensory detail and no lists."
  [reasoning]="A train leaves city A at 9:00 going 80 km/h; another leaves city B, 300 km away, at 9:30 going 100 km/h toward A. Work out exactly when and where they meet, then generalize to a formula and check it against the numbers."
)

run() { # <name> <thinking: true|false>
  jq -n --arg p "${prompts[$1]}" --argjson n "$max_tokens" --argjson think "$2" '{
      messages: [{role: "user", content: $p}], max_tokens: $n,
      temperature: 0, top_k: 1, seed: 1, cache_prompt: false,
      chat_template_kwargs: {enable_thinking: $think}}' |
    curl -sf --max-time 1800 "http://127.0.0.1:$port/v1/chat/completions" \
      -H 'Content-Type: application/json' -d @-
}

row() { # <name> <thinking>
  run "$1" "$2" | jq -r --arg label "$label" --arg name "$1" --arg think "$2" '.timings |
      [$label, $name, ("think=" + (if $think == "true" then "1" else "0" end)),
       (.predicted_n | tostring) + " tok", ((.predicted_per_second * 10 | round) / 10 | tostring) + " tok/s",
       (if (.draft_n // 0) == 0 then "draft -"
        else "draft \(.draft_n_accepted)/\(.draft_n) (\((.draft_n_accepted / .draft_n * 100 | round))%)" end)]
      | @tsv'
}

run code false > /dev/null # warm-up, discarded
for name in code sql prose reasoning; do row "$name" false; done
for name in code reasoning; do row "$name" true; done
