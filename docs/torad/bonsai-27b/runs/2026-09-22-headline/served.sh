#!/usr/bin/env bash
# Served decode, PrismML prism-b10709-9a9394a vs torad-labs/llama.cpp 3520147, public pack, RTX 5070 Ti.
# Each leg: start llama-server with the runtime's flags, run bench-server.sh, stop it. Without MTP A-B-A-B;
# then the fork with its MTP draft head, twice. Prism b10709 refuses this pack's MTP head at load
# (server-prism-mtp.log: the head reads the Hadamard-latent token table without the inverse transform; the
# fork's a0af7ec restores it), so its MTP leg is recorded as refused, not measured.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd); bench=$here/../../bench-server.sh; tmpl=$here/../../chat-template.jinja
# PRISM_BIN, FORK_BIN: the two builds' bin dirs; MODEL: the public pack (README step 2); GPU: the card's index
# (0 by default; these runs used 1, an RTX 5070 Ti)
PR=${PRISM_BIN:?set PRISM_BIN to a prism-b10709-9a9394a build/bin}; FK=${FORK_BIN:?set FORK_BIN to a torad-labs/llama.cpp build/bin}
M=${MODEL:?set MODEL to Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf}
common=(-m "$M" --alias bonsai-2-27b -ngl 99 -fa on -c 65536 -np 1 --jinja --cache-type-k q4_0 --cache-type-v q4_0
        --chat-template-file "$tmpl" --reasoning-format deepseek --temp 1.0 --top-p 0.95 --top-k 20
        --host 127.0.0.1 --port 18090)
fork_only=(-cts q8_0 --attn-mask-bits)
mtp=(--spec-type draft-mtp --spec-draft-n-max 2 -ctkd q4_0 -ctvd q4_0)
leg() { # <label> <bin dir> <extra args...>
  local label=$1 bin=$2; shift 2
  CUDA_VISIBLE_DEVICES=${GPU:-0} "$bin/llama-server" "${common[@]}" "$@" > "$here/server-$label.log" 2>&1 &
  local pid=$!
  for _ in $(seq 1 120); do
    curl -sf http://127.0.0.1:18090/health > /dev/null && break
    kill -0 $pid 2> /dev/null || break
    sleep 2
  done
  if ! kill -0 $pid 2> /dev/null; then
    printf '%s\tserver refused to start: %s\n' "$label" "$(grep -m1 ' E ' "$here/server-$label.log")" | tee -a "$here/served.tsv"
    return
  fi
  curl -s http://127.0.0.1:18090/v1/models > "$here/models-$label.json"
  curl -s http://127.0.0.1:18090/props | jq '{n_ctx: .default_generation_settings.n_ctx, total_slots, build_info}' > "$here/props-$label.json"
  bash "$bench" 18090 "$label" 512 | tee -a "$here/served.tsv"
  kill $pid; wait $pid 2>/dev/null
}
: > "$here/served.tsv"
leg prism-nomtp-1 "$PR"
leg fork-nomtp-1  "$FK" "${fork_only[@]}"
leg prism-nomtp-2 "$PR"
leg fork-nomtp-2  "$FK" "${fork_only[@]}"
leg fork-mtp-1    "$FK" "${fork_only[@]}" "${mtp[@]}"
leg fork-mtp-2    "$FK" "${fork_only[@]}" "${mtp[@]}"
leg prism-mtp     "$PR" "${mtp[@]}"
