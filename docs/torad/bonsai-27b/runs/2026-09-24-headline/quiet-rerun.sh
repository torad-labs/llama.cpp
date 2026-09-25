#!/usr/bin/env bash
# matrix.sh and d0-decode.sh, run only once the machine is quiet: the 1-minute load under 6 for 120 s
# straight (a shared 16-core host whose daytime load of 20-50 put ±8-12 tok/s on the decode rows,
# 2026-09-24), then under the gate card's lock, the load logged every 10 s through the run.
#   quiet-rerun.sh [deadline hours, default 16]
set -uo pipefail
cd "$(dirname "$0")" || exit 1
trap 'echo "terminated by SIGTERM at $(date +%T)"; exit 143' TERM
trap 'echo "terminated by SIGHUP at $(date +%T)"; exit 129' HUP
# PRISM_BIN, FORK_BIN, MODEL, GPU as matrix.sh takes them; this run: GPU=1 (an RTX 5070 Ti), FORK_BIN the
# published engine-c008fe8 build (rig's prebuilt), PRISM_BIN a prism-b10709-9a9394a build with README step 1's flags
export CUDA_DEVICE_ORDER=PCI_BUS_ID PRISM_BIN FORK_BIN MODEL GPU
deadline=$((SECONDS + ${1:-16} * 3600)) quiet=0
while ((quiet < 120)); do
  ((SECONDS < deadline)) || { echo "no quiet window before the deadline"; exit 1; }
  if awk -v l="$(cut -d' ' -f1 /proc/loadavg)" 'BEGIN { exit !(l < 6) }'; then quiet=$((quiet + 10)); else quiet=0; fi
  sleep 10
done
echo "quiet at $(date +%T), load $(cut -d' ' -f1-3 /proc/loadavg)"
(while :; do echo "$(date +%T) $(cut -d' ' -f1 /proc/loadavg)"; sleep 10; done) > quiet-rerun-load.log &
mon=$!
flock /run/user/1000/rig-gatecard.lock bash -c 'for s in matrix.sh d0-decode.sh; do
  echo "== $s $(date +%T)"; bash $s > $s.out 2>&1; echo "   exit $? $(date +%T)"; done'
kill $mon
echo "load during the rerun: max $(awk '{print $2}' quiet-rerun-load.log | sort -g | tail -1), samples $(wc -l < quiet-rerun-load.log)"
cat d0-decode.tsv
