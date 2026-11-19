#!/usr/bin/env bash
# mkjob NAME MODE "STAGES" LIMIT_MINUTES [FILE...]: the kit folder of one job, hard-linked from kit/ as it stands,
# with its own mode, stages and time limit, and the files named after them (parts of a large input) linked beside.
# A job's folder is made anew each time, so kit/ itself never carries a job's parts or settings.
set -euo pipefail
P=$(cd "$(dirname "$0")" && pwd)
name=$1 mode=$2 stages=$3 limit=$4
shift 4
out=$P/jobs/$name
rm -rf "$out"
mkdir -p "$P/jobs"
cp -al "$P/kit" "$out"
# written as new files: a write through a hard link would change kit/ too
rm -f "$out/mode.txt" "$out/stages.txt" "$out/job.toml"
printf '%s\n' "$mode" > "$out/mode.txt"
printf '%s\n' "$stages" > "$out/stages.txt"
sed "s/^limit_minutes = .*/limit_minutes = $limit/" "$P/kit/job.toml" > "$out/job.toml"
for f in "$@"; do
    if [ -d "$f" ]; then
        cp -al "$f" "$out/parity/$(basename "$f")"
    else
        ln "$f" "$out/"
    fi
done
printf '%s: %s, mode %s, stages: %s, %s\n' "$name" "$(du -sh "$out" | cut -f1)" "$mode" "$stages" "$(grep limit "$out/job.toml")"
