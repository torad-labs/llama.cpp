#!/usr/bin/env bash
# upload-parts KNOWN_HOSTS HOST PORT REMOTE_DIR PARALLEL FILE...: each file to the box's folder under its own name,
# through vast's ssh proxy with rig's pinned host key. One rsync per file, resumed where it stopped
# (--append-verify re-reads what is already there), retried until it lands, PARALLEL files at a time.
# One long transfer through that proxy was cut after 4.5 minutes on 2026-11-19, so no file here is larger than 1 GiB.
set -uo pipefail
kh=$1 host=$2 port=$3 dir=$4 par=$5
shift 5
export UP_SSH="ssh -T -o BatchMode=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=$kh -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -p $port"
export UP_DEST="root@$host:$dir/"
one() {
    local f=$1 try
    for try in 1 2 3 4 5 6 7 8; do
        if rsync --append-verify -e "$UP_SSH" "$f" "$UP_DEST" 2>/dev/null; then
            printf '%s ok %s (try %s)\n' "$(date -u +%T)" "$(basename "$f")" "$try"
            return 0
        fi
        printf '%s retry %s (try %s failed)\n' "$(date -u +%T)" "$(basename "$f")" "$try"
        python3 -I -c "import time; time.sleep(5)"
    done
    printf '%s FAILED %s\n' "$(date -u +%T)" "$(basename "$f")"
    return 1
}
export -f one
printf '%s\0' "$@" | xargs -0 -P "$par" -n 1 bash -c 'one "$1"' _
