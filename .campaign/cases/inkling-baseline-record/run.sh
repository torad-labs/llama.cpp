#!/bin/sh
# Cases for inkling-baseline-record: TORAD.md records the merge of upstream d3c3e242c
# the way the upstream v1.0.0 merge is recorded there, in an "Upstream ..." section of
# its own that names the upstream commit.
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$ROOT" || { echo "run.sh: cannot cd to $ROOT" >&2; exit 2; }

T=$(mktemp -d /tmp/inkling-baseline-record.XXXXXX)

fail() { printf '%s\n' "$*" > "$REASON"; exit 1; }

case_torad_md_names_upstream_commit() {
    [ -f TORAD.md ] || fail "TORAD.md is missing"
    grep -q 'd3c3e242c' TORAD.md || fail "TORAD.md does not name upstream commit d3c3e242c"
}

case_upstream_section_records_merge() {
    [ -f TORAD.md ] || fail "TORAD.md is missing"
    # A "## Upstream ..." (or "### Upstream ...") section other than the v1.0.0 one,
    # whose heading or body, up to the next heading of its level or above, names d3c3e242c.
    awk '
        /^#+ / {
            level = length($1); title = substr($0, level + 2)
            if (insec && level <= seclevel) insec = 0
            if (!insec && (level == 2 || level == 3) && title ~ /^Upstream/ && title != "Upstream v1.0.0") {
                insec = 1; seclevel = level
            }
        }
        insec && /d3c3e242c/ { found = 1 }
        END { exit found ? 0 : 1 }
    ' TORAD.md || fail "no Upstream section of its own in TORAD.md names d3c3e242c"
}

run_case() {
    name=$1; fn=$2
    REASON="$T/$name.reason"; : > "$REASON"
    ( set +e; "$fn" ) > "$T/$name.log" 2>&1
    rc=$?
    if [ "$rc" -eq 0 ]; then
        echo "PASS $name"
    else
        why=$(head -n 1 "$REASON" | tr -d '\r')
        [ -n "$why" ] || why="exited $rc: $(tail -n 1 "$T/$name.log" | tr -d '\r')"
        echo "FAIL $name: $why"
    fi
}

run_case torad-md-names-upstream-commit   case_torad_md_names_upstream_commit
run_case upstream-section-records-merge   case_upstream_section_records_merge

rm -rf "$T"
exit 0
