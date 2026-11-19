"""The trial of a sweep that the held-out requests accept most from.

  pick.py CACHE SWEEP

Reads CACHE/SWEEP/*/summary.json, prints a line a trial, and last the best trial's tuned draft as a path under CACHE.
Ends with 1 when no trial is above the draft it started from.
"""
from __future__ import annotations

import glob
import json
import os
import sys

KEY = "accepted_3"
WALKED = "walked_3"   # what the server's rounds keep, when the trials walked them (--walk): the trials are picked by it


def main() -> None:
    cache, sweep = sys.argv[1], sys.argv[2]
    trials = []
    for path in sorted(glob.glob(os.path.join(cache, sweep, "*", "summary.json"))):
        with open(path, encoding="utf-8") as f:
            summary = json.load(f)
        name = os.path.basename(os.path.dirname(path))
        if "before" not in summary or "after" not in summary:
            print("%s: no evaluation after training" % name)
            continue
        before, after = summary["before"][KEY], summary["after"][KEY]
        walked = "walked_before" in summary and "walked_after" in summary
        if walked:
            print("%s: anchors evenly apart accept %.4f before, %.4f after; the server's rounds keep %.4f of a most of %.4f before, %.4f of %.4f after" % (
                name, before, after, summary["walked_before"][WALKED], summary["walked_before"]["walked_ceiling_3"],
                summary["walked_after"][WALKED], summary["walked_after"]["walked_ceiling_3"]))
            before, after = summary["walked_before"][WALKED], summary["walked_after"][WALKED]
        if not os.path.isfile(os.path.join(cache, sweep, name, "draft-tuned.gguf")):
            print("%s: no tuned draft" % name)
            continue
        print("%s: %.4f before, %.4f after, of a ceiling of %.4f" % (name, before, after, summary["after"]["ceiling_3"]))
        trials.append((after, before, name))
    if not trials:
        raise SystemExit("no trial of %s has a tuned draft and an evaluation" % sweep)
    after, before, name = max(trials)
    if after <= before:
        raise SystemExit("no trial is above the draft as served: the best, %s, has %.4f after %.4f" % (name, after, before))
    print("best: %s" % name)
    print("%s/%s/draft-tuned.gguf" % (sweep, name))


if __name__ == "__main__":
    main()
