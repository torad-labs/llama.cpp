#!/usr/bin/env python3
"""Compare full next-token scores at identical teacher-forced inputs.

Each arm stores rows.json (build_sha256, model_sha256, n_vocab, and rows)
and logits.f32le (one dense little-endian F32 vector per row, in manifest
order). Rows specify client,
token_position, selected_id, and condition. The capture defines condition:
prompt_sha256, forced_tokens, query_position, batch_sha256, kv_mask_sha256,
and suppressed. Record the actual launch grid separately: a CTA permutation
does not change the conditional input. No server is started by this tool.
"""

import argparse
import array
from contextlib import ExitStack
import heapq
import json
import math
from pathlib import Path
import sys

ARMS = ("stock_a", "stock_b", "attention", "greedy")
PAIRS = (
    ("stock_a", "stock_b"),
    ("attention", "stock_a"), ("attention", "stock_b"),
    ("greedy", "stock_a"), ("greedy", "stock_b"),
)
CONDITION = {"prompt_sha256", "forced_tokens", "query_position", "batch_sha256", "kv_mask_sha256", "suppressed"}


def read_manifest(path):
    with (path / "rows.json").open() as f:
        m = json.load(f)
    rows = {(row["client"], row["token_position"]): (i, row) for i, row in enumerate(m["rows"])}
    if not rows or len(rows) != len(m["rows"]) or m["n_vocab"] < 2:
        raise ValueError(f"{path}: expected distinct scored rows and a full vocabulary")
    if (path / "logits.f32le").stat().st_size != 4 * m["n_vocab"] * len(rows):
        raise ValueError(f"{path}: logit file size does not match its manifest")
    return (m["build_sha256"], m["model_sha256"]), m["n_vocab"], rows


def row_scores(f, index, n_vocab, allowed):
    f.seek(4 * n_vocab * index)
    scores = array.array("f")
    scores.fromfile(f, n_vocab)
    if sys.byteorder == "big":
        scores.byteswap()
    if any(not math.isfinite(scores[i]) for i in allowed):
        raise ValueError("cannot normalize nonfinite scores for eligible tokens")
    top, runner_up = heapq.nlargest(2, allowed, key=lambda i: (scores[i], -i))
    logits = [scores[i] - scores[top] for i in allowed]
    log_z = math.log(math.fsum(math.exp(x) for x in logits))
    return top, scores[top] - scores[runner_up], [x - log_z for x in logits]


def check_selected(path):
    """Rows whose recorded token is the one a strict greedy scan of the recorded scores picks.

    The scan starts at token 0 and moves only to a strictly larger score, after the excluded tokens drop to
    minus infinity: the same pick the sampler's greedy step makes. Returns the matches, the rows, and the
    (client, position) of each mismatch."""
    _, n_vocab, rows = read_manifest(path)
    bad = []
    with (path / "logits.f32le").open("rb") as f:
        for (client, position), (index, row) in sorted(rows.items()):
            f.seek(4 * n_vocab * index)
            scores = array.array("f")
            scores.fromfile(f, n_vocab)
            if sys.byteorder == "big":
                scores.byteswap()
            for token in row["condition"]["suppressed"]:
                if 0 <= token < n_vocab:
                    scores[token] = -math.inf
            if max(range(n_vocab), key=scores.__getitem__) != row["selected_id"]:
                bad.append((client, position))
    return len(rows) - len(bad), len(rows), bad


def kl(log_a, log_b):
    return max(0.0, math.fsum(math.exp(a) * (a - b) for a, b in zip(log_a, log_b)))


def compare(paths):
    arms = ARMS + (("stack",) if paths.get("stack") is not None else ())
    pairs = PAIRS + ((("stack", "stock_a"), ("stack", "stock_b")) if "stack" in arms else ())
    manifests = {arm: read_manifest(paths[arm]) for arm in arms}
    provenance, n_vocab, stock_rows = manifests["stock_a"]
    if any(ids != provenance or n != n_vocab or rows.keys() != stock_rows.keys() for ids, n, rows in manifests.values()):
        raise ValueError("the arms do not use the same build, model, vocabulary, and scored positions")

    report = []
    with ExitStack() as stack:
        files = {arm: stack.enter_context((paths[arm] / "logits.f32le").open("rb")) for arm in arms}
        for key in sorted(stock_rows):
            client, position = key
            condition = stock_rows[key][1]["condition"]
            rows = {arm: manifests[arm][2][key] for arm in arms}
            if not CONDITION.issubset(condition) or any(row["condition"] != condition for _, row in rows.values()):
                raise ValueError(f"client {client}, token {position}: conditional input does not match")
            suppressed = set(condition["suppressed"])
            allowed = [i for i in range(n_vocab) if i not in suppressed]
            if len(allowed) < 2:
                raise ValueError(f"client {client}, token {position}: expected at least two eligible tokens")
            values = {arm: row_scores(files[arm], rows[arm][0], n_vocab, allowed) for arm in arms}
            entry = {"client": client, "token_position": position}
            for a, b in pairs:
                na, ga, la = values[a]
                nb, gb, lb = values[b]
                entry[f"{a}/{b}"] = {
                    "kl_ab": kl(la, lb), "kl_ba": kl(lb, la),
                    "top_a": na, "top_b": nb, "margin_a": ga, "margin_b": gb,
                    "selected_a": rows[a][1]["selected_id"], "selected_b": rows[b][1]["selected_id"],
                }
            report.append(entry)
    return {"build_sha256": provenance[0], "model_sha256": provenance[1], "n_vocab": n_vocab, "rows": report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in ARMS:
        parser.add_argument(arm, type=Path)
    parser.add_argument("--stack", type=Path, help="optional capture after standalone qualification")
    try:
        print(json.dumps(compare(vars(parser.parse_args())), indent=2, allow_nan=False))
    except (OSError, EOFError, ValueError, KeyError) as exc:
        parser.exit(1, f"replay comparison: {exc}\n")


if __name__ == "__main__":
    main()
