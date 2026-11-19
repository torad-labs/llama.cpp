"""Choose what the recorder reads, and where Inkling writes on its own.

Reads what `dspark_tool prepare` measured (each request's length in tokens, where its model turns start, and how far
it agrees with the donor request from the first token), and writes four plans for `dspark_tool capture`:

  plan-heldout.jsonl   sessions no training run ever reads
  plan-a.jsonl         the first part of the training recording, enough for a quick fine-tune
  plan-b.jsonl         the rest of it
  plan-smoke.jsonl     a couple of dozen small requests that walk every path of the recorder

Everything is chosen by a hash of the seed and the request's id, so the same inputs give the same plans.

Usage: make_plan.py --prepared prepared.jsonl --heldout requests-heldout.jsonl --train requests-train.jsonl --out DIR
"""
import argparse
import bisect
import collections
import hashlib
import json
import os

ROW_BYTES_REAL = 8 * 4608 * 2  # eight taps of the real model, bf16


def h64(*parts) -> int:
    m = hashlib.blake2b(digest_size=8)
    for p in parts:
        m.update(str(p).encode())
        m.update(b"\0")
    return int.from_bytes(m.digest(), "big")


def unit(*parts) -> float:
    return h64(*parts) / 2.0 ** 64


def source_class(source: str) -> str:
    s = source.lower()
    if "mimo" in s:
        return "mimo"
    if "glm53" in s:
        return "glm"
    return "long"


def read_split(path: str, split: str, into: dict) -> None:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            into[r["id"]] = {"split": split, "assistant_turns": int(r.get("assistant_turns", -1))}


def keep_points(lcps: list, min_keep: int, min_saving: int) -> tuple:
    """The two donor positions a slot keeps, chosen so the most tokens are read once instead of once a request."""
    ordered = sorted(lcps)
    n = len(ordered)
    values = sorted({k for k in ordered if k >= min_keep})
    reach = {k: n - bisect.bisect_left(ordered, k) for k in values}  # requests that agree with the donor that far

    one, one_k = 0, 0
    two, two_k = 0, (0, 0)
    for j, k2 in enumerate(values):
        s2 = k2 * reach[k2]
        if s2 > one:
            one, one_k = s2, k2
        for k1 in values[:j]:
            s = s2 + k1 * (reach[k1] - reach[k2])
            if s > two:
                two, two_k = s, (k1, k2)
    if one < min_saving:
        return 0, 0, 0, 0
    if two - one < min_saving:
        return 0, one_k, 0, one
    k1, k2 = two_k
    return k1, k2, k1 * (reach[k1] - reach[k2]), k2 * reach[k2]


def select(rows: list, budget: float, shares: dict, seed: int) -> dict:
    """Requests of each class in a fixed order, up to the class's part of the budget. A class too small to fill
    its part leaves the rest to the others."""
    pools = {c: sorted((r for r in rows if r["class"] == c), key=lambda r: (r["costly"], h64(seed, "order", r["id"])))
             for c in shares}
    want = {c: budget * shares[c] for c in shares}
    taken = {c: [] for c in shares}
    used = {c: 0 for c in shares}
    while True:
        for c in shares:
            pool = pools[c]
            while len(taken[c]) < len(pool) and used[c] < want[c]:
                r = pool[len(taken[c])]
                taken[c].append(r)
                used[c] += r["cost"]
        spent = [c for c in shares if len(taken[c]) == len(pools[c])]
        open_ = [c for c in shares if c not in spent]
        lacking = sum(max(0.0, want[c] - used[c]) for c in spent)
        if lacking < 1 or not open_:
            return taken
        for c in spent:
            want[c] = used[c]
        total = sum(shares[c] for c in open_)
        for c in open_:
            want[c] += lacking * shares[c] / total


def head_of(taken: dict, budget: float) -> tuple:
    """The start of each class's list, in the proportions the whole selection has."""
    whole = sum(r["cost"] for rs in taken.values() for r in rs)
    first, rest = [], []
    for rs in taken.values():
        part = budget * sum(r["cost"] for r in rs) / max(1, whole)
        used = 0
        for r in rs:
            if used < part:
                first.append(r)
                used += r["cost"]
            else:
                rest.append(r)
    return first, rest


def choose_cuts(r: dict, rate: float, long_share: float, g_short: int, g_long: int, seed: int) -> list:
    out = []
    for pos in r["cuts"]:
        if pos <= r["shared_len"] or unit(seed, "cut", r["id"], pos) >= rate:
            continue
        g_max = g_long if unit(seed, "length", r["id"], pos) < long_share else g_short
        out.append({"pos": pos, "g_max": g_max, "seed": h64(seed, "seed", r["id"], pos) & 0x7FFFFFFF})
    return out


def smoke_cuts(r: dict, g_short: int, g_long: int, seed: int) -> list:
    """The first model turn, one in the middle and the last, so the recorder goes back to a cut and reads on."""
    able = [pos for pos in r["cuts"] if pos > r["shared_len"]]
    picks = sorted({able[0], able[len(able) // 2], able[-1]})
    out = []
    for i, pos in enumerate(picks):
        out.append({"pos": pos, "g_max": g_long if (h64(seed, "smoke", r["id"]) + i) % 2 == 0 else g_short,
                    "seed": h64(seed, "seed", r["id"], pos) & 0x7FFFFFFF})
    return out


def write_plan(path: str, header: dict, entries: list, seed: int) -> dict:
    # requests that keep the most of the donor go first, so a slot does not read the donor's prefix again and again
    entries = sorted(entries, key=lambda e: (-e[0]["shared_len"], h64(seed, "plan", e[0]["id"])))
    stats = {"requests": len(entries), "rows": 0, "cuts": 0, "generated_at_most": 0,
             "by_class": collections.Counter(), "by_shared_len": collections.Counter(),
             "by_source": collections.Counter(), "by_g_max": collections.Counter(), "model_turns": 0}
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(header, separators=(",", ":")) + "\n")
        for r, cuts in entries:
            f.write(json.dumps({"type": "request", "id": r["id"], "file": r["file"], "n_tokens": r["n_tokens"],
                                "shared_len": r["shared_len"], "cuts": cuts}, separators=(",", ":")) + "\n")
            stats["rows"] += r["cost"]
            stats["cuts"] += len(cuts)
            stats["generated_at_most"] += sum(c["g_max"] for c in cuts)
            stats["by_class"][r["class"]] += r["cost"]
            stats["by_shared_len"][str(r["shared_len"])] += 1
            stats["by_source"][r["source"]] += 1
            stats["model_turns"] += sum(1 for pos in r["cuts"] if pos > r["shared_len"])
            for c in cuts:
                stats["by_g_max"][str(c["g_max"])] += 1
    for k in ("by_class", "by_shared_len", "by_source", "by_g_max"):
        stats[k] = dict(sorted(stats[k].items()))
    stats["gb_at_eight_taps"] = round((stats["rows"] + stats["generated_at_most"]) * ROW_BYTES_REAL / 1e9, 1)
    return stats


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prepared", required=True)
    p.add_argument("--heldout", required=True)
    p.add_argument("--train", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tokens", type=float, default=10e6, help="positions the training recording reads")
    p.add_argument("--tokens-a", type=float, default=1.5e6, help="positions of its first part")
    p.add_argument("--tokens-heldout", type=float, default=1e6)
    p.add_argument("--mix", default="long=0.55,mimo=0.15,glm=0.30")
    p.add_argument("--written", type=float, default=0.09,
                   help="what Inkling writes itself at most, as a share of the positions read")
    p.add_argument("--written-heldout", type=float, default=0.20)
    p.add_argument("--long-share", type=float, default=0.25, help="share of those continuations that run long")
    p.add_argument("--g-short", type=int, default=160)
    p.add_argument("--g-long", type=int, default=640)
    p.add_argument("--min-keep", type=int, default=512)
    p.add_argument("--min-saving", type=int, default=200000)
    p.add_argument("--smoke", default="12,6,6", help="smoke requests that keep the long prefix, the short one, none")
    p.add_argument("--seed", type=int, default=20261119)
    args = p.parse_args()

    shares = {}
    for item in args.mix.split(","):
        name, value = item.split("=")
        shares[name] = float(value)
    total_share = sum(shares.values())
    shares = {c: v / total_share for c, v in shares.items()}

    known = {}
    read_split(args.heldout, "heldout", known)
    read_split(args.train, "train", known)

    header = None
    rows = []
    with open(args.prepared, encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            if j["type"] == "header":
                header = j
            elif j["type"] == "request":
                j["file"] = "r%06d" % len(rows)
                rows.append(j)
    if header is None:
        raise SystemExit("%s has no header" % args.prepared)

    # what cannot be recorded, and why
    dropped = collections.Counter()
    turn_gap = collections.defaultdict(collections.Counter)
    valid = []
    for r in rows:
        info = known.get(r["id"])
        if info is None:
            dropped["in no requests file"] += 1
            continue
        r["split"] = info["split"]
        r["class"] = source_class(r["source"])
        if r["error"]:
            dropped["does not parse"] += 1
            continue
        if not r["cuts"] or r["n_tokens"] < 2:
            dropped["no model turn"] += 1
            continue
        if not r["truncated"] and info["assistant_turns"] >= 0:
            # one cut where each model turn starts, and one at the end, where the next turn would start
            gap = len(r["cuts"]) - (info["assistant_turns"] + 1)
            turn_gap[r["source"]]["more" if gap > 0 else "fewer" if gap < 0 else "equal"] += 1
            if gap > 0:
                dropped["more model turns than the session has"] += 1
                continue
        valid.append(r)
    if len(valid) < 0.9 * len(rows):
        print("dropped:", dict(dropped))
        print("model turns against the sessions' own count:", {s: dict(c) for s, c in turn_gap.items()})
        raise SystemExit("only %d of %d requests can be recorded" % (len(valid), len(rows)))

    k_tools, k_sys, saved_tools, saved_sys = keep_points([r["lcp"] for r in valid], args.min_keep, args.min_saving)
    for r in valid:
        r["shared_len"] = 0
        for k in (k_tools, k_sys):
            if 0 < k <= r["lcp"] and k < r["n_tokens"] and k < r["cuts"][-1]:
                r["shared_len"] = k
        r["cost"] = r["n_tokens"] - r["shared_len"]
        first = next((pos for pos in r["cuts"] if pos > r["shared_len"]), r["n_tokens"])
        # most of what it would record comes before its first model turn
        r["costly"] = 1 if (first - r["shared_len"]) > 0.5 * r["cost"] else 0

    plan_header = {"type": "header", "donor": header["donor"], "k_tools": k_tools, "k_sys": k_sys,
                   "max_tokens": header["max_tokens"]}

    os.makedirs(args.out, exist_ok=True)
    report = {"seed": args.seed, "donor": header["donor"], "donor_tokens": header["donor_tokens"],
              "k_tools": k_tools, "k_sys": k_sys, "saved_by_k_tools": saved_tools, "saved_by_k_sys": saved_sys,
              "requests": len(rows), "recordable": len(valid), "dropped": dict(dropped),
              "model_turns_against_sessions": {s: dict(c) for s, c in sorted(turn_gap.items())},
              "truncated": sum(1 for r in valid if r["truncated"]),
              "with_grammar": sum(1 for r in valid if r["grammar"]),
              "tokens_of_all": sum(r["n_tokens"] for r in valid),
              "rows_of_all": sum(r["cost"] for r in valid),
              "costly": sum(r["costly"] for r in valid), "plans": {}}

    lcp_counts = collections.Counter(r["lcp"] for r in valid)
    report["lcp_most_common"] = [[k, n] for k, n in lcp_counts.most_common(12)]
    per = collections.defaultdict(lambda: [0, 0, 0, collections.Counter()])
    for r in valid:
        a = per[(r["split"], r["source"], r["prefix"])]
        a[0] += 1
        a[1] += r["n_tokens"]
        a[2] += r["cost"]
        a[3][r["shared_len"]] += 1
    report["by_source"] = [{"split": k[0], "source": k[1], "prefix": k[2], "requests": a[0], "tokens": a[1],
                            "rows": a[2], "shared_len": {str(s): n for s, n in sorted(a[3].items())}}
                           for k, a in sorted(per.items())]

    def plan(name: str, chosen: list, written: float) -> None:
        # the share of model turns to cut at, from how much may be written and how long a continuation may run
        turns = sum(1 for r in chosen for pos in r["cuts"] if pos > r["shared_len"])
        mean = args.long_share * args.g_long + (1.0 - args.long_share) * args.g_short
        rate = min(1.0, written * sum(r["cost"] for r in chosen) / max(1.0, turns * mean))
        entries = [(r, choose_cuts(r, rate, args.long_share, args.g_short, args.g_long, args.seed)) for r in chosen]
        report["plans"][name] = write_plan(os.path.join(args.out, "plan-%s.jsonl" % name), plan_header, entries, args.seed)

    heldout = [r for r in valid if r["split"] == "heldout"]
    train = [r for r in valid if r["split"] == "train"]

    taken = select(heldout, args.tokens_heldout, shares, args.seed)
    plan("heldout", [r for rs in taken.values() for r in rs], args.written_heldout)

    taken = select(train, args.tokens, shares, args.seed)
    first, rest = head_of(taken, args.tokens_a)
    plan("a", first, args.written)
    plan("b", rest, args.written)

    # the smoke: the smallest requests of each kind that have a model turn to return to
    counts = [int(v) for v in args.smoke.split(",")]
    kinds = [k_sys, k_tools, 0]
    smoke, seen = [], set()
    for kind, count in zip(kinds, counts):
        if kind in seen:
            continue
        seen.add(kind)
        able = [r for r in valid if r["shared_len"] == kind and sum(1 for pos in r["cuts"] if pos > kind) >= 3]
        able.sort(key=lambda r: (r["cost"], r["id"]))
        smoke.extend(able[:count])
    entries = [(r, smoke_cuts(r, args.g_short, args.g_long, args.seed)) for r in smoke]
    report["plans"]["smoke"] = write_plan(os.path.join(args.out, "plan-smoke.jsonl"), plan_header, entries, args.seed)

    with open(os.path.join(args.out, "plan-stats.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
        f.write("\n")

    print("donor %s, %d tokens; a slot keeps its first %d and %d tokens (read once instead of %d and %d times over)" % (
        header["donor"], header["donor_tokens"], k_tools, k_sys,
        saved_tools // max(1, k_tools), saved_sys // max(1, k_sys)))
    print("%d requests, %d recordable, %d cut short at %d tokens, %d with a grammar; dropped: %s" % (
        len(rows), len(valid), report["truncated"], header["max_tokens"], report["with_grammar"], dict(dropped)))
    print("model turns against the sessions' own count:", report["model_turns_against_sessions"])
    print("agreement with the donor, most common lengths:", report["lcp_most_common"])
    for item in report["by_source"]:
        print("  %-8s %-58s %-12s %5d requests %9d tokens %9d rows  %s" % (
            item["split"], item["source"], item["prefix"], item["requests"], item["tokens"], item["rows"],
            item["shared_len"]))
    for name, s in report["plans"].items():
        print("plan %-8s %5d requests %9d rows %5d cuts of %d model turns, at most %7d written, about %6.1f GB; "
              "classes %s; kept prefix %s; lengths %s" % (
                  name, s["requests"], s["rows"], s["cuts"], s["model_turns"], s["generated_at_most"],
                  s["gb_at_eight_taps"], s["by_class"], s["by_shared_len"], s["by_g_max"]))


if __name__ == "__main__":
    main()
