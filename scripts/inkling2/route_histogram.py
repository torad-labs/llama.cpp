#!/usr/bin/env python3
"""Route-hit histogram for Inkling 2 Small from stored MMQ_ROUTE serving traces.

Ledger row inkling-route-histogram. Answers whether GLM-style hot-expert per-card
replication transfers to Inkling 2 Small serving, using ONLY what is already on disk:
CPU-side replay over stored route records. No GPU window runs; the RTX PRO 6000 box
stays fenced.

What exists on disk (found 2026-11-22): five stores of `MMQ_ROUTE_*` lines written by
the engine hook `GGML_CUDA_MMQ_ROUTE_LOG=N` during the 2026-11-18 tp4 four-card window
(producer: ggml_cuda_mmq_route_log::record — site/eval/cols/used/tiles8/tiles/j only).
Each record carries the launch's routed columns (`cols` = tokens x top_k slots) and how
many DISTINCT experts hold at least one column (`used`). NO expert identities exist in
any store, so the falsifier's exact top-1 share cannot be measured from these bytes;
what they measure for real is breadth concentration per layer vs the uniform
expectation E[distinct] = n_experts * (1 - (1 - top_k/n_experts)^B), B = cols/top_k,
exact under uniform within-token-distinct routing over independent tokens.

Modes:
  build (default)   parse --stores, aggregate, write --out JSON artifact.
  --check [PATH]    validate the artifact's internal consistency only (no stores):
                    totals equal rows*slots*topk relations, phase shares sum to 1,
                    stored uniform expectations re-derive from model params, verdict
                    agrees with what it claims was measurable.

The five default stores live outside this repo on the local rig research tree; pass
--stores explicitly when they move. Records repeat once per device context with
identical values and once per weight product of the same routing; dedup keeps one
launch event per (store, eval, layer) — first occurrence wins — and counts the rest as
product repeats.
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time

DEFAULT_STORES = [
    "/home/marcos/Documents/dev/infra/rig/local-nvme/research/inkling2-tp4-20261118/stock-glm-pool-engine/expert-routes.jsonl",
    "/home/marcos/Documents/dev/infra/rig/local-nvme/research/inkling2-tp4-20261118/stock-default-pool-sliced/expert-routes.jsonl",
    "/home/marcos/Documents/dev/infra/rig/local-nvme/research/inkling2-tp4-20261118/stock-prequalification-engine/expert-routes.jsonl",
    "/home/marcos/Documents/dev/infra/rig/local-nvme/research/inkling2-tp4-20261118/stock-full-capacity-host/expert-routes.jsonl",
    "/home/marcos/Documents/dev/infra/rig/local-nvme/research/inkling2-tp4-20261118/results/stock-shared-pool-engine/expert-routes.jsonl",
]
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
DEFAULT_OUT = os.path.join(REPO_ROOT, "reports", "inkling2-route-histogram.json")

N_EXPERTS = 384
TOP_K = 8            # routed slots per token per layer
ROUTED_LAYERS = 48   # every block routes: blk.0 .. blk.47 (96 sites = gate_up + down)
PHASE_BOUNDARY = 256 # launches at or below are decode-shaped; larger are prefill-shaped

DECIDER_CAPTURE = (
    "One window on a rented box with --expert-trace plus a producer variant that keeps a "
    "384-bin per-site hit histogram across evals and dumps JSONL at teardown (the site "
    "name already carries the layer); >=1000 decode steps at shallow AND deep context. "
    "GLM_ROUTE_STATS multiplicity[64] during an Inkling window is a coarser fallback."
)


def phase_of(cols):
    return "decode" if cols <= PHASE_BOUNDARY else "prefill"


def uniform_expected_distinct(n_experts, top_k, tokens):
    """E[number of distinct experts hit | `tokens` independent tokens each routing
    `top_k` DISTINCT experts chosen uniformly among all k-subsets]. Exact."""
    return n_experts * (1.0 - (1.0 - float(top_k) / n_experts) ** tokens)


def parse_site_layer(name):
    """'blk.7.ffn_gate_up_exps.weight' -> 7 ; None for names this reader does not know."""
    parts = name.split(".")
    if len(parts) < 3 or not name.startswith("blk."):
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


class StoreCounts(object):
    def __init__(self):
        self.lines = 0
        self.raw_route_records = 0
        self.launches = 0            # deduped (eval, layer) events with routing
        self.product_repeats = 0     # records dropped as the same routing seen again
        self.device_repeat_records = 0
        self.bad_cols = 0
        self.unknown_sites = 0
        self.end_notes = []          # MMQ_ROUTE_END lines: ring capacity ran/kept facts

    def as_dict(self):
        d = dict(self.__dict__)
        d["end_notes"] = list(d["end_notes"])
        return d


def parse_store(path):
    """Stream one store; yield launch dicts {layer, cols, used}. Bounded memory.
    Dedupe: first record per (eval, layer); later ones count as repeats only."""
    counts = StoreCounts()
    sites = {}      # site index -> layer (from MMQ_ROUTE_SITE lines of THIS file)
    seen = set()    # packed (eval, layer)
    launches = []
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            counts.lines += 1
            if line.startswith("MMQ_ROUTE_SITE ") or line.startswith("MMQ_ROUTE_END"):
                if line.startswith("MMQ_ROUTE_SITE "):
                    try:
                        j = json.loads(line[len("MMQ_ROUTE_SITE "):])
                    except ValueError:
                        continue
                    layer = parse_site_layer(str(j.get("name", "")))
                    if layer is not None:
                        sites[int(j["site"])] = layer
                else:
                    try:
                        endj = json.loads(line[len("MMQ_ROUTE_END"):].strip())
                        counts.end_notes.append(
                            {"ran": endj.get("ran"), "kept": endj.get("kept"),
                             "capacity": endj.get("capacity"), "ctx": endj.get("ctx")})
                    except ValueError:
                        pass
                continue
            if not line.startswith("MMQ_ROUTE "):
                continue
            try:
                j = json.loads(line[len("MMQ_ROUTE "):])
            except ValueError:
                continue
            counts.raw_route_records += 1
            layer = sites.get(int(j.get("site", -1)))
            if layer is None or not (0 <= layer < ROUTED_LAYERS):
                counts.unknown_sites += 1
                continue
            cols, used = int(j["cols"]), int(j["used"])
            key = (int(j["eval"]) << 8) | layer
            if key in seen:
                counts.product_repeats += 1
                continue
            seen.add(key)
            if cols % TOP_K != 0 or cols <= 0:
                counts.bad_cols += 1
                continue
            counts.launches += 1
            launches.append((layer, phase_of(cols), cols, used))
    return launches, counts


def stats(values):
    """mean/p10/median/p90 over a list of numbers (sorted copy; empty -> zeros)."""
    if not values:
        return {"records": 0}
    vs = sorted(values)
    n = len(vs)

    def pct(p):
        i = min(n - 1, max(0, int(math.ceil(p * n)) - 1))
        return vs[i]
    mean = sum(vs) / float(n)
    return {
        "records": n,
        "mean": round(mean, 6),
        "p10": round(pct(0.10), 6),
        "median": round(pct(0.50), 6),
        "p90": round(pct(0.90), 6),
    }


def sha256_file(path, cap_bytes=None):
    h = hashlib.sha256()
    read = 0
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
            read += len(chunk)
            if cap_bytes is not None and read > cap_bytes:
                break
    return h.hexdigest()


def by_cols_table(by_cols_ratios):
    """Per batch size: launch counts and mean breadth ratio per phase."""
    table = []
    for cols in sorted(by_cols_ratios):
        row = {"cols": cols, "tokens_per_launch": cols // TOP_K}
        for phase in ("decode", "prefill"):
            rs = by_cols_ratios[cols].get(phase) or []
            row["launches_" + phase] = len(rs)
            row["ratio_to_uniform_mean_" + phase] = round(sum(rs) / len(rs), 6) if rs else None
        table.append(row)
    return table


def build(stores, out_path):
    per_layer_launches = {}   # layer -> phase -> list of used
    cols_hist = {"decode": {}, "prefill": {}}
    all_used = {"decode": [], "prefill": []}      # pooled across layers (breadth rollup)
    by_cols_ratios = {}       # cols -> {phase: [used/E_uniform]} for the by-batch table
    store_meta = []
    unknown_hits_stores = 0

    slot_draws_total = 0
    launches_total = 0
    raw_route_records_total = 0
    product_repeats_total = 0
    device_repeat_total = 0
    bad_cols_total = 0
    unknown_sites_total = 0

    expert_hit_histograms_present = False  # a future richer record format would set this

    for path in stores:
        if not os.path.exists(path):
            raise SystemExit("missing store: %s" % path)
        launches, counts = parse_store(path)
        meta = {
            "path": path,
            "bytes": os.path.getsize(path),
            "sha256": sha256_file(path),
        }
        meta.update(counts.as_dict())
        meta["launches"] = len(launches)
        store_meta.append(meta)

        slot_draws_total += sum(cols for (_l, _p, cols, _u) in launches)
        launches_total += len(launches)
        raw_route_records_total += counts.raw_route_records
        product_repeats_total += counts.product_repeats
        device_repeat_total += max(0, counts.raw_route_records - counts.launches
                                   - counts.product_repeats - counts.unknown_sites
                                   - counts.bad_cols)
        bad_cols_total += counts.bad_cols
        unknown_sites_total += counts.unknown_sites

        for (layer, phase, cols, used) in launches:
            per_layer_launches.setdefault(layer, {"decode": [], "prefill": []})[phase].append(
                (cols, used))
            all_used[phase].append((cols, used))
            bucket = cols_hist[phase]
            bucket[str(cols)] = bucket.get(str(cols), 0) + 1
            e_uniform = uniform_expected_distinct(N_EXPERTS, TOP_K, cols // TOP_K)
            if e_uniform > 0:
                by_cols_ratios.setdefault(cols, {}).setdefault(phase, []).append(used / e_uniform)

    # ratios are computed per record against that record's own uniform expectation
    def ratios(pairs):
        out = []
        for cols, used in pairs:
            tokens = cols // TOP_K
            e_uniform = uniform_expected_distinct(N_EXPERTS, TOP_K, tokens)
            if e_uniform > 0:
                out.append(used / e_uniform)
        return out

    layers_out = []
    layer_share_sum = 0.0
    for layer in range(ROUTED_LAYERS):
        row = {"layer": layer}
        row["launch_share_of_all"] = None
        entry = per_layer_launches.get(layer)
        total_here = sum(len(entry[p]) if entry else 0 for p in ("decode", "prefill")) if entry else 0
        share = total_here / float(launches_total) if launches_total else 0.0
        layer_share_sum += share
        row["launch_share_of_all"] = round(share, 9)
        for phase in ("decode", "prefill"):
            pairs = entry[phase] if entry else []
            st = stats([u for (_c, u) in pairs])
            rs = stats(ratios(pairs))
            mean_cols = (sum(c for c, _u in pairs) / len(pairs)) if pairs else None
            block = {
                "records": st.get("records", 0),
                "used_mean": st.get("mean"),
                "used_p10": st.get("p10"),
                "used_median": st.get("median"),
                "used_p90": st.get("p90"),
                "ratio_to_uniform_mean": rs.get("mean"),
                "ratio_to_uniform_p10": rs.get("p10"),
                "ratio_to_uniform_p90": rs.get("p90"),
                "uniform_expected_at_mean_cols":
                    round(uniform_expected_distinct(N_EXPERTS, TOP_K, mean_cols // TOP_K), 6)
                    if mean_cols else None,
                # absent by construction from these stores; named, not extrapolated:
                "per_expert_hit_counts": None,
                "top_k_shares": None,
                "hhi": None,
                "absent_because": "stored MMQ_ROUTE records are per-launch aggregates "
                                  "(cols/used); no expert identities exist on disk",
            }
            row[phase] = block
        row["all"] = {"records": total_here}
        layers_out.append(row)

    pooled = {}
    for phase in ("decode", "prefill"):
        rs_all = ratios(all_used[phase])
        st_u = stats([u for (_c, u) in all_used[phase]])
        st_r = stats(rs_all)
        pooled[phase] = {
            "launches": st_u.get("records", 0),
            "slot_draws": sum(c for c, _u in all_used[phase]),
            "used_mean": st_u.get("mean"),
            "used_p10": st_u.get("p10"),
            "used_median": st_u.get("median"),
            "used_p90": st_u.get("p90"),
            "ratio_to_uniform_mean": st_r.get("mean"),
            "ratio_to_uniform_p10": st_r.get("p10"),
            "ratio_to_uniform_p90": st_r.get("p90"),
            "uniform_expected_at_mean_cols":
                round(uniform_expected_distinct(N_EXPERTS, TOP_K,
                                                (sum(c for c, _u in all_used[phase]) // max(1, len(all_used[phase]))) // TOP_K), 6)
                if all_used[phase] else None,
        }

    artifact = {
        "schema_version": 1,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generator": "scripts/inkling2/route_histogram.py",
        "provenance_note": "CPU-side aggregation over stored MMQ_ROUTE serving traces "
                           "(GGML_CUDA_MMQ_ROUTE_LOG) recorded during the 2026-11-18 tp4 "
                           "four-card Inkling-2-Small-Q3_K_M windows. No GPU window was run.",
        "model": {"name": "Inkling-2-Small-Q3_K_M", "n_experts": N_EXPERTS,
                  "top_k": TOP_K, "routed_layers": ROUTED_LAYERS},
        "phase_boundary_cols": PHASE_BOUNDARY,
        "uniform_mean_share_per_expert": round(float(TOP_K) / N_EXPERTS, 9),
        "falsifier_threshold_x_uniform": 1.5,
        "verdict": {
            # branch (a)/(b) need per-expert shares; disk stores cannot produce them
            "measured_top1_share": None,
            "branch": "c",
            "label": "inconclusive_from_stores_breadth_measured",
            "reason":
                "Every stored route record is a per-launch aggregate (cols, used); no "
                "expert identities exist in any store, so the top-1 share bar of 1.5x "
                "the uniform mean (0.03125) cannot be evaluated from bytes on disk. "
                "Measured breadth evidence IS recorded here per layer and phase.",
            "decider_capture": DECIDER_CAPTURE,
        },
        "totals": {
            "launches": launches_total,
            "slot_draws": slot_draws_total,          # sum over launches of cols (= tokens*top_k)
            "raw_route_records": raw_route_records_total,
            "accounting_sum": (launches_total + product_repeats_total + device_repeat_total
                               + bad_cols_total + unknown_sites_total),
            "product_repeats_deduped": product_repeats_total,
            "device_repeats_or_other_nonfirst": device_repeat_total,
            "bad_cols_records": bad_cols_total,
            "unknown_site_records": unknown_sites_total,
            "expert_hit_histograms_present": expert_hit_histograms_present or unknown_hits_stores > 0,
        },
        "stores": store_meta,
        "phase_launch_share": {
            p: round(len(all_used[p]) / float(launches_total), 9) if launches_total else None
            for p in ("decode", "prefill")
        },
        "cols_histogram_by_phase": cols_hist,
        "by_batch_size": by_cols_table(by_cols_ratios),
        "pooled_breadth": pooled,
        "per_layer": layers_out,
        "record_format_note":
            "Each stored record also carries tiles8=sum_e ceil(h_e/8), tiles=sum_e "
            "ceil(h_e/J) and the launch tile width j; on decode rows j=8 so those collapse. "
            "They are additional aggregates of the same hidden per-expert hit vector "
            "{h_e} that no store records directly, kept here as provenance only.",
        "phase_caveat":
            "'decode'/'prefill' are batch-shape labels (cols vs PHASE_BOUNDARY), not "
            "serving-phase truths: a small prefill tail chunk can land in the "
            "small-batch bucket, and each window's ring kept only its first evals, "
            "whose phase mix differs by window. Compare by_batch_size rows before "
            "reading any pooled phase number.",
        "depth_gap": {
            "recorded": True,
            "text": "Records carry (eval, site, cols, used) only; no context-position or "
                    "prompt-depth metadata survives in any store, and each device context's "
                    "record ring kept only its FIRST evals (capacity 65536/ctx, see "
                    "stores[].end_notes ran/kept). Which phase those early evals belong to "
                    "differs by window, so a shallow vs deep split is NOT derivable from "
                    "disk today and is not extrapolated.",
        },
    }
    with open(out_path + ".tmp", "w") as fh:
        json.dump(artifact, fh, indent=1, sort_keys=False)
        fh.write("\n")
    os.replace(out_path + ".tmp", out_path)
    return artifact


# ---------------------------------------------------------------------------
# --check: internal consistency only, reads just the artifact
# ---------------------------------------------------------------------------

def check(path):
    with open(path) as fh:
        d = json.load(fh)
    problems = []

    n_experts = d["model"]["n_experts"]
    top_k = d["model"]["top_k"]
    if d["uniform_mean_share_per_expert"] != round(float(top_k) / n_experts, 9):
        problems.append("uniform_mean_share_per_expert disagrees with model params")

    totals = d["totals"]
    hist = d["cols_histogram_by_phase"]

    # rows*slots*topk relations: slot_draws must equal sum over launches of cols,
    # recomputed here purely from the histogram the artifact carries.
    recomputed_cols_sum = 0
    recomputed_launches = 0
    for phase in ("decode", "prefill"):
        for cols_s, count in hist[phase].items():
            recomputed_cols_sum += int(cols_s) * count
            recomputed_launches += count
    if recomputed_cols_sum != totals["slot_draws"]:
        problems.append("slot_draws != sum(cols*count) recomputed from cols_histogram")
    if recomputed_launches != totals["launches"]:
        problems.append("launches != recomputed launch count from cols_histogram")
    if totals.get("accounting_sum") != totals.get("raw_route_records"):
        problems.append("record accounting does not close: "
                        "accounting_sum %r != raw_route_records %r"
                        % (totals.get("accounting_sum"), totals.get("raw_route_records")))
    store_launches = sum(s.get("launches", 0) for s in d.get("stores", []))
    if store_launches != totals["launches"]:
        problems.append("sum(store launches)=%d != totals.launches=%d"
                        % (store_launches, totals["launches"]))

    # per-layer record accounting closes against the total
    layer_records = sum(row["all"]["records"] for row in d["per_layer"])
    if layer_records != totals["launches"]:
        problems.append("sum(per_layer records)=%d != totals.launches=%d"
                        % (layer_records, totals["launches"]))
    share_sum = sum(row["launch_share_of_all"] for row in d["per_layer"])
    if abs(share_sum - 1.0) > 1e-6:
        problems.append("per-layer launch shares sum to %.9f, expected 1" % share_sum)

    # phase shares close to 1 (shares are a distribution over two phases)
    ph = d["phase_launch_share"]
    if ph["decode"] is not None and abs(ph["decode"] + ph["prefill"] - 1.0) > 1e-6:
        problems.append("phase launch shares sum to %.9f, expected 1" % (ph["decode"] + ph["prefill"]))

    # stored uniform expectations re-derive from model params (at each phase's mean cols)
    tol = max(1e-4, n_experts * 1e-7)
    for phase in ("decode", "prefill"):
        pb = d["pooled_breadth"][phase]
        launches_here = pb["launches"]
        if launches_here:
            slot_draws_here = pb["slot_draws"]
            mean_cols = int(round(slot_draws_here / float(launches_here)))
            tokens = mean_cols // top_k
            expect = round(uniform_expected_distinct(n_experts, top_k, tokens), 6)
            got = pb["uniform_expected_at_mean_cols"]
            if got is not None and abs(got - expect) > tol * 48:
                problems.append(
                    "%s: uniform_expected_at_mean_cols %.6f != recomputed %.6f"
                    % (phase, got, expect))

    # used stat ordering inside each layer/phase block
    for row in d["per_layer"]:
        for phase in ("decode", "prefill"):
            b = row[phase]
            vals = [b.get("used_p10"), b.get("used_median"), b.get("used_p90")]
            present = [v for v in vals if v is not None]
            if len(present) == 3 and not (present[0] <= present[1] <= present[2]):
                problems.append("layer %d %s: p10>median or median>p90" % (row["layer"], phase))
            if b["records"] and b.get("used_mean") is None:
                problems.append("layer %d %s: records>0 but no used_mean" % (row["layer"], phase))

    # verdict consistency with what the artifact claims measurable
    v = d["verdict"]
    if totals["expert_hit_histograms_present"]:
        if v["measured_top1_share"] is None:
            problems.append("hit histograms present but measured_top1_share is null")
    else:
        if v["measured_top1_share"] is not None:
            problems.append("no hit histograms on disk but a top-1 share was claimed")
        if v["branch"] != "c":
            problems.append("without expert ids only branch c may fire, got %r" % v["branch"])
    bar = round(1.5 * float(top_k) / n_experts, 9)
    if v["measured_top1_share"] is not None and v["measured_top1_share"] >= bar:
        problems.append("top-1 share above the replication bar but verdict still says transfer")

    for problem in problems:
        print("CHECK FAIL: " + problem, file=sys.stderr)
    return 0 if not problems else 1


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stores", nargs="*", default=DEFAULT_STORES,
                   help="MMQ_ROUTE_* stores to aggregate (default: the five tp4 windows)")
    p.add_argument("--out", default=DEFAULT_OUT, help="artifact path to write or check")
    p.add_argument("--check", nargs="?", default=False, const=True, metavar="PATH",
                   help="validate the artifact's internal consistency and exit; PATH "
                        "defaults to --out when given bare as --check")
    args = p.parse_args()

    if args.check is True:
        sys.exit(check(args.out))
    if args.check:
        sys.exit(check(args.check))
    artifact = build(args.stores, args.out)
    pooled = artifact["pooled_breadth"]
    dec = pooled["decode"]
    print(json.dumps({
        "verdict_branch": artifact["verdict"]["branch"],
        "launches_total": artifact["totals"]["launches"],
        "slot_draws_total": artifact["totals"]["slot_draws"],
        "decode_launches": dec["launches"] if isinstance(dec, dict) else None,
        "decode_ratio_to_uniform_mean":
            round(dec["ratio_to_uniform_mean"], 6) if isinstance(dec, dict) else None,
        "out": args.out,
    }))


if __name__ == "__main__":
    main()
