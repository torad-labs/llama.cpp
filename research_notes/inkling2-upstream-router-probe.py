#!/usr/bin/env python3
"""Provenance probe for Inkling 2 Small's eleven short router rows (doc 21 §1).

Answers ONE question with bytes: does the stock thinkingmachines/Inkling-2-Small
checkpoint also carry eleven ~0.686-norm router-gate rows per layer, or were they
added by the abliteration conversion that produced the served pack?

Method: the upstream repo ships safetensors shards (NOT gguf). Router gates are
`model.llm.layers.<L>.mlp.gate.weight`, dtype bfloat16 (hf_quant_config group),
shape [384, 4608]. For each shard we range-read only:
  - a header window (safetensors = u64 length + JSON table), once,
  - the exact byte span of each router tensor (3.54 MiB bf16), one at a time.
Host memory stays bounded: no stage holds more than one [384,4608] tensor plus
small accumulators; results land on disk per layer as they are measured.

The comparison side comes from the cached exact F32 routers of
Inkling-2-Small-abliterated-e4853745-Q3_K_M-imatrix.gguf
(~/.cache/inkling2-weightread/routers.npz, float32 [48, 384, 4608], read exactly
from its F32 ffn_gate_inp tensors — research_notes/inkling2-weight-read.py).
That cache is a local artifact from the prior seat's read; compacting it costs
one full decompress pass (~700 MB peak, seconds) and happens at most twice
(norms+samples now, union-index rows after fetch). No HTTP path materializes
full tensors.

Everything lands under --outdir; nothing is written inside any repo.
Run with python3 -I and pass paths as arguments (downloads are untrusted data).
"""
import json
import os
import struct
import sys
import time
import urllib.request

BASE = "https://huggingface.co/thinkingmachines/Inkling-2-Small/resolve/main/"
REPO_REF = "thinkingmachines/Inkling-2-Small @ main"
NLAY, NEXP, NHID = 48, 384, 4608
SHORT = 0.9        # eleven rows sit at ~0.686; unit rows at 1.000 +- 1e-4
SEED = 20261122    # fixed probe-row sample seed: same indices both checkpoints


def stage_done(outdir, name):
    return os.path.exists(os.path.join(outdir, ".done-" + name))


def touch(outdir, name):
    open(os.path.join(outdir, ".done-" + name), "w").write("")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fetch_range(shard, lo, hi):
    """GET bytes [lo,hi) of a shard path under BASE. Returns (bytes, commit)."""
    req = urllib.request.Request(BASE + shard,
                                 headers={"Range": f"bytes={lo}-{hi - 1}",
                                          "User-Agent": "torad-weightread-probe"})
    last = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                code = r.getcode()
                if code not in (200, 206):
                    raise IOError(f"HTTP {code}")
                data = r.read(hi - lo)
                commit = str(r.headers.get("X-Repo-Commit") or r.headers.get("ETag", ""))
                if len(data) != hi - lo:
                    raise IOError(f"short read {len(data)} != {hi - lo}")
                return data, commit
        except Exception as e:  # noqa: BLE001 — retry any transport error
            last = e
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"{shard} [{lo},{hi}): {last}")


def parse_header_window(window):
    """window covers the first >=8+n bytes of a safetensors file."""
    n = struct.unpack_from("<Q", window, 0)[0]
    need = 8 + n
    meta = json.loads(bytes(window[8:need]).decode("utf-8"))
    return need, meta


DT_NBYTES = {"BF16": 2, "F16": 2, "F32": 4}
DTNP = {"BF16": "<f2", "F32": "<f4", "F16": "<f2"}

WANT_TENSOR = ".mlp.gate.weight"


# ---------------------------------------------------------------------------
# stage: headers  (plan.json: per-layer shard/offsets; resume-safe)
# ---------------------------------------------------------------------------

def stage_headers(index_path, outdir):
    plan_f = os.path.join(outdir, "upstream-header-plan.json")
    if stage_done(outdir, "headers"):
        return json.load(open(plan_f))
    wm = json.load(open(index_path))["weight_map"]
    shards = {}
    for L in range(NLAY):
        tname = f"model.llm.layers.{L}{WANT_TENSOR}"
        shards.setdefault(wm[tname], []).append(L)
    log(f"{len(shards)} shards hold router gates")
    layers, hdrs, commit_ref = {}, {}, ""
    for shard in sorted(shards):
        win_len, parsed = 262144, None
        for _ in range(6):                     # grow the window until JSON parses whole
            raw, commit = fetch_range(shard, 0, win_len)
            commit_ref = commit or commit_ref
            try:
                parsed = parse_header_window(raw)
                break
            except Exception:
                win_len *= 4
        assert parsed is not None, f"{shard}: safetensors header did not parse"
        need, meta = parsed
        keep = {}
        for L in shards[shard]:
            tname = f"model.llm.layers.{L}{WANT_TENSOR}"
            ent = meta[tname]
            assert ent["shape"] == [NEXP, NHID], (tname, ent["shape"])
            nb = int(ent["data_offsets"][1] - ent["data_offsets"][0])
            assert abs(nb - NEXP * NHID * DT_NBYTES[ent["dtype"]]) <= 1, (tname, nb)
            layers[str(L)] = {"shard": shard, "beg": int(ent["data_offsets"][0]),
                              "end": int(ent["data_offsets"][1]), "dtype": ent["dtype"]}
            keep[tname] = {k: ent[k] for k in ("dtype", "shape", "data_offsets")}
        hdrs[shard] = {"header_bytes": need, "n_tensors": len(meta), "keep": keep}
        log(f"{shard}: header {need} B, router tensors kept: {len(keep)}")
    plan = {"layers": layers, "headers": hdrs,
            "repo_commit": commit_ref, "repo_ref": REPO_REF}
    json.dump(plan, open(plan_f, "w"), indent=1)
    touch(outdir, "headers")
    return plan


# ---------------------------------------------------------------------------
# stage: ablated norms + sample rows  (one full pass over the local cache)
# ---------------------------------------------------------------------------

def ab_norms_and_samples(ablated_routers_path, outdir):
    import numpy as np
    nf = os.path.join(outdir, "ablated-router-norms.npz")
    rf = os.path.join(outdir, "ablated-probe-rows.npz")
    if not os.path.exists(nf) or not os.path.exists(rf):
        z = np.load(ablated_routers_path)["weights"]     # [48,384,4608] f32 exact
        layers_order = np.load(ablated_routers_path)["layers"]
        assert list(layers_order[[0, -1]]) == [0, NLAY - 1], layers_order
        nrm = np.linalg.norm(z.astype(np.float64), axis=2)
        np.savez(nf, norms=nrm)
        rng = np.random.RandomState(SEED)
        unit_sample = rng.choice(NEXP, size=24, replace=False).astype(int)
        rows = {}
        for L in range(NLAY):
            sel = np.where(nrm[L] < SHORT)[0]
            rows[f"L{L}_short_idx"] = sel.astype(np.int16)
            rows[f"L{L}_short"] = z[L][sel].astype(np.float32)
            rows[f"L{L}_unit_idx"] = unit_sample.astype(np.int16)
            rows[f"L{L}_unit"] = z[L][unit_sample].astype(np.float32)
        np.savez_compressed(rf, **rows)
        del z
        log("ablated cache compacted to norms + probe rows (one full pass)")
    return nf, rf


# ---------------------------------------------------------------------------
# stage: fetch upstream routers  (streamed per tensor; recorded as it lands)
# ---------------------------------------------------------------------------

def stage_fetch(plan, outdir):
    import numpy as np
    plan_f = os.path.join(outdir, "upstream-header-plan.json")
    if not isinstance(plan, dict):                    # resumed run loads it instead
        plan = json.load(open(plan_f))
    nf = os.path.join(outdir, "upstream-router-norms.npz")
    rf = os.path.join(outdir, "upstream-probe-rows.npz")
    jl_f = os.path.join(outdir, "fetched-layers.jsonl")
    done = set()
    if os.path.exists(jl_f) and os.path.exists(nf) and os.path.exists(rf):
        done = {int(json.loads(l)["layer"]) for l in open(jl_f)}
        log(f"resume: {len(done)} layers already fetched")
    rng = np.random.RandomState(SEED)
    unit_sample = rng.choice(NEXP, size=24, replace=False).astype(int)
    norms = np.zeros((NLAY, NEXP), dtype=np.float64)
    if done:
        norms[:] = np.load(nf)["norms"]
    rows, t0, payload = {}, time.time(), 0
    if done:
        zr = np.load(rf)
        rows = {k: zr[k] for k in zr.files}
    jl = open(jl_f, "a")
    for LS in range(NLAY):
        if LS in done:
            continue
        e = plan["layers"][str(LS)]
        data0 = plan["headers"][e["shard"]]["header_bytes"]
        raw, commit = fetch_range(e["shard"], e["beg"] + data0, e["end"] + data0)
        w = np.frombuffer(raw, dtype=DTNP[e["dtype"]]
                          ).astype(np.float32).reshape(NEXP, NHID)
        nrm = np.linalg.norm(w.astype(np.float64), axis=1)     # [384]
        norms[LS] = nrm
        sel = np.where(nrm < SHORT)[0]
        rows[f"L{LS}_short_idx"] = sel.astype(np.int16)
        rows[f"L{LS}_short"] = w[sel].astype(np.float32)
        rows[f"L{LS}_unit_idx"] = unit_sample.astype(np.int16)
        rows[f"L{LS}_unit"] = w[unit_sample].astype(np.float32)
        payload += len(raw)
        jl.write(json.dumps({"layer": LS, "bytes": len(raw),
                             "n_short_lt_0.9": int(sel.size),
                             "commit": commit,
                             "elapsed_s": round(time.time() - t0, 1)}) + "\n")
        jl.flush()
        np.savez(nf, norms=norms)                    # 147 KB — per-layer checkpointing
        np.savez_compressed(rf, **rows)
        if LS % 8 == 0:
            log(f"layer {LS}: fetched ({payload >> 20} MiB router payload so far)")
    jl.close()
    json.dump({"bytes_router_payload": payload, "unit_sample_indices":
               [int(i) for i in unit_sample], "short_threshold": SHORT},
              open(os.path.join(outdir, "fetch-log.json"), "w"))
    touch(outdir, "fetch")


# ---------------------------------------------------------------------------
# stage: row vectors at arbitrary indices (small ranged reads / one cache pass)
# ---------------------------------------------------------------------------

def fetch_up_rows(plan, outdir, want):
    """want: {L: sorted list of row indices}. Ranged-reads each missing row's own
    9 KiB span from HF; merges into upstream-probe-rows.npz as L{L}_xref/_xref_idx."""
    import numpy as np
    rf = os.path.join(outdir, "upstream-probe-rows.npz")
    zr = np.load(rf)
    rows = {k: zr[k] for k in zr.files}
    t0 = time.time()
    nget = 0
    for LS in sorted(want):
        e = plan["layers"][str(LS)]
        data0 = plan["headers"][e["shard"]]["header_bytes"]
        dsize = DT_NBYTES[e["dtype"]]
        idxs = [int(i) for i in want[LS]]
        ki, kv = f"L{LS}_xref_idx", f"L{LS}_xref"
        oldmap = ({int(k): rows[kv][j] for j, k in enumerate(rows[ki])}
                  if kv in rows else {})
        missing = [i for i in idxs if i not in oldmap]
        for i in missing:
            lo = e["beg"] + data0 + i * NHID * dsize
            raw, _ = fetch_range(e["shard"], lo, lo + NHID * dsize)
            oldmap[i] = np.frombuffer(raw, dtype=DTNP[e["dtype"]]
                                      ).astype(np.float32).reshape(NHID)
            nget += 1
        merged_i = sorted(oldmap)
        rows[ki] = np.array(merged_i, dtype=np.int16)
        rows[kv] = (np.stack([oldmap[i] for i in merged_i]).astype(np.float32)
                    if merged_i else np.zeros((0, NHID), dtype=np.float32))
    np.savez_compressed(rf, **rows)
    np.savez_compressed(rf, **rows)
    log(f"upstream gapfill: {nget} row reads "
        f"(~{nget * NHID * 2 >> 10} KiB), {round(time.time() - t0, 1)}s")


def ab_rows_at(ablated_routers_path, outdir, want):
    """want: {L: indices}. One full pass over the cached routers.npz extracting
    exactly those rows; merges into ablated-probe-rows.npz as L{L}_xref."""
    import numpy as np
    rf = os.path.join(outdir, "ablated-probe-rows.npz")
    zr = np.load(rf)
    rows = {k: zr[k] for k in zr.files}
    todo_any = False
    for L, idxs in sorted(want.items()):
        ki = f"L{L}_xref_idx"
        have = set(int(i) for i in rows[ki]) if ki in rows else set()
        if not set(idxs) <= have:
            todo_any = True
    if todo_any or any(f"L{L}_xref" not in rows for L in want):
        z = np.load(ablated_routers_path)["weights"]     # one decompress pass
        for LS, idxs in sorted(want.items()):
            arr = np.array(sorted(set(int(i) for i in idxs)), dtype=int)
            rows[f"L{LS}_xref_idx"] = arr.astype(np.int16)
            rows[f"L{LS}_xref"] = z[int(LS)][arr].astype(np.float32) \
                if arr.size else np.zeros((0, NHID), dtype=np.float32)
        np.savez_compressed(rf, **rows)
        del z
        log("ablated xref rows extracted (one full cache pass)")
    return rf


# ---------------------------------------------------------------------------
# stage: compare — the verdict numbers
# ---------------------------------------------------------------------------

def stage_compare(outdir):
    import numpy as np
    from math import comb

    nab = np.load(os.path.join(outdir, "ablated-router-norms.npz"))["norms"]
    nup = np.load(os.path.join(outdir, "upstream-router-norms.npz"))["norms"]
    abr = np.load(os.path.join(outdir, "ablated-probe-rows.npz"))
    upr = np.load(os.path.join(outdir, "upstream-probe-rows.npz"))

    def rowmap(zr, L, kind):
        key = f"L{L}_{kind}"
        if key not in zr.files:
            return {}
        return {int(i): zr[key][j] for j, i in enumerate(zr[f"L{L}_{kind}_idx"])}

    def cos(a, b):
        d = float(np.linalg.norm(a) * np.linalg.norm(b))
        return round(float(np.dot(a, b)) / d, 6)

    setsA = {L: set(int(i) for i in np.where(nab[L] < SHORT)[0]) for L in range(NLAY)}
    setsB = {L: set(int(i) for i in np.where(nup[L] < SHORT)[0]) for L in range(NLAY)}

    res = {"threshold_short_norm": SHORT,
           "repo_ref": REPO_REF,
           "unit_rows_mean_norm_ablated": float(nab[nab > SHORT].mean()),
           "unit_rows_std_norm_ablated": float(nab[nab > SHORT].std()),
           "unit_rows_mean_norm_upstream": float(nup[nup > SHORT].mean()),
           "unit_rows_std_norm_upstream": float(nup[nup > SHORT].std()),
           "norm_min_max_upstream_unit": [float(nup.min()), None],
           "overlap": {}, "per_layer_table": []}
    unit_cos, shared_cos, xref_cos = [], [], []
    tot_ov, exp_tot = 0, 0.0
    for L in range(NLAY):
        A, B = setsA[L], setsB[L]
        ov = len(A & B)
        na, nb = len(A), len(B)
        tot_ov += ov
        exp_tot += na * nb / NEXP if nb else 0.0
        p = None
        if na == 11 and nb == 11:
            lo_k = max(ov, 0)
            p = sum(comb(11, k) * comb(373, 11 - k) for k in range(lo_k, 12)) / comb(384, 11)
        # pair cosines wherever both sides carry the row vector
        am, bm = rowmap(abr, L, "short"), rowmap(upr, L, "xref")
        ambm = {i: (am[i], bm[i]) for i in (A & set(bm)) & set(am)}
        xc = [cos(a, b) for a, b in ambm.values()]
        xref_cos.extend(xc)
        ashort_bsame = {i: (am[i], rowmap(upr, L, "short").get(i)) for i in A}
        sc = [cos(*v) for v in ashort_bsame.values() if v[1] is not None]
        shared_cos.extend(sc)
        ua, ub = rowmap(abr, L, "unit"), rowmap(upr, L, "unit")
        common_units = sorted(set(ua) & set(ub))
        unit_cos.extend(cos(ua[i], ub[i]) for i in common_units)
        res["overlap"][str(L)] = {
            "ablated_short": sorted(A), "upstream_short": sorted(B),
            "shared": sorted(A & B), "expected_chance": round(na * nb / NEXP, 3)
            if nb else 0.0,
            "p_hypergeom_11v11_or_more": (round(p, 9) if p is not None else None)}
        res["per_layer_table"].append({"layer": L,
                                       "ablated_short_norms":
                                           [round(float(nab[L][i]), 4) for i in sorted(A)],
                                       "upstream_short_norms":
                                           [round(float(nup[L][i]), 4) for i in sorted(B)],
                                       "shared": len(A & B)})
    # short-row norm signatures (either side may be EMPTY — that is a verdict, not an error)
    parts_ab = [nab[L][sorted(setsA[L])] for L in range(NLAY) if setsA[L]]
    parts_up = [nup[L][sorted(setsB[L])] for L in range(NLAY) if setsB[L]]
    s_ab = np.concatenate(parts_ab) if parts_ab else np.zeros(0)
    s_up = np.concatenate(parts_up) if parts_up else np.zeros(0)
    res["upstream_all_row_norm_pcts"] = {
        "p001": round(float(np.percentile(nup, 0.1)), 6),
        "p01": round(float(np.percentile(nup, 1)), 6),
        "p50": round(float(np.percentile(nup, 50)), 6),
        "min": round(float(nup.min()), 6), "max": round(float(nup.max()), 6)}
    res["ablated_short_row_norm_stats"] = {"count": int(s_ab.size),
                                           "mean": float(s_ab.mean()),
                                           "std": float(s_ab.std()),
                                           "min": float(s_ab.min()),
                                           "max": float(s_ab.max())}
    res["upstream_short_row_norm_stats"] = {"count": int(s_up.size),
                                            "mean": float(s_up.mean()) if s_up.size else None,
                                            "std": float(s_up.std()) if s_up.size else None,
                                            "min": float(s_up.min()) if s_up.size else None,
                                            "max": float(s_up.max()) if s_up.size else None}
    res["cos_unit_sample_ablated_vs_upstream"] = {
        "n": len(unit_cos), "mean": round(float(np.mean(unit_cos)), 6),
        "min": round(float(np.min(unit_cos)), 6), "max": round(float(np.max(unit_cos)), 6)}
    res["cos_shared_rows_11v11"] = {
        "n": len(shared_cos),
        "vals": shared_cos[:64],
        "mean": round(float(np.mean(shared_cos)), 6) if shared_cos else None}
    res["cos_xref_rows_at_same_index"] = {
        "n": len(xref_cos),
        "mean": round(float(np.mean(xref_cos)), 6) if xref_cos else None,
        "min": round(float(np.min(xref_cos)), 6) if xref_cos else None}
    res["total_shared"] = tot_ov
    res["total_expected_chance"] = round(exp_tot, 2)
    counts_b = [len(setsB[L]) for L in range(NLAY)]
    res["upstream_short_counts_per_layer"] = counts_b
    counts_a = [len(setsA[L]) for L in range(NLAY)]
    res["ablated_short_counts_per_layer"] = counts_a

    json.dump(res, open(os.path.join(outdir, "eleven-rows-provenance.json"), "w"),
              indent=1)
    touch(outdir, "compare")
    log(f"VERDICT NUMBERS: upstream short rows/layer "
        f"min={min(counts_b)} max={max(counts_b)} total={sum(counts_b)}; "
        f"ablated min={min(counts_a)} max={max(counts_a)} total={sum(counts_a)}; "
        f"shared {tot_ov} vs chance {res['total_expected_chance']}; "
        f"unit-sample cos mean {res['cos_unit_sample_ablated_vs_upstream']['mean']}")
    return res


# ---------------------------------------------------------------------------

def main():
    outdir = os.path.abspath(sys.argv[1])
    index_path = sys.argv[2]
    ablated_routers = sys.argv[3]
    only = sys.argv[4] if len(sys.argv) > 4 else "all"
    os.makedirs(outdir, exist_ok=True)

    import numpy as np

    plan_f = os.path.join(outdir, "upstream-header-plan.json")
    plan = stage_headers(index_path, outdir)
    nf_ab, rf_ab = ab_norms_and_samples(ablated_routers, outdir)

    def sets_from(norms_f):
        n = np.load(norms_f)["norms"]
        return {L: sorted(int(i) for i in np.where(n[L] < SHORT)[0]) for L in range(NLAY)}

    A = sets_from(nf_ab)
    xref_for_fetch = {L: A[L] for L in range(NLAY)}   # upstream rows at ablated-short idx
    stage_fetch(plan, outdir)
    B = sets_from(os.path.join(outdir, "upstream-router-norms.npz"))
    union = {L: sorted(set(A[L]) | set(B[L])) for L in range(NLAY)}
    if only in ("all", "compare"):
        fetch_up_rows(plan, outdir, want=union)
        ab_rows_at(ablated_routers, outdir, want={L: union[L] for L in range(NLAY)})
        stage_compare(outdir)


if __name__ == "__main__":
    main()
