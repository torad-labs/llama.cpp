#!/usr/bin/env python3
"""Analyze weight-read artifacts -> structured findings for doc 21.

Reads the outdir produced by inkling2-weight-read.py and prints a machine-
readable JSON of computed results plus human-readable highlights:
  * router structure: norms, eigen spectra, duplicate-pair counts;
  * expert energy concentration per layer (gate/up and down separately):
    top-k shares, Gini, participation ratio, gate<->down correlation;
  * cross-LAYER stability: correlation between layers' expert-energy vectors,
    mean pairwise |corr|, best/worst pairs;
  * fingerprint cosine structure within and across layers;
  * sampled zero-code fraction per layer.
"""
import json
import sys

import numpy as np


def gini(x):
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = x.size
    if n == 0 or x.sum() <= 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


def main(outdir):
    R = np.load(f"{outdir}/routers.npz")
    W = R["weights"]                      # (L,384,4608) f32
    layers = R["layers"].tolist()
    L, NE, D = W.shape
    print(f"routers: {W.shape}, layers {min(layers)}..{max(layers)}", flush=True)

    EB = np.load(f"{outdir}/expert-blocks.npz")
    EGU = EB["energy_gateup"]             # (Lg,384) proxy energy gate/up
    EDN = EB["energy_down"]
    gu_layers = EB["gateup_layers"].tolist()
    dn_layers = EB["down_layers"].tolist()
    zfrac = EB["zero_code_frac_gateup_sampled"]

    FP = np.load(f"{outdir}/fingerprints.npz")
    FPGU = FP["gu"]                       # (L,384,4,128)
    meta = json.loads(str(FP["meta"]))
    print(f"fingerprints: gu {FPGU.shape}", flush=True)

    out = {}

    # ---- expert-energy concentration ------------------------------------
    def concentration(vec):
        v = vec / vec.sum()
        order = np.argsort(v)[::-1]
        return {
            "top16_share": float(v[order[:16]].sum()),
            "top32_share": float(v[order[:32]].sum()),
            "top64_share": float(v[order[:64]].sum()),
            "top96_share": float(v[order[:96]].sum()),
            "gini": gini(vec),
            "participation_ratio": float((vec.sum() ** 2) / np.square(vec).sum()),
        }

    conc_gu = [concentration(EGU[i]) for i in range(EGU.shape[0])]
    conc_dn = [concentration(EDN[i]) for i in range(EDN.shape[0])]
    out["gate_up_concentration"] = {"layers": gu_layers, "stats": conc_gu}
    out["down_concentration"] = {"layers": dn_layers, "stats": conc_dn}

    # gate/up vs down energy correlation per layer
    common = sorted(set(gu_layers) & set(dn_layers))
    gi = {l: i for i, l in enumerate(gu_layers)}
    di = {l: i for i, l in enumerate(dn_layers)}
    gd_corr = []
    for l in common:
        a = EGU[gi[l]]
        b = EDN[di[l]]
        if a.std() > 0 and b.std() > 0:
            gd_corr.append(float(np.corrcoef(a, b)[0, 1]))
    out["gate_vs_down_energy_correlation"] = {
        "per_layer": dict(zip([str(c) for c in common], gd_corr)),
        "mean": float(np.mean(gd_corr)),
    }

    # ---- cross-layer expert-energy stability ------------------------------
    def cross_layer(mat, label):
        C = mat / (mat.sum(axis=1, keepdims=True) + 1e-30)
        K = C.shape[0]
        corr = np.corrcoef(C)
        tri = corr[np.triu_indices(K, 1)]
        best_i = np.unravel_index(np.argmax(corr - np.eye(K)), corr.shape)
        worst_i = np.unravel_index((corr + np.eye(K)).argmin(), corr.shape)
        return {
            "matrix_label": label,
            "offdiag_mean": float(tri.mean()),
            "offdiag_abs_max": float(np.abs(tri).max()),
            "best_pair": [int(best_i[0]), int(best_i[1]), float(corr[best_i])],
            "worst_pair": [int(worst_i[0]), int(worst_i[1]), float(corr[worst_i])],
            "full_matrix": corr.tolist(),
        }
    out["cross_layer_gateup"] = cross_layer(EGU, "gate_up_energy")
    out["cross_layer_down"] = cross_layer(EDN, "down_energy")

    # energy vs router-norm coupling: do high-energy experts have hot routers?
    norms_l = np.linalg.norm(W, axis=2)      # (L,384)
    cc = []
    for i, l in enumerate(sorted(layers)):
        gl = gu_layers.index(l) if l in gu_layers else None
        if gl is not None and EGU[gl].std() > 0:
            cc.append(float(np.corrcoef(norms_l[i], EGU[gl])[0, 1]))
    out["router_norm_vs_expert_energy_corr"] = {"per_layer": dict(zip(map(str, sorted(layers)), cc)),
                                                "mean": float(np.mean(cc)) if cc else None}

    # ---- fingerprint cosine structure ------------------------------------
    def fp_cos(mat):
        # mat (L,NE,S,F): flatten samples per expert to one vector each
        V = mat.reshape(mat.shape[0] * mat.shape[1], -1)
        Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-30)
        Lk = mat.shape[0]
        NE = mat.shape[1]
        within = []
        across_idx = []
        corr_within_all = []
        for li in range(Lk):
            blk = Vn[li * NE:(li + 1) * NE]
            g = blk @ blk.T
            tri = g[np.triu_indices(NE, 1)]
            corr_within_all.append(tri)
        G = Vn @ Vn.T     # full (L*NE)^2 — too big? L*NE ~18k -> fine
        return np.stack(corr_within_all), G

    Wcorr, Gfull = fp_cos(FPGU)
    out["fingerprint_within_layer"] = {
        "per_layer_mean_cos": [float(t.mean()) for t in Wcorr],
        "per_layer_absmax_cos": [float(np.abs(t).max()) for t in Wcorr],
        "pairs_above_099_per_layer": [int((t > 0.99).sum()) for t in Wcorr],
    }
    # cross-layer same-expert fingerprint correlation
    NE = FPGU.shape[1]
    Lk = FPGU.shape[0]
    flat = FPGU.reshape(Lk * NE, -1)
    flatn = flat / (np.linalg.norm(flat, axis=1, keepdims=True) + 1e-30)
    same_expert_cross = []
    li_sorted = gu_layers
    for a in range(Lk):
        for b in range(a + 1, Lk):
            va = flatn[a * NE:(a + 1) * NE]
            vb = flatn[b * NE:(b + 1) * NE]
            c = np.diag(va @ vb.T)
            same_expert_cross.append(float(c.mean()))
    out["fingerprint_same_expert_across_layers_mean"] = float(np.mean(same_expert_cross))

    out["zero_code_frac_sampled_gateup"] = {
        "layers": gu_layers, "values": zfrac.tolist(),
        "mean": float(zfrac.mean()),
    }

    json.dump(out, open(f"{outdir}/analysis.json", "w"), indent=1)

    # human highlights
    print("\n=== HIGHLIGHTS ===", flush=True)
    print("gate/up energy concentration (layer means):")
    print("  top16 %.3f top32 %.3f top64 %.3f gini-mean %.4f" % (
        np.mean([c["top16_share"] for c in conc_gu]),
        np.mean([c["top32_share"] for c in conc_gu]),
        np.mean([c["top64_share"] for c in conc_gu]),
        np.mean([c["gini"] for c in conc_gu])))
    print("down energy concentration:")
    print("  top16 %.3f top32 %.3f top64 %.3f gini-mean %.4f" % (
        np.mean([c["top16_share"] for c in conc_dn]),
        np.mean([c["top32_share"] for c in conc_dn]),
        np.mean([c["top64_share"] for c in conc_dn]),
        np.mean([c["gini"] for c in conc_dn])))
    print("cross-layer gate/up offdiag mean corr: %.4f" %
          out["cross_layer_gateup"]["offdiag_mean"])
    print("router-norm vs expert-energy corr mean: %.4f" %
          (out["router_norm_vs_expert_energy_corr"]["mean"] or 0))
    print("zero-code sampled frac mean: %.5f" % zfrac.mean())
    wl = out["fingerprint_within_layer"]
    import statistics
    print("within-layer expert-fp cos: mean-of-means %.4f, absmax %.4f, pairs>0.99 max/layer %d" % (
        statistics.mean(wl["per_layer_mean_cos"]),
        max(wl["per_layer_absmax_cos"]),
        max(wl["pairs_above_099_per_layer"])))


if __name__ == "__main__":
    main(sys.argv[1])
