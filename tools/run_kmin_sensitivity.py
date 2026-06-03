"""k_min floor sensitivity sweep (F6 prototype, not run for this submission).

The Stage-2 constrained-BIC fallback (paper eq. 5) selects
  lambda*_fallback = argmin_{lambda : k(lambda) >= k_min} BIC(lambda)
with k_min = max(p, 10). This script re-runs the lambda-grid sweep on
the five fallback snapshots (Oct 2008, Mar 2009, Jan 2020, Mar 2020,
Jun 2020) capturing per-grid-point precision matrix, edge count, and
BIC, then picks lambda*_fallback for k_min in {p/2, p, 2p}. For each
new selection it re-runs Stage-3 (lead/lag cascade) and Stage-4
(network metrics) on the modified adjacency, leaving the five
non-fallback snapshots' cached Stage-3/4 outputs intact. Stage-5 NSI
is recomputed for all ten snapshots under each k_min choice.

Outputs:
  results/kmin_sensitivity/results.json
  results/kmin_sensitivity/per_kmin_nsi.tsv

Expected compute: ~45-65 min on Apple M5. The paper's main-text F6
limitation enumerates this sweep as future work; the script is the
prototype used to scope it. The results directory is empty in the
submission cache.
"""

import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import graphical_lasso

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from src.config import SNAPSHOTS, GLASSO_LAMBDA_RANGE, GLASSO_N_LAMBDAS, SNAPSHOTS_DIR
from src.stage2_precision.glasso_filter import (compute_ebic,
                                                precision_to_partial_corr,
                                                build_adjacency_matrix,
                                                _gamma_for_ratio,
                                                _delta_for_ratio)
from src.stage3_direction.lead_lag import assign_directions_snapshot
from src.stage4_network.analysis import (erdos_renyi_test, pagerank_analysis,
                                         community_analysis, motif_significance,
                                         build_nx_graph)
from src.stage5_nsi.stress_index import NSI_WEIGHTS_4CH

OUT_DIR = ROOT / "results" / "kmin_sensitivity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FALLBACK_LABELS = ["Oct 2008 Peak", "Mar 2009 Recovery",
                   "Jan 2020 Pre-shock", "Mar 2020 Peak",
                   "Jun 2020 Stable"]
KMIN_RATIOS = {"p_half": 0.5, "p": 1.0, "two_p": 2.0}

NSI_WEIGHTS = np.asarray(NSI_WEIGHTS_4CH, dtype=float)
assert NSI_WEIGHTS.shape == (4,) and np.isclose(NSI_WEIGHTS.sum(), 1.0)


ebic_gamma_for_ratio = _gamma_for_ratio
shrinkage_delta = _delta_for_ratio


def lambda_grid_sweep(R_avg, n_samples):
    """Full lambda sweep on a single snapshot's R_avg. Returns list of
    per-grid-point dicts with (lambda, n_edges, ebic, precision)."""
    p = R_avg.shape[0]
    ratio = n_samples / p
    delta = shrinkage_delta(ratio)
    gamma = ebic_gamma_for_ratio(ratio)

    R = (1.0 - delta) * R_avg + delta * np.eye(p)
    eigvals = np.linalg.eigvalsh(R)
    if eigvals.min() < 1e-6:
        R = R + (1e-4 - eigvals.min()) * np.eye(p)

    lambda_min, lambda_max = GLASSO_LAMBDA_RANGE
    lambdas = np.logspace(np.log10(lambda_max), np.log10(lambda_min),
                          GLASSO_N_LAMBDAS)
    trace = []
    n_failed = 0
    for lam in lambdas:
        try:
            _, prec = graphical_lasso(R, alpha=lam, max_iter=500, mode="cd")
            ebic = compute_ebic(prec, R, n_samples, gamma=gamma)
            mask = np.abs(prec) > 1e-10
            np.fill_diagonal(mask, False)
            n_edges = int(mask.sum() // 2)
            trace.append({"lambda": float(lam), "n_edges": n_edges,
                          "ebic": float(ebic), "precision": prec,
                          "shrunk_R": R})
        except Exception:
            n_failed += 1
    assert len(trace) > 0, "no GLASSO convergence on this snapshot"
    return trace, {"p": p, "n": n_samples, "ratio": float(ratio),
                   "delta": float(delta), "gamma": float(gamma),
                   "n_failed": int(n_failed),
                   "n_grid_visited": len(trace)}


def pick_lambda(trace, k_min):
    """Constrained-BIC argmin on the trace."""
    eligible = [r for r in trace if r["n_edges"] >= k_min]
    if not eligible:
        # No grid point satisfies k >= k_min — return the densest available
        # (smallest lambda with largest edge count).
        densest = max(trace, key=lambda r: r["n_edges"])
        return densest, "NO_ELIGIBLE_DENSEST_FALLBACK"
    chosen = min(eligible, key=lambda r: r["ebic"])
    return chosen, "OK"


def build_adjacency(precision):
    pc = precision_to_partial_corr(precision)
    return build_adjacency_matrix(pc, precision)


def run_stage3_one(adj, returns_window, tickers, R_avg):
    return assign_directions_snapshot(adj, returns_window, tickers,
                                      R_avg, method="both")


def run_stage4_one(directed_adj, tickers, regime, sector_map,
                   n_motif_rewires=100, n_er_sims=200):
    G = build_nx_graph(directed_adj, tickers)
    n_nodes, n_edges = G.number_of_nodes(), G.number_of_edges()
    if n_edges < 5:
        return None
    er = erdos_renyi_test(G, n_sims=n_er_sims)
    pr = pagerank_analysis(G, tickers)
    comm = community_analysis(G, tickers, sector_map)
    motifs = None
    if n_edges >= 10:
        try:
            motifs = motif_significance(G, n_rewires=n_motif_rewires)
        except Exception as e:
            print(f"      motif failed: {e}")
    return {
        "graph": G, "erdos_renyi": er, "pagerank": pr,
        "community": comm, "motifs": motifs, "tickers": tickers,
        "regime": regime, "n_nodes": n_nodes, "n_edges": n_edges,
    }


def compute_nsi_channels(stage1_corrs, stage3, stage4):
    """Build the four NSI raw channels per snapshot.
    Returns dict label -> (s, h, rho, mu) matching the snapshot NSI
    code in src/stage5_nsi/stress_index.py:114-117.
    """
    labels = list(stage4.keys())
    # Match production: ffl_baseline = mean of signed baseline FFL Zs,
    # motif_shift = ffl_z / abs(ffl_baseline). src/stage5_nsi/stress_index.py:65,87.
    base_ffl_z = [data["motifs"]["z_scores"].get("feed_forward_loop", 0.0)
                  for data in stage4.values()
                  if data.get("regime") == "baseline" and data.get("motifs")]
    if not base_ffl_z:
        raise RuntimeError("missing baseline FFL Z for normalisation")
    ffl_baseline = float(np.mean(base_ffl_z))
    base_abs = abs(ffl_baseline) if ffl_baseline != 0.0 else 1.0
    assert base_abs > 0

    channels = {}
    for lab in labels:
        s4 = stage4[lab]
        n_nodes = s4["n_nodes"]
        n_edges_dir = s4["n_edges"]
        # NSI sparsity = 1 - directed-density:
        density = n_edges_dir / (n_nodes * (n_nodes - 1) / 2)
        s = 1.0 - density
        h = float(s4["pagerank"]["hhi_top10"])  # top-10 PageRank HHI (NSI hub channel)
        rho = float(np.mean(stage1_corrs[lab]["R_avg"][np.triu_indices(n_nodes, k=1)]))
        m = s4.get("motifs")
        z_ffl = float(m["z_scores"]["feed_forward_loop"]) if m else 0.0
        mu = z_ffl / base_abs
        channels[lab] = (s, h, rho, mu)
    return channels


def normalise_and_nsi(channels):
    """Min/max-normalise across snapshots and apply NSI weights."""
    labels = list(channels.keys())
    X = np.array([channels[lab] for lab in labels])  # (10, 4)

    def mm(col):
        lo, hi = col.min(), col.max()
        return (col - lo) / (hi - lo) if hi > lo else np.zeros_like(col)

    Xn = np.stack([mm(X[:, k]) for k in range(4)], axis=1)
    nsi = Xn @ NSI_WEIGHTS
    return {labels[i]: float(nsi[i]) for i in range(len(labels))}


def load_canonical():
    with open(SNAPSHOTS_DIR / "stage1_results.pkl", "rb") as f:
        s1 = pickle.load(f)
    with open(SNAPSHOTS_DIR / "stage2_results.pkl", "rb") as f:
        s2 = pickle.load(f)
    with open(SNAPSHOTS_DIR / "stage3_results.pkl", "rb") as f:
        s3 = pickle.load(f)
    with open(SNAPSHOTS_DIR / "stage4_results.pkl", "rb") as f:
        s4 = pickle.load(f)
    return s1, s2, s3, s4


def load_returns():
    from src.stage1_data.download import run_download
    _, _, returns = run_download(force=False)
    return returns


def load_sector_map():
    from src.stage1_data.download import run_download
    info, _, _ = run_download(force=False)
    return {row["Symbol"]: row["GICS Sector"] for _, row in info.iterrows()}


def window_slice(returns, snap_label):
    for lab, start, end, regime in SNAPSHOTS:
        if lab == snap_label:
            mask = ((returns.index >= pd.Timestamp(start)) &
                    (returns.index <= pd.Timestamp(end)))
            return returns.loc[mask], regime
    raise KeyError(snap_label)


def main():
    t_total = time.time()
    print("[kmin-sens] loading canonical caches...")
    s1, s2_canon, s3_canon, s4_canon = load_canonical()
    print("[kmin-sens] loading returns + sector map...")
    returns = load_returns()
    sector_map = load_sector_map()

    print(f"\n[kmin-sens] re-running lambda grid sweep on "
          f"{len(FALLBACK_LABELS)} fallback snapshots...")
    traces = {}
    grid_meta = {}
    for lab in FALLBACK_LABELS:
        t0 = time.time()
        s1_snap = s1["snapshot_correlations"][lab]
        trace, meta = lambda_grid_sweep(s1_snap["R_avg"], s1_snap["n_days"])
        traces[lab] = trace
        grid_meta[lab] = meta
        print(f"  {lab:22s}: p={meta['p']}, n={meta['n']}, ratio={meta['ratio']:.3f}, "
              f"delta={meta['delta']:.3f}, grid_visited={meta['n_grid_visited']}/25, "
              f"failed={meta['n_failed']} | {time.time() - t0:.1f}s")

    print("\n[kmin-sens] computing lambda*_fallback per k_min...")
    selections = {}  # (kmin_key, label) -> chosen dict
    for k_key, k_ratio in KMIN_RATIOS.items():
        selections[k_key] = {}
        for lab in FALLBACK_LABELS:
            p = grid_meta[lab]["p"]
            k_min = int(k_ratio * p)
            chosen, status = pick_lambda(traces[lab], k_min)
            selections[k_key][lab] = {
                "k_min": int(k_min), "k_min_ratio": float(k_ratio),
                "lambda": float(chosen["lambda"]),
                "n_edges": int(chosen["n_edges"]),
                "ebic": float(chosen["ebic"]),
                "status": status,
                "precision": chosen["precision"],
            }
            print(f"  k_min={k_key:6s} ({k_min:>4d}) | {lab:22s} "
                  f"lambda*={chosen['lambda']:.4f}  k={chosen['n_edges']:>5d}  {status}")

    print("\n[kmin-sens] running Stage-3 on modified adjacencies "
          "(fallback snapshots only)...")
    s3_by_kmin = {k: {} for k in KMIN_RATIOS}
    for k_key in KMIN_RATIOS:
        for lab in FALLBACK_LABELS:
            chosen = selections[k_key][lab]
            adj = build_adjacency(chosen["precision"])
            tickers = s1["snapshot_correlations"][lab]["tickers"]
            R_avg = s1["snapshot_correlations"][lab]["R_avg"]
            returns_window, regime = window_slice(returns, lab)
            print(f"  Stage-3 [{k_key}] {lab}: {int(adj.sum()) // 2} input edges")
            t0 = time.time()
            res = run_stage3_one(adj, returns_window, tickers, R_avg)
            res["tickers"] = tickers
            res["regime"] = regime
            s3_by_kmin[k_key][lab] = res
            print(f"    done in {time.time() - t0:.1f}s | "
                  f"directed={res['n_directed']}, mutual={res['n_bidirectional']}")

    print("\n[kmin-sens] running Stage-4 on modified directed graphs...")
    s4_by_kmin = {k: dict(s4_canon) for k in KMIN_RATIOS}
    for k_key in KMIN_RATIOS:
        if k_key == "p":
            # Sanity: re-run on the same adjacency as canonical, expect
            # near-match (Louvain/motif stochastic seeding may shift slightly).
            pass
        for lab in FALLBACK_LABELS:
            t0 = time.time()
            s3_one = s3_by_kmin[k_key][lab]
            s4_one = run_stage4_one(
                s3_one["directed_adj"], s3_one["tickers"],
                s3_one["regime"], sector_map,
                n_motif_rewires=100, n_er_sims=200)
            if s4_one is None:
                # Graph too sparse; drop the snapshot from s4 (Stage-5 will skip)
                s4_by_kmin[k_key].pop(lab, None)
                print(f"    Stage-4 [{k_key}] {lab}: SKIPPED (too few edges)")
                continue
            s4_by_kmin[k_key][lab] = s4_one
            print(f"  Stage-4 [{k_key}] {lab}: |Z|={abs(s4_one['erdos_renyi']['z_scores']['clustering']):.1f}, "
                  f"Q={s4_one['community']['modularity']:.3f}, "
                  f"edges={s4_one['n_edges']} | {time.time() - t0:.1f}s")

    print("\n[kmin-sens] computing NSI per k_min...")
    nsi_by_kmin = {}
    channels_by_kmin = {}
    for k_key in KMIN_RATIOS:
        ch = compute_nsi_channels(s1["snapshot_correlations"],
                                   None, s4_by_kmin[k_key])
        nsi = normalise_and_nsi(ch)
        nsi_by_kmin[k_key] = nsi
        channels_by_kmin[k_key] = ch
        # Rank
        rank = sorted(nsi.items(), key=lambda kv: -kv[1])
        top2 = {rank[0][0], rank[1][0]}
        top2_crises = (top2 == {"Oct 2008 Peak", "Mar 2020 Peak"})
        mar = nsi.get("Mar 2020 Peak", float("nan"))
        twentytwo = nsi.get("2022 Rate Hikes", float("nan"))
        print(f"  k_min={k_key:6s} | top1={rank[0][0]:22s} ({rank[0][1]:.4f}) "
              f"top2_crises={top2_crises}  M20-2022={mar - twentytwo:+.4f}")

    print("\n[kmin-sens] writing outputs...")
    # Strip precision matrices for JSON
    out_selections = {
        k_key: {lab: {kk: v for kk, v in d.items() if kk != "precision"}
                for lab, d in by_lab.items()}
        for k_key, by_lab in selections.items()
    }
    # Strip non-serialisable from stage4 outputs
    def s4_summary(s4):
        out = {}
        for lab, v in s4.items():
            out[lab] = {
                "n_nodes": v["n_nodes"], "n_edges": v["n_edges"],
                "regime": v["regime"],
                "z_clustering": float(v["erdos_renyi"]["z_scores"]["clustering"]),
                "hhi": float(v["pagerank"]["hhi_top10"]),
                "gini": float(v["pagerank"]["gini"]),
                "modularity": float(v["community"]["modularity"]),
                "purity": float(v["community"]["purity"]),
                "z_ffl": (float(v["motifs"]["z_scores"]["feed_forward_loop"])
                          if v["motifs"] else 0.0),
                "z_mr":  (float(v["motifs"]["z_scores"]["mutual_regulation"])
                          if v["motifs"] else 0.0),
                "z_sim": (float(v["motifs"]["z_scores"]["single_input_module"])
                          if v["motifs"] else 0.0),
            }
        return out

    payload = {
        "grid_meta": grid_meta,
        "selections": out_selections,
        "channels": {k: {lab: list(ch) for lab, ch in by_lab.items()}
                     for k, by_lab in channels_by_kmin.items()},
        "nsi_by_kmin": nsi_by_kmin,
        "stage4_summary_by_kmin": {k: s4_summary(v) for k, v in s4_by_kmin.items()},
        "fallback_labels": FALLBACK_LABELS,
        "kmin_ratios": KMIN_RATIOS,
        "nsi_weights": NSI_WEIGHTS.tolist(),
    }
    (OUT_DIR / "results.json").write_text(json.dumps(payload, indent=2))
    print(f"  wrote {OUT_DIR / 'results.json'}")

    # Quick TSV: snapshot x k_min NSI table
    labels_order = [s[0] for s in SNAPSHOTS]
    with open(OUT_DIR / "per_kmin_nsi.tsv", "w") as f:
        f.write("snapshot\tregime\t" + "\t".join(KMIN_RATIOS.keys()) + "\n")
        for lab in labels_order:
            regime = next(s[3] for s in SNAPSHOTS if s[0] == lab)
            cells = [f"{nsi_by_kmin[k].get(lab, float('nan')):.4f}" for k in KMIN_RATIOS]
            f.write(f"{lab}\t{regime}\t" + "\t".join(cells) + "\n")
    print(f"  wrote {OUT_DIR / 'per_kmin_nsi.tsv'}")

    print(f"\n[kmin-sens] DONE in {(time.time() - t_total)/60:.1f} min")


if __name__ == "__main__":
    main()
