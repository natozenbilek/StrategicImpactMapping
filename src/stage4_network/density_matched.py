"""Density-matched community analysis for Stage-4 Q3 robustness.

Stage-2 GLASSO produces snapshots with very different edge densities,
and Newman-Girvan modularity is *not* density-invariant: sparser
graphs mechanically achieve larger Q because k_i k_j / 2m shrinks
faster than the realised intra-community mass. To remove the confound
we threshold every snapshot at a shared edge budget K (top-K |R_avg|,
Mantegna 1999) and report Louvain Q + sector purity + the relative
modularity

    Q_rel = (Q - E[Q_rand]) / E[Q_rand],

with E[Q_rand] estimated from Erdos-Renyi G(n, m) draws at matched
density. Q_rel is the cleanest density-invariant test of the
community-dissolution hypothesis.
"""
from pathlib import Path
import json
import pickle
from collections import Counter

import numpy as np
import pandas as pd
import networkx as nx
import community as community_louvain

from src.config import SNAPSHOTS_DIR, LOUVAIN_RESOLUTION


def build_threshold_network(corr_matrix, tickers, n_edges):
    """Top-K threshold network on |R| with at most ``n_edges`` edges.

    Uses np.partition for the K-th order statistic (O(p^2)); ties at
    the threshold fall through to an explicit argsort so the realised
    edge count never exceeds K.
    """
    p = corr_matrix.shape[0]
    triu_i, triu_j = np.triu_indices(p, k=1)
    abs_corrs = np.abs(corr_matrix[triu_i, triu_j])

    if n_edges >= len(abs_corrs):
        keep = np.ones(len(abs_corrs), dtype=bool)
    else:
        threshold = np.partition(abs_corrs, -n_edges)[-n_edges]
        keep = abs_corrs >= threshold
        if keep.sum() > n_edges:
            order = np.argsort(-abs_corrs)
            keep = np.zeros(len(abs_corrs), dtype=bool)
            keep[order[:n_edges]] = True

    G = nx.Graph()
    G.add_nodes_from(range(p))
    for k in np.where(keep)[0]:
        G.add_edge(int(triu_i[k]), int(triu_j[k]), weight=float(abs_corrs[k]))
    return G


def louvain_Q_purity(G, tickers, sector_map,
                     resolution=LOUVAIN_RESOLUTION, seed=42):
    """Louvain modularity + GICS sector purity on G."""
    partition = community_louvain.best_partition(G, resolution=resolution,
                                                  random_state=seed)
    Q = community_louvain.modularity(partition, G)
    comm_labels = set(partition.values())
    n = len(partition)
    correct = 0
    for comm in comm_labels:
        sector_counts = Counter()
        for node, c in partition.items():
            if c != comm:
                continue
            t = tickers[node] if node < len(tickers) else None
            if t and t in sector_map:
                sector_counts[sector_map[t]] += 1
        if sector_counts:
            correct += sector_counts.most_common(1)[0][1]
    return Q, (correct / n if n > 0 else 0.0), len(comm_labels)


def expected_Q_random(n, m, n_trials=50,
                      resolution=LOUVAIN_RESOLUTION, seed=42):
    """Mean Louvain Q over n_trials ER G(n, m) draws."""
    rng = np.random.RandomState(seed)
    Qs = []
    for k in range(n_trials):
        G_rand = nx.gnm_random_graph(n, m, seed=int(rng.randint(0, 2 ** 31)))
        if G_rand.number_of_edges() == 0:
            continue
        partition = community_louvain.best_partition(
            G_rand, resolution=resolution, random_state=seed + k)
        Qs.append(community_louvain.modularity(partition, G_rand))
    return float(np.mean(Qs)) if Qs else np.nan


def run_density_matched_analysis(
    stage1_results=None, stage2_results=None, sp500_info=None,
    target_density=None, n_random_trials=50, cache=True, force=False,
):
    """Density-matched Q3 robustness across all snapshots.

    Default K = min Stage-2 edge count (truncate to sparsest crisis
    density). target_density overrides this and sets K from p directly.
    """
    cache_path = SNAPSHOTS_DIR / "density_matched_results.pkl"
    meta_path = SNAPSHOTS_DIR / "density_matched_results.meta.json"
    params = {"target_density": target_density,
              "n_random_trials": n_random_trials}
    if cache and not force and cache_path.exists() and meta_path.exists():
        with open(meta_path) as f:
            cached_params = json.load(f)
        if cached_params == params:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        print(f"[density-matched] cache params changed "
              f"({cached_params} != {params}); re-running.")

    if stage1_results is None:
        with open(SNAPSHOTS_DIR / "stage1_results.pkl", "rb") as f:
            stage1_results = pickle.load(f)
    if stage2_results is None:
        with open(SNAPSHOTS_DIR / "stage2_results.pkl", "rb") as f:
            stage2_results = pickle.load(f)
    if sp500_info is None:
        root = Path(__file__).resolve().parent.parent.parent
        sp500_info = pd.read_parquet(root / "data" / "sp500_info.parquet")

    sector_map = dict(zip(sp500_info["Symbol"], sp500_info["GICS Sector"]))
    snap_corrs = stage1_results["snapshot_correlations"]

    n_assets = next(iter(snap_corrs.values()))["R_avg"].shape[0]
    max_possible_edges = n_assets * (n_assets - 1) // 2
    if target_density is None:
        target_edges = min(stage2_results[lbl]["n_edges"] for lbl in stage2_results)
    else:
        target_edges = int(target_density * max_possible_edges)

    print(f"[Density-matched Q3] Target edges = {target_edges} "
          f"({100 * target_edges / max_possible_edges:.2f}% density)")

    rows = []
    for label in snap_corrs:
        R = snap_corrs[label]["R_avg"]
        tickers = snap_corrs[label]["tickers"]
        regime = snap_corrs[label]["regime"]
        G = build_threshold_network(R, tickers, n_edges=target_edges)
        Q, purity, n_comm = louvain_Q_purity(G, tickers, sector_map)
        n = G.number_of_nodes(); m = G.number_of_edges()
        Q_rand = expected_Q_random(n, m, n_trials=n_random_trials)
        Q_rel = (Q - Q_rand) / Q_rand if Q_rand > 1e-6 else np.nan
        rows.append({
            "snapshot": label, "regime": regime, "n_edges": m,
            "Q": Q, "Q_random": Q_rand, "Q_rel": Q_rel,
            "purity": purity, "n_communities": n_comm,
        })
        print(f"  {label:25s} ({regime:8s}): Q={Q:.4f}, "
              f"Q_rand={Q_rand:.4f}, Q_rel={Q_rel:+.3f}, "
              f"purity={purity:.2f}, #comm={n_comm}")

    df = pd.DataFrame(rows)
    if cache:
        with open(cache_path, "wb") as f:
            pickle.dump(df, f)
        with open(meta_path, "w") as f:
            json.dump(params, f)
        print(f"\n  Cached to {cache_path}")
    return df


if __name__ == "__main__":
    df = run_density_matched_analysis(force=True)
    print("\n" + "=" * 70)
    print("DENSITY-MATCHED COMMUNITY ANALYSIS (Q3 ROBUSTNESS)")
    print("=" * 70)
    print(df.to_string(index=False,
                       formatters={"Q": "{:.4f}".format,
                                   "Q_random": "{:.4f}".format,
                                   "Q_rel": "{:+.3f}".format,
                                   "purity": "{:.2f}".format}))
    print("\n--- Group means ---")
    print(df.groupby("regime")[["Q", "Q_rel", "purity"]].mean().to_string(
        formatters={"Q": "{:.4f}".format,
                    "Q_rel": "{:+.3f}".format,
                    "purity": "{:.3f}".format}))
