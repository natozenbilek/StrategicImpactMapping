"""Stage 4: network-level analysis on the Stage-3 directed graph.

Four mesoscale questions per snapshot:

    Q1  Non-randomness: ER G(n,m) clustering Z (|Z|>1.96 rejects null).
    Q2  Leadership:     PageRank concentration (Gini + full-network HHI).
    Q3  Communities:    Louvain modularity + GICS sector purity.
    Q4  Motif patterns: MAN triadic census (FFL/MR/SIM) vs dyad-preserving null.

Each per-snapshot result is pickled to SNAPSHOTS_DIR/stage4_results.pkl
and consumed by Stage 5 (NSI) and the robustness modules.

Cache key layout on ``pagerank``: ``hhi_top10`` is the genuine top-10
normalised Herfindahl (NSI hub-concentration channel, in [0.1, 1]);
``hhi_full`` is the full-network Herfindahl over every node (paper
Tab. tab:q1q2 column, in [1/p, 1]); ``hhi`` is an alias for
``hhi_top10`` kept for backward compatibility with the NSI driver.
"""
import json
import pickle
from collections import Counter

import numpy as np
import pandas as pd
import networkx as nx
import community as community_louvain

from src.config import (
    SNAPSHOTS_DIR, PAGERANK_DAMPING, ERDOS_RENYI_N_SIMS,
    LOUVAIN_RESOLUTION, MOTIF_N_REWIRES,
)


# --- Q1: ER non-randomness --------------------------------------------

def erdos_renyi_test(G, n_sims=ERDOS_RENYI_N_SIMS, seed=2026):
    """Test G against an ER G(n,m) null on the undirected projection.

    Both empirical and null operate on the symmetrised graph: m is the
    undirected-projection edge count of G, and the null is drawn with
    ``directed=False``. Reports Z-scores for clustering and path length
    plus is_non_random = |Z_C| > 1.96 (two-sided 5%).
    """
    n = G.number_of_nodes()
    G_undir = G.to_undirected()
    m_undir = G_undir.number_of_edges()
    assert n >= 3, f"ER test requires n >= 3, got {n}"

    emp_clustering = nx.average_clustering(G_undir)
    try:
        if nx.is_connected(G_undir):
            emp_path_len = nx.average_shortest_path_length(G_undir)
        else:
            lcc = max(nx.connected_components(G_undir), key=len)
            emp_path_len = nx.average_shortest_path_length(G_undir.subgraph(lcc))
    except Exception:
        emp_path_len = np.nan

    emp_mean_degree = np.mean([d for _, d in G.degree()])

    rng = np.random.RandomState(seed)
    null_clustering, null_path_len, null_mean_degree = [], [], []
    for _ in range(n_sims):
        sub_seed = int(rng.randint(0, 2 ** 31 - 1))
        G_rand = nx.gnm_random_graph(n, m_undir, seed=sub_seed)
        null_clustering.append(nx.average_clustering(G_rand))
        try:
            if nx.is_connected(G_rand):
                null_path_len.append(nx.average_shortest_path_length(G_rand))
            else:
                lcc = max(nx.connected_components(G_rand), key=len)
                null_path_len.append(
                    nx.average_shortest_path_length(G_rand.subgraph(lcc)))
        except Exception:
            pass
        null_mean_degree.append(2 * G_rand.number_of_edges() / n)

    def z_score(emp, null_vals):
        arr = np.asarray(null_vals, dtype=float)
        sigma = float(arr.std())
        if sigma == 0:
            # Degenerate ER null (e.g. all draws produce a graph with the
            # same clustering coefficient) — happens on extremely sparse
            # bottom-N panels. Return NaN; downstream consumers filter.
            return float("nan")
        return (emp - float(arr.mean())) / sigma

    z_c = z_score(emp_clustering, null_clustering)
    z_l = z_score(emp_path_len, null_path_len) if null_path_len else np.nan

    return {
        "empirical": {
            "clustering": emp_clustering,
            "path_length": emp_path_len,
            "mean_degree": emp_mean_degree,
        },
        "null_mean": {
            "clustering": float(np.mean(null_clustering)),
            "path_length": (float(np.mean(null_path_len))
                            if null_path_len else np.nan),
            "mean_degree": float(np.mean(null_mean_degree)),
        },
        "z_scores": {"clustering": z_c, "path_length": z_l},
        "is_non_random": bool(np.isfinite(z_c) and abs(z_c) > 1.96),
    }


# --- Q2: PageRank concentration ---------------------------------------

def pagerank_analysis(G, tickers, damping=PAGERANK_DAMPING, top_k=20):
    """PageRank + top-k leaders + top-10 HHI + full-network HHI + Gini.

    PageRank uses networkx's default `weight='weight'`; build_nx_graph
    sets edge weights to the Stage-2 |partial correlation| magnitude.
    `hhi_top10` re-normalises the top-10 PageRank scores so they sum
    to 1 within the sub-sample, then computes Herfindahl: it lives in
    [1/10, 1] and is robust to single-node spikes whose impact would
    dominate the full-network Herfindahl. `hhi_full` keeps the prior
    sum-over-all-nodes HHI for backward comparison.
    """
    pr = nx.pagerank(G, alpha=damping)
    p = G.number_of_nodes()
    assert abs(sum(pr.values()) - 1.0) < 1e-6, "PageRank does not sum to 1"
    sorted_pr = sorted(pr.items(), key=lambda x: x[1], reverse=True)
    top_nodes = [
        {"rank": i + 1,
         "ticker": tickers[node_idx] if node_idx < len(tickers) else str(node_idx),
         "pagerank": score}
        for i, (node_idx, score) in enumerate(sorted_pr[:top_k])
    ]
    top10_scores = np.array([s for _, s in sorted_pr[:10]], dtype=float)
    top10_sum = float(top10_scores.sum())
    hhi_top10 = float(np.sum((top10_scores / top10_sum) ** 2)) if top10_sum > 0 else 0.0
    hhi_full = sum(s ** 2 for s in pr.values())
    assert 0.1 - 1e-9 <= hhi_top10 <= 1.0 + 1e-9, f"hhi_top10={hhi_top10} out of [0.1, 1]"
    assert 1.0 / p - 1e-9 <= hhi_full <= 1.0 + 1e-9, f"hhi_full={hhi_full} out of [1/p, 1]"
    gini = _gini_coefficient(list(pr.values()))
    assert 0.0 <= gini <= 1.0 + 1e-9, f"Gini={gini} out of [0, 1]"
    return {
        "pagerank_scores": pr,
        "top_nodes": top_nodes,
        "hhi_top10": hhi_top10,  # top-10 normalised concentration (NSI channel)
        "hhi_full": hhi_full,    # full-network Herfindahl (legacy / paper Tab q1q2)
        "hhi": hhi_top10,        # alias = canonical NSI input
        "gini": gini,
    }


def _gini_coefficient(values):
    """Sorted-rank Gini of a non-negative score vector; 0 on empty/all-zero."""
    arr = np.sort(np.array(values))
    n = len(arr)
    if n == 0 or arr.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * arr) / (n * np.sum(arr))) - (n + 1) / n


# --- Q3: Louvain communities + GICS purity ----------------------------

def community_analysis(G, tickers, sector_map=None, resolution=LOUVAIN_RESOLUTION):
    """Louvain modularity on G.to_undirected() + sector purity vs GICS.

    Modularity uses python-louvain's default `weight='weight'`; weights
    are the |partial correlation| magnitudes carried over from Stage 2.
    Purity normalises by the count of nodes whose tickers exist in
    sector_map (not by the partition size), so unclassified tickers
    do not depress the score.
    """
    G_undir = G.to_undirected()
    partition = community_louvain.best_partition(
        G_undir, resolution=resolution, random_state=42)
    modularity = community_louvain.modularity(partition, G_undir)
    assert -1.0 <= modularity <= 1.0 + 1e-9, f"Q={modularity} out of [-1, 1]"
    comm_labels = set(partition.values())
    comm_sizes = Counter(partition.values())

    purity = np.nan
    community_sectors = {}
    if sector_map:
        n_classified = sum(1 for node in partition
                           if node < len(tickers)
                           and tickers[node] in sector_map)
        correct = 0
        for comm in comm_labels:
            members = [node for node, c in partition.items() if c == comm]
            sector_counts = Counter()
            for node in members:
                t = tickers[node] if node < len(tickers) else None
                if t and t in sector_map:
                    sector_counts[sector_map[t]] += 1
            if sector_counts:
                dom = sector_counts.most_common(1)[0]
                correct += dom[1]
                community_sectors[comm] = {
                    "dominant_sector": dom[0], "size": len(members),
                    "sector_distribution": dict(sector_counts),
                }
        purity = correct / n_classified if n_classified > 0 else 0.0
        assert 0.0 <= purity <= 1.0 + 1e-9, f"Purity={purity} out of [0, 1]"

    return {
        "partition": partition,
        "modularity": modularity,
        "n_communities": len(comm_labels),
        "community_sizes": dict(comm_sizes),
        "purity": purity,
        "community_sectors": community_sectors,
    }


# --- Q4: MAN triadic motifs -------------------------------------------

def count_triadic_motifs(G):
    """Extract three named MAN classes from networkx.triadic_census.

        FFL  = 030T  (A->B, B->C, A->C)
        MR   = 111D  (A<->B, C->A) -- networkx "downstream" convention:
                                      asymmetric arc points INTO the
                                      mutual pair.
        SIM  = 021D  (A->B, A->C)
    """
    census = nx.triadic_census(G)
    return {
        "feed_forward_loop":   int(census.get("030T", 0)),
        "mutual_regulation":   int(census.get("111D", 0)),
        "single_input_module": int(census.get("021D", 0)),
    }


def _dyad_preserving_rewire(G, n_swaps_per_class=None, rng=None):
    """Maslov-Sneppen rewire that also preserves the mutual-dyad count.

    Vanilla in/out-degree-preserving rewires break mutual pairs and
    inflate single-edge triads in the null, biasing Z-scores of 021D
    and 030T negative on any graph with appreciable bidirectional
    density. We partition edges into asymmetric and mutual sets, rewire
    each set independently, and reject swaps that would change the
    mutual-pair count.

    Asymmetric swap (u->v),(x->y) -> (u->y),(x->v): rejected if either
    target arc already exists OR if either target's reverse exists
    (would create a new mutual dyad).

    Mutual swap {u<->v},{x<->y} -> {u<->y},{x<->v}: requires all four
    target directed arcs absent and four distinct endpoints.

    Swap budget is 20x partition size; failure to reach the target
    leaves the null closer to empirical wiring (conservative).
    """
    if rng is None:
        rng = np.random.RandomState()
    H = G.copy()

    mutual_pairs, asym_edges = [], []
    seen_mutuals = set()
    for u, v in H.edges():
        if H.has_edge(v, u):
            key = (min(u, v), max(u, v))
            if key not in seen_mutuals:
                seen_mutuals.add(key)
                mutual_pairs.append((u, v))
        else:
            asym_edges.append((u, v))

    n_asym, n_mut = len(asym_edges), len(mutual_pairs)
    if n_swaps_per_class is None:
        n_swaps_asym, n_swaps_mut = n_asym, n_mut
    else:
        n_swaps_asym = n_swaps_mut = n_swaps_per_class

    # Asymmetric rewires.
    max_tries = max(n_swaps_asym * 20, 50)
    successful = tries = 0
    while n_asym >= 2 and successful < n_swaps_asym and tries < max_tries:
        tries += 1
        i, j = rng.randint(0, n_asym, size=2)
        if i == j:
            continue
        u, v = asym_edges[i]
        x, y = asym_edges[j]
        if u in (x, y) or v in (x, y):
            continue
        if H.has_edge(u, y) or H.has_edge(x, v):
            continue
        if H.has_edge(y, u) or H.has_edge(v, x):
            continue  # would create a new mutual dyad
        H.remove_edge(u, v); H.remove_edge(x, y)
        H.add_edge(u, y); H.add_edge(x, v)
        asym_edges[i] = (u, y); asym_edges[j] = (x, v)
        successful += 1

    # Mutual rewires.
    max_tries = max(n_swaps_mut * 20, 50)
    successful = tries = 0
    while n_mut >= 2 and successful < n_swaps_mut and tries < max_tries:
        tries += 1
        i, j = rng.randint(0, n_mut, size=2)
        if i == j:
            continue
        u, v = mutual_pairs[i]
        x, y = mutual_pairs[j]
        if len({u, v, x, y}) < 4:
            continue
        if (H.has_edge(u, y) or H.has_edge(y, u)
                or H.has_edge(x, v) or H.has_edge(v, x)):
            continue
        H.remove_edge(u, v); H.remove_edge(v, u)
        H.remove_edge(x, y); H.remove_edge(y, x)
        H.add_edge(u, y); H.add_edge(y, u)
        H.add_edge(x, v); H.add_edge(v, x)
        mutual_pairs[i] = (u, y); mutual_pairs[j] = (x, v)
        successful += 1

    return H


def motif_significance(G, n_rewires=MOTIF_N_REWIRES, seed=42):
    """Z-scores + Milo significance profile vs the dyad-preserving null.

        Z_i = (N_i - mean N^null_i) / std N^null_i
        SP_i = Z_i / sqrt(sum_j Z_j^2)
    """
    emp_counts = count_triadic_motifs(G)
    null_counts = {k: [] for k in emp_counts}
    rng = np.random.RandomState(seed)

    for i in range(n_rewires):
        G_rand = _dyad_preserving_rewire(G, rng=rng)
        counts = count_triadic_motifs(G_rand)
        for k, v in counts.items():
            null_counts[k].append(v)
        if (i + 1) % 25 == 0:
            print(f"      Null model {i + 1}/{n_rewires}...")

    z_scores = {}
    degenerate = []
    for motif_name in emp_counts:
        arr = np.asarray(null_counts[motif_name], dtype=float)
        mu = float(arr.mean())
        sigma = float(arr.std())
        if sigma == 0:
            # Bottom-N / low-density panels can collapse the rewire null
            # to a single configuration (e.g. graph too sparse for the
            # dyad-preserving rewire to find any swap). Record a NaN Z
            # rather than abort the whole panel; downstream consumers
            # filter NaN.
            z_scores[motif_name] = float("nan")
            degenerate.append(motif_name)
        else:
            z_scores[motif_name] = (emp_counts[motif_name] - mu) / sigma
    if degenerate:
        print(f"      [warn] degenerate rewire null (sigma=0): "
              f"{', '.join(degenerate)} -- Z set to NaN")

    z_vec = np.array([v for v in z_scores.values() if not np.isnan(v)])
    if z_vec.size > 0:
        norm = float(np.sqrt(np.sum(z_vec ** 2)))
    else:
        norm = 0.0
    motif_names = list(z_scores.keys())
    if norm > 0:
        sp_vec = np.array(list(z_scores.values())) / norm
    else:
        sp_vec = np.full(len(motif_names), float("nan"))
    significance_profile = {motif_names[i]: float(sp_vec[i])
                            for i in range(len(motif_names))}

    return {
        "empirical_counts": emp_counts,
        "z_scores": z_scores,
        "significance_profile": significance_profile,
    }


# --- Driver -----------------------------------------------------------

def build_nx_graph(directed_adj, tickers):
    """DiGraph from a (p, p) directed adjacency matrix; weight=|adj_ij|."""
    p = directed_adj.shape[0]
    assert directed_adj.shape == (p, p), f"expected square, got {directed_adj.shape}"
    assert p == len(tickers), f"ticker count {len(tickers)} != p={p}"
    G = nx.DiGraph()
    G.add_nodes_from(range(p))
    for i in range(p):
        for j in range(p):
            if directed_adj[i, j] > 0:
                G.add_edge(i, j, weight=float(directed_adj[i, j]))
    return G


def run_stage4(input_results, sp500_info=None, run_motifs=True,
               n_motif_rewires=100, n_er_sims=200, force=False):
    """Q1-Q4 per snapshot.

    Cache is invalidated if the parameter tuple
    (run_motifs, n_motif_rewires, n_er_sims) differs from the sidecar
    meta JSON written alongside the pickle.
    """
    cache_path = SNAPSHOTS_DIR / "stage4_results.pkl"
    meta_path = SNAPSHOTS_DIR / "stage4_results.meta.json"
    params = {
        "run_motifs": run_motifs,
        "n_motif_rewires": n_motif_rewires,
        "n_er_sims": n_er_sims,
    }
    if not force and cache_path.exists() and meta_path.exists():
        with open(meta_path) as f:
            cached_params = json.load(f)
        if cached_params == params:
            print("[Stage 4] Loading cached results...")
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        print(f"[Stage 4] Cache params changed "
              f"({cached_params} != {params}); re-running.")

    sector_map = {}
    if sp500_info is not None:
        for _, row in sp500_info.iterrows():
            sector_map[row["Symbol"]] = row["GICS Sector"]

    print("[Stage 4] Network Analysis and Strategic Characterization...")
    results = {}

    for label, data in input_results.items():
        tickers = data["tickers"]
        regime = data["regime"]
        print(f"\n  === Snapshot: {label} ({regime}) ===")

        directed_adj = data["directed_adj"]
        G = build_nx_graph(directed_adj, tickers)

        n_nodes, n_edges = G.number_of_nodes(), G.number_of_edges()
        print(f"    Graph: {n_nodes} nodes, {n_edges} edges")

        print(f"    Q1: Erdos-Renyi test ({n_er_sims} sims)...")
        er_result = erdos_renyi_test(G, n_sims=n_er_sims)
        print(f"      Clustering Z={er_result['z_scores']['clustering']:.2f}, "
              f"Non-random: {er_result['is_non_random']}")

        print(f"    Q2: PageRank analysis...")
        pr_result = pagerank_analysis(G, tickers)
        top3 = [(n["ticker"], f"{n['pagerank']:.4f}") for n in pr_result["top_nodes"][:3]]
        print(f"      Top-3: {top3}")
        print(f"      HHI={pr_result['hhi']:.4f}, Gini={pr_result['gini']:.4f}")

        print(f"    Q3: Louvain community detection...")
        comm_result = community_analysis(G, tickers, sector_map)
        print(f"      Communities: {comm_result['n_communities']}, "
              f"Q={comm_result['modularity']:.4f}, "
              f"Purity={comm_result['purity']:.4f}")

        motif_result = None
        if run_motifs:
            print(f"    Q4: Motif analysis ({n_motif_rewires} rewires)...")
            motif_result = motif_significance(G, n_rewires=n_motif_rewires)
            print(f"      Z-scores: {motif_result['z_scores']}")

        results[label] = {
            "graph": G,
            "erdos_renyi": er_result,
            "pagerank": pr_result,
            "community": comm_result,
            "motifs": motif_result,
            "tickers": tickers,
            "regime": regime,
            "n_nodes": n_nodes,
            "n_edges": n_edges,
        }

    with open(cache_path, "wb") as f:
        pickle.dump(results, f)
    with open(meta_path, "w") as f:
        json.dump(params, f)
    print(f"\n  Cached to {cache_path}")
    return results


if __name__ == "__main__":
    stage3_path = SNAPSHOTS_DIR / "stage3_results.pkl"
    if stage3_path.exists():
        with open(stage3_path, "rb") as f:
            stage3 = pickle.load(f)
        results = run_stage4(stage3, run_motifs=False)
        print("\n[Stage 4] Complete!")
    else:
        print("Run Stage 3 first.")
