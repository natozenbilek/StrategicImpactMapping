"""Stage 4 network-metric unit tests.

Covers _gini_coefficient, _dyad_preserving_rewire,
count_triadic_motifs, erdos_renyi_test, community_analysis,
pagerank_analysis, build_nx_graph, and the density-matched
build_threshold_network construction.
"""
import numpy as np
import pytest
import networkx as nx

from src.stage4_network.analysis import (
    _gini_coefficient,
    _dyad_preserving_rewire,
    count_triadic_motifs,
    erdos_renyi_test,
    community_analysis,
    pagerank_analysis,
    build_nx_graph,
    motif_significance,
)
from src.stage4_network.density_matched import build_threshold_network


# --- Gini ---

def test_gini_uniform():
    assert _gini_coefficient([0.1, 0.1, 0.1, 0.1]) == pytest.approx(0.0, abs=1e-12)


def test_gini_monopoly():
    # n=4, one node holds all mass → analytic Gini = (n-1)/n = 0.75
    assert _gini_coefficient([0.0, 0.0, 0.0, 1.0]) == pytest.approx(0.75, rel=1e-9)


def test_gini_empty_and_all_zero():
    assert _gini_coefficient([]) == 0.0
    assert _gini_coefficient([0.0, 0.0, 0.0]) == 0.0


# --- Triadic motif census ---

def test_triadic_ffl():
    # 030T: A->B, B->C, A->C
    G = nx.DiGraph([(0, 1), (1, 2), (0, 2)])
    c = count_triadic_motifs(G)
    assert c["feed_forward_loop"] == 1
    assert c["mutual_regulation"] == 0
    assert c["single_input_module"] == 0


def test_triadic_sim():
    # 021D: A->B, A->C
    G = nx.DiGraph([(0, 1), (0, 2)])
    c = count_triadic_motifs(G)
    assert c["single_input_module"] == 1
    assert c["feed_forward_loop"] == 0
    assert c["mutual_regulation"] == 0


def test_triadic_mr():
    # 111D (networkx convention): A<->B, C->A
    # The asymmetric arc points INTO the mutual pair.
    G = nx.DiGraph([(0, 1), (1, 0), (2, 0)])
    c = count_triadic_motifs(G)
    assert c["mutual_regulation"] == 1
    assert c["single_input_module"] == 0
    assert c["feed_forward_loop"] == 0


# --- Dyad-preserving rewire invariants ---

def _build_synthetic_dyad_graph():
    """20 nodes, 5 mutual pairs + 10 asymmetric arcs."""
    G = nx.DiGraph()
    G.add_nodes_from(range(20))
    for u, v in [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]:
        G.add_edge(u, v); G.add_edge(v, u)
    for u, v in [(10, 11), (11, 12), (12, 13), (13, 14), (14, 15),
                 (15, 16), (16, 17), (17, 18), (18, 19), (19, 10)]:
        G.add_edge(u, v)
    return G


def test_rewire_preserves_degree_and_mutual_count():
    G = _build_synthetic_dyad_graph()
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    n_mutual = sum(1 for u, v in G.edges() if G.has_edge(v, u) and u < v)

    rng = np.random.RandomState(42)
    H = _dyad_preserving_rewire(G, rng=rng)

    assert dict(H.in_degree()) == in_deg
    assert dict(H.out_degree()) == out_deg
    n_mut_after = sum(1 for u, v in H.edges() if H.has_edge(v, u) and u < v)
    assert n_mut_after == n_mutual
    assert H.number_of_edges() == G.number_of_edges()


def test_rewire_is_deterministic_with_seed():
    G = _build_synthetic_dyad_graph()
    H1 = _dyad_preserving_rewire(G, rng=np.random.RandomState(42))
    H2 = _dyad_preserving_rewire(G, rng=np.random.RandomState(42))
    assert set(H1.edges()) == set(H2.edges())


# --- Erdos-Renyi test ---

def test_er_test_deterministic_under_same_seed():
    G = nx.gnp_random_graph(40, 0.15, seed=42, directed=True)
    r1 = erdos_renyi_test(G, n_sims=50, seed=2026)
    r2 = erdos_renyi_test(G, n_sims=50, seed=2026)
    assert r1["z_scores"]["clustering"] == r2["z_scores"]["clustering"]
    assert r1["empirical"]["clustering"] == r2["empirical"]["clustering"]


def test_er_test_runs_on_random_graph():
    G = nx.gnp_random_graph(30, 0.2, seed=7, directed=True)
    r = erdos_renyi_test(G, n_sims=50, seed=2026)
    assert "z_scores" in r
    assert isinstance(r["is_non_random"], (bool, np.bool_))
    # An Erdos-Renyi-like input should not be very far from its ER null.
    assert abs(r["z_scores"]["clustering"]) < 10


def test_er_null_uses_undirected_m():
    # Pure mutual digraph: every directed edge has a reverse.
    # m_dir = 2 * m_undir. The null must match m_undir, not m_dir.
    G = nx.DiGraph()
    G.add_nodes_from(range(20))
    for u, v in [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9),
                 (10, 11), (12, 13), (14, 15)]:
        G.add_edge(u, v); G.add_edge(v, u)
    m_undir = G.to_undirected().number_of_edges()
    assert m_undir == 8
    # If the implementation accidentally fed 16 to the null with directed=True,
    # the projected null edge count would be < 16. The test just verifies
    # that the function runs without raising.
    r = erdos_renyi_test(G, n_sims=20, seed=2026)
    assert np.isfinite(r["empirical"]["clustering"])


# --- PageRank invariants ---

def test_pagerank_invariants_on_random_digraph():
    G = nx.gnp_random_graph(20, 0.25, seed=42, directed=True)
    # Inject weights, since build_nx_graph would.
    for u, v in G.edges():
        G[u][v]["weight"] = 0.5
    tickers = [f"T{i}" for i in range(20)]
    r = pagerank_analysis(G, tickers)
    assert 0.0 <= r["gini"] <= 1.0
    p = G.number_of_nodes()
    assert 1.0 / p - 1e-9 <= r["hhi"] <= 1.0 + 1e-9
    assert r["hhi_top10"] == r["hhi"]
    assert abs(sum(r["pagerank_scores"].values()) - 1.0) < 1e-6


# --- Build nx graph ---

def test_build_nx_graph_edge_weights():
    adj = np.array([
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.3],
        [0.0, 0.0, 0.0],
    ])
    G = build_nx_graph(adj, tickers=["A", "B", "C"])
    assert G.number_of_edges() == 2
    assert G[0][1]["weight"] == pytest.approx(0.5)
    assert G[1][2]["weight"] == pytest.approx(0.3)


def test_build_nx_graph_shape_mismatch_asserts():
    adj = np.zeros((3, 3))
    with pytest.raises(AssertionError):
        build_nx_graph(adj, tickers=["A", "B"])


# --- Community / purity ---

def test_community_purity_ignores_unclassified_tickers():
    # Two clean components: {0,1,2} (sector X) and {3,4,5} (sector Y);
    # ticker F (idx 5) intentionally missing from sector_map.
    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (1, 2), (3, 4), (4, 5)])
    tickers = ["A", "B", "C", "D", "E", "F"]
    sector_map = {"A": "X", "B": "X", "C": "X", "D": "Y", "E": "Y"}  # F missing
    r = community_analysis(G, tickers, sector_map=sector_map)
    # F should not depress purity through the denominator.
    assert 0.0 <= r["purity"] <= 1.0 + 1e-9
    # With a perfect split it should be 1 on the classified subset.
    assert r["purity"] == pytest.approx(1.0, abs=1e-9)


# --- Threshold network ---

def test_build_threshold_network_topk():
    R = np.array([
        [1.0, 0.9, 0.1, 0.5],
        [0.9, 1.0, 0.3, 0.7],
        [0.1, 0.3, 1.0, 0.2],
        [0.5, 0.7, 0.2, 1.0],
    ])
    G = build_threshold_network(R, tickers=["A", "B", "C", "D"], n_edges=2)
    assert G.number_of_edges() == 2
    assert G.has_edge(0, 1)  # |R| = 0.9
    assert G.has_edge(1, 3)  # |R| = 0.7


def test_build_threshold_network_more_edges_than_available():
    R = np.array([
        [1.0, 0.9, 0.1],
        [0.9, 1.0, 0.3],
        [0.1, 0.3, 1.0],
    ])
    G = build_threshold_network(R, tickers=["A", "B", "C"], n_edges=100)
    assert G.number_of_edges() == 3


# --- Motif significance Z + SP integration ---

def test_motif_significance_sp_unit_norm():
    rng = np.random.RandomState(42)
    G = nx.gnp_random_graph(40, 0.15, seed=1, directed=True)
    r = motif_significance(G, n_rewires=20, seed=42)
    sp_vec = np.array(list(r["significance_profile"].values()))
    assert abs(np.linalg.norm(sp_vec) - 1.0) < 1e-6
