"""Stage 5 NSI unit + smoke tests.

Synthetic units exercise compute_nsi_components, compute_rolling_nsi,
compute_volume_weighted_nsi, _log_volume_weights, and the cluster-
bootstrap partition invariants. Smoke tests run against the cached
Stage-1/4 pickles when present.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.stage5_nsi import stress_index as si
from src.stage5_nsi.stress_index import (
    NSI_WEIGHTS_4CH, ROLLING_PRECISION_EPS, ROLLING_PSD_FLOOR,
    compute_nsi_components, compute_rolling_nsi,
    backtest_rolling_nsi_vs_vix,
)
from src.stage5_nsi.volume_weighted_nsi import (
    _log_volume_weights, compute_volume_weighted_nsi,
)


ROOT = Path(__file__).resolve().parents[1]
SNAP_DIR = ROOT / "results" / "snapshots"
HAVE_CACHE = (SNAP_DIR / "stage1_results.pkl").exists() and \
             (SNAP_DIR / "stage4_results.pkl").exists()


def _fake_stage4(label, regime, *, n_nodes=20, n_edges=10, hhi=0.05,
                 modularity=0.5, gini=0.4, z_ffl=2.0, n_communities=4,
                 purity=0.6):
    """Minimal Stage-4 record matching the contract of compute_nsi_components."""
    return {
        "regime": regime,
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "community": {"modularity": modularity, "n_communities": n_communities,
                       "purity": purity},
        "pagerank": {"hhi_top10": hhi, "gini": gini,
                     "pagerank_scores": {i: 1.0 / n_nodes for i in range(n_nodes)}},
        "motifs": {"z_scores": {"feed_forward_loop": z_ffl,
                                  "mutual_regulation": -1.0}},
    }


def _fake_stage1_corr(p=20):
    R = np.eye(p) * 0.9 + 0.05
    np.fill_diagonal(R, 1.0)
    return {"mean_corr": float(R[np.triu_indices_from(R, k=1)].mean()),
            "R_avg": R}


# --- compute_nsi_components ---------------------------------------------------

def test_nsi_weights_sum_to_one():
    assert abs(sum(NSI_WEIGHTS_4CH) - 1.0) < 1e-12


def test_compute_nsi_minimal_shape():
    labels = ["A", "B", "C", "D"]
    regimes = ["baseline", "crisis", "stress", "recovery"]
    s4 = {l: _fake_stage4(l, r, z_ffl=z) for l, r, z in
          zip(labels, regimes, (1.0, 5.0, 2.0, 1.5))}
    s1c = {l: _fake_stage1_corr() for l in labels}
    df = compute_nsi_components(s4, s1c)
    assert len(df) == 4
    assert (df["nsi"] >= 0).all() and (df["nsi"] <= 1).all()
    for col in ("network_sparsity_norm", "hhi_top10_norm",
                "mean_corr_norm", "motif_shift_norm"):
        assert df[col].between(0, 1).all(), col


def test_compute_nsi_requires_motifs():
    s4 = {"A": _fake_stage4("A", "baseline")}
    s4["A"]["motifs"] = None
    with pytest.raises(AssertionError, match="motif"):
        compute_nsi_components(s4, {"A": _fake_stage1_corr()})


def test_compute_nsi_requires_correlations():
    s4 = {"A": _fake_stage4("A", "baseline")}
    with pytest.raises(AssertionError, match="snapshot_correlations"):
        compute_nsi_components(s4, None)


def test_compute_nsi_constant_channel_zero():
    """When a channel is constant across snapshots, normalised version
    must collapse to 0 (paper convention)."""
    labels = ["A", "B", "C"]
    s4 = {l: _fake_stage4(l, "baseline", z_ffl=1.0) for l in labels}
    s4["B"]["regime"] = "crisis"
    s1c = {l: _fake_stage1_corr() for l in labels}
    df = compute_nsi_components(s4, s1c)
    # All hhi values are equal → norm channel is 0
    assert (df["hhi_top10_norm"] == 0.0).all()


def test_compute_nsi_top_rank_follows_max_channels():
    """The snapshot maximising every channel scores the highest NSI."""
    labels = ["calm", "panic", "mild"]
    regimes = ["baseline", "crisis", "stress"]
    s4 = {
        "calm":  _fake_stage4("calm",  "baseline", n_edges=80, hhi=0.02, z_ffl=1.0),
        "panic": _fake_stage4("panic", "crisis",   n_edges=5,  hhi=0.20, z_ffl=10.0),
        "mild":  _fake_stage4("mild",  "stress",   n_edges=40, hhi=0.10, z_ffl=4.0),
    }
    s1c = {
        "calm":  {"mean_corr": 0.10},
        "panic": {"mean_corr": 0.70},
        "mild":  {"mean_corr": 0.40},
    }
    df = compute_nsi_components(s4, s1c).set_index("snapshot")
    assert df.loc["panic", "nsi"] > df.loc["mild", "nsi"] > df.loc["calm", "nsi"]


# --- _log_volume_weights ------------------------------------------------------

def test_log_volume_weights_sum_to_one():
    adv = pd.Series([1e8, 2e8, 5e7, 1e9], index=["A", "B", "C", "D"])
    w = _log_volume_weights(["A", "B", "C", "D"], adv)
    assert np.isclose(w.sum(), 1.0)
    assert (w > 0).all()


def test_log_volume_weights_uniform_when_zero():
    adv = pd.Series([0.0, 0.0], index=["A", "B"])
    w = _log_volume_weights(["A", "B"], adv)
    assert np.allclose(w, 0.5)


def test_log_volume_weights_missing_ticker_zero():
    adv = pd.Series([1e8], index=["A"])
    w = _log_volume_weights(["A", "MISS"], adv)
    assert w[0] > 0 and w[1] == 0


# --- compute_rolling_nsi (synthetic) ------------------------------------------

def test_rolling_nsi_shape_and_range():
    rng = np.random.default_rng(0)
    p = 30
    n = 800
    X = rng.standard_normal((n, p)) * 0.02
    cols = [f"T{i}" for i in range(p)]
    idx = pd.bdate_range("2010-01-04", periods=n)
    returns = pd.DataFrame(X, index=idx, columns=cols)
    df = compute_rolling_nsi(returns, window_days=252, step_days=63, n_assets=p)
    assert len(df) > 0
    assert df["nsi"].between(0, 1).all()
    assert ROLLING_PRECISION_EPS == 1e-10
    assert ROLLING_PSD_FLOOR == 1e-4


# --- backtest_rolling_nsi_vs_vix ---------------------------------------------

def test_backtest_keys_and_range():
    idx = pd.bdate_range("2010-01-04", periods=400)
    nsi = pd.Series(np.linspace(0.1, 0.9, len(idx)), index=idx)
    vix = pd.Series(20 + 5 * np.sin(np.linspace(0, 6, len(idx))), index=idx)
    res = backtest_rolling_nsi_vs_vix(nsi, vix)
    assert "contemporaneous_corr" in res
    assert "lagged_correlations" in res
    for k in res["lagged_correlations"]:
        assert k.startswith("lag")
    assert -1 <= res["contemporaneous_corr"] <= 1


# --- Cluster bootstrap partition invariants ----------------------------------

CLUSTERS = [
    ['Oct 2008 Peak', 'Mar 2009 Recovery'],
    ['2011-2012 Calm'],
    ['2015 Calm'],
    ['2018 VolShock'],
    ['Jan 2020 Pre-shock', 'Mar 2020 Peak', 'Jun 2020 Stable'],
    ['2022 Rate Hikes'],
    ['2025 Contemporary'],
]
SNAPS_10 = ['Oct 2008 Peak', 'Mar 2009 Recovery', '2011-2012 Calm', '2015 Calm',
            '2018 VolShock', 'Jan 2020 Pre-shock', 'Mar 2020 Peak',
            'Jun 2020 Stable', '2022 Rate Hikes', '2025 Contemporary']


def test_cluster_partition_complete():
    flat = [s for c in CLUSTERS for s in c]
    assert sorted(flat) == sorted(SNAPS_10)


def test_cluster_partition_disjoint():
    flat = [s for c in CLUSTERS for s in c]
    assert len(flat) == len(set(flat)), "clusters overlap"


def test_cluster_bootstrap_min_length():
    """Every 7-cluster resample yields at least 7 snapshots
    (smallest cluster has 1 element)."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        chosen = rng.integers(0, len(CLUSTERS), size=len(CLUSTERS))
        flat = []
        for ci in chosen:
            flat.extend(CLUSTERS[int(ci)])
        assert len(flat) >= 7


# --- Smoke against the cached pickles ----------------------------------------

@pytest.mark.skipif(not HAVE_CACHE, reason="Stage 1/4 cache not present")
def test_snapshot_nsi_matches_cache():
    """Recomputing snapshot NSI from Stage 1/4 caches matches the
    Stage 5 cache to four decimals."""
    with open(SNAP_DIR / "stage1_results.pkl", "rb") as f:
        s1 = pickle.load(f)
    with open(SNAP_DIR / "stage4_results.pkl", "rb") as f:
        s4 = pickle.load(f)
    with open(SNAP_DIR / "stage5_results.pkl", "rb") as f:
        s5 = pickle.load(f)
    df = compute_nsi_components(s4, s1["snapshot_correlations"])
    cached = s5["snapshot_nsi"].set_index("snapshot")["nsi"]
    fresh = df.set_index("snapshot")["nsi"]
    assert np.allclose(cached.values, fresh.reindex(cached.index).values, atol=1e-9)


@pytest.mark.skipif(not HAVE_CACHE, reason="Stage 1/4 cache not present")
def test_vw_nsi_smoke():
    """VW-NSI runs end-to-end on the production cache and stays in [0,1]."""
    with open(SNAP_DIR / "stage1_results.pkl", "rb") as f:
        s1 = pickle.load(f)
    with open(SNAP_DIR / "stage4_results.pkl", "rb") as f:
        s4 = pickle.load(f)
    df = compute_volume_weighted_nsi(s4, s1["snapshot_correlations"])
    assert len(df) == len(s4)
    assert df["nsi_vw"].between(0, 1).all()
    for col in ("network_sparsity_vw", "hhi_vw", "mean_corr_vw", "motif_shift_vw"):
        assert col in df.columns
