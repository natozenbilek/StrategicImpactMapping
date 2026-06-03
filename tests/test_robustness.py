"""Robustness-layer unit tests.

Covers the four density-controlled / inferential modules that feed
\\S sec:robust of the manuscript and Appendix \\S app:robust /
\\S app:nsi_weight_sens:

* :mod:`src.stage4_network.crisis_signals` -- mutual-dyad fraction,
  cross-sector edge fraction at the auto-sized density-matched top-k
  (5% of p_min(p_min-1)/2), and small-world sigma computed off the
  Stage-4 ER-null draws.
* :mod:`paper._nsi_weight_sensitivity` -- one-sided exact-permutation
  p-value on the C(20, 7) = 77,520 partitions of the 20-snapshot panel.

Stage-4 plumbing tests (build_threshold_network, erdos_renyi_test,
motif_significance, etc.) live in tests/test_stage4_network.py.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.stage4_network.crisis_signals import (
    DEFAULT_TARGET_DENSITY,
    cross_sector_edge_fraction,
    mutual_dyad_fraction,
    small_world_coefficients,
)
from paper._nsi_weight_sensitivity import exact_perm_p_one_sided


# --- mutual_dyad_fraction ------------------------------------------------

def test_mutual_dyad_fraction_formula():
    s3 = {
        "snap_A": {"regime": "crisis",   "n_directed": 100, "n_bidirectional": 20},
        "snap_B": {"regime": "baseline", "n_directed": 200, "n_bidirectional":  0},
        "snap_C": {"regime": "stress",   "n_directed":  50, "n_bidirectional": 50},
    }
    df = mutual_dyad_fraction(s3).set_index("snapshot")
    assert df.loc["snap_A", "mutual_fraction"] == pytest.approx(20 / 120)
    assert df.loc["snap_B", "mutual_fraction"] == pytest.approx(0.0)
    assert df.loc["snap_C", "mutual_fraction"] == pytest.approx(0.5)
    assert (df["mutual_fraction"].between(0.0, 1.0)).all()


def test_mutual_dyad_fraction_no_edges_returns_nan():
    s3 = {"empty": {"regime": "baseline", "n_directed": 0, "n_bidirectional": 0}}
    df = mutual_dyad_fraction(s3)
    assert math.isnan(df["mutual_fraction"].iloc[0])


def test_mutual_dyad_fraction_preserves_directed_plus_mutual():
    s3 = {f"s{i}": {"regime": "baseline",
                    "n_directed": int(np.random.default_rng(i).integers(10, 200)),
                    "n_bidirectional": int(np.random.default_rng(i + 99).integers(0, 50))}
          for i in range(5)}
    df = mutual_dyad_fraction(s3).set_index("snapshot")
    for k, v in s3.items():
        total = v["n_directed"] + v["n_bidirectional"]
        assert df.loc[k, "directed"] + df.loc[k, "mutual"] == total


# --- cross_sector_edge_fraction -----------------------------------------

def _toy_stage1(p=10, n_snaps=3, seed=0):
    rng = np.random.default_rng(seed)
    snap_corrs = {}
    tickers = [f"T{i}" for i in range(p)]
    for k in range(n_snaps):
        A = rng.standard_normal((p, p))
        R = (A + A.T) / 2
        np.fill_diagonal(R, 1.0)
        snap_corrs[f"snap_{k}"] = {"R_avg": R, "tickers": tickers,
                                   "regime": "baseline"}
    return {"snapshot_correlations": snap_corrs}


def test_cross_sector_default_target_density_is_five_percent():
    assert DEFAULT_TARGET_DENSITY == 0.05


def test_cross_sector_default_auto_sizes_topk_to_five_percent():
    p = 200
    s1 = _toy_stage1(p=p, n_snaps=2, seed=1)
    tickers = s1["snapshot_correlations"]["snap_0"]["tickers"]
    sp500_info = pd.DataFrame({
        "Symbol": tickers,
        "GICS Sector": ["A" if i % 2 == 0 else "B" for i in range(p)],
    })
    df = cross_sector_edge_fraction(sp500_info, stage1=s1)
    # default budget auto-sizes to 5% density at the sparsest panel.
    expected_topk = max(p, int(round(DEFAULT_TARGET_DENSITY * p * (p - 1) / 2)))
    assert int(df["top_k"].iloc[0]) == expected_topk
    # all four reported columns add up to top_k.
    for _, row in df.iterrows():
        assert row["within_sec"] + row["cross_sec"] == row["top_k"]
        assert 0.0 <= row["cs_fraction"] <= 1.0


def test_cross_sector_explicit_topk_arg_overrides_default():
    p = 30
    s1 = _toy_stage1(p=p, n_snaps=2, seed=2)
    tickers = s1["snapshot_correlations"]["snap_0"]["tickers"]
    sp500_info = pd.DataFrame({
        "Symbol": tickers,
        "GICS Sector": ["S" for _ in tickers],
    })
    df = cross_sector_edge_fraction(sp500_info, stage1=s1, top_k=15)
    assert (df["top_k"] == 15).all()
    # single-sector panel: every edge is within-sector.
    assert (df["cross_sec"] == 0).all()
    assert (df["cs_fraction"] == 0.0).all()


def test_cross_sector_target_density_overrides_default():
    p = 30
    s1 = _toy_stage1(p=p, n_snaps=2, seed=3)
    tickers = s1["snapshot_correlations"]["snap_0"]["tickers"]
    sp500_info = pd.DataFrame({
        "Symbol": tickers,
        "GICS Sector": ["A"] * 15 + ["B"] * 15,
    })
    df = cross_sector_edge_fraction(sp500_info, stage1=s1, target_density=0.10)
    expected_topk = max(p, int(round(0.10 * p * (p - 1) / 2)))
    assert int(df["top_k"].iloc[0]) == expected_topk


# --- small_world_coefficients -------------------------------------------

def _toy_stage4_for_sigma(c_emp, c_null, l_emp, l_null, z_c=5.0):
    return {
        "snap": {
            "regime": "crisis",
            "erdos_renyi": {
                "empirical": {"clustering": c_emp, "path_length": l_emp},
                "null_mean": {"clustering": c_null, "path_length": l_null},
                "z_scores": {"clustering": z_c},
            },
        },
    }


def test_small_world_sigma_matches_humphries_gurney_formula():
    s4 = _toy_stage4_for_sigma(c_emp=0.40, c_null=0.10, l_emp=2.5, l_null=3.0)
    row = small_world_coefficients(s4).iloc[0]
    assert row["C_ratio"] == pytest.approx(4.0)
    assert row["L_emp"] / row["L_null"] == pytest.approx(2.5 / 3.0)
    assert row["sigma"] == pytest.approx(4.0 / (2.5 / 3.0))


def test_small_world_sigma_nan_when_l_null_zero():
    s4 = _toy_stage4_for_sigma(c_emp=0.40, c_null=0.10, l_emp=2.5, l_null=0.0)
    row = small_world_coefficients(s4).iloc[0]
    assert math.isnan(row["sigma"])


def test_small_world_sigma_nan_when_c_null_zero():
    s4 = _toy_stage4_for_sigma(c_emp=0.40, c_null=0.0, l_emp=2.5, l_null=3.0)
    row = small_world_coefficients(s4).iloc[0]
    assert math.isnan(row["sigma"])


# --- exact_perm_p_one_sided ---------------------------------------------

def test_exact_perm_p_one_sided_floor_is_one_over_C20_7():
    # Crisis indices hold the panel's seven highest values -> observed
    # difference is the sample maximum, so exactly one of C(20,7)=77,520
    # assignments matches (canonical crisis_idx = [0, 4, 5, 6, 7, 11, 16]).
    nsi = np.zeros(20)
    for rank, idx in enumerate([0, 4, 5, 6, 7, 11, 16]):
        nsi[idx] = 7 - rank      # 7,6,5,4,3,2,1 -> seven strict highest
    p = exact_perm_p_one_sided(nsi)
    assert p == pytest.approx(1.0 / 77520)


def test_exact_perm_p_one_sided_uniform_input_gives_one():
    nsi = np.ones(20)
    p = exact_perm_p_one_sided(nsi)
    assert p == 1.0


def test_exact_perm_p_one_sided_total_partitions_equal_C20_7():
    # Spot-check the floor / ceiling relationship.
    rng = np.random.default_rng(11)
    nsi = rng.standard_normal(20)
    p = exact_perm_p_one_sided(nsi)
    # 1/77520 <= p <= 77520/77520
    assert 1.0 / 77520 - 1e-12 <= p <= 1.0 + 1e-12
    # p is always an integer multiple of 1/77520.
    n = round(p * 77520)
    assert abs(p * 77520 - n) < 1e-9
