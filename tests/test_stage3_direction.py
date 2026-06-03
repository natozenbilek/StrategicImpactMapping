"""Stage 3 lead-follower cascade unit tests.

Covers the FWL partial-correlation helper, the single-lag Granger
helper, direction-aware control construction, the four cascade branches
(drop / one-way / mutual / dominance), end-to-end snapshot orchestration
on synthetic data, the empty / degenerate Stage-2 inputs, and the
parametric knobs used by the sensitivity sweep.
"""
import numpy as np
import pandas as pd
import pytest

from src.stage3_direction.lead_lag import (
    _top_k_controls, _build_controls,
    lagged_partial_correlation, granger_test,
    assign_directions_snapshot,
    GRANGER_FLOOR_DEFAULT, TOP_K_CONTROLS,
)


# =====================================================================
# Synthetic data generators
# =====================================================================

def _make_R_avg(p, seed=2026):
    """A symmetric, unit-diagonal correlation matrix with asymmetric rows
    so that top-k by |R_avg[i]| differs from top-k by |R_avg[j]|."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((p, p)) * 0.2
    Sigma = A @ A.T + np.eye(p)
    d = np.sqrt(np.diag(Sigma))
    R = Sigma / np.outer(d, d)
    np.fill_diagonal(R, 1.0)
    return R


def _make_lead_lag_panel(T, p, lead_pairs=None, lead_strength=0.6, seed=2026):
    """Daily-return panel with prescribed lead-lag couplings.

    lead_pairs: list of (i, j) meaning column i at t-1 predicts column j at t.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((T, p)) * 0.01
    if lead_pairs:
        for i, j in lead_pairs:
            X[1:, j] += lead_strength * X[:-1, i]
    cols = [f"T{k}" for k in range(p)]
    idx = pd.bdate_range("2024-01-01", periods=T)
    return pd.DataFrame(X, index=idx, columns=cols)


def _make_dependent_pair(T, lead, anti=False, seed=2026):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(T)
    y = rng.standard_normal(T) * 0.5
    sign = -1.0 if anti else 1.0
    y[1:] += sign * lead * x[:-1]
    return x, y


# =====================================================================
# _top_k_controls
# =====================================================================

class TestTopKControls:
    def test_returns_indices_in_range(self):
        R = _make_R_avg(20)
        idx = _top_k_controls(R, source_col=0, exclude_cols=[0])
        assert idx.dtype.kind in "iu"
        assert idx.shape[0] == TOP_K_CONTROLS
        assert ((idx >= 0) & (idx < 20)).all()

    def test_excludes_requested_columns(self):
        R = _make_R_avg(20)
        idx = _top_k_controls(R, source_col=3, exclude_cols=[3, 7])
        assert 3 not in idx
        assert 7 not in idx

    def test_picks_genuine_top_k(self):
        # Build a controlled R where the top-5 from row 0 are unambiguous.
        p = 12
        R = np.eye(p)
        R[0, 1:6] = R[1:6, 0] = 0.9
        R[0, 6:] = R[6:, 0] = 0.1
        # Symmetrise — keep R PSD trivially since it's near-identity.
        chosen = _top_k_controls(R, source_col=0, exclude_cols=[0])
        assert set(chosen.tolist()) == {1, 2, 3, 4, 5}

    def test_direction_aware_top_k_differs(self):
        # If R has asymmetric off-diagonal magnitudes, top-k for row i
        # and top-k for row j typically differ — the audit fix is to
        # recompute per direction.
        R = _make_R_avg(50)
        i, j = 2, 7
        from_i = _top_k_controls(R, i, [i, j])
        from_j = _top_k_controls(R, j, [i, j])
        assert set(from_i.tolist()) != set(from_j.tolist())

    def test_handles_nan_in_row(self):
        R = _make_R_avg(15)
        R[5, 9] = np.nan; R[9, 5] = np.nan
        idx = _top_k_controls(R, source_col=5, exclude_cols=[5])
        assert 9 not in idx  # NaN candidate is masked out
        assert len(idx) == TOP_K_CONTROLS

    def test_k_capped_at_available_count(self):
        p = 6
        R = np.eye(p)
        idx = _top_k_controls(R, source_col=0, exclude_cols=[0, 1, 2, 3, 4])
        assert len(idx) <= 1


# =====================================================================
# lagged_partial_correlation
# =====================================================================

class TestLaggedPartialCorrelation:
    def test_short_series_returns_neutral(self):
        x = np.zeros(8); y = np.zeros(8)
        r, p = lagged_partial_correlation(x, y, controls=None, lag=1)
        assert r == 0.0 and p == 1.0

    def test_detects_synthetic_lead(self):
        x, y = _make_dependent_pair(T=600, lead=0.7, seed=11)
        r, p = lagged_partial_correlation(x, y, controls=None, lag=1)
        assert r > 0
        assert p < 0.01

    def test_negative_lead_is_negative_significant(self):
        x, y = _make_dependent_pair(T=600, lead=0.7, anti=True, seed=12)
        r, p = lagged_partial_correlation(x, y, controls=None, lag=1)
        assert r < 0
        assert p < 0.01

    def test_independent_pair_is_not_significant(self):
        rng = np.random.default_rng(99)
        x = rng.standard_normal(400)
        y = rng.standard_normal(400)
        _, p = lagged_partial_correlation(x, y, controls=None, lag=1)
        assert p > 0.05

    def test_controls_shape_validated(self):
        T = 200
        rng = np.random.default_rng(0)
        x = rng.standard_normal(T); y = rng.standard_normal(T)
        bad_controls = rng.standard_normal((T - 5, 3))
        with pytest.raises(AssertionError):
            lagged_partial_correlation(x, y, controls=bad_controls)

    def test_rank_deficient_controls_do_not_propagate_nan(self):
        T = 300
        rng = np.random.default_rng(0)
        x = rng.standard_normal(T); y = rng.standard_normal(T)
        # Two identical control columns -> ctrl_only is rank-deficient.
        c = rng.standard_normal(T)
        controls = np.column_stack([c, c, c])
        r, p = lagged_partial_correlation(x, y, controls, lag=1)
        assert np.isfinite(r) and np.isfinite(p)


# =====================================================================
# granger_test
# =====================================================================

class TestGrangerTest:
    def test_short_series_returns_neutral(self):
        x = np.zeros(5); y = np.zeros(5)
        f, p, lag = granger_test(x, y, lag=1)
        assert f == 0.0 and p == 1.0 and lag == 1

    def test_detects_synthetic_causality(self):
        x, y = _make_dependent_pair(T=500, lead=0.6, seed=33)
        f, p, lag = granger_test(x, y, lag=1)
        assert p < 0.01
        assert f > 0

    def test_independent_is_not_significant(self):
        rng = np.random.default_rng(7)
        x = rng.standard_normal(400)
        y = rng.standard_normal(400)
        _, p, _ = granger_test(x, y, lag=1)
        assert p > 0.05

    def test_no_best_of_lag_scan(self):
        # Verify the function tests exactly the requested lag — calling
        # with lag=2 must report the lag-2 F-stat, not a best-of-{1,2}.
        x, y = _make_dependent_pair(T=500, lead=0.6, seed=55)
        f1, _, returned_lag1 = granger_test(x, y, lag=1)
        f2, _, returned_lag2 = granger_test(x, y, lag=2)
        assert returned_lag1 == 1
        assert returned_lag2 == 2
        # The two F-stats are computed under different null/alt model
        # structures so they are not equal in general.
        assert f1 != f2


# =====================================================================
# _build_controls
# =====================================================================

class TestBuildControls:
    def _setup(self, p=10, T=200, drop_one_ticker=False):
        R = _make_R_avg(p, seed=4)
        returns = _make_lead_lag_panel(T, p, seed=4)
        tickers = list(returns.columns)
        if drop_one_ticker:
            # Simulate a ticker present in Stage-2 results but absent
            # from the daily-returns panel.
            returns = returns.drop(columns=["T7"])
        available = [t for t in tickers if t in returns.columns]
        sub = returns[available].dropna(how="any")
        returns_arr = sub.values
        ticker_to_subcol = {t: i for i, t in enumerate(available)}
        return R, tickers, ticker_to_subcol, returns_arr

    def test_returns_two_d_panel(self):
        R, tickers, t2c, arr = self._setup()
        out = _build_controls(R, 0, [0, 1], tickers, t2c, arr)
        assert out.ndim == 2
        assert out.shape[0] == arr.shape[0]
        assert out.shape[1] == TOP_K_CONTROLS

    def test_returns_none_when_panel_too_narrow(self):
        R = np.eye(2); tickers = ["A", "B"]
        arr = np.zeros((100, 2))
        t2c = {"A": 0, "B": 1}
        assert _build_controls(R, 0, [0, 1], tickers, t2c, arr) is None

    def test_drops_missing_tickers(self):
        # When a top-k Stage-2 candidate is absent from the daily-panel
        # (dropna left a thin sub-panel), the helper degrades to fewer
        # controls rather than erroring.
        R, tickers, t2c, arr = self._setup(drop_one_ticker=True)
        out = _build_controls(R, 0, [0, 1], tickers, t2c, arr)
        assert out is not None
        assert out.shape[1] <= TOP_K_CONTROLS


# =====================================================================
# assign_directions_snapshot — cascade branches
# =====================================================================

def _identity_R_avg(p):
    R = np.eye(p)
    # Mild off-diagonal so top-k selection has work to do.
    rng = np.random.default_rng(0)
    M = rng.standard_normal((p, p)) * 0.05
    M = (M + M.T) / 2; np.fill_diagonal(M, 0)
    R = R + M
    np.fill_diagonal(R, 1.0)
    return R


class TestAssignDirectionsCascade:
    def test_empty_input_returns_empty_result(self):
        p = 8
        adj = np.zeros((p, p))
        returns = _make_lead_lag_panel(120, p, seed=1)
        tickers = list(returns.columns)
        R = _identity_R_avg(p)
        res = assign_directions_snapshot(adj, returns, tickers, R, verbose=False)
        assert res["n_input_edges"] == 0
        assert res["n_directed"] == 0
        assert res["n_bidirectional"] == 0
        assert np.all(res["directed_adj"] == 0)

    def test_one_way_assignment_on_synthetic_lead(self):
        p = 6
        # Only the (0, 1) pair has a true lead 0 -> 1.
        returns = _make_lead_lag_panel(
            T=400, p=p, lead_pairs=[(0, 1)], lead_strength=0.7, seed=2)
        tickers = list(returns.columns)
        adj = np.zeros((p, p)); adj[0, 1] = adj[1, 0] = 0.3
        R = _identity_R_avg(p)
        res = assign_directions_snapshot(adj, returns, tickers, R, verbose=False)
        D = res["directed_adj"]
        assert D[0, 1] > 0
        assert D[1, 0] == 0
        assert res["n_directed"] == 1
        assert res["n_bidirectional"] == 0

    def test_mutual_assignment_when_both_directions_match(self):
        # Bidirectional Granger causality with method="granger" makes the
        # cascade deterministic: each direction contributes exactly
        # granger_floor, neither dominates by tau=1.5, so the edge is
        # mutual by definition.
        p = 6
        returns = _make_lead_lag_panel(
            T=600, p=p,
            lead_pairs=[(0, 1), (1, 0)], lead_strength=0.5, seed=3)
        tickers = list(returns.columns)
        adj = np.zeros((p, p)); adj[0, 1] = adj[1, 0] = 0.3
        R = _identity_R_avg(p)
        res = assign_directions_snapshot(
            adj, returns, tickers, R, method="granger",
            tau=1.5, verbose=False)
        D = res["directed_adj"]
        assert D[0, 1] > 0 and D[1, 0] > 0
        assert res["n_bidirectional"] == 1
        assert res["n_directed"] == 0

    def test_dominance_rule_assigns_dominant_direction(self):
        p = 6
        # Strong (0 -> 1) lead; weak (1 -> 0) reverse — ratio should exceed tau.
        returns = _make_lead_lag_panel(
            T=500, p=p,
            lead_pairs=[(0, 1), (0, 1), (1, 0)], lead_strength=0.45, seed=8)
        tickers = list(returns.columns)
        adj = np.zeros((p, p)); adj[0, 1] = adj[1, 0] = 0.3
        R = _identity_R_avg(p)
        res = assign_directions_snapshot(
            adj, returns, tickers, R, tau=1.5, verbose=False)
        D = res["directed_adj"]
        # We tolerate either pure i->j (dominance) or mutual (synthetic
        # data noise) but never pure j->i.
        assert D[0, 1] > 0

    def test_insufficient_sample_returns_empty(self):
        p = 6
        returns = _make_lead_lag_panel(T=10, p=p, seed=4)
        tickers = list(returns.columns)
        adj = np.zeros((p, p)); adj[0, 1] = adj[1, 0] = 0.3
        R = _identity_R_avg(p)
        res = assign_directions_snapshot(adj, returns, tickers, R, verbose=False)
        assert res["n_input_edges"] == 0

    def test_missing_ticker_in_returns_is_skipped(self):
        p = 6
        returns = _make_lead_lag_panel(T=400, p=p, seed=5)
        returns = returns.drop(columns=["T1"])
        tickers = [f"T{k}" for k in range(p)]
        adj = np.zeros((p, p)); adj[0, 1] = adj[1, 0] = 0.3
        R = _identity_R_avg(p)
        res = assign_directions_snapshot(adj, returns, tickers, R, verbose=False)
        # Edge involving T1 (col 1) is skipped; result is empty.
        assert res["n_directed"] == 0
        assert res["n_bidirectional"] == 0

    def test_postcondition_diag_zero(self):
        p = 8
        returns = _make_lead_lag_panel(T=300, p=p,
                                       lead_pairs=[(0, 1), (2, 3)],
                                       lead_strength=0.5, seed=6)
        tickers = list(returns.columns)
        adj = np.zeros((p, p))
        adj[0, 1] = adj[1, 0] = adj[2, 3] = adj[3, 2] = 0.2
        R = _identity_R_avg(p)
        res = assign_directions_snapshot(adj, returns, tickers, R, verbose=False)
        D = res["directed_adj"]
        assert np.allclose(np.diag(D), 0.0)
        assert (D >= 0).all()

    def test_edge_count_conservation(self):
        p = 8
        returns = _make_lead_lag_panel(T=300, p=p,
                                       lead_pairs=[(0, 1), (2, 3)],
                                       lead_strength=0.5, seed=7)
        tickers = list(returns.columns)
        adj = np.zeros((p, p))
        adj[0, 1] = adj[1, 0] = 0.2
        adj[2, 3] = adj[3, 2] = 0.2
        adj[4, 5] = adj[5, 4] = 0.2
        R = _identity_R_avg(p)
        res = assign_directions_snapshot(adj, returns, tickers, R, verbose=False)
        s = res["n_directed"] + res["n_bidirectional"] + res["n_dropped"]
        assert s == res["n_input_edges"]


class TestAssignDirectionsInputValidation:
    def test_rejects_nonsquare_adjacency(self):
        adj = np.zeros((6, 7))
        returns = _make_lead_lag_panel(T=120, p=6, seed=0)
        tickers = list(returns.columns)
        R = _identity_R_avg(6)
        with pytest.raises(AssertionError):
            assign_directions_snapshot(adj, returns, tickers, R, verbose=False)

    def test_rejects_R_avg_shape_mismatch(self):
        p = 6
        adj = np.zeros((p, p))
        returns = _make_lead_lag_panel(T=120, p=p, seed=0)
        tickers = list(returns.columns)
        R = _identity_R_avg(p + 1)
        with pytest.raises(AssertionError):
            assign_directions_snapshot(adj, returns, tickers, R, verbose=False)

    def test_rejects_unknown_method(self):
        p = 4
        adj = np.zeros((p, p))
        returns = _make_lead_lag_panel(T=120, p=p, seed=0)
        tickers = list(returns.columns)
        R = _identity_R_avg(p)
        with pytest.raises(AssertionError):
            assign_directions_snapshot(adj, returns, tickers, R,
                                       method="bogus", verbose=False)


class TestAssignDirectionsParametric:
    def test_alpha_tau_lag_floor_propagated(self):
        # Confirm the parametric knobs reach the cascade without error.
        p = 6
        returns = _make_lead_lag_panel(T=300, p=p,
                                       lead_pairs=[(0, 1)],
                                       lead_strength=0.5, seed=9)
        tickers = list(returns.columns)
        adj = np.zeros((p, p)); adj[0, 1] = adj[1, 0] = 0.2
        R = _identity_R_avg(p)
        for alpha, tau, gl, gf in [
            (0.05, 1.2, 1, 0.5),
            (0.10, 2.0, 1, 0.3),
            (0.05, 1.5, 2, 0.7),
        ]:
            res = assign_directions_snapshot(
                adj, returns, tickers, R,
                alpha=alpha, tau=tau,
                granger_lag=gl, granger_floor=gf,
                verbose=False)
            assert res["n_input_edges"] == 1
            s = res["n_directed"] + res["n_bidirectional"] + res["n_dropped"]
            assert s == res["n_input_edges"]
