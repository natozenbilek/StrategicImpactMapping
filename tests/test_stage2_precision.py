"""Stage 2 precision-matrix filter unit tests.

Covers fit_glasso_ebic branch paths, EBIC/shrinkage tier helpers, the
spectral safety net, the constrained-BIC fallback, and the end-to-end
run_stage2 driver invariants.
"""
import numpy as np
import pytest

from src.stage2_precision.glasso_filter import (
    compute_ebic, fit_glasso_ebic, precision_to_partial_corr,
    build_adjacency_matrix, run_stage2,
    _gamma_for_ratio, _delta_for_ratio, _apply_spectral_safety_net,
    SHRINKAGE_TIERS, SPECTRAL_TRIGGER, SPECTRAL_TARGET,
)
from src.config import (
    EBIC_GAMMA, EBIC_GAMMA_MID, EBIC_RATIO_LOW, EBIC_RATIO_MID,
)


def _make_factor_correlation(p, n_factors=2, scale=0.4, seed=2026):
    """Build a well-conditioned, factor-structured PD correlation."""
    rng = np.random.default_rng(seed)
    F = rng.standard_normal((p, n_factors))
    Sigma = scale * (F @ F.T) + np.eye(p)
    d = np.sqrt(np.diag(Sigma))
    return Sigma / np.outer(d, d)


# =====================================================================
# Tier helpers
# =====================================================================

class TestGammaForRatio:
    def test_zero_branch_below_low_threshold(self):
        assert _gamma_for_ratio(0.0) == 0.0
        assert _gamma_for_ratio(EBIC_RATIO_LOW - 1e-9) == 0.0

    def test_mid_branch(self):
        assert _gamma_for_ratio(EBIC_RATIO_LOW) == EBIC_GAMMA_MID
        assert _gamma_for_ratio(EBIC_RATIO_MID - 1e-9) == EBIC_GAMMA_MID

    def test_high_branch(self):
        assert _gamma_for_ratio(EBIC_RATIO_MID) == EBIC_GAMMA
        assert _gamma_for_ratio(100.0) == EBIC_GAMMA


class TestDeltaForRatio:
    @pytest.mark.parametrize("ratio,expected", [
        (0.05, 0.55), (0.19, 0.55),
        (0.20, 0.45), (0.29, 0.45),
        (0.30, 0.35), (0.49, 0.35),
        (0.50, 0.15), (0.99, 0.15),
        (1.00, 0.05), (1.99, 0.05),
        (2.00, 0.005), (100.0, 0.005),
    ])
    def test_tier_lookup(self, ratio, expected):
        assert _delta_for_ratio(ratio) == expected

    def test_tier_table_monotone_decreasing(self):
        # Each successive tier's delta should not exceed its predecessor.
        deltas = [d for _, d in SHRINKAGE_TIERS]
        assert deltas == sorted(deltas, reverse=True)


# =====================================================================
# Spectral safety net
# =====================================================================

class TestSpectralSafetyNet:
    def test_no_op_on_strictly_psd(self):
        R = _make_factor_correlation(20)
        assert float(np.linalg.eigvalsh(R).min()) >= SPECTRAL_TRIGGER
        out = _apply_spectral_safety_net(R)
        assert np.allclose(out, R, atol=1e-12)

    def test_shifts_when_lambda_min_below_trigger(self):
        R = _make_factor_correlation(20)
        # Inject a -1e-3 leakage uniformly so lambda_min < TRIGGER.
        R_leaky = R - (float(np.linalg.eigvalsh(R).min()) + 1e-3) * np.eye(20)
        assert float(np.linalg.eigvalsh(R_leaky).min()) < SPECTRAL_TRIGGER

        out = _apply_spectral_safety_net(R_leaky)
        assert abs(float(np.linalg.eigvalsh(out).min()) - SPECTRAL_TARGET) < 1e-12

    def test_target_strictly_above_trigger(self):
        # Otherwise a shifted matrix could itself need a second shift.
        assert SPECTRAL_TARGET > SPECTRAL_TRIGGER


# =====================================================================
# compute_ebic
# =====================================================================

class TestComputeEbic:
    def test_gamma_zero_equals_BIC(self):
        p, n = 8, 200
        rng = np.random.default_rng(0)
        A = rng.standard_normal((p, p))
        Omega = A @ A.T + 5 * np.eye(p)
        S = np.linalg.inv(Omega)

        ebic = compute_ebic(Omega, S, n_samples=n, gamma=0.0)
        _, logdet = np.linalg.slogdet(Omega)
        ll = 0.5 * n * (logdet - np.trace(S @ Omega))
        mask = np.abs(Omega) > 1e-10
        np.fill_diagonal(mask, False)
        k = int(mask.sum() // 2)
        assert abs(ebic - (-2 * ll + k * np.log(n))) < 1e-8

    def test_gamma_positive_adds_4kgammalogp(self):
        p, n, gamma = 8, 200, 0.5
        rng = np.random.default_rng(0)
        A = rng.standard_normal((p, p))
        Omega = A @ A.T + 5 * np.eye(p)
        S = np.linalg.inv(Omega)

        ebic_0 = compute_ebic(Omega, S, n_samples=n, gamma=0.0)
        ebic_g = compute_ebic(Omega, S, n_samples=n, gamma=gamma)
        mask = np.abs(Omega) > 1e-10
        np.fill_diagonal(mask, False)
        k = int(mask.sum() // 2)
        assert abs((ebic_g - ebic_0) - 4 * k * gamma * np.log(p)) < 1e-8

    @pytest.mark.parametrize("p", [2, 4, 6, 8])
    def test_inf_on_even_p_negative_definite(self, p):
        # det(-I_p) = (-1)^p = +1 for even p (slogdet sign +1); the
        # eigenvalue check is what catches this case.
        out = compute_ebic(-np.eye(p), np.eye(p), n_samples=100, gamma=0.0)
        assert np.isinf(out)

    @pytest.mark.parametrize("p", [3, 5, 7])
    def test_inf_on_odd_p_negative_definite(self, p):
        # det(-I_p) = -1 for odd p; the slogdet sign check rejects.
        out = compute_ebic(-np.eye(p), np.eye(p), n_samples=100, gamma=0.0)
        assert np.isinf(out)


# =====================================================================
# precision_to_partial_corr / build_adjacency_matrix
# =====================================================================

class TestPartialCorr:
    @pytest.fixture
    def omega(self):
        rng = np.random.default_rng(0)
        A = rng.standard_normal((6, 6))
        return A @ A.T + 5 * np.eye(6)

    def test_unit_diagonal(self, omega):
        Pc = precision_to_partial_corr(omega)
        assert np.allclose(np.diag(Pc), 1.0, atol=1e-12)

    def test_symmetric(self, omega):
        Pc = precision_to_partial_corr(omega)
        assert np.allclose(Pc, Pc.T, atol=1e-12)

    def test_formula(self, omega):
        Pc = precision_to_partial_corr(omega)
        expected = -omega[0, 1] / np.sqrt(omega[0, 0] * omega[1, 1])
        assert abs(Pc[0, 1] - expected) < 1e-10


class TestBuildAdjacency:
    @pytest.fixture
    def omega(self):
        rng = np.random.default_rng(0)
        A = rng.standard_normal((6, 6))
        return A @ A.T + 5 * np.eye(6)

    def test_symmetric_zero_diag_nonneg(self, omega):
        Pc = precision_to_partial_corr(omega)
        adj = build_adjacency_matrix(Pc, omega)
        assert np.allclose(adj, adj.T, atol=1e-12)
        assert np.allclose(np.diag(adj), 0.0)
        assert np.all(adj >= 0)

    def test_respects_precision_support(self, omega):
        omega = omega.copy()
        omega[0, 1] = 0.0
        omega[1, 0] = 0.0
        Pc = precision_to_partial_corr(omega)
        adj = build_adjacency_matrix(Pc, omega)
        assert adj[0, 1] == 0.0
        assert adj[1, 0] == 0.0


# =====================================================================
# fit_glasso_ebic input asserts
# =====================================================================

class TestFitGlassoInputAsserts:
    def test_non_square_rejected(self):
        with pytest.raises(AssertionError, match="square"):
            fit_glasso_ebic(np.eye(5)[:, :4], n_samples=100)

    def test_non_symmetric_rejected(self):
        R = np.eye(5)
        R[0, 1] = 0.5  # off-diagonal asymmetry
        with pytest.raises(AssertionError, match="symmetric"):
            fit_glasso_ebic(R, n_samples=100)

    def test_non_unit_diag_rejected(self):
        R = 2.0 * np.eye(5)  # covariance, not correlation
        with pytest.raises(AssertionError, match="unit diagonal"):
            fit_glasso_ebic(R, n_samples=100)

    def test_non_finite_rejected(self):
        R = _make_factor_correlation(8)
        R[0, 0] = np.nan
        R[0, 1] = np.nan
        R[1, 0] = np.nan  # keep symmetric
        with pytest.raises((ValueError, AssertionError)):
            fit_glasso_ebic(R, n_samples=100)


# =====================================================================
# fit_glasso_ebic happy path + branches
# =====================================================================

class TestFitGlassoEbicHappyPath:
    def test_returns_floor_clearing_graph(self):
        R = _make_factor_correlation(15)
        out = fit_glasso_ebic(R, n_samples=200)
        assert out["n_edges"] >= max(15, 10)

    def test_precision_psd_symmetric(self):
        R = _make_factor_correlation(15)
        out = fit_glasso_ebic(R, n_samples=200)
        prec = out["precision"]
        assert np.allclose(prec, prec.T, atol=1e-8)
        assert float(np.linalg.eigvalsh((prec + prec.T) / 2).min()) > 0

    def test_unconstrained_keys_present(self):
        R = _make_factor_correlation(15)
        out = fit_glasso_ebic(R, n_samples=200)
        for key in ("lambda_unconstr", "n_edges_unconstr", "ebic_unconstr",
                    "fallback_fired"):
            assert key in out

    def test_density_matches_n_edges(self):
        R = _make_factor_correlation(15)
        out = fit_glasso_ebic(R, n_samples=200)
        p = 15
        expected = out["n_edges"] / (p * (p - 1) / 2)
        assert abs(out["density"] - expected) < 1e-12


class TestFallbackBranch:
    """Constrained-BIC fallback triggering and the no-candidate raise."""

    def test_postcondition_holds_on_short_window(self):
        # Strong common factor + small n => unconstrained-BIC argmin
        # tends to the sparse end; the constrained-BIC fallback should
        # take over so the postcondition n_edges >= max(p, 10) holds.
        rng = np.random.default_rng(2026)
        p = 50
        F = rng.standard_normal((p, 1)) * 1.5
        Sigma = F @ F.T + 0.1 * np.eye(p)
        d = np.sqrt(np.diag(Sigma))
        R = Sigma / np.outer(d, d)
        out = fit_glasso_ebic(R, n_samples=15)
        assert out["n_edges"] >= max(p, 10)

    def test_fired_flag_consistent_with_n_edges(self):
        # When fallback_fired, the unconstrained argmin is sub-degree
        # and the final n_edges came from the constrained selector;
        # when not fired, the two coincide.
        for seed in range(4):
            R = _make_factor_correlation(20, seed=seed)
            for n in [25, 80, 250]:
                out = fit_glasso_ebic(R, n_samples=n)
                if out["fallback_fired"]:
                    assert out["n_edges_unconstr"] < max(20, 10)
                    assert out["n_edges"] >= max(20, 10)
                else:
                    assert out["n_edges"] == out["n_edges_unconstr"]
                assert out["n_edges"] >= max(20, 10)

    def test_no_constrained_candidate_raises(self):
        # Pure identity input => GLASSO returns a diagonal precision at
        # every positive lambda (k=0); no grid point clears k>=k_min,
        # so the constrained-BIC selector has nothing to argmin over.
        p = 20
        R = np.eye(p)
        with pytest.raises(RuntimeError, match="no constrained-BIC candidate"):
            fit_glasso_ebic(R, n_samples=100)


# =====================================================================
# run_stage2 end-to-end
# =====================================================================

class TestRunStage2EndToEnd:
    def test_two_snapshots_fixture(self, tmp_path, monkeypatch):
        import src.stage2_precision.glasso_filter as g2
        monkeypatch.setattr(g2, "SNAPSHOTS_DIR", tmp_path)

        snaps = {}
        for i, label in enumerate(["FixtureA", "FixtureB"]):
            R = _make_factor_correlation(20, seed=2026 + i)
            snaps[label] = {
                "R_avg": R, "n_days": 200,
                "tickers": [f"T{j}" for j in range(20)],
                "regime": "fixture",
            }
        results = g2.run_stage2(snaps, force=True)

        assert set(results.keys()) == {"FixtureA", "FixtureB"}
        for label, r in results.items():
            assert np.allclose(r["adjacency"], r["adjacency"].T, atol=1e-8)
            assert np.all(np.isfinite(r["precision"]))
            assert np.all(np.isfinite(r["adjacency"]))
            assert r["n_edges"] >= max(20, 10)
            for key in ("lambda_opt", "density", "ebic", "lambda_unconstr",
                        "n_edges_unconstr", "ebic_unconstr", "fallback_fired",
                        "applied_gamma", "applied_delta", "n_failed_lambdas",
                        "lambdas_visited"):
                assert key in r
        # Cache was written.
        assert (tmp_path / "stage2_results.pkl").exists()

    def test_skips_snapshot_with_non_finite_R(self, tmp_path, monkeypatch, capsys):
        import src.stage2_precision.glasso_filter as g2
        monkeypatch.setattr(g2, "SNAPSHOTS_DIR", tmp_path)

        R_good = _make_factor_correlation(15)
        R_bad = R_good.copy()
        R_bad[0, 0] = np.nan
        snaps = {
            "Good": {"R_avg": R_good, "n_days": 150,
                     "tickers": [f"T{i}" for i in range(15)], "regime": "ok"},
            "Bad":  {"R_avg": R_bad,  "n_days": 150,
                     "tickers": [f"T{i}" for i in range(15)], "regime": "bad"},
        }
        results = g2.run_stage2(snaps, force=True)
        assert "Good" in results
        assert "Bad" not in results

    def test_loads_from_cache(self, tmp_path, monkeypatch):
        import src.stage2_precision.glasso_filter as g2
        monkeypatch.setattr(g2, "SNAPSHOTS_DIR", tmp_path)

        R = _make_factor_correlation(15)
        snaps = {"X": {"R_avg": R, "n_days": 150,
                       "tickers": [f"T{i}" for i in range(15)],
                       "regime": "calm"}}
        first = g2.run_stage2(snaps, force=True)
        # Mutate the input; cache load should return the original.
        snaps["X"]["n_days"] = -1
        second = g2.run_stage2(snaps, force=False)
        assert first["X"]["n_edges"] == second["X"]["n_edges"]


# =====================================================================
# Solver failure handling (FloatingPointError skip path)
# =====================================================================

class TestSolverFailureHandling:
    """When sklearn's graphical_lasso raises FloatingPointError on a
    grid point, fit_glasso_ebic skips it, increments n_failed_lambdas,
    and continues. Other exception types propagate."""

    def test_floatingpoint_error_skipped_not_propagated(
            self, monkeypatch, capsys):
        import src.stage2_precision.glasso_filter as g2

        call_count = {"n": 0}
        real_glasso = g2.graphical_lasso

        def flaky_glasso(R, alpha, **kw):
            call_count["n"] += 1
            if call_count["n"] <= 3:
                raise FloatingPointError("Non SPD result: synthetic")
            return real_glasso(R, alpha, **kw)

        monkeypatch.setattr(g2, "graphical_lasso", flaky_glasso)

        R = _make_factor_correlation(20)
        out = g2.fit_glasso_ebic(R, n_samples=200)
        assert out["n_edges"] >= max(20, 10)
        assert out["n_failed_lambdas"] >= 3
        assert "non-convergent" in capsys.readouterr().out

    def test_unexpected_exception_propagates(self, monkeypatch):
        import src.stage2_precision.glasso_filter as g2

        def broken_glasso(R, alpha, **kw):
            raise RuntimeError("genuine bug, not a solver hiccup")

        monkeypatch.setattr(g2, "graphical_lasso", broken_glasso)
        R = _make_factor_correlation(15)
        with pytest.raises(RuntimeError, match="genuine bug"):
            g2.fit_glasso_ebic(R, n_samples=200)


# =====================================================================
# EBIC gamma-tier active branches via fit_glasso_ebic
# =====================================================================

class TestEbicGammaActiveTier:
    """Integration-level coverage that the gamma=0.25 and gamma=0.5
    EBIC tiers are reachable through fit_glasso_ebic and yield a
    well-defined, floor-clearing graph. On the n=500 main panel only
    gamma=0 fires; these tests exercise the small-panel branches that
    activate in the multipanel sweep."""

    @staticmethod
    def _factor_R(p, n_factors=3, scale=0.4, seed=2026):
        rng = np.random.default_rng(seed)
        F = rng.standard_normal((p, n_factors)) * scale
        Sigma = F @ F.T + np.eye(p)
        d = np.sqrt(np.diag(Sigma))
        return Sigma / np.outer(d, d)

    def test_gamma_high_tier_full_path(self):
        # n/p = 8.82 -> gamma = 0.5 tier
        p = 50
        n = 441
        assert (n / p) >= EBIC_RATIO_MID
        R = self._factor_R(p)
        out = fit_glasso_ebic(R, n_samples=n)
        assert out["applied_gamma"] == EBIC_GAMMA
        assert out["n_edges"] >= max(p, 10)
        assert np.isfinite(out["ebic"])

    def test_gamma_mid_tier_full_path(self):
        # n/p = 4.0 -> gamma = 0.25 tier
        p = 50
        n = 200
        assert EBIC_RATIO_LOW <= (n / p) < EBIC_RATIO_MID
        R = self._factor_R(p)
        out = fit_glasso_ebic(R, n_samples=n)
        assert out["applied_gamma"] == EBIC_GAMMA_MID
        assert out["n_edges"] >= max(p, 10)
        assert np.isfinite(out["ebic"])

    def test_gamma_zero_tier_full_path(self):
        # n/p = 1.0 -> gamma = 0 tier (covers the n=500 main panel)
        p = 50
        n = 50
        assert (n / p) < EBIC_RATIO_LOW
        R = self._factor_R(p)
        out = fit_glasso_ebic(R, n_samples=n)
        assert out["applied_gamma"] == 0.0


# =====================================================================
# Early-stop counter behaviour
# =====================================================================

class TestEarlyStopBehaviour:
    """The grid traversal terminates once 8 consecutive converged
    points have worsened the BIC relative to the previous point AND a
    constrained candidate exists. We exercise this with a synthetic
    BIC trajectory injected via monkey-patched compute_ebic."""

    def test_early_stop_breaks_grid_before_full_sweep(self, monkeypatch):
        import src.stage2_precision.glasso_filter as g2

        call_idx = {"k": 0}

        def faked_ebic(prec, S, n_samples, gamma=0.0):
            # Index-monotone trajectory: minimum at i=0, then strict
            # monotone increase. With patience=8 the loop should break
            # after the 9th converged call (1 best + 8 consecutive
            # worse), well before exhausting the 25-point grid.
            i = call_idx["k"]
            call_idx["k"] += 1
            return -100.0 + float(i)

        monkeypatch.setattr(g2, "compute_ebic", faked_ebic)
        R = _make_factor_correlation(20)
        out = g2.fit_glasso_ebic(R, n_samples=200)
        # Compute calls = converged lambdas visited (one per try block).
        assert call_idx["k"] < 25, \
            f"early-stop did not engage: visited {call_idx['k']} of 25"
        assert call_idx["k"] >= 9, \
            "early-stop fired too early (need patience+1 to break)"
        assert out["n_edges"] >= max(20, 10)

    def test_lambdas_visited_does_not_exceed_grid_size(self):
        # The grid has 25 points; visited + failed must not exceed it.
        R = _make_factor_correlation(15)
        out = fit_glasso_ebic(R, n_samples=200)
        assert len(out["lambdas_visited"]) <= 25
        assert (len(out["lambdas_visited"])
                + out["n_failed_lambdas"]) <= 25


# =====================================================================
# Spectral safety net invoked through the full fit_glasso_ebic path
# =====================================================================

class TestSpectralSafetyNetIntegration:
    """Safety net runs exactly once per fit (post-shrinkage, pre-GLASSO).
    On the production n=500 panel the trigger does not fire because
    shrinkage alone clears the threshold; the integration test only
    verifies the call site, not the firing condition (covered by the
    unit test in TestSpectralSafetyNet)."""

    def test_safety_net_called_exactly_once_per_fit(self, monkeypatch):
        import src.stage2_precision.glasso_filter as g2

        call_count = {"n": 0}
        real_safety = g2._apply_spectral_safety_net

        def counted(M):
            call_count["n"] += 1
            return real_safety(M)

        monkeypatch.setattr(g2, "_apply_spectral_safety_net", counted)

        R = _make_factor_correlation(15)
        g2.fit_glasso_ebic(R, n_samples=200)
        assert call_count["n"] == 1, (
            f"safety net must run once per fit; got {call_count['n']}")
