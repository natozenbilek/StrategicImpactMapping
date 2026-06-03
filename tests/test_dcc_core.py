"""
Unit tests for the A-DCC numerical primitives in :mod:`src.utils.dcc_core`.

The tests verify the four mathematical guarantees that the rest of
Stage 1 relies on:

1. The intercept formula Omega = (1 - a - b) Qbar - g Nbar reproduces
   the closed-form stationarity condition E[Q_t] = Qbar of Cappiello,
   Engle, and Sheppard (2006, eq. 13).
2. The Q_t update preserves shape, symmetry, and the asymmetry sign:
   joint negative shocks must drive Q_t strictly above the response
   to the corresponding positive shocks.
3. The Q_t -> R_t normalisation produces a symmetric matrix with unit
   diagonal and off-diagonal entries in [-1, 1].
4. The Cholesky-based log-likelihood contribution is finite on
   well-conditioned inputs and falls back gracefully (returning
   ``None``) on a singular correlation matrix.

Run with ``python -m pytest tests/ -v``.
"""
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.dcc_core import (
    adcc_intercept, adcc_qt_update, normalize_qt_to_rt,
    adcc_loglikelihood_contribution
)


# --- Fixtures --------------------------------------------------------

def _make_synthetic_data(T=500, k=10, seed=42):
    """
    Synthesise a standardised-residual panel with mild correlation.

    Generates IID standard normals, applies a low-rank perturbation
    L L' (with L drawn from N(0, 0.09)) to a unit-diagonal target,
    transforms via the Cholesky factor of the resulting correlation
    matrix, then re-standardises columnwise to mean zero and unit
    variance.

    Parameters
    ----------
    T : int, optional
        Number of time-series observations.
    k : int, optional
        Number of cross-sectional series.
    seed : int, optional
        Seed for ``np.random.RandomState`` to fix the panel for
        reproducible test runs.

    Returns
    -------
    z : ndarray of shape (T, k)
        Synthetic standardised-residual panel.
    """
    rng = np.random.RandomState(seed)
    z = rng.randn(T, k)
    # Inject mild cross-sectional dependence so that correlation
    # tests are not exercised on a trivially diagonal target.
    L = rng.randn(k, k) * 0.3
    cov = np.eye(k) + L @ L.T
    d_inv = np.diag(1.0 / np.sqrt(np.diag(cov)))
    corr = d_inv @ cov @ d_inv
    chol = np.linalg.cholesky(corr)
    z = z @ chol.T
    # Re-standardise so the empirical first and second moments match
    # the standard-normal assumption used in the A-DCC likelihood.
    z = (z - z.mean(axis=0)) / z.std(axis=0)
    return z


@pytest.fixture
def synthetic_data():
    """Default-size synthetic standardised-residual panel for tests."""
    return _make_synthetic_data()


@pytest.fixture
def small_data():
    """Compact ``T=200``, ``k=5`` panel for fast unit tests."""
    return _make_synthetic_data(T=200, k=5, seed=123)


# --- Tests: A-DCC core functions -------------------------------------

class TestADCCIntercept:
    """Tests for the A-DCC unconditional intercept ``(1-a-b)Qbar - g Nbar``."""

    def test_returns_correct_shape(self):
        """The intercept must inherit the ``(k, k)`` shape of ``Qbar``."""
        k = 5
        Q_bar = np.eye(k)
        N_bar = np.eye(k) * 0.3
        result = adcc_intercept(0.02, 0.95, 0.01, Q_bar, N_bar)
        assert result.shape == (k, k)

    def test_identity_when_no_asymmetry(self):
        """With g = 0 the intercept reduces to (1 - a - b) Qbar."""
        k = 5
        Q_bar = np.eye(k)
        N_bar = np.zeros((k, k))
        a, b = 0.02, 0.95
        result = adcc_intercept(a, b, 0, Q_bar, N_bar)
        expected = (1 - a - b) * Q_bar
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_stationarity_condition(self):
        """E[Q_t] equals Qbar under the unconditional A-DCC moment."""
        k = 5
        rng = np.random.RandomState(42)
        z = rng.randn(1000, k)
        Q_bar = np.corrcoef(z, rowvar=False)
        n = np.minimum(z, 0)
        N_bar = (n.T @ n) / len(z)

        a, b, g = 0.02, 0.93, 0.02

        # Derivation (Cappiello, Engle, Sheppard 2006, Proposition 1):
        #   Omega        = (1 - a - b) Qbar - g Nbar
        #   E[Q_t]       = Omega + a E[zz'] + b E[Q_t] + g E[nn']
        #                = Omega + a Qbar + b E[Q_t] + g Nbar
        #   (1 - b) E[Q_t] = (1 - a - b) Qbar - g Nbar + a Qbar + g Nbar
        #                  = (1 - b) Qbar
        #   E[Q_t]       = Qbar
        Omega = adcc_intercept(a, b, g, Q_bar, N_bar)
        E_Qt = np.linalg.solve(
            (1 - b) * np.eye(k * k),
            (Omega + a * Q_bar + g * N_bar).flatten()
        ).reshape(k, k)
        np.testing.assert_allclose(E_Qt, Q_bar, atol=1e-8)


class TestQtUpdate:
    """Tests for the recursive ``Q_t`` update of A-DCC."""

    def test_returns_correct_shape(self):
        """The recursion preserves the input ``(k, k)`` shape."""
        k = 5
        Q_prev = np.eye(k)
        z_prev = np.random.randn(k)
        const = np.eye(k) * 0.03
        result = adcc_qt_update(Q_prev, z_prev, const, 0.02, 0.95, 0.01)
        assert result.shape == (k, k)

    def test_symmetric(self):
        """``Q_t`` stays symmetric when ``Q_{t-1}`` and the intercept are."""
        k = 5
        rng = np.random.RandomState(42)
        Q_prev = np.eye(k) + rng.randn(k, k) * 0.1
        Q_prev = (Q_prev + Q_prev.T) / 2  # symmetrise for a valid Q_{t-1}
        z_prev = rng.randn(k)
        const = np.eye(k) * 0.03
        result = adcc_qt_update(Q_prev, z_prev, const, 0.02, 0.95, 0.01)
        np.testing.assert_allclose(result, result.T, atol=1e-10)

    def test_negative_residuals_affect_qt(self):
        """Joint negative shocks raise Q_t relative to symmetric ones."""
        k = 5
        Q_prev = np.eye(k)
        z_neg = -np.ones(k)
        z_pos = np.ones(k)
        const = np.eye(k) * 0.03

        Q_neg = adcc_qt_update(Q_prev, z_neg, const, 0.02, 0.95, 0.05)
        Q_pos = adcc_qt_update(Q_prev, z_pos, const, 0.02, 0.95, 0.05)

        # The asymmetric term g (n n') is non-zero only on negative
        # entries, so the negative-shock update inherits an extra
        # rank-one increment over the positive-shock update.
        assert np.mean(Q_neg) > np.mean(Q_pos)


class TestNormalization:
    """Tests for the ``diag(Q)^{-1/2} Q diag(Q)^{-1/2}`` normalisation."""

    def test_diagonal_ones(self):
        """The normalised matrix has unit diagonal entries."""
        k = 5
        rng = np.random.RandomState(42)
        Q = np.eye(k) + rng.randn(k, k) * 0.1
        Q = (Q + Q.T) / 2
        R = normalize_qt_to_rt(Q)
        np.testing.assert_allclose(np.diag(R), np.ones(k), atol=1e-10)

    def test_symmetric(self):
        """The normalisation preserves symmetry."""
        k = 5
        rng = np.random.RandomState(42)
        Q = np.eye(k) + rng.randn(k, k) * 0.1
        Q = (Q + Q.T) / 2
        R = normalize_qt_to_rt(Q)
        np.testing.assert_allclose(R, R.T, atol=1e-10)

    def test_bounded(self):
        """Off-diagonal correlations lie in [-1, 1] modulo round-off."""
        k = 5
        rng = np.random.RandomState(42)
        Q = np.eye(k) + rng.randn(k, k) * 0.1
        Q = (Q + Q.T) / 2
        R = normalize_qt_to_rt(Q)
        assert np.all(R >= -1 - 1e-10)
        assert np.all(R <= 1 + 1e-10)


class TestLogLikelihood:
    """Tests for the per-step A-DCC quasi-log-likelihood contribution."""

    def test_finite_output(self):
        """The contribution is finite on a well-conditioned input."""
        k = 5
        R = np.eye(k)
        z = np.random.randn(k)
        ll = adcc_loglikelihood_contribution(R, z)
        assert ll is not None
        assert np.isfinite(ll)

    def test_identity_correlation(self):
        """At R = I the contribution collapses to -0.5 (0 + z'z - z'z) = 0."""
        k = 5
        R = np.eye(k)
        z = np.random.randn(k)
        ll = adcc_loglikelihood_contribution(R, z)
        # log|I| = 0 and z' I^{-1} z = z' z, so the volatility term
        # cancels exactly and the contribution is identically zero.
        assert abs(ll) < 1e-10

    def test_singular_matrix_returns_none(self):
        """Singular ``R`` (rank-0 zero matrix) returns ``None`` rather than NaN."""
        k = 5
        R = np.zeros((k, k))  # rank 0; both Cholesky and slogdet must fail
        z = np.random.randn(k)
        ll = adcc_loglikelihood_contribution(R, z)
        assert ll is None


class TestGARCHPercentScaleInvariance:
    """The percent pre-scale on returns must leave the standardised
    residuals unchanged. Paper §A.4.1 calls this scale-invariance
    explicitly; the codepath in :func:`src.stage1_data.dcc_garch.fit_single_garch`
    multiplies returns by 100 before the arch fit and relies on
    z = ε/σ being independent of the multiplicative pre-scale.
    """

    def test_std_resid_invariant_under_constant_scale(self):
        """Closed-form: GARCH(1,1) with (ω,α,β) on r and (c²ω,α,β) on c·r
        produces identical standardised residuals z = ε/σ.

        Run on a hand-driven recursion so the test isolates the algebraic
        scale-invariance from the iterative arch optimiser. This is the
        property that justifies the percent pre-scale in
        :func:`src.stage1_data.dcc_garch.fit_single_garch` —
        rescale=False, then r ← 100 r is safe iff the recursion is
        invariant up to the matched (ω, c²ω) pair.
        """
        rng = np.random.RandomState(20260523)
        T = 1500
        eps_innov = rng.standard_normal(T)
        omega, alpha, beta = 0.05, 0.07, 0.90
        c = 100.0

        # Native scale path.
        sigma2_n = np.empty(T); eps_n = np.empty(T)
        sigma2_n[0] = 1.0; eps_n[0] = np.sqrt(sigma2_n[0]) * eps_innov[0]
        for t in range(1, T):
            sigma2_n[t] = omega + alpha * (eps_n[t - 1] ** 2) + beta * sigma2_n[t - 1]
            eps_n[t] = np.sqrt(sigma2_n[t]) * eps_innov[t]
        z_native = eps_n / np.sqrt(sigma2_n)

        # Scaled path with matched (c²ω, α, β) and same innovations.
        sigma2_s = np.empty(T); eps_s = np.empty(T)
        sigma2_s[0] = c ** 2 * 1.0
        eps_s[0] = np.sqrt(sigma2_s[0]) * eps_innov[0]
        for t in range(1, T):
            sigma2_s[t] = (c ** 2 * omega + alpha * (eps_s[t - 1] ** 2)
                           + beta * sigma2_s[t - 1])
            eps_s[t] = np.sqrt(sigma2_s[t]) * eps_innov[t]
        z_scaled = eps_s / np.sqrt(sigma2_s)

        # ε_s = c ε_n and σ_s = c σ_n hold exactly under matched ω scaling,
        # so the ratio is bit-identical up to floating-point round-off.
        np.testing.assert_allclose(z_native, z_scaled, atol=1e-12, rtol=1e-12)


class TestIGARCHLimitFiniteness:
    """At the IGARCH boundary alpha + beta = 1 the unconditional variance
    is undefined, but the *conditional* variance recursion remains
    well-posed at every step and the standardised residuals stay finite.
    Paper §A.4.1 admits the 5 boundary tickers (ALB, CVNA, DELL, NXPI,
    VRT) on exactly this basis.
    """

    def test_std_resid_finite_on_alpha_plus_beta_eq_one(self):
        """Hand-driven IGARCH(1,1) recursion produces finite z_{i,t} for all t."""
        rng = np.random.RandomState(20260523)
        omega, alpha, beta = 1e-6, 0.1, 0.9  # alpha + beta = 1.0 exactly
        T = 2000
        eps = rng.standard_normal(T)
        sigma2 = np.empty(T)
        eps_t = np.empty(T)
        sigma2[0] = 1.0
        eps_t[0] = np.sqrt(sigma2[0]) * eps[0]
        for t in range(1, T):
            sigma2[t] = omega + alpha * (eps_t[t - 1] ** 2) + beta * sigma2[t - 1]
            eps_t[t] = np.sqrt(sigma2[t]) * eps[t]
        z = eps_t / np.sqrt(sigma2)

        assert np.all(np.isfinite(z))
        assert np.all(sigma2 > 0)


class TestADCCEstimateStationarityGuard:
    """The init-time stationarity guard in :func:`estimate_adcc` must
    reject any seed with a0 + b0 + g0 >= 1 - η before L-BFGS-B sees
    it. This is the behaviour Algorithm 1 (line 4) describes; the
    audit found the previous implementation only enforced it inside
    the objective function.
    """

    def test_on_boundary_seed_rejected_at_init(self):
        """The (0.05, 0.90, 0.05) sum = 1.00 fixed seed must be init-rejected.

        Post-2026-05-26 Dirichlet rescale: surviving + rejected_init can be
        strictly less than attempted because a rescaled Dirichlet draw can
        clear the init guard but then drift to a+b+g >= 1-η during
        L-BFGS-B and be discarded by the post-fit guard. The bookkeeping
        below asserts the invariant that survives the post-fit channel.
        """
        from src.stage1_data.dcc_garch import (
            ADCC_STATIONARITY_SLACK, estimate_adcc,
        )
        # A small synthetic z panel that admits A-DCC estimation.
        z = _make_synthetic_data(T=400, k=8, seed=20260523)
        out = estimate_adcc(z)
        # At least the on-boundary fixed seed (0.05, 0.90, 0.05, sum=1.00)
        # must be rejected at init under the η=1e-3 slack.
        assert out["n_rejected_init"] >= 1
        # Surviving plus init-rejected cannot exceed attempted; the gap
        # (if any) is filled by post-fit-guard rejections.
        assert (out["n_surviving_seeds"] + out["n_rejected_init"]
                <= out["n_attempted_seeds"])
        assert out["a"] + out["b"] + out["g"] < 1.0 - ADCC_STATIONARITY_SLACK


class TestEstimateADCCDeterminism:
    """The L-BFGS-B multi-start MLE is seeded by an RNG (default
    ``rng_seed=2026``) and the surviving-seed selection is a sort by
    log-likelihood; two runs on the same input must therefore return
    bit-identical ``(a, b, g)`` and ``loglik``. Guards against an
    unseeded RNG, parallel-order-dependent reduction, or any other
    silent non-determinism in the estimator.
    """

    def test_same_input_same_output(self):
        from src.stage1_data.dcc_garch import estimate_adcc
        z = _make_synthetic_data(T=400, k=8, seed=4242)
        out1 = estimate_adcc(z, rng_seed=2026)
        out2 = estimate_adcc(z, rng_seed=2026)
        assert out1["a"] == out2["a"]
        assert out1["b"] == out2["b"]
        assert out1["g"] == out2["g"]
        assert out1["loglik"] == out2["loglik"]
        assert out1["best_seed_label"] == out2["best_seed_label"]
        assert out1["n_surviving_seeds"] == out2["n_surviving_seeds"]


class TestADCCMultiModalityFlag:
    """The ``multi_modality_flag`` output of :func:`estimate_adcc` must
    track the documented 1e-4 threshold rule from Algorithm 1: the flag
    is True iff the maximum parameter-space gap across surviving seeds
    exceeds 1e-4. Without a self-consistency test the threshold could
    silently drift away from the paper's stated value.
    """

    def test_flag_matches_threshold_rule(self):
        """flag == (max_param_spread > 1e-4) on every run, by construction."""
        from src.stage1_data.dcc_garch import estimate_adcc
        z = _make_synthetic_data(T=400, k=8, seed=20260524)
        out = estimate_adcc(z)
        assert "multi_modality_flag" in out
        assert "max_param_spread" in out
        assert out["multi_modality_flag"] == bool(out["max_param_spread"] > 1e-4)

    def test_metadata_keys_present(self):
        """estimate_adcc must return the full audit-trail metadata set."""
        from src.stage1_data.dcc_garch import estimate_adcc
        z = _make_synthetic_data(T=400, k=8, seed=20260524)
        out = estimate_adcc(z)
        for key in ("a", "b", "g", "Q_bar", "N_bar", "success", "loglik",
                    "best_seed_label", "n_surviving_seeds",
                    "n_attempted_seeds", "n_rejected_init",
                    "max_param_spread", "multi_modality_flag"):
            assert key in out, f"missing key {key!r} in estimate_adcc output"
        # 5 fixed + 2 Dirichlet = 7 seeds attempted by construction.
        assert out["n_attempted_seeds"] == 7
        # Post-2026-05-26 rescale: the on-boundary fixed seed
        # (0.05, 0.90, 0.05) is the only structural init-reject; the two
        # Dirichlet draws now land u<1-η for u~U(0.85, 0.97) and clear
        # the init guard. They may still be rejected post-fit if L-BFGS-B
        # drifts to the boundary, but that bookkeeping is not part of
        # n_rejected_init.
        assert out["n_rejected_init"] >= 1


class TestExtractSnapshotCorrelations:
    """Happy-path and PSD-skip guard behaviour of the per-snapshot
    $\\bar R$ build (``src.stage1_data.dcc_garch
    .extract_snapshot_correlations``).
    """

    @staticmethod
    def _z_frame(T=400, k=12, seed=7):
        z = _make_synthetic_data(T=T, k=k, seed=seed)
        idx = pd.bdate_range("2010-01-01", periods=T)
        cols = [f"TKR{i}" for i in range(k)]
        return pd.DataFrame(z, index=idx, columns=cols)

    @staticmethod
    def _adcc_params(z_df):
        z = z_df.values
        Q_bar = np.corrcoef(z, rowvar=False)
        n = np.minimum(z, 0.0)
        N_bar = (n.T @ n) / len(z)
        return {"a": 0.02, "b": 0.95, "g": 0.01,
                "Q_bar": Q_bar, "N_bar": N_bar}

    def test_happy_path_produces_valid_R_avg(self):
        from src.stage1_data.dcc_garch import extract_snapshot_correlations
        z_df = self._z_frame()
        params = self._adcc_params(z_df)
        snapshots = [("S1", "2010-03-01", "2010-09-30", "baseline")]
        out = extract_snapshot_correlations(z_df, params, snapshots=snapshots)
        assert "S1" in out
        R = out["S1"]["R_avg"]
        k = R.shape[0]
        np.testing.assert_allclose(np.diag(R), np.ones(k), atol=1e-10)
        np.testing.assert_allclose(R, R.T, atol=1e-10)
        assert np.all(R >= -1 - 1e-8)
        assert np.all(R <= 1 + 1e-8)
        assert out["S1"]["regime"] == "baseline"
        assert out["S1"]["n_days"] > 0

    def test_skips_window_with_too_few_alive_tickers(self):
        from src.stage1_data.dcc_garch import extract_snapshot_correlations
        z_df = self._z_frame(k=8)
        params = self._adcc_params(z_df)
        snapshots = [("Tiny", "2010-01-01", "2010-01-05", "baseline")]
        out = extract_snapshot_correlations(z_df, params, snapshots=snapshots)
        assert "Tiny" not in out

    def test_winsorisation_clips_extreme_outliers(self):
        """A single |z|=1e6 outlier inside the window must be clipped
        rather than blowing up the rank-one update; without the clip,
        ``Q_t`` exits the PSD cone and ``R_avg`` becomes NaN."""
        from src.stage1_data.dcc_garch import extract_snapshot_correlations
        z_df = self._z_frame(T=300, k=10, seed=11)
        z_df.iloc[150, 0] = 1e6
        params = self._adcc_params(z_df)
        snapshots = [("Outlier", "2010-01-01", "2010-12-31", "baseline")]
        out = extract_snapshot_correlations(z_df, params, snapshots=snapshots)
        assert "Outlier" in out
        assert np.all(np.isfinite(out["Outlier"]["R_avg"]))

    def test_a_b_g_pinned_to_input_params(self):
        """The per-window recursion must use the (a, b, g) passed in;
        sanity-check that changing the params changes R_avg."""
        from src.stage1_data.dcc_garch import extract_snapshot_correlations
        z_df = self._z_frame()
        snapshots = [("S1", "2010-03-01", "2010-09-30", "baseline")]
        p_low = self._adcc_params(z_df)
        p_low.update({"a": 0.001, "b": 0.998, "g": 0.0005})
        p_high = self._adcc_params(z_df)
        p_high.update({"a": 0.15, "b": 0.80, "g": 0.04})
        out_low = extract_snapshot_correlations(z_df, p_low,
                                                snapshots=snapshots)
        out_high = extract_snapshot_correlations(z_df, p_high,
                                                 snapshots=snapshots)
        R_low = out_low["S1"]["R_avg"]
        R_high = out_high["S1"]["R_avg"]
        iu = np.triu_indices_from(R_low, k=1)
        # Different transient-dynamics scalars produce a different
        # off-diagonal average; equal would mean the params were ignored.
        assert not np.allclose(R_low[iu], R_high[iu], atol=1e-6)


class TestRunStage1CacheRoundTrip:
    """``run_stage1`` is the integration entry point invoked from
    ``run_pipeline.py``. It writes a pickle under
    ``SNAPSHOTS_DIR / 'stage1_results.pkl'`` and short-circuits on
    subsequent calls when ``force=False``. This test pins the
    round-trip on a tiny synthetic returns panel.
    """

    def test_cache_write_and_short_circuit(self, tmp_path, monkeypatch):
        import src.stage1_data.dcc_garch as s1
        # Redirect the cache and shrink the snapshot set so the test
        # finishes in seconds. One short snapshot keeps R_avg work
        # negligible while still exercising the full pipeline path.
        monkeypatch.setattr(s1, "SNAPSHOTS_DIR", tmp_path)
        monkeypatch.setattr(s1, "SNAPSHOTS",
                            [("Smoke", "2018-06-01", "2018-09-30", "baseline")])

        rng = np.random.default_rng(20260524)
        T, k = 800, 12
        idx = pd.bdate_range("2016-01-01", periods=T)
        # Mild persistent volatility so the GARCH fits behave like
        # real-world tickers rather than IID standard normals.
        eps = rng.standard_normal((T, k))
        scale = np.cumprod(1 + rng.normal(0, 0.005, size=(T, k)), axis=0)
        returns = pd.DataFrame(eps * 0.012 * scale, index=idx,
                               columns=[f"TKR{i:02d}" for i in range(k)])

        res = s1.run_stage1(returns, n_assets=None, force=True)
        assert "garch_results" in res
        assert "adcc_params" in res
        assert "snapshot_correlations" in res
        assert (tmp_path / "stage1_results.pkl").exists()
        assert "Smoke" in res["snapshot_correlations"]
        R = res["snapshot_correlations"]["Smoke"]["R_avg"]
        assert np.all(np.isfinite(R))
        np.testing.assert_allclose(np.diag(R), np.ones(R.shape[0]),
                                   atol=1e-10)

        # Second call with force=False must hit the cache and return
        # the same dict by value. We tweak the input returns so a
        # cache miss would visibly change the output; equality proves
        # the short-circuit fired.
        res_cached = s1.run_stage1(returns * 999, n_assets=None,
                                   force=False)
        assert res_cached["adcc_params"]["a"] == res["adcc_params"]["a"]
        assert res_cached["adcc_params"]["b"] == res["adcc_params"]["b"]
        assert res_cached["adcc_params"]["g"] == res["adcc_params"]["g"]
        assert (list(res_cached["snapshot_correlations"].keys())
                == list(res["snapshot_correlations"].keys()))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
