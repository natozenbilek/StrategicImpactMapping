"""Stage 1: A-DCC GARCH estimation (Cappiello-Engle-Sheppard 2006).

Two-step QMLE: univariate GARCH(1,1)-Student-t margins per ticker
standardise returns to z_{i,t}; the scalar A-DCC parameters (a,b,g)
are then fit by L-BFGS-B on the resulting residual panel. The window-
averaged correlation R̄ per snapshot feeds Stage 2.

The panel is left unbalanced: delisted tickers contribute their pre-
delisting window, late-IPO tickers their post-IPO window, so crisis-
era cross-sections remain observable where they were active.
"""
import time
import pickle

import numpy as np
import pandas as pd
from arch import arch_model
from scipy.optimize import minimize

from src.config import (
    SNAPSHOTS_DIR, GARCH_P, GARCH_Q, GARCH_DIST, SNAPSHOTS,
    ADCC_MAX_ITER, ADCC_SUBSET_SIZE,
)
from src.utils.dcc_core import (
    adcc_intercept, adcc_qt_update, normalize_qt_to_rt,
    adcc_loglikelihood_contribution,
)


# Short-coverage tickers (e.g. MER reconstructed from a 98-day Wayback
# panel) cannot reach GARCH convergence; the arch_model fit either
# raises or returns degenerate parameters. Below this threshold we
# substitute a static-variance z-scoring fallback so the ticker still
# enters the standardized-residual matrix and the per-snapshot A-DCC.
# Disclosed in paper Lim.(ii) for the augmented-panel run; not used on
# the baseline panel since all tickers there clear the 100-obs floor.
SHORT_COVERAGE_GARCH_THRESHOLD = 200

# Pre-MLE tail clip on the 100-asset balanced subset z-panel. With the
# 2026-05-28 CRSP migration the 1985-2024 window adds three severe
# crash tails (1987 Black Monday, 2008 GFC, 2020 COVID) that push GARCH
# standardised residuals past |z|=10 on a handful of days. Without
# clipping, the rank-one (z z') and (n n') updates inflate N_bar's
# spectral magnitude enough that the closed-form A-DCC intercept
# (1-a-b)Qbar - g Nbar loses PSD even for canonical (a,b,g) seeds, and
# every L-BFGS-B iterate falls into the 1e10 penalty branch. Clipping
# to ±Z_CLIP=10 matches the per-snapshot R-bar build in
# extract_snapshot_correlations and restores PSD on the intercept.
Z_CLIP = 10.0


def fit_single_garch(args):
    """Per-ticker GARCH(p,q)-t fit on the asset's native lifespan.

    Returns are scaled to percent so omega stays out of the 1e-8
    boundary regime. ``rescale=False`` blocks arch's auto-rescale on
    top of that. Returns None for <30 obs, a dict with ``__failure__``
    on any exception, otherwise the parameters, standardised residuals,
    and conditional volatility. Tickers with 30 <= n < 200 obs use a
    static z-score fallback in place of GARCH (see
    SHORT_COVERAGE_GARCH_THRESHOLD).
    """
    ticker, returns_series = args
    try:
        y = returns_series.dropna() * 100
        n = len(y)
        if n < 30:
            return ticker, None
        if n < SHORT_COVERAGE_GARCH_THRESHOLD:
            mu = float(y.mean())
            sd = float(y.std(ddof=1))
            assert sd > 0, f"{ticker}: zero std on short-coverage panel"
            std_resid = (y - mu) / sd
            cond_vol = pd.Series(sd * np.ones(n), index=y.index)
            assert np.all(np.isfinite(std_resid)), (
                f"{ticker}: short-fallback std_resid non-finite")
            return ticker, {
                "params": {"omega": sd * sd, "alpha": 0.0, "beta": 0.0},
                "std_resid": std_resid,
                "cond_vol": cond_vol,
                "aic": float("nan"), "bic": float("nan"),
                "__short_fallback__": True,
                "__n_obs__": n,
            }
        result = arch_model(y, vol="Garch", p=GARCH_P, q=GARCH_Q,
                            dist=GARCH_DIST, mean="Constant",
                            rescale=False).fit(
            disp="off", show_warning=False)
        std_resid = result.std_resid
        cond_vol = result.conditional_volatility
        # IGARCH-limit tickers that breach alpha+beta<1 at floating-
        # point precision (ALB, CVNA, DELL, VRT on the n=500 panel;
        # CVNA, DELL exactly at 1.0, ALB ~2e-11 above, VRT ~1e-6 above)
        # have undefined unconditional variance but the conditional
        # recursion stays well-posed; pin that the admitted std_resid
        # is finite, since downstream A-DCC has no NaN tolerance.
        assert np.all(np.isfinite(std_resid)), (
            f"{ticker}: non-finite std_resid (n_nonfinite="
            f"{int((~np.isfinite(std_resid)).sum())})")
        assert np.all(cond_vol > 0), (
            f"{ticker}: non-positive cond_vol "
            f"(min={float(cond_vol.min()):.3e})")
        return ticker, {
            "params": {
                "omega": result.params.get("omega", np.nan),
                "alpha": result.params.get("alpha[1]", np.nan),
                "beta": result.params.get("beta[1]", np.nan),
            },
            "std_resid": std_resid,
            "cond_vol": cond_vol,
            "aic": result.aic, "bic": result.bic,
        }
    except Exception as e:
        return ticker, {"__failure__": f"{type(e).__name__}: {str(e)[:120]}"}


def estimate_univariate_garch(returns):
    """Sequential per-asset GARCH fits.

    Sequential because arch Result objects hold compiled-extension refs
    that don't pickle under Windows spawn-based multiprocessing.
    """
    print("[Stage 1, GARCH] Estimating univariate GARCH(1,1)...")
    args_list = [(col, returns[col]) for col in returns.columns]
    results, failed, failure_reasons = {}, [], {}
    for i, args in enumerate(args_list):
        ticker, res = fit_single_garch(args)
        if res is not None and "__failure__" not in res:
            results[ticker] = res
        else:
            failed.append(ticker)
            if res is not None:
                failure_reasons[ticker] = res.get("__failure__", "unknown")
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(args_list)} assets...")

    print(f"  Completed: {len(results)} succeeded, {len(failed)} failed")
    if failed:
        print(f"  Failed tickers: {failed[:10]}{'...' if len(failed) > 10 else ''}")
        for t, r in list(failure_reasons.items())[:3]:
            print(f"    {t}: {r}")
    return results


def build_standardized_residuals(garch_results, returns):
    """Reindex per-asset z series onto the master return grid (unbalanced)."""
    valid_tickers = [t for t in returns.columns if t in garch_results]
    z = pd.DataFrame({t: garch_results[t]["std_resid"] for t in valid_tickers})
    z = z.reindex(returns.index)
    assert z.shape[0] == returns.shape[0]
    assert z.shape[1] == len(valid_tickers)
    print(f"  Standardized residuals: {z.shape[0]} days x {z.shape[1]} "
          f"assets (unbalanced; coverage median "
          f"{int(z.count().median())} obs/ticker)")
    return z


_adcc_eval_count = 0

# Stationarity slack η: keeps L-BFGS-B one step inside the open boundary
# a+b+g<1. Without it, the optimiser can converge to a floating-point
# boundary point that produces an IGARCH-limit A-DCC whose unconditional
# pseudo-correlation is undefined. Matches the η=1e-3 in paper Alg. 1.
ADCC_STATIONARITY_SLACK = 1e-3


def _adcc_loglikelihood(params, z, Q_bar, N_bar):
    """Negative A-DCC QMLE; 1e10 penalty for infeasible iterates."""
    global _adcc_eval_count
    _adcc_eval_count += 1

    a, b, g = params
    T, k = z.shape
    if _adcc_eval_count % 10 == 0:
        print(f"\r  A-DCC optimization: eval #{_adcc_eval_count}, "
              f"a={a:.4f} b={b:.4f} g={g:.4f}  ", end="", flush=True)

    # Box bounds alone don't enforce a+b+g < 1; pushing to boundary
    # destabilises the recursion.
    if a < 0 or b < 0 or g < 0 or (a + b + g) >= 1.0 - ADCC_STATIONARITY_SLACK:
        return 1e10

    const = adcc_intercept(a, b, g, Q_bar, N_bar)
    if np.linalg.eigvalsh(const).min() < -1e-8:
        return 1e10

    Q_t = Q_bar.copy()
    loglik = 0.0
    for t in range(1, T):
        Q_t = adcc_qt_update(Q_t, z[t - 1, :], const, a, b, g)
        R_t = normalize_qt_to_rt(Q_t)
        ll_t = adcc_loglikelihood_contribution(R_t, z[t, :])
        if ll_t is None:
            return 1e10
        loglik += ll_t
    return -loglik


def estimate_adcc(z, rng_seed=2026):
    """L-BFGS-B multi-start MLE for (a, b, g).

    Seven starting points: five fixed and two Dirichlet(1,1,1) random
    draws (RNG seed 2026 for reproducibility). The fifth fixed seed
    (0.05, 0.90, 0.05) sums to 1.00 and is rejected at initialisation
    by the a+b+g >= 1-η guard (η=ADCC_STATIONARITY_SLACK). Best
    log-likelihood across the surviving seeds wins; a multi-modality
    flag fires if the surviving seeds disagree by more than 1e-4 in
    parameter space.

    Tail-clipping: z is clipped to ``[-Z_CLIP, Z_CLIP]`` before Q_bar,
    N_bar, and the per-t Q_t recursion are formed. On the CRSP 1985-2024
    panel the 1987 / 2008 / 2020 crash tails push standardised residuals
    well past |z|=10 on a handful of days, which inflates ``N_bar``
    spectral magnitude enough that the closed-form intercept
    ``(1-a-b) Q_bar - g N_bar`` loses PSD even for canonical (a,b,g)
    starts -- every L-BFGS-B iterate then falls into the 1e10 penalty
    branch (the failure pattern observed on the pre-clip MLE run).
    """
    global _adcc_eval_count
    assert z.ndim == 2 and z.shape[0] > z.shape[1], z.shape
    T, k = z.shape
    print(f"[Stage 1, A-DCC] Estimating A-DCC parameters ({T} obs x {k} assets)...")

    z = np.clip(z, -Z_CLIP, Z_CLIP)

    col_std = z.std(axis=0)
    if np.any(col_std <= 1e-10):
        bad = np.where(col_std <= 1e-10)[0]
        raise ValueError(
            f"A-DCC estimate: {len(bad)} asset column(s) have near-zero "
            f"variance (cols {bad[:5].tolist()}).")

    Q_bar = np.corrcoef(z, rowvar=False)
    if np.any(np.isnan(Q_bar)):
        raise ValueError("A-DCC estimate: Q_bar contains NaN entries.")
    assert Q_bar.shape == (k, k)

    n = np.minimum(z, 0)
    # Apple Accelerate's matmul raises spurious divide-by-zero / overflow /
    # invalid RuntimeWarnings when one operand has many exact zeros (n is
    # min(z, 0); ~50% of entries are zero by construction). The output is
    # finite and matches reference BLAS; we suppress the warning chatter
    # and assert finiteness explicitly so any real numerical breakage
    # still trips the assert.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        N_bar = (n.T @ n) / T
    assert np.all(np.isfinite(N_bar)), "N_bar non-finite (real numerical issue, not the Accelerate spurious warning)"
    assert N_bar.shape == (k, k)

    bounds = [(1e-6, 0.3), (0.5, 0.999), (1e-6, 0.3)]

    fixed_starts = [
        (np.array([0.02, 0.95, 0.01]), "canonical Cappiello et al. 2006"),
        (np.array([0.01, 0.97, 0.01]), "high persistence, low shock"),
        (np.array([0.10, 0.85, 0.02]), "high shock response"),
        (np.array([0.04, 0.94, 0.01]), "moderate persistence, low shock"),
        (np.array([0.05, 0.90, 0.05]), "on-boundary stationarity test"),
    ]
    # Dirichlet(1,1,1) draws lie on the closed simplex (sum == 1), which
    # the a+b+g >= 1-η init guard always rejects. Rescale by u ~ U(0.85,
    # 0.97) so the sum lands strictly inside the open stationarity region
    # (well below 1-η = 0.999) and the random seeds actually contribute
    # to the multi-start search. Fixes the structural-rejection bug noted
    # in the 2026-05-26 audit.
    rng = np.random.default_rng(rng_seed)
    dirichlet_starts = []
    for i in range(2):
        raw = rng.dirichlet([1.0, 1.0, 1.0])
        u = float(rng.uniform(0.85, 0.97))
        dirichlet_starts.append((u * raw, f"Dirichlet draw {i + 1}"))
    all_starts = fixed_starts + dirichlet_starts
    eta = ADCC_STATIONARITY_SLACK

    t0 = time.time()
    surviving = []  # (fun, x, label) for seeds that complete the optimiser
    n_rejected_init = 0
    for si, (x0, label) in enumerate(all_starts):
        sum0 = float(x0.sum())
        print(f"    start {si + 1}/{len(all_starts)} ({label}): "
              f"a0={x0[0]:.3f} b0={x0[1]:.2f} g0={x0[2]:.3f}...",
              end="", flush=True)
        if sum0 >= 1.0 - eta:
            print(f" rejected at init (a0+b0+g0={sum0:.4f} >= 1-{eta:g})")
            n_rejected_init += 1
            continue
        _adcc_eval_count = 0
        try:
            res = minimize(_adcc_loglikelihood, x0, args=(z, Q_bar, N_bar),
                           method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": ADCC_MAX_ITER, "ftol": 1e-8})
        except Exception as exc:
            print(f" failed: {type(exc).__name__}")
            continue
        print(f" LL={-res.fun:.1f} ({_adcc_eval_count} evals)")
        a_, b_, g_ = res.x
        if a_ + b_ + g_ >= 1.0 - eta:
            # Optimiser drifted onto / through the stationarity boundary;
            # discard so we don't crown a non-stationary seed as best.
            print(f"      discarded (post-fit a+b+g={a_ + b_ + g_:.4f} >= 1-η)")
            continue
        surviving.append((res.fun, res.x, label))

    if not surviving:
        raise RuntimeError("A-DCC MLE failed for all starting points")

    surviving.sort(key=lambda t: t[0])
    best_fun, best_x, best_label = surviving[0]
    a, b, g = best_x
    assert a + b + g < 1.0 - eta, (a, b, g, a + b + g)
    assert a >= 0 and b >= 0 and g >= 0, (a, b, g)

    # Multi-modality flag: paper Alg. 1 promises the surviving seeds
    # agree within 1e-4 of θ*. Report the maximum parameter-space gap.
    if len(surviving) >= 2:
        max_dev = max(float(np.max(np.abs(x - best_x)))
                      for _, x, _ in surviving[1:])
        multi_modality = bool(max_dev > 1e-4)
    else:
        max_dev = 0.0
        multi_modality = False

    print(f"\n  A-DCC params (best of {len(surviving)} surviving / "
          f"{len(all_starts)} attempted, {n_rejected_init} rejected at init): "
          f"a={a:.4f}, b={b:.4f}, g={g:.4f} "
          f"(sum={a + b + g:.4f}) [{time.time() - t0:.0f}s total]")
    print(f"  Best seed: '{best_label}'. Max param spread across "
          f"surviving seeds: {max_dev:.2e} "
          f"({'multi-modality flag SET' if multi_modality else 'within 1e-4 — unimodal'})")

    return {"a": float(a), "b": float(b), "g": float(g),
            "Q_bar": Q_bar, "N_bar": N_bar,
            "success": True, "loglik": -best_fun,
            "best_seed_label": best_label,
            "n_surviving_seeds": len(surviving),
            "n_attempted_seeds": len(all_starts),
            "n_rejected_init": n_rejected_init,
            "max_param_spread": max_dev,
            "multi_modality_flag": multi_modality}


def extract_snapshot_correlations(z_df, adcc_params, snapshots=None,
                                  min_coverage=0.80):
    """One window-averaged R̄ per snapshot.

    z is winsorised to [-10, 10] before the per-window recursion: |z|>10
    indicates GARCH mis-specification on a low-quality ticker and a
    single such outlier can push Q_t out of the PSD cone via the
    rank-one update a z z'. The MLE step does NOT winsorise (see
    estimate_adcc); the clip applies only to the per-window R̄ build.
    """
    if snapshots is None:
        snapshots = SNAPSHOTS
    a, b, g = adcc_params["a"], adcc_params["b"], adcc_params["g"]
    assert 0 <= a and 0 <= b and 0 <= g and a + b + g < 1, (a, b, g)

    snapshot_correlations = {}
    for label, start, end, regime in snapshots:
        window = z_df.loc[pd.Timestamp(start):pd.Timestamp(end)]
        window_len = len(window)
        if window_len == 0:
            print(f"  WARNING: No data for snapshot '{label}' ({start} to {end})")
            continue

        obs_per_ticker = window.count()
        min_obs = max(30, int(min_coverage * window_len))
        alive = obs_per_ticker[obs_per_ticker >= min_obs].index
        sub = window[alive].dropna(how="any")
        if sub.shape[1] < 10 or len(sub) < 30:
            print(f"  WARNING: '{label}' has only {sub.shape[1]} alive "
                  f"tickers x {len(sub)} days — skipping")
            continue

        col_std = sub.std(axis=0)
        sub = sub.loc[:, col_std > 1e-8]
        z_snap = np.clip(sub.values, -10.0, 10.0)
        T_snap, k_snap = z_snap.shape

        Q_bar = np.corrcoef(z_snap, rowvar=False)
        if np.any(np.isnan(Q_bar)):
            print(f"  WARNING: '{label}' Q_bar has NaN — skipping")
            continue
        n_neg = np.minimum(z_snap, 0)
        # Same spurious Accelerate matmul warnings as in estimate_adcc;
        # see the comment there for the rationale behind np.errstate.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            N_bar = (n_neg.T @ n_neg) / T_snap
        assert np.all(np.isfinite(N_bar)), (
            f"'{label}': N_bar non-finite (real numerical issue)")

        const = adcc_intercept(a, b, g, Q_bar, N_bar)
        Q_t = Q_bar.copy()
        R_acc = np.zeros_like(Q_bar)
        n_steps = n_skipped = 0
        for t in range(1, T_snap):
            Q_t_new = adcc_qt_update(Q_t, z_snap[t - 1, :], const, a, b, g)
            diag_Q = np.diag(Q_t_new)
            # Defensive PSD guard: a single bad outlier post-winsorisation
            # can still drive diag(Q_t) negative; skip that step so NaN
            # doesn't propagate into R_avg.
            if np.any(~np.isfinite(diag_Q)) or np.any(diag_Q <= 0):
                n_skipped += 1
                continue
            Q_t = Q_t_new
            R_t = normalize_qt_to_rt(Q_t)
            if np.any(~np.isfinite(R_t)):
                n_skipped += 1
                continue
            R_acc += R_t
            n_steps += 1

        if n_steps == 0:
            print(f"  WARNING: '{label}' produced no valid R_t step "
                  f"({n_skipped} skipped) — skipping snapshot")
            continue

        R_avg = R_acc / n_steps
        np.fill_diagonal(R_avg, 1.0)
        assert R_avg.shape == (k_snap, k_snap)
        assert np.all(np.isfinite(R_avg))
        assert np.allclose(R_avg, R_avg.T, atol=1e-10)
        snapshot_correlations[label] = {
            "R_avg": R_avg, "n_days": T_snap,
            "regime": regime, "tickers": list(sub.columns),
        }
        iu = np.triu_indices_from(R_avg, k=1)
        skipped_msg = f", skipped {n_skipped}" if n_skipped > 0 else ""
        print(f"  Snapshot '{label}': {T_snap} days x {k_snap} alive tickers, "
              f"mean corr = {R_avg[iu].mean():.3f} "
              f"({n_steps} valid steps{skipped_msg})")

    return snapshot_correlations


def run_stage1(returns, n_assets=None, force=False):
    """Full Stage 1: GARCH margins + A-DCC MLE + per-snapshot R̄."""
    cache_path = SNAPSHOTS_DIR / "stage1_results.pkl"
    if not force and cache_path.exists():
        print("[Stage 1] Loading cached results...")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Coverage-based sub-universe preserves delisted high-coverage names
    # (e.g. CFC) that alphabetical truncation would lose.
    if n_assets is not None:
        coverage = returns.count().sort_values(ascending=False)
        keep = set(coverage.head(n_assets).index)
        keep = [c for c in returns.columns if c in keep]
        returns = returns[keep]
        print(f"  Using top {len(keep)} assets by coverage for estimation")

    garch_results = estimate_univariate_garch(returns)
    z_df = build_standardized_residuals(garch_results, returns)

    # (a,b,g) governs transient dynamics; Q̄, N̄ encode the cross-
    # section and are restored per snapshot below. Pooling the scalars
    # on the always-alive ≥99%-coverage subset and re-instating the
    # unconditional matrices per window is the decomposition justified
    # in \appref{app:stage1}.
    T_full = len(z_df)
    coverage = z_df.count()
    always_alive = coverage[coverage >= 0.99 * T_full].index.tolist()
    # Multipanel sweep edge case: ADV-ranked top-N for small N can pull
    # in late-listed mega-caps (META, GOOGL, etc.) whose coverage is
    # below the 99 % bar, leaving the always-alive set empty. Fall back
    # to the top-ADCC_SUBSET_SIZE tickers by coverage and dropna to the
    # common window so the MLE still has a balanced residual matrix.
    subset_threshold = 0.99
    if not always_alive:
        ranked = coverage.sort_values(ascending=False)
        always_alive = ranked.head(ADCC_SUBSET_SIZE).index.tolist()
        subset_threshold = float(ranked.head(ADCC_SUBSET_SIZE).min()) / T_full
        print(f"  [Stage 1] No tickers above 99% coverage on this panel; "
              f"using top-{len(always_alive)} by coverage "
              f"(min coverage {subset_threshold:.3f}*T)")
    print(f"  A-DCC parameter estimation: {len(always_alive)} always-alive "
          f"tickers (>={subset_threshold:.2f} of {T_full}-day residual panel)")

    subset_cols = always_alive[:ADCC_SUBSET_SIZE]
    z_subset = z_df[subset_cols].dropna(how="any").values
    assert z_subset.shape[1] == len(subset_cols)
    # When the fallback fires, the common balanced window can be much
    # shorter than the ≥99% case. Require at least 252 trading days (~1 year)
    # so the L-BFGS-B MLE has enough information.
    assert z_subset.shape[0] >= 252, (
        f"A-DCC MLE input too short: {z_subset.shape[0]} balanced days "
        f"after dropna on {len(subset_cols)}-ticker subset")
    adcc_params = estimate_adcc(z_subset)

    adcc_params["Q_bar_subset"] = adcc_params["Q_bar"]
    adcc_params["N_bar_subset"] = adcc_params["N_bar"]
    adcc_params["subset_tickers"] = list(subset_cols)
    adcc_params["subset_k"] = len(subset_cols)
    adcc_params["subset_T"] = int(z_subset.shape[0])
    adcc_params["full_k"] = z_df.shape[1]

    snapshot_correlations = extract_snapshot_correlations(z_df, adcc_params)

    results = {
        "garch_results": garch_results, "z_df": z_df,
        "adcc_params": adcc_params,
        "snapshot_correlations": snapshot_correlations,
        "tickers": z_df.columns.tolist(),
    }
    with open(cache_path, "wb") as f:
        pickle.dump(results, f)
    print(f"  Cached to {cache_path}")
    return results


if __name__ == "__main__":
    from src.stage1_data.download import run_download
    info, prices, returns = run_download()
    results = run_stage1(returns, n_assets=50)  # dev run on 50 assets
    print("\n[Stage 1] Complete!")
    print(f"  A-DCC params: a={results['adcc_params']['a']:.4f}, "
          f"b={results['adcc_params']['b']:.4f}, "
          f"g={results['adcc_params']['g']:.4f}")
    print(f"  Snapshots computed: {list(results['snapshot_correlations'].keys())}")
