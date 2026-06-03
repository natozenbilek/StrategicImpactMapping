"""Stage 2: sparse precision-matrix filter via Graphical LASSO.

Given the Stage-1 R̄ we estimate a sparse precision Omega = R̄^{-1} under
an L1 penalty and extract the partial correlation

    rho_{ij|rest} = -omega_{ij} / sqrt(omega_{ii} omega_{jj}).

Penalty selection: EBIC (Foygel-Drton 2010) with an n/p-tiered gamma,
which reduces to plain BIC on the n=500 panel (n/p <= 1.01). An
identity-target shrinkage (Ledoit-Wolf form, delta drawn from an
n/p-tiered schedule rather than the LW optimal-shrinkage formula)
keeps the sklearn CD solver inside its convergence basin on the
short crisis windows. A spectral safety net follows as a defensive
guard on the shrunken matrix; on the n=500 panel the shrinkage tier
already lifts every post-shrinkage lambda_min above the trigger so
the safety net is a no-op in production, retained for downstream
panel configurations with smaller delta. When the unconstrained-BIC
argmin yields a sub-degree graph (k < max(p, 10)) we fall back to
the constrained-BIC selector

    lambda_fallback = argmin_{lambda : k(lambda) >= max(p, 10)} BIC(lambda).
"""
import warnings
import numpy as np
from sklearn.covariance import graphical_lasso
from sklearn.exceptions import ConvergenceWarning
import pickle

from src.config import (
    SNAPSHOTS_DIR, GLASSO_LAMBDA_RANGE, GLASSO_N_LAMBDAS,
    EBIC_GAMMA, EBIC_GAMMA_MID, EBIC_RATIO_LOW, EBIC_RATIO_MID,
)

# Non-convergent lambdas are normal at the dense end of the grid;
# handled by try/except + fallback. Suppress per-lambda warnings.
warnings.filterwarnings("ignore", category=ConvergenceWarning,
                        module="sklearn.covariance")

# Identity-target shrinkage tiers (paper Table tab:shrinkage-tiers,
# appendix app:stage2 "Identity-target shrinkage tiers"). Lower n/p =>
# heavier identity pull.
SHRINKAGE_TIERS = (
    (0.20, 0.55),
    (0.30, 0.45),
    (0.50, 0.35),
    (1.00, 0.15),
    (2.00, 0.05),
    (float("inf"), 0.005),
)

# Spectral safety net thresholds (paper sec:stage2, appendix
# app:rbar_psd + Algorithm alg:stage2_glasso). Any eigenvalue below
# TRIGGER is shifted up so the smallest eigenvalue lands at TARGET >> 0,
# guaranteeing strict PSD for graphical_lasso.
SPECTRAL_TRIGGER = 1e-6
SPECTRAL_TARGET = 1e-4


def _gamma_for_ratio(ratio):
    """EBIC γ tier given n/p ratio (paper sec:stage2, appendix
    app:stage2 \"The EBIC penalty for choosing lambda\")."""
    if ratio < EBIC_RATIO_LOW:
        return 0.0
    if ratio < EBIC_RATIO_MID:
        return EBIC_GAMMA_MID
    return EBIC_GAMMA


def _delta_for_ratio(ratio):
    """Identity-target shrinkage δ tier given n/p ratio."""
    for hi, delta in SHRINKAGE_TIERS:
        if ratio < hi:
            return delta
    return SHRINKAGE_TIERS[-1][1]


def _apply_spectral_safety_net(M):
    """Shift M's spectrum up if λ_min < SPECTRAL_TRIGGER.

    Strictly PSDifies the input. No-op when M is already SPD with
    λ_min >= SPECTRAL_TRIGGER. Returns the (possibly shifted) matrix.
    """
    p = M.shape[0]
    lam_min = float(np.linalg.eigvalsh(M).min())
    if lam_min < SPECTRAL_TRIGGER:
        out = M + (SPECTRAL_TARGET - lam_min) * np.eye(p)
    else:
        out = M
    lam_min_out = float(np.linalg.eigvalsh(out).min())
    assert lam_min_out >= SPECTRAL_TRIGGER - 1e-12, (
        f"spectral safety net failed: input lam_min={lam_min:.2e}, "
        f"output lam_min={lam_min_out:.2e}, trigger={SPECTRAL_TRIGGER:.2e}")
    return out


def compute_ebic(precision, sample_cov, n_samples, gamma=EBIC_GAMMA):
    """EBIC = -2 * L(Omega; S) + k log n + 4 k gamma log p.

    L = (n/2) (log|Omega| - tr(S Omega)). k counts off-diagonal nonzeros
    of Omega (i.e. undirected edges). Returns +inf on non-PD precision.

    The slogdet sign check alone misses even-p negative-definite matrices
    (det(-I_p) = (-1)^p = +1 for even p), so we add an explicit eigenvalue
    check below.
    """
    p = precision.shape[0]
    sign, logdet = np.linalg.slogdet(precision)
    if sign <= 0 or not np.isfinite(logdet):
        return np.inf
    if float(np.linalg.eigvalsh((precision + precision.T) / 2).min()) <= 0:
        return np.inf

    loglik = 0.5 * n_samples * (logdet - np.trace(sample_cov @ precision))
    mask = np.abs(precision) > 1e-10
    np.fill_diagonal(mask, False)
    k = mask.sum() // 2
    return -2 * loglik + k * np.log(n_samples) + 4 * k * gamma * np.log(p)


def fit_glasso_ebic(corr_matrix, n_samples, n_lambdas=GLASSO_N_LAMBDAS):
    """GLASSO + EBIC selection + constrained-BIC fallback.

    Returns the chosen result with the unconstrained-BIC argmin also
    exposed (lambda_unconstr / n_edges_unconstr / fallback_fired) so the
    multipanel cache can be re-audited without re-running.
    """
    assert corr_matrix.ndim == 2 and corr_matrix.shape[0] == corr_matrix.shape[1], \
        f"corr_matrix must be square, got shape {corr_matrix.shape}"
    assert np.allclose(corr_matrix, corr_matrix.T, atol=1e-8), \
        "corr_matrix must be symmetric"
    assert np.allclose(np.diag(corr_matrix), 1.0, atol=1e-6), \
        "corr_matrix must have unit diagonal (correlation, not covariance)"

    p = corr_matrix.shape[0]
    min_edges = max(p, 10)  # k >= p floor for constrained-BIC

    ratio = n_samples / p
    gamma = _gamma_for_ratio(ratio)
    delta = _delta_for_ratio(ratio)

    if not np.all(np.isfinite(corr_matrix)):
        bad = (~np.isfinite(corr_matrix)).sum()
        raise ValueError(
            f"GLASSO input has {bad} non-finite entries out of "
            f"{corr_matrix.size}.")

    # n/p-tiered Ledoit-Wolf identity shrinkage. Conservative vs LW
    # optimum but keeps graphical_lasso(mode='cd', max_iter=500)
    # converging without solver chaining.
    corr_matrix = (1.0 - delta) * corr_matrix + delta * np.eye(p)
    corr_matrix = _apply_spectral_safety_net(corr_matrix)

    # High -> low traversal so we start sparse (fast) and densify; the
    # EBIC U-curve minimum lies somewhere in the middle.
    lambda_min, lambda_max = GLASSO_LAMBDA_RANGE
    lambdas = np.logspace(np.log10(lambda_max), np.log10(lambda_min), n_lambdas)

    best_ebic = np.inf
    best_result = None
    fallback_result = None  # BIC-optimal lambda s.t. k >= min_edges
    consecutive_worse = 0
    EARLY_STOP_PATIENCE = 8
    prev_ebic = np.inf
    n_failed = 0
    lambdas_visited = []  # converged lambdas in traversal order

    for lam in lambdas:
        try:
            cov_est, prec_est = graphical_lasso(
                corr_matrix, alpha=lam, max_iter=500, mode="cd")
            ebic = compute_ebic(prec_est, corr_matrix, n_samples, gamma=gamma)
            mask = np.abs(prec_est) > 1e-10
            np.fill_diagonal(mask, False)
            n_edges = mask.sum() // 2

            rec = {
                "precision": prec_est, "covariance": cov_est, "lambda": lam,
                "ebic": ebic, "n_edges": n_edges,
                "density": n_edges / (p * (p - 1) / 2),
            }
            lambdas_visited.append(float(lam))
            if ebic < best_ebic:
                best_ebic = ebic; best_result = rec
                consecutive_worse = 0
            elif ebic > prev_ebic:
                consecutive_worse += 1
            else:
                consecutive_worse = 0
            prev_ebic = ebic

            if n_edges >= min_edges:
                if fallback_result is None or ebic < fallback_result["ebic"]:
                    fallback_result = rec

            # Stop only after EBIC clearly past its minimum AND a viable
            # fallback already exists; non-convergent lambdas don't tick
            # the counter so middle-of-grid solver noise doesn't trigger
            # spurious early termination.
            if (consecutive_worse >= EARLY_STOP_PATIENCE
                    and fallback_result is not None
                    and best_result is not None):
                break
        except (FloatingPointError, ValueError):
            # sklearn graphical_lasso raises FloatingPointError on "Non SPD
            # result" and ValueError on the dense-end coord-descent blow-ups.
            # Other exception types reflect genuine bugs and propagate.
            n_failed += 1
            continue

    if n_failed > 0:
        print(f"    [glasso] {n_failed}/{len(lambdas)} lambdas non-convergent "
              f"(n/p={ratio:.2f}, delta={delta:.2f})")
    if best_result is None:
        raise RuntimeError("GLASSO failed for all lambda values")

    fallback_fired = (best_result["n_edges"] < min_edges
                      and fallback_result is not None)
    if fallback_fired:
        print(f"    EBIC-optimal too sparse ({best_result['n_edges']} edges), "
              f"using fallback (λ={fallback_result['lambda']:.4f}, "
              f"{fallback_result['n_edges']} edges)")
        chosen = dict(fallback_result)
    else:
        chosen = dict(best_result)

    # Expose unconstrained argmin alongside the constrained selection.
    chosen["lambda_unconstr"] = best_result["lambda"]
    chosen["n_edges_unconstr"] = best_result["n_edges"]
    chosen["ebic_unconstr"] = best_result["ebic"]
    chosen["fallback_fired"] = fallback_fired
    # Tier and grid-traversal diagnostics for assertion-suite re-audit
    # and downstream sensitivity scripts (paper/_kmin_sensitivity.py).
    chosen["applied_gamma"] = float(gamma)
    chosen["applied_delta"] = float(delta)
    chosen["n_failed_lambdas"] = int(n_failed)
    chosen["lambdas_visited"] = list(lambdas_visited)

    # Algorithm 1 invariant: the returned graph clears the k>=k_min floor.
    # Reachable only when the unconstrained argmin is sub-degree AND no
    # grid point ever produced k>=k_min (e.g. degenerate input or a grid
    # too coarse to resolve the BIC minimum).
    if chosen["n_edges"] < min_edges:
        raise RuntimeError(
            f"GLASSO produced no constrained-BIC candidate: best λ="
            f"{best_result['lambda']:.4f} has {best_result['n_edges']} edges < "
            f"k_min={min_edges}, and no λ on the grid yielded k>=k_min. "
            f"Snapshot data may be degenerate or the λ grid too coarse.")
    return chosen


def precision_to_partial_corr(precision):
    """rho_{ij|rest} = -omega_{ij} / sqrt(omega_ii omega_jj); diag set to 1."""
    d = np.sqrt(np.diag(precision))
    d_outer = np.maximum(np.outer(d, d), 1e-10)
    partial_corr = -precision / d_outer
    np.fill_diagonal(partial_corr, 1.0)
    return partial_corr


def build_adjacency_matrix(partial_corr, precision):
    """Edge (i,j) iff |omega_{ij}| > 1e-10; weight is |rho_{ij|rest}|."""
    mask = np.abs(precision) > 1e-10
    np.fill_diagonal(mask, False)
    adj = np.where(mask, np.abs(partial_corr), 0.0)
    adj = 0.5 * (adj + adj.T)  # absorb solver-induced asymmetry
    np.fill_diagonal(adj, 0.0)
    return adj


def run_stage2(snapshot_correlations, force=False):
    """Driver: GLASSO + EBIC selection per snapshot, cache results."""
    cache_path = SNAPSHOTS_DIR / "stage2_results.pkl"
    if not force and cache_path.exists():
        print("[Stage 2] Loading cached results...")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print("[Stage 2] GLASSO Precision Matrix Filtering...")
    results = {}
    for label, snap_data in snapshot_correlations.items():
        R = snap_data["R_avg"]
        n_days = snap_data["n_days"]
        tickers = snap_data["tickers"]
        regime = snap_data["regime"]
        p = R.shape[0]
        print(f"\n  Snapshot: {label} ({p} assets, {n_days} days, {regime})")

        if not np.all(np.isfinite(R)):
            bad = (~np.isfinite(R)).sum()
            print(f"    SKIP: R_avg has {bad} non-finite entries out of {R.size}")
            continue

        g = fit_glasso_ebic(R, n_samples=n_days)
        pc = precision_to_partial_corr(g["precision"])
        adj = build_adjacency_matrix(pc, g["precision"])

        # Stage 2 -> Stage 3 boundary invariants (Algorithm 1 postconditions).
        assert np.all(np.isfinite(g["precision"])), \
            f"{label}: precision contains non-finite entries"
        assert np.allclose(g["precision"], g["precision"].T, atol=1e-8), \
            f"{label}: precision not symmetric"
        assert float(np.linalg.eigvalsh(
            (g["precision"] + g["precision"].T) / 2).min()) > 0, \
            f"{label}: precision not positive-definite"
        assert np.allclose(adj, adj.T, atol=1e-8), \
            f"{label}: adjacency not symmetric"
        assert np.all(np.isfinite(adj)), \
            f"{label}: adjacency contains non-finite entries"
        assert g["n_edges"] >= max(p, 10), \
            f"{label}: n_edges={g['n_edges']} violates k>=max(p,10)={max(p,10)}"

        results[label] = {
            "adjacency": adj, "partial_corr": pc, "precision": g["precision"],
            "tickers": tickers, "regime": regime,
            "R_avg": R,
            "lambda_opt": g["lambda"], "n_edges": g["n_edges"],
            "density": g["density"], "ebic": g["ebic"],
            "lambda_unconstr": g["lambda_unconstr"],
            "n_edges_unconstr": g["n_edges_unconstr"],
            "ebic_unconstr": g["ebic_unconstr"],
            "fallback_fired": g["fallback_fired"],
            "applied_gamma": g["applied_gamma"],
            "applied_delta": g["applied_delta"],
            "n_failed_lambdas": g["n_failed_lambdas"],
            "lambdas_visited": g["lambdas_visited"],
        }
        print(f"    λ* = {g['lambda']:.4f}, edges = {g['n_edges']}, "
              f"density = {g['density']:.4f}")

    with open(cache_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\n  Cached to {cache_path}")
    return results


if __name__ == "__main__":
    stage1_path = SNAPSHOTS_DIR / "stage1_results.pkl"
    if stage1_path.exists():
        with open(stage1_path, "rb") as f:
            stage1 = pickle.load(f)
        results = run_stage2(stage1["snapshot_correlations"])
        print("\n[Stage 2] Complete!")
        for label, r in results.items():
            print(f"  {label}: {r['n_edges']} edges, density={r['density']:.4f}")
    else:
        print("Run Stage 1 first.")
