"""Stage 3: lead-follower direction assignment.

For every Stage-2 edge (i, j) a two-test cascade assigns one of
{i->j, j->i, mutual, undetermined}:

1. Lagged partial correlation of x_{t-1} and y_t after partialling out
   y_{t-1} and the top-5 contemporaneous controls of the *source* asset
   (top-5 by |R_avg[s, k]|, k != i, j). Only a positive significant
   residual is admitted as evidence that s leads t; a significant
   negative residual is treated as anti-correlation rather than
   direction. By Frisch-Waugh-Lovell the residual correlation is the
   partial correlation given the controls.
2. Bivariate Granger F at a single lag (statsmodels SSR-F). A
   significant result contributes an unsigned floor confidence
   (default 0.5).

Per-direction confidence c_{ij}, c_{ji} are max-aggregated across the
two tests. Decision rule (default tau = DIRECTION_RATIO_THRESH = 1.5):

    both zero            -> drop
    exactly one positive -> assign that direction
    one >= tau * other   -> assign the dominant direction
    both positive, neither dominates -> mutual

The controls and the dominance threshold are exposed as keyword
arguments on ``assign_directions_snapshot`` so the sensitivity sweep in
``src/stage3_direction/sensitivity_sweep.py`` can vary them without
mutating the module-level config defaults.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
from scipy import stats
import pickle

from src.config import (
    SNAPSHOTS_DIR, SNAPSHOTS, LAG_ORDER,
    GRANGER_MAX_LAG, SIGNIFICANCE_LEVEL, DIRECTION_RATIO_THRESH,
)

GRANGER_FLOOR_DEFAULT = 0.5
TOP_K_CONTROLS = 5


def _top_k_controls(R_avg, source_col, exclude_cols, k=TOP_K_CONTROLS):
    """Stage-2 column indices of the k assets most correlated with ``source_col``
    in ``R_avg``, excluding the entries in ``exclude_cols``.

    ``R_avg`` is the window-averaged DCC correlation produced by Stage 1
    (paper §3.3, appendix Algorithm 1: top-k by |\\bar R_{sk}|).
    """
    assert R_avg.ndim == 2 and R_avg.shape[0] == R_avg.shape[1], \
        f"R_avg must be square, got {R_avg.shape}"
    abs_corrs = np.abs(R_avg[source_col]).astype(float, copy=True)
    abs_corrs[list(exclude_cols)] = -1.0
    # numpy puts NaN last under argsort; mask them out explicitly.
    abs_corrs[~np.isfinite(abs_corrs)] = -1.0
    k_eff = min(k, int((abs_corrs > -1.0).sum()))
    if k_eff <= 0:
        return np.empty(0, dtype=int)
    return np.argsort(abs_corrs)[-k_eff:]


def lagged_partial_correlation(x, y, controls, lag=LAG_ORDER):
    """corr(y_t - W beta_y, x_{t-lag} - W beta_x), W = [y_{t-lag}, controls].

    By Frisch-Waugh-Lovell this is the partial correlation of x_{t-lag}
    and y_t conditional on W (paper §3.3 eq:fwl). Returns (0, 1) on
    degenerate input.

    ``controls`` is the (T, k) contemporaneous control panel (aligned
    with ``y``); it is sliced to ``controls[lag:]`` internally so the
    conditioning row matches y_t.
    """
    T = len(x)
    assert len(y) == T, f"x/y length mismatch: {T} vs {len(y)}"
    if T < lag + 10:
        return 0.0, 1.0

    y_t = y[lag:]
    x_lagged = x[:T - lag]
    y_lagged = y[:T - lag]
    if controls is not None and controls.shape[1] > 0:
        assert controls.shape[0] == T, \
            f"controls T={controls.shape[0]} mismatches x T={T}"
        ctrl_only = np.column_stack([y_lagged, controls[lag:]])
    else:
        ctrl_only = y_lagged.reshape(-1, 1)

    try:
        # numpy 2.0's BLAS-backed matmul emits spurious divide/overflow
        # warnings for well-conditioned inputs under some OpenBLAS builds;
        # suppress them locally — the explicit isfinite checks below
        # still guard correctness.
        with np.errstate(divide="ignore", over="ignore", under="ignore",
                         invalid="ignore"):
            beta_y, *_ = np.linalg.lstsq(ctrl_only, y_t, rcond=None)
            beta_x, *_ = np.linalg.lstsq(ctrl_only, x_lagged, rcond=None)
            if not (np.all(np.isfinite(beta_y)) and np.all(np.isfinite(beta_x))):
                return 0.0, 1.0
            resid_y = y_t - ctrl_only @ beta_y
            resid_x = x_lagged - ctrl_only @ beta_x
        if not (np.all(np.isfinite(resid_x)) and np.all(np.isfinite(resid_y))):
            return 0.0, 1.0
        r, p = stats.pearsonr(resid_x, resid_y)
        if not np.isfinite(r) or not np.isfinite(p):
            return 0.0, 1.0
        return float(r), float(p)
    except (np.linalg.LinAlgError, ValueError):
        return 0.0, 1.0


def granger_test(x, y, lag=GRANGER_MAX_LAG):
    """SSR-F Granger F-test at a single lag: does x at t-lag Granger-cause y_t?

    Returns (F, p, lag). No best-of-lags scan: the function tests
    *exactly* ``lag`` so the cascade is free of multiple-comparison
    selection bias. The sensitivity sweep iterates over lag values
    externally.
    """
    assert lag >= 1, f"lag must be >= 1, got {lag}"
    if len(x) < 3 * lag + 5:
        return 0.0, 1.0, lag
    try:
        df = pd.DataFrame(np.column_stack([y, x]), columns=["y", "x"])
        # ``verbose`` was removed from grangercausalitytests in
        # statsmodels 0.14; the function is silent by default now.
        result = grangercausalitytests(df, maxlag=lag)
        if lag not in result:
            return 0.0, 1.0, lag
        fstat = float(result[lag][0]["ssr_ftest"][0])
        pval = float(result[lag][0]["ssr_ftest"][1])
        if not (np.isfinite(fstat) and np.isfinite(pval)):
            return 0.0, 1.0, lag
        return fstat, pval, lag
    except (ValueError, RuntimeError, np.linalg.LinAlgError):
        return 0.0, 1.0, lag


def _empty_snapshot_result(p):
    return {"directed_adj": np.zeros((p, p)), "edge_details": [],
            "n_directed": 0, "n_bidirectional": 0,
            "n_dropped": 0, "n_input_edges": 0}


def assign_directions_snapshot(adjacency, returns_window, tickers, R_avg,
                               method="both",
                               alpha=SIGNIFICANCE_LEVEL,
                               tau=DIRECTION_RATIO_THRESH,
                               lag=LAG_ORDER,
                               granger_lag=GRANGER_MAX_LAG,
                               granger_floor=GRANGER_FLOOR_DEFAULT,
                               verbose=True):
    """Run the two-test cascade for every Stage-2 edge in this snapshot.

    Returns dict with directed_adj (asymmetric for one-way, symmetric
    for mutual), per-edge diagnostics, and n_directed / n_bidirectional
    / n_dropped / n_input_edges.

    ``R_avg`` is the Stage-1 window-averaged DCC correlation matrix; it
    drives the FWL conditioning-set selection per direction. ``method``
    controls which tests contribute to the per-direction confidence:
    ``"both"`` (default), ``"lagged_partial"`` (FWL only), or
    ``"granger"`` (Granger only).
    """
    p = adjacency.shape[0]
    assert adjacency.shape == (p, p), \
        f"adjacency must be square, got {adjacency.shape}"
    assert np.allclose(np.diag(adjacency), 0.0, atol=1e-10), \
        "adjacency diagonal must be zero"
    assert R_avg.shape == (p, p), \
        f"R_avg shape {R_avg.shape} != adjacency shape {(p, p)}"
    assert len(tickers) == p, \
        f"tickers length {len(tickers)} != p {p}"
    assert method in ("both", "lagged_partial", "granger"), \
        f"unknown method {method!r}"
    assert 0.0 < alpha < 1.0, f"alpha out of (0,1): {alpha}"
    assert tau >= 1.0, f"tau must be >= 1 to define dominance, got {tau}"

    available = [t for t in tickers if t in returns_window.columns]

    # Stage-3 needs a balanced sub-panel; Stage-1c's >=80% coverage
    # filter keeps this dropna small.
    sub = returns_window[available].dropna(how="any")
    returns_arr = sub.values
    ticker_to_subcol = {t: i for i, t in enumerate(available)}
    if len(returns_arr) < 30:
        if verbose:
            print(f"    WARNING: only {len(returns_arr)} balanced days for "
                  f"{len(available)} tickers — skipping direction assignment")
        return _empty_snapshot_result(p)

    directed_adj = np.zeros((p, p))
    edge_details = []
    edges = [(i, j) for i in range(p) for j in range(i + 1, p) if adjacency[i, j] > 0]
    if verbose:
        print(f"    Testing {len(edges)} edges for directionality "
              f"(alpha={alpha}, tau={tau}, lag={lag}, granger_lag={granger_lag}, "
              f"granger_floor={granger_floor})...")

    for idx, (i, j) in enumerate(edges):
        ti, tj = tickers[i], tickers[j]
        if ti not in ticker_to_subcol or tj not in ticker_to_subcol:
            continue
        ci, cj = ticker_to_subcol[ti], ticker_to_subcol[tj]
        x = returns_arr[:, ci]; y = returns_arr[:, cj]

        # Direction (i -> j): controls = top-5 by |R_avg[i, *]| excluding
        # i, j. Direction (j -> i): controls = top-5 by |R_avg[j, *]|.
        # See Algorithm 1 in appendix app:stage3.
        controls_ij = _build_controls(R_avg, i, [i, j], tickers,
                                      ticker_to_subcol, returns_arr)
        controls_ji = _build_controls(R_avg, j, [i, j], tickers,
                                      ticker_to_subcol, returns_arr)

        ij_conf = ji_conf = 0.0
        ij_pval = ji_pval = 1.0
        detail = {"edge": (ti, tj), "weight": adjacency[i, j]}

        if method in ("lagged_partial", "both"):
            corr_ij, pval_ij = lagged_partial_correlation(x, y, controls_ij, lag=lag)
            corr_ji, pval_ji = lagged_partial_correlation(y, x, controls_ji, lag=lag)
            detail["lpc_ij"] = (corr_ij, pval_ij)
            detail["lpc_ji"] = (corr_ji, pval_ji)
            # Only positive significant residuals count as lead evidence;
            # negative significant is anti-correlation, not direction.
            if pval_ij < alpha and corr_ij > 0:
                ij_conf = max(ij_conf, corr_ij); ij_pval = min(ij_pval, pval_ij)
            if pval_ji < alpha and corr_ji > 0:
                ji_conf = max(ji_conf, corr_ji); ji_pval = min(ji_pval, pval_ji)

        if method in ("granger", "both"):
            f_ij, gpval_ij, lag_ij = granger_test(x, y, lag=granger_lag)
            f_ji, gpval_ji, lag_ji = granger_test(y, x, lag=granger_lag)
            detail["gc_ij"] = (f_ij, gpval_ij, lag_ij)
            detail["gc_ji"] = (f_ji, gpval_ji, lag_ji)
            if gpval_ij < alpha:
                ij_conf = max(ij_conf, granger_floor); ij_pval = min(ij_pval, gpval_ij)
            if gpval_ji < alpha:
                ji_conf = max(ji_conf, granger_floor); ji_pval = min(ji_pval, gpval_ji)

        if ij_conf == 0 and ji_conf == 0:
            detail["direction"] = "undetermined"
        elif ij_conf > 0 and ji_conf == 0:
            directed_adj[i, j] = adjacency[i, j]
            detail["direction"] = f"{ti} -> {tj}"
        elif ji_conf > 0 and ij_conf == 0:
            directed_adj[j, i] = adjacency[i, j]
            detail["direction"] = f"{tj} -> {ti}"
        elif ij_conf >= tau * ji_conf:
            directed_adj[i, j] = adjacency[i, j]
            detail["direction"] = f"{ti} -> {tj}"
        elif ji_conf >= tau * ij_conf:
            directed_adj[j, i] = adjacency[i, j]
            detail["direction"] = f"{tj} -> {ti}"
        else:
            directed_adj[i, j] = adjacency[i, j]
            directed_adj[j, i] = adjacency[i, j]
            detail["direction"] = f"{ti} <-> {tj}"
        edge_details.append(detail)

        if verbose and (idx + 1) % 200 == 0:
            print(f"      Processed {idx + 1}/{len(edges)} edges...")

    # Postcondition asserts: structural invariants of the directed graph.
    assert directed_adj.shape == (p, p)
    assert np.allclose(np.diag(directed_adj), 0.0, atol=1e-12), \
        "directed_adj has nonzero diagonal (self-loop leak)"
    assert np.all(directed_adj >= 0), "directed_adj has negative weight"

    n_directed = int(np.sum((directed_adj > 0) & (directed_adj.T == 0)))
    n_bidirectional = int(np.sum((directed_adj > 0) & (directed_adj.T > 0))) // 2
    n_input_edges = len(edges)
    n_dropped = n_input_edges - n_directed - n_bidirectional
    assert n_dropped >= 0, \
        f"edge accounting: input={n_input_edges} directed={n_directed} mutual={n_bidirectional}"
    if verbose:
        print(f"    Results: {n_directed} directed, {n_bidirectional} mutual, "
              f"{n_dropped} dropped (no significant direction) / {n_input_edges} input")
    return {
        "directed_adj": directed_adj,
        "edge_details": edge_details,
        "n_directed": n_directed,
        "n_bidirectional": n_bidirectional,
        "n_dropped": int(n_dropped),
        "n_input_edges": n_input_edges,
    }


def _build_controls(R_avg, source_col, exclude_cols, tickers,
                    ticker_to_subcol, returns_arr):
    """Top-k controls in the sub-panel column space for a given source.

    Picks top-k Stage-2 indices by |R_avg[source_col]| (excluding
    ``exclude_cols``), filters to those tickers that survived the
    balanced-sub-panel dropna, and returns the (T, k_eff) control panel
    or None if no controls survive. k_eff may be smaller than k when
    candidates were dropped by the dropna; the FWL is still well-defined.
    """
    if returns_arr.shape[1] <= 2:
        return None
    cand_idx = _top_k_controls(R_avg, source_col, exclude_cols)
    sub_cols = [ticker_to_subcol[tickers[k]] for k in cand_idx
                if tickers[k] in ticker_to_subcol]
    if not sub_cols:
        return None
    return returns_arr[:, sub_cols]


def run_stage3(stage2_results, returns, force=False):
    """Driver: per-snapshot direction assignment.

    ``stage2_results`` must contain a per-snapshot ``R_avg`` entry
    (added by ``run_stage2``); regenerate the Stage-2 cache if loading
    fails this check.
    """
    cache_path = SNAPSHOTS_DIR / "stage3_results.pkl"
    if not force and cache_path.exists():
        print("[Stage 3] Loading cached results...")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print("[Stage 3] Lead-Follower Direction Assignment...")
    results = {}
    for label, s2_data in stage2_results.items():
        adj = s2_data["adjacency"]
        tickers = s2_data["tickers"]
        if "R_avg" not in s2_data:
            raise KeyError(
                f"Stage-2 result for '{label}' has no R_avg field. Regenerate "
                f"Stage 2 with `--stages 2 --force` (run_stage2 stores R_avg "
                f"since the 2026-05-24 Stage-3 audit fix)."
            )
        R_avg = s2_data["R_avg"]

        for snap_label, start, end, regime in SNAPSHOTS:
            if snap_label == label:
                break
        else:
            continue

        mask = ((returns.index >= pd.Timestamp(start)) &
                (returns.index <= pd.Timestamp(end)))
        returns_window = returns.loc[mask]
        if len(returns_window) < 30:
            print(f"  Skipping '{label}': insufficient data ({len(returns_window)} days)")
            continue

        print(f"\n  Snapshot: {label} ({len(returns_window)} days)")
        direction_result = assign_directions_snapshot(
            adj, returns_window, tickers, R_avg, method="both")
        results[label] = {**direction_result, "tickers": tickers,
                          "regime": s2_data["regime"]}

    with open(cache_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\n  Cached to {cache_path}")
    return results


if __name__ == "__main__":
    stage1_path = SNAPSHOTS_DIR / "stage1_results.pkl"
    stage2_path = SNAPSHOTS_DIR / "stage2_results.pkl"
    if stage1_path.exists() and stage2_path.exists():
        with open(stage1_path, "rb") as f:
            stage1 = pickle.load(f)
        with open(stage2_path, "rb") as f:
            stage2 = pickle.load(f)
        returns_path = (Path(__file__).resolve().parent.parent.parent
                        / "data" / "sp500_returns.parquet")
        returns = pd.read_parquet(returns_path)
        results = run_stage3(stage2, returns)
        print("\n[Stage 3] Complete!")
    else:
        print("Run Stages 1 and 2 first.")
