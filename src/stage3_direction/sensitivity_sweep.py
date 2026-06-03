"""Stage 3 cascade joint sensitivity (F3 / app:sens).

Sweeps the lead/follower decision rule over a grid of
(alpha, tau, granger_lag, granger_floor) and tabulates per-snapshot
(n_directed, n_mutual, n_dropped, mutual_fraction). Reuses the
``assign_directions_snapshot`` cascade verbatim — only the threshold
arguments vary, so the sweep is a parametric re-run of Stage 3, not a
separate code path.

Output schema (pickled to ``SNAPSHOTS_DIR / stage3_sensitivity.pkl``)::

    {
      "grid": [
        {"alpha": 0.05, "tau": 1.5, "granger_lag": 1, "granger_floor": 0.5,
         "snapshots": {
            label: {"n_directed": int, "n_mutual": int, "n_dropped": int,
                    "n_input": int, "mutual_fraction": float, "regime": str},
            ...
         },
         "crisis_mean_mutual_fraction": float,
         "noncrisis_mean_mutual_fraction": float,
        },
        ...
      ],
      "headline_cell": (0.05, 1.5, 1, 0.5),
    }
"""
from itertools import product
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from src.config import (
    SNAPSHOTS, SNAPSHOTS_DIR,
    SIGNIFICANCE_LEVEL, DIRECTION_RATIO_THRESH, GRANGER_MAX_LAG, LAG_ORDER,
)
from src.stage3_direction.lead_lag import (
    assign_directions_snapshot, GRANGER_FLOOR_DEFAULT,
)


# Grid spec from appendix F3 / app:sens.
ALPHA_GRID = (0.05, 0.10)
TAU_GRID = (1.2, 1.5, 2.0)
LAG_GRID = (1, 2)
FLOOR_GRID = (0.3, 0.5, 0.7)

HEADLINE_CELL = (SIGNIFICANCE_LEVEL, DIRECTION_RATIO_THRESH,
                 GRANGER_MAX_LAG, GRANGER_FLOOR_DEFAULT)


def _snapshot_window(label, returns):
    for snap_label, start, end, regime in SNAPSHOTS:
        if snap_label == label:
            mask = ((returns.index >= pd.Timestamp(start)) &
                    (returns.index <= pd.Timestamp(end)))
            return returns.loc[mask], regime
    return None, None


def _aggregate_regime(per_snapshot):
    """Crisis vs non-crisis mean mutual-fraction (regime == 'crisis')."""
    crisis = [v["mutual_fraction"] for v in per_snapshot.values()
              if v["regime"] == "crisis" and v["n_input"] > 0]
    noncrisis = [v["mutual_fraction"] for v in per_snapshot.values()
                 if v["regime"] != "crisis" and v["n_input"] > 0]
    return (float(np.mean(crisis)) if crisis else float("nan"),
            float(np.mean(noncrisis)) if noncrisis else float("nan"))


def run_one_cell(stage2_results, returns, alpha, tau, granger_lag,
                 granger_floor, verbose=False):
    """Cascade re-run for every Stage-2 snapshot at one parameter cell."""
    per_snapshot = {}
    for label, s2_data in stage2_results.items():
        returns_window, regime = _snapshot_window(label, returns)
        if returns_window is None or len(returns_window) < 30:
            continue
        if "R_avg" not in s2_data:
            raise KeyError(
                f"Stage-2 result for '{label}' has no R_avg — regenerate "
                f"Stage 2 cache before running the Stage-3 sensitivity sweep."
            )
        res = assign_directions_snapshot(
            s2_data["adjacency"], returns_window, s2_data["tickers"],
            s2_data["R_avg"],
            method="both",
            alpha=alpha, tau=tau, lag=LAG_ORDER,
            granger_lag=granger_lag, granger_floor=granger_floor,
            verbose=verbose,
        )
        n_dir = res["n_directed"]
        n_mut = res["n_bidirectional"]
        n_in = res["n_input_edges"]
        denom = n_dir + n_mut
        per_snapshot[label] = {
            "n_directed": n_dir,
            "n_mutual": n_mut,
            "n_dropped": res["n_dropped"],
            "n_input": n_in,
            "mutual_fraction": (n_mut / denom) if denom > 0 else 0.0,
            "regime": regime,
        }
    crisis_mu, noncrisis_mu = _aggregate_regime(per_snapshot)
    return {
        "alpha": alpha,
        "tau": tau,
        "granger_lag": granger_lag,
        "granger_floor": granger_floor,
        "snapshots": per_snapshot,
        "crisis_mean_mutual_fraction": crisis_mu,
        "noncrisis_mean_mutual_fraction": noncrisis_mu,
    }


def _alpha_tau_lag_cells():
    """12-cell α × τ × maxlag grid at the headline Granger floor."""
    headline_floor = HEADLINE_CELL[3]
    return [(a, t, lag, headline_floor)
            for a, t, lag in product(ALPHA_GRID, TAU_GRID, LAG_GRID)]


def _floor_cells():
    """3-cell Granger-floor sweep at the headline (alpha, tau, lag)."""
    a, t, lag, _ = HEADLINE_CELL
    return [(a, t, lag, f) for f in FLOOR_GRID]


def default_grid():
    """Headline 12-cell α × τ × lag grid plus the 3-cell floor sweep.

    Deduplicates the overlap (the headline cell appears in both).
    """
    cells = _alpha_tau_lag_cells() + _floor_cells()
    seen = set()
    out = []
    for c in cells:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def run_sweep(stage2_results, returns, grid=None, verbose=True,
              cache_path=None, force=False):
    """Run the full Stage-3 sensitivity grid.

    ``grid`` is an iterable of (alpha, tau, granger_lag, granger_floor)
    tuples; ``None`` uses :func:`default_grid` (14 unique cells: 12
    α·τ·lag + 3 c_floor variants at the headline α·τ·lag, with the
    headline c_floor=0.5 cell deduplicated across the two sub-grids).
    """
    if cache_path is None:
        cache_path = SNAPSHOTS_DIR / "stage3_sensitivity.pkl"
    cache_path = Path(cache_path)
    if not force and cache_path.exists():
        if verbose:
            print(f"[Stage 3 sweep] Loading cached results from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    if grid is None:
        grid = default_grid()
    grid = list(grid)
    results = {"grid": [], "headline_cell": HEADLINE_CELL}
    for i, (alpha, tau, gl, gf) in enumerate(grid):
        if verbose:
            print(f"[Stage 3 sweep] Cell {i + 1}/{len(grid)}: "
                  f"alpha={alpha}, tau={tau}, granger_lag={gl}, "
                  f"granger_floor={gf}")
        cell = run_one_cell(stage2_results, returns,
                            alpha=alpha, tau=tau,
                            granger_lag=gl, granger_floor=gf,
                            verbose=False)
        results["grid"].append(cell)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(results, f)
    if verbose:
        print(f"[Stage 3 sweep] Cached to {cache_path}")
    return results


def to_dataframe(sweep_results):
    """Flatten the sweep dict into a long-form DataFrame.

    Columns: alpha, tau, granger_lag, granger_floor, snapshot, regime,
    n_directed, n_mutual, n_dropped, n_input, mutual_fraction.
    """
    rows = []
    for cell in sweep_results["grid"]:
        for label, snap in cell["snapshots"].items():
            rows.append({
                "alpha": cell["alpha"],
                "tau": cell["tau"],
                "granger_lag": cell["granger_lag"],
                "granger_floor": cell["granger_floor"],
                "snapshot": label,
                "regime": snap["regime"],
                "n_directed": snap["n_directed"],
                "n_mutual": snap["n_mutual"],
                "n_dropped": snap["n_dropped"],
                "n_input": snap["n_input"],
                "mutual_fraction": snap["mutual_fraction"],
            })
    return pd.DataFrame(rows)


def headline_table(sweep_results):
    """Per-cell aggregate (crisis vs non-crisis mean mutual fraction)."""
    rows = []
    for cell in sweep_results["grid"]:
        rows.append({
            "alpha": cell["alpha"],
            "tau": cell["tau"],
            "granger_lag": cell["granger_lag"],
            "granger_floor": cell["granger_floor"],
            "crisis_mu_mean": cell["crisis_mean_mutual_fraction"],
            "noncrisis_mu_mean": cell["noncrisis_mean_mutual_fraction"],
            "separation": (cell["crisis_mean_mutual_fraction"]
                           - cell["noncrisis_mean_mutual_fraction"]),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    stage2_path = SNAPSHOTS_DIR / "stage2_results.pkl"
    if not stage2_path.exists():
        print("Run Stages 1 and 2 first.")
    else:
        with open(stage2_path, "rb") as f:
            stage2 = pickle.load(f)
        returns_path = (Path(__file__).resolve().parent.parent.parent
                        / "data" / "sp500_returns.parquet")
        returns = pd.read_parquet(returns_path)
        sweep = run_sweep(stage2, returns)
        print("\n[Stage 3 sweep] Headline table:")
        print(headline_table(sweep).to_string(index=False))
