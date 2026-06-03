"""Recompute ADF stationarity diagnostics on the post-CRSP standardised
residual panel.

Writes results/snapshots/stage1_adf_diagnostics.json with full-panel and
per-snapshot rejection counts at alpha=0.05, matching the schema the
old N=500 cache used so paper/_generate_extra.py can rebuild
tables/adf_rejection_per_snapshot.tex without touching its loader.
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import SNAPSHOTS

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "results" / "snapshots" / "stage1_results.pkl"
OUT = ROOT / "results" / "snapshots" / "stage1_adf_diagnostics.json"

ALPHA = 0.05


def adf_pvalue(series: pd.Series) -> float:
    """Return ADF p-value or NaN if test fails (constant series etc)."""
    s = series.dropna().values
    if len(s) < 30:
        return float("nan")
    try:
        # autolag="AIC" matches the May-25 sidecar's default
        _, p, *_ = adfuller(s, autolag="AIC", regression="c")
        return float(p)
    except Exception:
        return float("nan")


def main():
    t0 = time.time()
    print(f"[adf] Loading {CACHE}")
    with open(CACHE, "rb") as f:
        s1 = pickle.load(f)
    z = s1["z_df"]
    z.index = pd.to_datetime(z.index)
    T, p = z.shape
    print(f"[adf] z_df = ({T}, {p})")

    # Full-panel ADF on each ticker's full non-missing residual lifespan
    print("[adf] Full-panel ADF on per-ticker lifespans...")
    full_pvals = {}
    for i, t in enumerate(z.columns):
        full_pvals[t] = adf_pvalue(z[t])
        if (i + 1) % 200 == 0:
            print(f"  [adf] {i+1}/{p} done")
    full_pvals = pd.Series(full_pvals)
    n_tested_full = full_pvals.notna().sum()
    n_reject_full = (full_pvals < ALPHA).sum()
    full_panel = {
        "n_days": int(T),
        "n_tickers_total": int(p),
        "n_tested": int(n_tested_full),
        "n_reject": int(n_reject_full),
        "reject_rate": float(n_reject_full / n_tested_full) if n_tested_full else 0.0,
    }
    print(f"[adf] full-panel reject {n_reject_full}/{n_tested_full} = "
          f"{full_panel['reject_rate']:.4f}")

    # Per-snapshot ADF on each ticker's window-restricted residual series
    print("[adf] Per-snapshot ADF...")
    per_snapshot = {}
    for label, start, end, regime in SNAPSHOTS:
        z_win = z.loc[start:end]
        # Pre-filter tickers with at least 30 non-missing obs in the window
        cnt = z_win.count()
        elig = cnt[cnt >= 30].index
        pvals = {}
        for t in elig:
            pvals[t] = adf_pvalue(z_win[t])
        pvals = pd.Series(pvals)
        n_tested = pvals.notna().sum()
        n_reject = (pvals < ALPHA).sum()
        per_snapshot[label] = {
            "n_days": int(z_win.shape[0]),
            "n_tested": int(n_tested),
            "n_reject": int(n_reject),
            "reject_rate": float(n_reject / n_tested) if n_tested else 0.0,
        }
        print(f"  {label:25s}  n_days={z_win.shape[0]:>4d}  "
              f"tested={n_tested:>4d}  reject={n_reject:>4d}  "
              f"rate={per_snapshot[label]['reject_rate']:.4f}")

    payload = {
        "alpha": ALPHA,
        "full_panel": full_panel,
        "per_snapshot": per_snapshot,
        "wall_seconds": time.time() - t0,
    }
    with open(OUT, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[adf] Wrote {OUT}  (wall = {payload['wall_seconds']:.1f} s)")


if __name__ == "__main__":
    main()
