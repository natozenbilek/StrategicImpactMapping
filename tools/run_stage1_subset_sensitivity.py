"""GICS-proportional 100-ticker subset sensitivity for A-DCC scalars.

The default Stage-1 100-asset subset is the first 100 always-alive
tickers in the panel's column order, which is arbitrary (cf.
\\S{app:stage1} ``The 100-asset always-alive subset''). To check that
the column-order choice does not drive the (â, b̂, ĝ) estimate, this
script samples two alternative 100-ticker subsets stratified by GICS
sector, runs the full A-DCC MLE on each, and writes the comparison to a
side-car JSON.

Output: results/snapshots/stage1_subset_sensitivity.json with the
per-subset (a, b, g, sum, loglik), the sector composition, and the
absolute deviation from the cached headline fit.
"""
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.stage1_data.dcc_garch import estimate_adcc

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "results" / "snapshots" / "stage1_results.pkl"
INFO = ROOT / "data" / "sp500_info.parquet"
OUT = ROOT / "results" / "snapshots" / "stage1_subset_sensitivity.json"

SEEDS = [2026, 2027]
SUBSET_SIZE = 100
COVERAGE_THRESHOLD = 0.99


def stratified_subset(always_alive, sectors, seed, k=SUBSET_SIZE):
    """Sample `k` tickers from `always_alive` with sector quotas
    proportional to each sector's share of the always-alive pool.

    Sectors with a fractional quota that rounds to 0 still get one slot
    if the pool contains at least one ticker from that sector — losing
    the smallest sectors entirely would bias the subset toward the
    plurality sectors. Surplus from rounding is distributed by sorting
    sectors on the descending fractional remainder.
    """
    pool = pd.Series({t: sectors[t] for t in always_alive if t in sectors})
    counts = pool.value_counts()
    total = counts.sum()
    rng = np.random.default_rng(seed)
    raw = (counts / total) * k
    quota = np.floor(raw).astype(int)
    # Bump every sector with at least one available ticker to 1
    quota = np.maximum(quota, 1)
    # Adjust to hit exactly k
    while quota.sum() > k:
        # Trim from the largest sector with quota > 1
        candidates = quota[quota > 1]
        sec_to_trim = candidates.idxmax()
        quota[sec_to_trim] -= 1
    while quota.sum() < k:
        # Add to the sector with the largest fractional remainder among
        # those that have room to grow (quota < pool size)
        room = counts - quota
        remainders = (raw - np.floor(raw))[room > 0]
        if remainders.empty:
            break
        sec_to_bump = remainders.idxmax()
        quota[sec_to_bump] += 1

    selected = []
    for sec, n in quota.items():
        sec_pool = pool[pool == sec].index.tolist()
        if n >= len(sec_pool):
            selected.extend(sec_pool)
        else:
            selected.extend(rng.choice(sec_pool, size=n, replace=False).tolist())
    assert len(selected) == k, (len(selected), k)
    return sorted(selected), dict(quota)


def main():
    print(f"[sensitivity] Loading {CACHE}")
    with open(CACHE, "rb") as f:
        s1 = pickle.load(f)
    z_df = s1["z_df"]
    headline = s1["adcc_params"]
    print(f"[sensitivity] Headline (â,b̂,ĝ) = "
          f"({headline['a']:.4f}, {headline['b']:.4f}, {headline['g']:.4f})")
    print(f"[sensitivity] Headline loglik = {headline['loglik']:.2f}")

    print(f"[sensitivity] Loading {INFO}")
    info = pd.read_parquet(INFO)
    sectors = dict(zip(info["Symbol"], info["GICS Sector"]))

    T_full = len(z_df)
    coverage = z_df.count()
    always_alive = coverage[coverage >= COVERAGE_THRESHOLD * T_full].index.tolist()
    print(f"[sensitivity] always_alive = {len(always_alive)} tickers")

    pool_with_sector = [t for t in always_alive if t in sectors]
    pool_counts = Counter(sectors[t] for t in pool_with_sector)
    print(f"[sensitivity] pool sector distribution:")
    for sec, n in sorted(pool_counts.items(), key=lambda x: -x[1]):
        print(f"  {sec:25s}  {n:3d}  ({100*n/len(pool_with_sector):.1f}%)")

    results = []
    for seed in SEEDS:
        print(f"\n[sensitivity] === seed {seed} ===")
        subset, quota = stratified_subset(always_alive, sectors, seed)
        print(f"[sensitivity] subset size = {len(subset)}; quota = {quota}")
        z_subset = z_df[subset].dropna(how="any").values
        print(f"[sensitivity] z_subset shape = {z_subset.shape}")
        assert z_subset.shape[1] == SUBSET_SIZE

        fit = estimate_adcc(z_subset, rng_seed=seed)
        a, b, g = fit["a"], fit["b"], fit["g"]
        s = a + b + g
        results.append({
            "seed": seed,
            "tickers": subset,
            "sector_quota": quota,
            "n_balanced_days": int(z_subset.shape[0]),
            "a": a, "b": b, "g": g,
            "sum_abg": s,
            "loglik": fit["loglik"],
            "best_seed_label": fit["best_seed_label"],
            "n_surviving_seeds": fit["n_surviving_seeds"],
            "n_rejected_init": fit["n_rejected_init"],
            "max_param_spread": fit["max_param_spread"],
            "multi_modality_flag": bool(fit["multi_modality_flag"]),
            "abs_deviation": {
                "a": abs(a - headline["a"]),
                "b": abs(b - headline["b"]),
                "g": abs(g - headline["g"]),
            },
        })
        print(f"[sensitivity] seed {seed}: a={a:.4f}, b={b:.4f}, g={g:.4f}, "
              f"sum={s:.4f}, loglik={fit['loglik']:.1f}")
        print(f"[sensitivity] |Δa|={abs(a-headline['a']):.3e}, "
              f"|Δb|={abs(b-headline['b']):.3e}, "
              f"|Δg|={abs(g-headline['g']):.3e}")

    max_dev = max(max(r["abs_deviation"].values()) for r in results)
    print(f"\n[sensitivity] max |Δθ| across both alt subsets vs headline = {max_dev:.3e}")

    payload = {
        "headline_fit": {"a": headline["a"], "b": headline["b"],
                         "g": headline["g"], "loglik": headline["loglik"],
                         "subset_tickers_head": list(headline["subset_tickers"])[:5],
                         "subset_size": len(headline["subset_tickers"])},
        "always_alive_pool_size": len(always_alive),
        "always_alive_pool_sectors": dict(pool_counts),
        "coverage_threshold": COVERAGE_THRESHOLD,
        "subset_size": SUBSET_SIZE,
        "max_abs_deviation_vs_headline": max_dev,
        "subsets": results,
    }
    def _coerce(o):
        # json.dump can't serialise numpy scalar / pandas Series values
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"unserialisable {type(o).__name__}")

    with open(OUT, "w") as f:
        json.dump(payload, f, indent=2, default=_coerce)
    print(f"[sensitivity] Wrote {OUT}")


if __name__ == "__main__":
    main()
