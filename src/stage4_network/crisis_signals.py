"""Density- and sample-invariant crisis-signal metrics for §3.4 robustness.

Each metric is designed to be insensitive to GLASSO sparsification or
varying snapshot-window length, so crisis-vs-baseline contrasts cannot
be explained by edge-count or sample-size confounds.

    small_world_coefficients   Watts-Strogatz sigma (Humphries-Gurney).
    cross_sector_edge_fraction Top-K threshold-network cross-sector share.
    mutual_dyad_fraction       n_mutual / (n_directed + n_mutual).
    crisis_signal_summary      All three joined per snapshot.

The mutual-dyad fraction is the strongest crisis-vs-stress separator on
the ten-snapshot panel (Cohen's d = 7.16, exact-permutation p = 0.022).
"""
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from src.config import SNAPSHOTS_DIR


def _load_caches():
    """Load Stage-1..Stage-4 pickle caches."""
    with open(SNAPSHOTS_DIR / "stage1_results.pkl", "rb") as f:
        s1 = pickle.load(f)
    with open(SNAPSHOTS_DIR / "stage2_results.pkl", "rb") as f:
        s2 = pickle.load(f)
    with open(SNAPSHOTS_DIR / "stage3_results.pkl", "rb") as f:
        s3 = pickle.load(f)
    with open(SNAPSHOTS_DIR / "stage4_results.pkl", "rb") as f:
        s4 = pickle.load(f)
    return s1, s2, s3, s4


def small_world_coefficients(stage4=None):
    """Watts-Strogatz sigma = (C_emp/C_null) / (L_emp/L_null) per snapshot.

    Reads C and L from the Stage-4 ER-null draws; no recomputation.
    """
    if stage4 is None:
        _, _, _, stage4 = _load_caches()

    rows = []
    for label, d in stage4.items():
        er = d["erdos_renyi"]
        c_emp = er["empirical"]["clustering"]
        c_null = er["null_mean"]["clustering"]
        l_emp = er["empirical"]["path_length"]
        l_null = er["null_mean"]["path_length"]
        c_ratio = c_emp / c_null if c_null > 0 else np.nan
        l_ratio = l_emp / l_null if (l_null and l_emp) else np.nan
        sigma = c_ratio / l_ratio if l_ratio else np.nan
        rows.append({
            "snapshot": label, "regime": d["regime"],
            "C_emp": c_emp, "C_null": c_null, "C_ratio": c_ratio,
            "L_emp": l_emp, "L_null": l_null, "sigma": sigma,
            "Z_C": abs(er["z_scores"]["clustering"]),
        })
    return pd.DataFrame(rows)


DEFAULT_TARGET_DENSITY = 0.05  # 5%-of-p_min density-matched budget.
# top_k = round(0.05 * p_min*(p_min-1)/2); on the full-coverage headline cache
# (p_min=655) this is top_k=10709 (5.0% density at the sparsest panel, 2024
# Contemporary). Sized to the sparsest snapshot so the shared budget never
# exceeds any panel's edge supply.


def cross_sector_edge_fraction(sp500_info, stage1=None,
                               top_k=None, target_density=None):
    """Density-matched cross-GICS-sector edge share per snapshot.

    For each snapshot we form the top-K |R_avg| threshold network and
    report the share of kept edges whose endpoints lie in different
    GICS sectors. K is shared across snapshots either as ``top_k`` or
    via ``target_density * p_min (p_min-1)/2`` where p_min is the
    sparsest panel; the default ``target_density = DEFAULT_TARGET_DENSITY``
    matches the paper's Tab tab:cross-sector edge budget.
    """
    if stage1 is None:
        stage1, _, _, _ = _load_caches()

    sector_map = dict(zip(sp500_info["Symbol"], sp500_info["GICS Sector"]))
    snap_corrs = stage1["snapshot_correlations"]

    p_min = min(d["R_avg"].shape[0] for d in snap_corrs.values())
    if top_k is None:
        if target_density is None:
            target_density = DEFAULT_TARGET_DENSITY
        top_k = max(p_min, int(round(target_density * p_min * (p_min - 1) / 2)))
        density = top_k / (p_min * (p_min - 1) / 2)
        print(f"  [density-matched] top_k = {top_k} "
              f"(target_density={target_density:.3f}, p_min={p_min}, "
              f"effective density={density:.4f})")
    assert top_k > 0

    rows = []
    for label, data in snap_corrs.items():
        R = data["R_avg"]
        tickers = data["tickers"]
        p = R.shape[0]
        triu_i, triu_j = np.triu_indices(p, k=1)
        abs_corrs = np.abs(R[triu_i, triu_j])
        order = np.argsort(-abs_corrs)[:top_k]

        within = cross = 0
        for idx in order:
            si = sector_map.get(tickers[triu_i[idx]], "Unknown")
            sj = sector_map.get(tickers[triu_j[idx]], "Unknown")
            if si == sj:
                within += 1
            else:
                cross += 1
        total = within + cross
        rows.append({
            "snapshot": label, "regime": data["regime"], "top_k": top_k,
            "within_sec": within, "cross_sec": cross,
            "cs_fraction": cross / total if total > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def mutual_dyad_fraction(stage3=None):
    """f_mut = n_mutual / (n_directed + n_mutual) per snapshot."""
    if stage3 is None:
        _, _, stage3, _ = _load_caches()

    rows = []
    for label, d in stage3.items():
        n_dir = d["n_directed"]
        n_mut = d["n_bidirectional"]
        total = n_dir + n_mut
        rows.append({
            "snapshot": label, "regime": d["regime"],
            "directed": n_dir, "mutual": n_mut,
            "mutual_fraction": n_mut / total if total > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def crisis_signal_summary(sp500_info=None):
    """Three robustness metrics joined per snapshot."""
    if sp500_info is None:
        root = Path(__file__).resolve().parent.parent.parent
        sp500_info = pd.read_parquet(root / "data" / "sp500_info.parquet")

    s1, _, s3, s4 = _load_caches()
    sw = small_world_coefficients(s4).set_index("snapshot")
    cs = cross_sector_edge_fraction(sp500_info, s1).set_index("snapshot")
    md = mutual_dyad_fraction(s3).set_index("snapshot")

    return (sw[["regime", "sigma"]]
            .join(cs[["cs_fraction"]])
            .join(md[["mutual_fraction"]])
            .reset_index())


if __name__ == "__main__":
    summary = crisis_signal_summary()
    print("=" * 80)
    print("CRISIS-SIGNAL METRIC SUMMARY")
    print("=" * 80)
    print(summary.to_string(index=False,
                            formatters={"sigma": "{:7.2f}".format,
                                        "cs_fraction": "{:.3f}".format,
                                        "mutual_fraction": "{:.3f}".format}))
    print("\n--- Group means by regime ---")
    grp = summary.groupby("regime")[
        ["sigma", "cs_fraction", "mutual_fraction"]].mean()
    print(grp.to_string(float_format="{:.3f}".format))
