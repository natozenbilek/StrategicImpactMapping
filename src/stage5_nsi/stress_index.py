"""Stage 5: Network Stress Index (NSI) + VIX benchmark.

NSI_t = w_s tilde_s + w_h tilde_h + w_rho tilde_rho + w_mu tilde_mu,

with channels (sparsity 1-d, full-network PageRank HHI, mean A-DCC
correlation, FFL motif shift) and weights (0.25, 0.20, 0.35, 0.20)
summing to 1. Each channel is min/max-normalised across snapshots.

Two NSI variants:
* compute_nsi_components - snapshot-level (one row per regime window)
* compute_rolling_nsi    - simplified pipeline on rolling 252-day windows

Cache key `hhi_top10` carries the top-10 PageRank HHI (top-10 scores
re-normalised to sum to 1, then Herfindahl). `pagerank.hhi_full` holds
the full-network HHI for the paper Q2 table.
"""
import numpy as np
import pandas as pd
import pickle

from src.config import (
    DATA_DIR, SNAPSHOTS_DIR,
    ROLLING_WINDOW_DAYS, ROLLING_STEP_DAYS,
    ROLLING_GLASSO_ALPHA, ROLLING_MIN_ASSETS,
)
from src.stage5_nsi.vix_continuity import build_vix_continuity

NSI_WEIGHTS_4CH = (0.25, 0.20, 0.35, 0.20)
assert abs(sum(NSI_WEIGHTS_4CH) - 1.0) < 1e-12
# Precision-matrix entry below this magnitude is treated as a zero edge
# in the rolling NSI; matches Algorithm 1 (appendix.tex, app:stage5).
ROLLING_PRECISION_EPS = 1e-10
# Spectral-safety floor for sample Pearson matrices: shift so that
# lambda_min(C) >= ROLLING_PSD_FLOOR before GLASSO. Disclosed in
# Algorithm 1 (appendix.tex).
ROLLING_PSD_FLOOR = 1e-4


def download_vix(start=None, end=None):
    """VIX-VXO continuity Series (Whaley 2009 splice at 2003-09-22).

    Reads FRED VXOCLS + VIXCLS from data/crsp/ via build_vix_continuity;
    no network call. Optional start/end slice the returned Series.
    """
    cont = build_vix_continuity()
    if start is not None:
        cont = cont.loc[cont.index >= pd.Timestamp(start)]
    if end is not None:
        cont = cont.loc[cont.index <= pd.Timestamp(end)]
    assert cont.notna().any(), "VIX continuity empty after slicing"
    return cont


def compute_nsi_components(stage4_results, snapshot_correlations):
    """Snapshot-level NSI: four channels (s, h, rho, mu) -> composite.

    Requires Stage-1 mean-correlation and Stage-4 FFL motif outputs on
    every snapshot (the production pipeline always supplies both).
    """
    assert snapshot_correlations is not None, \
        "snapshot_correlations is required (Stage-1 mean correlation channel)"
    has_motifs = all(d.get("motifs") for d in stage4_results.values())
    assert has_motifs, \
        "Every snapshot must carry a Stage-4 motif block (FFL channel)"

    # Anchor motif channel to the mean baseline FFL Z so crisis values
    # are interpretable as multiples of calm-period FFL prevalence.
    baseline_motifs = [
        data["motifs"]["z_scores"].get("feed_forward_loop", 0.0)
        for data in stage4_results.values()
        if data["regime"] == "baseline"
    ]
    assert baseline_motifs, "Need at least one baseline snapshot for motif anchor"
    ffl_baseline = float(np.mean(baseline_motifs))
    assert ffl_baseline != 0.0, "Baseline mean FFL Z must be non-zero"

    records = []
    for label, data in stage4_results.items():
        n_nodes = data["n_nodes"]
        n_edges = data["n_edges"]  # directed (mutual pairs = 2 edges)
        max_edges = n_nodes * (n_nodes - 1) / 2  # undirected pair count
        assert max_edges > 0
        density = n_edges / max_edges  # m_dir / C(n,2); in [0, 2]

        sc = snapshot_correlations.get(label)
        assert sc is not None, f"Missing Stage-1 correlation for {label}"
        if "mean_corr" in sc:
            mean_corr = float(sc["mean_corr"])
        else:
            R = sc["R_avg"]
            mean_corr = float(R[np.triu_indices_from(R, k=1)].mean())

        ffl_z = data["motifs"]["z_scores"].get("feed_forward_loop", 0.0)
        # ffl_z can be NaN when the rewire null collapses (sparse / degenerate
        # bottom-N panels, see src/stage4_network/analysis.py). Fall back to
        # 0.0 so the motif channel contributes nothing rather than poisoning
        # the whole NSI vector.
        if not np.isfinite(ffl_z):
            ffl_z = 0.0
        motif_shift = ffl_z / abs(ffl_baseline) if ffl_baseline != 0 else 0.0
        if not np.isfinite(motif_shift):
            motif_shift = 0.0

        records.append({
            "snapshot": label,
            "regime": data["regime"],
            "modularity": data["community"]["modularity"],
            "network_sparsity": 1.0 - density,
            "hhi_top10": data["pagerank"]["hhi_top10"],  # top-10 normalised HHI
            "gini": data["pagerank"]["gini"],
            "density": density,
            "mean_corr": mean_corr,
            "motif_shift": motif_shift,
            "n_communities": data["community"]["n_communities"],
            "purity": data["community"]["purity"],
            "n_edges": n_edges,
        })

    df = pd.DataFrame(records)
    # Defensive: degenerate panels (no edges, no valid GLASSO fit) can
    # still produce NaN sparsity / HHI / mean_corr on a per-snapshot
    # basis. Replace with 0 so the panel-level NSI vector is well
    # defined; the affected cells will read 0 in the appendix tables,
    # which is the correct null answer for "no signal".
    channel_cols = ["network_sparsity", "hhi_top10", "mean_corr", "motif_shift"]
    df[channel_cols] = df[channel_cols].fillna(0.0)

    for col in ("network_sparsity", "hhi_top10", "mean_corr", "motif_shift"):
        cmin, cmax = float(df[col].min()), float(df[col].max())
        df[f"{col}_norm"] = ((df[col] - cmin) / (cmax - cmin)
                             if cmax > cmin else 0.0)
        if cmax > cmin:
            assert df[f"{col}_norm"].between(-1e-9, 1 + 1e-9).all(), col

    w_s, w_h, w_rho, w_mu = NSI_WEIGHTS_4CH
    df["nsi"] = (w_s * df["network_sparsity_norm"]
                 + w_h * df["hhi_top10_norm"]
                 + w_rho * df["mean_corr_norm"]
                 + w_mu * df["motif_shift_norm"])
    assert df["nsi"].between(-1e-9, 1 + 1e-9).all(), "NSI out of [0,1]"
    return df


def compute_rolling_nsi(returns, window_days=ROLLING_WINDOW_DAYS,
                        step_days=ROLLING_STEP_DAYS, n_assets=50):
    """Monthly rolling NSI on 252-day windows.

    Simplified pipeline: sample correlation -> fixed-alpha GLASSO ->
    Louvain Q + mean rho + density. Three-channel NSI =
    0.35 (1 - Q_norm) + 0.35 mean_corr_norm + 0.30 density_norm.
    """
    from sklearn.covariance import graphical_lasso
    from sklearn.exceptions import ConvergenceWarning
    import warnings
    import networkx as nx
    import community as community_louvain

    dates = returns.index
    T = len(dates)

    # Static asset selection avoids universe drift across windows.
    if returns.shape[1] > n_assets:
        top_tickers = returns.count().nlargest(n_assets).index
        returns_sub = returns[top_tickers]
    else:
        returns_sub = returns

    records = []
    n_skipped_assets = 0
    n_skipped_edges = 0
    n_glasso_fail = 0
    n_windows = (T - window_days) // step_days + 1
    print(f"[Stage 5] Computing rolling NSI ({n_windows} windows, "
          f"{n_assets} assets, step={step_days} days)...")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for i in range(0, T - window_days, step_days):
            window = returns_sub.iloc[i:i + window_days].dropna(axis=1)
            if window.shape[1] < ROLLING_MIN_ASSETS:
                n_skipped_assets += 1
                continue
            end_date = dates[i + window_days - 1]
            corr = window.corr().values
            p = corr.shape[0]

            # Spectral safety net: shift to lambda_min >= ROLLING_PSD_FLOOR
            # before GLASSO. Mirrors the snapshot Stage-2 fix-up.
            eigvals = np.linalg.eigvalsh(corr)
            shift = max(0.0, ROLLING_PSD_FLOOR - float(eigvals.min()))
            if shift > 0:
                corr = corr + shift * np.eye(p)

            try:
                _, precision = graphical_lasso(
                    corr, alpha=ROLLING_GLASSO_ALPHA, max_iter=200)
            except (FloatingPointError, np.linalg.LinAlgError, ValueError) as exc:
                n_glasso_fail += 1
                print(f"  [skip] GLASSO failed at {end_date.date()}: {type(exc).__name__}")
                continue

            mask = np.abs(precision) > ROLLING_PRECISION_EPS
            np.fill_diagonal(mask, False)
            d_inv = np.sqrt(np.maximum(np.diag(precision), 1e-12))
            pcorr_abs = np.abs(precision / np.outer(d_inv, d_inv))
            adj = np.where(mask, pcorr_abs, 0.0)
            np.fill_diagonal(adj, 0.0)

            G = nx.from_numpy_array(adj)
            if G.number_of_edges() < 5:
                n_skipped_edges += 1
                continue
            partition = community_louvain.best_partition(G, random_state=42)
            Q = community_louvain.modularity(partition, G)
            mean_corr = float(corr[np.triu_indices_from(corr, k=1)].mean())
            max_edges = p * (p - 1) / 2
            density = G.number_of_edges() / max_edges

            records.append({
                "date": end_date, "modularity": Q,
                "mean_correlation": mean_corr, "density": density,
                "n_edges": G.number_of_edges(),
            })

            if len(records) % 20 == 0:
                print(f"  Window {len(records)}/{n_windows}: "
                      f"date={end_date.strftime('%Y-%m-%d')}, "
                      f"Q={Q:.3f}, mean_corr={mean_corr:.3f}")

    print(f"  Skipped: {n_skipped_assets} asset-count, {n_skipped_edges} "
          f"edge-count, {n_glasso_fail} GLASSO-fail")

    df = pd.DataFrame(records)
    if len(df) == 0:
        print("  WARNING: No valid windows computed")
        return df
    df.set_index("date", inplace=True)

    for col in ("modularity", "mean_correlation", "density"):
        cmin, cmax = float(df[col].min()), float(df[col].max())
        df[f"{col}_norm"] = ((df[col] - cmin) / (cmax - cmin)
                             if cmax > cmin else 0.0)

    # Modularity enters as (1 - Q_norm) so community breakdown
    # contributes positively to NSI.
    df["nsi"] = (0.35 * (1.0 - df["modularity_norm"])
                 + 0.35 * df["mean_correlation_norm"]
                 + 0.30 * df["density_norm"])
    assert df["nsi"].between(-1e-9, 1 + 1e-9).all()

    print(f"\n  Rolling NSI computed: {len(df)} data points")
    print(f"  Date range: {df.index[0].strftime('%Y-%m-%d')} to "
          f"{df.index[-1].strftime('%Y-%m-%d')}")
    return df


def backtest_rolling_nsi_vs_vix(nsi_series, vix, threshold_percentile=90):
    """Rolling-NSI vs VIX exceedance concordance + contemporaneous /
    lagged Pearson.

    Lag keys 'lag5' ... 'lag63' are NSI grid steps (each ~21 trading
    days = one trading month).
    """
    common_dates = nsi_series.index.intersection(vix.index)
    assert len(common_dates) > 30, \
        f"NSI / VIX share only {len(common_dates)} dates"
    nsi_a = nsi_series.loc[common_dates]
    vix_a = vix.loc[common_dates]

    nsi_th = float(np.percentile(nsi_a.dropna(), threshold_percentile))
    vix_th = float(np.percentile(vix_a.dropna(), threshold_percentile))
    nsi_spikes = nsi_a[nsi_a > nsi_th].index
    vix_spikes = vix_a[vix_a > vix_th].index

    lead_times = []
    for vd in vix_spikes:
        preceding = nsi_spikes[nsi_spikes < vd]
        if len(preceding) > 0:
            lead_days = (vd - preceding[-1]).days
            if lead_days <= 90:
                lead_times.append(lead_days)

    lagged = {}
    for lag in (5, 10, 15, 21, 42, 63):
        shifted = nsi_a.shift(lag)
        common = shifted.dropna().index.intersection(vix_a.dropna().index)
        if len(common) > 20:
            lagged[f"lag{lag}"] = float(shifted.loc[common].corr(vix_a.loc[common]))

    return {
        "contemporaneous_corr": float(nsi_a.corr(vix_a)),
        "lagged_correlations": lagged,
        "mean_lead_days": float(np.mean(lead_times)) if lead_times else np.nan,
        "median_lead_days": float(np.median(lead_times)) if lead_times else np.nan,
        "n_signals": len(lead_times),
        "n_vix_spikes": len(vix_spikes),
        "signal_rate": (len(lead_times) / len(vix_spikes)
                        if len(vix_spikes) > 0 else 0.0),
    }


def run_stage5(stage4_results=None, returns=None,
               snapshot_correlations=None, force=False):
    """Driver: snapshot NSI (Part A) + rolling NSI + VIX benchmark (Part B)."""
    cache_path = SNAPSHOTS_DIR / "stage5_results.pkl"
    if not force and cache_path.exists():
        print("[Stage 5] Loading cached results...")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    if stage4_results:
        print("[Stage 5a] Computing snapshot NSI components...")
        nsi_components = compute_nsi_components(
            stage4_results, snapshot_correlations=snapshot_correlations)
        results["snapshot_nsi"] = nsi_components
        print(nsi_components[["snapshot", "regime", "nsi",
                              "modularity", "mean_corr"]].to_string())

    if returns is not None:
        rolling_nsi = compute_rolling_nsi(returns, n_assets=50)
        results["rolling_nsi"] = rolling_nsi
        if len(rolling_nsi) > 0:
            print("\n[Stage 5b] Backtesting rolling NSI vs VIX...")
            vix = download_vix()
            if len(vix) > 0:
                backtest = backtest_rolling_nsi_vs_vix(rolling_nsi["nsi"], vix)
                results["backtest"] = backtest
                print(f"  Contemporaneous corr(NSI, VIX): "
                      f"{backtest['contemporaneous_corr']:.3f}")
                print(f"  Lagged correlations: {backtest['lagged_correlations']}")
                print(f"  Mean lead time: {backtest['mean_lead_days']:.1f} days")
                print(f"  Signal rate: {backtest['signal_rate']:.1%}")

    with open(cache_path, "wb") as f:
        pickle.dump(results, f)
    return results


if __name__ == "__main__":
    print("[Stage 5] Run via main pipeline (run_pipeline.py)")
