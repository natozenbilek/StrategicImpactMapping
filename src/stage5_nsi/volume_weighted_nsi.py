"""Volume-weighted NSI variant (F1 extension).

The default snapshot NSI weights every ticker equally inside each
channel. The volume-weighted variant downweights low-ADV tickers,
isolating the systemic-risk signal from the speculative-tail
amplification documented in Appendix app:vw-nsi.

For each snapshot the per-ticker weight is

    w_i = log(1 + ADV_i) / sum_j log(1 + ADV_j),

i.e. log-volume-weighted shares summing to 1 per snapshot. The four
NSI channels are then recomputed with these weights:

    rho_vw   = sum_{i<j} (w_i w_j) * R_ij / sum_{i<j} (w_i w_j)
    hhi_vw   = sum_i w_i * r_i^2
    s_vw     = 1 - density_vw  (density_vw = weighted edge fraction)
    mu_vw    = same FFL motif shift (graph-level, unchanged)

Output schema mirrors the standard NSI; rows carry ``_vw`` suffixes
to avoid clobbering the unweighted result when persisted.
"""
import pickle

import numpy as np
import pandas as pd

from src.config import DATA_DIR, SNAPSHOTS_DIR
from src.stage5_nsi.stress_index import NSI_WEIGHTS_4CH


def _log_volume_weights(tickers, adv_series):
    """Per-ticker log-volume weights summing to 1."""
    raw = np.array([np.log1p(float(adv_series.get(t, 0.0))) for t in tickers])
    total = float(raw.sum())
    if total > 0:
        w = raw / total
    else:
        w = np.full_like(raw, 1.0 / len(raw))
    assert np.isclose(w.sum(), 1.0)
    assert (w >= -1e-12).all()
    return w


def compute_volume_weighted_nsi(stage4_results, snapshot_correlations,
                                adv_path=DATA_DIR / "sp500_adv.parquet"):
    """Volume-weighted NSI components per snapshot.

    Reuses the four channels (s, h, rho, mu) and the (0.25, 0.20, 0.35,
    0.20) weights of compute_nsi_components but reweights rho and HHI
    channels by log-volume. The motif channel is unaffected (motifs are
    graph-level, not per-ticker).
    """
    if not adv_path.exists():
        raise FileNotFoundError(
            f"ADV cache missing at {adv_path}. Run "
            "`python -m tools.download_volume` first.")
    adv_series = pd.read_parquet(adv_path).iloc[:, 0]

    baseline_motifs = [
        d["motifs"]["z_scores"].get("feed_forward_loop", 0.0)
        for d in stage4_results.values()
        if d["regime"] == "baseline" and d.get("motifs")
    ]
    assert baseline_motifs, "Need at least one baseline snapshot with motifs"
    ffl_baseline = float(np.mean(baseline_motifs))

    records = []
    for label, data in stage4_results.items():
        tickers = data["tickers"]
        p = len(tickers)
        w = _log_volume_weights(tickers, adv_series)

        rho_vw = 0.0
        if label in snapshot_correlations:
            R = snapshot_correlations[label].get("R_avg")
            if R is not None and R.shape[0] == p:
                W = np.outer(w, w)
                np.fill_diagonal(W, 0.0)
                denom = float(W.sum())
                rho_vw = float((W * R).sum() / denom) if denom > 0 else 0.0

        # Volume-weighted PageRank Herfindahl: sum_i w_i s_i^2. PageRank
        # scores are keyed by integer node index 0..p-1 (Stage 4 convention).
        pr_scores = data["pagerank"].get("pagerank_scores", {})
        if pr_scores and len(pr_scores) == p:
            scores = np.array([pr_scores.get(i, 0.0) for i in range(p)])
            assert np.isclose(scores.sum(), 1.0, atol=1e-4), \
                f"PageRank scores do not sum to 1 in {label}"
            hhi_vw = float(np.sum(w * scores ** 2))
        else:
            hhi_vw = data["pagerank"]["hhi_top10"]

        # Weighted edge density = sum_{(i,j) in E} w_i w_j / sum_{i<j} w_i w_j.
        # Closed form for denominator: 0.5 * (sum(w)^2 - sum(w^2)).
        max_w_pairs = 0.5 * (float(w.sum()) ** 2 - float((w ** 2).sum()))
        G = data.get("graph")
        if G is not None and max_w_pairs > 0:
            edge_w = 0.0
            for u, v in G.edges():
                u_int, v_int = int(u), int(v)
                assert 0 <= u_int < p and 0 <= v_int < p, \
                    f"Edge ({u},{v}) out of node-index range in {label}"
                edge_w += w[u_int] * w[v_int]
            density_vw = edge_w / max_w_pairs
        else:
            density_vw = (data["n_edges"] / (p * (p - 1) / 2)
                          if p > 1 else 0.0)
        s_vw = 1.0 - density_vw

        ffl_z = data["motifs"]["z_scores"].get("feed_forward_loop", 0.0) \
            if data.get("motifs") else 0.0
        motif_shift = ffl_z / abs(ffl_baseline) if ffl_baseline != 0 else 0.0

        records.append({
            "snapshot": label,
            "regime": data["regime"],
            "network_sparsity_vw": s_vw,
            "hhi_vw": hhi_vw,
            "mean_corr_vw": rho_vw,
            "motif_shift_vw": motif_shift,
        })

    df = pd.DataFrame(records)
    assert len(df) == len(stage4_results)

    for col in ("network_sparsity_vw", "hhi_vw", "mean_corr_vw", "motif_shift_vw"):
        cmin, cmax = float(df[col].min()), float(df[col].max())
        df[f"{col}_norm"] = ((df[col] - cmin) / (cmax - cmin)
                             if cmax > cmin else 0.0)

    w_s, w_h, w_rho, w_mu = NSI_WEIGHTS_4CH
    df["nsi_vw"] = (
        w_s * df["network_sparsity_vw_norm"]
        + w_h * df["hhi_vw_norm"]
        + w_rho * df["mean_corr_vw_norm"]
        + w_mu * df["motif_shift_vw_norm"]
    )
    assert df["nsi_vw"].between(-1e-9, 1 + 1e-9).all()
    return df


def run_volume_weighted_nsi(force=False):
    """Driver: load Stage 1 + Stage 4 caches, compute VW-NSI, persist."""
    cache_path = SNAPSHOTS_DIR / "nsi_volume_weighted.pkl"
    if not force and cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    with open(SNAPSHOTS_DIR / "stage1_results.pkl", "rb") as f:
        s1 = pickle.load(f)
    with open(SNAPSHOTS_DIR / "stage4_results.pkl", "rb") as f:
        s4 = pickle.load(f)

    df = compute_volume_weighted_nsi(s4, s1["snapshot_correlations"])
    with open(cache_path, "wb") as f:
        pickle.dump(df, f)
    print(f"  Cached to {cache_path}")
    return df


if __name__ == "__main__":
    df = run_volume_weighted_nsi(force=True)
    print(df[["snapshot", "regime", "mean_corr_vw", "hhi_vw", "nsi_vw"]]
          .to_string(index=False))
