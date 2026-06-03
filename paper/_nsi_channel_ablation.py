"""NSI channel ablation table (appendix Tab. nsi-ablation).

Drops each of the four NSI channels (s, h, rho, mu) one at a time,
renormalises the remaining three weights to sum to 1, and recomputes
the composite NSI on the min/max-normalised channels from the Stage 5
cache. For every ablation row reports crisis-mean / non-crisis-mean,
their difference Delta, and the one-sided exact-permutation p-value
over C(n, n_crisis) partitions of the active panel, plus the
per-snapshot NSI for Oct 2008 GFC and Mar 2020 COVID.

Panel composition is read from src.config.SNAPSHOTS so the same script
works on both the pre-CRSP 10-snapshot Yahoo cache and the post-CRSP
20-snapshot CRSP cache.
"""
import json
import math
import pickle
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import SNAPSHOTS  # noqa: E402
from src.stage5_nsi.stress_index import NSI_WEIGHTS_4CH  # noqa: E402

SNAP_DIR = ROOT / "results" / "snapshots"
OUT_DIR = ROOT / "results" / "nsi_channel_ablation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHANNEL_COLS = ("network_sparsity_norm", "hhi_top10_norm",
                "mean_corr_norm", "motif_shift_norm")
CHANNEL_LABELS = ("s", "h", "rho", "mu")
SNAPS_ORDER = [s[0] for s in SNAPSHOTS]
REGIMES = [s[3] for s in SNAPSHOTS]
CRISIS_IDX = [i for i, r in enumerate(REGIMES) if r == "crisis"]
N_PANEL = len(SNAPS_ORDER)
N_CRISIS = len(CRISIS_IDX)
TOTAL_PARTITIONS = math.comb(N_PANEL, N_CRISIS)
# Oct 2008 GFC and Mar 2020 COVID are reported as the modern-crisis anchors;
# they are the two crises common to both pre- and post-CRSP panels.
ANCHOR_LABELS = ("Oct 2008 GFC", "Mar 2020 COVID",
                 "Oct 2008 Peak", "Mar 2020 Peak")
OCT2008_IDX = next((i for i, lbl in enumerate(SNAPS_ORDER)
                    if lbl in ("Oct 2008 GFC", "Oct 2008 Peak")), None)
MAR2020_IDX = next((i for i, lbl in enumerate(SNAPS_ORDER)
                    if lbl in ("Mar 2020 COVID", "Mar 2020 Peak")), None)
assert OCT2008_IDX is not None and MAR2020_IDX is not None
assert OCT2008_IDX in CRISIS_IDX and MAR2020_IDX in CRISIS_IDX


def load_normalised_channels():
    """Pull min/max-normalised channels from the Stage 5 cache."""
    with open(SNAP_DIR / "stage5_results.pkl", "rb") as f:
        s5 = pickle.load(f)
    df = s5["snapshot_nsi"].set_index("snapshot").loc[SNAPS_ORDER].reset_index()
    X = df[list(CHANNEL_COLS)].values  # shape (N_PANEL, 4)
    assert X.shape == (N_PANEL, 4), f"unexpected channel shape {X.shape}"
    assert np.isfinite(X).all()
    return df, X


def exact_perm_p_one_sided(nsi, crisis_idx=CRISIS_IDX):
    """One-sided exact perm: count partitions whose crisis-mean minus
    non-crisis-mean meets or exceeds the observed value
    (floor 1/C(n, n_crisis))."""
    n = len(nsi)
    other = [i for i in range(n) if i not in crisis_idx]
    observed = nsi[crisis_idx].mean() - nsi[other].mean()
    n_crisis = len(crisis_idx)
    count = 0
    total = 0
    for combo in combinations(range(n), n_crisis):
        cm = nsi[list(combo)].mean()
        nm = nsi[[i for i in range(n) if i not in combo]].mean()
        if cm - nm >= observed - 1e-12:
            count += 1
        total += 1
    assert total == TOTAL_PARTITIONS, \
        f"partition count {total} != C({n}, {n_crisis}) = {TOTAL_PARTITIONS}"
    return count / total, observed


def renorm_weights_dropping(idx):
    """Set weight[idx] = 0 and rescale the remaining three to sum to 1."""
    w = np.array(NSI_WEIGHTS_4CH, dtype=float).copy()
    w[idx] = 0.0
    s = w.sum()
    assert s > 0
    return w / s


def main():
    df, X = load_normalised_channels()

    rows = []
    # Baseline
    w_base = np.array(NSI_WEIGHTS_4CH)
    nsi = X @ w_base
    p, delta = exact_perm_p_one_sided(nsi)
    rows.append({
        "config": f"Baseline NSI (s, h, rho, mu)",
        "weights": [round(float(x), 4) for x in w_base],
        "oct2008": round(float(nsi[OCT2008_IDX]), 4),
        "mar2020": round(float(nsi[MAR2020_IDX]), 4),
        "crisis_mean": round(float(nsi[CRISIS_IDX].mean()), 4),
        "noncrisis_mean": round(float(np.delete(nsi, CRISIS_IDX).mean()), 4),
        "delta": round(float(delta), 4),
        "p_one_sided": round(float(p), 4),
    })

    # Single-channel drops
    for i, name in enumerate(CHANNEL_LABELS):
        w = renorm_weights_dropping(i)
        nsi = X @ w
        p, delta = exact_perm_p_one_sided(nsi)
        rows.append({
            "config": f"Drop {name} ({['no sparsity','no HHI','no mean corr','no motif shift'][i]})",
            "weights": [round(float(x), 4) for x in w],
            "oct2008": round(float(nsi[OCT2008_IDX]), 4),
            "mar2020": round(float(nsi[MAR2020_IDX]), 4),
            "crisis_mean": round(float(nsi[CRISIS_IDX].mean()), 4),
            "noncrisis_mean": round(float(np.delete(nsi, CRISIS_IDX).mean()), 4),
            "delta": round(float(delta), 4),
            "p_one_sided": round(float(p), 4),
        })

    print(f"{'Configuration':<32} {'Weights':<34} {'Oct08':>7} {'Mar20':>7} "
          f"{'CrM':>7} {'NCM':>7} {'Delta':>8} {'p':>7}")
    print("-" * 120)
    for r in rows:
        print(f"{r['config']:<32} {str(r['weights']):<34} "
              f"{r['oct2008']:>7.4f} {r['mar2020']:>7.4f} "
              f"{r['crisis_mean']:>7.4f} {r['noncrisis_mean']:>7.4f} "
              f"{r['delta']:>+8.4f} {r['p_one_sided']:>7.4f}")

    out_json = OUT_DIR / "results.json"
    out_json.write_text(json.dumps({"rows": rows}, indent=2))
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
