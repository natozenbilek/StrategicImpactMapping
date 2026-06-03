"""Generate the addon figures and tables introduced in the 2026-05-27
section/figure audit.

Reads cached pickles + JSON sidecars under ``results/snapshots/`` and
emits:

  figures/fig_app_nsi_vix_scatter.pdf      -- snapshot NSI vs window-mean
                                              VIX scatter with OLS line
                                              and rank annotation.
  figures/fig_app_rolling_nsi.pdf          -- rolling NSI + daily VIX
                                              dual-axis time series, with
                                              the lagged cross-correlogram
                                              as the bottom panel.
  figures/fig_app_nsi_components.pdf       -- four-panel small multiples
                                              of the raw NSI channels per
                                              snapshot.
  figures/fig_app_mle_convergence.pdf      -- multi-start MLE (a,b,g)
                                              dispersion across the
                                              100-asset sub-panel seeds.
  figures/fig_app_sector_heatmap.pdf       -- 11x11 GICS sector cross-edge
                                              density heatmap for the two
                                              crises and the 2025 baseline.

The power-MDE and per-snapshot |Z|/sweep/ADF tables are now produced by
paper/_generate_extra.py (Cohen's-d annotated lattice + verbose regime
columns); the legacy power_quantization figure and the bonferroni_z /
stage3_sweep_grid / adf_rejection table emitters were retired in the
2026-05-27 cleanup pass to keep one canonical generator per artefact.

Run from the repository root::

    .venv/bin/python paper/_generate_addon_assets.py

Every cell of every emitted table is read off the live pickle/JSON
caches; the only constants are the snapshot ordering, the regime palette
(reused from generate_figures.py for visual consistency), and the LaTeX
column headers.
"""

import json
import math
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
SNAP = ROOT / "results" / "snapshots"
FIG = Path(__file__).resolve().parent / "figures"
TAB = Path(__file__).resolve().parent / "tables"
FIG.mkdir(exist_ok=True)
TAB.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})

# Imperial palette mirrored from generate_figures.py.
REGIME_COLOR = {
    "crisis":   "#B2182B",
    "stress":   "#E08214",
    "recovery": "#1A9850",
    "baseline": "#2166AC",
}
IMPERIAL_IVORY = "#F5E9D7"
IMPERIAL_GOLD  = "#E08214"
IMPERIAL_GRAPHITE = "#2A2E33"

SECTOR_COLORS = {
    "Information Technology":  "#1F4E79",
    "Financials":              "#B8860B",
    "Health Care":             "#2E7D32",
    "Consumer Discretionary":  "#C0392B",
    "Industrials":             "#5E35B2",
    "Communication Services":  "#00838F",
    "Consumer Staples":        "#AD1457",
    "Energy":                  "#6D4C41",
    "Utilities":               "#F39C12",
    "Materials":               "#546E7A",
    "Real Estate":             "#EF6C00",
}

SECTOR_ORDER = [
    "Information Technology", "Financials", "Health Care",
    "Consumer Discretionary", "Industrials", "Communication Services",
    "Consumer Staples", "Energy", "Utilities", "Materials", "Real Estate",
]

SHORT_LABEL = {
    "Oct 1987 Black Monday": "Oct 1987",
    "1990-91 Recession":     "1990-91",
    "1993 Calm":             "1993",
    "Dec 1994 Tequila":      "Dec 1994",
    "Oct 1997 Asian Crisis": "Oct 1997",
    "Oct 1998 LTCM":         "Oct 1998",
    "Apr 2000 Dot-com":      "Apr 2000",
    "Sep 2001 9/11":         "Sep 2001",
    "Jul 2002 WorldCom":     "Jul 2002",
    "2005 Calm":             "2005",
    "2007 Subprime Buildup": "2007",
    "Oct 2008 GFC":          "Oct 2008",
    "Mar 2009 Recovery":     "Mar 2009",
    "2013 Calm":             "2013",
    "2017 Calm":             "2017",
    "Q4 2018 VolShock":      "Q4 2018",
    "Mar 2020 COVID":        "Mar 2020",
    "Jun 2020 Stable":       "Jun 2020",
    "2022 Rate Hikes":       "2022",
    "2024 Contemporary":     "2024",
}

SNAP_ORDER = list(SHORT_LABEL.keys())

import sys as _sys
_sys.path.insert(0, str(ROOT))
from src.config import SNAPSHOTS as _SNAPSHOTS
# Analytical snapshot windows (label -> start, end) straight from the
# pipeline config, so the window-mean VXO/VIX read here coincides with
# the windows NSI is computed on -- the 20-window CRSP panel.
WINDOWS = {lab: (start, end) for (lab, start, end, _reg) in _SNAPSHOTS}


# ----------------------------------------------------------------------
# Load caches
# ----------------------------------------------------------------------
print("Loading caches...")
with open(SNAP / "stage1_results.pkl", "rb") as f:
    stage1 = pickle.load(f)
with open(SNAP / "stage3_results.pkl", "rb") as f:
    stage3 = pickle.load(f)
with open(SNAP / "stage4_results.pkl", "rb") as f:
    stage4 = pickle.load(f)
with open(SNAP / "stage5_results.pkl", "rb") as f:
    stage5 = pickle.load(f)
with open(SNAP / "stage3_sensitivity.pkl", "rb") as f:
    stage3_sens = pickle.load(f)
with open(SNAP / "stage1_adf_diagnostics.json") as f:
    adf = json.load(f)
# Spliced VXO+VIX continuity (1986+) so the pre-2004 snapshots have a
# benchmark to read; the stored level lives in a "Close" column.
_vix_cont = pd.read_parquet(ROOT / "data" / "vix_continuity.parquet")
vix_s = _vix_cont["Close"] if "Close" in _vix_cont.columns else _vix_cont.iloc[:, 0]

vix_s.index = pd.to_datetime(vix_s.index)

sp500 = pd.read_parquet(ROOT / "data" / "sp500_info.parquet")
sector_map = dict(zip(sp500["Symbol"], sp500["GICS Sector"]))

nsi_df = stage5["snapshot_nsi"].copy()
rolling_df = stage5["rolling_nsi"].copy()

assert set(SNAP_ORDER).issubset(set(nsi_df["snapshot"])), \
    f"snapshot ordering mismatch: missing {set(SNAP_ORDER)-set(nsi_df['snapshot'])}"


def window_mean_vix(label: str) -> float:
    a, b = WINDOWS[label]
    s = vix_s.loc[a:b]
    assert len(s) > 0, f"VIX window {label} empty"
    return float(s.mean())


nsi_df["vix_mean"] = nsi_df["snapshot"].map(window_mean_vix)
nsi_df["color"] = nsi_df["regime"].map(REGIME_COLOR)
nsi_df["short"] = nsi_df["snapshot"].map(SHORT_LABEL)


# ----------------------------------------------------------------------
# F-app-1: NSI-VIX scatter with OLS line + cluster-bootstrap-style CI
# ----------------------------------------------------------------------
def cluster_bootstrap_pearson(x, y, clusters, B=5000, seed=2026):
    """Cluster bootstrap Pearson r, resampling whole clusters with
    replacement. Matches the snapshot-clustering convention in
    paper/_inference.py so the new figure's CI lands in the same band
    as the macro \\InfClusterPearsonCILow / CIHigh used in the text."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x); y = np.asarray(y); clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    rs = np.empty(B)
    for b in range(B):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.flatnonzero(clusters == c) for c in pick])
        if idx.size < 3:
            rs[b] = np.nan
            continue
        rs[b] = np.corrcoef(x[idx], y[idx])[0, 1]
    rs = rs[~np.isnan(rs)]
    return float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


# Cluster labels mirror _inference.py CLUSTERS on the CRSP panel: the GFC
# pair (Oct 2008 GFC + Mar 2009 Recovery) and the COVID pair (Mar 2020
# COVID + Jun 2020 Stable) are resampled atomically; the other sixteen
# snapshots are singletons (each its own cluster).
_CL_PAIRS = {"Oct 2008 GFC": "GFC", "Mar 2009 Recovery": "GFC",
             "Mar 2020 COVID": "COVID", "Jun 2020 Stable": "COVID"}
CLUSTER = {s: _CL_PAIRS.get(s, s) for s in SNAP_ORDER}

x = nsi_df["nsi"].to_numpy()
y = nsi_df["vix_mean"].to_numpy()
clusters = nsi_df["snapshot"].map(CLUSTER).to_numpy()
r_pearson = float(np.corrcoef(x, y)[0, 1])
ci_lo, ci_hi = cluster_bootstrap_pearson(x, y, clusters)
slope, intercept = np.polyfit(x, y, 1)
xx = np.linspace(x.min() - 0.02, x.max() + 0.02, 100)
yy = slope * xx + intercept

# Rank concordance: Spearman on the same 10 pairs (rank panel).
rho_s, _ = stats.spearmanr(x, y)

fig, ax = plt.subplots(figsize=(5.2, 4.4))
for _, r in nsi_df.iterrows():
    ax.scatter(r["nsi"], r["vix_mean"], s=110, color=r["color"],
               edgecolors=IMPERIAL_GRAPHITE, linewidths=0.8, zorder=4)
    # offset label so it does not overlap the marker
    ax.annotate(r["short"], (r["nsi"], r["vix_mean"]),
                xytext=(7, 4), textcoords="offset points",
                fontsize=7.5, color=IMPERIAL_GRAPHITE)
ax.plot(xx, yy, color=IMPERIAL_GRAPHITE, linewidth=1.2, zorder=3,
        label=f"OLS fit (slope={slope:.2f})")
ax.set_xlabel("Snapshot NSI")
ax.set_ylabel("Window-mean VIX")
ax.set_title("Snapshot NSI vs. window-mean VIX")
ax.grid(alpha=0.25, linestyle=":")
ax.text(0.03, 0.97,
        f"Pearson $r={r_pearson:.3f}$\n"
        f"Cluster boot. 95% CI [{ci_lo:.3f}, {ci_hi:.3f}]\n"
        f"Spearman $\\rho={rho_s:.3f}$",
        transform=ax.transAxes, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.35", fc=IMPERIAL_IVORY,
                  ec=IMPERIAL_GOLD, lw=0.6))

import matplotlib.lines as ml
ax.legend(handles=[ml.Line2D([0], [0], color=IMPERIAL_GRAPHITE,
                              lw=1.2, label="OLS fit")] +
                 [ml.Line2D([0], [0], marker="o", linestyle="none",
                            markerfacecolor=c,
                            markeredgecolor=IMPERIAL_GRAPHITE,
                            markersize=9, label=r) for r, c in REGIME_COLOR.items()],
          loc="lower right", frameon=True, framealpha=0.95,
          facecolor=IMPERIAL_IVORY, edgecolor=IMPERIAL_GOLD)

fig.tight_layout()
fig.savefig(FIG / "fig_app_nsi_vix_scatter.pdf", bbox_inches="tight")
fig.savefig(FIG / "fig_app_nsi_vix_scatter.png", bbox_inches="tight")
plt.close(fig)
print("  fig_app_nsi_vix_scatter ok")


# ----------------------------------------------------------------------
# F-app-2: Rolling NSI + daily VIX overlay + lagged cross-correlogram
# ----------------------------------------------------------------------
roll_nsi = rolling_df["nsi"].dropna().copy()
roll_nsi.index = pd.to_datetime(roll_nsi.index)
# Align VIX to the rolling grid exactly as paper/_inference.py does for
# the block-bootstrap observed r (\InfRollingR): reindex the VXO+VIX
# continuity to each NSI grid date by nearest trading day, then keep the
# common non-missing index. No distance tolerance is applied, so the
# contemporaneous r reported here equals the headline \InfRollingR used
# throughout the paper and the appendix block-bootstrap.
vix_aligned = vix_s.reindex(roll_nsi.index, method="nearest")
_common = roll_nsi.index.intersection(vix_aligned.dropna().index)
roll_nsi = roll_nsi.loc[_common]
vix_aligned = vix_aligned.loc[_common]

# contemporaneous Pearson (identical computation to paper/_inference.py)
r_roll = float(np.corrcoef(roll_nsi.values, vix_aligned.values)[0, 1])

# lagged cross-correlation in grid steps (positive lag = NSI leads VIX)
max_lag = 24
lags = np.arange(-max_lag, max_lag + 1)
ccf = []
for L in lags:
    if L >= 0:
        a = roll_nsi.iloc[:len(roll_nsi) - L]
        b = vix_aligned.iloc[L:]
    else:
        a = roll_nsi.iloc[-L:]
        b = vix_aligned.iloc[:len(vix_aligned) + L]
    valid = (~a.isna()) & (~b.values.astype(float).__ne__(b.values.astype(float)))
    # cleaner: drop NaNs from b
    bb = pd.Series(b.values, index=a.index)
    mask = (~a.isna()) & (~bb.isna())
    if mask.sum() < 5:
        ccf.append(np.nan)
    else:
        ccf.append(float(np.corrcoef(a[mask], bb[mask])[0, 1]))
ccf = np.array(ccf)

fig, axes = plt.subplots(2, 1, figsize=(8.0, 5.6),
                         gridspec_kw={"hspace": 0.45, "height_ratios": [1.3, 1.0]})

ax = axes[0]
ax.plot(roll_nsi.index, roll_nsi.values, color=REGIME_COLOR["baseline"],
        linewidth=1.2,
        label="Rolling NSI (252-day window, 21-day step)")
ax.set_ylabel("Rolling NSI", color=REGIME_COLOR["baseline"])
ax.tick_params(axis="y", colors=REGIME_COLOR["baseline"])
ax.grid(alpha=0.25, linestyle=":")
# Shade snapshot windows on this background so reader can tie the two
# NSIs together
for snap, (a, b) in WINDOWS.items():
    a, b = pd.Timestamp(a), pd.Timestamp(b)
    if a > roll_nsi.index.max() or b < roll_nsi.index.min():
        continue
    reg = nsi_df.set_index("snapshot").loc[snap, "regime"]
    ax.axvspan(a, b, color=REGIME_COLOR[reg], alpha=0.13, zorder=1)
ax2 = ax.twinx()
ax2.plot(vix_aligned.index, vix_aligned.values, color=REGIME_COLOR["crisis"],
         linewidth=0.75, alpha=0.75, label="Daily VIX (rolling-grid sampled)")
ax2.set_ylabel("VIX", color=REGIME_COLOR["crisis"])
ax2.tick_params(axis="y", colors=REGIME_COLOR["crisis"])
ax.set_title(rf"Rolling NSI vs. VXO/VIX (contemporaneous Pearson $r={r_roll:.3f}$)")

ax = axes[1]
ax.bar(lags, ccf,
       color=[REGIME_COLOR["crisis"] if c < 0 else REGIME_COLOR["recovery"]
              for c in ccf],
       edgecolor=IMPERIAL_GRAPHITE, linewidth=0.3)
ax.axhline(0, color=IMPERIAL_GRAPHITE, linewidth=0.6)
ax.axvline(0, color=IMPERIAL_GRAPHITE, linewidth=0.6, linestyle=":")
# 95% CI bar under no autocorrelation (very loose; for visual reference)
n = int((~roll_nsi.isna()).sum())
ci = 1.96 / math.sqrt(max(n - 1, 1))
for s in (-ci, ci):
    ax.axhline(s, color="gray", linewidth=0.5, linestyle="--")
ax.set_xlabel("Lag (rolling-NSI grid steps; positive = NSI leads VIX)")
ax.set_ylabel("Cross-correlation")
ax.set_title(rf"Lagged cross-correlogram (dashed $\pm 1.96/\sqrt{{n}}$ reference)")

fig.tight_layout()
fig.savefig(FIG / "fig_app_rolling_nsi.pdf", bbox_inches="tight")
fig.savefig(FIG / "fig_app_rolling_nsi.png", bbox_inches="tight")
plt.close(fig)
print(f"  fig_app_rolling_nsi ok (r_roll={r_roll:.3f}, max|ccf|={np.nanmax(np.abs(ccf)):.3f})")


# ----------------------------------------------------------------------
# F-app-3: NSI components small multiples
# ----------------------------------------------------------------------
nsi_df["sparsity"] = 1.0 - nsi_df["density"]
nsi_ord = nsi_df.set_index("snapshot").reindex(SNAP_ORDER).reset_index()
xs = np.arange(len(SNAP_ORDER))
colors = nsi_ord["regime"].map(REGIME_COLOR).to_numpy()
short = [SHORT_LABEL[s] for s in SNAP_ORDER]

fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.4),
                         gridspec_kw={"hspace": 0.45, "wspace": 0.28})
panels = [
    ("(a) Sparsity $1-d$",          "sparsity"),
    ("(b) Top-10 HHI",              "hhi_top10"),
    ("(c) Mean A-DCC correlation $\\bar\\rho$", "mean_corr"),
    ("(d) FFL motif shift $\\mu$",  "motif_shift"),
]
for ax, (title, col) in zip(axes.ravel(), panels):
    ax.bar(xs, nsi_ord[col].to_numpy(), color=colors, edgecolor="#222",
           linewidth=0.3)
    ax.set_xticks(xs)
    ax.set_xticklabels(short, rotation=35, ha="right", fontsize=7.5)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25, linestyle=":")
import matplotlib.lines as ml
fig.legend(handles=[ml.Line2D([0], [0], marker="s", linestyle="none",
                              markerfacecolor=c, markeredgecolor="white",
                              markersize=9, label=r)
                    for r, c in REGIME_COLOR.items()],
           loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04),
           frameon=False)
fig.suptitle("NSI raw channels per snapshot (chronological)", y=0.995)
fig.tight_layout()
fig.savefig(FIG / "fig_app_nsi_components.pdf", bbox_inches="tight")
fig.savefig(FIG / "fig_app_nsi_components.png", bbox_inches="tight")
plt.close(fig)
print("  fig_app_nsi_components ok")


# ----------------------------------------------------------------------
# F-app-4: Multi-start MLE (a,b,g) convergence dispersion
# ----------------------------------------------------------------------
# Headline A-DCC scalars and the cross-seed dispersion are read directly
# from the Stage-1 cache (adcc_params), so this figure tracks the same
# panel the pipeline estimated -- no separate sensitivity side-car.
_ap = stage1["adcc_params"]
headline = {"a": float(_ap["a"]), "b": float(_ap["b"]), "g": float(_ap["g"])}
_spread = float(_ap.get("max_param_spread", 0.0))
_nseed = int(_ap.get("n_surviving_seeds", 0))
fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.4),
                         gridspec_kw={"wspace": 0.4})
labels = ("a", "b", "g")
for ax, k in zip(axes, labels):
    val = headline[k]
    # Show the converged estimate as a bar with the cross-seed spread band
    # (the same +/- max_param_spread/2 envelope across the surviving seeds).
    ax.bar([0], [val], width=0.5, color="#2166AC", edgecolor="#222",
           linewidth=0.4)
    ax.errorbar([0], [val], yerr=_spread / 2.0, color="#B2182B",
                capsize=4, linewidth=1.0, label="cross-seed spread")
    pad = max(_spread * 1.2, val * 0.02, 1e-4)
    ax.set_ylim(max(0.0, val - pad), val + pad)
    ax.set_xticks([0])
    ax.set_xticklabels([f"{_nseed} seeds"], fontsize=7.5)
    ax.set_xlim(-0.7, 0.7)
    ax.set_title(f"$\\hat{{{k}}}$")
    ax.set_ylabel("Estimate")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3),
                        useMathText=True)
# No suptitle/footer: the LaTeX caption carries the descriptive text.
fig.tight_layout()
fig.savefig(FIG / "fig_app_mle_convergence.pdf", bbox_inches="tight")
fig.savefig(FIG / "fig_app_mle_convergence.png", bbox_inches="tight")
plt.close(fig)
print("  fig_app_mle_convergence ok")


# Power-MDE figure was retired here in the 2026-05-27 cleanup; the
# Cohen's-d annotated lattice in paper/_generate_extra.py (fig_power_mde)
# is the canonical Cohen's-d power view, and the appendix wires it up
# at fig:power-mde.


# ----------------------------------------------------------------------
# F-app-6: Sector x sector cross-edge heatmap (3 snapshots)
# ----------------------------------------------------------------------
PICK = [("Oct 2008 GFC", "(a) Oct 2008 GFC"),
        ("Mar 2020 COVID", "(b) Mar 2020 COVID"),
        ("2024 Contemporary", "(c) 2024 baseline")]

def sector_edge_matrix(snap):
    """Directed cross-sector edge counts normalised by row-sector node
    count (so the heatmap reads as 'edges per source-sector node into
    each target sector', a density rather than a raw count)."""
    rec = stage3[snap]
    adj = rec["directed_adj"]
    tickers = rec["tickers"]
    sec = [sector_map.get(t, "Unknown") for t in tickers]
    # node count per source sector
    counts = pd.Series(sec).value_counts().to_dict()
    M = np.zeros((len(SECTOR_ORDER), len(SECTOR_ORDER)))
    idx = {s: i for i, s in enumerate(SECTOR_ORDER)}
    # adj is a sparse matrix or dense; coerce to coo
    import scipy.sparse as sp
    if sp.issparse(adj):
        coo = adj.tocoo()
        rows, cols = coo.row, coo.col
    else:
        rows, cols = np.nonzero(np.asarray(adj))
    for r, c in zip(rows, cols):
        sr, sc = sec[r], sec[c]
        if sr in idx and sc in idx:
            M[idx[sr], idx[sc]] += 1
    # Normalise by source sector node count
    for s, i in idx.items():
        if counts.get(s, 0) > 0:
            M[i, :] = M[i, :] / counts[s]
    return M

mats = [sector_edge_matrix(s) for s, _ in PICK]
vmax = max(m.max() for m in mats)

fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.0),
                         gridspec_kw={"wspace": 0.05})
short_sec = [s.replace("Information ", "Info ")
              .replace("Communication ", "Comm. ")
              .replace("Consumer ", "Cons. ")
              .replace("Discretionary", "Disc.")
              .replace("Real Estate", "RealEst")
             for s in SECTOR_ORDER]
for ax, (snap, title), M in zip(axes, PICK, mats):
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(SECTOR_ORDER)))
    ax.set_yticks(np.arange(len(SECTOR_ORDER)))
    ax.set_xticklabels(short_sec, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(short_sec if ax is axes[0] else [], fontsize=7)
    ax.set_title(title)
    ax.set_xlabel("Target sector")
    if ax is axes[0]:
        ax.set_ylabel("Source sector")
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, pad=0.02,
                    label="Directed edges per source-sector node")
fig.tight_layout()
fig.savefig(FIG / "fig_app_sector_heatmap.pdf", bbox_inches="tight")
fig.savefig(FIG / "fig_app_sector_heatmap.png", bbox_inches="tight")
plt.close(fig)
print("  fig_app_sector_heatmap ok")


# The Bonferroni-10, Stage-3 sweep, and ADF-rejection tables were
# retired here on 2026-05-27; paper/_generate_extra.py now owns those
# three artefacts under the labels tab:bonf-z-per-snapshot,
# tab:stage3-sensitivity, and tab:adf-rejection-per-snapshot.

print("\nAll addon assets generated.")
