"""
Publication-quality figure generator for the IEEE journal manuscript.

This script regenerates Fig. 1-7 of the paper from the Stage-1, Stage-2,
Stage-4, and (optionally) Stage-5 cached pickles produced by
``run_pipeline.py``. The figures are intentionally re-aggregated here
rather than computed in the per-stage modules so that the manuscript's
visual encoding (colour palette, axis ranges, annotation thresholds)
stays decoupled from the underlying numerical pipeline. All figures
are written as both PDF (vector) and PNG (300 dpi) into ``paper/figures``.

Run from the project root::

    py paper/generate_figures.py

Outputs
-------
fig1_impact_map.pdf
    Strategic impact map: directed networks for two crisis snapshots
    (Oct 2008 GFC, Mar 2020 COVID) and the 2024 baseline panel,
    drawn with a shared ticker-keyed layout so that densification
    is read as "same nodes, fewer edges" rather than "different
    universes".
fig2_success_criteria.pdf
    Four-panel proposal-criteria dashboard: ER clustering Z (Q1),
    PageRank Gini (Q2), modularity / sector purity (Q3), and the
    FFL / MR / SIM motif Z-scores (Q4).
fig3_motif_profile.pdf
    Significance-profile (SP) trajectory of the three motifs across
    snapshots, with crisis windows shaded.
fig4_correlation_heatmap.pdf
    Sector-block-ordered A-DCC correlation matrices for the two
    crisis windows and two baseline windows.
fig5_pagerank_leaders.pdf
    Top-3 PageRank leaders per snapshot, displayed as a union heatmap
    so that snapshot-specific leaders (e.g. DOC in Mar 2020) are not
    suppressed by global ranking.
fig6_robustness.pdf
    Density-invariant robustness diagnostics: clustering-excess Z-score
    |Z_C| (replacing the withdrawn small-world sigma, undefined on the
    fragmented Stage-3 graphs) and density-matched modularity Q.

Note: an earlier ``fig7_nsi.pdf`` (NSI bar chart with three components)
was removed in 2026-05-27. The current paper uses a four-channel NSI
(s, h, rho_bar, mu); the legacy figure rendered only three and
hard-coded an outdated VIX-correlation annotation, so it was deleted
rather than carried as a stale artefact. The conclusion section in
paper.tex now embeds an equivalent NSI ranking chart as a TikZ block
(no external image dependency).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Force UTF-8 stdout on Windows hosts so the en-dash / em-dash glyphs
# in the print banners do not crash the script under cp1254.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from pathlib import Path

# --- Paths -------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
SNAP_DIR = ROOT / "results" / "snapshots"
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# --- Matplotlib defaults ----------------------------------------------
# Serif body font, 9 pt base, 300 dpi raster fallback. Matches the
# IEEE conference-proceedings template used by the manuscript build.
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# --- Regime palette ---------------------------------------------------
# Cool diverging register matched to the fig9 viridis heatmap and the
# fig4 RdBu correlation maps: crisis at the warm (high-stress) end,
# baseline at the cool end, stress/recovery between. Tuned to stay
# distinguishable in greyscale and to read as a single stress axis
# rather than four unrelated hues.
REGIME_COLORS = {
    "crisis":   "#B2182B",   # deep red (RdBu warm extreme)
    "stress":   "#E08214",   # amber
    "recovery": "#1A9850",   # teal-green
    "baseline": "#2166AC",   # steel blue (RdBu cool extreme)
}

# Neutral accents: a soft panel fill, a highlight, and graphite for
# text/strokes. IMPERIAL_* names retained so existing call sites keep
# working; the values are now the cool-register neutrals.
IMPERIAL_IVORY    = "#EEF2F5"   # soft cool-grey panel fill
IMPERIAL_GOLD_LT  = "#E08214"   # amber highlight (matches stress)
IMPERIAL_GRAPHITE = "#2A2E33"

# --- Sector palette ----------------------------------------------------
# Eleven mutually-distinct hues on a cool-leaning register (no
# fluorescent pastels). Used by Fig. 1 (node fill) and the multi-panel
# sector legend.
SECTOR_COLORS = {
    "Information Technology":  "#2166AC",   # steel blue
    "Financials":              "#E08214",   # amber
    "Health Care":             "#1A9850",   # teal-green
    "Consumer Discretionary":  "#B2182B",   # deep red
    "Industrials":             "#762A83",   # violet
    "Communication Services":  "#0E7C7B",   # peacock teal
    "Consumer Staples":        "#C51B7D",   # magenta-rose
    "Energy":                  "#8C6D31",   # bronze
    "Utilities":               "#F1A340",   # light amber
    "Materials":               "#5A6B7B",   # slate
    "Real Estate":             "#D6604D",   # coral
}

# Compact x-axis labels reused across every multi-snapshot figure.
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

# Canonical chronological ordering used by every multi-snapshot panel
# so that bar order, heatmap column order, and line-plot x-axes agree.
SNAP_ORDER = [
    "Oct 1987 Black Monday", "1990-91 Recession", "1993 Calm",
    "Dec 1994 Tequila", "Oct 1997 Asian Crisis", "Oct 1998 LTCM",
    "Apr 2000 Dot-com", "Sep 2001 9/11", "Jul 2002 WorldCom",
    "2005 Calm", "2007 Subprime Buildup", "Oct 2008 GFC",
    "Mar 2009 Recovery", "2013 Calm", "2017 Calm", "Q4 2018 VolShock",
    "Mar 2020 COVID", "Jun 2020 Stable", "2022 Rate Hikes",
    "2024 Contemporary",
]

# --- Inference-macro lookup -------------------------------------------
# The paper writes statistical macro values to
# paper/output/inference_macros.tex; figures should read from the same
# file so caption and figure cannot diverge. _read_macros returns a
# {name -> string} dict; values are kept as strings so the figure can
# decide whether to cast to float or render verbatim.
import re as _re
_MACRO_RE = _re.compile(r"\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}")
_MACRO_FILE = Path(__file__).resolve().parent / "output" / "inference_macros.tex"

def _read_macros():
    out = {}
    if not _MACRO_FILE.exists():
        return out
    with open(_MACRO_FILE, "r", encoding="utf-8") as fh:
        for line in fh:
            m = _MACRO_RE.match(line.strip())
            if m:
                out[m.group(1)] = m.group(2)
    return out

INF_MACROS = _read_macros()

# --- Load Stage 1 / 2 / 4 caches (proposal-aligned scope) --------------
# Stage 5 (NSI) is loaded lazily inside the Fig. 7 block so that
# proposal-only re-runs of Fig. 1-6 succeed even when stage5_results.pkl
# does not exist.
print("Loading cached results...")
with open(SNAP_DIR / "stage1_results.pkl", "rb") as f:
    stage1 = pickle.load(f)
with open(SNAP_DIR / "stage2_results.pkl", "rb") as f:
    stage2 = pickle.load(f)
with open(SNAP_DIR / "stage4_results.pkl", "rb") as f:
    stage4 = pickle.load(f)

snapshot_corr = stage1["snapshot_correlations"]
sp500_info = pd.read_parquet(ROOT / "data" / "sp500_info.parquet")
sector_map = dict(zip(sp500_info["Symbol"], sp500_info["GICS Sector"]))


# --- Figure 1: Strategic Impact Map -----------------------------------
def draw_directed_network(label, ax, title, anchor_pos=None, min_degree=2):
    """
    Render one panel of Fig. 1 (the strategic impact map).

    The visual hero of Fig. 1 is the *edge set*, not the node set:
    nodes are drawn as uniform sector-coloured dots while edge width
    and alpha encode |partial correlation| via a three-tier quantile
    binning. This makes the density gap between panels (e.g. 1884 vs
    456 edges) read as a visible fullness difference rather than being
    masked by PageRank-sized hub bubbles.

    Parameters
    ----------
    label : str
        Snapshot key into ``stage4`` (e.g. ``"Oct 2008 GFC"``).
    ax : matplotlib.axes.Axes
        Target panel axis.
    title : str
        Panel-letter caption (e.g. ``"(a) Oct 2008 GFC"``).
    anchor_pos : dict[str, (float, float)] or None
        Shared ticker-keyed layout dictionary so that crisis and
        baseline panels position the same ticker at the same xy
        coordinates. Tickers absent from the anchor receive a local
        spring-layout fallback.
    min_degree : int
        Hide nodes with fewer than this many incident edges. The
        default of 2 suppresses leaf nodes that would otherwise
        clutter the baseline panel.

    Notes
    -----
    Edge density is computed on the unfiltered graph G (not on the
    ``min_degree``-filtered subgraph H) so that the printed
    ``density %`` annotation matches the underlying network rather
    than the trimmed visualisation.
    """
    data = stage4[label]
    G = data["graph"]  # NetworkX DiGraph (nodes are int indices)
    tickers = data["tickers"]
    pr = data["pagerank"]["pagerank_scores"]

    # Hide leaf and isolated nodes so the sparse baseline panel reads
    # as visibly sparse; the filter is applied to the full degree, not
    # after intersection with the anchor layout.
    keep = [n for n in G.nodes() if G.degree(n) >= min_degree]
    H = G.subgraph(keep).copy()

    # Use the shared ticker-keyed anchor layout when available;
    # tickers that did not enter the anchor (e.g. dropped from one
    # snapshot's universe) receive a local spring-layout fallback.
    if anchor_pos is not None:
        pos = {}
        orphans = []
        for n in H.nodes():
            t = tickers[n] if n < len(tickers) else None
            if t and t in anchor_pos:
                pos[n] = anchor_pos[t]
            else:
                orphans.append(n)
        if orphans:
            sub_pos = nx.spring_layout(H.subgraph(orphans), seed=42, k=0.4)
            pos.update(sub_pos)
    else:
        pos = nx.spring_layout(H, seed=42,
                               k=2.8 / np.sqrt(max(1, H.number_of_nodes())),
                               iterations=80)

    # Three-tier edge rendering (low / mid / high |weight|): edges are
    # drawn first and in tiered alpha/width so that strong partial
    # correlations remain readable through crowded regions. This is the
    # key mechanism that makes the panel edge-dominant rather than
    # node-dominant.
    edges_list = list(H.edges())
    if edges_list:
        weights = np.array([abs(H[u][v].get("weight", 1.0))
                            for u, v in edges_list])
        if len(weights) > 6 and weights.max() > weights.min():
            q1, q2 = np.quantile(weights, [0.33, 0.67])
        else:
            q1 = q2 = float(weights.mean() if len(weights) else 0)

        tiers = [
            ([e for e, w in zip(edges_list, weights) if w <= q1],
             0.22, 0.35),
            ([e for e, w in zip(edges_list, weights) if q1 < w <= q2],
             0.38, 0.55),
            ([e for e, w in zip(edges_list, weights) if w > q2],
             0.55, 0.80),
        ]
        for elist, e_alpha, e_width in tiers:
            if not elist:
                continue
            nx.draw_networkx_edges(H, pos, ax=ax, edgelist=elist,
                                   alpha=e_alpha, width=e_width,
                                   edge_color="#202020", arrows=True,
                                   arrowsize=4, arrowstyle="-|>",
                                   connectionstyle="arc3,rad=0.08")

    # Uniform tiny nodes coloured by GICS sector only; the deliberate
    # absence of PageRank size scaling keeps the panel edge-dominated.
    node_colors = [SECTOR_COLORS.get(sector_map.get(tickers[n], ""), "#cccccc")
                   for n in H.nodes()]
    nx.draw_networkx_nodes(H, pos, ax=ax, node_size=14,
                           node_color=node_colors, alpha=0.95,
                           edgecolors="white", linewidths=0.25)

    # Label only the top-three PageRank nodes per panel. This keeps
    # the figure readable at column width while still naming the
    # leaders the manuscript discusses.
    kept = set(H.nodes())
    top3 = sorted(((n, s) for n, s in pr.items() if n in kept),
                  key=lambda x: x[1], reverse=True)[:3]
    labels = {n: tickers[n] for n, _ in top3 if n < len(tickers)}
    nx.draw_networkx_labels(H, pos, labels=labels, ax=ax,
                            font_size=6.5, font_weight="bold")

    # Title carries the unfiltered edge count and density so the reader
    # has a numerical anchor that matches the visual sparsity contrast.
    n_full = G.number_of_nodes()
    m_full = G.number_of_edges()
    density_pct = 100 * m_full / (n_full * (n_full - 1)) if n_full > 1 else 0
    ax.set_title(f"{title}\n{m_full} edges,  density {density_pct:.2f}%",
                 fontsize=9, fontweight="bold")
    ax.axis("off")


print("\n[Fig 1] Strategic Impact Maps (two crises + baseline)...")

# Build the shared ticker-keyed layout from the *union* of the three
# panels' edge sets. Using the union (rather than per-panel layouts)
# ensures every ticker that appears in any panel keeps a stable xy
# position, so the crisis-vs-baseline contrast reads as "same nodes,
# fewer edges" instead of as a relayout artefact.
_fig1_labels = ("Oct 2008 GFC", "Mar 2020 COVID", "2024 Contemporary")
_union = nx.DiGraph()
for _lbl in _fig1_labels:
    _d = stage4[_lbl]
    _tk = _d["tickers"]
    for _u, _v in _d["graph"].edges():
        if _u < len(_tk) and _v < len(_tk):
            _union.add_edge(_tk[_u], _tk[_v])
anchor_pos = nx.spring_layout(
    _union, seed=42,
    k=2.8 / np.sqrt(max(1, _union.number_of_nodes())),
    iterations=120)
print(f"  Shared layout: {_union.number_of_nodes()} tickers, "
      f"{_union.number_of_edges()} union edges")

fig, axes = plt.subplots(1, 3, figsize=(7.16, 3.0))

draw_directed_network("Oct 2008 GFC", axes[0],
                      "(a) Oct 2008 GFC", anchor_pos=anchor_pos)
draw_directed_network("Mar 2020 COVID", axes[1],
                      "(b) Mar 2020 COVID", anchor_pos=anchor_pos)
draw_directed_network("2024 Contemporary", axes[2],
                      "(c) 2024 Baseline", anchor_pos=anchor_pos)

# Sector legend lists only sectors actually present across the three
# panels; this avoids a swatch for sectors that never appear in Fig. 1.
sectors_used = set()
for label_key in ("Oct 2008 GFC", "Mar 2020 COVID", "2024 Contemporary"):
    for t in stage4[label_key]["tickers"]:
        s = sector_map.get(t)
        if s:
            sectors_used.add(s)
handles = [mpatches.Patch(color=c, label=s)
           for s, c in SECTOR_COLORS.items() if s in sectors_used]
fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=6.5,
           frameon=False, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout(rect=[0, 0.10, 1, 1])
fig.savefig(FIG_DIR / "fig1_impact_map.pdf")
fig.savefig(FIG_DIR / "fig1_impact_map.png", dpi=300)
print(f"  Saved fig1_impact_map.pdf")


# --- Figure 2: Success-Criteria Dashboard ------------------------------
# Four-panel summary of the proposal §6.4 success criteria (Q1-Q4):
# ER clustering Z, PageRank Gini, modularity Q + sector purity, and
# the directed motif Z-score significance profile.
print("\n[Fig 2] Success Criteria Dashboard...")

# Re-aggregate the per-snapshot endpoints into a single DataFrame so
# that all four panels iterate over the same row order and colour
# vector without having to re-look-up Stage-4 keys per panel.
rows = []
for label in SNAP_ORDER:
    if label not in stage4:
        continue
    d = stage4[label]
    rows.append({
        "snapshot": label,
        "short": SHORT_LABEL[label],
        "regime": d["regime"],
        "edges":  d["n_edges"],
        "zC":     abs(d["erdos_renyi"]["z_scores"]["clustering"]),
        "gini":   d["pagerank"]["gini"],
        "hhi":    d["pagerank"]["hhi_top10"],
        "Q":      d["community"]["modularity"],
        "purity": d["community"]["purity"],
        "ffl_z":  d["motifs"]["z_scores"]["feed_forward_loop"] if d.get("motifs") else np.nan,
        "mr_z":   d["motifs"]["z_scores"]["mutual_regulation"] if d.get("motifs") else np.nan,
        "sim_z":  d["motifs"]["z_scores"]["single_input_module"] if d.get("motifs") else np.nan,
    })
df = pd.DataFrame(rows)
colors = [REGIME_COLORS[r] for r in df["regime"]]
x = np.arange(len(df))

fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.2))

# --- Panel (a): Erdos-Renyi clustering |Z| (Q1) ---
ax = axes[0, 0]
ax.bar(x, df["zC"], color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
ax.axhline(2, color="black", linestyle="--", linewidth=0.6, alpha=0.5)
ax.set_ylabel("$|Z_C|$", fontweight="bold")
ax.set_title("(a) Erdos-Renyi deviation (Q1)", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(df["short"], rotation=40, ha="right", fontsize=7)
ax.text(len(df) - 0.4, 2.2, "Non-random threshold", fontsize=6.5,
        ha="right", va="bottom", color="#555555", style="italic")

# --- Panel (b): PageRank Gini concentration (Q2) ---
ax = axes[0, 1]
ax.bar(x, df["gini"], color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
ax.set_ylabel("Gini coefficient", fontweight="bold")
ax.set_title("(b) PageRank concentration (Q2)", fontweight="bold")
ax.set_ylim(0, max(0.6, df["gini"].max() * 1.1))
ax.set_xticks(x)
ax.set_xticklabels(df["short"], rotation=40, ha="right", fontsize=7)

# --- Panel (c): Louvain modularity Q + sector purity (Q3) ---
ax = axes[1, 0]
w = 0.38
ax.bar(x - w/2, df["Q"], w, color=REGIME_COLORS["baseline"],
       label="Modularity $Q$",
       alpha=0.9, edgecolor="white", linewidth=0.5)
ax.bar(x + w/2, df["purity"], w, color=REGIME_COLORS["stress"],
       label="Purity",
       alpha=0.9, edgecolor="white", linewidth=0.5)
ax.set_ylabel("Value", fontweight="bold")
ax.set_title("(c) Community structure (Q3)", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(df["short"], rotation=40, ha="right", fontsize=7)
ax.legend(loc="upper right", fontsize=7)
ax.set_ylim(0, 1.0)

# --- Panel (d): Directed motif Z-scores (Q4) ---
ax = axes[1, 1]
w = 0.28
# Motif palette: claret for FFL (the only positively-deviating motif on
# this panel), emerald for MR, navy for SIM. Matches the imperial regime
# tones so the dashboard reads as one chromatic system.
ax.bar(x - w, df["ffl_z"], w, color=REGIME_COLORS["crisis"],
       label="FFL (feed-forward)",
       alpha=0.9, edgecolor="white", linewidth=0.5)
ax.bar(x,     df["mr_z"],  w, color=REGIME_COLORS["recovery"],
       label="MR (mutual reg.)",
       alpha=0.9, edgecolor="white", linewidth=0.5)
ax.bar(x + w, df["sim_z"], w, color=REGIME_COLORS["baseline"],
       label="SIM (single input)",
       alpha=0.9, edgecolor="white", linewidth=0.5)
ax.axhline(0, color="black", linewidth=0.6)
ax.set_ylabel("Motif Z-score", fontweight="bold")
ax.set_title("(d) Motif significance (Q4)", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(df["short"], rotation=40, ha="right", fontsize=7)
ax.legend(loc="lower left", fontsize=6.5, ncol=1, framealpha=0.9)

# Shared regime legend below the 2x2 grid.
handles = [mpatches.Patch(color=c, label=r.capitalize())
           for r, c in REGIME_COLORS.items()]
fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7.5,
           frameon=False, bbox_to_anchor=(0.5, -0.015))

plt.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig(FIG_DIR / "fig2_success_criteria.pdf")
fig.savefig(FIG_DIR / "fig2_success_criteria.png", dpi=300)
print("  Saved fig2_success_criteria.pdf")


# --- Figure 3: Motif Significance-Profile trajectory -------------------
# Plots the FFL / MR / SIM significance-profile values across the ten
# snapshots in chronological order; crisis windows are shaded so the
# regime transitions in the SP series stand out at column width.
print("\n[Fig 3] Motif Significance Profile...")
fig, ax = plt.subplots(figsize=(7.16, 3.2))

# Build the per-snapshot SP table; snapshots whose motif analysis was
# skipped (or did not converge) are omitted entirely.
sp_df = []
for label in SNAP_ORDER:
    if label not in stage4 or stage4[label].get("motifs") is None:
        continue
    sp = stage4[label]["motifs"]["significance_profile"]
    sp_df.append({
        "short": SHORT_LABEL[label],
        "regime": stage4[label]["regime"],
        "FFL": sp["feed_forward_loop"],
        "MR":  sp["mutual_regulation"],
        "SIM": sp["single_input_module"],
    })
sp_df = pd.DataFrame(sp_df)
x = np.arange(len(sp_df))

ax.plot(x, sp_df["FFL"], "o-", color=REGIME_COLORS["crisis"],
        label="FFL (feed-forward loop)",
        linewidth=1.6, markersize=6, markeredgecolor="white",
        markeredgewidth=0.5)
ax.plot(x, sp_df["MR"],  "s-", color=REGIME_COLORS["recovery"],
        label="MR (mutual regulation)",
        linewidth=1.6, markersize=6, markeredgecolor="white",
        markeredgewidth=0.5)
ax.plot(x, sp_df["SIM"], "^-", color=REGIME_COLORS["baseline"],
        label="SIM (single input module)",
        linewidth=1.6, markersize=6, markeredgecolor="white",
        markeredgewidth=0.5)
ax.axhline(0, color=IMPERIAL_GRAPHITE, linewidth=0.6, alpha=0.7)

# Shade crisis snapshots so the regime shift at the GFC and COVID
# peaks is visible without colouring the lines themselves. Tint uses
# the regime claret so the same crisis hue cues across all figures.
for i, row in sp_df.iterrows():
    if row["regime"] == "crisis":
        ax.axvspan(i - 0.35, i + 0.35, alpha=0.14,
                   color=REGIME_COLORS["crisis"], zorder=0)

ax.set_ylabel("Normalised Z-score (SP)", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(sp_df["short"], rotation=30, ha="right", fontsize=8)
ax.set_title("Motif Significance Profile Across Market Regimes\n"
             "(shaded = crisis peak; $SP_i = Z_i / \\sqrt{\\sum_j Z_j^2}$)",
             fontweight="bold")
ax.legend(loc="upper right", fontsize=8, framealpha=0.95)
ax.set_ylim(-1.05, 1.05)
ax.grid(True, alpha=0.25, linewidth=0.5)

plt.tight_layout()
fig.savefig(FIG_DIR / "fig3_motif_profile.pdf")
fig.savefig(FIG_DIR / "fig3_motif_profile.png", dpi=300)
print("  Saved fig3_motif_profile.pdf")


# --- Figure 4: A-DCC Correlation Heatmaps -----------------------------
# 2x2 grid of snapshot-averaged R-bar matrices for the two crisis
# windows and two baseline windows, sector-block-ordered along the
# diagonal so within-sector clustering is visible.
print("\n[Fig 4] Correlation heatmaps (2 crises + 2 baselines)...")
fig, axes = plt.subplots(2, 2, figsize=(7.16, 6.2))

panels = [
    ("Oct 2008 GFC",       axes[0, 0], "(a) Oct 2008 Crisis"),
    ("Mar 2020 COVID",     axes[0, 1], "(b) Mar 2020 Crisis"),
    ("2017 Calm",          axes[1, 0], "(c) 2017 Calm baseline"),
    ("2024 Contemporary",  axes[1, 1], "(d) 2024 Contemporary baseline"),
]

im = None
for label, ax, panel in panels:
    R = snapshot_corr[label]["R_avg"]
    tickers = snapshot_corr[label]["tickers"]

    # Reorder rows / columns by GICS sector so within-sector blocks
    # appear on the diagonal; tie-break by ticker symbol for stable
    # ordering across snapshots.
    sectors = [sector_map.get(t, "Other") for t in tickers]
    order = sorted(range(len(tickers)), key=lambda i: (sectors[i], tickers[i]))
    R_sorted = R[np.ix_(order, order)]

    im = ax.imshow(R_sorted, cmap="RdBu_r", vmin=-0.1, vmax=0.8, aspect="auto")
    triu = np.triu_indices_from(R_sorted, k=1)
    mean_r = R_sorted[triu].mean()
    ax.set_title(f"{panel}\n" + r"$\bar{\rho}$" + f" = {mean_r:.3f}",
                 fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])

fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7,
             label="Pairwise correlation")
fig.savefig(FIG_DIR / "fig4_correlation_heatmap.pdf")
fig.savefig(FIG_DIR / "fig4_correlation_heatmap.png", dpi=300)
print("  Saved fig4_correlation_heatmap.pdf")


# --- Figure 5: PageRank Leader Heatmap --------------------------------
# Per-snapshot PageRank scores for the union of each window's top-3
# leaders, displayed as a row-per-ticker heatmap so that snapshot-
# specific hubs (e.g. DOC in Mar 2020) are not suppressed by ranking
# against the global mean.
print("\n[Fig 5] PageRank leader matrix...")

# Build the union of top-3 leaders across all snapshots, then sort
# rows by each ticker's max PR so the heatmap reads roughly
# descending in the most-frequent leader rows.
top3_set = set()
ticker_max_pr = {}
for label in SNAP_ORDER:
    if label not in stage4:
        continue
    pr = stage4[label]["pagerank"]["pagerank_scores"]
    tickers = stage4[label]["tickers"]
    for n, score in pr.items():
        if n < len(tickers):
            t = tickers[n]
            ticker_max_pr[t] = max(ticker_max_pr.get(t, 0), score)
    pr_sorted = sorted(pr.items(), key=lambda kv: kv[1], reverse=True)[:3]
    for n, _ in pr_sorted:
        if n < len(tickers):
            top3_set.add(tickers[n])

top_tickers = sorted(top3_set, key=lambda t: ticker_max_pr[t], reverse=True)

# Assemble the (ticker x snapshot) matrix; missing entries (ticker
# absent from a snapshot's universe) remain NaN and render as the
# colormap's "bad" colour so absences are visually distinguishable
# from low-but-present scores.
mat = np.full((len(top_tickers), len(SNAP_ORDER)), np.nan)
for j, label in enumerate(SNAP_ORDER):
    if label not in stage4:
        continue
    pr = stage4[label]["pagerank"]["pagerank_scores"]
    tickers = stage4[label]["tickers"]
    tk2pr = {tickers[n]: pr[n] for n in pr if n < len(tickers)}
    for i, t in enumerate(top_tickers):
        if t in tk2pr:
            mat[i, j] = tk2pr[t]

fig, ax = plt.subplots(figsize=(7.16, 5.5))
im = ax.imshow(mat, cmap="viridis", aspect="auto")
ax.set_xticks(range(len(SNAP_ORDER)))
ax.set_xticklabels([SHORT_LABEL[s] for s in SNAP_ORDER],
                   rotation=35, ha="right", fontsize=8)
ax.set_yticks(range(len(top_tickers)))
ax.set_yticklabels(top_tickers, fontsize=8)
plt.colorbar(im, ax=ax, shrink=0.9, label="PageRank score")

# Outline crisis snapshot columns in claret to match the rest of the
# manuscript's regime palette.
crisis_cols = [i for i, s in enumerate(SNAP_ORDER) if stage4.get(s, {}).get("regime") == "crisis"]
for ci in crisis_cols:
    ax.add_patch(plt.Rectangle((ci - 0.5, -0.5), 1, len(top_tickers),
                               fill=False, edgecolor="#8C2D2D", linewidth=1.8))

ax.set_title("Top-3 PageRank Leaders Per Snapshot (Union Across Regimes)\n"
             "(red boxes = crisis snapshots)", fontweight="bold")
plt.tight_layout()
fig.savefig(FIG_DIR / "fig5_pagerank_leaders.pdf")
fig.savefig(FIG_DIR / "fig5_pagerank_leaders.png", dpi=300)
print("  Saved fig5_pagerank_leaders.pdf")


# --- Figure 6: Robustness Checks --------------------------------------
# Two density-invariant diagnostics: the clustering-excess Z-score
# |Z_C| against a matched G(n,m) ER null (replacing the withdrawn
# small-world sigma, whose path-length term is undefined on the
# fragmented Stage-3 graphs -- see appendix app:sigma) and the
# density-matched modularity Q. The density match is loaded from a
# separately-cached pickle and the panel is silently omitted if the
# cache does not exist.
print("\n[Fig 6] Robustness checks (clustering |Z_C| + density-matched Q)...")

# Per-snapshot clustering-excess Z-score, read directly from Stage-4's
# ER null bookkeeping. Clustering is a *local* statistic, so unlike the
# small-world sigma it stays well-defined when the graph fragments into
# many components; the null is the same G(n,m) bootstrap, so |Z_C| is
# density-matched by construction.
sw_rows = []
for label in SNAP_ORDER:
    if label not in stage4:
        continue
    er = stage4[label]["erdos_renyi"]
    sw_rows.append({
        "short": SHORT_LABEL[label],
        "regime": stage4[label]["regime"],
        "z_clustering": er["z_scores"]["clustering"],
    })
sw_df = pd.DataFrame(sw_rows)

# Density-matched modularity is produced by a separate script
# (src/stage4_network/density_matched.py); when the cache is missing
# the panel is skipped without aborting the figure run.
try:
    with open(SNAP_DIR / "density_matched_results.pkl", "rb") as f:
        dm_df = pickle.load(f)
    dm_df["short"] = dm_df["snapshot"].map(SHORT_LABEL)
    dm_df = dm_df.set_index("snapshot").reindex(SNAP_ORDER).reset_index()
    dm_df["short"] = dm_df["snapshot"].map(SHORT_LABEL)
    dm_available = True
except FileNotFoundError:
    print("  (density_matched_results.pkl not found — "
          "run src/stage4_network/density_matched.py first)")
    dm_available = False

if dm_available:
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.2))

    # --- Panel (a): clustering-excess Z-score (replaces withdrawn sigma) ---
    # Bars are |Z_C| against the matched ER null. The dashed claret and
    # dotted graphite lines mark the crisis and baseline regime means;
    # they sit almost on top of one another, which is the panel's point:
    # clustering is strongly non-random in every window yet does not
    # discriminate crises (see app:sigma for the verdict).
    ax = axes[0]
    x = np.arange(len(sw_df))
    colors_sw = [REGIME_COLORS[r] for r in sw_df["regime"]]
    zc = sw_df["z_clustering"].abs().astype(float)
    ax.bar(x, zc, color=colors_sw, alpha=0.88,
           edgecolor=IMPERIAL_GRAPHITE, linewidth=0.5)
    cri_mean = float(zc[sw_df["regime"] == "crisis"].mean())
    base_mean = float(zc[sw_df["regime"] == "baseline"].mean())
    ax.axhline(cri_mean, color=REGIME_COLORS["crisis"], linestyle="--",
               linewidth=0.9, alpha=0.9)
    ax.axhline(base_mean, color=IMPERIAL_GRAPHITE, linestyle=":",
               linewidth=0.9, alpha=0.9)
    ax.text(0.03, 0.97,
            f"crisis {cri_mean:.1f} $\\approx$ baseline {base_mean:.1f}\n"
            "(non-discriminating)",
            transform=ax.transAxes, fontsize=6.5, ha="left", va="top",
            color=IMPERIAL_GRAPHITE, style="italic")
    ax.set_ylabel(r"Clustering excess $|Z_C|$ vs. ER null",
                  fontweight="bold")
    ax.set_title("(a) Clustering excess $|Z_C|$ (Q1, density-matched)",
                 fontweight="bold", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(sw_df["short"], rotation=40, ha="right", fontsize=7)
    ax.set_ylim(0, float(zc.max()) * 1.18)
    # In-bar value annotations (clustering Z is O(10-70) on a linear axis).
    for xi, val in zip(x, zc):
        if not np.isfinite(val):
            continue
        ax.text(xi, val + 0.8, f"{val:.0f}",
                ha="center", va="bottom", fontsize=6.0,
                color=IMPERIAL_GRAPHITE, fontweight="bold")

    # --- Panel (b): density-matched Q and Q_rel ---
    ax = axes[1]
    colors_dm = [REGIME_COLORS[r] for r in dm_df["regime"]]
    width = 0.38
    x = np.arange(len(dm_df))
    ax.bar(x - width/2, dm_df["Q"], width, color=REGIME_COLORS["baseline"],
           label="$Q$ (density-matched)", alpha=0.9,
           edgecolor="white", linewidth=0.5)
    ax.bar(x + width/2, dm_df["Q_rel"], width, color=REGIME_COLORS["stress"],
           label=r"$Q_{\mathrm{rel}}$",
           alpha=0.9, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Value", fontweight="bold")
    ax.set_title("(b) Density-matched $Q$ (Q3, corrected)",
                 fontweight="bold", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(dm_df["short"], rotation=40, ha="right", fontsize=7)
    ax.legend(loc="upper left", fontsize=7)
    ymax = max(dm_df["Q"].max(), dm_df["Q_rel"].max()) * 1.15
    ax.set_ylim(0, ymax)

    # Headline annotation summarising the Q3 finding: crisis vs
    # baseline mean Q plus the cross-sector edge fraction at the
    # k = 4637 budget reported in §IV-C of the manuscript.
    crisis_mean_Q = dm_df.loc[dm_df["regime"] == "crisis", "Q"].mean()
    base_mean_Q   = dm_df.loc[dm_df["regime"] == "baseline", "Q"].mean()
    ax.text(0.98, 0.97,
            (f"crisis Q = {crisis_mean_Q:.3f}\n"
             f"baseline Q = {base_mean_Q:.3f}\n"
             f"cross-sector edges (top-4637): +31 %"),
            transform=ax.transAxes, ha="right", va="top",
            fontsize=6.5, color="#222222",
            bbox=dict(facecolor="white", edgecolor="#888888",
                      linewidth=0.4, boxstyle="round,pad=0.3", alpha=0.92))

    # Shared regime legend below the two panels.
    handles = [mpatches.Patch(color=c, label=r.capitalize())
               for r, c in REGIME_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7.5,
               frameon=False, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(FIG_DIR / "fig6_robustness.pdf")
    fig.savefig(FIG_DIR / "fig6_robustness.png", dpi=300)
    print("  Saved fig6_robustness.pdf")


# --- Figure 7 removed (2026-05-27) ------------------------------------
# The legacy fig7_nsi.pdf rendered NSI with three normalised components
# (s, h, rho_bar) and an early hard-coded VIX-correlation annotation.
# The current paper has migrated to a four-channel NSI (with the FFL
# motif-shift channel mu) and to auto-generated inference macros, so the
# figure had drifted out of sync on both axes. The conclusion section in
# paper.tex now embeds an equivalent NSI ranking chart as native TikZ,
# which keeps the visual self-consistent with tab:nsi without an
# external image asset. Intentionally not regenerated here.


# --- Figure 5 (main paper): NSI--VIX time series overlay --------------
# Daily VIX line with snapshot windows shaded by regime, and a
# secondary axis showing the snapshot NSI bars + window-mean VIX
# circles aligned to the same calendar. Generated only when both
# stage5_results.pkl (snapshot NSI) and data/vix.parquet (daily VIX
# series) are available.
print("Building Fig. 5 (NSI--VIX time series overlay)...")
_vix_parquet = ROOT / "data" / "vix_continuity.parquet"
# Enabled post-#11: the spliced VXO+VIX continuity series
# (data/vix_continuity.parquet -- FRED VXOCLS 1986+ spliced into VIXCLS
# from 2003-09-22 under the Whaley convention) covers all twenty CRSP
# analytical windows from Oct 1987 onward, so the pre-2004 snapshots now
# have a benchmark to overlay; the figure binds to src.config.SNAPSHOTS.
_FIG5_NSI_READY = True
if (_FIG5_NSI_READY and (SNAP_DIR / "stage5_results.pkl").exists()
        and _vix_parquet.exists()):
    with open(SNAP_DIR / "stage5_results.pkl", "rb") as f:
        _stage5 = pickle.load(f)
    nsi_df = _stage5["snapshot_nsi"].copy()
    vix_df = pd.read_parquet(_vix_parquet)
    # vix_continuity.parquet stores the spliced level in a "Close" column;
    # fall back to the first column if the schema has been renamed.
    vix_col = "VIX" if "VIX" in vix_df.columns else vix_df.columns[0]
    vix_daily = vix_df[vix_col].dropna()

    # Shading windows ARE the analytical snapshot windows from
    # src.config.SNAPSHOTS (label -> start, end): the shaded span and the
    # window-mean VXO+VIX read on the bottom panel then coincide exactly
    # with the windows NSI is computed on -- no plot-only widening.
    from src.config import SNAPSHOTS as _SNAPSHOTS
    SNAP_DATES = {lab: (start, end) for (lab, start, end, _reg) in _SNAPSHOTS}
    # snapshot_nsi schema uses 'snapshot' column (not 'label'); keep the
    # local variable name `label` in the loop below since it iterates over
    # SNAP_ORDER strings.
    nsi_df["snapshot"] = nsi_df["snapshot"].astype(str)
    nsi_df = nsi_df.set_index("snapshot").reindex(SNAP_ORDER).reset_index()

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(7.0, 4.6), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.4], "hspace": 0.08}
    )

    # Top panel: continuous daily VIX with shaded snapshot windows.
    ax_top.plot(vix_daily.index, vix_daily.values,
                color="#444444", linewidth=0.6, zorder=2)
    for label in SNAP_ORDER:
        regime = nsi_df.loc[nsi_df["snapshot"] == label, "regime"].iloc[0]
        d0, d1 = SNAP_DATES[label]
        ax_top.axvspan(pd.Timestamp(d0), pd.Timestamp(d1),
                       color=REGIME_COLORS[regime], alpha=0.18, zorder=1)
    ax_top.set_ylabel("Daily VXO/VIX", fontweight="bold")
    ax_top.set_title("VXO+VIX continuity and snapshot NSI, 1986--2024",
                     fontweight="bold")
    ax_top.grid(axis="y", linestyle=":", alpha=0.4)

    # Bottom panel: snapshot NSI bars at window-midpoints + window-mean
    # VIX as open circles on a twinned right axis.
    midpoints = [pd.Timestamp(d0) + (pd.Timestamp(d1) - pd.Timestamp(d0)) / 2
                 for d0, d1 in (SNAP_DATES[l] for l in SNAP_ORDER)]
    bar_colors = [REGIME_COLORS[r] for r in nsi_df["regime"]]
    ax_bot.bar(midpoints, nsi_df["nsi"].values,
               width=120, color=bar_colors, edgecolor="#222222",
               linewidth=0.6, zorder=2)
    for xi, v in zip(midpoints, nsi_df["nsi"].values):
        ax_bot.text(xi, v + 0.015, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=6.5, color="#333333", fontweight="bold")
    ax_bot.set_ylabel("Snapshot NSI", fontweight="bold")
    ax_bot.set_ylim(0, 0.95)
    ax_bot.set_xlabel("Year")
    ax_bot.grid(axis="y", linestyle=":", alpha=0.4)

    ax_vix = ax_bot.twinx()
    # Window-mean VIX over each snapshot window (right axis).
    win_vix = []
    for d0, d1 in (SNAP_DATES[l] for l in SNAP_ORDER):
        mask = (vix_daily.index >= pd.Timestamp(d0)) & \
               (vix_daily.index <= pd.Timestamp(d1))
        win_vix.append(vix_daily.loc[mask].mean())
    ax_vix.scatter(midpoints, win_vix, marker="o", s=42,
                   facecolors="white", edgecolors="#222222",
                   linewidth=1.0, zorder=3, label="Window-mean VIX")
    ax_vix.set_ylabel("Window-mean VIX", fontweight="bold")

    # Regime legend, pushed to lower-left of the bottom panel so it does
    # not collide with the Pearson r annotation in the top-right corner
    # nor obscure the headline bar at 2020.
    regime_handles = [mpatches.Patch(color=c, label=r.capitalize())
                      for r, c in REGIME_COLORS.items()]
    leg = ax_bot.legend(handles=regime_handles + [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="white", markeredgecolor=IMPERIAL_GRAPHITE,
                   markersize=7, label="Window-mean VIX")],
        loc="lower left", ncol=5, fontsize=7, frameon=True,
        facecolor=IMPERIAL_IVORY, edgecolor=IMPERIAL_GOLD_LT,
        bbox_to_anchor=(0.0, 1.005))
    ax_bot.add_artist(leg)

    # Pearson r annotation in the bottom-right of the lower panel,
    # read live from inference_macros.tex so the figure cannot drift
    # from the paper's macro values (the prior hardcoded 0.769 / [.47, .97]
    # silently lagged a cache refresh and was caught only at visual audit).
    _r  = INF_MACROS.get("InfClusterPearson",       "?.???")
    _lo = INF_MACROS.get("InfClusterPearsonCILow",  "?.???")
    _hi = INF_MACROS.get("InfClusterPearsonCIHigh", "?.???")
    assert "?" not in _r,  "InfClusterPearson missing from inference_macros.tex"
    assert "?" not in _lo, "InfClusterPearsonCILow missing from inference_macros.tex"
    assert "?" not in _hi, "InfClusterPearsonCIHigh missing from inference_macros.tex"
    ax_bot.text(0.98, 0.97,
                f"Pearson $r$ = {_r}  (cluster-bootstrap 95% CI [{_lo}, {_hi}])",
                transform=ax_bot.transAxes, ha="right", va="top",
                fontsize=7, color=IMPERIAL_GRAPHITE,
                bbox=dict(facecolor=IMPERIAL_IVORY, edgecolor=IMPERIAL_GOLD_LT,
                          linewidth=0.6, boxstyle="round,pad=0.32",
                          alpha=0.96))

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig5_nsi_vix_overlay.pdf")
    fig.savefig(FIG_DIR / "fig5_nsi_vix_overlay.png", dpi=300)
    print("  Saved fig5_nsi_vix_overlay.pdf")
else:
    print("  [deferred] fig5_nsi_vix_overlay not regenerated: CRSP panel "
          "spans 1987-2024 but VIX data starts 2004 (index predates 1990); "
          "needs 20-window SNAP_DATES + pre-VIX vol proxy (#11).")


# --- Figure A.1 (appendix): Multipanel NSI heatmap --------------------
# 10 snapshots x 20 (panel size x kind) cells, shaded by NSI value.
# Reads results/multipanel/n{N}_{kind}/stage5_results.pkl per cell
# when the multipanel sweep cache exists; the per-panel directory naming
# (no underscore between "n" and the size) matches tools.run_multipanel.
print("Building Fig. A.1 (multipanel NSI heatmap)...")
MULTI_DIR = ROOT / "results" / "multipanel"
_panel_sizes = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
_panel_kinds = ["coverage", "adv"]
if MULTI_DIR.exists():
    grid = np.full((len(SNAP_ORDER), len(_panel_sizes) * 2), np.nan)
    col_labels = []
    col = 0
    for kind in _panel_kinds:
        for N in _panel_sizes:
            col_labels.append(f"N={N}\n{kind}")
            cache_path = MULTI_DIR / f"n{N}_{kind}" / "stage5_results.pkl"
            if not cache_path.exists():
                col += 1
                continue
            with open(cache_path, "rb") as f:
                _s5 = pickle.load(f)
            df = _s5["snapshot_nsi"]
            # snapshot_nsi schema: 'snapshot' column, NOT 'label'.
            df = df.set_index("snapshot")
            for row, lbl in enumerate(SNAP_ORDER):
                if lbl in df.index:
                    grid[row, col] = df.loc[lbl, "nsi"]
            col += 1

    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    im = ax.imshow(grid, aspect="auto", cmap="viridis",
                   vmin=0.10, vmax=0.95)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=70, ha="right", fontsize=6.5)
    ax.set_yticks(range(len(SNAP_ORDER)))
    ax.set_yticklabels([SHORT_LABEL[l] for l in SNAP_ORDER], fontsize=8)
    # Vertical separator between coverage and ADV blocks.
    ax.axvline(len(_panel_sizes) - 0.5, color="white", linewidth=2.0)
    ax.text(len(_panel_sizes) / 2 - 0.5, -1.0, "coverage",
            ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.text(len(_panel_sizes) * 1.5 - 0.5, -1.0, "ADV",
            ha="center", va="bottom", fontsize=9, fontweight="bold")
    # Cell annotations for the headline N=500 columns (high-readability
    # zone where the two rules near-coincide).
    for row in range(len(SNAP_ORDER)):
        for col_idx in (len(_panel_sizes) - 1,
                        2 * len(_panel_sizes) - 1):
            v = grid[row, col_idx]
            if not np.isnan(v):
                ax.text(col_idx, row, f"{v:.2f}",
                        ha="center", va="center", fontsize=6.5,
                        color="white" if v > 0.55 else "black",
                        fontweight="bold")
    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("NSI", fontweight="bold")
    ax.set_title("Multipanel NSI heatmap "
                 r"($\{50,100,\ldots,500\}\times\{$coverage, ADV$\}$)",
                 fontweight="bold", pad=22)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig9_multipanel_nsi_heatmap.pdf")
    fig.savefig(FIG_DIR / "fig9_multipanel_nsi_heatmap.png", dpi=300)
    print("  Saved fig9_multipanel_nsi_heatmap.pdf")
else:
    print(f"  [warn] {MULTI_DIR} not found -- "
          "fig9_multipanel_nsi_heatmap skipped")


# --- Figure D.1 (appendix): Stage-2 lambda* selection diagnostic ------
# Per-snapshot unconstrained-BIC argmin lambda*_unconstr vs the final
# constrained-BIC lambda* (after the k>=p fallback), with the matching
# edge counts. Read directly from stage2_results.pkl: the 25-point BIC
# curve itself is not persisted, but the selected (lambda*, k) pair plus
# the fallback flag fully determine this selection diagnostic, so no
# per-lambda trace file is required. Mirrors tab:stage2-trajectory.
print("Building Fig. D.1 (Stage-2 lambda* selection diagnostic)...")
_labels = [s for s in SNAP_ORDER if s in stage2]
lam_unconstr = np.array([float(stage2[s]["lambda_unconstr"]) for s in _labels])
lam_opt = np.array([float(stage2[s]["lambda_opt"]) for s in _labels])
k_unconstr = np.array([float(stage2[s]["n_edges_unconstr"]) for s in _labels])
k_opt = np.array([float(stage2[s]["n_edges"]) for s in _labels])
fb = np.array([bool(stage2[s].get("fallback_fired", False)) for s in _labels])
p_vals = np.array([len(stage2[s]["tickers"]) for s in _labels])

xs = np.arange(len(_labels))
w = 0.38
_c_unc = "#9AA0A6"
_c_fin = REGIME_COLORS["recovery"]
fig, (ax_l, ax_k) = plt.subplots(
    2, 1, figsize=(7.2, 5.6), sharex=True,
    gridspec_kw={"hspace": 0.10}
)

# --- Top: unconstrained-BIC argmin vs final (constrained) lambda* ---
ax_l.bar(xs - w / 2, lam_unconstr, width=w, color=_c_unc,
         edgecolor="#222222", linewidth=0.5,
         label=r"$\lambda^\star$ unconstrained (BIC argmin)")
ax_l.bar(xs + w / 2, lam_opt, width=w, color=_c_fin,
         edgecolor="#222222", linewidth=0.5,
         label=r"$\lambda^\star$ final (post-fallback)")
for i in range(len(_labels)):
    if fb[i]:
        ax_l.scatter([xs[i] + w / 2], [lam_opt[i] + 0.05], marker=r"$\dagger$",
                     s=55, color="#8C2D2D", zorder=5)
ax_l.axhline(1.0, color="gray", linewidth=0.6, linestyle=":")
ax_l.set_ylabel(r"$\lambda^\star$", fontweight="bold")
ax_l.set_ylim(0, 1.20)
ax_l.grid(axis="y", linestyle=":", alpha=0.4)
ax_l.legend(loc="upper center", fontsize=7, ncol=2, frameon=True)
ax_l.set_title("Stage-2 selection: unconstrained vs. constrained "
               r"$\lambda^\star$  ($\dagger$ = $k\geq p$ fallback fired)",
               fontweight="bold")

# --- Bottom: edge counts k at each argmin (log) ---
# The unconstrained argmin is the empty graph (k=0) on every fallback
# row; plot 0 as 0.5 so the zero bars stay visible on the log axis.
ku = np.where(k_unconstr > 0, k_unconstr, 0.5)
ax_k.bar(xs - w / 2, ku, width=w, color=_c_unc,
         edgecolor="#222222", linewidth=0.5,
         label=r"$k(\lambda^\star)$ unconstrained")
ax_k.bar(xs + w / 2, k_opt, width=w, color=_c_fin,
         edgecolor="#222222", linewidth=0.5,
         label=r"$k(\lambda^\star)$ final")
ax_k.axhspan(p_vals.min(), p_vals.max(), color="#8C2D2D", alpha=0.10, zorder=0)
ax_k.axhline(p_vals.min(), color="#8C2D2D", linestyle="--", linewidth=0.7,
             label=r"$k_{\min}\approx p$ floor (%d--%d)"
                   % (int(p_vals.min()), int(p_vals.max())))
ax_k.set_yscale("log")
ax_k.set_ylabel(r"edges $k(\lambda^\star)$", fontweight="bold")
ax_k.grid(axis="y", linestyle=":", alpha=0.4)
ax_k.legend(loc="upper left", fontsize=7, ncol=2, frameon=True)
ax_k.set_xticks(xs)
ax_k.set_xticklabels([SHORT_LABEL[s] for s in _labels], rotation=40,
                     ha="right", fontsize=7)

plt.tight_layout()
fig.savefig(FIG_DIR / "fig10_stage2_bic_trajectory.pdf")
fig.savefig(FIG_DIR / "fig10_stage2_bic_trajectory.png", dpi=300)
print(f"  Saved fig10_stage2_bic_trajectory.pdf ({len(_labels)} snapshots, "
      f"{int(fb.sum())} fallback-firing)")


print(f"\n[DONE] All figures saved to {FIG_DIR}")
for f in ["fig1_impact_map", "fig2_success_criteria", "fig3_motif_profile",
          "fig4_correlation_heatmap", "fig5_nsi_vix_overlay",
          "fig5_pagerank_leaders", "fig6_robustness",
          "fig9_multipanel_nsi_heatmap", "fig10_stage2_bic_trajectory"]:
    print(f"  {f}.pdf")
