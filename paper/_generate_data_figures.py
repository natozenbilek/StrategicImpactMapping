"""Appendix Section 3 data-construction figures.

Builds three vector PDFs in ``paper/figures``:
  - fig_app_snapshot_timeline.pdf  (snapshot windows on VIX backdrop)
  - fig_app_gics_bias.pdf          (panel-vs-2008 GICS pair bar)
  - fig_app_contamination.pdf      (CFC/SBNY/SW/AMCR signature 2x2)

Reads only data/ (no results/). The contamination panels do a fresh
yfinance auto_adjust=False pull that is cached to
data/contamination_raw.parquet (~30s first call).
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import SNAPSHOTS

DATA_DIR = ROOT / "data"
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Imperial regime palette (matched to generate_figures.py).
REGIME_COLORS = {
    "crisis":   "#7B1E2B",   # Romanov claret bordo
    "stress":   "#C9A227",   # imperial gold
    "recovery": "#1E5945",   # Romanov emerald
    "baseline": "#1F2A44",   # Petersburg navy
}
IMPERIAL_IVORY    = "#F5E9D7"
IMPERIAL_GOLD_LT  = "#E0B84C"
IMPERIAL_GRAPHITE = "#2A2E33"

SECTOR_COLORS = {
    "Information Technology":  "#1F2A44",   # Petersburg navy
    "Financials":              "#C9A227",   # imperial gold
    "Health Care":             "#2D5A3D",   # forest emerald
    "Consumer Discretionary":  "#A33B30",   # terracotta
    "Industrials":             "#5B3A6E",   # royal purple
    "Communication Services":  "#0E5B6E",   # peacock teal
    "Consumer Staples":        "#8A2444",   # claret-rose
    "Energy":                  "#6B4423",   # russet
    "Utilities":               "#D8893A",   # warm amber
    "Materials":               "#52606D",   # graphite
    "Real Estate":             "#B45F23",   # rust
}

GICS_SECTOR_ORDER = [
    "Energy", "Materials", "Industrials", "Consumer Discretionary",
    "Consumer Staples", "Health Care", "Financials",
    "Information Technology", "Communication Services",
    "Utilities", "Real Estate",
]

MSCI_2008_WEIGHTS = {
    "Energy": 13.0,
    "Materials": 3.3,
    "Industrials": 11.0,
    "Consumer Discretionary": 8.3,
    "Consumer Staples": 12.8,
    "Health Care": 14.7,
    "Financials": 13.3,
    "Information Technology": 15.4,
    "Communication Services": 3.4,
    "Utilities": 4.2,
    "Real Estate": 1.2,
}

WIKIPEDIA_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
WIKIPEDIA_CACHE = DATA_DIR / "wikipedia_constituents.parquet"
CONTAMINATION_DROPS = ["CFC", "SBNY"]

CONTAMINATION_TICKERS = ["CFC", "SBNY", "SW", "AMCR"]
CONTAMINATION_META = {
    "CFC":  {"type": "merger → ticker re-use",   "cutoff": "2008-07 BoA acquisition"},
    "SBNY": {"type": "failure → ticker re-use",  "cutoff": "2023-03-12 FDIC seizure"},
    "SW":   {"type": "pre-listing placeholder",       "cutoff": "post-2024-07-15 only"},
    "AMCR": {"type": "pre-listing placeholder",       "cutoff": "post-2019-06-11 only"},
}
CONTAMINATION_CACHE = DATA_DIR / "contamination_raw.parquet"


def _assert(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _max_flatline_run(series):
    """Longest run of consecutive unchanged-price days in a Close series."""
    diff = series.diff().abs()
    flat = (diff < 1e-12).astype(int).fillna(0).to_numpy()
    max_run = cur = 0
    for v in flat:
        if v:
            cur += 1
            if cur > max_run:
                max_run = cur
        else:
            cur = 0
    return int(max_run)


def _zero_return_fraction(close):
    """Fraction of trading days with |log-return| < 1e-12 over the realised sample."""
    log_ret = np.log(close / close.shift(1)).dropna()
    if len(log_ret) == 0:
        return float("nan")
    return float((log_ret.abs() < 1e-12).sum() / len(log_ret))


def _stagger_rows(snapshots):
    """Greedy row assignment so overlapping bands sit on different rows."""
    parsed = []
    for label, start, end, regime in snapshots:
        parsed.append((label, pd.Timestamp(start), pd.Timestamp(end), regime))
    rows = []
    assignment = []
    for label, start, end, regime in parsed:
        placed = False
        for i, row_end in enumerate(rows):
            if start > row_end:
                assignment.append(i)
                rows[i] = end
                placed = True
                break
        if not placed:
            assignment.append(len(rows))
            rows.append(end)
    _assert(len(assignment) == len(snapshots), "row assignment length mismatch")
    return assignment, parsed


def figure_timeline(out_path):
    """Snapshot windows on a VIX backdrop, 2004-2025."""
    vix = pd.read_parquet(DATA_DIR / "vix.parquet")
    _assert("Close" in vix.columns, "vix.parquet missing 'Close' column")
    _assert(len(vix) > 5000, f"vix.parquet rows={len(vix)} unexpectedly small")
    vix = vix.sort_index()
    _assert(vix.index.min() <= pd.Timestamp("2004-12-31"), "vix start later than expected")
    _assert(vix.index.max() >= pd.Timestamp("2025-01-01"), "vix end earlier than expected")
    _assert(float(vix["Close"].min()) > 5.0 and float(vix["Close"].max()) < 100.0,
            "vix Close out of [5,100] sanity range")

    row_idx, parsed = _stagger_rows(SNAPSHOTS)
    n_rows = max(row_idx) + 1
    _assert(n_rows in (1, 2, 3), f"unexpected row count {n_rows}")

    fig, ax = plt.subplots(figsize=(10.5, 3.8))
    ax_vix = ax.twinx()

    ax_vix.plot(vix.index, vix["Close"], color="#888888", linewidth=0.55,
                alpha=0.85, zorder=1)
    ax_vix.set_ylabel("VIX (daily close)", fontsize=9, color="#555555")
    ax_vix.tick_params(axis="y", colors="#555555", labelsize=7)
    ax_vix.set_ylim(0, max(85, float(vix["Close"].max()) * 1.05))
    ax_vix.spines["top"].set_visible(False)

    band_height = 0.55 / n_rows
    band_gap = 0.04 / max(1, n_rows - 1) if n_rows > 1 else 0.0

    # Label positions sit on alternating high/low slots above the bands
    # with a connector line so adjacent windows (Jan/Mar/Jun 2020) read
    # without overlap. The connector also makes the figure scan from
    # band -> label without ambiguity.
    label_slots = [0.92, 0.81, 0.97, 0.86]   # cyclic vertical slots
    sorted_idx = sorted(range(len(parsed)),
                        key=lambda i: parsed[i][1])  # by start date
    slot_of = {}
    for k, i in enumerate(sorted_idx):
        slot_of[i] = label_slots[k % len(label_slots)]

    for idx, ((label, start, end, regime), row) in enumerate(zip(parsed, row_idx)):
        y0 = 0.10 + row * (band_height + band_gap)
        rect = mpatches.Rectangle(
            (mdates.date2num(start), y0),
            mdates.date2num(end) - mdates.date2num(start),
            band_height,
            facecolor=REGIME_COLORS[regime], edgecolor=IMPERIAL_GRAPHITE,
            linewidth=0.5, alpha=0.72, zorder=3,
        )
        ax.add_patch(rect)
        mid = start + (end - start) / 2
        y_lbl = slot_of[idx]
        # Faint connector from band top to label baseline.
        ax.plot([mid, mid], [y0 + band_height, y_lbl - 0.012],
                color=IMPERIAL_GRAPHITE, linewidth=0.35,
                linestyle=":", alpha=0.6, zorder=2)
        ax.text(mid, y_lbl, label,
                ha="center", va="bottom", fontsize=6.5,
                color=IMPERIAL_GRAPHITE, fontweight="bold",
                rotation=0, zorder=4,
                bbox=dict(boxstyle="round,pad=0.18", facecolor=IMPERIAL_IVORY,
                          edgecolor="none", alpha=0.85))

    ax.set_xlim(pd.Timestamp("2004-01-01"), pd.Timestamp("2026-03-01"))
    ax.set_ylim(0, 1.0)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", labelsize=8)
    ax.set_xlabel("Calendar year", fontsize=10)

    legend_patches = [
        mpatches.Patch(color=REGIME_COLORS[r], label=r.capitalize(), alpha=0.75)
        for r in ["crisis", "stress", "recovery", "baseline"]
    ]
    # Legend pushed below the x-axis so the label-box slots above the
    # bands stay readable; the earlier top-left position collided with
    # the 2008-2009 cluster of labels.
    ax.legend(handles=legend_patches, loc="upper center",
              bbox_to_anchor=(0.5, -0.13), ncol=4, frameon=False,
              fontsize=8.0, handlelength=1.3, columnspacing=1.2)

    durations_days = np.array([
        (pd.Timestamp(end) - pd.Timestamp(start)).days
        for _, start, end, _ in SNAPSHOTS
    ])
    ratio = durations_days.max() / durations_days.min()
    _assert(4.0 < ratio < 6.0, f"expected ~4.8x ratio, got {ratio:.2f}")

    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    _assert(out_path.exists() and out_path.stat().st_size > 5_000,
            f"{out_path} not written or too small")


def _fetch_wikipedia_constituents():
    """Live S&P 500 constituent fetch via bs4 (lxml-free); cached for reuse."""
    if WIKIPEDIA_CACHE.exists():
        return pd.read_parquet(WIKIPEDIA_CACHE)
    import requests
    from bs4 import BeautifulSoup

    resp = requests.get(WIKIPEDIA_SP500_URL,
                        headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    table = (soup.find("table", {"id": "constituents"})
             or soup.find("table", class_="wikitable"))
    _assert(table is not None, "Wikipedia constituents table not found")

    hdr_cells = table.find("tr").find_all("th")
    hdrs = [th.get_text(strip=True) for th in hdr_cells]
    sec_idx = next((i for i, h in enumerate(hdrs)
                    if h.replace(" ", "").lower() == "gicssector"), None)
    sym_idx = next((i for i, h in enumerate(hdrs)
                    if h.lower() == "symbol"), 0)
    _assert(sec_idx is not None, f"GICS Sector column not in headers: {hdrs}")

    rows = []
    for tr in table.find_all("tr")[1:]:
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(cells) > max(sec_idx, sym_idx):
            rows.append((cells[sym_idx], cells[sec_idx]))
    df = pd.DataFrame(rows, columns=["Symbol", "GICS Sector"])
    df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
    df.to_parquet(WIKIPEDIA_CACHE)
    return df


def _load_panel_universe():
    """Live Wikipedia constituents (panel proxy ~501-503); parquet fallback."""
    try:
        df = _fetch_wikipedia_constituents()
        df = df[~df["Symbol"].isin(CONTAMINATION_DROPS)].reset_index(drop=True)
        if len(df) > 400:
            return df, "Wikipedia constituents"
    except Exception as e:
        print(f"  Wikipedia fetch failed ({e}); falling back to sp500_info.parquet")
    df = pd.read_parquet(DATA_DIR / "sp500_info.parquet")
    return df, "offline parquet"


def figure_gics_bias(out_path):
    """Panel-count-share vs 2008 MSCI cap-share pair bars per GICS sector."""
    info, source = _load_panel_universe()
    _assert("GICS Sector" in info.columns, f"{source}: missing GICS Sector")
    _assert(len(info) > 50, f"{source}: only {len(info)} rows")
    panel_n = len(info)

    counts = info["GICS Sector"].value_counts()
    panel_share = {s: 100.0 * counts.get(s, 0) / panel_n for s in GICS_SECTOR_ORDER}
    _assert(abs(sum(panel_share.values()) - 100.0) < 1e-6,
            f"panel shares sum {sum(panel_share.values())} != 100")
    _assert(abs(sum(MSCI_2008_WEIGHTS.values()) - 100.0) < 1.0,
            "MSCI 2008 weights do not sum to ~100")

    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    x = np.arange(len(GICS_SECTOR_ORDER))
    width = 0.4

    panel_vals = [panel_share[s] for s in GICS_SECTOR_ORDER]
    msci_vals = [MSCI_2008_WEIGHTS[s] for s in GICS_SECTOR_ORDER]

    bars_panel = ax.bar(
        x - width / 2, panel_vals, width,
        color=[SECTOR_COLORS[s] for s in GICS_SECTOR_ORDER],
        edgecolor="black", linewidth=0.4, alpha=0.85,
        label=f"Panel count-share ($N={panel_n}$, {source})",
    )
    bars_msci = ax.bar(
        x + width / 2, msci_vals, width,
        color=[SECTOR_COLORS[s] for s in GICS_SECTOR_ORDER],
        edgecolor="black", linewidth=0.4, alpha=0.42, hatch="///",
        label="MSCI 2008 market-cap share",
    )

    for i, sector in enumerate(GICS_SECTOR_ORDER):
        delta = panel_share[sector] - MSCI_2008_WEIGHTS[sector]
        color = "#1f7a1f" if delta > 0 else "#a01515"
        sign = "+" if delta > 0 else ""
        ymax = max(panel_share[sector], MSCI_2008_WEIGHTS[sector])
        ax.text(i, ymax + 0.6, f"{sign}{delta:.1f}pp",
                ha="center", va="bottom", fontsize=7.5,
                color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([s.replace(" ", "\n", 1) for s in GICS_SECTOR_ORDER],
                       rotation=0, fontsize=7.5)
    ax.set_ylabel("Share (%)", fontsize=10)
    ax.set_ylim(0, max(max(panel_vals), max(msci_vals)) * 1.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linewidth=0.4, alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False, fontsize=8.5)

    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    _assert(out_path.exists() and out_path.stat().st_size > 5_000,
            f"{out_path} not written or too small")


def _load_contamination_raw():
    """Fresh yfinance pull (auto_adjust=False) for the 4 contaminated symbols, cached."""
    if CONTAMINATION_CACHE.exists():
        cached = pd.read_parquet(CONTAMINATION_CACHE)
        have = set(cached.columns.get_level_values(0)) if isinstance(cached.columns, pd.MultiIndex) else set(cached.columns)
        if all(t in have for t in CONTAMINATION_TICKERS):
            return cached
    import yfinance as yf

    frames = {}
    for sym in CONTAMINATION_TICKERS:
        df = yf.download(
            sym, start="2004-01-01", end="2025-12-31",
            auto_adjust=False, progress=False, threads=False,
        )
        _assert(df is not None and len(df) > 0, f"yfinance returned empty for {sym}")
        close_col = df["Close"]
        if hasattr(close_col, "columns"):
            close_col = close_col.iloc[:, 0]
        frames[sym] = close_col.rename(sym)

    out = pd.concat(frames.values(), axis=1)
    out.columns = list(frames.keys())
    out.index.name = "Date"
    out.to_parquet(CONTAMINATION_CACHE)
    return out


def figure_contamination(out_path):
    """2x2 raw-close panels for CFC, SBNY, SW, AMCR with signature annotations."""
    raw = _load_contamination_raw()
    _assert(all(t in raw.columns for t in CONTAMINATION_TICKERS),
            f"contamination cache missing columns; got {list(raw.columns)}")

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.2))
    panel_letters = ["(a)", "(b)", "(c)", "(d)"]

    for ax, sym, letter in zip(axes.flat, CONTAMINATION_TICKERS, panel_letters):
        series = raw[sym].dropna()
        _assert(len(series) > 50, f"{sym}: only {len(series)} non-NaN rows")
        meta = CONTAMINATION_META[sym]

        ax.plot(series.index, series.values, color="#222222",
                linewidth=0.55, alpha=0.92)

        zfrac = _zero_return_fraction(series)
        max_run = _max_flatline_run(series)
        _assert(0.0 <= zfrac <= 1.0, f"{sym}: zero-fraction {zfrac} out of [0,1]")
        _assert(max_run >= 0 and max_run < len(series),
                f"{sym}: flatline run {max_run} out of range")

        annot = (
            f"{sym}\n"
            f"zero-return frac: {zfrac:.1%}\n"
            f"max flatline run: {max_run} d\n"
            f"signature: {meta['type']}\n"
            f"cutoff: {meta['cutoff']}"
        )
        ax.text(0.02, 0.96, annot, transform=ax.transAxes,
                ha="left", va="top", fontsize=7.3,
                family="serif",
                bbox=dict(boxstyle="round,pad=0.32",
                          facecolor="white", edgecolor="#888888",
                          linewidth=0.5, alpha=0.92))

        ax.set_title(f"{letter} {sym}: raw Yahoo close (auto_adjust=False)",
                     fontsize=9.5, loc="left")
        ax.set_ylabel("Close (USD)", fontsize=9)
        ax.tick_params(axis="both", labelsize=7.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, linewidth=0.3, alpha=0.3)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_locator(mdates.YearLocator(3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    _assert(out_path.exists() and out_path.stat().st_size > 5_000,
            f"{out_path} not written or too small")


FIGURES = {
    "timeline":      ("fig_app_snapshot_timeline.pdf",  figure_timeline),
    "gics":          ("fig_app_gics_bias.pdf",          figure_gics_bias),
    "contamination": ("fig_app_contamination.pdf",      figure_contamination),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", choices=list(FIGURES.keys()),
                        help="Render a single figure by key.")
    parser.add_argument("--all", action="store_true",
                        help="Render every figure.")
    args = parser.parse_args()

    if not args.all and args.figure is None:
        parser.error("pass --all or --figure {timeline,gics,contamination}")

    targets = list(FIGURES.keys()) if args.all else [args.figure]
    for key in targets:
        name, fn = FIGURES[key]
        out = FIG_DIR / name
        print(f"[{key}] -> {out}")
        fn(out)
        size_kb = out.stat().st_size / 1024
        print(f"        wrote {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
