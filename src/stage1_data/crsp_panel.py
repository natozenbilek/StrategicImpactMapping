"""CRSP-based Stage 1a panel: drop-in replacement for Yahoo download.

Pulls the consolidated CRSP DSF long parquet plus the constituents,
stocknames, and delisting metadata under data/crsp/ and assembles
(info, prices, returns) in the same schema run_download() returns.

Conventions
-----------
* prices: split-adjusted dollar close, prc / cfacpr * cfacpr_first, with
  cfacpr_first the panel-window first valid cfacpr per PERMNO. Dividends
  NOT folded in (this is the CRSP back-adjusted dollar series, not a
  total return index).
* returns: log(1 + ret), with ret the CRSP simple total-return series
  (split + dividend adjusted). Identical-by-design to log(1+Engle-style)
  daily total return.

The two series are therefore NOT linked by log(prices).diff() == returns;
the dividend yield is the gap. Downstream Stage 1 GARCH consumes returns
only; prices is for tables, dollar-axis plots, and volume normalization.

Delisting splice: when delisting.csv carries a non-null DLRET on a date
inside the panel window, the delisting return is appended on its
dlstdt; if DSF already has a return on that date, the two are combined
multiplicatively. WAMU (PERMNO 81593) has null DLRET because the FDIC
2008-09-26 seizure is recorded as a trading suspension, not a formal
delisting -- its bankruptcy-day return is implicit in the final DSF prc
trajectory ($46 -> $0.16); no splice is applied.

Ticker resolution: most-recent stocknames ticker per PERMNO, with manual
overrides for the eight distressed-2008 names so paper-facing labels
read LEH/BSC/WAMU rather than the post-reorg COOP etc. Cross-PERMNO
ticker collisions (81 of them, e.g. WM = Waste Management 2024 vs
Washington Mutual 2008) get a _PERMNO suffix.
"""
import datetime
import json
import numpy as np
import pandas as pd

from src.config import DATA_DIR

CRSP_DIR = DATA_DIR / "crsp"
DSF_PATH = CRSP_DIR / "dsf_long.parquet"
CONST_PATH = CRSP_DIR / "constituents.csv"
NAMES_PATH = CRSP_DIR / "stocknames.csv"
DELIST_PATH = CRSP_DIR / "delisting.csv"
CACHE_META_PATH = DATA_DIR / "cache_metadata.json"

# Paper-facing label overrides for the eight distressed-2008 PERMNOs.
# Latest stocknames ticker resolves to post-reorg names (e.g. WAMU
# PERMNO 81593 -> COOP, the 2018 Mr Cooper Group reorganization);
# the crisis-era labels are what every section of the manuscript uses.
TICKER_OVERRIDES = {
    80599: "LEH",
    68304: "BSC",
    52919: "MER",
    51043: "FNM",
    75789: "FRE",
    81593: "WAMU",
    66800: "AIG",
    25785: "F",
}

# Coarse SIC -> 11-sector GICS bucket. Exact GICS taxonomy lives in a
# crosswalk outside CRSP; this is sufficient for sector-purity and
# node-colour plots and aligns with the GICS_SECTORS labels in config.
_SIC_BANDS = [
    (1000, 1499, "Materials"),
    (1500, 1799, "Industrials"),
    (2000, 2199, "Consumer Staples"),
    (2200, 2399, "Consumer Discretionary"),
    (2400, 2599, "Industrials"),
    (2600, 2699, "Materials"),
    (2700, 2799, "Communication Services"),
    (2800, 2899, "Materials"),
    (2900, 2999, "Energy"),
    (3000, 3199, "Consumer Discretionary"),
    (3200, 3299, "Materials"),
    (3300, 3499, "Industrials"),
    (3500, 3599, "Industrials"),
    (3600, 3699, "Information Technology"),
    (3700, 3799, "Industrials"),
    (3800, 3899, "Health Care"),
    (3900, 3999, "Consumer Discretionary"),
    (4000, 4799, "Industrials"),
    (4800, 4899, "Communication Services"),
    (4900, 4999, "Utilities"),
    (5000, 5199, "Consumer Staples"),
    (5200, 5999, "Consumer Discretionary"),
    (6000, 6499, "Financials"),
    (6500, 6599, "Real Estate"),
    (6700, 6799, "Financials"),
    (7000, 7299, "Consumer Discretionary"),
    (7300, 7399, "Information Technology"),
    (7400, 7799, "Industrials"),
    (7800, 7999, "Communication Services"),
    (8000, 8099, "Health Care"),
    (8100, 8999, "Industrials"),
]


def _sic_to_gics(siccd) -> str:
    if pd.isna(siccd):
        return "Unknown"
    s = int(siccd)
    for lo, hi, label in _SIC_BANDS:
        if lo <= s <= hi:
            return label
    return "Unknown"


def _resolve_tickers(stocknames: pd.DataFrame) -> pd.DataFrame:
    """Build per-PERMNO (ticker, comnam, siccd) with overrides + collision suffix."""
    assert {'permno','ticker','comnam','nameenddt','siccd'} <= set(stocknames.columns)
    sn = stocknames.sort_values(['permno','nameenddt']).copy()
    latest = sn.drop_duplicates(subset='permno', keep='last')[
        ['permno','ticker','comnam','siccd']
    ].copy()
    latest['ticker'] = latest['ticker'].fillna('UNK')
    for permno, ticker in TICKER_OVERRIDES.items():
        latest.loc[latest['permno'] == permno, 'ticker'] = ticker
    counts = latest['ticker'].value_counts()
    collision = set(counts[counts > 1].index)
    latest['ticker'] = [
        f"{t}_{p}" if t in collision else t
        for t, p in zip(latest['ticker'], latest['permno'])
    ]
    assert latest['ticker'].is_unique, "ticker resolution failed: duplicates remain"
    return latest.reset_index(drop=True)


def _build_returns(dsf: pd.DataFrame, p2t: dict) -> pd.DataFrame:
    """Pivot CRSP `ret` to (date x ticker) and convert to log(1+ret)."""
    df = dsf[['permno','date','ret']].dropna(subset=['ret']).copy()
    df['log_ret'] = np.log1p(df['ret'])
    assert df['log_ret'].abs().max() < 5, (
        f"impossible log return magnitude: {df['log_ret'].abs().max():.3f}"
    )
    df['ticker'] = df['permno'].map(p2t)
    df = df.dropna(subset=['ticker'])
    wide = df.pivot(index='date', columns='ticker', values='log_ret').sort_index()
    wide.index.name = None
    return wide


def _build_prices(dsf: pd.DataFrame, p2t: dict) -> pd.DataFrame:
    """Split-adjusted dollar close: prc / cfacpr * cfacpr_first per PERMNO."""
    df = dsf[['permno','date','prc','cfacpr']].dropna(subset=['prc','cfacpr']).copy()
    # CRSP encodes bid-ask midpoint as a negative prc; treat as positive.
    df['prc'] = df['prc'].abs()
    df = df[df['cfacpr'] > 0]
    df = df.sort_values(['permno','date'])
    first_cfac = df.groupby('permno')['cfacpr'].transform('first')
    df['adj'] = df['prc'] / df['cfacpr'] * first_cfac
    assert (df['adj'] > 0).all(), "non-positive adjusted price after splice"
    df['ticker'] = df['permno'].map(p2t)
    df = df.dropna(subset=['ticker'])
    wide = df.pivot(index='date', columns='ticker', values='adj').sort_index()
    wide.index.name = None
    return wide


def _splice_delisting(returns: pd.DataFrame, delisting: pd.DataFrame,
                      p2t: dict, start_ts, end_ts) -> int:
    """Append non-null DLRET on delisting day; combine multiplicatively if needed.
    Returns number of splices applied (in place)."""
    dl = delisting.dropna(subset=['dlret']).copy()
    dl['dlret'] = pd.to_numeric(dl['dlret'], errors='coerce')
    dl = dl.dropna(subset=['dlret'])
    dl['ticker'] = dl['permno'].map(p2t)
    dl = dl.dropna(subset=['ticker'])
    n = 0
    for _, r in dl.iterrows():
        d, t, val = r['dlstdt'], r['ticker'], float(r['dlret'])
        if d < start_ts or d > end_ts or t not in returns.columns:
            continue
        if d not in returns.index:
            cand = returns.index[returns.index <= d]
            if len(cand) == 0:
                continue
            d = cand[-1]
        existing = returns.at[d, t]
        # DLRET == -1 (total bankruptcy wipeout, dlstcd 574) clipped to
        # -0.9999 so log1p stays finite; the resulting -9.21 still
        # encodes a ~99.99% loss and downstream GARCH handles it.
        if pd.isna(existing):
            v = max(val, -0.9999)
            returns.at[d, t] = np.log1p(v)
        else:
            combined = max(np.exp(existing) * (1 + val) - 1, -0.9999)
            returns.at[d, t] = np.log1p(combined)
        n += 1
    return n


def build_panel(start: str = "1985-01-01", end: str = "2024-12-31"):
    """Return (info, prices, returns) for the CRSP DSF window [start, end]."""
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)

    print("[CRSP Stage 1a] loading DSF + metadata...")
    dsf = pd.read_parquet(DSF_PATH)
    dsf['date'] = pd.to_datetime(dsf['date'])
    stocknames = pd.read_csv(NAMES_PATH, parse_dates=['namedt','nameenddt'])
    delisting = pd.read_csv(DELIST_PATH, parse_dates=['dlstdt'])

    dsf_w = dsf[(dsf['date'] >= start_ts) & (dsf['date'] <= end_ts)].copy()
    print(f"  DSF rows in window {start}..{end}: {len(dsf_w):,}")
    assert len(dsf_w) > 0, "empty DSF window"

    tmap = _resolve_tickers(stocknames)
    p2t = dict(zip(tmap['permno'], tmap['ticker']))
    print(f"  PERMNO -> ticker entries: {len(tmap):,}")

    returns = _build_returns(dsf_w, p2t)
    print(f"  returns shape: {returns.shape}")

    n_splice = _splice_delisting(returns, delisting, p2t, start_ts, end_ts)
    print(f"  delisting splices applied: {n_splice}")

    prices = _build_prices(dsf_w, p2t)
    print(f"  prices shape:  {prices.shape}")

    # Align prices and returns on a common ticker set so downstream
    # callers can assume returns.columns == prices.columns.
    common = sorted(set(prices.columns) & set(returns.columns))
    prices = prices[common]
    returns = returns[common]
    assert list(prices.columns) == list(returns.columns)

    info = tmap.rename(columns={'comnam': 'Security', 'ticker': 'Symbol'}).copy()
    info['GICS Sector'] = info['siccd'].apply(_sic_to_gics)
    info['GICS Sub-Industry'] = info['GICS Sector']
    info['PERMNO'] = info['permno']
    info = info[['Symbol','Security','GICS Sector','GICS Sub-Industry','PERMNO']]
    info = info[info['Symbol'].isin(set(common))].reset_index(drop=True)
    assert len(info) == len(common), (
        f"info / panel cardinality mismatch: {len(info)} vs {len(common)}"
    )
    print(f"  info rows: {len(info)}")

    return info, prices, returns


def _write_cache_metadata(n_tickers: int):
    payload = {
        "fetched_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "n_tickers": int(n_tickers),
        "source": "CRSP DSF (data/crsp/dsf_long.parquet)",
    }
    CACHE_META_PATH.write_text(json.dumps(payload, indent=2))


def _read_cache_metadata():
    if not CACHE_META_PATH.exists():
        return None
    try:
        return json.loads(CACHE_META_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def run_download_crsp(force: bool = False):
    """Drop-in replacement for run_download(): cache (info, prices, returns)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    prices_path = DATA_DIR / "sp500_prices.parquet"
    returns_path = DATA_DIR / "sp500_returns.parquet"
    info_path = DATA_DIR / "sp500_info.parquet"

    if (not force and prices_path.exists()
            and returns_path.exists() and info_path.exists()):
        print("[CRSP Stage 1a] loading cached panel...")
        info = pd.read_parquet(info_path)
        prices = pd.read_parquet(prices_path)
        returns = pd.read_parquet(returns_path)
        meta = _read_cache_metadata()
        fetched = meta["fetched_at"] if meta and "fetched_at" in meta else "unknown"
        src = meta.get("source", "unknown") if meta else "unknown"
        print(f"  prices {prices.shape}, returns {returns.shape}, info {len(info)}")
        print(f"  fetched_at: {fetched}   source: {src}")
        return info, prices, returns

    info, prices, returns = build_panel()
    # Downcast pandas nullable Float64 (capital F) to numpy float64 so
    # downstream callers like np.linalg.lstsq (Stage 3 lagged_partial_
    # correlation) see a native numeric array, not dtype('O'). CRSP DSF
    # parquet loads as nullable by default; the cast is one extra pass
    # over the panel at write time.
    prices = prices.astype(float)
    returns = returns.astype(float)
    prices.to_parquet(prices_path)
    returns.to_parquet(returns_path)
    info.to_parquet(info_path)
    _write_cache_metadata(len(info))
    print(f"[CRSP Stage 1a] cached to {DATA_DIR}")
    return info, prices, returns


if __name__ == "__main__":
    info, prices, returns = run_download_crsp(force=True)
    print(f"\nSummary:")
    print(f"  Tickers     : {len(info)}")
    print(f"  Date range  : {prices.index[0].date()} -> {prices.index[-1].date()}")
    print(f"  Prices shape: {prices.shape}")
    print(f"  Returns shp : {returns.shape}")
    print(f"\nSector distribution:")
    print(info["GICS Sector"].value_counts().to_string())
