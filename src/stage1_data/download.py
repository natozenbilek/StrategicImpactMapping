"""Stage 1a: S&P 500 download and log-return panel.

Pulls adjusted close prices for current S&P 500 constituents plus a
curated list of crisis-era delisted names (LEH, BSC, WAMU, ...). The
panel is left unbalanced by design: a global dropna would erase every
date prior to the most recent IPO and purge every delisted ticker,
making the 2008 cross-section invisible at exactly the windows where
those tickers matter most. Per-asset GARCH and per-snapshot A-DCC
handle the resulting NaN gaps without a globally balanced grid.
"""
import datetime
import json
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
import warnings

from src.config import DATA_DIR, DATA_START, DATA_END

CACHE_META_PATH = DATA_DIR / "cache_metadata.json"

warnings.filterwarnings("ignore", category=FutureWarning)


# Crisis-era distressed / near-bankruptcy tickers probed alongside the
# live S&P 500 list. Each name is included so that whatever Yahoo does
# return for the symbol enters the panel; the coverage filter in
# clean_prices then admits the ones with enough data and silently drops
# the rest. Per-name outcomes on the current Yahoo backend:
#
#   * Absent (Yahoo returns no adjusted-close series): LEH, BSC, WAMU,
#     MER, FNM, FRE (2008 failures); SIVB, FRC (2023 failures).
#   * Partial: GM — only post-2010 "new GM" survives; the pre-2009
#     bankruptcy series is absent.
#   * Continuous: AIG (survived the 2008 government bailout), F (Ford
#     never filed). Both are kept in this list rather than relying on
#     the Wikipedia constituent fetch because AIG was removed from the
#     index in 2008 and re-added later, and the crisis-era information
#     content of both names matters for the 2008 cross-section.
#
# CFC and SBNY are deliberately *omitted* from this list: the Yahoo
# series returned under each symbol is contaminated by post-event
# ticker reassignment (CFC post-2008-07 is a separate small-cap; SBNY
# post-2024-08 is a sub-$3 penny stock unrelated to Signature Bank).
# See app:contamination in the appendix for the diagnostic detail.
HISTORICAL_TICKERS = [
    ("LEH",  "Lehman Brothers",         "Financials"),
    ("BSC",  "Bear Stearns",            "Financials"),
    ("WAMU", "Washington Mutual",       "Financials"),
    ("MER",  "Merrill Lynch",           "Financials"),
    ("FNM",  "Fannie Mae",              "Financials"),
    ("FRE",  "Freddie Mac",             "Financials"),
    ("AIG",  "AIG",                     "Financials"),
    ("GM",   "General Motors",          "Consumer Discretionary"),
    ("F",    "Ford Motor",              "Consumer Discretionary"),
    ("SIVB", "SVB Financial",           "Financials"),
    ("FRC",  "First Republic",          "Financials"),
]

# Manual truncation cutoffs for tickers where Yahoo's symbol-level
# history mixes two unrelated issuers across a known re-use boundary.
# The dict maps ticker -> earliest valid date (inclusive); all prior
# observations are masked to NaN before downstream stages see them.
# SW = Smurfit Westrock listed 2024-07-15 (Smurfit Kappa + WestRock
# merger); pre-merger SW data on Yahoo is a flat $6 series for an
# unrelated issuer. AMCR = Amcor PLC listed 2019-06-11 (Bemis spin);
# pre-spin AMCR is a flat ~$28-43 series for an earlier unrelated
# listing under the same symbol.
TICKER_CUTOFFS = {
    "SW":   "2024-07-15",
    "AMCR": "2019-06-11",
}


def get_sp500_tickers():
    """Current S&P 500 constituents from Wikipedia. Raises on any failure
    (network outage, page-format change, suspiciously small response).

    A silent fallback to a hand-maintained mega-cap list would let the
    pipeline run on a panel an order of magnitude smaller than the one
    documented in the paper; we prefer a loud abort so the operator
    notices and either restores connectivity or rebuilds the panel
    deliberately.
    """
    import requests
    from io import StringIO
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    resp = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                        headers=headers, timeout=15)
    resp.raise_for_status()
    df = pd.read_html(StringIO(resp.text))[0]
    df = df[["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"]].copy()
    df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
    if not 480 <= len(df) <= 520:
        raise RuntimeError(
            f"Wikipedia parse returned {len(df)} S&P 500 symbols, outside "
            f"the sanity band [480, 520]. Page format likely changed; "
            f"inspect https://en.wikipedia.org/wiki/List_of_S%26P_500_companies "
            f"and update get_sp500_tickers if the first wikitable is no "
            f"longer the constituent list."
        )
    print(f"  Fetched {len(df)} tickers from Wikipedia")
    return df


def download_prices(tickers, start=DATA_START, end=DATA_END, batch_size=50,
                    max_failure_fraction=0.20, retry_singleton=True):
    """Adjusted-close download in 50-ticker batches (Yahoo URL length limit).

    A batch that raises is logged and skipped; the function aborts if
    more than ``max_failure_fraction`` of batches fail, so a transient
    Yahoo outage that silently delivers a tiny panel cannot pass
    through to downstream stages.

    When ``retry_singleton=True`` any ticker that comes back from a
    batch download as an all-NaN column (a yfinance-level within-batch
    failure mode that does not surface as an exception) is retried as a
    single-ticker call. This guards against the v1.2.x regression where
    batches of $\\geq 50$ tickers sometimes return a partial DataFrame
    with most columns silently empty. The singleton retry adds at most
    a few seconds per failed ticker and recovers in practice.
    """
    all_prices = []
    n_batches = (len(tickers) - 1) // batch_size + 1
    failed_batches = []
    singleton_recovered = 0
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        print(f"  Downloading batch {i // batch_size + 1}/{n_batches} "
              f"({len(batch)} tickers)...")
        try:
            data = yf.download(" ".join(batch), start=start, end=end,
                               auto_adjust=True, progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                prices = data["Close"]
            else:
                prices = data[["Close"]]
                prices.columns = batch
            if retry_singleton:
                empty_cols = [c for c in batch
                              if c in prices.columns
                              and prices[c].isna().all()]
                for ticker in empty_cols:
                    try:
                        df1 = yf.download(ticker, start=start, end=end,
                                          auto_adjust=True, progress=False)
                        if isinstance(df1.columns, pd.MultiIndex):
                            col = df1[("Close", ticker)]
                        else:
                            col = df1["Close"]
                        if not col.isna().all():
                            prices[ticker] = col.reindex(prices.index)
                            singleton_recovered += 1
                    except Exception:
                        continue
            all_prices.append(prices)
        except Exception as e:
            print(f"  WARNING: Batch failed: {e}")
            failed_batches.append((i // batch_size + 1, str(e)[:120]))
            continue

    if not all_prices:
        raise RuntimeError("No data downloaded. Check internet connection.")

    failure_fraction = len(failed_batches) / n_batches
    if failed_batches:
        print(f"  Batch failure rate: {len(failed_batches)}/{n_batches} "
              f"({failure_fraction:.1%})")
    if singleton_recovered:
        print(f"  Singleton-retry recovered {singleton_recovered} tickers "
              f"that came back empty from their batch.")
    if failure_fraction > max_failure_fraction:
        details = "\n".join(f"    batch {b}: {msg}" for b, msg in failed_batches)
        raise RuntimeError(
            f"Yahoo download batch-failure rate {failure_fraction:.1%} "
            f"exceeds the {max_failure_fraction:.0%} guard. Refusing to "
            f"continue with a partial panel:\n{details}"
        )
    return pd.concat(all_prices, axis=1)


def _apply_ticker_cutoffs(prices, cutoffs=TICKER_CUTOFFS):
    """Mask observations before a known ticker-reassignment date as NaN.

    Some Yahoo symbols carry two separate issuers stitched together
    across a merger/spin-off. Truncating to the post-event regime
    keeps the legitimate series and drops the phantom history.
    """
    out = prices.copy()
    index_tz = getattr(out.index, "tz", None)
    for t, cutoff in cutoffs.items():
        if t not in out.columns:
            continue
        cutoff_ts = pd.Timestamp(cutoff)
        if index_tz is not None and cutoff_ts.tz is None:
            cutoff_ts = cutoff_ts.tz_localize(index_tz)
        mask = out.index < cutoff_ts
        n_masked = int(mask.sum())
        out.loc[mask, t] = np.nan
        print(f"  Truncated {t}: masked {n_masked} pre-{cutoff} obs "
              f"(ticker re-use boundary)")
    return out


def _detect_contaminated(returns, max_zero_frac=0.20, max_flatline=20):
    """Return list of tickers whose return series looks like a re-used
    symbol: dominated by zero-returns or by long unchanged-price runs.

    A genuine S&P 500 stock has < 5% zero-return days. CFC, SW, AMCR,
    SBNY all sit above 30%. The flatline check catches series held at
    a single price for weeks at a time (typical of a delisted symbol
    that Yahoo backfilled with a vendor placeholder).

    Returns a list of (ticker, zero_fraction, max_consecutive_zero_run)
    tuples for every ticker that exceeds either threshold. An empty
    list means the panel is clean under this detector.
    """
    bad = []
    for t in returns.columns:
        r = returns[t].dropna()
        if len(r) < 20:
            continue
        zero_frac = float((r == 0).sum()) / len(r)
        max_run = cur = 0
        for v in (r == 0):
            cur = cur + 1 if v else 0
            if cur > max_run:
                max_run = cur
        if zero_frac > max_zero_frac or max_run > max_flatline:
            bad.append((t, zero_frac, max_run))
    return bad


def clean_prices(prices, min_obs_frac=0.05):
    """Drop low-coverage tickers, mask ticker-reuse cutoffs, ffill
    <=5-day gaps; preserve unbalanced grid.

    A previous version followed the ffill with a global dropna, which
    erased every date prior to the latest IPO and purged every delisted
    ticker. The current logic keeps the grid unbalanced and lets the
    per-snapshot routines handle window-level alignment.

    Ticker-reuse cutoffs (TICKER_CUTOFFS) are applied before coverage
    filtering so a ticker like SW (real post-2024-07-15 only) is judged
    on its legitimate observations only.
    """
    prices = _apply_ticker_cutoffs(prices)

    n_days = len(prices)
    min_obs = int(n_days * min_obs_frac)
    valid_counts = prices.count()
    valid_tickers = valid_counts[valid_counts >= min_obs].index
    dropped = set(prices.columns) - set(valid_tickers)
    prices = prices[valid_tickers].copy()
    print(f"  Retained {len(valid_tickers)} tickers with >= {min_obs} obs "
          f"({min_obs_frac:.0%} of sample)")
    if dropped:
        print(f"  Dropped {len(dropped)} low-coverage tickers "
              f"(e.g. {sorted(dropped)[:5]})")

    # Bound ffill to 5 days so genuine multi-week delistings stay NaN.
    prices = prices.ffill(limit=5)

    first_valid = prices.apply(lambda c: c.first_valid_index()).min()
    last_valid = prices.apply(lambda c: c.last_valid_index()).max()
    print(f"  Panel spans: {first_valid.date()} -> {last_valid.date()} "
          f"({prices.shape[0]} days x {prices.shape[1]} tickers, unbalanced)")
    return prices


def compute_log_returns(prices):
    """r_{i,t} = log P_{i,t} - log P_{i,t-1}; drop first row + all-NaN rows."""
    returns = np.log(prices).diff().iloc[1:]
    return returns.dropna(how="all")


def _write_cache_metadata(n_tickers):
    """Record the fresh-pull timestamp in a sidecar JSON. Parquet `attrs`
    do not survive the to_parquet round-trip; a sidecar file is the
    minimum-fuss way to make cache staleness visible to operators
    (the CVNA split miss the audit caught was on a cache whose pull
    date was no longer recoverable from the parquet mtime).
    """
    payload = {
        "fetched_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "n_tickers": int(n_tickers),
    }
    CACHE_META_PATH.write_text(json.dumps(payload, indent=2))


def _read_cache_metadata():
    if not CACHE_META_PATH.exists():
        return None
    try:
        return json.loads(CACHE_META_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _merge_augmented_panel(info, prices):
    """Append the three free-public-sourced 2008-era distressed names
    (MER from Wayback, FNMA + FMCC from live Yahoo) to the panel.
    Returns (info, prices, returns) with three extra columns each.

    The augmented parquets are assembled by tools/build_augmented_panel.py
    from tools/distressed_2008_panel/*.csv; in-memory only, the on-disk
    baseline sp500_prices.parquet is left untouched.
    """
    aug_path = DATA_DIR / "sp500_distressed_augmented.parquet"
    assert aug_path.exists(), (
        f"augmented panel not built: {aug_path} missing. "
        f"Run tools/build_augmented_panel.py first.")
    aug = pd.read_parquet(aug_path)
    expected_cols = {"MER", "FNMA", "FMCC"}
    assert set(aug.columns) == expected_cols, (
        f"unexpected augmented columns: {set(aug.columns)} vs {expected_cols}")
    # The augmented parquet is built on the baseline DatetimeIndex; sanity
    # check the alignment so a stale cache surfaces loud.
    assert aug.index.equals(prices.index), (
        "augmented panel index does not match baseline prices index; "
        "rebuild via tools/build_augmented_panel.py")

    overlap = expected_cols & set(prices.columns)
    assert not overlap, (
        f"augmented tickers {overlap} already present in baseline panel; "
        f"the augmented panel should add net-new columns only.")

    prices_aug = pd.concat([prices, aug], axis=1)
    assert prices_aug.shape[1] == prices.shape[1] + 3, (
        f"augmented merge cardinality mismatch: {prices.shape[1]} + 3 vs {prices_aug.shape[1]}")
    returns_aug = compute_log_returns(prices_aug)

    aug_info_rows = pd.DataFrame([
        {"Symbol": "MER",  "Security": "Merrill Lynch (Wayback augmented)",
         "GICS Sector": "Financials", "GICS Sub-Industry": "Financials"},
        {"Symbol": "FNMA", "Security": "Fannie Mae (Yahoo pink-sheet)",
         "GICS Sector": "Financials", "GICS Sub-Industry": "Financials"},
        {"Symbol": "FMCC", "Security": "Freddie Mac (Yahoo pink-sheet)",
         "GICS Sector": "Financials", "GICS Sub-Industry": "Financials"},
    ])
    info_aug = (pd.concat([info, aug_info_rows], ignore_index=True)
                  .drop_duplicates(subset="Symbol", keep="first")
                  .reset_index(drop=True))
    assert len(info_aug) == len(info) + 3, "augmented info merge cardinality mismatch"

    print(f"  [AUGMENTED] +3 tickers (MER + FNMA + FMCC): "
          f"prices {prices_aug.shape}, returns {returns_aug.shape}, info {info_aug.shape}")
    return info_aug, prices_aug, returns_aug


def run_download(force=False, augmented=False):
    """CRSP-backed Stage 1a entry point (Yahoo path deprecated 2026-05-28).

    Delegates to src.stage1_data.crsp_panel.run_download_crsp. The
    `augmented` flag is a no-op because the CRSP panel already carries
    the eight crisis-era distressed names (LEH/BSC/MER/FNM/FRE/WAMU/AIG/F)
    via their PERMNOs. The Yahoo helpers (get_sp500_tickers,
    download_prices, clean_prices, _merge_augmented_panel, ...) remain in
    this module for reproducibility of the pre-2026-05-28 panel only.
    """
    from src.stage1_data.crsp_panel import run_download_crsp
    if augmented:
        print("  [INFO] --augmented is a no-op on the CRSP panel "
              "(distressed names already present via PERMNO).")
    return run_download_crsp(force=force)


if __name__ == "__main__":
    info, prices, returns = run_download()
    print(f"\nSummary:")
    print(f"  Tickers: {len(info)}")
    print(f"  Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"  Returns shape: {returns.shape}")
    print(f"\nSector distribution:")
    print(info["GICS Sector"].value_counts().to_string())
