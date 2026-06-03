"""
Fetch daily volume for the S&P 500 universe and compute mean dollar
volume per ticker.

The main pipeline stores the split-and-dividend-adjusted close in
:file:`sp500_prices.parquet` (as Stage 1 needs for log-return
computation), so reusing it for the dollar-volume product against a
separately-fetched ``auto_adjust=False`` volume column under-states
turnover for dividend-paying tickers by the cumulative-dividend
correction factor that the adjusted close embeds. The split
correction does cancel — Yahoo's ``auto_adjust=False`` pipeline
applies the cumulative split factor to both close and volume — but
the dividend correction does not, so e.g. MO is under-stated by
~2.2x and T by ~1.9x over the 2004-2025 sample, while non-dividend
names (TSLA, AMZN) are unaffected (see appendix section 3.6 for the
full table). The correct construction is therefore to fetch close and
volume from a single ``auto_adjust=False`` Yahoo request and
multiply them in memory.

The volume column is cached to :file:`sp500_volume.parquet` (the
``auto_adjust=False`` split-adjusted volume, suitable for any
downstream use that requires per-day share-count comparability); the
close is consumed in-memory by :func:`compute_adv` and not persisted,
since the main pipeline already stores the dividend-adjusted close in
:file:`sp500_prices.parquet`. The mean dollar volume per ticker is
cached to :file:`sp500_adv.parquet`.

Two ticker-reuse cutoffs (SW from 2024-07-15, AMCR from 2019-06-11)
mirror :mod:`src.stage1_data.download` so the ADV for each re-used
symbol is computed only over the legitimate post-listing observations,
matching the panel composition used downstream.

Usage
-----
    python -m tools.download_volume

The download is sequential and rate-limited by Yahoo Finance; a full
universe pull takes roughly 15-25 minutes. The cache is reused on
subsequent runs of :mod:`tools.run_multipanel` when ``kind="adv"``.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from src.config import DATA_DIR, DATA_START, DATA_END
from src.stage1_data.download import _apply_ticker_cutoffs


VOLUME_PATH = DATA_DIR / "sp500_volume.parquet"
ADV_PATH = DATA_DIR / "sp500_adv.parquet"


def fetch_volume_and_raw_close(tickers: list[str],
                               start: str = DATA_START,
                               end: str = DATA_END,
                               max_retries: int = 3,
                               max_missing_fraction: float = 0.02,
                               ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pull daily close and split-adjusted volume per ticker.

    Returns ``(close, volume)`` aligned on the same date index and
    ticker columns, both from a single ``auto_adjust=False`` Yahoo
    download so that the product ``close * volume`` is the actual
    daily dollar volume. ``auto_adjust=False`` applies the cumulative
    split factor to both columns but leaves the dividend correction
    out of the close, which is the correct convention for a dollar-
    volume product (the dividend adjustment is purely a return-side
    correction and would otherwise under-state historical turnover
    for dividend-paying tickers; see the module docstring).

    Each ticker is attempted up to ``max_retries`` times with linear
    backoff (1s, 2s, 3s) to ride out transient Yahoo HTTP 429 / 500
    responses. After the loop the caller asserts that no more than
    ``max_missing_fraction`` of the requested tickers came back empty;
    a silent ticker-drop would desync this cache from
    ``sp500_prices.parquet`` and produce the exact internal
    inconsistency the audit caught (BNY/VEEV missing from volume).

    The pull is sequential to avoid Yahoo's rate-limit guard; expect
    15-25 minutes on the full 501-ticker universe over 2004-2025.
    """
    close_recs: dict[str, pd.Series] = {}
    vol_recs: dict[str, pd.Series] = {}
    failed: list[tuple[str, str]] = []
    for i, ticker in enumerate(tickers, start=1):
        last_exc: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                df = yf.download(ticker, start=start, end=end,
                                 progress=False, auto_adjust=False)
                if isinstance(df.columns, pd.MultiIndex):
                    close = df[("Close", ticker)]
                    vol = df[("Volume", ticker)]
                else:
                    close = df["Close"]
                    vol = df["Volume"]
                if close.dropna().empty or vol.dropna().empty:
                    raise RuntimeError("empty close/volume response")
                close_recs[ticker] = close.astype(float)
                vol_recs[ticker] = vol.astype(float)
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                if attempt < max_retries:
                    time.sleep(attempt)  # 1s, 2s, ...
        if last_exc is not None:
            print(f"[{i:3d}/{len(tickers)}] {ticker}: failed after "
                  f"{max_retries} retries ({last_exc})")
            failed.append((ticker, str(last_exc)[:120]))
        # Light throttle to be polite to the API.
        time.sleep(0.1)
        if i % 25 == 0:
            print(f"  {i}/{len(tickers)} tickers fetched "
                  f"({len(failed)} failed so far)")

    missing_fraction = len(failed) / max(1, len(tickers))
    if missing_fraction > max_missing_fraction:
        details = "\n".join(f"    {t}: {msg}" for t, msg in failed[:30])
        raise RuntimeError(
            f"Yahoo volume fetch missing {len(failed)}/{len(tickers)} "
            f"tickers ({missing_fraction:.1%}), above the "
            f"{max_missing_fraction:.0%} guard. Refusing to write a "
            f"partial cache that would desync from sp500_prices.parquet:\n"
            f"{details}"
        )
    if failed:
        print(f"  {len(failed)} tickers permanently failed "
              f"(below the {max_missing_fraction:.0%} guard, accepted): "
              f"{[t for t, _ in failed]}")
    return pd.DataFrame(close_recs), pd.DataFrame(vol_recs)


def compute_adv(close: pd.DataFrame,
                volume: pd.DataFrame) -> pd.Series:
    """Mean dollar-volume per ticker = ``mean(close * volume)``.

    Both inputs MUST come from the same ``auto_adjust=False`` Yahoo
    download (so the dividend correction does not sit on the close
    only); see the module docstring for the reason. The two frames
    are aligned pairwise on the intersection of their date indices
    and ticker columns, and missing entries are skipped by the pandas
    mean.
    """
    common_cols = close.columns.intersection(volume.columns)
    common_idx = close.index.intersection(volume.index)
    p = close.loc[common_idx, common_cols]
    v = volume.loc[common_idx, common_cols]
    dv = p * v
    return dv.mean(axis=0, skipna=True).rename("mean_dollar_volume")


def main() -> int:
    prices_path = DATA_DIR / "sp500_prices.parquet"
    if not prices_path.exists():
        print(f"Prices cache missing at {prices_path}; run Stage 1 download first.")
        return 1

    prices = pd.read_parquet(prices_path)
    tickers = list(prices.columns)
    print(f"Fetching raw close and volume for {len(tickers)} tickers...")

    raw_close, volume = fetch_volume_and_raw_close(tickers)

    # Apply the same TICKER_CUTOFFS as the main download so the ADV
    # for re-used symbols (SW, AMCR) is computed only over legitimate
    # post-listing observations and matches the panel that downstream
    # stages actually see.
    raw_close = _apply_ticker_cutoffs(raw_close)
    volume = _apply_ticker_cutoffs(volume)

    adv = compute_adv(raw_close, volume)

    # All cross-cache checks run BEFORE persisting so a mismatch never
    # reaches disk. The appendix (app:adv) advertises that the writer
    # validates against sp500_prices.parquet pre-persist; a raise here
    # therefore leaves the existing on-disk caches untouched.
    #
    # Cross-cache integrity guard: prices, volume, and ADV must agree on
    # the ticker set, otherwise downstream multipanel selection silently
    # drops tickers that exist in one cache but not the others. The audit
    # caught a real instance of this (BNY/VEEV in prices after a refresh,
    # BK/CTRA still in volume/ADV from the older Wikipedia pull).
    prices_cols = set(prices.columns)
    volume_cols = set(volume.columns)
    adv_rows = set(adv.index)
    if not (prices_cols == volume_cols == adv_rows):
        only_prices = sorted(prices_cols - volume_cols - adv_rows)
        only_volume = sorted(volume_cols - prices_cols)
        only_adv = sorted(adv_rows - prices_cols)
        raise RuntimeError(
            f"Cache ticker-set mismatch after volume fetch (pre-persist):\n"
            f"  in prices only:  {only_prices}\n"
            f"  in volume only:  {only_volume}\n"
            f"  in adv only:     {only_adv}\n"
            f"Re-run `python -m src.stage1_data.download` (with the "
            f"prices cache deleted) and then this tool to bring the "
            f"three caches back into lockstep. The on-disk "
            f"sp500_volume.parquet and sp500_adv.parquet are unchanged."
        )

    # Numeric scale guard. On any date both modes share as an anchor,
    # auto_adjust=True (prices cache) and auto_adjust=False (raw_close
    # from this fresh pull) should agree to within the no-corporate-
    # action band: dividend adjustments are cumulative backwards from
    # each call's own anchor date, so on the shared most-recent date
    # the only thing that can drive a numerical gap is a split between
    # the prices-cache pull date and today. A |log| > 0.3 gap (~35%
    # price ratio) signals a stale prices cache; the audit caught this
    # for CVNA in 2026-05 (5x split between an older prices pull and
    # the fresh volume pull). The fix is to refresh the prices cache
    # via `python -m src.stage1_data.download` with force=True.
    common_idx = prices.index.intersection(raw_close.index)
    common_cols = prices.columns.intersection(raw_close.columns)
    if len(common_idx) == 0 or len(common_cols) == 0:
        raise RuntimeError(
            "No overlap between prices cache and the freshly-pulled "
            "raw close; cannot run the numeric scale check. The on-disk "
            "sp500_volume.parquet and sp500_adv.parquet are unchanged."
        )
    anchor = common_idx[-1]
    p_anchor = prices.loc[anchor, common_cols]
    r_anchor = raw_close.loc[anchor, common_cols]
    both_finite = p_anchor.notna() & r_anchor.notna() & (r_anchor != 0)
    ratios = (p_anchor[both_finite] / r_anchor[both_finite]).abs()
    log_dev = np.log(ratios).abs()
    suspect = log_dev[log_dev > 0.3].sort_values(ascending=False)
    if len(suspect) > 0:
        details = "\n".join(
            f"    {t}: prices={p_anchor[t]:.4f}, raw_close="
            f"{r_anchor[t]:.4f}, ratio={ratios[t]:.3f}"
            for t in suspect.index[:20]
        )
        raise RuntimeError(
            f"Numeric scale mismatch between prices cache and the "
            f"freshly-pulled raw close on {anchor.date()} "
            f"(|log(prices/raw_close)| > 0.3 for {len(suspect)} "
            f"tickers). Likely a corporate-action split between the "
            f"prices-cache pull and today; refresh via\n"
            f"  python -m src.stage1_data.download   "
            f"(with `force=True` or the prices parquet deleted)\n"
            f"then re-run this tool. The on-disk sp500_volume.parquet "
            f"and sp500_adv.parquet are unchanged:\n"
            + details
        )

    # All pre-persist checks passed; commit the caches to disk.
    volume.to_parquet(VOLUME_PATH)
    print(f"Volume cached to {VOLUME_PATH}")
    adv.to_frame().to_parquet(ADV_PATH)
    print(f"ADV cached to {ADV_PATH}; range = "
          f"[{adv.min():.2e}, {adv.max():.2e}]")
    print(f"Integrity check: all three caches agree on "
          f"{len(prices_cols)} tickers.")
    print(f"Numeric scale check: {len(ratios)} tickers within "
          f"|log(prices/raw_close)| <= 0.3 on {anchor.date()}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
