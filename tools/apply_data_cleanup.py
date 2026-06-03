"""Apply pre-Stage-1 contamination cleanup to an existing cache.

When to run this
----------------
This tool is a one-shot remediation for caches built before the
audit-fixed src/stage1_data/download.py and tools/download_volume.py
were in place. A fresh

    python -m src.stage1_data.download
    python -m tools.download_volume

produces a clean cache directly (CFC and SBNY excluded from the
HISTORICAL_TICKERS probe, SW and AMCR truncated via TICKER_CUTOFFS
inside clean_prices, and the ADV computed from raw close x raw
volume) and does not need this tool. Running this script on a
freshly-downloaded cache is therefore a no-op for prices, volume,
and info; it will still drop adv rows for the never-present
CFC/SBNY (also a no-op) and re-verify the contamination detector
(a true safety check).

Operates in place on data/sp500_{prices,returns,volume,info,adv}.parquet.

Cleanup actions
---------------
1. Drop CFC and SBNY columns entirely. Yahoo's adjusted-close history
   under both symbols is post-event ticker reassignment to unrelated
   issuers (CFC post-2008-07 is a separate small-cap; SBNY post-2024-08
   is a sub-$3 penny stock). No legitimate Countrywide / Signature
   Bank window-relevant data is recoverable from Yahoo.

2. Mask SW pre-2024-07-15 as NaN. SW = Smurfit Westrock listed on the
   merger date; pre-merger SW Yahoo series is a flat $6 unrelated
   listing.

3. Mask AMCR pre-2019-06-11 as NaN. AMCR = Amcor PLC listed on the
   Bemis spin-off date; pre-spin AMCR Yahoo series is flat ~$28-43
   data for an earlier unrelated issuer.

4. Recompute log-returns from the cleaned prices.

5. Restrict the ADV cache to surviving ticker rows. ADV values are
   not recomputed here (the previous recomputation re-introduced the
   auto_adjust mismatch that tools/download_volume.py was fixed to
   avoid); if TICKER_CUTOFFS changes, re-run download_volume to
   refresh the truncated ADV values.

6. Verify no contaminated ticker remains under the same detector that
   would have caught the four above (>20% zero-returns or >20-day
   flatline).
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

from src.stage1_data.download import (
    TICKER_CUTOFFS as _DOWNLOAD_TICKER_CUTOFFS,
    _apply_ticker_cutoffs,
    _detect_contaminated,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

DROP_TICKERS = ["CFC", "SBNY"]
# Re-exported from src.stage1_data.download so both pre-Stage-1 entry
# points apply the same per-ticker cutoffs; a drift between the two
# would silently mask different sub-windows across pipelines. The
# matching cross-module test lives in tests/test_stage1_data.py.
TICKER_CUTOFFS = _DOWNLOAD_TICKER_CUTOFFS


def cleanup_prices(prices):
    out = prices.copy()

    # 1. Drop CFC / SBNY entirely
    drop_present = [t for t in DROP_TICKERS if t in out.columns]
    if drop_present:
        out = out.drop(columns=drop_present)
        print(f"  Dropped {drop_present} (ticker re-use, no recoverable data)")

    # 2-3. Truncate SW / AMCR pre-cutoff via the shared primitive so
    # tz-aware caches behave identically across both pre-Stage-1 paths.
    return _apply_ticker_cutoffs(out)


def cleanup_volume(volume):
    out = volume.copy()
    drop_present = [t for t in DROP_TICKERS if t in out.columns]
    if drop_present:
        out = out.drop(columns=drop_present)
    return _apply_ticker_cutoffs(out)


def cleanup_info(info, surviving_tickers):
    return info[info["Symbol"].isin(surviving_tickers)].reset_index(drop=True)


def cleanup_adv(adv, surviving_tickers):
    """Restrict the ADV cache to surviving ticker rows.

    ADV is computed in :mod:`tools.download_volume` from a single
    ``auto_adjust=False`` Yahoo download so that close and volume
    carry the same (split-only) adjustment; recomputing it here from
    sp500_prices.parquet (dividend-adjusted close) x sp500_volume.parquet
    (split-adjusted volume) would re-introduce the auto_adjust
    mismatch that understates dollar volume for dividend-paying
    tickers by the cumulative-dividend factor (median ~1.13x, max
    ~2.2x; appendix §3.6). The cleanup pass therefore only drops
    the rows for tickers that left the panel and leaves the per-
    ticker values untouched. If TICKER_CUTOFFS changes, re-run
    :mod:`tools.download_volume` to refresh the truncated ADV values.
    """
    keep = [t for t in adv.index if t in surviving_tickers]
    return adv.loc[keep].copy()


def main():
    print("=== apply_data_cleanup.py ===")
    print(f"Working dir: {DATA}")

    p_path  = DATA / "sp500_prices.parquet"
    r_path  = DATA / "sp500_returns.parquet"
    v_path  = DATA / "sp500_volume.parquet"
    i_path  = DATA / "sp500_info.parquet"
    a_path  = DATA / "sp500_adv.parquet"

    print("\n[1/5] Loading existing cache...")
    prices  = pd.read_parquet(p_path)
    returns = pd.read_parquet(r_path)
    volume  = pd.read_parquet(v_path)
    info    = pd.read_parquet(i_path)
    adv     = pd.read_parquet(a_path)
    print(f"  prices: {prices.shape}")
    print(f"  returns: {returns.shape}")
    print(f"  volume: {volume.shape}")
    print(f"  info: {info.shape}")
    print(f"  adv: {adv.shape}")

    print("\n[2/5] Applying contamination cleanup...")
    new_prices = cleanup_prices(prices)
    new_volume = cleanup_volume(volume)

    print("\n[3/5] Recomputing log-returns...")
    new_returns = (np.log(new_prices) - np.log(new_prices.shift(1))).iloc[1:]
    new_returns = new_returns.dropna(how="all")
    print(f"  new returns shape: {new_returns.shape}")

    surviving = set(new_prices.columns)
    new_info = cleanup_info(info, surviving)
    new_adv = cleanup_adv(adv, surviving)
    print(f"  surviving tickers: {len(surviving)} (was {prices.shape[1]})")

    print("\n[4/5] Contamination re-scan (>20% zero-returns OR >20-day flatline):")
    bad = _detect_contaminated(new_returns)
    if bad:
        print("  STILL CONTAMINATED:")
        for t, zf, mr in bad:
            print(f"    {t}: zero_frac={zf:.1%}, max_flatline={mr}d")
        print("  ABORT — refusing to write cache.")
        sys.exit(1)
    print("  Clean. No ticker remains above the contamination threshold.")

    print("\n[5/5] Writing back to data/...")
    new_prices.to_parquet(p_path)
    new_returns.to_parquet(r_path)
    new_volume.to_parquet(v_path)
    new_info.to_parquet(i_path)
    new_adv.to_parquet(a_path)
    print("  Done.")

    # Final per-snapshot alive-count report
    print("\nPost-cleanup alive counts per snapshot (>=80% in-window coverage):")
    SNAPS = [
        ("Oct 2008 Peak",      "2008-08-01", "2008-12-31"),
        ("Mar 2009 Recovery",  "2009-01-01", "2009-06-30"),
        ("2011-2012 Calm",     "2011-01-01", "2012-12-31"),
        ("2015 Calm",          "2015-01-01", "2015-12-31"),
        ("2018 VolShock",      "2018-01-01", "2018-12-31"),
        ("Jan 2020 Pre-shock", "2019-09-01", "2020-01-31"),
        ("Mar 2020 Peak",      "2020-01-01", "2020-06-30"),
        ("Jun 2020 Stable",    "2020-04-01", "2020-09-30"),
        ("2022 Rate Hikes",    "2022-01-01", "2022-12-31"),
        ("2025 Contemporary",  "2025-01-01", "2025-12-31"),
    ]
    for label, s, e in SNAPS:
        w = new_prices.loc[s:e]
        alive = (w.count() >= int(0.8 * len(w))).sum()
        print(f"  {label:22s}: {alive} alive (of {new_prices.shape[1]})")


if __name__ == "__main__":
    main()
