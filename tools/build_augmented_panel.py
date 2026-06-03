"""Assemble the three distressed-name CSVs (MER from Wayback, FNMA/FMCC from
live Yahoo) into a single augmented prices parquet aligned to the baseline
sp500_prices.parquet DatetimeIndex.

Outputs
-------
data/sp500_distressed_augmented.parquet
    shape (5534, 3), columns = [MER, FNMA, FMCC], values = adjusted close
    matching the sp500_prices.parquet convention. MER non-NaN only over
    2008-07-31 .. 2008-12-17 (Wayback span). FNMA/FMCC non-NaN full range.

data/sp500_distressed_augmented_volume.parquet
    shape (5534, 3), volume column. MER volume non-NaN ONLY over the
    20081101 snapshot segment (2008-07-31 .. 2008-10-31) where the volume
    is NYSE-realistic; the 20081218 segment (2008-11-03 .. 2008-12-17) has
    volume dropped because Yahoo had begun applying a retroactive BAC-merger
    adjustment that mangled the volume column (close prices stayed coherent).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data"
SRC_DIR = REPO / "tools" / "distressed_2008_panel"

BASELINE_PRICES = DATA_DIR / "sp500_prices.parquet"
OUT_PRICES = DATA_DIR / "sp500_distressed_augmented.parquet"
OUT_VOLUME = DATA_DIR / "sp500_distressed_augmented_volume.parquet"

MER_WAYBACK_AUTHORITATIVE = "20081101130053"


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
    return df


def main() -> None:
    assert BASELINE_PRICES.exists(), f"baseline prices missing: {BASELINE_PRICES}"
    baseline = pd.read_parquet(BASELINE_PRICES)
    target_index = baseline.index
    assert target_index.is_unique, "baseline index has duplicates"
    print(f"baseline index: {target_index.min().date()} .. {target_index.max().date()}, {len(target_index)} rows")

    mer = load_csv(SRC_DIR / "MER.csv")
    fnma = load_csv(SRC_DIR / "FNMA.csv")
    fmcc = load_csv(SRC_DIR / "FMCC.csv")

    # === Prices ===
    prices = pd.DataFrame(index=target_index)
    prices["MER"] = mer["AdjClose"].reindex(target_index)
    prices["FNMA"] = fnma["AdjClose"].reindex(target_index)
    prices["FMCC"] = fmcc["AdjClose"].reindex(target_index)

    # Sanity: MER non-NaN window
    mer_non_nan = prices["MER"].dropna().index
    assert len(mer_non_nan) == 98, f"MER should have 98 non-NaN rows, got {len(mer_non_nan)}"
    assert mer_non_nan.min() == pd.Timestamp("2008-07-31"), f"MER first date mismatch: {mer_non_nan.min()}"
    assert mer_non_nan.max() == pd.Timestamp("2008-12-17"), f"MER last date mismatch: {mer_non_nan.max()}"

    # Sanity: FNMA / FMCC near-full coverage; small NaN tail acceptable for
    # non-trading days. Just confirm Oct 2008 window is fully populated.
    oct2008 = slice("2008-08-01", "2008-12-31")
    assert prices.loc[oct2008, "FNMA"].notna().all(), "FNMA has NaN in Oct 2008 window"
    assert prices.loc[oct2008, "FMCC"].notna().all(), "FMCC has NaN in Oct 2008 window"

    # Sanity: positivity on non-NaN
    for col in ["MER", "FNMA", "FMCC"]:
        non_nan = prices[col].dropna()
        assert (non_nan > 0).all(), f"{col} has non-positive prices"

    prices.to_parquet(OUT_PRICES)
    print(f"wrote {OUT_PRICES}: shape={prices.shape}, MER non-NaN={prices['MER'].notna().sum()}, "
          f"FNMA non-NaN={prices['FNMA'].notna().sum()}, FMCC non-NaN={prices['FMCC'].notna().sum()}")

    # === Volume ===
    volume = pd.DataFrame(index=target_index)
    # MER: only keep the authoritative-snapshot rows (where source_snapshot == 20081101130053)
    mer_auth_mask = mer["source_snapshot"].astype(str) == MER_WAYBACK_AUTHORITATIVE
    mer_auth = mer.loc[mer_auth_mask, "Volume"]
    assert len(mer_auth) == 66, f"MER authoritative-volume rows should be 66, got {len(mer_auth)}"
    assert (mer_auth > 1e6).all(), "MER authoritative volume should be NYSE-realistic (>1M daily)"
    volume["MER"] = mer_auth.reindex(target_index)

    volume["FNMA"] = fnma["Volume"].reindex(target_index)
    volume["FMCC"] = fmcc["Volume"].reindex(target_index)

    volume.to_parquet(OUT_VOLUME)
    print(f"wrote {OUT_VOLUME}: shape={volume.shape}, MER non-NaN={volume['MER'].notna().sum()}, "
          f"FNMA non-NaN={volume['FNMA'].notna().sum()}, FMCC non-NaN={volume['FMCC'].notna().sum()}")


if __name__ == "__main__":
    main()
