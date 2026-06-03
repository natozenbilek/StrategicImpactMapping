"""CRSP-based volume + ADV rebuild for the multipanel sweep.

Produces the two caches the sweep selectors expect, in the same schema
the legacy tools.download_volume wrote from Yahoo (501-ticker, 2004-25):

* data/sp500_volume.parquet  - DatetimeIndex (CRSP trading grid 1985-
  2024) x ticker, CRSP `vol` (shares; split-adjusted by CRSP via cfacshr).
* data/sp500_adv.parquet     - ticker index x single column
  ``mean_dollar_volume``, computed as mean over the panel window of
  |prc| * vol (the CRSP dollar-volume product, split-invariant because
  the cfacshr/cfacpr factors cancel in the prc*vol product).

Re-uses src.stage1_data.crsp_panel._resolve_tickers for the PERMNO ->
ticker mapping (overrides + collision suffix) so the column names match
sp500_returns.parquet exactly. Both caches replace the Yahoo .bak_yahoo
siblings under data/.
"""
import pandas as pd
import numpy as np

from src.config import DATA_DIR
from src.stage1_data.crsp_panel import _resolve_tickers

DSF_PATH = DATA_DIR / "crsp" / "dsf_long.parquet"
NAMES_PATH = DATA_DIR / "crsp" / "stocknames.csv"
VOLUME_PATH = DATA_DIR / "sp500_volume.parquet"
ADV_PATH = DATA_DIR / "sp500_adv.parquet"

# Match the Stage-1 returns panel window so volume / returns share the
# same DatetimeIndex. SEDCO (PERMNO 48549) DSF rows pre-1985 are kept in
# data/crsp/dsf_long.parquet for completeness but excluded here.
WINDOW_START = "1985-01-01"
WINDOW_END = "2024-12-31"


def build():
    print("[CRSP volume] loading DSF + stocknames...")
    dsf = pd.read_parquet(DSF_PATH)
    dsf["date"] = pd.to_datetime(dsf["date"])
    dsf = dsf[(dsf["date"] >= WINDOW_START) & (dsf["date"] <= WINDOW_END)].copy()
    stocknames = pd.read_csv(NAMES_PATH, parse_dates=["namedt", "nameenddt"])

    tmap = _resolve_tickers(stocknames)
    p2t = dict(zip(tmap["permno"], tmap["ticker"]))

    dsf["ticker"] = dsf["permno"].map(p2t)
    dsf = dsf.dropna(subset=["ticker", "vol"])
    # Negative prc encodes a bid-ask midpoint in CRSP; treat as positive
    # so dollar-volume stays >= 0.
    dsf["prc_abs"] = dsf["prc"].abs()
    dsf["dvol"] = dsf["prc_abs"] * dsf["vol"]

    print(f"  rows: {len(dsf):,}  unique tickers: {dsf['ticker'].nunique()}")

    print("[CRSP volume] pivoting wide volume panel...")
    vol_wide = dsf.pivot_table(
        index="date", columns="ticker", values="vol", aggfunc="first"
    ).sort_index()
    vol_wide.index.name = None
    assert vol_wide.shape[0] > 0, "empty volume panel"
    print(f"  volume shape: {vol_wide.shape}")

    print("[CRSP volume] computing mean dollar volume (ADV)...")
    adv = (
        dsf.dropna(subset=["dvol"])
        .groupby("ticker")["dvol"]
        .mean()
        .rename("mean_dollar_volume")
        .sort_values(ascending=False)
    )
    assert (adv > 0).all(), "non-positive ADV after groupby"
    print(f"  ADV entries: {len(adv)}")
    print("  top 5 ADV:")
    print(adv.head(5).apply(lambda x: f"{x:,.0f}").to_string())

    # Downcast nullable Float64 to numpy float64 for downstream lstsq
    # / corr callers (same fix as crsp_panel.run_download_crsp).
    vol_wide = vol_wide.astype(float)
    adv = adv.astype(float)
    vol_wide.to_parquet(VOLUME_PATH)
    adv.to_frame().to_parquet(ADV_PATH)
    print(f"\nwrote: {VOLUME_PATH}")
    print(f"wrote: {ADV_PATH}")


if __name__ == "__main__":
    build()
