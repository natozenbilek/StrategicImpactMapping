"""Pull FNMA (Fannie Mae) and FMCC (Freddie Mac) daily adjusted close + volume
from Yahoo Finance for 2004-01-02 .. 2025-12-30.

Both names entered conservatorship 2008-09-07 and were delisted from NYSE
2010-06-16, then continued trading as OTC pink-sheet securities under the
FNMA / FMCC tickers (current Yahoo Finance retrieves both with full
2004-present history). They cover the 2008 Peak window without any reach
into archived sources.

Output: tools/distressed_2008_panel/FNMA.csv, FMCC.csv with columns
Date, Open, High, Low, Close, Volume, AdjClose. yfinance is called with
auto_adjust=False so that Close (raw) and AdjClose (dividend+split adjusted)
are both available; downstream pipeline uses AdjClose to match the existing
sp500_prices.parquet convention (which is auto_adjust=True == AdjClose).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

OUT_DIR = Path(__file__).resolve().parent / "distressed_2008_panel"

START = "2004-01-02"
END = "2025-12-31"

TICKERS = ["FNMA", "FMCC"]


def fetch_one(ticker: str) -> pd.DataFrame:
    df = yf.download(
        ticker, start=START, end=END,
        auto_adjust=False, progress=False, threads=False,
    )
    if df is None or df.empty:
        raise RuntimeError(f"{ticker}: empty frame from yfinance")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={"Adj Close": "AdjClose"})
    expected = {"Open", "High", "Low", "Close", "Volume", "AdjClose"}
    missing = expected - set(df.columns)
    assert not missing, f"{ticker}: missing columns {missing}"
    df = df[["Open", "High", "Low", "Close", "Volume", "AdjClose"]].sort_index()
    assert df.index.is_unique, f"{ticker}: duplicate dates"
    assert (df["Close"] > 0).all(), f"{ticker}: non-positive Close"
    assert (df["AdjClose"] > 0).all(), f"{ticker}: non-positive AdjClose"
    return df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for tkr in TICKERS:
        print(f"Fetching {tkr} ...")
        df = fetch_one(tkr)
        out = OUT_DIR / f"{tkr}.csv"
        df.to_csv(out, index_label="Date")
        peak = df.loc["2008-08-01":"2008-12-31"]
        print(f"  {tkr}: total rows={len(df)}, span={df.index.min().date()}..{df.index.max().date()}")
        print(f"        Oct 2008 window rows: {len(peak)}, "
              f"AdjClose range: {peak['AdjClose'].min():.2f}..{peak['AdjClose'].max():.2f}")


if __name__ == "__main__":
    main()
