"""Reconstruct Merrill Lynch (MER) daily OHLCV for 2008-07-31 .. 2008-12-17
from two Wayback Machine snapshots of Yahoo Finance's historical-price page.

MER was delisted on 2009-01-01 (Bank of America acquisition closed at a
0.8595 share-exchange ratio); the live Yahoo Finance API returns an empty
shell for the ticker and Stooq does not carry it. Two Wayback snapshots
survive that cover the paper's Oct 2008 Peak window (2008-08-01 .. 2008-12-31):

  20081101130053  -> default 3-month page yields 31-Jul-2008 .. 31-Oct-2008
                     (67 trading rows). Volumes are NYSE-realistic
                     (30M-200M daily). Authoritative source.

  20081218052907  -> default 3-month page yields 16-Sep-2008 .. 17-Dec-2008
                     (67 trading rows). Volumes are unrealistically low
                     (50k-400k) because the snapshot was captured during
                     a window in which Yahoo had begun applying a
                     retroactive BAC-merger adjustment; close prices
                     stay coherent but the volume column is unreliable.
                     Used ONLY for the Nov 1 .. Dec 17 2008 close prices
                     to extend the panel beyond the 20081101 snapshot's
                     cut-off, with volume column dropped on those rows.

The two snapshots overlap on 16-Sep .. 31-Oct; on the overlap we keep the
20081101 row (authoritative volume + close). Output: a single CSV with
columns Date, Open, High, Low, Close, AdjClose, Volume — Volume blank
for the 20081218-derived rows.
"""

from __future__ import annotations

import io
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

warnings.filterwarnings("ignore")

OUT_DIR = Path(__file__).resolve().parent / "distressed_2008_panel"
OUT_CSV = OUT_DIR / "MER.csv"

SNAPSHOT_AUTHORITATIVE = "20081101130053"
SNAPSHOT_EXTENSION = "20081218052907"
URL_TEMPLATE = "https://web.archive.org/web/{ts}id_/http://finance.yahoo.com/q/hp?s=MER"

DATE_RE = re.compile(r"^(\d{1,2})-(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-(\d{2,4})$")
MONTHS = {m: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], start=1)}

UA = {"User-Agent": "academic-research-probe/1.0 (single-author replication)"}


@dataclass
class ParsedRow:
    date: pd.Timestamp
    open_: float
    high: float
    low: float
    close: float
    volume: Optional[int]
    adj_close: float


def parse_date(token: str) -> Optional[pd.Timestamp]:
    m = DATE_RE.match(token.strip())
    if not m:
        return None
    d, mon, y = int(m.group(1)), MONTHS[m.group(2)], int(m.group(3))
    if y < 100:
        y += 2000  # snapshots are 2008, two-digit forms safe
    return pd.Timestamp(year=y, month=mon, day=d)


def parse_snapshot_html(html: str) -> List[ParsedRow]:
    soup = BeautifulSoup(html, "html.parser")
    rows: List[ParsedRow] = []
    for tr in soup.find_all("tr"):
        cells = [c.get_text(strip=True) for c in tr.find_all(["td", "th"])]
        if len(cells) < 7:
            continue
        ts = parse_date(cells[0])
        if ts is None:
            continue
        # Skip dividend annotation rows: they have shape [date, "$ x Dividend"]
        # — but filter above already drops len<7.
        try:
            open_ = float(cells[1].replace(",", ""))
            high = float(cells[2].replace(",", ""))
            low = float(cells[3].replace(",", ""))
            close = float(cells[4].replace(",", ""))
            volume = int(cells[5].replace(",", ""))
            adj = float(cells[6].replace(",", ""))
        except (ValueError, IndexError):
            continue
        assert open_ > 0 and high > 0 and low > 0 and close > 0, (
            f"non-positive price on {ts.date()}: {cells}")
        assert volume >= 0, f"negative volume on {ts.date()}: {volume}"
        rows.append(ParsedRow(ts, open_, high, low, close, volume, adj))
    return rows


def fetch(timestamp: str) -> str:
    url = URL_TEMPLATE.format(ts=timestamp)
    r = requests.get(url, headers=UA, timeout=60)
    r.raise_for_status()
    return r.text


def to_frame(rows: List[ParsedRow], keep_volume: bool) -> pd.DataFrame:
    df = pd.DataFrame([{
        "Date": r.date,
        "Open": r.open_,
        "High": r.high,
        "Low": r.low,
        "Close": r.close,
        "Volume": r.volume if keep_volume else pd.NA,
        "AdjClose": r.adj_close,
    } for r in rows])
    return df.set_index("Date").sort_index()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Fetching authoritative snapshot {SNAPSHOT_AUTHORITATIVE} ...")
    df_a = to_frame(parse_snapshot_html(fetch(SNAPSHOT_AUTHORITATIVE)), keep_volume=True)
    print(f"  -> {len(df_a)} rows, {df_a.index.min().date()} .. {df_a.index.max().date()}")
    assert len(df_a) >= 60, f"authoritative snapshot too short: {len(df_a)}"
    assert df_a["Volume"].max() > 1e7, (
        f"authoritative snapshot volume should be NYSE-realistic; got max={df_a['Volume'].max()}")

    print(f"Fetching extension snapshot {SNAPSHOT_EXTENSION} ...")
    df_b_full = to_frame(parse_snapshot_html(fetch(SNAPSHOT_EXTENSION)), keep_volume=False)
    print(f"  -> {len(df_b_full)} rows, {df_b_full.index.min().date()} .. {df_b_full.index.max().date()}")
    assert len(df_b_full) >= 60, f"extension snapshot too short: {len(df_b_full)}"

    # Trim the extension to dates strictly AFTER the authoritative snapshot ends.
    cutoff = df_a.index.max()
    df_b = df_b_full.loc[df_b_full.index > cutoff].copy()
    print(f"  -> {len(df_b)} rows after trimming to dates > {cutoff.date()}")

    combined = pd.concat([df_a, df_b], axis=0).sort_index()
    assert combined.index.is_unique, "duplicate dates after concat"
    assert combined.index.is_monotonic_increasing, "non-monotonic dates"
    assert combined["Close"].notna().all(), "missing Close prices"
    assert (combined["Close"] > 0).all(), "non-positive Close detected"

    # Save with the source-segment column so downstream code can audit.
    combined["source_snapshot"] = SNAPSHOT_AUTHORITATIVE
    combined.loc[combined.index > cutoff, "source_snapshot"] = SNAPSHOT_EXTENSION

    combined.to_csv(OUT_CSV)
    print(f"\nWrote {OUT_CSV}")
    print(f"  span: {combined.index.min().date()} .. {combined.index.max().date()}")
    print(f"  rows: {len(combined)}")
    print(f"  by source: {combined['source_snapshot'].value_counts().to_dict()}")
    print(f"  close range: {combined['Close'].min():.2f} .. {combined['Close'].max():.2f}")


if __name__ == "__main__":
    main()
