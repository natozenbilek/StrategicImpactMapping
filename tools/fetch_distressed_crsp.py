"""WRDS/CRSP gold-standard fetch for the six 2008-era distressed tickers.

Replaces the free-public-source patch (Wayback + Yahoo pink sheet) that
tools/fetch_mer_wayback.py + tools/fetch_fnma_fmcc_yahoo.py produced.
CRSP carries delisting-return-aware total returns for LEH, BSC, WAMU,
MER, FNM, FRE (and AIG/F as survival controls), keyed on PERMNO so
Yahoo's ticker-reuse contamination (CFC, SBNY, WM=>WaMu==WasteMgmt) is
structurally impossible.

USAGE
-----
Requires an active WRDS subscription seat (institutional). Set
WRDS_USERNAME env var or pass on first call:

    python -m tools.fetch_distressed_crsp --username <wrds_user>

First call prompts for the WRDS password and persists a ~/.pgpass entry
for subsequent passwordless connections (standard wrds package behaviour).

OUTPUT
------
tools/distressed_2008_panel/crsp_<TICKER>.csv  per-ticker daily panel
  Date, PERMNO, Open, High, Low, Close, Volume, AdjClose, RawReturn, DelistingReturn
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

OUT_DIR = Path(__file__).resolve().parent / "distressed_2008_panel"

# Tickers we want CRSP coverage for. The first six are the augmented-panel
# targets (free-public sources covered only 3/6). AIG and F are survival
# controls — both already present in the live Yahoo panel; pulling them
# from CRSP too lets us cross-validate the Yahoo total-return convention
# against CRSP's RET column (which already folds in dividends).
TARGET_TICKERS = ["LEH", "BSC", "WAMU", "MER", "FNM", "FRE", "AIG", "F"]

DATE_START = "2004-01-02"
DATE_END = "2009-12-31"

# Tickers can be reused across PERMNOs (Yahoo's CFC, SBNY, WM, MERR issues
# repeat in CRSP if you do not filter by namedt/nameenddt). We require the
# (namedt, nameenddt) interval to cover at least one 2008 trading day,
# which is the regime the paper's Oct 2008 Peak snapshot lives in.
NAME_OVERLAP_PROBE_DATE = "2008-01-15"  # before BSC (May 2008), LEH (Sep 17), WAMU (Sep 26) delistings


def fetch_permno_map(db, tickers):
    """Resolve ticker -> PERMNO valid on NAME_OVERLAP_PROBE_DATE.

    crsp.stocknames carries (PERMNO, ticker, namedt, nameenddt) intervals;
    a ticker can map to multiple PERMNOs historically. We restrict to the
    interval covering Sep 30 2008 so we get the 2008-era listing for each
    name (e.g. LEH = Lehman Brothers PERMNO, not whatever reuse came after).
    """
    placeholders = ",".join(f"'{t}'" for t in tickers)
    q = f"""
        SELECT permno, ticker, comnam, namedt, nameenddt
        FROM crsp.stocknames
        WHERE ticker IN ({placeholders})
          AND namedt <= '{NAME_OVERLAP_PROBE_DATE}'
          AND nameenddt >= '{NAME_OVERLAP_PROBE_DATE}'
    """
    df = db.raw_sql(q)
    print(f"  resolved {len(df)} ticker rows on {NAME_OVERLAP_PROBE_DATE}:")
    for _, row in df.iterrows():
        print(f"    PERMNO {int(row['permno']):>6d}  ticker {row['ticker']:<6s}  "
              f"{row['comnam']:<40s}  {row['namedt']} -> {row['nameenddt']}")
    missing = set(tickers) - set(df["ticker"].str.upper())
    if missing:
        print(f"  WARNING: not in stocknames on probe date: {sorted(missing)}")
    return df


def fetch_daily_panel(db, permnos):
    """Pull crsp.dsf: prc, ret, retx, vol, shrout, cfacpr for each PERMNO."""
    placeholders = ",".join(str(int(p)) for p in permnos)
    q = f"""
        SELECT permno, date, prc, openprc, askhi, bidlo, ret, retx,
               vol, shrout, cfacpr, cfacshr
        FROM crsp.dsf
        WHERE permno IN ({placeholders})
          AND date BETWEEN '{DATE_START}' AND '{DATE_END}'
        ORDER BY permno, date
    """
    df = db.raw_sql(q)
    print(f"  pulled {len(df)} (PERMNO, date) rows from crsp.dsf")
    return df


def fetch_delisting(db, permnos):
    """Pull crsp.dsedelist: delisting date / code / DLRET / DLRETX."""
    placeholders = ",".join(str(int(p)) for p in permnos)
    q = f"""
        SELECT permno, dlstdt, dlstcd, dlret, dlretx
        FROM crsp.dsedelist
        WHERE permno IN ({placeholders})
    """
    df = db.raw_sql(q)
    print(f"  pulled {len(df)} delisting events")
    for _, row in df.iterrows():
        print(f"    PERMNO {int(row['permno']):>6d}  "
              f"{row['dlstdt']}  code={int(row['dlstcd'])}  "
              f"DLRET={row['dlret']}  DLRETX={row['dlretx']}")
    return df


def build_per_ticker_csv(permno_map, daily, delist, ticker):
    """Splice daily prices + delisting return into one tidy CSV.

    Adjusted close convention to match Yahoo auto_adjust=True:
        AdjClose_t = |prc_t| / cfacpr_t * cfacpr_FIRST
    Negative prc values in CRSP are bid-ask midpoints (rare on these
    names); abs() pins them to a positive level. cfacpr is the cumulative
    price-adjustment factor (splits + spinoffs + dividends). The ratio
    against the first-date cfacpr re-bases the series to that date so
    the level matches Yahoo's auto_adjust=True output.
    """
    ticker = ticker.upper()
    p = permno_map[permno_map["ticker"].str.upper() == ticker]
    if p.empty:
        return None
    permno = int(p["permno"].iloc[0])
    d = daily[daily["permno"] == permno].copy().sort_values("date")
    if d.empty:
        return None
    d["prc_abs"] = d["prc"].abs()
    base_cfac = d["cfacpr"].iloc[0]
    d["AdjClose"] = d["prc_abs"] / d["cfacpr"] * base_cfac
    d["RawReturn"] = d["ret"]

    dl = delist[delist["permno"] == permno]
    delist_ret_val = float(dl["dlret"].iloc[0]) if not dl.empty and pd.notna(dl["dlret"].iloc[0]) else None
    d["DelistingReturn"] = pd.NA
    if delist_ret_val is not None and not dl.empty:
        d.loc[d["date"] == dl["dlstdt"].iloc[0], "DelistingReturn"] = delist_ret_val

    out = d[["date", "permno", "openprc", "askhi", "bidlo", "prc_abs",
             "vol", "AdjClose", "RawReturn", "DelistingReturn"]].rename(columns={
        "date": "Date", "permno": "PERMNO",
        "openprc": "Open", "askhi": "High", "bidlo": "Low",
        "prc_abs": "Close", "vol": "Volume",
    })
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", default=os.environ.get("WRDS_USERNAME"),
                        help="WRDS username (or set WRDS_USERNAME env var)")
    args = parser.parse_args()

    if not args.username:
        parser.error("WRDS username required: --username or WRDS_USERNAME env var")

    try:
        import wrds
    except ImportError:
        sys.exit("ERROR: wrds package not installed. Run: pip install wrds")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Connecting to WRDS as {args.username} ...")
    db = wrds.Connection(wrds_username=args.username)

    print("\n[1/3] Resolving ticker -> PERMNO on 2008-09-30 ...")
    permno_map = fetch_permno_map(db, TARGET_TICKERS)
    if permno_map.empty:
        sys.exit("ERROR: no PERMNOs resolved; check ticker set and WRDS access")

    permnos = permno_map["permno"].astype(int).unique().tolist()
    print(f"\n[2/3] Pulling daily panel for {len(permnos)} PERMNOs ...")
    daily = fetch_daily_panel(db, permnos)

    print(f"\n[3/3] Pulling delisting events ...")
    delist = fetch_delisting(db, permnos)

    print("\n[Writing per-ticker CSVs]")
    for ticker in TARGET_TICKERS:
        out = build_per_ticker_csv(permno_map, daily, delist, ticker)
        if out is None:
            print(f"  {ticker}: SKIP (no CRSP data)")
            continue
        out_path = OUT_DIR / f"crsp_{ticker}.csv"
        out.to_csv(out_path, index=False)
        print(f"  {ticker}: wrote {len(out)} rows -> {out_path.name}  "
              f"AdjClose range {out['AdjClose'].min():.2f} .. {out['AdjClose'].max():.2f}")

    db.close()
    print("\nDone. Next step: rebuild data/sp500_distressed_augmented.parquet "
          "from the new crsp_*.csv files; tools/build_augmented_panel.py "
          "needs a small variant for the CRSP path.")


if __name__ == "__main__":
    main()
