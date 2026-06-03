"""CRSP S&P 500 historical panel pull, 1995-2025.

For the friend with WRDS+CRSP access: this script pulls the full S&P 500
historical constituent panel over a 30-year window, batched to stay under
the per-query memory/row limit. Output: parquet files + two CSVs (small)
in the working directory.

USAGE
-----
    pip install wrds pandas pyarrow
    python fetch_crsp_sp500_full.py --username <wrds_user>

First call prompts for the WRDS password; subsequent calls use the
~/.pgpass cache. DUO multi-factor approval is required at login.

PULLS
-----
1. crsp.msp500list      -> S&P 500 membership intervals per PERMNO
2. crsp.stocknames      -> PERMNO -> ticker / SIC / company name with intervals
3. crsp.dsf             -> Daily Stock File (prc, ret, retx, vol, shrout, cfacpr, cfacshr)
                          batched at BATCH=200 PERMNOs per query
4. crsp.dsedelist       -> Delisting events (DLSTDT, DLSTCD, DLRET, DLRETX)

OUTPUT
------
    ./crsp_sp500_full/
        constituents.csv        ~2000 rows  (PERMNO, start, ending)
        stocknames.csv          ~5000 rows  (PERMNO, ticker, comnam, namedt, nameenddt, siccd)
        dsf_batch_000.parquet ..dsf_batch_NNN.parquet   ~200-400 MB total
        delisting.csv           ~500 rows   (PERMNO, dlstdt, dlstcd, dlret, dlretx)

DELIVERY
--------
Zip the crsp_sp500_full/ directory and send (Google Drive / WeTransfer / etc.).
Expected wall time on a normal WRDS connection: 30-90 minutes.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

OUT_DIR = Path("crsp_sp500_full")
DATE_START_DEFAULT = "1985-01-01"  # covers 1987 Black Monday, LTCM, dot-com, GFC, COVID
DATE_END_DEFAULT = "2025-12-31"
BATCH_PERMNOS = 200  # tune down if your WRDS tier caps row count per query


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", default=os.environ.get("WRDS_USERNAME"),
                        help="WRDS username (or set WRDS_USERNAME env var)")
    parser.add_argument("--start-date", default=DATE_START_DEFAULT,
                        help=f"Panel start date (default {DATE_START_DEFAULT}). "
                             f"Earliest sane: 1957-03-04 (S&P 500 inception). "
                             f"CRSP DSF coverage starts 1925-07-01.")
    parser.add_argument("--end-date", default=DATE_END_DEFAULT,
                        help=f"Panel end date (default {DATE_END_DEFAULT}).")
    parser.add_argument("--batch", type=int, default=BATCH_PERMNOS,
                        help="PERMNOs per DSF query batch (default 200)")
    args = parser.parse_args()

    if not args.username:
        parser.error("--username required (or set WRDS_USERNAME)")

    try:
        import wrds
        import pandas as pd
    except ImportError as e:
        sys.exit(f"Missing dependency: {e}. Run: pip install wrds pandas pyarrow")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Connecting to WRDS as {args.username} ...")
    db = wrds.Connection(wrds_username=args.username)

    # 1. S&P 500 historical constituent list
    print(f"\n[1/4] crsp.msp500list  ({args.start_date} .. {args.end_date}) ...")
    constituents = db.raw_sql(f"""
        SELECT permno, start, ending
        FROM crsp.msp500list
        WHERE start <= '{args.end_date}' AND ending >= '{args.start_date}'
        ORDER BY permno, start
    """)
    constituents.to_csv(OUT_DIR / "constituents.csv", index=False)
    permnos = sorted(set(int(p) for p in constituents["permno"].dropna()))
    print(f"  -> {len(constituents)} membership intervals, {len(permnos)} unique PERMNOs")

    # 2. Stocknames mapping (PERMNO -> ticker history, with namedt/nameenddt intervals)
    print(f"\n[2/4] crsp.stocknames ...")
    placeholders = ",".join(str(p) for p in permnos)
    stocknames = db.raw_sql(f"""
        SELECT permno, ticker, comnam, namedt, nameenddt, siccd
        FROM crsp.stocknames
        WHERE permno IN ({placeholders})
        ORDER BY permno, namedt
    """)
    stocknames.to_csv(OUT_DIR / "stocknames.csv", index=False)
    print(f"  -> {len(stocknames)} ticker-history rows")

    # 3. Daily Stock File, batched
    print(f"\n[3/4] crsp.dsf (batched at {args.batch} PERMNOs per query) ...")
    n_batches = (len(permnos) + args.batch - 1) // args.batch
    total_rows = 0
    t0 = time.time()
    for i, start in enumerate(range(0, len(permnos), args.batch)):
        batch = permnos[start:start + args.batch]
        batch_placeholders = ",".join(str(p) for p in batch)
        df = db.raw_sql(f"""
            SELECT permno, date, ret, retx, prc, openprc, askhi, bidlo,
                   vol, shrout, cfacpr, cfacshr
            FROM crsp.dsf
            WHERE permno IN ({batch_placeholders})
              AND date BETWEEN '{args.start_date}' AND '{args.end_date}'
            ORDER BY permno, date
        """)
        out = OUT_DIR / f"dsf_batch_{i:03d}.parquet"
        df.to_parquet(out, compression="snappy")
        total_rows += len(df)
        elapsed = time.time() - t0
        print(f"  batch {i+1}/{n_batches}: {len(batch):>3} PERMNOs -> {len(df):>7} rows "
              f"({out.name}, total {total_rows:,} rows, elapsed {elapsed:.0f}s)")

    # 4. Delisting events
    print(f"\n[4/4] crsp.dsedelist ...")
    delist = db.raw_sql(f"""
        SELECT permno, dlstdt, dlstcd, dlret, dlretx
        FROM crsp.dsedelist
        WHERE permno IN ({placeholders})
    """)
    delist.to_csv(OUT_DIR / "delisting.csv", index=False)
    n_with_return = delist["dlret"].notna().sum()
    print(f"  -> {len(delist)} delisting events, {n_with_return} with non-null DLRET")

    db.close()

    print(f"\nDone. Total elapsed {(time.time()-t0)/60:.1f} min.")
    print(f"Output directory: {OUT_DIR.absolute()}")
    print(f"  - constituents.csv  ({len(constituents)} rows)")
    print(f"  - stocknames.csv    ({len(stocknames)} rows)")
    print(f"  - {n_batches} dsf_batch_*.parquet  ({total_rows:,} rows total)")
    print(f"  - delisting.csv     ({len(delist)} rows)")
    print(f"\nZip and send.")


if __name__ == "__main__":
    main()
