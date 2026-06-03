"""LSEG Datastream pull: S&P 500 historical panel 1995-2025.

For the friend with LSEG Datastream Web Service (DSWS) credentials.
Companion to tools/fetch_crsp_sp500_full.py; together the two outputs
let us cross-validate every panel value against an independent
commercial source (CRSP + Datastream RI).

REQUIRES
--------
LSEG Datastream Web Service (DSWS) access. Two Python libraries can hit it:

    pip install DatastreamPy        (legacy, simple, well-documented)
        OR
    pip install lseg-data           (current LSEG Data Library)

USAGE
-----
    export DSWS_USERNAME=<your_dsws_user>
    export DSWS_PASSWORD=<your_dsws_pass>
    python fetch_lseg_sp500_full.py

DATA PULLED
-----------
1. LS&PCOMP constituent history       -> monthly snapshots of S&P 500 membership
                                         (~1500-2000 unique securities over 30 years)
2. Per-constituent daily Return Index -> RI field (total return, splits + dividends adjusted)
3. Per-constituent daily Price        -> P field (raw close)
4. Per-constituent daily Volume       -> VO field (thousands)
5. Per-constituent market value       -> MV field (market cap, useful for ADV channel)

Output: ./lseg_sp500_full/*.parquet  (~200-400 MB compressed)

NOTES
-----
- Datastream codes for delisted securities use the @DEAD suffix
  (e.g. "LEHMAN BROS HDG@DEAD" for Lehman, "BEAR STEARNS COS@DEAD" for Bear).
- The RI field on the constituent's @DEAD code carries the full price + delisting
  return history; equivalent to CRSP's RET + DLRET fold-in.
- DSWS has per-request limits (typically 50 securities x 5 fields x ~12,000
  data points per call). The script batches accordingly.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

OUT_DIR = Path("lseg_sp500_full")
DATE_START = "1995-01-01"
DATE_END = "2025-12-31"
BATCH_SIZE = 50  # DSWS soft limit on securities per request


def main() -> None:
    user = os.environ.get("DSWS_USERNAME")
    pwd = os.environ.get("DSWS_PASSWORD")
    if not (user and pwd):
        sys.exit("Set DSWS_USERNAME and DSWS_PASSWORD env vars first.")

    # Prefer DatastreamPy (simpler API); fall back to lseg-data if absent.
    try:
        import DatastreamPy as ds
        client = ds.DataClient(None, user, pwd)
        api = "DatastreamPy"
    except ImportError:
        try:
            import lseg.data as ld
            ld.open_session()
            api = "lseg-data"
        except ImportError:
            sys.exit("Install one of: pip install DatastreamPy  OR  pip install lseg-data")

    import pandas as pd

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Using {api} API")

    # ----------------------------------------------------------------
    # Step 1: S&P 500 historical constituent list
    # ----------------------------------------------------------------
    # Datastream code for S&P 500: LS&PCOMP
    # The "L#" prefix on a list name returns the historical constituent
    # set; combined with a date range, DSWS returns membership intervals.
    print(f"\n[1/2] LS&PCOMP historical constituents {DATE_START} .. {DATE_END} ...")

    if api == "DatastreamPy":
        # request the historical list at monthly snapshots
        const = client.get_data(
            tickers="LS&PCOMP",
            fields=["LCNST"],          # List of CoNSTituents
            start=DATE_START,
            end=DATE_END,
            freq="M",
        )
    else:
        const = ld.get_data(
            universe="LS&PCOMP",
            fields=["LCNST"],
            parameters={"SDate": DATE_START, "EDate": DATE_END, "Frq": "M"},
        )

    const.to_csv(OUT_DIR / "constituents_monthly.csv", index=False)
    # Flatten to unique-security set
    if "LCNST" in const.columns:
        all_ids = set()
        for cell in const["LCNST"].dropna():
            # LCNST cell is typically a comma- or pipe-delimited list of codes
            tokens = [t.strip() for t in str(cell).replace("|", ",").split(",") if t.strip()]
            all_ids.update(tokens)
        unique_codes = sorted(all_ids)
    else:
        sys.exit("LCNST field not returned; check DSWS access tier.")

    print(f"  -> {len(unique_codes)} unique Datastream codes ever-member of S&P 500")

    pd.DataFrame({"datastream_code": unique_codes}).to_csv(
        OUT_DIR / "unique_codes.csv", index=False)

    # ----------------------------------------------------------------
    # Step 2: Daily RI + P + VO + MV per constituent, batched
    # ----------------------------------------------------------------
    print(f"\n[2/2] Daily time series (RI, P, VO, MV), batched at {BATCH_SIZE} codes ...")

    fields = ["RI", "P", "VO", "MV"]
    n_batches = (len(unique_codes) + BATCH_SIZE - 1) // BATCH_SIZE
    t0 = time.time()
    total_rows = 0

    for i, start in enumerate(range(0, len(unique_codes), BATCH_SIZE)):
        batch = unique_codes[start:start + BATCH_SIZE]
        if api == "DatastreamPy":
            df = client.get_data(
                tickers=batch,
                fields=fields,
                start=DATE_START,
                end=DATE_END,
                freq="D",
            )
        else:
            df = ld.get_data(
                universe=batch,
                fields=fields,
                parameters={"SDate": DATE_START, "EDate": DATE_END, "Frq": "D"},
            )

        out = OUT_DIR / f"daily_batch_{i:03d}.parquet"
        df.to_parquet(out, compression="snappy")
        total_rows += len(df)
        elapsed = time.time() - t0
        print(f"  batch {i+1}/{n_batches}: {len(batch):>3} codes -> {len(df):>7} rows "
              f"({out.name}, total {total_rows:,}, elapsed {elapsed:.0f}s)")
        # Be polite to the API
        time.sleep(0.5)

    print(f"\nDone. Total elapsed {(time.time()-t0)/60:.1f} min.")
    print(f"Output directory: {OUT_DIR.absolute()}")
    print(f"  - constituents_monthly.csv  (monthly LS&PCOMP membership)")
    print(f"  - unique_codes.csv          ({len(unique_codes)} unique codes)")
    print(f"  - {n_batches} daily_batch_*.parquet  ({total_rows:,} rows)")
    print(f"\nZip and send.")


if __name__ == "__main__":
    main()
