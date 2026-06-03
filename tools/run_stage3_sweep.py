"""CLI driver for the Stage-3 sensitivity sweep (appendix F3 / app:sens).

Usage::

    python -m tools.run_stage3_sweep                     # default 14-cell grid
    python -m tools.run_stage3_sweep --force             # ignore cache
    python -m tools.run_stage3_sweep --csv out.csv       # also write long-form CSV
    python -m tools.run_stage3_sweep --headline-csv h.csv

The sweep reuses ``src.stage3_direction.sensitivity_sweep.run_sweep``
and writes its cache to ``SNAPSHOTS_DIR / stage3_sensitivity.pkl``.
"""
import argparse
import pickle
from pathlib import Path

import pandas as pd

from src.config import SNAPSHOTS_DIR
from src.stage3_direction.sensitivity_sweep import (
    run_sweep, to_dataframe, headline_table,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true",
                    help="Ignore cached sweep results and recompute.")
    ap.add_argument("--csv", type=Path, default=None,
                    help="Optional path for the long-form per-cell-snapshot CSV.")
    ap.add_argument("--headline-csv", type=Path, default=None,
                    help="Optional path for the headline crisis/non-crisis "
                    "mean mutual-fraction table.")
    args = ap.parse_args()

    stage2_path = SNAPSHOTS_DIR / "stage2_results.pkl"
    if not stage2_path.exists():
        ap.error(f"missing {stage2_path}; run Stage 2 first")

    with open(stage2_path, "rb") as f:
        stage2 = pickle.load(f)
    returns_path = Path(__file__).resolve().parent.parent / "data" / "sp500_returns.parquet"
    if not returns_path.exists():
        ap.error(f"missing {returns_path}; run Stage 1 download first")
    returns = pd.read_parquet(returns_path)

    sweep = run_sweep(stage2, returns, force=args.force)
    headline = headline_table(sweep)
    print("\nHeadline (crisis vs non-crisis mean mutual fraction):")
    print(headline.to_string(index=False))

    if args.csv is not None:
        long_df = to_dataframe(sweep)
        long_df.to_csv(args.csv, index=False)
        print(f"\nLong-form CSV written to {args.csv}")
    if args.headline_csv is not None:
        headline.to_csv(args.headline_csv, index=False)
        print(f"Headline CSV written to {args.headline_csv}")


if __name__ == "__main__":
    main()
