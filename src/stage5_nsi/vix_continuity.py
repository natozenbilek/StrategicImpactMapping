"""VIX-VXO continuity series (Whaley 2009 splice).

CBOE changed the VIX calculation methodology on 2003-09-22 from S&P 100
implied volatility (old VIX, now relabelled VXO) to S&P 500 near-the-
money implied variance (new VIX). For panels that span the 1987 Black
Monday window (VIX has no pre-1990 coverage), the Whaley (2009) splice
concatenates VXO[..2003-09-22) with VIX[2003-09-22..], using the new
VIX value on the methodology-change day itself.

Source files
------------
* data/crsp/fred_vxo.csv  - FRED VXOCLS, 1986-01-02 .. 2021-09-23
* data/crsp/fred_vix.csv  - FRED VIXCLS, 1990-01-02 .. 2026-05-26

Splice gap: VXO last day 2021-09-23 sits well after VIX coverage starts
(1990) and the splice cutoff (2003-09-22), so no continuity hole opens.
"""
import pandas as pd

from src.config import DATA_DIR

CRSP_DIR = DATA_DIR / "crsp"
VXO_PATH = CRSP_DIR / "fred_vxo.csv"
VIX_PATH = CRSP_DIR / "fred_vix.csv"
CONTINUITY_PATH = DATA_DIR / "vix_continuity.parquet"

SPLICE_DATE = pd.Timestamp("2003-09-22")


def _load_fred_series(path, value_col) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
    assert value_col in df.columns, f"{path.name}: missing column {value_col}"
    s = df[value_col].astype(float).dropna()
    assert s.index.is_monotonic_increasing, f"{path.name} not date-sorted"
    return s


def build_vix_continuity(force: bool = False) -> pd.Series:
    """Construct VXO[..2003-09-22) + VIX[2003-09-22..] continuity Series."""
    if not force and CONTINUITY_PATH.exists():
        df = pd.read_parquet(CONTINUITY_PATH)
        s = df["Close"]
        s.index = pd.DatetimeIndex(s.index)
        return s

    vxo = _load_fred_series(VXO_PATH, "VXOCLS")
    vix = _load_fred_series(VIX_PATH, "VIXCLS")

    pre = vxo.loc[vxo.index < SPLICE_DATE]
    post = vix.loc[vix.index >= SPLICE_DATE]
    assert len(pre) > 0 and len(post) > 0, "empty splice segment"

    cont = pd.concat([pre, post]).sort_index()
    cont.name = "Close"
    assert not cont.index.duplicated().any(), "duplicate dates after splice"

    peak = pd.Timestamp("1987-10-19")
    if peak in cont.index:
        v = float(cont.loc[peak])
        assert 140 < v < 160, f"1987-10-19 VXO peak {v:.2f} outside [140, 160]"

    n_pre = (cont.index < SPLICE_DATE).sum()
    n_post = (cont.index >= SPLICE_DATE).sum()
    print(f"[VIX continuity] VXO pre-splice: {n_pre} | VIX post-splice: {n_post}")
    print(f"  range: {cont.index.min().date()} .. {cont.index.max().date()}")
    print(f"  1987-10-19 peak: {cont.get(peak, float('nan')):.2f}")

    cont.to_frame("Close").to_parquet(CONTINUITY_PATH)
    return cont


if __name__ == "__main__":
    s = build_vix_continuity(force=True)
    print(f"\ncontinuity series: {len(s)} obs, "
          f"min={s.min():.2f}, max={s.max():.2f}, mean={s.mean():.2f}")
