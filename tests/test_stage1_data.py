"""
Unit tests for the pre-Stage-1 Yahoo data-construction helpers.

DEPRECATED 2026-05-28: every test in this module exercises the Yahoo
download path (TICKER_CUTOFFS, _detect_contaminated, clean_prices,
download_prices, run_download, compute_adv, apply_data_cleanup) which
the CRSP migration replaced with src.stage1_data.crsp_panel. The Yahoo
helpers in src.stage1_data.download stay in the tree for pre-migration
reproducibility only; their tests no longer represent live behaviour and
are skipped at collection time. A CRSP-equivalent suite belongs to Task
[8] (V1+V2+V3 review on the new cache).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.skip(
    reason="Yahoo-path tests deprecated post 2026-05-28 CRSP migration; "
           "CRSP equivalents pending Task [8]."
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.stage1_data.download import (
    TICKER_CUTOFFS,
    _apply_ticker_cutoffs,
    _detect_contaminated,
    clean_prices,
    compute_log_returns,
    download_prices,
    run_download,
)
from tools.download_volume import compute_adv
from tools import apply_data_cleanup


# --- Fixtures --------------------------------------------------------

def _synthetic_prices(T: int = 252, k: int = 5, seed: int = 42) -> pd.DataFrame:
    """Geometric-Brownian-motion price panel, business-day-indexed."""
    rng = np.random.default_rng(seed)
    log_p = np.cumsum(rng.normal(0.0005, 0.012, size=(T, k)), axis=0) + 4.6
    idx = pd.bdate_range("2020-01-01", periods=T)
    cols = [f"TKR{i}" for i in range(k)]
    return pd.DataFrame(np.exp(log_p), index=idx, columns=cols)


# --- _apply_ticker_cutoffs -------------------------------------------

def test_cutoffs_mask_pre_cutoff_only():
    """All observations strictly before the cutoff date become NaN;
    on-and-after dates and other tickers remain identical."""
    df = _synthetic_prices(T=252, k=3)
    cutoff_idx = 100
    cutoff_date = df.index[cutoff_idx]
    out = _apply_ticker_cutoffs(
        df, cutoffs={"TKR0": cutoff_date.strftime("%Y-%m-%d")}
    )
    # TKR0 strictly before cutoff: all NaN
    assert out.iloc[:cutoff_idx]["TKR0"].isna().all()
    # TKR0 from cutoff onward: untouched
    np.testing.assert_array_equal(
        out.iloc[cutoff_idx:]["TKR0"].values,
        df.iloc[cutoff_idx:]["TKR0"].values,
    )
    # Other tickers untouched in full
    pd.testing.assert_frame_equal(out[["TKR1", "TKR2"]], df[["TKR1", "TKR2"]])


def test_cutoffs_ignore_missing_ticker():
    """A cutoff entry for a ticker not in the panel is a silent no-op."""
    df = _synthetic_prices(T=100, k=2)
    out = _apply_ticker_cutoffs(df, cutoffs={"NOTHERE": "2020-03-01"})
    pd.testing.assert_frame_equal(out, df)


def test_cutoffs_do_not_mutate_input():
    """The input frame must not be modified in place."""
    df = _synthetic_prices(T=50, k=2)
    df_before = df.copy()
    _ = _apply_ticker_cutoffs(df, cutoffs={"TKR0": "2020-01-15"})
    pd.testing.assert_frame_equal(df, df_before)


# --- _detect_contaminated --------------------------------------------

def test_detect_contaminated_flags_high_zero_frac():
    """A series with >20% zero-return days is flagged."""
    rng = np.random.default_rng(0)
    n = 500
    returns = rng.normal(0, 0.01, size=n)
    # Force 30% of returns to be exactly zero (above the 20% threshold)
    returns[rng.choice(n, size=int(0.30 * n), replace=False)] = 0.0
    df = pd.DataFrame({"BAD": returns})
    bad = _detect_contaminated(df)
    assert len(bad) == 1
    assert bad[0][0] == "BAD"
    assert bad[0][1] > 0.20


def test_detect_contaminated_flags_long_flat_run():
    """A 25-day flat run trips the detector even when the overall
    zero-fraction is well below the 20% threshold."""
    rng = np.random.default_rng(1)
    n = 500
    returns = rng.normal(0, 0.01, size=n)
    returns[100:125] = 0.0  # 25-day flat run; overall zero_frac ~5%
    df = pd.DataFrame({"FLAT": returns})
    bad = _detect_contaminated(df)
    assert len(bad) == 1
    assert bad[0][0] == "FLAT"
    assert bad[0][2] >= 25


def test_detect_contaminated_quiet_on_clean_series():
    """A normal Gaussian return series is not flagged."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({"GOOD": rng.normal(0, 0.01, size=500)})
    assert _detect_contaminated(df) == []


def test_detect_contaminated_skips_short_series():
    """Series with <20 non-NaN observations are skipped silently."""
    df = pd.DataFrame({"SHORT": [0.0, 0.0, 0.0, np.nan, np.nan]})
    assert _detect_contaminated(df) == []


# --- clean_prices ----------------------------------------------------

def test_clean_prices_preserves_unbalanced_grid():
    """Per-ticker partial coverage is retained: no global dropna."""
    df = _synthetic_prices(T=200, k=3)
    df.iloc[:50, 0] = np.nan  # TKR0 starts 50 days late but still > 5%
    out = clean_prices(df, min_obs_frac=0.05)
    assert "TKR0" in out.columns
    assert out.shape[0] == 200  # row count preserved


def test_clean_prices_drops_below_coverage_threshold():
    """A ticker below ``min_obs_frac`` is dropped; sibling tickers stay."""
    df = _synthetic_prices(T=2000, k=3)
    df.iloc[:1980, 0] = np.nan  # TKR0 has 20 obs (<100 = 5% of 2000)
    out = clean_prices(df, min_obs_frac=0.05)
    assert "TKR0" not in out.columns
    assert {"TKR1", "TKR2"}.issubset(set(out.columns))


def test_clean_prices_ffill_is_bounded_to_five_days():
    """A 7-day gap is ffilled for the first five days only; the tail
    two days remain NaN so genuine multi-week delistings stay visible."""
    df = _synthetic_prices(T=200, k=2)
    last_valid = df.iloc[19, 0]
    df.iloc[20:27, 0] = np.nan  # 7-day gap at rows 20..26
    out = clean_prices(df, min_obs_frac=0.05)
    # Rows 20..24 are within the 5-day ffill budget; rows 25, 26 are not.
    for i in range(20, 25):
        assert out.iloc[i, 0] == pytest.approx(last_valid)
    assert pd.isna(out.iloc[25, 0])
    assert pd.isna(out.iloc[26, 0])


# --- compute_log_returns ---------------------------------------------

def test_compute_log_returns_drops_first_row():
    """The first calendar row has no lagged price, so it is dropped."""
    df = _synthetic_prices(T=100, k=3)
    ret = compute_log_returns(df)
    assert ret.shape[0] == df.shape[0] - 1
    assert ret.index[0] == df.index[1]


def test_compute_log_returns_matches_elementwise_log_diff():
    """Reference: ``log(p_t / p_{t-1})``."""
    df = _synthetic_prices(T=100, k=2)
    ret = compute_log_returns(df)
    expected = (np.log(df["TKR0"]) - np.log(df["TKR0"].shift(1))).iloc[1:]
    pd.testing.assert_series_equal(ret["TKR0"], expected, check_names=False)


def test_compute_log_returns_drops_all_nan_rows():
    """An all-NaN row in returns (e.g.\\ a market holiday with no prices
    at all) is dropped so downstream stages see only trading days."""
    df = _synthetic_prices(T=100, k=2)
    df.iloc[50:55, :] = np.nan
    ret = compute_log_returns(df)
    # The function's contract is dropna(how='all'); verify no row is
    # entirely NaN after the diff.
    assert not ret.isna().all(axis=1).any()


# --- compute_adv -----------------------------------------------------

def test_compute_adv_product_of_inputs():
    """ADV = mean(close * volume) on the intersection of indices and
    columns. With constant inputs the mean is the product itself."""
    idx = pd.bdate_range("2020-01-01", periods=10)
    close = pd.DataFrame({"A": [10.0] * 10, "B": [20.0] * 10}, index=idx)
    volume = pd.DataFrame({"A": [1e6] * 10, "B": [2e6] * 10}, index=idx)
    adv = compute_adv(close, volume)
    assert adv["A"] == pytest.approx(10.0 * 1e6)
    assert adv["B"] == pytest.approx(20.0 * 2e6)


def test_compute_adv_skips_pairwise_nan():
    """Missing entries are dropped pairwise; the per-ticker mean is the
    mean over the surviving (close, volume) pairs."""
    idx = pd.bdate_range("2020-01-01", periods=10)
    close = pd.DataFrame({"A": [10.0] * 5 + [np.nan] * 5}, index=idx)
    volume = pd.DataFrame({"A": [1e6] * 10}, index=idx)
    adv = compute_adv(close, volume)
    assert adv["A"] == pytest.approx(10.0 * 1e6)


def test_compute_adv_restricts_to_common_columns():
    """A ticker present in only one of the two frames is dropped from
    the output."""
    idx = pd.bdate_range("2020-01-01", periods=5)
    close = pd.DataFrame({"A": [10.0] * 5, "B": [20.0] * 5}, index=idx)
    volume = pd.DataFrame({"A": [1e6] * 5}, index=idx)
    adv = compute_adv(close, volume)
    assert list(adv.index) == ["A"]


def test_compute_adv_skips_volume_nan():
    """Pairwise NaN skip applies on the volume side too; the per-ticker
    mean is over the surviving (close, volume) pairs."""
    idx = pd.bdate_range("2020-01-01", periods=10)
    close = pd.DataFrame({"A": [10.0] * 10}, index=idx)
    volume = pd.DataFrame({"A": [1e6] * 6 + [np.nan] * 4}, index=idx)
    adv = compute_adv(close, volume)
    assert adv["A"] == pytest.approx(10.0 * 1e6)


# --- cross-module integrity ------------------------------------------

def test_ticker_cutoffs_match_across_modules():
    """The two pre-Stage-1 entry points must apply the same per-ticker
    cutoffs; a divergence would silently mask a different sub-window
    on one path."""
    assert apply_data_cleanup.TICKER_CUTOFFS == TICKER_CUTOFFS


# --- run_download contamination raise --------------------------------

def test_run_download_raises_on_contamination(tmp_path, monkeypatch):
    """run_download must abort with RuntimeError if the post-cleanup
    detector flags a ticker (>20% zero-return or >20-day flatline)."""
    import src.stage1_data.download as dl

    # Redirect cache dir so the test does not touch real artefacts.
    monkeypatch.setattr(dl, "DATA_DIR", tmp_path)
    monkeypatch.setattr(dl, "HISTORICAL_TICKERS", [])

    # Fabricate a 2-ticker prices frame: TKR_OK is GBM, TKR_BAD is a
    # flatline that will exceed both detector thresholds after diff.
    T = 300
    rng = np.random.default_rng(0)
    log_p = np.cumsum(rng.normal(0.0005, 0.012, size=T)) + 4.6
    idx = pd.bdate_range("2020-01-01", periods=T)
    prices = pd.DataFrame({"TKR_OK": np.exp(log_p),
                           "TKR_BAD": [100.0] * T}, index=idx)

    monkeypatch.setattr(dl, "download_prices", lambda tickers: prices)
    monkeypatch.setattr(dl, "get_sp500_tickers",
                        lambda: pd.DataFrame([
                            ("TKR_OK",  "ok",  "Materials", "Materials"),
                            ("TKR_BAD", "bad", "Materials", "Materials"),
                        ], columns=["Symbol", "Security",
                                    "GICS Sector", "GICS Sub-Industry"]))

    with pytest.raises(RuntimeError, match="Contamination detector"):
        dl.run_download(force=True)


# --- download_prices batch-failure guard -----------------------------

def test_download_prices_aborts_on_high_failure_rate(monkeypatch):
    """download_prices must raise when the batch-failure fraction
    exceeds the guard (default 20%) so a partial Yahoo outage cannot
    pass a tiny panel through to downstream stages."""
    import src.stage1_data.download as dl

    call_count = {"i": 0}

    def fake_yf_download(symbols, start, end, auto_adjust, progress):
        call_count["i"] += 1
        # Fail every other batch -> 50% failure rate, above the 20% guard.
        if call_count["i"] % 2 == 0:
            raise RuntimeError("simulated Yahoo outage")
        tickers = symbols.split()
        idx = pd.bdate_range(start, periods=20)
        data = pd.DataFrame({t: [100.0] * 20 for t in tickers}, index=idx)
        data.columns = pd.MultiIndex.from_product([["Close"], tickers])
        return data

    monkeypatch.setattr(dl.yf, "download", fake_yf_download)

    tickers = [f"TKR{i:03d}" for i in range(200)]  # 4 batches of 50
    with pytest.raises(RuntimeError, match="batch-failure"):
        dl.download_prices(tickers, batch_size=50)


# --- download_prices singleton retry --------------------------------

def test_download_prices_retry_singleton_recovers_empty_column(monkeypatch):
    """When a batch returns an all-NaN column for some ticker, the
    singleton retry path must re-fetch that ticker on its own and
    populate the column. Guards against the yfinance 1.2.x regression
    where a multi-ticker batch silently delivers an empty column."""
    import src.stage1_data.download as dl

    state = {"batch_calls": 0, "singleton_calls": 0}

    def fake_yf_download(symbols, start, end, auto_adjust, progress):
        tickers = symbols.split()
        idx = pd.bdate_range(start, periods=20)
        if len(tickers) == 1:
            # Singleton retry path: return clean data.
            state["singleton_calls"] += 1
            t = tickers[0]
            data = pd.DataFrame({t: [50.0] * 20}, index=idx)
            data.columns = pd.MultiIndex.from_product([["Close"], [t]])
            return data
        # Batch path: TKR_BAD comes back as all NaN; others fine.
        state["batch_calls"] += 1
        data = {}
        for t in tickers:
            data[t] = [np.nan] * 20 if t == "TKR_BAD" else [100.0] * 20
        df = pd.DataFrame(data, index=idx)
        df.columns = pd.MultiIndex.from_product([["Close"], tickers])
        return df

    monkeypatch.setattr(dl.yf, "download", fake_yf_download)
    out = dl.download_prices(["TKR_OK1", "TKR_BAD", "TKR_OK2"],
                             batch_size=50)
    # The singleton retry filled TKR_BAD; the batch alone would have
    # left it all-NaN.
    assert not out["TKR_BAD"].isna().all()
    assert state["singleton_calls"] == 1


# --- clean_prices coverage threshold (no floor) ----------------------

def test_clean_prices_uses_only_fractional_threshold():
    """After the audit, clean_prices no longer applies a 50-day floor
    in addition to the fractional rule: min_obs is exactly
    int(n_days * min_obs_frac). Verify with a short panel where 5% is
    well below 50 obs so the old max(50, ...) would have changed the
    answer."""
    df = _synthetic_prices(T=200, k=3)
    # TKR0 has 15 valid obs; 5% of 200 = 10, so it must survive under
    # the fractional-only rule (it would have been dropped under the
    # old max(50, 10) = 50 floor).
    df.iloc[15:, 0] = np.nan
    out = clean_prices(df, min_obs_frac=0.05)
    assert "TKR0" in out.columns


# --- compute_log_returns ffill interaction ---------------------------

def test_compute_log_returns_ffill_run_below_detector_threshold():
    """A 5-day ffill on a single gap must produce no more than 5
    consecutive zero log-returns and must keep the overall zero-return
    fraction well below the contamination-detector cut-off (20%). This
    pins the disclosure in appendix app:data rule 2."""
    df = _synthetic_prices(T=1000, k=1)
    df.iloc[500:505, 0] = np.nan  # 5-day gap inside the sample
    cleaned = clean_prices(df, min_obs_frac=0.05)
    returns = compute_log_returns(cleaned)
    # The ffill fills the gap forward at the pre-gap price; the diff of
    # equal-valued consecutive entries gives zero log-returns.
    r = returns["TKR0"].dropna()
    zero_frac = float((r == 0).sum()) / len(r)
    assert zero_frac < 0.20  # well below detector threshold
    max_run = cur = 0
    for v in (r == 0):
        cur = cur + 1 if v else 0
        max_run = max(max_run, cur)
    assert max_run <= 5  # ffill is bounded to 5 days


# --- get_sp500_tickers size sanity / no fallback ---------------------

def test_get_sp500_tickers_aborts_on_degenerate_response(monkeypatch):
    """A Wikipedia response with a suspiciously small count must abort,
    not silently degrade. Guards against the silent-fallback regression
    that the audit removed."""
    import src.stage1_data.download as dl

    fake_html = (
        "<html><body><table>"
        "<tr><th>Symbol</th><th>Security</th>"
        "<th>GICS Sector</th><th>GICS Sub-Industry</th></tr>"
        + "".join(
            f"<tr><td>TKR{i}</td><td>Name{i}</td>"
            f"<td>Information Technology</td><td>Sub{i}</td></tr>"
            for i in range(60)  # 60 rows < 480 sanity lower bound
        )
        + "</table></body></html>"
    )

    class FakeResp:
        text = fake_html

        def raise_for_status(self):
            pass

    monkeypatch.setattr(
        "requests.get", lambda *a, **kw: FakeResp()
    )
    with pytest.raises(RuntimeError, match="sanity band"):
        dl.get_sp500_tickers()


def test_get_sp500_tickers_propagates_network_error(monkeypatch):
    """A network failure must propagate as an exception rather than
    silently returning a built-in mega-cap list. The fallback was
    removed in the audit; if anyone re-introduces it, this test
    catches the regression."""
    import src.stage1_data.download as dl

    def boom(*a, **kw):
        raise RuntimeError("simulated network outage")

    monkeypatch.setattr("requests.get", boom)
    with pytest.raises(RuntimeError):
        dl.get_sp500_tickers()


# --- HISTORICAL_TICKERS integrity ------------------------------------

def test_historical_tickers_no_duplicates_and_valid_sectors():
    """HISTORICAL_TICKERS must have unique symbols and reference only
    GICS sectors that exist in src.config.GICS_SECTORS."""
    from src.stage1_data.download import HISTORICAL_TICKERS
    from src.config import GICS_SECTORS

    symbols = [t[0] for t in HISTORICAL_TICKERS]
    assert len(symbols) == len(set(symbols)), \
        f"HISTORICAL_TICKERS has duplicate symbols: {symbols}"
    valid_sectors = set(GICS_SECTORS.values())
    for sym, _name, sector in HISTORICAL_TICKERS:
        assert sector in valid_sectors, \
            f"{sym} has unknown GICS sector {sector!r}"


# --- apply_data_cleanup.cleanup_adv regression guard -----------------

def test_cleanup_adv_does_not_recompute():
    """cleanup_adv must only restrict rows; it must NOT recompute ADV
    values from sp500_prices.parquet (the dividend-adjusted close)
    times sp500_volume.parquet (the split-adjusted volume), which was
    the pre-audit bug. Verify by checking that the per-ticker values
    for surviving rows are byte-identical to the input."""
    adv_in = pd.DataFrame(
        {"mean_dollar_volume": [1e8, 2e8, 3e8, 4e8]},
        index=["A", "B", "C", "D"],
    )
    surviving = {"A", "C", "D"}
    out = apply_data_cleanup.cleanup_adv(adv_in, surviving)
    assert set(out.index) == surviving
    # Values for surviving rows must be unchanged (no recomputation).
    for t in surviving:
        assert out.loc[t, "mean_dollar_volume"] == \
               adv_in.loc[t, "mean_dollar_volume"]


# --- cross-cache integrity smoke test (skipped if no real cache) -----

def test_cache_files_share_ticker_set():
    """sp500_prices.parquet, sp500_volume.parquet, and sp500_adv.parquet
    must reference the same ticker set; a mismatch means a refresh of
    one cache without the others (the BNY/VEEV vs BK/CTRA drift caught
    by the audit). Skips silently when caches are absent so the test
    suite stays runnable in CI."""
    from src.config import DATA_DIR

    p_path = DATA_DIR / "sp500_prices.parquet"
    v_path = DATA_DIR / "sp500_volume.parquet"
    a_path = DATA_DIR / "sp500_adv.parquet"
    if not (p_path.exists() and v_path.exists() and a_path.exists()):
        pytest.skip("cache files not present; integrity check skipped")

    prices = pd.read_parquet(p_path)
    volume = pd.read_parquet(v_path)
    adv = pd.read_parquet(a_path)
    assert set(prices.columns) == set(volume.columns), (
        f"prices vs volume mismatch: "
        f"only-in-prices={sorted(set(prices.columns) - set(volume.columns))}, "
        f"only-in-volume={sorted(set(volume.columns) - set(prices.columns))}"
    )
    assert set(prices.columns) == set(adv.index), (
        f"prices vs adv mismatch: "
        f"only-in-prices={sorted(set(prices.columns) - set(adv.index))}, "
        f"only-in-adv={sorted(set(adv.index) - set(prices.columns))}"
    )


# --- apply_data_cleanup.cleanup_prices / cleanup_volume --------------

def test_cleanup_prices_drops_cfc_sbny_and_masks_sw_amcr():
    """Drop CFC/SBNY entirely; mask SW pre-2024-07-15 and AMCR
    pre-2019-06-11 with NaN. Other tickers untouched."""
    idx = pd.bdate_range("2018-01-01", periods=2000)
    df = pd.DataFrame({
        "AAA":  100.0,
        "CFC":  10.0,
        "SBNY": 20.0,
        "SW":   30.0,
        "AMCR": 40.0,
    }, index=idx)
    out = apply_data_cleanup.cleanup_prices(df)

    assert "CFC" not in out.columns
    assert "SBNY" not in out.columns
    for col in ("AAA", "SW", "AMCR"):
        assert col in out.columns

    sw_cutoff = pd.Timestamp("2024-07-15")
    assert out.loc[out.index < sw_cutoff, "SW"].isna().all()
    assert not out.loc[out.index >= sw_cutoff, "SW"].isna().any()

    amcr_cutoff = pd.Timestamp("2019-06-11")
    assert out.loc[out.index < amcr_cutoff, "AMCR"].isna().all()
    assert not out.loc[out.index >= amcr_cutoff, "AMCR"].isna().any()

    # AAA never had a cutoff entry; values must be unchanged.
    np.testing.assert_array_equal(out["AAA"].values, df["AAA"].values)


def test_cleanup_volume_mirrors_cleanup_prices_mask():
    """cleanup_volume applies the same DROP_TICKERS + TICKER_CUTOFFS
    semantics as cleanup_prices so the two caches stay in lockstep."""
    idx = pd.bdate_range("2018-01-01", periods=2000)
    df = pd.DataFrame({"CFC": 1.0, "SBNY": 1.0, "SW": 1.0,
                       "AMCR": 1.0, "AAA": 1.0},
                      index=idx)
    out = apply_data_cleanup.cleanup_volume(df)

    assert "CFC" not in out.columns and "SBNY" not in out.columns

    sw_cutoff = pd.Timestamp("2024-07-15")
    assert out.loc[out.index < sw_cutoff, "SW"].isna().all()

    amcr_cutoff = pd.Timestamp("2019-06-11")
    assert out.loc[out.index < amcr_cutoff, "AMCR"].isna().all()


def test_cleanup_prices_handles_tz_aware_index():
    """A tz-aware index must produce the same mask as a tz-naive one.
    Pre-audit cleanup_prices used a bare ``pd.Timestamp(cutoff)`` which
    would either raise a tz-mismatch comparison or silently apply the
    wrong mask; routing through _apply_ticker_cutoffs fixes both paths."""
    idx = pd.bdate_range("2018-01-01", periods=2000, tz="UTC")
    df = pd.DataFrame({"SW": 1.0, "AAA": 2.0}, index=idx)
    out = apply_data_cleanup.cleanup_prices(df)

    sw_cutoff = pd.Timestamp("2024-07-15", tz="UTC")
    assert out.loc[out.index < sw_cutoff, "SW"].isna().all()
    assert not out.loc[out.index >= sw_cutoff, "SW"].isna().any()


def test_cleanup_volume_handles_tz_aware_index():
    """Same tz robustness check as for cleanup_prices."""
    idx = pd.bdate_range("2018-01-01", periods=2000, tz="UTC")
    df = pd.DataFrame({"AMCR": 1.0, "AAA": 2.0}, index=idx)
    out = apply_data_cleanup.cleanup_volume(df)

    amcr_cutoff = pd.Timestamp("2019-06-11", tz="UTC")
    assert out.loc[out.index < amcr_cutoff, "AMCR"].isna().all()
    assert not out.loc[out.index >= amcr_cutoff, "AMCR"].isna().any()


# --- run_download cache-load path ------------------------------------

def test_run_download_cache_load_returns_existing_frames(tmp_path, monkeypatch):
    """force=False with all three parquets present returns them without
    touching Yahoo. Guards against a regression where the cache-load
    branch silently re-downloads."""
    import src.stage1_data.download as dl

    monkeypatch.setattr(dl, "DATA_DIR", tmp_path)
    monkeypatch.setattr(dl, "CACHE_META_PATH", tmp_path / "cache_metadata.json")

    idx = pd.bdate_range("2024-01-01", periods=10)
    prices = pd.DataFrame({"AAA": [1.0] * 10}, index=idx)
    returns = pd.DataFrame({"AAA": [0.01] * 9}, index=idx[1:])
    info = pd.DataFrame({
        "Symbol":           ["AAA"],
        "Security":         ["x"],
        "GICS Sector":      ["Financials"],
        "GICS Sub-Industry": ["Financials"],
    })
    prices.to_parquet(tmp_path / "sp500_prices.parquet")
    returns.to_parquet(tmp_path / "sp500_returns.parquet")
    info.to_parquet(tmp_path / "sp500_info.parquet")

    def boom(*a, **kw):
        raise AssertionError(
            "download_prices/get_sp500_tickers invoked on the cache-load path"
        )
    monkeypatch.setattr(dl, "download_prices", boom)
    monkeypatch.setattr(dl, "get_sp500_tickers", boom)

    out_info, out_prices, out_returns = dl.run_download(force=False)
    # Parquet round-trip drops DatetimeIndex.freq (BusinessDay → None);
    # values are identical, only the freq attribute differs.
    pd.testing.assert_frame_equal(out_prices, prices, check_freq=False)
    pd.testing.assert_frame_equal(out_returns, returns, check_freq=False)
    pd.testing.assert_frame_equal(out_info, info)


# --- get_sp500_tickers BRK.B / BF.B normalisation --------------------

def test_get_sp500_tickers_normalises_dot_to_hyphen(monkeypatch):
    """Wikipedia 'BRK.B' / 'BF.B' must come back as 'BRK-B' / 'BF-B'
    (Yahoo uses hyphenated class suffixes). regex=False is required so
    that the dot is replaced literally; regex=True would treat the dot
    as 'any character' and corrupt other symbols."""
    import src.stage1_data.download as dl

    rows = []
    for i in range(490):
        rows.append(
            f"<tr><td>TKR{i}</td><td>Name{i}</td>"
            f"<td>Information Technology</td><td>Sub{i}</td></tr>"
        )
    rows.append(
        "<tr><td>BRK.B</td><td>Berkshire B</td>"
        "<td>Financials</td><td>Multi-Sector Holdings</td></tr>"
    )
    rows.append(
        "<tr><td>BF.B</td><td>Brown-Forman B</td>"
        "<td>Consumer Staples</td><td>Distillers</td></tr>"
    )
    fake_html = (
        "<html><body><table>"
        "<tr><th>Symbol</th><th>Security</th>"
        "<th>GICS Sector</th><th>GICS Sub-Industry</th></tr>"
        + "".join(rows)
        + "</table></body></html>"
    )

    class FakeResp:
        text = fake_html

        def raise_for_status(self):
            pass

    monkeypatch.setattr("requests.get", lambda *a, **kw: FakeResp())
    info = dl.get_sp500_tickers()
    syms = set(info["Symbol"])
    assert "BRK-B" in syms
    assert "BF-B" in syms
    assert "BRK.B" not in syms
    assert "BF.B" not in syms


# --- _detect_contaminated boundary edges ----------------------------

def test_detect_contaminated_zero_frac_boundary_strict():
    """zero_frac > 0.20 is strict: exactly 0.20 is NOT flagged, 0.21 is."""
    # Construct a deterministic series with exactly 20% zeros at the end,
    # the rest a non-zero constant. This produces both zero_frac=0.20 and
    # a 20-day flat run, so both thresholds sit at the boundary.
    n = 100
    at_boundary = np.array([0.01] * 80 + [0.0] * 20)
    assert _detect_contaminated(pd.DataFrame({"BOUND": at_boundary})) == [], (
        "0.20 zero-fraction with 20-day flat run is at both cutoffs; "
        "code uses strict > and must not flag."
    )

    just_above = np.array([0.01] * 79 + [0.0] * 21)
    flagged = _detect_contaminated(pd.DataFrame({"OVER": just_above}))
    assert len(flagged) == 1
    assert flagged[0][0] == "OVER"


# --- download_volume.main cross-cache mismatch ----------------------

def test_download_volume_main_raises_on_ticker_set_mismatch(tmp_path, monkeypatch):
    """The set-level integrity guard must abort if the volume fetch
    drops a ticker that prices still carries (BNY/VEEV vs BK/CTRA-style
    audit finding)."""
    import tools.download_volume as dvol

    monkeypatch.setattr(dvol, "DATA_DIR", tmp_path)
    monkeypatch.setattr(dvol, "VOLUME_PATH", tmp_path / "sp500_volume.parquet")
    monkeypatch.setattr(dvol, "ADV_PATH", tmp_path / "sp500_adv.parquet")

    idx = pd.bdate_range("2024-01-01", periods=20)
    prices = pd.DataFrame({"AAA": [100.0] * 20, "BBB": [50.0] * 20}, index=idx)
    prices.to_parquet(tmp_path / "sp500_prices.parquet")

    def fake_fetch(tickers, *a, **kw):
        idx2 = pd.bdate_range("2024-01-01", periods=20)
        # BBB silently dropped — this is the desync the guard must catch.
        c = pd.DataFrame({"AAA": [100.0] * 20}, index=idx2)
        v = pd.DataFrame({"AAA": [1e6] * 20}, index=idx2)
        return c, v

    monkeypatch.setattr(dvol, "fetch_volume_and_raw_close", fake_fetch)
    with pytest.raises(RuntimeError, match="ticker-set mismatch"):
        dvol.main()


def test_download_volume_main_raises_on_numeric_scale_mismatch(tmp_path, monkeypatch):
    """The numeric scale guard must abort when the prices cache and the
    fresh raw_close diverge on the shared anchor date by more than the
    no-corporate-action band. The CVNA 2026-05 mismatch is the
    motivating case: a 5x split between the prices-cache pull and the
    fresh volume pull yielded |log|=ln5≈1.6, far above the 0.3 cutoff."""
    import tools.download_volume as dvol

    monkeypatch.setattr(dvol, "DATA_DIR", tmp_path)
    monkeypatch.setattr(dvol, "VOLUME_PATH", tmp_path / "sp500_volume.parquet")
    monkeypatch.setattr(dvol, "ADV_PATH", tmp_path / "sp500_adv.parquet")

    idx = pd.bdate_range("2024-01-01", periods=20)
    # Prices cache claims AAA = 100; fresh raw_close says AAA = 20.
    # Ticker set matches (no false trigger on the set guard), only the
    # numeric guard should fire.
    prices = pd.DataFrame({"AAA": [100.0] * 20, "BBB": [50.0] * 20}, index=idx)
    prices.to_parquet(tmp_path / "sp500_prices.parquet")

    def fake_fetch(tickers, *a, **kw):
        idx2 = pd.bdate_range("2024-01-01", periods=20)
        c = pd.DataFrame({"AAA": [20.0] * 20, "BBB": [50.0] * 20}, index=idx2)
        v = pd.DataFrame({"AAA": [1e6] * 20, "BBB": [2e6] * 20}, index=idx2)
        return c, v

    monkeypatch.setattr(dvol, "fetch_volume_and_raw_close", fake_fetch)
    with pytest.raises(RuntimeError, match="scale mismatch"):
        dvol.main()


def test_download_volume_main_passes_when_caches_agree(tmp_path, monkeypatch):
    """Sanity test for the happy path: matching ticker sets and prices
    within the no-corporate-action band exit main() with code 0."""
    import tools.download_volume as dvol

    monkeypatch.setattr(dvol, "DATA_DIR", tmp_path)
    monkeypatch.setattr(dvol, "VOLUME_PATH", tmp_path / "sp500_volume.parquet")
    monkeypatch.setattr(dvol, "ADV_PATH", tmp_path / "sp500_adv.parquet")

    idx = pd.bdate_range("2024-01-01", periods=20)
    prices = pd.DataFrame({"AAA": [100.0] * 20, "BBB": [50.0] * 20}, index=idx)
    prices.to_parquet(tmp_path / "sp500_prices.parquet")

    def fake_fetch(tickers, *a, **kw):
        idx2 = pd.bdate_range("2024-01-01", periods=20)
        # Match prices exactly — both guards must stay silent.
        c = pd.DataFrame({"AAA": [100.0] * 20, "BBB": [50.0] * 20}, index=idx2)
        v = pd.DataFrame({"AAA": [1e6] * 20, "BBB": [2e6] * 20}, index=idx2)
        return c, v

    monkeypatch.setattr(dvol, "fetch_volume_and_raw_close", fake_fetch)
    rc = dvol.main()
    assert rc == 0
    assert (tmp_path / "sp500_volume.parquet").exists()
    assert (tmp_path / "sp500_adv.parquet").exists()


# --- fetch_volume_and_raw_close retry path --------------------------

def test_fetch_volume_and_raw_close_retries_then_succeeds(monkeypatch):
    """First attempt raises; second attempt returns clean data. The
    function must populate both close and volume columns and incur
    exactly one retry per ticker."""
    import tools.download_volume as dvol

    state = {"calls_per_ticker": {}}

    def fake_yf_download(ticker, start, end, progress, auto_adjust):
        state["calls_per_ticker"][ticker] = (
            state["calls_per_ticker"].get(ticker, 0) + 1
        )
        if state["calls_per_ticker"][ticker] == 1:
            raise RuntimeError("simulated HTTP 429")
        idx = pd.bdate_range(start, periods=10)
        df = pd.DataFrame(
            {("Close", ticker): [100.0] * 10,
             ("Volume", ticker): [1e6] * 10},
            index=idx,
        )
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        return df

    monkeypatch.setattr(dvol.yf, "download", fake_yf_download)
    monkeypatch.setattr(dvol.time, "sleep", lambda *_: None)

    close, vol = dvol.fetch_volume_and_raw_close(
        ["AAA"],
        start="2024-01-01", end="2024-01-31",
        max_retries=3, max_missing_fraction=0.0,
    )
    assert "AAA" in close.columns
    assert "AAA" in vol.columns
    assert state["calls_per_ticker"]["AAA"] == 2
    assert not close["AAA"].isna().any()
    assert not vol["AAA"].isna().any()
