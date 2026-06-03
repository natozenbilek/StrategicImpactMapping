# Data

**No market data is shipped in this repository.** The pipeline is built on the
CRSP US Daily Stock File, whose license **prohibits raw redistribution**. The
files the pipeline expects under `data/` are therefore git-ignored and must be
rebuilt from source by anyone with the appropriate access.

## What the pipeline expects here

Built by the fetch/build scripts into `data/` (DatetimeIndex × ticker unless noted):

| File | Content | Source |
|---|---|---|
| `sp500_prices.parquet`  | split-adjusted close, 1985–2024 | CRSP `dsf` |
| `sp500_returns.parquet` | `log(1+ret)` total return        | CRSP `dsf.ret` |
| `sp500_info.parquet`    | PERMNO → ticker, GICS (from SIC) | CRSP `stocknames` |
| `sp500_volume.parquet`  | daily volume                     | CRSP `dsf.vol` |
| `sp500_adv.parquet`     | mean dollar volume per PERMNO    | derived |
| `vix_continuity.parquet`| Whaley VXO+VIX splice (1986→)    | FRED `VXOCLS`+`VIXCLS` (public) |

Panel: ~1,442 historical S&P 500 PERMNOs over 1985-01-02 … 2024-12-31
(10,080 trading days), keyed point-in-time on `crsp.msp500list`.

## How to rebuild

1. **CRSP (requires a WRDS subscription with CRSP access):**
   ```bash
   pip install wrds pandas pyarrow
   python tools/fetch_crsp_sp500_full.py --username <your_wrds_user>
   # first call prompts for the WRDS password + DUO; caches to ~/.pgpass
   python -m tools.build_crsp_volume      # -> sp500_{volume,adv}.parquet
   ```
2. **VIX continuity (public, no subscription):** built automatically at pipeline
   start by `src/stage5_nsi/vix_continuity.py` from FRED `VXOCLS`+`VIXCLS`
   (spliced 2003-09-22), cached to `data/vix_continuity.parquet`.

No credentials are stored in this repository; WRDS/Refinitiv access is read from
an interactive prompt, `~/.pgpass`, or environment variables at run time.

## Derived caches and the assertion report

The Stage-1…5 result caches (`results/snapshots/stage{1..5}_results.pkl`), the
panel-size sweep (`results/multipanel/`), and the 836-check assertion-suite
report are **not** in this repository (they embed CRSP-derived values and are
large). They will be deposited on **Zenodo** (DOI minted on acceptance) as the
reproduction archive referenced in the paper's "Code and data availability".
