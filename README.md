# Strategic Impact Mapping: A Directed-Network Framework for S&P 500 Crisis Regimes

Reproducible code for an end-to-end **directed-network pipeline** that turns the
daily S&P 500 return panel into a sparse directed "strategic impact map" and a
single composite stress reading, evaluated across twenty regime snapshots over
1985–2024.

> **Status:** author **preprint**, submitted to *IEEE Transactions on Network
> Science and Engineering* (TNSE). The compiled manuscript and its 75-page
> companion appendix are in [`paper/`](paper/) (`paper.pdf`, `appendix.pdf`).

---

## What this is

A single panel-scale workflow that integrates four ingredients prior S&P 500
work uses only in part, run per snapshot on the **survivorship-free, historical-
constituent CRSP panel** (point-in-time `crsp.msp500list` membership, PERMNO-
indexed; 1,442 PERMNOs, 655–889 alive per snapshot, 10,080 trading days; the
2008-era distress exits LEH/BSC/MER/FNM/FRE/WAMU enter with their full
pre-bankruptcy trajectory):

| Stage | Does | Key methods |
|---|---|---|
| **1. A-DCC GARCH** | time-varying conditional correlations → window-average R̄ | GARCH(1,1)-*t*; globally pooled (a,b,g) on a 100-asset always-alive subset |
| **2. Graphical LASSO** | sparse partial-correlation graph | adaptive EBIC; **n/p-tiered identity shrinkage** + **constrained-BIC fallback** (recovers a non-empty graph at 19/20 windows) |
| **3. Lead/follower** | direct each surviving edge | two-test cascade: lagged Frisch–Waugh–Lovell partial correlation **and** bivariate Granger *F* |
| **4. Network metrics** | four topological readouts | ER clustering \|Z_C\|, PageRank Gini/Herfindahl, Louvain modularity + GICS sector purity, MAN triads (FFL/MR/SIM) |
| **5. NSI** | composite stress score in [0,1] | weighted sparsity + hub-concentration + mean-ρ + motif-shift |

**Twenty snapshots:** 7 crisis peaks (Oct 1987 Black Monday, Oct 1997 Asian,
Oct 1998 LTCM, Apr 2000 Dot-com, Sep 2001 9/11, Oct 2008 GFC, Mar 2020 COVID),
6 non-crisis stress windows, 2 recoveries, 5 calm baselines.

**Two elements are the stated novelty:** (i) the integration itself, and (ii)
the two Stage-2 numerical modifications above. A formal inference layer (exact
permutation tests over all C(20,7)=77,520 label assignments, cluster- and
block-bootstrap confidence sets, Benjamini–Hochberg/Yekutieli + Bonferroni-20
multiplicity control) and an **836-check numerical-invariant assertion suite**
(805 PASS / 31 expected n/p<1 near-PSD + GARCH-boundary FAILs) back every
reported number.

## Headline findings (honest summary)

- **PageRank Gini concentration** is the strongest crisis-vs-non-crisis
  discriminator (Cohen's *d* = 1.04; the only channel clearing the uncorrected
  one-sided permutation test, *p* = 0.022) — but **no channel survives
  multiplicity correction** (smallest Benjamini–Hochberg *q* = 0.152).
- Q4 (mutual-dyad fraction, *d* = 0.68) and Q3 (cross-sector edge fraction)
  move in the predicted direction; Q1 is a non-randomness sanity check.
- The **NSI ranks Oct-2008 GFC first but does not separate crisis from
  non-crisis at the regime-mean level** (motivates a weight redesign, F1).
- Snapshot NSI is **positively but not significantly** associated with the CBOE
  VIX-continuity benchmark (Pearson *r* = 0.321; cluster-bootstrap 95% CI
  includes zero); a rolling counterpart is sign-inconsistent across horizons.

This is a methodology-and-characterization paper for a network venue: the
contribution is the integrated framework and its rigorous, openly-disclosed
evaluation (including negative results), not a working crisis predictor.

## Repository layout

```
.
├── run_pipeline.py          # orchestrates Stages 1–5 -> results/snapshots/stage{1..5}_results.pkl
├── src/                     # pipeline
│   ├── stage1_data/         #   A-DCC GARCH + CRSP panel construction
│   ├── stage2_precision/    #   Graphical LASSO + shrinkage tiers + constrained-BIC fallback
│   ├── stage3_direction/    #   lead/follower cascade + sensitivity sweep
│   ├── stage4_network/      #   ER / PageRank / Louvain / MAN-motif metrics + dyad-preserving null
│   ├── stage5_nsi/          #   NSI composite, rolling NSI, VIX continuity, volume-weighted NSI
│   ├── _assertions/         #   836-check numerical-invariant suite
│   └── utils/               #   A-DCC core, shared helpers
├── tools/                   # data fetch (WRDS/FRED/Refinitiv), multipanel sweep, diagnostics
├── tests/                   # pytest suite (212 tests)
├── paper/                   # manuscript PDFs (preprint) + figure/table/inference generators
│   ├── paper.pdf, appendix.pdf
│   └── generate_figures.py, _inference.py, _generate_*.py, _nsi_*.py
└── data/README.md           # how to obtain CRSP (data is NOT shipped — license)
```

## Install

Python **3.9.6** (pinned in `requirements.txt` / `environment.yml`):

```bash
pip install -r requirements.txt        # or: conda env create -f environment.yml
```

Core stack: numpy 2.0.2, scipy 1.13.1, pandas 2.2.3, scikit-learn 1.6.1,
networkx 3.2.1, statsmodels 0.14.6, arch 7.2.0, python-louvain 0.16
(+ matplotlib 3.9.4, pyarrow 21.0.0). A `Dockerfile` is provided.

## Data

**Not included** — the CRSP US Daily Stock File license prohibits
redistribution. See [`data/README.md`](data/README.md) for the expected schema
and how to rebuild it (WRDS subscription required for CRSP; the VIX continuity
series is public via FRED). The derived Stage-1…5 caches, the multipanel sweep
summary, and the assertion-suite report will be deposited on **Zenodo** (DOI on
acceptance).

## Reproduce

```bash
# 1. fetch data (needs WRDS/CRSP access — see data/README.md)
python tools/fetch_crsp_sp500_full.py --username <wrds_user>
python -m tools.build_crsp_volume

# 2. run the five-stage pipeline (~30 min on Apple M5)
python run_pipeline.py

# 3. 836-check numerical-invariant assertion suite
python -m src._assertions.invariants

# 4. panel-size / liquidity sweep  (N in {50..500} x {coverage, ADV, deciles})
python -m tools.run_multipanel --all --workers 8

# 5. Stage-3 (alpha, tau, lag, c_floor) sensitivity grid
python -m tools.run_stage3_sweep --csv out.csv

# 6. tests
python -m pytest tests/

# 7. regenerate the paper's figures / tables / inference macros from the caches
python paper/generate_figures.py
python -m paper._inference
python -m paper._generate_multipanel_tables
```

**Determinism:** NumPy `default_rng` streams are pinned (seeds 2026 / 42); the
Stage-1 multi-start MLE and the sklearn coordinate-descent solver are
deterministic but non-seeded (host-BLAS dependent) and reproduce within rank
order across re-runs. Full per-module seed table and per-stage run-times are in
the appendix "Reproducibility" section.

## License & citation

- **Code** (`src/`, `tools/`, `tests/`, `paper/*.py`, `run_pipeline.py`):
  MIT — see [`LICENSE`](LICENSE).
- **Manuscript** PDFs: © 2026 the author, posted as a preprint; the version of
  record will be under IEEE copyright on acceptance.
- Please cite via [`CITATION.cff`](CITATION.cff).

## Author

**Nezih Arhan Tözenbilek** — Department of Computer Engineering, Hacettepe
University, Ankara, Turkey.
