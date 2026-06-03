"""
Pipeline configuration for *Strategic Impact Mapping in Financial Markets*.

This module centralises every numeric constant, file path, and snapshot
specification used downstream by Stages 1-5. Stages 1-4 implement the
pre-registered econometric pipeline (A-DCC GARCH -> Graphical LASSO ->
lead-follower direction assignment -> network-level metrics); Stage 5
aggregates the Stage-1, Stage-3, and Stage-4 outputs into the composite
Network Stress Index (NSI) reported in the manuscript.

All defaults are documented at the point of definition; adjusting any
hyper-parameter here is sufficient to alter the entire pipeline without
editing the per-stage modules.
"""
from pathlib import Path
import datetime

# --- Project paths -----------------------------------------------------
# All paths are derived from the repository root so the pipeline is
# location-independent and reproducible across machines.
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
SNAPSHOTS_DIR = RESULTS_DIR / "snapshots"

# --- S&P 500 universe and sample window --------------------------------
# CRSP-backed panel (migration 2026-05-28): 1985-01-02 to 2024-12-31,
# 10,080 trading days, drawn from data/crsp/dsf_long.parquet. The 1985
# start gives the 1987 Black Monday window the 24-day Stage-1 burn-in;
# the 2024 end is the CRSP vintage cutoff at the Bilkent WRDS pull.
# DATA_START / DATA_END are read only by the legacy Yahoo helpers in
# src/stage1_data/download.py (kept for pre-2026-05-28 reproducibility);
# the CRSP entry src/stage1_data/crsp_panel.py defaults to the same
# window directly in build_panel().
DATA_START = "1985-01-01"
DATA_END = "2024-12-31"

# --- Ten temporal snapshots --------------------------------------------
# Each tuple is (label, start_date, end_date, regime). Regimes follow
# the manuscript taxonomy: crisis (GFC and COVID peaks), recovery (six
# months after each crisis), stress (high-volatility, non-systemic
# events), and baseline (calm windows for control). Snapshot boundaries
# are calendar-based; per-window observation counts after Stage-1
# coverage filtering are reported in Tab. SNAPSHOTS of the manuscript.
# Twenty snapshots, chronological. Event windows (crisis + stress +
# recovery) are uniform 6 calendar months peak-centered (~127 trading
# days each); baseline windows are uniform 12 calendar months (~252
# trading days). Inter-snapshot window uniformity makes Stage-2 EBIC
# selection and Stage-4 ER/rewire null distributions intra-group
# comparable; the prior 4.8x window-length asymmetry is closed.
#
# Selection rationale: every event window centres on the VXO/VIX peak
# of the named episode (Whaley 2009 continuity, verified against
# data/vix_continuity.parquet); every baseline is a full calendar year
# whose VIX max < 26 and mean < 16 (deciles of the 1986-2024
# distribution). Two prior baselines failed this check and are
# replaced: 2015 Calm contained Aug 2015 yuan-deval (VIX 40.74),
# 2011-2012 Calm contained 2011 EU-debt / US-downgrade (VIX 48); their
# slots go to 1993 (VIX mean 12.65), 2013 (14.23), and 2017 (11.09,
# historic low). Jan 2020 Pre-shock is dropped as the COVID-cluster
# over-representation already covered by Mar 2020 + Jun 2020.
#
# n_crisis = 7; exact-permutation power floor 1/C(20,7) = 1.29e-5.
SNAPSHOTS = [
    # --- Crises (7), 6-month peak-centered windows ---
    ("Oct 1987 Black Monday", "1987-07-19", "1988-01-19", "crisis"),    # peak 1987-10-19, VXO 150.19
    ("1990-91 Recession",     "1990-05-23", "1990-11-23", "stress"),    # peak 1990-08-23 (Iraq + NBER recession), VIX 38.07
    ("1993 Calm",             "1993-01-01", "1993-12-31", "baseline"),  # mid-90s low-vol regime, VIX mean 12.65
    ("Dec 1994 Tequila",      "1994-09-20", "1995-03-20", "stress"),    # peak 1994-12-20 (Mexican peso devaluation)
    ("Oct 1997 Asian Crisis", "1997-07-27", "1998-01-27", "crisis"),    # peak 1997-10-27, VIX 39.96
    ("Oct 1998 LTCM",         "1998-07-08", "1999-01-08", "crisis"),    # peak 1998-10-08, VIX 48.56
    ("Apr 2000 Dot-com",      "2000-01-14", "2000-07-14", "crisis"),    # peak 2000-04-14, VIX 39.33
    ("Sep 2001 9/11",         "2001-06-20", "2001-12-20", "crisis"),    # peak 2001-09-20, VIX 49.04
    ("Jul 2002 WorldCom",     "2002-04-23", "2002-10-23", "stress"),    # peak 2002-07-23, VIX 50.48
    ("2005 Calm",             "2005-01-01", "2005-12-31", "baseline"),  # pre-GFC low-vol, VIX mean 12.81
    ("2007 Subprime Buildup", "2007-08-12", "2008-02-12", "stress"),    # peak 2007-11-12, VIX 31.09 (GFC precursor)
    ("Oct 2008 GFC",          "2008-08-20", "2009-02-20", "crisis"),    # peak 2008-11-20, VIX 80.86
    ("Mar 2009 Recovery",     "2009-04-01", "2009-09-30", "recovery"),  # post-2009-03-09 trough rally; pure-recovery 6m
    ("2013 Calm",             "2013-01-01", "2013-12-31", "baseline"),  # post-GFC stable baseline, VIX mean 14.23
    ("2017 Calm",             "2017-01-01", "2017-12-31", "baseline"),  # historic VIX low, mean 11.09
    ("Q4 2018 VolShock",      "2018-09-24", "2019-03-24", "stress"),    # Dec 2018 Christmas low, VIX 36
    ("Mar 2020 COVID",        "2019-12-16", "2020-06-16", "crisis"),    # peak 2020-03-16, VIX 82.69
    ("Jun 2020 Stable",       "2020-06-17", "2020-12-17", "recovery"),  # post-COVID rally
    ("2022 Rate Hikes",       "2022-03-07", "2022-09-07", "stress"),    # Fed June hike cycle, VIX 36.45
    ("2024 Contemporary",     "2024-01-01", "2024-12-31", "baseline"),  # recent perspective, VIX mean 15.55
]

# --- Rolling-NSI sampling grid (Stage 5) -------------------------------
# A 252-trading-day window with a 21-trading-day stride yields one NSI
# observation per business month over the full sample, matching the
# resolution at which the CBOE VIX is plotted in Stage-5 backtests.
ROLLING_WINDOW_DAYS = 252
ROLLING_STEP_DAYS = 21

# --- Stage 1: univariate GARCH -----------------------------------------
# Student-t innovations accommodate the leptokurtic tails documented for
# daily equity returns (Cont 2001); orders (1, 1) are the parsimonious
# default, sufficient to capture the volatility persistence on this
# panel under information-criteria checks.
GARCH_P = 1
GARCH_Q = 1
GARCH_DIST = "studentst"

# --- Stage 2: Graphical LASSO ------------------------------------------
# A 25-point log-spaced lambda grid bounded below by 0.05 skips the
# guaranteed-non-convergent low-lambda regime on n/p < 0.5 windows; the
# upper bound 1.0 is the empirical 'empty-graph' ceiling on this panel.
# EBIC penalty gamma follows Foygel and Drton (2010); 0.5 is the
# high-dimensional default recommended for n/p < 1.
GLASSO_LAMBDA_RANGE = (0.05, 1.0)
GLASSO_N_LAMBDAS = 25
EBIC_GAMMA = 0.5

# Tier-dependent EBIC penalty: when the sample-to-dimension ratio is
# small the standard penalty over-shrinks the graph to emptiness, so we
# fall back to BIC (gamma = 0). When the ratio is large enough for
# concentration to bite, we use the canonical gamma = 0.5. The mid
# tier interpolates linearly. Thresholds match the manuscript's
# Stage-2 disclosure.
EBIC_RATIO_LOW = 3.0
EBIC_RATIO_MID = 5.0
EBIC_GAMMA_MID = 0.25

# --- Stage 3: direction assignment -------------------------------------
# Lag order 1 matches the daily horizon of the underlying returns; a
# longer Granger order would mix multi-day information flows with the
# pairwise lead-follower test. The 1.5x dominance ratio is the
# minimum geometric margin a direction must clear over its reverse to
# be classified as a one-way edge rather than mutual; sensitivity is
# disclosed in F3 of the manuscript.
LAG_ORDER = 1
GRANGER_MAX_LAG = 1
SIGNIFICANCE_LEVEL = 0.05
DIRECTION_RATIO_THRESH = 1.5

# --- Stage 4: network-level metrics ------------------------------------
# PageRank damping 0.85 follows Page et al. (1999); 200 ER null draws
# keep the empirical Z-score Monte-Carlo standard error below 0.07 on
# the densest baseline. Louvain resolution 1.0 is the canonical
# Newman-Girvan modularity objective (Blondel et al. 2008). Motif
# rewiring uses the Maslov-Sneppen (2002) dyad-preserving null with
# 100 rewires per snapshot.
PAGERANK_DAMPING = 0.85
ERDOS_RENYI_N_SIMS = 200
LOUVAIN_RESOLUTION = 1.0
MOTIF_N_REWIRES = 100

# --- Stage 5: rolling-NSI channel weights ------------------------------
# Prior weights for the simplified rolling NSI (three channels: no motif
# channel because Stage 3 is omitted in the rolling pipeline). The
# snapshot-level NSI uses a separate four-channel split (sparsity,
# PageRank Herfindahl, mean correlation, FFL motif shift) defined in
# src.stage5_nsi.stress_index.NSI_WEIGHTS_4CH; do not confuse the two.
# These three weights sum to one and emphasise the mean-correlation
# channel as the strongest direct systemic signal.
ROLLING_NSI_WEIGHTS_3CH = {
    "network_sparsity": 0.35,
    "pagerank_concentration": 0.25,
    "mean_corr": 0.40,
}

# --- GICS sector codes -------------------------------------------------
# Two-digit GICS codes (S&P / MSCI 2024 revision) used for sector
# purity, cross-sector edge fraction, and node-colour assignment in
# Fig. 1 of the manuscript.
GICS_SECTORS = {
    10: "Energy",
    15: "Materials",
    20: "Industrials",
    25: "Consumer Discretionary",
    30: "Consumer Staples",
    35: "Health Care",
    40: "Financials",
    45: "Information Technology",
    50: "Communication Services",
    55: "Utilities",
    60: "Real Estate",
}

# --- A-DCC estimation hyper-parameters ---------------------------------
# 500 L-BFGS-B iterations are sufficient for convergence on every
# initialisation tested; the 100-asset MLE subset trades a small
# information loss for a tractable Hessian update at p = 100 (Engle,
# Ledoit, and Wolf 2019).
ADCC_MAX_ITER = 500
ADCC_SUBSET_SIZE = 100

# --- Rolling-NSI Stage-5 hyper-parameters ------------------------------
# A fixed Graphical LASSO penalty (alpha = 0.15) and a 20-asset minimum
# alive-ticker count keep the rolling pipeline computationally cheap
# while remaining comparable to the snapshot Stage-2 output in
# off-diagonal density at calm baselines.
ROLLING_GLASSO_ALPHA = 0.15
ROLLING_MIN_ASSETS = 20

# --- Statistical-inference hyper-parameters ----------------------------
# The §7 inference layer (exact permutation, BH FDR, cluster bootstrap,
# stationary block bootstrap, Bonferroni-10) lives in paper/_inference.py.
# All replica counts, block lengths, and RNG seeds are pinned at the call
# site there so the published numbers are reproducible from a single
# `python -m paper._inference` invocation. No constants are exported from
# here to avoid two sources of truth.
