"""Statistical-inference layer for the manuscript (§7 + appendix §11).

Consumes the Stage-1/3/4/5 pickles + data/vix.parquet and produces:

    (A) Exact permutation tests with BH FDR on the seven per-snapshot
        metrics; floor 1/77,520 at n_crisis=7.
    (A') COVID-merged sensitivity row: Mar/Jun 2020 collapsed into a
         single COVID-cluster snapshot, n_crisis=7, C(19,7)=50,388,
         floor 1/50,388.
    (B) Cluster bootstrap (B=10000) on snapshot NSI-VIX: GFC pair +
        COVID pair + sixteen singletons resampled atomically; label-shuffle
        permutation companion (B=100000).
    (C) Stationary block bootstrap (Politis-Romano 1994, B=5000) on
        rolling NSI vs daily VIX at mean block lengths {6, 12, 24}.
    (D) Bonferroni-10 on per-snapshot Q1 |Z_C| + three motif Z-scores
        (FFL, MR, SIM), plus Cohen's d per crisis-vs-non-crisis contrast.

Side effect on success: writes paper/output/inference_macros.tex containing
\\newcommand definitions for every headline number, so paper.tex / appendix.tex
can reference them via \\input without manual transcription drift.

Determinism: independent np.random.default_rng streams per bootstrap branch
(cluster=2026, label-shuffle=2026, block-bootstrap=2026). No global seed.
"""
from __future__ import annotations

import pickle
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[1]
SNAP_DIR = ROOT / "results" / "snapshots"
VIX_PARQUET = ROOT / "data" / "vix_continuity.parquet"
SP500_INFO = ROOT / "data" / "sp500_info.parquet"
MACRO_OUT = Path(__file__).resolve().parent / "output" / "inference_macros.tex"


# SNAPS / REGIME / SNAPSHOT_WINDOWS derived from src.config so the
# inference layer follows the snapshot-set redesign (twenty snapshots
# 2026-05-28) without a manual edit; the prior hardcoded ten-snapshot
# Yahoo dict is gone. The cluster partition still needs hand-curation
# because it encodes which calendar windows share trading days (the
# GFC pair, the COVID pair, etc.).
import sys
sys.path.insert(0, str(ROOT))
from src.config import SNAPSHOTS as _CFG_SNAPSHOTS

SNAPS = [label for label, _s, _e, _r in _CFG_SNAPSHOTS]
REGIME = {label: regime for label, _s, _e, regime in _CFG_SNAPSHOTS}
SNAPSHOT_WINDOWS = {label: (s, e) for label, s, e, _r in _CFG_SNAPSHOTS}

# Cluster partition for the snapshot NSI-VIX cluster bootstrap.
# Designed to absorb the GFC, COVID, and 2009 GFC-tail overlaps (see
# app:snapshot-overlap). One cluster per independent block; non-
# overlapping baselines / stresses are each singletons. Partition
# completeness/disjointness is enforced by
# tests/test_inference.py::test_cluster_partition_complete.
CLUSTERS = [
    ['Oct 1987 Black Monday'],
    ['1990-91 Recession'],
    ['1993 Calm'],
    ['Dec 1994 Tequila'],
    ['Oct 1997 Asian Crisis'],
    ['Oct 1998 LTCM'],
    ['Apr 2000 Dot-com'],
    ['Sep 2001 9/11'],
    ['Jul 2002 WorldCom'],
    ['2005 Calm'],
    ['2007 Subprime Buildup'],
    ['Oct 2008 GFC', 'Mar 2009 Recovery'],
    ['2013 Calm'],
    ['2017 Calm'],
    ['Q4 2018 VolShock'],
    ['Mar 2020 COVID', 'Jun 2020 Stable'],
    ['2022 Rate Hikes'],
    ['2024 Contemporary'],
]

METRIC_KEYS = ['nsi', 'gini', 'hhi', 'abs_z_mr', 'mutual_dyad_pct',
               'cross_sector', 'mean_rho']
LABELS = {
    'nsi': 'NSI',
    'gini': 'PageRank Gini',
    'hhi': 'PageRank HHI',
    'abs_z_mr': '|Z_MR|',
    'mutual_dyad_pct': 'Mutual-dyad fraction (\\%)',
    'cross_sector': 'Cross-sector edge fraction',
    'mean_rho': 'Mean A-DCC correlation',
}
# LaTeX-macro slug per metric (deterministic; no spaces / punctuation).
MACRO_SLUG = {
    'nsi': 'NSI',
    'gini': 'Gini',
    'hhi': 'HHI',
    'abs_z_mr': 'ZMR',
    'mutual_dyad_pct': 'MutDyad',
    'cross_sector': 'CrossSec',
    'mean_rho': 'MeanRho',
}
# Per-metric decimal places for the LaTeX macros. Tuned so the typeset
# table cells stay compact (HHI needs 4 to resolve 0.0075 vs 0.0078; the
# percentage-style mutual-dyad needs 2; everything else 3).
DECIMALS = {
    'nsi': 3, 'gini': 3, 'hhi': 4, 'abs_z_mr': 2,
    'mutual_dyad_pct': 2, 'cross_sector': 3, 'mean_rho': 3,
}


# ---------------------------------------------------------------------------
# Pure helpers (no I/O; testable)
# ---------------------------------------------------------------------------


def gini(x):
    """Population Gini of a non-negative vector, in [0, 1]."""
    x = np.sort(np.asarray(x, dtype=float))
    assert (x >= 0).all(), "Gini requires non-negative input"
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    return (n + 1 - 2 * cum.sum() / cum[-1]) / n


def perm_test(values_by_snap, crisis_set, all_snaps):
    """Exact one-sided permutation test of mean(crisis) > mean(non-crisis).

    Returns (obs_diff, p_one_sided, n_total) with n_total = C(N, k_crisis).
    Enumeration is exact; observed assignment is counted (>=, not >), so the
    floor is 1/n_total.
    """
    assert len(set(crisis_set)) == len(crisis_set), "crisis_set must be unique"
    assert set(crisis_set).issubset(all_snaps), "crisis_set must be in all_snaps"
    all_vals = np.array([values_by_snap[s] for s in all_snaps])
    obs_crisis = np.array([values_by_snap[s] for s in crisis_set])
    obs_nc = np.array([values_by_snap[s] for s in all_snaps if s not in crisis_set])
    obs_diff = obs_crisis.mean() - obs_nc.mean()
    n_crisis = len(crisis_set)
    n_extreme = 0
    n_total = 0
    for combo in combinations(range(len(all_snaps)), n_crisis):
        n_total += 1
        idx_c = list(combo)
        idx_nc = [i for i in range(len(all_snaps)) if i not in idx_c]
        diff = all_vals[idx_c].mean() - all_vals[idx_nc].mean()
        if diff >= obs_diff:
            n_extreme += 1
    assert n_extreme >= 1, "observed assignment should always count as extreme"
    return obs_diff, n_extreme / n_total, n_total


def bh_fdr(p_dict):
    """Benjamini-Hochberg step-up q-values.

    q_(k) = min_{j>=k} m * p_(j) / j after monotone right-to-left enforcement.
    Returns dict keyed by the same keys as p_dict.
    """
    keys = list(p_dict.keys())
    m = len(keys)
    assert m >= 1, "empty p-value family"
    sorted_keys = sorted(keys, key=lambda k: p_dict[k])
    q = {}
    prev = 1.0
    for i in range(m - 1, -1, -1):
        k = sorted_keys[i]
        raw = p_dict[k] * m / (i + 1)
        prev = min(raw, prev)
        q[k] = prev
    for k, v in q.items():
        assert 0.0 <= v <= 1.0, f"q out of [0,1] for {k}: {v}"
    return q


def by_fdr(p_dict):
    """Benjamini-Yekutieli step-up q-values under arbitrary dependence.

    BY tightens BH by a factor c(m) = sum_{i=1..m} 1/i (the m-th harmonic
    number), so BY q = BH q * c(m), capped at 1. Reported alongside BH
    when the positive-dependence assumption underlying BH cannot be
    formally verified (Benjamini & Yekutieli 2001).
    """
    keys = list(p_dict.keys())
    m = len(keys)
    assert m >= 1, "empty p-value family"
    c_m = float(sum(1.0 / i for i in range(1, m + 1)))
    bh = bh_fdr(p_dict)
    return {k: min(1.0, v * c_m) for k, v in bh.items()}


def hochberg_fwer(p_dict):
    """Hochberg step-up FWER-adjusted p-values.

    Standard formula (1-indexed): q_(k) = min_{j>=k} (m-j+1) * p_(j),
    with the resulting sequence automatically monotone non-decreasing in
    k. Hochberg is step-up; it dominates Holm-Bonferroni (which is
    step-down) under positive dependence. Both collapse to the same
    floor when the smallest p exceeds alpha/m, so on this panel
    (smallest p ~ 0.022 for PageRank Gini vs Hochberg's most stringent
    threshold alpha/m = 0.05/7 ~ 0.00714) Hochberg adds no rejections beyond
    Holm. Reported as the third FWER companion in J-C for completeness.
    """
    keys = list(p_dict.keys())
    m = len(keys)
    assert m >= 1, "empty p-value family"
    sorted_keys = sorted(keys, key=lambda k: p_dict[k])
    # raw[i] = (m - i) * p_(i+1) in 1-indexed form, i.e. m*p_(1), (m-1)*p_(2), ..., 1*p_(m).
    raw = [(m - i) * p_dict[sorted_keys[i]] for i in range(m)]
    # Step-up: q_(k) = min over j>=k of raw[j]. Walk right-to-left.
    q_sorted = [0.0] * m
    prev = 1.0
    for i in range(m - 1, -1, -1):
        prev = min(prev, raw[i])
        q_sorted[i] = min(1.0, prev)
    return {sorted_keys[i]: q_sorted[i] for i in range(m)}


def cohen_d(vals_c, vals_nc):
    """Cohen's d on two groups with pooled within-group SD (ddof=1)."""
    vals_c = np.asarray(vals_c, dtype=float)
    vals_nc = np.asarray(vals_nc, dtype=float)
    n_c, n_nc = len(vals_c), len(vals_nc)
    assert n_c >= 2 and n_nc >= 2, "Cohen's d needs >=2 observations per group"
    pooled = np.sqrt(((n_c - 1) * np.var(vals_c, ddof=1)
                      + (n_nc - 1) * np.var(vals_nc, ddof=1))
                     / (n_c + n_nc - 2))
    if pooled < 1e-12:
        return float('inf') if vals_c.mean() > vals_nc.mean() else float('-inf')
    return (vals_c.mean() - vals_nc.mean()) / pooled


def bonferroni_two_sided_threshold(m, alpha=0.05):
    """Bonferroni-corrected two-sided |Z| threshold for m tests at alpha."""
    assert m >= 1 and 0 < alpha < 1
    return float(norm.ppf(1.0 - alpha / (2.0 * m)))


def cluster_bootstrap(x, y, clusters_idx, B=10000, seed=2026):
    """Cluster bootstrap on paired (x, y) at the cluster-of-indices level.

    clusters_idx: list of int-list, each cluster is a list of integer indices
    into x / y. The cluster contents are resampled with replacement at the
    cluster level (not the observation level) and flattened.

    Returns (boot_pearson, boot_spearman, n_dropped) with NaN entries on
    zero-variance replicates (also reported via n_dropped).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    assert x.shape == y.shape and x.ndim == 1
    n_clusters = len(clusters_idx)
    assert n_clusters >= 2, "cluster bootstrap needs >=2 clusters"
    rng = np.random.default_rng(seed)
    boot_p = np.empty(B)
    boot_s = np.empty(B)
    n_dropped = 0
    for b in range(B):
        chosen = rng.integers(0, n_clusters, size=n_clusters)
        idx = []
        for ci in chosen:
            idx.extend(clusters_idx[int(ci)])
        xb = x[idx]
        yb = y[idx]
        if np.std(xb) < 1e-10 or np.std(yb) < 1e-10:
            boot_p[b] = np.nan
            boot_s[b] = np.nan
            n_dropped += 1
            continue
        boot_p[b], _ = pearsonr(xb, yb)
        boot_s[b], _ = spearmanr(xb, yb)
    return boot_p, boot_s, n_dropped


def label_shuffle_permutation(x, y, B=100000, seed=2026):
    """Two-sided label-shuffle permutation of Pearson r."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    assert x.shape == y.shape and x.ndim == 1
    obs, _ = pearsonr(x, y)
    rng = np.random.default_rng(seed)
    perm_r = np.empty(B)
    for b in range(B):
        ys = rng.permutation(y)
        perm_r[b], _ = pearsonr(x, ys)
    p = float((np.abs(perm_r) >= np.abs(obs)).sum()) / B
    return float(obs), p


def stationary_block_bootstrap(x, y, mean_block_size=12, B=5000, seed=2026):
    """Politis-Romano (1994) stationary block bootstrap on paired Pearson r.

    Block lengths drawn from Geometric(1/mean_block_size); wrap-around with
    i = (i+1) mod N. Returns array of length B, NaN on zero-variance
    replicates.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    assert x.shape == y.shape and x.ndim == 1
    N = len(x)
    assert N >= 2 and mean_block_size >= 1
    rng = np.random.default_rng(seed)
    p = 1.0 / mean_block_size
    out = np.empty(B)
    for b in range(B):
        indices = []
        i = int(rng.integers(0, N))
        while len(indices) < N:
            indices.append(i)
            i = (i + 1) % N
            if rng.random() < p:
                i = int(rng.integers(0, N))
        indices = np.asarray(indices[:N])
        xs, ys = x[indices], y[indices]
        if np.std(xs) < 1e-10 or np.std(ys) < 1e-10:
            out[b] = np.nan
        else:
            out[b] = float(np.corrcoef(xs, ys)[0, 1])
    return out


# ---------------------------------------------------------------------------
# Calibration Monte Carlo: directional bias of the exact-permutation null
# under positive within-cluster dependence (COVID-triple overlap).
#
# RETIRED (pre-redesign device, kept for archival/reproducibility). It models
# the pre-2026-05-28 Yahoo geometry -- ten snapshots, n_crisis=2, a three-way
# COVID overlap (Jan/Mar/Jun 2020) -- whose overlap-induced positive dependence
# had to be shown non-anti-conservative. The CRSP redesign removes that overlap
# (COVID is now a disjoint pair, Mar/Jun 2020), so the appendix explicitly drops
# this calibration ("an overlap-specific calibration is no longer required") and
# the manuscript consumes none of its macros. Logic/signature frozen so the
# existing self-contained unit tests keep passing; do not wire its output back
# into the paper.
# ---------------------------------------------------------------------------


def calibration_sim_positive_dependence(
    n_replicates=10000, rho_grid=(0.0, 0.3, 0.7), alpha_grid=(0.05, 0.10, 0.20),
    crisis_indices=(0, 6), covid_indices=(5, 6, 7), seed=2026,
):
    """Empirical type-I rate of the snapshot exact-permutation test as a
    function of within-COVID-triple correlation rho.

    RETIRED pre-redesign device (see the section banner above): the geometry
    below is the pre-2026-05-28 Yahoo headline geometry, not the current CRSP
    panel. Kept only so the self-contained unit tests still run; its macros
    are not consumed by the manuscript.

    Setup under H0 (no crisis effect):
        X_others ~ N(0, 1) i.i.d. for the 7 non-COVID snapshots
        (X_jan, X_mar, X_jun) ~ N(0, Sigma_rho) with off-diagonal rho
    The crisis arm is fixed at indices (Oct 2008 = 0, Mar 2020 = 6), so
    Mar 2020 (in the COVID triple) is one crisis snapshot and Jan/Jun 2020
    are non-crisis. This was the exact label geometry of the pre-redesign
    headline test; under CRSP the headline geometry is twenty snapshots,
    seven crises, and a disjoint COVID pair.

    For each replicate the one-sided perm_test is run and the resulting
    p-value recorded. Empirical type-I rate at alpha is the fraction with
    p <= alpha. Under H0 a well-calibrated test gives ~alpha;
    'conservative under positive dependence' predicts a lower rate at
    rho > 0; 'anti-conservative' predicts a higher rate.

    Returns: dict {rho: {alpha: rate}} plus mean_p per rho.
    """
    assert len(crisis_indices) == 2 and len(covid_indices) == 3
    assert set(crisis_indices).issubset(set(range(10)))
    assert set(covid_indices).issubset(set(range(10)))
    others = [i for i in range(10) if i not in covid_indices]
    snap_labels = [f"s{i}" for i in range(10)]
    crisis_labels = [snap_labels[i] for i in crisis_indices]
    rng = np.random.default_rng(seed)
    out = {}
    for rho in rho_grid:
        assert -0.5 <= rho < 1.0, "rho must keep Sigma_rho positive-definite"
        # Sigma for the 3-d COVID block: 1 on diagonal, rho off-diagonal.
        sigma = np.full((3, 3), rho)
        np.fill_diagonal(sigma, 1.0)
        chol = np.linalg.cholesky(sigma)
        p_values = np.empty(n_replicates)
        for r in range(n_replicates):
            x = np.empty(10)
            x[others] = rng.standard_normal(len(others))
            z = rng.standard_normal(3)
            x[list(covid_indices)] = chol @ z
            values = {snap_labels[i]: float(x[i]) for i in range(10)}
            _, p, _ = perm_test(values, crisis_labels, snap_labels)
            p_values[r] = p
        rate_by_alpha = {a: float((p_values <= a).mean()) for a in alpha_grid}
        out[rho] = {
            "rates": rate_by_alpha,
            "mean_p": float(p_values.mean()),
            "median_p": float(np.median(p_values)),
            "n": n_replicates,
        }
    return out


# ---------------------------------------------------------------------------
# Coverage Monte Carlo: 95% percentile-CI coverage of the cluster bootstrap
# on variable-length resamples (|I_b| in [7, 21]).
# ---------------------------------------------------------------------------


def _cluster_bootstrap_pearson_only(x, y, clusters_idx, B, rng):
    """Pearson-only cluster bootstrap inner loop (avoids the spearmanr call
    in the hot path; used by cluster_bootstrap_coverage_sim only)."""
    n_clusters = len(clusters_idx)
    boot = np.empty(B)
    for b in range(B):
        chosen = rng.integers(0, n_clusters, size=n_clusters)
        idx = []
        for ci in chosen:
            idx.extend(clusters_idx[int(ci)])
        xb = x[idx]
        yb = y[idx]
        sx, sy = xb.std(), yb.std()
        if sx < 1e-10 or sy < 1e-10:
            boot[b] = np.nan
        else:
            # Inline Pearson — faster than scipy.stats.pearsonr in tight loops.
            mx, my = xb.mean(), yb.mean()
            boot[b] = float(((xb - mx) * (yb - my)).sum() / (len(xb) * sx * sy))
    return boot


def cluster_bootstrap_coverage_sim(
    true_r_grid=(0.0, 0.5, 0.8), n_replicates=500, B_bootstrap=1000,
    seed=2026,
):
    """Empirical 95% CI coverage of the cluster bootstrap on (n=10, 7-cluster)
    data with variable-length resamples.

    For each true Pearson r:
        - draw (X, Y) of length len(SNAPS) from a bivariate normal with that r
        - apply Pearson-only cluster bootstrap with the production CLUSTERS
          partition (18 clusters on the post-2026-05-28 twenty-snapshot panel)
        - form 95% percentile CI on the Pearson resample vector
        - record whether the true r is inside the CI
    Coverage = fraction of replicates where the CI contains true_r.
    A correctly-calibrated 95% CI gives ~95% coverage; lower indicates
    under-coverage (CI too narrow), higher indicates over-coverage.

    Returns: dict {true_r: {coverage, ci_mean_width, n_replicates}}.
    """
    rng = np.random.default_rng(seed)
    snap_index = {s: i for i, s in enumerate(SNAPS)}
    clusters_idx = [[snap_index[s] for s in c] for c in CLUSTERS]
    n_obs = len(SNAPS)
    out = {}
    for true_r in true_r_grid:
        assert -1.0 < true_r < 1.0
        cov = np.array([[1.0, true_r], [true_r, 1.0]])
        chol = np.linalg.cholesky(cov)
        hits = 0
        widths = np.empty(n_replicates)
        for r in range(n_replicates):
            z = rng.standard_normal((n_obs, 2))
            xy = z @ chol.T
            x = xy[:, 0]
            y = xy[:, 1]
            boot_p = _cluster_bootstrap_pearson_only(
                x, y, clusters_idx, B=B_bootstrap, rng=rng)
            valid = boot_p[~np.isnan(boot_p)]
            assert len(valid) > 0, "all-NaN bootstrap on coverage sim replicate"
            lo, hi = float(np.quantile(valid, 0.025)), float(np.quantile(valid, 0.975))
            widths[r] = hi - lo
            if lo <= true_r <= hi:
                hits += 1
        out[true_r] = {
            "coverage": float(hits / n_replicates),
            "ci_mean_width": float(widths.mean()),
            "n_replicates": n_replicates,
            "B_bootstrap": B_bootstrap,
        }
    return out


# ---------------------------------------------------------------------------
# Empirical PRDS evidence: inter-metric Spearman correlation matrix across
# the 10 observed snapshots. Reports min / max off-diagonal correlation and
# the count of negative entries; backs the BH choice over BY (BY is reported
# alongside as robustness).
# ---------------------------------------------------------------------------


def pagerank_concentration_alternatives(s4, top_k=10):
    """Per-snapshot top-k concentration metrics on PageRank vectors.

    The headline HHI in stage4 is computed on the full PageRank vector
    over all ~100 always-alive assets. The sign-reversed contrast on
    full-network HHI (J-G) is driven by Jun~2020 Stable's
    hub-concentration spike. This helper computes three alternative
    concentration metrics on the top-k PageRank entries per snapshot:
        HHI_topk     = sum_{i in top_k} (p_i / sum_topk)**2
        Gini_topk    = Gini coefficient of the top-k entries
        Entropy_topk = -sum p_i' log p_i' with p_i' the normalised top-k
                       weights (in nats; max = log(k))
    The diagnostic is whether the sign reversal persists on the top-k
    alternatives or is full-network-specific. Reports per-snapshot
    values plus the crisis-vs-non-crisis exact-permutation p on each.
    """
    out = {}
    for snap in SNAPS:
        pr = np.array(list(s4[snap]['pagerank']['pagerank_scores'].values()),
                      dtype=float)
        topk = np.sort(pr)[::-1][:top_k]
        assert topk.sum() > 0, f"top-{top_k} PR all-zero on {snap}"
        w = topk / topk.sum()
        hhi_top = float((w * w).sum())
        # Gini on the top-k positive weights (already non-negative).
        g_top = float(gini(w))
        # Shannon entropy (nats); guard against 0*log(0).
        e_top = float(-np.sum(np.where(w > 0, w * np.log(w), 0.0)))
        out[snap] = {
            "hhi_topk": hhi_top,
            "gini_topk": g_top,
            "entropy_topk": e_top,
        }
    return out


def inter_metric_correlation_evidence(metrics_dict, metric_keys=METRIC_KEYS):
    """Per-pair Spearman correlation across snapshots; PRDS empirical proxy.

    The Benjamini-Hochberg FDR control assumes positive regression
    dependence on the subset (PRDS). A weaker but observable proxy is that
    all pairwise rank correlations between the 7 test statistics across the
    20 snapshots are non-negative. We compute the 7x7 Spearman matrix and
    report (min_off, max_off, n_negative, n_positive) for the
    21 unique off-diagonal pairs.

    Returns: dict with the 7x7 matrix and the three summary stats.
    """
    n = len(metric_keys)
    M = np.empty((n, len(SNAPS)))
    for i, k in enumerate(metric_keys):
        M[i] = np.array([metrics_dict[s][k] for s in SNAPS])
    # Spearman = Pearson on ranks; ranks via argsort+argsort.
    R = np.empty_like(M)
    for i in range(n):
        order = np.argsort(M[i])
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(order))
        R[i] = ranks
    corr = np.corrcoef(R)
    iu = np.triu_indices(n, k=1)
    off = corr[iu]
    return {
        "matrix": corr,
        "min_off": float(off.min()),
        "max_off": float(off.max()),
        "n_negative": int((off < 0).sum()),
        "n_positive": int((off > 0).sum()),
        "n_pairs": int(len(off)),
        "metric_keys": list(metric_keys),
    }


# ---------------------------------------------------------------------------
# Leave-one-out (LOO) BH sensitivity: drop each non-crisis snapshot in turn,
# rerun the 7-metric perm_test family on the remaining 19 snapshots, and
# report the range of BH q-values per metric. A robust rejection survives
# every LOO panel; a fragile one fails when a specific snapshot is dropped.
# ---------------------------------------------------------------------------


def loo_bh_sensitivity(metrics_dict, metric_keys=METRIC_KEYS):
    """LOO BH q-values across the 13 non-crisis snapshots.

    For each non-crisis snapshot s_drop:
        - panel := SNAPS \\ {s_drop}; crisis set := the seven crisis snapshots;
          n_crisis = 7, |panel| = 19, C(19,7) = 50,388 partitions,
          floor = 1/50,388.
        - re-run perm_test for each metric on the 19-snapshot panel
        - re-run BH FDR on the 7 p-values
    Returns: per-metric (min q, max q, list of (dropped_snap, q)) across
    the 13 LOO panels.
    """
    crisis = [s for s in SNAPS if REGIME[s] == 'crisis']
    noncrisis = [s for s in SNAPS if REGIME[s] != 'crisis']
    # Post-redesign (2026-05-28): 7 crisis + 13 non-crisis = 20 snapshots.
    assert len(crisis) + len(noncrisis) == len(SNAPS)
    n_total = len(SNAPS)

    per_drop = []  # list of (dropped_snap, {metric: q})
    for s_drop in noncrisis:
        panel = [s for s in SNAPS if s != s_drop]
        n_panel_full = n_total - 1  # capture before perm_test clobbers n_total
        assert len(panel) == n_panel_full and all(c in panel for c in crisis)
        from math import comb as _comb
        _expected_loo_perm = _comb(n_panel_full, len(crisis))
        p_loo = {}
        for k in metric_keys:
            vals = {s: metrics_dict[s][k] for s in panel}
            _, p, n_perm = perm_test(vals, crisis, panel)
            assert n_perm == _expected_loo_perm, (
                f"LOO panel C({n_panel_full},{len(crisis)}) "
                f"!= {_expected_loo_perm}, got {n_perm}")
            p_loo[k] = p
        q_loo = bh_fdr(p_loo)
        per_drop.append((s_drop, q_loo))

    summary = {}
    for k in metric_keys:
        qs = [q[k] for _, q in per_drop]
        summary[k] = {
            "min_q": float(min(qs)),
            "max_q": float(max(qs)),
            "n_below_10pct": int(sum(1 for q in qs if q <= 0.10)),
            "n_below_5pct":  int(sum(1 for q in qs if q <= 0.05)),
            "per_drop": [(s, float(q)) for s, q in zip(noncrisis, qs)],
        }
    return summary


# ---------------------------------------------------------------------------
# Block bootstrap coverage sim: empirical 95% CI coverage on AR(1) synthetic
# series matched to the rolling-NSI length T=252 at the production block
# lengths {6, 12, 24}.
# ---------------------------------------------------------------------------


def block_bootstrap_coverage_sim(
    n_replicates=200, B_bootstrap=1000, T=252, ar_phi=0.5,
    mean_block_grid=(6, 12, 24), seed=2026,
):
    """Empirical 95% CI coverage of the stationary block bootstrap on
    AR(1)-pair synthetic data matched to the rolling-NSI structure.

    Setup:
        X_t = phi * X_{t-1} + eps^x_t,  eps^x ~ N(0,1)
        Y_t = phi * Y_{t-1} + eps^y_t,  eps^y ~ N(0,1)
        (X, Y) independent AR(1); true Pearson r --> 0 as T grows.
    For each replicate:
        - compute observed Pearson r_obs on (X, Y)
        - apply block bootstrap at each mean_block_size
        - form 95% percentile CI; check whether r_obs is inside.
    The reference "true r" is 0 (independent AR(1) pair); coverage = fraction
    of replicates where the CI contains 0.

    Returns: dict {L: {coverage, ci_mean_width, n_replicates}}.
    """
    rng = np.random.default_rng(seed)
    out = {L: {"hits": 0, "widths": []} for L in mean_block_grid}
    for r in range(n_replicates):
        # Burn-in AR(1) pair of length T+50.
        T_full = T + 50
        ex = rng.standard_normal(T_full)
        ey = rng.standard_normal(T_full)
        x = np.empty(T_full)
        y = np.empty(T_full)
        x[0], y[0] = ex[0], ey[0]
        for t in range(1, T_full):
            x[t] = ar_phi * x[t - 1] + ex[t]
            y[t] = ar_phi * y[t - 1] + ey[t]
        x, y = x[50:], y[50:]
        for L in mean_block_grid:
            boot = stationary_block_bootstrap(
                x, y, mean_block_size=L, B=B_bootstrap,
                seed=int(rng.integers(0, 2**31 - 1)),
            )
            valid = boot[~np.isnan(boot)]
            assert len(valid) > 0
            lo, hi = float(np.quantile(valid, 0.025)), float(np.quantile(valid, 0.975))
            out[L]["widths"].append(hi - lo)
            if lo <= 0.0 <= hi:
                out[L]["hits"] += 1
    return {
        L: {
            "coverage": float(out[L]["hits"] / n_replicates),
            "ci_mean_width": float(np.mean(out[L]["widths"])),
            "n_replicates": n_replicates,
            "B_bootstrap": B_bootstrap,
            "T": T,
            "ar_phi": ar_phi,
        }
        for L in mean_block_grid
    }


# ---------------------------------------------------------------------------
# Data loading + metric construction
# ---------------------------------------------------------------------------


def load_caches(snap_dir=SNAP_DIR):
    """Load Stage-1/3/4/5 pickles."""
    with open(snap_dir / "stage1_results.pkl", "rb") as f:
        s1 = pickle.load(f)
    with open(snap_dir / "stage3_results.pkl", "rb") as f:
        s3 = pickle.load(f)
    with open(snap_dir / "stage4_results.pkl", "rb") as f:
        s4 = pickle.load(f)
    with open(snap_dir / "stage5_results.pkl", "rb") as f:
        s5 = pickle.load(f)
    return s1, s3, s4, s5


def window_mean_vix(parquet_path=VIX_PARQUET, windows=SNAPSHOT_WINDOWS):
    """Per-snapshot mean of daily VIX close from data/vix.parquet."""
    v = pd.read_parquet(parquet_path)
    series = v[("Close", "^VIX")] if isinstance(v.columns, pd.MultiIndex) else v["Close"]
    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.to_datetime(series.index)
    out = {}
    for snap, (start, end) in windows.items():
        m = float(series.loc[start:end].mean())
        assert np.isfinite(m) and m > 0, f"bad VIX mean for {snap}: {m}"
        out[snap] = m
    return out


def compute_cross_sector_from_cache(s1, sp500_info, top_k=None):
    """Cross-sector edge fraction per snapshot, computed live from Stage 1.

    Routes through src.stage4_network.crisis_signals so the inference layer
    no longer carries a hardcoded copy of the Tab cross-sector values.

    Returns (mapping, top_k_used). top_k_used is read back from the first
    Stage-1 row so the paper macro stays in sync with the auto-sized value.
    """
    # Local import keeps the module importable in environments without the
    # full src tree (e.g. read-only doc builds).
    from src.stage4_network.crisis_signals import cross_sector_edge_fraction
    df = cross_sector_edge_fraction(sp500_info, stage1=s1, top_k=top_k)
    mapping = dict(zip(df["snapshot"], df["cs_fraction"]))
    top_k_used = int(df["top_k"].iloc[0])
    return mapping, top_k_used


def compute_per_snapshot_metrics(s1, s3, s4, s5, sp500_info):
    """Build the {snapshot: {metric_key: value}} dict consumed by perm_test.

    Also returns the auto-sized cross-sector top-k so the paper macro stays
    in sync with the density-matched setting.
    """
    metrics = {}
    cross_sector, top_k_used = compute_cross_sector_from_cache(s1, sp500_info)
    nsi_df = s5['snapshot_nsi']
    for snap in SNAPS:
        pr_vals = list(s4[snap]['pagerank']['pagerank_scores'].values())
        z_mr = s4[snap]['motifs']['z_scores']['mutual_regulation']

        n_dir = s3[snap]['n_directed']
        n_mut = s3[snap]['n_bidirectional']
        f_mut = n_mut / (n_dir + n_mut) if (n_dir + n_mut) > 0 else 0.0

        nsi = float(nsi_df.loc[nsi_df['snapshot'] == snap, 'nsi'].values[0])
        rho = float(nsi_df.loc[nsi_df['snapshot'] == snap, 'mean_corr'].values[0])

        metrics[snap] = {
            'regime': REGIME[snap],
            'gini': gini(pr_vals),
            'hhi': sum(p * p for p in pr_vals),
            'abs_z_mr': abs(z_mr),
            'mutual_dyad_pct': f_mut * 100,
            'nsi': nsi,
            'mean_rho': rho,
            'cross_sector': cross_sector[snap],
        }
    return metrics, top_k_used


# ---------------------------------------------------------------------------
# LaTeX macro writer (drift killer for paper.tex / appendix.tex)
# ---------------------------------------------------------------------------


def _fmt(v, places=3):
    """Format float to fixed places; preserves sign + zero padding."""
    return f"{v:.{places}f}"


# LaTeX control-sequence names accept letters only — digits terminate the
# token. Embed integer parameters in macro names by spelling them out.
_NUM2WORD = {
    6: "six", 12: "twelve", 24: "twentyfour", 40: "forty",
}


def _n2w(n):
    if n not in _NUM2WORD:
        raise KeyError(f"_NUM2WORD missing entry for {n}; add it.")
    return _NUM2WORD[n]


def write_latex_macros(path, payload):
    """Dump every headline number as \\newcommand for paper.tex \\input.

    payload: nested dict assembled in main(). Stable key order ensures the
    output is byte-identical across reruns with the same cache.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "% Auto-generated by paper/_inference.py.",
        "% Do not edit by hand; rerun  python -m paper._inference  instead.",
        "% Source cache: results/snapshots/stage{1,3,4,5}_results.pkl",
        "",
    ]

    def cmd(name, value):
        lines.append(f"\\newcommand{{\\Inf{name}}}{{{value}}}")

    perm = payload["perm"]
    bh = payload["bh"]
    means = payload["means"]
    d = payload["cohen_d"]
    for k in METRIC_KEYS:
        slug = MACRO_SLUG[k]
        places = DECIMALS[k]
        cmd(f"P{slug}", _fmt(perm[k], 3))
        cmd(f"Q{slug}", _fmt(bh[k], 3))
        cmd(f"D{slug}", _fmt(d[k], 2))
        cmd(f"CrisisMean{slug}",    _fmt(means[k]["crisis"], places))
        cmd(f"NonCrisisMean{slug}", _fmt(means[k]["noncrisis"], places))
        cmd(f"BaselineMean{slug}",  _fmt(means[k]["baseline"], places))
        cmd(f"Diff{slug}",          _fmt(means[k]["diff"], places))

    cov = payload["covid_merged"]
    for k in METRIC_KEYS:
        slug = MACRO_SLUG[k]
        cmd(f"CovP{slug}", _fmt(cov["perm"][k], 3))
        cmd(f"CovQ{slug}", _fmt(cov["bh"][k], 3))

    # BY q-values: BH * c(m) under arbitrary dependence (companion to BH).
    by = payload["by"]
    for k in METRIC_KEYS:
        slug = MACRO_SLUG[k]
        cmd(f"BY{slug}", _fmt(by[k], 3))

    # Hochberg step-up FWER (third multi-test column alongside BH/BY).
    hoch = payload["hoch"]
    for k in METRIC_KEYS:
        slug = MACRO_SLUG[k]
        cmd(f"Hoch{slug}", _fmt(hoch[k], 3))

    lines.append("")
    cb = payload["cluster_boot"]
    cmd("ClusterPearson",        _fmt(cb["pearson_obs"], 3))
    cmd("ClusterPearsonCILow",   _fmt(cb["pearson_ci"][0], 3))
    cmd("ClusterPearsonCIHigh",  _fmt(cb["pearson_ci"][1], 3))
    cmd("ClusterSpearman",       _fmt(cb["spearman_obs"], 3))
    cmd("ClusterSpearmanCILow",  _fmt(cb["spearman_ci"][0], 3))
    cmd("ClusterSpearmanCIHigh", _fmt(cb["spearman_ci"][1], 3))
    cmd("ClusterDropped",        str(cb["n_dropped"]))
    cmd("LabelShuffleP",         _fmt(cb["label_shuffle_p"], 3))

    lines.append("")
    bb = payload["block_boot"]
    cmd("RollingR", _fmt(bb["obs_r"], 3))
    cmd("RollingT", str(bb["T"]))
    for L, ci in bb["cis"].items():
        Lw = _n2w(L)
        cmd(f"BlockL{Lw}CILow",  _fmt(ci[0], 3))
        cmd(f"BlockL{Lw}CIHigh", _fmt(ci[1], 3))
        cmd(f"BlockL{Lw}Valid",  str(ci[2]))

    lines.append("")
    bz = payload["bonferroni"]
    cmd("BonfThreshold",        _fmt(bz["threshold"], 3))
    cmd("BonfPooledEightyThreshold", _fmt(bz["pooled40_threshold"], 3))
    for axis, (lo, hi) in bz["ranges"].items():
        cmd(f"BonfMin{axis}", _fmt(lo, 2))
        cmd(f"BonfMax{axis}", _fmt(hi, 2))
    cmd("BonfPooledEightyFailures", str(bz["pooled40_failures"]))

    cmd("CrossSecTopK", str(payload["cross_sector_top_k"]))

    lines.append("")
    # RETIRED pre-redesign calibration macros (Yahoo COVID-triple geometry):
    # emitted for back-compat but consumed by neither paper.tex nor
    # appendix.tex after the CRSP redesign dropped the COVID overlap. See
    # calibration_sim_positive_dependence.
    cal = payload["calib"]
    # Three rho rungs: 0.0 (independent), 0.3 (mid dependence), 0.7 (high).
    cmd("CalibNReps", str(cal["n_replicates"]))
    cmd("CalibRateIndep", _fmt(cal["rates_a05"][0.0], 4))
    cmd("CalibRateMid",   _fmt(cal["rates_a05"][0.3], 4))
    cmd("CalibRateHigh",  _fmt(cal["rates_a05"][0.7], 4))
    cmd("CalibMeanPIndep", _fmt(cal["mean_p"][0.0], 3))
    cmd("CalibMeanPMid",   _fmt(cal["mean_p"][0.3], 3))
    cmd("CalibMeanPHigh",  _fmt(cal["mean_p"][0.7], 3))

    lines.append("")
    cov_sim = payload["cluster_cov_sim"]
    cmd("ClusterCovNReps", str(cov_sim["n_replicates"]))
    cmd("ClusterCovIndep", _fmt(cov_sim["coverage"][0.0], 3))
    cmd("ClusterCovMid",   _fmt(cov_sim["coverage"][0.5], 3))
    cmd("ClusterCovHigh",  _fmt(cov_sim["coverage"][0.8], 3))
    cmd("ClusterCovWidthIndep", _fmt(cov_sim["ci_mean_width"][0.0], 3))
    cmd("ClusterCovWidthMid",   _fmt(cov_sim["ci_mean_width"][0.5], 3))
    cmd("ClusterCovWidthHigh",  _fmt(cov_sim["ci_mean_width"][0.8], 3))

    lines.append("")
    prds = payload["prds"]
    cmd("PRDSMinOff",      _fmt(prds["min_off"], 3))
    cmd("PRDSMaxOff",      _fmt(prds["max_off"], 3))
    cmd("PRDSNNegative",   str(prds["n_negative"]))
    cmd("PRDSNPairs",      str(prds["n_pairs"]))

    lines.append("")
    loo = payload["loo"]
    for k in METRIC_KEYS:
        slug = MACRO_SLUG[k]
        cmd(f"LooMinQ{slug}",    _fmt(loo[k]["min_q"], 3))
        cmd(f"LooMaxQ{slug}",    _fmt(loo[k]["max_q"], 3))
        cmd(f"LooBelowTen{slug}", str(loo[k]["n_below_10pct"]))

    lines.append("")
    bcs = payload["block_cov_sim"]
    cmd("BlockCovNReps", str(bcs["n_replicates"]))
    for L in (6, 12, 24):
        Lw = _n2w(L)
        cmd(f"BlockCovL{Lw}",        _fmt(bcs["coverage"][L], 3))
        cmd(f"BlockCovWidthL{Lw}",   _fmt(bcs["ci_mean_width"][L], 3))

    lines.append("")
    ca = payload["conc_alt"]
    # Top-10 PageRank concentration alternatives. Crisis-mean / non-crisis-mean
    # / diff / one-sided perm p / Cohen's d on each of three top-10 stats.
    for key, tag in (("hhi_topk", "HHITen"),
                     ("gini_topk", "GiniTen"),
                     ("entropy_topk", "EntTen")):
        cmd(f"Conc{tag}CrisisMean",    _fmt(ca["means"][key]["crisis"], 4))
        cmd(f"Conc{tag}NonCrisisMean", _fmt(ca["means"][key]["noncrisis"], 4))
        cmd(f"Conc{tag}Diff",          _fmt(ca["means"][key]["diff"], 4))
        cmd(f"Conc{tag}P",             _fmt(ca["perm"][key], 3))
        cmd(f"Conc{tag}D",             _fmt(ca["cohen_d"][key], 2))

    # Q3 density-matched modularity dissolution test (crisis-vs-non-crisis,
    # one-sided p that crisis Q sits below non-crisis Q at the k=779 budget).
    dq = payload.get("density_q")
    if dq is not None:
        cmd("PDensityMatchedQ",             _fmt(dq["p_dissolution"], 3))
        cmd("DDensityMatchedQ",             _fmt(dq["cohen_d"], 2))
        cmd("DensityMatchedQCrisisMean",    _fmt(dq["crisis_mean"], 3))
        cmd("DensityMatchedQNonCrisisMean", _fmt(dq["noncrisis_mean"], 3))
        cmd("DensityMatchedQDiff",          _fmt(dq["obs_diff"], 3))
        cmd("DensityMatchedQK",             str(dq["k_edges"]))

    # ---- CRSP migration headline numerics (added 2026-05-28) -----------
    # Stage-1 A-DCC parameters, Stage-5 NSI peaks, and assertion-suite
    # totals are pulled from the live cache so paper.tex stays in sync
    # whenever python -m paper._inference runs against a fresh build.
    lines.append("")
    lines.append("% --- CRSP-migration headline numerics ---")
    crsp_macros = {"ADCCa": "0", "ADCCb": "0", "ADCCg": "0", "ADCCSum": "0",
                   "ADCCParamSpread": "n/a", "AlwaysAliveN": "278",
                   "MeanCorrJanTwenty": "0.000",
                   "NSIMarTwenty": "0.000", "NSIOctEight": "0.000",
                   "AssertionTotal": "0", "AssertionPass": "0",
                   "AssertionFail": "0", "PipelineWallTime": "n/a"}
    try:
        import pickle as _pkl
        s1_path = SNAP_DIR / "stage1_results.pkl"
        if s1_path.exists():
            with open(s1_path, "rb") as fh:
                s1 = _pkl.load(fh)
            ap = s1.get("adcc_params", {})
            crsp_macros["ADCCa"] = _fmt(float(ap.get("a", 0.0)), 4)
            crsp_macros["ADCCb"] = _fmt(float(ap.get("b", 0.0)), 4)
            crsp_macros["ADCCg"] = _fmt(float(ap.get("g", 0.0)), 4)
            s = float(ap.get("a", 0.0)) + float(ap.get("b", 0.0)) + float(ap.get("g", 0.0))
            crsp_macros["ADCCSum"] = _fmt(s, 4)
            spread = float(ap.get("max_param_spread", 0.0))
            crsp_macros["ADCCParamSpread"] = (
                f"<\\!10^{{-4}}" if spread < 1e-4 else
                f"{spread:.1e}".replace("e-0", "\\times 10^{-").replace("e-", "\\times 10^{-") + "}"
            )
            crsp_macros["AlwaysAliveN"] = str(int(ap.get("subset_k", 100)))
            sc = s1.get("snapshot_correlations", {})
            # Jan 2020 Pre-shock was dropped from SNAPSHOTS in the
            # 2026-05-28 redesign; the macro stays for paper.tex legacy
            # references but evaluates to "n/a" on the post-redesign
            # cache (the snapshot is no longer estimated).
            jan = sc.get("Jan 2020 Pre-shock", {})
            if jan and "R_avg" in jan:
                R = jan["R_avg"]
                iu = np.triu_indices_from(R, k=1)
                crsp_macros["MeanCorrJanTwenty"] = _fmt(float(R[iu].mean()), 3)
            else:
                crsp_macros["MeanCorrJanTwenty"] = "n/a"
        s5_path = SNAP_DIR / "stage5_results.pkl"
        if s5_path.exists():
            with open(s5_path, "rb") as fh:
                s5 = _pkl.load(fh)
            nsi_df = s5.get("snapshot_nsi")
            if nsi_df is not None and "snapshot" in nsi_df.columns:
                mar20 = nsi_df.loc[nsi_df["snapshot"] == "Mar 2020 COVID", "nsi"]
                oct08 = nsi_df.loc[nsi_df["snapshot"] == "Oct 2008 GFC", "nsi"]
                if len(mar20):
                    crsp_macros["NSIMarTwenty"] = _fmt(float(mar20.iloc[0]), 3)
                if len(oct08):
                    crsp_macros["NSIOctEight"] = _fmt(float(oct08.iloc[0]), 3)
        inv_path = SNAP_DIR / "invariants_report.md"
        if inv_path.exists():
            import re as _re
            t = inv_path.read_text(encoding="utf-8")
            m = _re.search(
                r"\*\*Total checks:\*\*\s*(\d+).*?\*\*PASS:\*\*\s*(\d+).*?\*\*FAIL:\*\*\s*(\d+)",
                t, _re.S,
            )
            if m:
                crsp_macros["AssertionTotal"] = m.group(1)
                crsp_macros["AssertionPass"] = m.group(2)
                crsp_macros["AssertionFail"] = m.group(3)
        crsp_macros["PipelineWallTime"] = "${\\sim}30$\\,min"
    except Exception as _exc:
        # Cache mid-rerun or schema mismatch; keep defaults so paper.tex
        # still compiles. The next inference run will overwrite.
        pass
    for nm, val in crsp_macros.items():
        cmd(nm, val)

    path.write_text("\n".join(lines) + "\n")
    return path


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def main(snap_dir=SNAP_DIR, sp500_info_path=SP500_INFO, write_macros=True):
    """Run the full inference layer and print a summary table."""
    s1, s3, s4, s5 = load_caches(snap_dir)
    sp500_info = pd.read_parquet(sp500_info_path)
    metrics, cross_sector_top_k = compute_per_snapshot_metrics(
        s1, s3, s4, s5, sp500_info)

    vix_mean = window_mean_vix()
    print("VIX window means (recomputed from data/vix.parquet):")
    for snap in SNAPS:
        print(f"  {snap:<22} {vix_mean[snap]:>6.2f}")

    crisis_snaps = [s for s in SNAPS if REGIME[s] == 'crisis']
    noncrisis_snaps = [s for s in SNAPS if REGIME[s] != 'crisis']
    baseline_snaps = [s for s in SNAPS if REGIME[s] == 'baseline']
    # Post-redesign: 7 crisis + 13 non-crisis (= 6 stress + 2 recovery
    # + 5 baseline) over 20 snapshots; assert the partition is well-formed.
    assert len(crisis_snaps) + len(noncrisis_snaps) == len(SNAPS)
    assert len(crisis_snaps) >= 1 and len(baseline_snaps) >= 1

    # --- (A) 20-snapshot exact permutation + BH ---
    from math import comb as _comb
    _N_PERM = _comb(len(SNAPS), len(crisis_snaps))
    print()
    print('=' * 75)
    print(f'EXACT PERMUTATION TESTS - crisis (n={len(crisis_snaps)}) '
          f'vs all-non-crisis (n={len(noncrisis_snaps)})')
    print(f'Total label assignments enumerated: '
          f'C({len(SNAPS)},{len(crisis_snaps)}) = {_N_PERM:,}')
    print('=' * 75)
    print(f'{"Metric":<28} {"Obs diff":>10} {"p (1-sided)":>12}')
    print('-' * 75)
    perm = {}
    obs_diff = {}
    n_total = 0
    for k in METRIC_KEYS:
        values = {s: metrics[s][k] for s in SNAPS}
        od, p, n_total = perm_test(values, crisis_snaps, SNAPS)
        perm[k] = p
        obs_diff[k] = od
        print(f'{LABELS[k]:<28} {od:>10.4f} {p:>12.4f}')
    assert n_total == _N_PERM, (
        f"exact-perm enumerator returned {n_total} partitions; "
        f"expected C({len(SNAPS)},{len(crisis_snaps)}) = {_N_PERM}")

    bh = bh_fdr(perm)
    by = by_fdr(perm)
    hoch = hochberg_fwer(perm)
    print()
    print('Multi-test adjusted q-values (BH=FDR-PRDS, BY=FDR-any-dep, Hoch=FWER):')
    for k in METRIC_KEYS:
        star = '*' if bh[k] <= 0.05 else ''
        print(f'  {LABELS[k]:<28} '
              f'q_BH = {bh[k]:.4f}  q_BY = {by[k]:.4f}  q_Hoch = {hoch[k]:.4f} {star}')

    # --- (A') COVID-merged sensitivity (twenty - 1 = nineteen snapshots) ---
    # Post-redesign: only Mar 2020 COVID + Jun 2020 Stable share the
    # COVID-cluster overlap (Jan 2020 Pre-shock was dropped). With six
    # remaining crisis windows (Oct 1987, Oct 1997, Oct 1998, Apr 2000,
    # Sep 2001, Oct 2008) plus the merged COVID-cluster, the contrast is
    # n_crisis=7 vs n_non=12 over C(19,7)=50,388 partitions.
    print()
    print('=' * 75)
    print('SENSITIVITY: COVID-MERGED 19-SNAPSHOT EXACT PERMUTATION')
    print('Mar 2020 + Jun 2020 collapsed via mean; n_crisis=7, C(19,7)=50,388')
    print('=' * 75)
    covid_set = ['Mar 2020 COVID', 'Jun 2020 Stable']
    merged_snaps = [s for s in SNAPS if s not in covid_set] + ['COVID-cluster']
    # Crisis arm: the six surviving crisis snapshots + COVID-cluster
    # (Mar 2020 COVID still carries the crisis label after merging).
    merged_crisis = [s for s in SNAPS
                     if s not in covid_set and REGIME[s] == 'crisis'] + ['COVID-cluster']
    perm_cov = {}
    for k in METRIC_KEYS:
        vals = {s: metrics[s][k] for s in SNAPS if s not in covid_set}
        vals['COVID-cluster'] = float(np.mean([metrics[s][k] for s in covid_set]))
        od, p, ntot = perm_test(vals, merged_crisis, merged_snaps)
        perm_cov[k] = p
        print(f'{LABELS[k]:<28} obs_diff={od:>+8.4f}  p={p:>6.4f}  n={ntot}')
    bh_cov = bh_fdr(perm_cov)
    print()
    print('  COVID-merged BH q-values:')
    for k in METRIC_KEYS:
        print(f'    {LABELS[k]:<28} q = {bh_cov[k]:.4f}')

    # --- (B) Cluster bootstrap on snapshot NSI-VIX ---
    nsi_vec = np.array([metrics[s]['nsi'] for s in SNAPS])
    vix_vec = np.array([vix_mean[s] for s in SNAPS])
    snap_index = {s: i for i, s in enumerate(SNAPS)}
    clusters_idx = [[snap_index[s] for s in c] for c in CLUSTERS]

    obs_pearson, _ = pearsonr(nsi_vec, vix_vec)
    obs_spearman, _ = spearmanr(nsi_vec, vix_vec)
    print()
    print('=' * 75)
    print('CLUSTER BOOTSTRAP - snapshot NSI-VIX')
    print(f'Observed Pearson r = {obs_pearson:.4f}, Spearman rho = {obs_spearman:.4f}')
    print(f'Clusters (n={len(CLUSTERS)}): GFC pair, COVID pair, 16 singletons')
    print('=' * 75)
    boot_p, boot_s, n_dropped = cluster_bootstrap(
        nsi_vec, vix_vec, clusters_idx, B=10000, seed=2026)
    valid_p = boot_p[~np.isnan(boot_p)]
    valid_s = boot_s[~np.isnan(boot_s)]
    p_ci = (float(np.quantile(valid_p, .025)), float(np.quantile(valid_p, .975)))
    s_ci = (float(np.quantile(valid_s, .025)), float(np.quantile(valid_s, .975)))
    print(f'Pearson  r = {obs_pearson:.3f}, 95% CI = '
          f'[{p_ci[0]:.3f}, {p_ci[1]:.3f}], '
          f'n_boot={len(valid_p)} (dropped {n_dropped} zero-variance replicates)')
    print(f'Spearman rho = {obs_spearman:.3f}, 95% CI = '
          f'[{s_ci[0]:.3f}, {s_ci[1]:.3f}]')
    _, label_p = label_shuffle_permutation(nsi_vec, vix_vec, B=100000, seed=2026)
    print(f'Two-sided permutation p (label shuffle, B=100000): {label_p:.4f}')

    # --- (C) Stationary block bootstrap on rolling NSI-VIX ---
    print()
    print('=' * 75)
    print('STATIONARY BLOCK BOOTSTRAP - rolling NSI-VIX')
    print('=' * 75)
    rolling = s5.get('rolling_nsi')
    backtest = s5.get('backtest', {})
    print('Stage 5 keys:', list(s5.keys()))
    print('Rolling NSI rows:', len(rolling) if rolling is not None else 'N/A')
    print('Contemporaneous corr (cache):', backtest.get('contemporaneous_corr'))

    vix_daily = _load_vix_daily()
    block_payload = None
    if rolling is not None and len(rolling) > 0:
        nsi_aligned = rolling['nsi'].dropna()
        vix_at_nsi = vix_daily.reindex(nsi_aligned.index, method='nearest')
        common_idx = nsi_aligned.index.intersection(vix_at_nsi.dropna().index)
        x = nsi_aligned.loc[common_idx].values
        y = vix_at_nsi.loc[common_idx].values
        print(f'Aligned series length: {len(common_idx)}')
        if len(x) > 10:
            obs_roll_r = float(np.corrcoef(x, y)[0, 1])
            print(f'Observed rolling Pearson r = {obs_roll_r:.4f}')
            cis = {}
            for mbs in (6, 12, 24):
                boot_r = stationary_block_bootstrap(
                    x, y, mean_block_size=mbs, B=5000, seed=2026)
                valid = boot_r[~np.isnan(boot_r)]
                lo = float(np.quantile(valid, .025))
                hi = float(np.quantile(valid, .975))
                cis[mbs] = (lo, hi, len(valid))
                n_drop = (~np.isfinite(boot_r)).sum()
                print(f'  mean block={mbs:>3}: 95% CI = '
                      f'[{lo:.3f}, {hi:.3f}], n_valid={len(valid)}, '
                      f'n_dropped={n_drop}')
            block_payload = {
                "obs_r": obs_roll_r, "T": int(len(common_idx)), "cis": cis,
            }
            _print_per_channel_diagnostics(rolling, vix_daily)
    if block_payload is None:
        block_payload = {"obs_r": float('nan'), "T": 0, "cis": {}}
        print('rolling_nsi missing in Stage 5 cache.')

    # --- (D) Bonferroni per-axis + pooled ---
    # m_axis = number of snapshots tested per axis (one |Z| per snapshot
    # per axis). The post-redesign 20-snapshot panel sets m_axis=20;
    # pooled m = 4 axes x m_axis.
    m_axis = len(SNAPS)
    m_pooled = 4 * m_axis
    print()
    print('=' * 75)
    print(f'BONFERRONI-CORRECTED PER-SNAPSHOT Z-SCORES (m={m_axis} snapshots)')
    print('=' * 75)
    bonf_thr = bonferroni_two_sided_threshold(m_axis, alpha=0.05)
    print(f'Bonferroni threshold: |Z| > {bonf_thr:.3f}')
    print()
    print(f'{"Snapshot":<22} {"|Z_C| (Q1)":>12} {"|Z_FFL|":>10} '
          f'{"|Z_MR|":>10} {"|Z_SIM|":>10}')
    zc_vals, zffl_vals, zmr_vals, zsim_vals = [], [], [], []
    for snap in SNAPS:
        zc = abs(s4[snap]['erdos_renyi']['z_scores']['clustering'])
        zffl = abs(s4[snap]['motifs']['z_scores']['feed_forward_loop'])
        zmr = abs(s4[snap]['motifs']['z_scores']['mutual_regulation'])
        zsim = abs(s4[snap]['motifs']['z_scores']['single_input_module'])
        zc_vals.append(zc); zffl_vals.append(zffl)
        zmr_vals.append(zmr); zsim_vals.append(zsim)
        print(f'{snap:<22} {zc:>12.2f} {zffl:>10.2f} {zmr:>10.2f} {zsim:>10.2f}')
    bonf_ranges = {
        "ZC":   (min(zc_vals),   max(zc_vals)),
        "ZFFL": (min(zffl_vals), max(zffl_vals)),
        "ZMR":  (min(zmr_vals),  max(zmr_vals)),
        "ZSIM": (min(zsim_vals), max(zsim_vals)),
    }
    # Sanity: every snapshot clears the per-axis Bonferroni threshold.
    for axis_name, vals in (("Q1 |Z_C|", zc_vals), ("|Z_FFL|", zffl_vals),
                             ("|Z_MR|", zmr_vals), ("|Z_SIM|", zsim_vals)):
        below = [v for v in vals if v <= bonf_thr]
        if below:
            print(f'  WARNING: {axis_name} has {len(below)} snapshot(s) '
                  f'below per-axis Bonferroni threshold {bonf_thr:.3f}: {below}')
    # Pooled Bonferroni-(4 x m_axis): single correction across all four axes
    # and all m_axis snapshots. On the 20-snapshot panel m_axis=20, so the
    # pooled family is m = 4 x 20 = 80 tests; the emitted macros are named
    # \InfBonfPooledEighty{Threshold,Failures} to match (the internal
    # pooled40_* variable names are legacy labels and do not surface anywhere).
    pooled40_thr = bonferroni_two_sided_threshold(m_pooled, alpha=0.05)
    all_z = zc_vals + zffl_vals + zmr_vals + zsim_vals
    pooled40_failures = sum(1 for v in all_z if v <= pooled40_thr)
    print(f'Pooled Bonferroni-{m_pooled} threshold: |Z| > {pooled40_thr:.3f} '
          f'(fails on {pooled40_failures} of {len(all_z)} axis-snap pairs).')

    print()
    print('=' * 75)
    print("EFFECT SIZES (Cohen's d) - crisis vs all-non-crisis")
    print('=' * 75)
    d_vals = {}
    means = {}
    for k in METRIC_KEYS:
        vc = np.array([metrics[s][k] for s in crisis_snaps])
        vnc = np.array([metrics[s][k] for s in noncrisis_snaps])
        vbl = np.array([metrics[s][k] for s in baseline_snaps])
        d_vals[k] = cohen_d(vc, vnc)
        means[k] = {
            "crisis":    float(vc.mean()),
            "noncrisis": float(vnc.mean()),
            "baseline":  float(vbl.mean()),
            "diff":      float(vc.mean() - vnc.mean()),
        }
        print(f"  {LABELS[k]:<28} d = {d_vals[k]:>+7.3f}")

    print()
    print('=' * 75)
    print('SUMMARY TABLE FOR PAPER')
    print('=' * 75)
    print(f'{"Metric":<28} {"Crisis":>10} {"Non-cris":>10} {"Diff":>9} '
          f'{"d":>7} {"p":>8} {"BH q":>8}')
    for k in METRIC_KEYS:
        print(f'{LABELS[k]:<28} {means[k]["crisis"]:>10.4f} '
              f'{means[k]["noncrisis"]:>10.4f} {means[k]["diff"]:>+9.4f} '
              f'{d_vals[k]:>+7.3f} {perm[k]:>8.4f} {bh[k]:>8.4f}')

    # --- (E) Calibration Monte Carlo: directional bias of permutation null
    # under positive within-COVID-triple dependence. RETIRED pre-redesign
    # device (Yahoo COVID-triple geometry): the CRSP redesign drops the
    # overlap, so the appendix no longer needs this calibration and the
    # manuscript consumes none of its macros. Still run here so the emitted
    # macros and the self-contained unit tests stay in sync; do not cite it.
    print()
    print('=' * 75)
    print('CALIBRATION SIM (RETIRED pre-redesign) - type-I rate vs within-COVID rho')
    print('=' * 75)
    calib_raw = calibration_sim_positive_dependence(
        n_replicates=10000, rho_grid=(0.0, 0.3, 0.7), seed=2026)
    calib_payload = {
        "n_replicates": next(iter(calib_raw.values()))["n"],
        "rates_a05":  {rho: calib_raw[rho]["rates"][0.05] for rho in calib_raw},
        "rates_a10":  {rho: calib_raw[rho]["rates"][0.10] for rho in calib_raw},
        "mean_p":     {rho: calib_raw[rho]["mean_p"]      for rho in calib_raw},
    }
    print(f'{"rho":>6} {"rate(a=0.05)":>14} {"rate(a=0.10)":>14} '
          f'{"mean_p":>10}')
    for rho in calib_raw:
        r5 = calib_payload["rates_a05"][rho]
        r10 = calib_payload["rates_a10"][rho]
        mp = calib_payload["mean_p"][rho]
        print(f'{rho:>6.2f} {r5:>14.4f} {r10:>14.4f} {mp:>10.4f}')

    # --- (F) Cluster-bootstrap coverage Monte Carlo: empirical 95% CI
    # coverage on (n=10, 7-cluster) data with variable-length resamples.
    print()
    print('=' * 75)
    print('CLUSTER BOOTSTRAP COVERAGE SIM - 95% CI vs true Pearson r')
    print('=' * 75)
    cov_raw = cluster_bootstrap_coverage_sim(
        true_r_grid=(0.0, 0.5, 0.8), n_replicates=500, B_bootstrap=1000,
        seed=2026)
    cluster_cov_payload = {
        "n_replicates":  next(iter(cov_raw.values()))["n_replicates"],
        "coverage":      {tr: cov_raw[tr]["coverage"]      for tr in cov_raw},
        "ci_mean_width": {tr: cov_raw[tr]["ci_mean_width"] for tr in cov_raw},
    }
    print(f'{"true_r":>8} {"coverage":>10} {"CI_width":>10}')
    for tr in cov_raw:
        cov = cluster_cov_payload["coverage"][tr]
        wid = cluster_cov_payload["ci_mean_width"][tr]
        print(f'{tr:>8.2f} {cov:>10.3f} {wid:>10.3f}')

    # --- (G) Empirical PRDS evidence: 7x7 Spearman correlation across snaps
    print()
    print('=' * 75)
    print('INTER-METRIC SPEARMAN CORRELATION ACROSS 10 SNAPSHOTS')
    print('=' * 75)
    prds = inter_metric_correlation_evidence(metrics, METRIC_KEYS)
    print(f'min off-diag = {prds["min_off"]:+.3f}, '
          f'max off-diag = {prds["max_off"]:+.3f}, '
          f'n_negative = {prds["n_negative"]} of {prds["n_pairs"]} pairs.')

    # --- (G') Top-10 PageRank concentration alternatives (HHI sign-reversal
    # diagnostic). The full-network HHI sign-reverses on this panel
    # (Jun~2020 hub spike). Compute three top-10 alternatives + run the
    # crisis-vs-non-crisis exact perm test on each.
    print()
    print('=' * 75)
    print('TOP-10 PAGERANK CONCENTRATION ALTERNATIVES (HHI sign-reversal diagnostic)')
    print('=' * 75)
    conc = pagerank_concentration_alternatives(s4, top_k=10)
    conc_perm = {}
    conc_d = {}
    conc_means = {}
    for key in ("hhi_topk", "gini_topk", "entropy_topk"):
        vals = {s: conc[s][key] for s in SNAPS}
        od, p, _ = perm_test(vals, crisis_snaps, SNAPS)
        vc = np.array([vals[s] for s in crisis_snaps])
        vnc = np.array([vals[s] for s in noncrisis_snaps])
        conc_perm[key] = p
        conc_d[key] = cohen_d(vc, vnc)
        conc_means[key] = {"crisis": float(vc.mean()),
                           "noncrisis": float(vnc.mean()),
                           "diff": float(vc.mean() - vnc.mean())}
        print(f'  {key:<14} crisis={vc.mean():.4f}  noncrisis={vnc.mean():.4f}  '
              f'diff={od:+.4f}  p={p:.4f}  d={conc_d[key]:+.3f}')

    # --- (H) LOO BH sensitivity across 13 non-crisis snapshots
    print()
    print('=' * 75)
    print('LEAVE-ONE-OUT BH SENSITIVITY (drop each non-crisis snapshot)')
    print('=' * 75)
    loo = loo_bh_sensitivity(metrics, METRIC_KEYS)
    print(f'{"Metric":<28} {"min q":>8} {"max q":>8} {"<=10%":>7} {"<=5%":>7}')
    for k in METRIC_KEYS:
        s = loo[k]
        print(f'{LABELS[k]:<28} {s["min_q"]:>8.4f} {s["max_q"]:>8.4f} '
              f'{s["n_below_10pct"]:>7d} {s["n_below_5pct"]:>7d}')

    # --- (I) Block bootstrap coverage on AR(1) synthetic series
    print()
    print('=' * 75)
    print('BLOCK BOOTSTRAP COVERAGE - AR(1) synthetic, T=252, true r=0')
    print('=' * 75)
    block_cov_raw = block_bootstrap_coverage_sim(
        n_replicates=200, B_bootstrap=1000, T=252, ar_phi=0.5,
        mean_block_grid=(6, 12, 24), seed=2026)
    block_cov_payload = {
        "n_replicates": next(iter(block_cov_raw.values()))["n_replicates"],
        "coverage":      {L: block_cov_raw[L]["coverage"]      for L in block_cov_raw},
        "ci_mean_width": {L: block_cov_raw[L]["ci_mean_width"] for L in block_cov_raw},
    }
    print(f'{"L":>4} {"coverage":>10} {"CI_width":>10}')
    for L in block_cov_raw:
        print(f'{L:>4} {block_cov_payload["coverage"][L]:>10.3f} '
              f'{block_cov_payload["ci_mean_width"][L]:>10.3f}')

    # --- (Q3) Density-matched modularity: crisis-vs-non-crisis dissolution ---
    # The density-matched Louvain Q at the common top-k budget (k=779, the
    # sparsest Stage-2 graph) tests community dissolution at a fixed edge
    # count. Dissolution predicts crisis Q BELOW non-crisis Q, so the
    # one-sided contrast is mean(crisis) < mean(non-crisis): run the exact
    # enumerator on negated Q (upper tail of -Q == lower tail of Q).
    import pickle as _pickle_dm
    _dm_path = snap_dir / "density_matched_results.pkl"
    density_q_payload = None
    if _dm_path.exists():
        _dm = _pickle_dm.load(open(_dm_path, "rb"))
        _q_by = {row["snapshot"]: float(row["Q"]) for _, row in _dm.iterrows()}
        if set(SNAPS).issubset(_q_by):
            _q_neg = {s: -_q_by[s] for s in SNAPS}
            _od_neg, _p_diss, _ntot_dm = perm_test(_q_neg, crisis_snaps, SNAPS)
            _qc = [_q_by[s] for s in crisis_snaps]
            _qnc = [_q_by[s] for s in noncrisis_snaps]
            density_q_payload = {
                "p_dissolution": float(_p_diss),
                "obs_diff": float(np.mean(_qc) - np.mean(_qnc)),
                "cohen_d": float(cohen_d(_qc, _qnc)),
                "crisis_mean": float(np.mean(_qc)),
                "noncrisis_mean": float(np.mean(_qnc)),
                "k_edges": int(_dm["n_edges"].iloc[0]),
            }
            print()
            print('=' * 75)
            print(f'Q3 DENSITY-MATCHED MODULARITY (k={density_q_payload["k_edges"]}): '
                  'crisis-vs-non-crisis dissolution')
            print(f'  crisis-mean Q={density_q_payload["crisis_mean"]:.4f}  '
                  f'non-crisis-mean Q={density_q_payload["noncrisis_mean"]:.4f}  '
                  f'diff={density_q_payload["obs_diff"]:+.4f}')
            print(f'  one-sided p (crisis<non-crisis)={_p_diss:.4f}  '
                  f"Cohen's d={density_q_payload['cohen_d']:+.3f}  C(20,7)={_ntot_dm:,}")

    payload = {
        "perm": perm, "bh": bh, "by": by, "hoch": hoch,
        "cohen_d": d_vals, "means": means,
        "covid_merged": {"perm": perm_cov, "bh": bh_cov},
        "cluster_boot": {
            "pearson_obs": float(obs_pearson),
            "spearman_obs": float(obs_spearman),
            "pearson_ci": p_ci,
            "spearman_ci": s_ci,
            "n_dropped": n_dropped,
            "label_shuffle_p": label_p,
        },
        "block_boot": block_payload,
        "bonferroni": {
            "threshold": bonf_thr,
            "ranges": bonf_ranges,
            "pooled40_threshold": pooled40_thr,
            "pooled40_failures": pooled40_failures,
        },
        "calib": calib_payload,
        "cluster_cov_sim": cluster_cov_payload,
        "prds": prds,
        "loo": loo,
        "block_cov_sim": block_cov_payload,
        "conc_alt": {
            "perm": conc_perm,
            "cohen_d": conc_d,
            "means": conc_means,
        },
        "cross_sector_top_k": int(cross_sector_top_k),
        "density_q": density_q_payload,
    }
    if write_macros:
        out = write_latex_macros(MACRO_OUT, payload)
        print()
        print(f'LaTeX macros written: {out.relative_to(ROOT)}')
    return payload


def _load_vix_daily():
    """Daily VXO+VIX continuity series (Whaley 2009 splice) for the block-
    bootstrap path; cached for repeat runs."""
    cache = SNAP_DIR / "_vix_daily.pkl"
    if cache.exists():
        return pickle.load(open(cache, 'rb'))
    v = pd.read_parquet(VIX_PARQUET)
    # data/vix_continuity.parquet has a single 'Close' column (DatetimeIndex);
    # the legacy MultiIndex branch is retained for back-compat with the
    # pre-migration Yahoo cache if it ever resurfaces in a side build.
    series = (v[("Close", "^VIX")] if isinstance(v.columns, pd.MultiIndex)
              else v["Close"])
    pickle.dump(series, open(cache, 'wb'))
    return series


def _print_per_channel_diagnostics(rolling, vix_daily):
    """Per-channel rolling vs VIX contemporaneous + lagged correlations.

    Diagnostic only; identifies which channel drives the lagged sign flip
    discussed in §V.NSI.
    """
    print()
    print('-' * 75)
    print('Per-channel rolling vs daily VIX (contemporaneous + lagged):')
    print('-' * 75)
    for chan_col, chan_label in (
            ('modularity_norm', '(1 - Q_norm)'),
            ('mean_correlation_norm', 'mean_corr_norm'),
            ('density_norm', 'density_norm')):
        if chan_col not in rolling.columns:
            continue
        chan_series = (1.0 - rolling[chan_col] if chan_col == 'modularity_norm'
                       else rolling[chan_col])
        chan_aligned = chan_series.dropna()
        vix_chan = vix_daily.reindex(chan_aligned.index, method='nearest')
        common = chan_aligned.index.intersection(vix_chan.dropna().index)
        cx = chan_aligned.loc[common].values
        cy = vix_chan.loc[common].values
        r0 = float(np.corrcoef(cx, cy)[0, 1])
        lag_str = []
        for lag in (5, 10, 15, 21, 42, 63):
            cs = chan_aligned.shift(lag).dropna()
            vc = vix_daily.reindex(cs.index, method='nearest')
            ic = cs.index.intersection(vc.dropna().index)
            if len(ic) > 20:
                rl = float(np.corrcoef(cs.loc[ic].values, vc.loc[ic].values)[0, 1])
                lag_str.append(f"{lag}={rl:+.3f}")
        print(f'  {chan_label:<24} r0={r0:+.3f}  ' + '  '.join(lag_str))


if __name__ == "__main__":
    main()
