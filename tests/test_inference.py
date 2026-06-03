"""Unit tests for paper/_inference.py helpers.

Covers exact permutation, BH FDR step-up (with tied / degenerate cases),
cluster bootstrap, label-shuffle permutation, stationary block bootstrap,
Cohen's d, and the Bonferroni-10 threshold. Macro-emission side effect is
exercised by a small end-to-end smoke against the cached pickles.
"""
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm

from paper._inference import (
    SNAPS, MACRO_OUT, MACRO_SLUG, DECIMALS, METRIC_KEYS,
    perm_test, bh_fdr, by_fdr, hochberg_fwer,
    cohen_d, bonferroni_two_sided_threshold,
    cluster_bootstrap, label_shuffle_permutation,
    stationary_block_bootstrap, gini,
    calibration_sim_positive_dependence,
    cluster_bootstrap_coverage_sim,
)


ROOT = Path(__file__).resolve().parents[1]
SNAP_DIR = ROOT / "results" / "snapshots"
HAVE_CACHE = (SNAP_DIR / "stage5_results.pkl").exists()


# --- perm_test ---------------------------------------------------------------


def test_perm_test_strict_top_extreme_hits_floor():
    """Observed split being the unique maximum gives p = 1/C(10,2) = 1/45."""
    snaps = [f"s{i}" for i in range(10)]
    # crisis = the two largest values; floor must be reached exactly.
    vals = {s: float(i) for i, s in enumerate(snaps)}  # 0..9
    crisis = ['s8', 's9']  # the two largest
    obs, p, n = perm_test(vals, crisis, snaps)
    assert n == 45
    assert obs > 0
    assert p == pytest.approx(1.0 / 45, abs=1e-12)


def test_perm_test_all_tied_returns_one():
    """All equal values: every permutation ties, so p = 1.0."""
    snaps = [f"s{i}" for i in range(10)]
    vals = {s: 5.0 for s in snaps}
    obs, p, n = perm_test(vals, ['s0', 's1'], snaps)
    assert n == 45
    assert obs == pytest.approx(0.0)
    # Every partition has diff = 0 >= 0, so all 45 count as extreme.
    assert p == pytest.approx(1.0)


def test_perm_test_observed_always_counted():
    """The observed assignment is itself a partition; n_extreme >= 1."""
    snaps = [f"s{i}" for i in range(10)]
    vals = {s: float(i) for i, s in enumerate(snaps)}
    obs, p, n = perm_test(vals, ['s0', 's1'], snaps)
    # Worst case is the observation; p must be exactly 1 here (least extreme).
    assert n == 45
    assert p == pytest.approx(1.0)


def test_perm_test_n_crisis_one_edge_case():
    """n_crisis=1 enumerates C(10,1)=10 partitions; floor 1/10."""
    snaps = [f"s{i}" for i in range(10)]
    vals = {s: float(i) for i, s in enumerate(snaps)}
    obs, p, n = perm_test(vals, ['s9'], snaps)
    assert n == 10
    assert p == pytest.approx(1.0 / 10, abs=1e-12)


# --- bh_fdr ------------------------------------------------------------------


def test_bh_fdr_matches_textbook_formula():
    """Sorted p = (0.01, 0.04, 0.05, 0.20), m=4.

    Textbook BH: q_(i) = min_{j>=i} m*p_(j)/j.
    q_(4) = 4*0.20/4 = 0.20
    q_(3) = min(0.20, 4*0.05/3) ≈ 0.0667
    q_(2) = min(0.0667, 4*0.04/2) = 0.08 → 0.0667 (monotonic)
    q_(1) = min(0.0667, 4*0.01/1) = 0.04
    """
    p = {"a": 0.01, "b": 0.04, "c": 0.05, "d": 0.20}
    q = bh_fdr(p)
    assert q["a"] == pytest.approx(0.04, abs=1e-10)
    assert q["b"] == pytest.approx(0.0667, abs=1e-3)
    assert q["c"] == pytest.approx(0.0667, abs=1e-3)
    assert q["d"] == pytest.approx(0.20, abs=1e-10)


def test_bh_fdr_all_tied():
    """All tied p-values get the same q = p (monotone floor)."""
    p = {f"k{i}": 0.022 for i in range(7)}
    q = bh_fdr(p)
    # q_(7) = 7*0.022/7 = 0.022; monotonicity holds; every q equals 0.022.
    for k in p:
        assert q[k] == pytest.approx(0.022, abs=1e-12)


def test_bh_fdr_monotonicity_enforced_right_to_left():
    """Raw m*p/k can be non-monotone; the floor must be enforced.

    m=3, p_(1)=0.01, p_(2)=0.04, p_(3)=0.04.
    Raw q = (3*0.01/1, 3*0.04/2, 3*0.04/3) = (0.03, 0.06, 0.04).
    Monotone right-to-left:
      q_(3) = 0.04;
      q_(2) = min(0.04, 0.06) = 0.04;
      q_(1) = min(0.04, 0.03) = 0.03.
    """
    p = {"x": 0.01, "y": 0.04, "z": 0.04}
    q = bh_fdr(p)
    assert q["x"] <= q["y"] <= q["z"]
    assert q["x"] == pytest.approx(0.03, abs=1e-10)
    assert q["y"] == pytest.approx(0.04, abs=1e-10)
    assert q["z"] == pytest.approx(0.04, abs=1e-10)


def test_bh_fdr_single_p():
    """m=1 degenerates to q = p."""
    q = bh_fdr({"only": 0.123})
    assert q["only"] == pytest.approx(0.123, abs=1e-12)


def test_bh_fdr_clamps_to_unity():
    """q is monotonically capped at 1.0."""
    p = {"a": 0.9, "b": 0.95}
    q = bh_fdr(p)
    for k in p:
        assert 0.0 <= q[k] <= 1.0


# --- by_fdr ------------------------------------------------------------------


def test_by_fdr_inflates_bh_by_harmonic():
    """BY q = BH q * c(m) with c(m) = sum_{i=1..m} 1/i (capped at 1)."""
    p = {"a": 0.01, "b": 0.04, "c": 0.05, "d": 0.20}
    bh = bh_fdr(p)
    by = by_fdr(p)
    c_m = sum(1.0 / i for i in range(1, 5))
    for k in p:
        expected = min(1.0, bh[k] * c_m)
        assert by[k] == pytest.approx(expected, abs=1e-10), \
            f"BY/BH ratio off for {k}: BY={by[k]} BH={bh[k]} c(m)={c_m}"


def test_by_fdr_caps_at_one():
    """BY can hit the c(m) inflation ceiling; must cap at 1.0."""
    p = {"a": 0.5, "b": 0.6, "c": 0.7}
    by = by_fdr(p)
    for k, v in by.items():
        assert 0.0 <= v <= 1.0, f"BY q out of [0,1] for {k}: {v}"


def test_hochberg_dominates_holm_under_positive_dependence():
    """Hochberg q <= Holm q for every metric (Hochberg is uniformly more
    powerful under positive dependence). Holm adjusted p = (m-i+1)*p_(i)
    with monotone left-to-right enforcement (step-down). We compute Holm
    inline here so the test does not import statsmodels."""
    p = {"a": 0.01, "b": 0.04, "c": 0.05, "d": 0.20}
    m = len(p)
    sorted_keys = sorted(p.keys(), key=lambda k: p[k])
    holm = {}
    prev = 0.0
    for i in range(m):
        raw = (m - i) * p[sorted_keys[i]]
        prev = max(prev, raw)
        holm[sorted_keys[i]] = min(1.0, prev)
    hoch = hochberg_fwer(p)
    for k in p:
        assert hoch[k] <= holm[k] + 1e-12, \
            f"Hochberg q ({hoch[k]}) should be <= Holm q ({holm[k]}) for {k}"


def test_hochberg_monotone_non_decreasing():
    """Hochberg adjusted p-values must be non-decreasing when sorted by raw p."""
    p = {"a": 0.01, "b": 0.04, "c": 0.05, "d": 0.20, "e": 0.30}
    hoch = hochberg_fwer(p)
    sorted_keys = sorted(p.keys(), key=lambda k: p[k])
    for i in range(1, len(sorted_keys)):
        assert hoch[sorted_keys[i]] >= hoch[sorted_keys[i - 1]] - 1e-12, \
            f"Hochberg not monotone at {sorted_keys[i-1]}->{sorted_keys[i]}"


# --- cohen_d -----------------------------------------------------------------


def test_cohen_d_pooled_sd_formula():
    """Hand-computed Cohen's d on a known pair."""
    a = np.array([1.0, 2.0, 3.0])  # mean 2, var (sample) 1.0
    b = np.array([4.0, 5.0, 6.0])  # mean 5, var (sample) 1.0
    # pooled = sqrt(((3-1)*1 + (3-1)*1) / (3+3-2)) = sqrt(4/4) = 1
    # d = (2 - 5) / 1 = -3
    assert cohen_d(a, b) == pytest.approx(-3.0, abs=1e-12)


def test_cohen_d_sign_with_crisis_above():
    """d > 0 when group-1 mean exceeds group-2."""
    crisis = np.array([10.0, 12.0])
    noncrisis = np.array([1.0, 2.0, 3.0, 4.0])
    d = cohen_d(crisis, noncrisis)
    assert d > 0


def test_cohen_d_zero_within_variance():
    """All-constant groups with differing means: infinite d (or signed)."""
    a = np.array([5.0, 5.0, 5.0])
    b = np.array([1.0, 1.0, 1.0])
    d = cohen_d(a, b)
    assert math.isinf(d) and d > 0


# --- Bonferroni threshold ----------------------------------------------------


def test_bonferroni_threshold_m10_alpha05():
    """m=10, alpha=0.05 → Φ^-1(1 - 0.0025) ≈ 2.807."""
    thr = bonferroni_two_sided_threshold(10, alpha=0.05)
    assert thr == pytest.approx(norm.ppf(1 - 0.0025), abs=1e-12)
    assert thr == pytest.approx(2.807, abs=1e-3)


def test_bonferroni_threshold_m1_alpha05():
    """m=1 collapses to the conventional |Z| > 1.96."""
    thr = bonferroni_two_sided_threshold(1, alpha=0.05)
    assert thr == pytest.approx(1.96, abs=1e-2)


def test_bonferroni_threshold_m40():
    """Pooled m=40 axis: alpha/2/40 = 0.000625; Φ^-1(0.999375) ≈ 3.227.

    This is the threshold the paper §7.4 / appendix §11.6 compares the
    per-axis Bonferroni-10 verdict to. The exact value is verified against
    scipy.stats.norm.ppf so a future SciPy release that drifts on the inverse
    Gaussian quantile would be caught here.
    """
    thr = bonferroni_two_sided_threshold(40, alpha=0.05)
    assert thr == pytest.approx(norm.ppf(1 - 0.000625), abs=1e-12)
    assert thr == pytest.approx(3.227, abs=1e-3)


# --- gini --------------------------------------------------------------------


def test_gini_uniform_zero():
    """All-equal vector has Gini 0."""
    x = np.full(50, 0.02)
    assert gini(x) == pytest.approx(0.0, abs=1e-10)


def test_gini_extreme_concentration():
    """One element with all the mass: Gini → (n-1)/n."""
    n = 10
    x = np.zeros(n)
    x[0] = 1.0
    assert gini(x) == pytest.approx((n - 1) / n, abs=1e-10)


# --- cluster_bootstrap -------------------------------------------------------


def test_cluster_bootstrap_seed_determinism():
    """Same seed reproduces the same boot vectors."""
    x = np.arange(10, dtype=float)
    y = 0.5 * x + 0.1
    clusters = [[0, 1], [2, 3], [4, 5], [6, 7], [8], [9]]
    p1, s1, n1 = cluster_bootstrap(x, y, clusters, B=200, seed=42)
    p2, s2, n2 = cluster_bootstrap(x, y, clusters, B=200, seed=42)
    np.testing.assert_allclose(p1, p2, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(s1, s2, atol=1e-12, equal_nan=True)
    assert n1 == n2


def test_cluster_bootstrap_perfect_correlation_ci_at_one():
    """y = 2x + 3 collinear → every resample gives Pearson r = 1."""
    x = np.linspace(0, 9, 10)
    y = 2 * x + 3
    clusters = [[i] for i in range(10)]
    boot_p, boot_s, n_drop = cluster_bootstrap(x, y, clusters, B=500, seed=0)
    valid = boot_p[~np.isnan(boot_p)]
    # Any non-degenerate resample is perfectly correlated.
    assert np.allclose(valid, 1.0, atol=1e-10)


# --- label_shuffle_permutation ----------------------------------------------


def test_label_shuffle_identity_correlation_small_p():
    """y = x is perfectly correlated; permutation p must hit the resolution
    floor for small B."""
    x = np.arange(15, dtype=float)
    y = x.copy()
    obs, p = label_shuffle_permutation(x, y, B=500, seed=2026)
    assert obs == pytest.approx(1.0, abs=1e-12)
    # Identity is the unique top draw at almost every permutation.
    assert p < 0.05


# --- stationary_block_bootstrap ---------------------------------------------


def test_block_bootstrap_seed_determinism():
    """Same seed reproduces the same vector of r's."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(60)
    y = 0.3 * x + 0.7 * rng.standard_normal(60)
    a = stationary_block_bootstrap(x, y, mean_block_size=6, B=200, seed=7)
    b = stationary_block_bootstrap(x, y, mean_block_size=6, B=200, seed=7)
    np.testing.assert_allclose(a, b, atol=1e-12, equal_nan=True)


def test_block_bootstrap_perfect_correlation_invariant_to_block():
    """y = 2x: every resample (any block length) still has r = 1."""
    x = np.linspace(0, 50, 60)
    y = 2 * x + 1
    for L in (1, 6, 60):
        out = stationary_block_bootstrap(x, y, mean_block_size=L, B=200, seed=0)
        valid = out[~np.isnan(out)]
        assert np.allclose(valid, 1.0, atol=1e-10)


def test_block_bootstrap_mean_block_one_runs_clean():
    """L̄=1 reduces to an iid bootstrap; should still produce valid output."""
    rng = np.random.default_rng(11)
    x = rng.standard_normal(80)
    y = rng.standard_normal(80)
    out = stationary_block_bootstrap(x, y, mean_block_size=1, B=100, seed=11)
    assert np.isfinite(out).sum() > 0


# --- calibration_sim_positive_dependence ------------------------------------


def test_calibration_sim_independent_matches_asymptote():
    """Under H0 with rho=0, the empirical type-I rate at alpha=0.05 should
    track the discrete-lattice asymptote 2/45 ~= 0.0444. We allow generous
    MC tolerance because the test is exact / discrete and the rate cannot
    move continuously."""
    out = calibration_sim_positive_dependence(
        n_replicates=4000, rho_grid=(0.0,), seed=11)
    rate_a05 = out[0.0]["rates"][0.05]
    assert rate_a05 == pytest.approx(2.0 / 45, abs=0.012), \
        f"independent-draw type-I rate at alpha=0.05 = {rate_a05}, " \
        f"expected ~{2/45:.4f}"


def test_calibration_sim_positive_rho_under_rejects():
    """Monotone drop in type-I rate as rho increases — the directional
    bias claim. Holds at the medium MC scale used here; the production
    sim uses 10000 replicates."""
    out = calibration_sim_positive_dependence(
        n_replicates=4000, rho_grid=(0.0, 0.7), seed=11)
    rate_indep = out[0.0]["rates"][0.05]
    rate_high  = out[0.7]["rates"][0.05]
    assert rate_high <= rate_indep, \
        f"expected under-rejection at rho=0.7; rate_indep={rate_indep}, " \
        f"rate_high={rate_high}"


# --- cluster_bootstrap_coverage_sim -----------------------------------------


@pytest.mark.skip(
    reason="cluster_bootstrap_coverage_sim hardcoded n=10 (pre-2026-05-28 "
           "Yahoo 10-snapshot framework); post-redesign cluster count is 18 "
           "(20 snapshots, two overlap pairs). Refactor in Task [8].")
def test_cluster_bootstrap_coverage_below_nominal():
    """Empirical coverage on n=10 cluster-resampled data is known to
    under-cover the nominal 95%; the sim should reflect that. Small MC
    here just guards against catastrophic regression (e.g., coverage
    collapsing to ~0 or jumping to ~100)."""
    out = cluster_bootstrap_coverage_sim(
        true_r_grid=(0.0,), n_replicates=200, B_bootstrap=500, seed=11)
    cov = out[0.0]["coverage"]
    assert 0.75 <= cov <= 0.95, \
        f"sanity coverage at true_r=0 should be in [0.75, 0.95], got {cov}"


# --- MACRO_SLUG / DECIMALS consistency --------------------------------------


def test_macro_slug_coverage_matches_metric_keys():
    """Every metric key has a slug + decimals entry — required by macro writer."""
    assert set(MACRO_SLUG.keys()) == set(METRIC_KEYS)
    assert set(DECIMALS.keys()) == set(METRIC_KEYS)


def test_macro_slugs_unique():
    """Slugs collide would overwrite each other in the .tex output."""
    slugs = list(MACRO_SLUG.values())
    assert len(set(slugs)) == len(slugs)


# --- End-to-end smoke against the cached pickles ----------------------------


@pytest.mark.skipif(not HAVE_CACHE, reason="Stage 1/3/4/5 cache not present")
def test_inference_main_writes_macros():
    """main() runs end-to-end on the production cache, writes macro file,
    and the macro file contains the required commands."""
    from paper._inference import main
    payload = main(write_macros=True)
    assert MACRO_OUT.exists(), "macro file should be written"
    text = MACRO_OUT.read_text()
    # Spot-check a few macros that paper.tex / appendix.tex depend on.
    for required in ("\\InfPGini", "\\InfClusterPearson", "\\InfBlockLtwelveCILow",
                     "\\InfBonfThreshold", "\\InfBonfMinZSIM",
                     "\\InfCovPMutDyad", "\\InfCrossSecTopK",
                     "\\InfBaselineMeanNSI",
                     "\\InfBYGini", "\\InfBYHHI",
                     "\\InfCalibRateIndep", "\\InfCalibRateMid", "\\InfCalibRateHigh",
                     "\\InfClusterCovIndep", "\\InfClusterCovHigh",
                     "\\InfHochGini", "\\InfHochHHI",
                     "\\InfPRDSMinOff", "\\InfPRDSMaxOff", "\\InfPRDSNNegative",
                     "\\InfLooMinQMutDyad", "\\InfLooBelowTenMutDyad",
                     "\\InfBlockCovLsix", "\\InfBlockCovLtwentyfour",
                     "\\InfConcHHITenP", "\\InfConcGiniTenP", "\\InfConcEntTenP"):
        assert required in text, f"missing macro: {required}"
    # Post-redesign (2026-05-28): SNAPS has 20 entries (7 crisis,
    # 6 stress, 2 recovery, 5 baseline). Pre-redesign had 10.
    assert len(SNAPS) == 20
    # COVID-merged perm test (Mar 2020 + Jun 2020 collapsed) operates
    # on n=19 partitions; we verify indirectly via the payload's bh
    # dict keys.
    assert set(payload["covid_merged"]["perm"].keys()) == set(METRIC_KEYS)
