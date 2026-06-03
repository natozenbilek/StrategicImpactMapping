"""Numerical-invariant assertion suite for the five-stage pipeline.

Three families of checks run against ``results/snapshots/stage{1..5}_results.pkl``
and against synthetic inputs to the pipeline's pure helpers:

1. **Property tests** -- deterministic synthetic-input checks on
   :mod:`src.utils.dcc_core`, :mod:`src.stage2_precision.glasso_filter`,
   and :mod:`src.stage4_network.analysis` helpers (A-DCC intercept
   closed form, Q_t PSD preservation, R_t normalisation bounds, EBIC =
   closed-form BIC at gamma=0, partial-correlation formula, adjacency
   shape, Gini reference values, triadic-motif fixtures, dyad-preserving
   rewire degree/mutual-pair preservation).

2. **Cache-level checks** -- per stage, re-derive stored quantities
   from primary outputs and verify the derivation matches: A-DCC
   stationarity and intercept PSD; per-snapshot R_avg finiteness /
   symmetry / unit-diagonal / strict-PSD / near-PSD; GLASSO precision
   PSD, partial-corr / adjacency well-formedness, n_edges + density
   consistency, constrained-BIC k>=p floor, EBIC gamma-tier and
   shrinkage delta-tier matching the n/p rule; Stage 3 directed-adj
   non-negativity + n_directed + n_mutual + n_dropped = n_input edges;
   Stage 4 PageRank sums to 1, Gini/HHI/modularity/purity in admissible
   ranges, Louvain partition covers all nodes, motif counts non-negative
   and significance profile unit-norm; Stage 5 weight configurations
   sum to 1, NSI in [0,1] and recomputable from *_norm columns, each
   normalised channel in [0,1].

3. **Cross-stage consistency** -- per snapshot, R_avg / adjacency /
   directed_adj / Stage-4 graph share dimensions, and Stages 2-4 carry
   the Stage-1 ticker list verbatim.

Run with ``python -m src._assertions.invariants``. Emits
``results/snapshots/invariants_report.md`` plus a one-line PASS/FAIL
verdict to stdout. The 436-check suite catches regressions (silent NaN,
sign flip, weight drift) at the pickle boundary rather than downstream.
"""
from __future__ import annotations

import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import SNAPSHOTS_DIR
from src.stage2_precision.glasso_filter import (
    _gamma_for_ratio as _expected_gamma,
    _delta_for_ratio as _expected_delta,
)


# ---------------------------------------------------------------------
# Constants from the manuscript (cross-checked against config.py)
# ---------------------------------------------------------------------

PAPER_FALLBACK_SNAPSHOTS = {
    "Oct 2008 Peak", "Mar 2009 Recovery",
    "Jan 2020 Pre-shock", "Mar 2020 Peak", "Jun 2020 Stable",
}


TOL_DIAG = 1e-10
TOL_SYM = 1e-8
TOL_STRICT_PSD = -1e-6
TOL_NEAR_PSD = -5e-2
TOL_PR_SUM = 1e-6
TOL_NSI_RECOMPUTE = 1e-10


@dataclass
class Check:
    """One invariant outcome."""

    stage: str
    snapshot: str
    name: str
    passed: bool
    detail: str = ""


def _is_symmetric(M: np.ndarray, tol: float = TOL_SYM) -> bool:
    return np.allclose(M, M.T, atol=tol, rtol=0.0)


# =====================================================================
# 1. PROPERTY TESTS (synthetic inputs to pure helper functions)
# =====================================================================

def property_tests() -> list[Check]:
    """Run deterministic synthetic-input tests on the pipeline helpers."""
    out: list[Check] = []

    out += _property_tests_dcc_core()
    out += _property_tests_stage2()
    out += _property_tests_stage4()

    return out


def _property_tests_dcc_core() -> list[Check]:
    """Tests for :mod:`src.utils.dcc_core` primitives."""
    out: list[Check] = []

    from src.utils.dcc_core import (
        adcc_intercept, adcc_qt_update, normalize_qt_to_rt,
        adcc_loglikelihood_contribution,
    )

    rng = np.random.default_rng(2026)
    k = 5

    # Synthesise a PSD Qbar and Nbar.
    X = rng.standard_normal((200, k))
    Q_bar = (X.T @ X) / 200
    Nx = np.minimum(X, 0)
    N_bar = (Nx.T @ Nx) / 200

    a, b, g = 0.02, 0.95, 0.01

    # Property 1: adcc_intercept exactly equals (1-a-b) Qbar - g Nbar.
    Omega = adcc_intercept(a, b, g, Q_bar, N_bar)
    Omega_expected = (1 - a - b) * Q_bar - g * N_bar
    out.append(Check(
        "property", "dcc_core", "adcc_intercept_closed_form",
        bool(np.allclose(Omega, Omega_expected, atol=1e-12)),
        f"max |diff| = {np.max(np.abs(Omega - Omega_expected)):.2e}",
    ))

    # Property 2: adcc_qt_update preserves PSD when input is PSD and the
    # intercept is PSD. We construct a PSD Q_prev and a PSD intercept
    # (small a, b, g, Q_bar diagonal-dominant) and check the output.
    Q_prev = Q_bar.copy()
    z_prev = rng.standard_normal(k)
    Q_next = adcc_qt_update(Q_prev, z_prev, Omega, a, b, g)
    lam_min_next = float(np.linalg.eigvalsh((Q_next + Q_next.T) / 2).min())
    out.append(Check(
        "property", "dcc_core", "adcc_qt_update_preserves_PSD",
        bool(lam_min_next >= -1e-10),
        f"lambda_min(Q_t) = {lam_min_next:.2e} (intercept PSD: "
        f"{np.linalg.eigvalsh(Omega).min():.2e})",
    ))

    # Property 3: normalize_qt_to_rt yields unit diagonal, symmetric R_t
    # with off-diagonals in [-1, 1] (within tolerance for clean Q_t).
    R_t = normalize_qt_to_rt(Q_next)
    diag_ok = bool(np.allclose(np.diag(R_t), 1.0, atol=1e-12))
    sym_ok = _is_symmetric(R_t, tol=1e-12)
    off = R_t[np.triu_indices_from(R_t, k=1)]
    bounded = bool(np.all(np.abs(off) <= 1.0 + 1e-10))
    out += [
        Check("property", "dcc_core", "normalize_qt_to_rt_unit_diagonal",
              diag_ok, f"max |diag-1| = {np.max(np.abs(np.diag(R_t)-1)):.2e}"),
        Check("property", "dcc_core", "normalize_qt_to_rt_symmetric",
              sym_ok, f"max |R - R^T| = {np.max(np.abs(R_t - R_t.T)):.2e}"),
        Check("property", "dcc_core", "normalize_qt_to_rt_offdiag_bounded",
              bounded, f"max |off| = {np.max(np.abs(off)):.6f}"),
    ]

    # Property 4: adcc_loglikelihood_contribution returns a finite scalar
    # on a well-posed PD R_t and z, and returns None on a non-PD input.
    ll = adcc_loglikelihood_contribution(R_t, z_prev)
    out.append(Check(
        "property", "dcc_core", "adcc_loglikelihood_contribution_finite_on_PD",
        bool(ll is not None and np.isfinite(ll)),
        f"LL = {ll}",
    ))
    R_bad = -np.eye(k)
    ll_bad = adcc_loglikelihood_contribution(R_bad, z_prev)
    out.append(Check(
        "property", "dcc_core",
        "adcc_loglikelihood_contribution_None_on_non_PD",
        bool(ll_bad is None),
        f"LL on -I = {ll_bad}",
    ))

    return out


def _property_tests_stage2() -> list[Check]:
    """Tests for :mod:`src.stage2_precision.glasso_filter` helpers."""
    out: list[Check] = []

    from src.stage2_precision.glasso_filter import (
        compute_ebic, precision_to_partial_corr, build_adjacency_matrix,
    )

    rng = np.random.default_rng(2026)
    p = 6

    # Build a small toy Gaussian model: random PD precision Omega.
    # The PD lift (+5 I) keeps the spectrum well away from the boundary
    # so multivariate_normal does not warn under the rng draw below.
    Aglas = rng.standard_normal((p, p))
    Omega = Aglas @ Aglas.T + 5.0 * np.eye(p)
    Sigma = np.linalg.inv(Omega)
    n = 200
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        X = rng.multivariate_normal(np.zeros(p), Sigma, size=n)
        S = (X - X.mean(0)).T @ (X - X.mean(0)) / (n - 1)

    # Property 1: compute_ebic at gamma = 0 equals -2 LL + k log n, with
    # LL = (n/2)(log|Omega| - tr(S Omega)) and k = off-diagonal nonzero
    # count of Omega.
    ebic = compute_ebic(Omega, S, n_samples=n, gamma=0.0)
    sign, logdet = np.linalg.slogdet(Omega)
    ll = 0.5 * n * (logdet - np.trace(S @ Omega))
    mask = np.abs(Omega) > 1e-10
    np.fill_diagonal(mask, False)
    k_edges = int(mask.sum() // 2)
    bic_closed = -2 * ll + k_edges * np.log(n)
    out.append(Check(
        "property", "stage2", "compute_ebic_equals_BIC_at_gamma_zero",
        bool(abs(ebic - bic_closed) < 1e-8),
        f"ebic={ebic:.6f}, closed-form BIC={bic_closed:.6f}",
    ))

    # Property 2: compute_ebic returns +inf when Omega is not PD.
    Omega_bad = -np.eye(p)
    ebic_bad = compute_ebic(Omega_bad, S, n_samples=n, gamma=0.0)
    out.append(Check(
        "property", "stage2", "compute_ebic_inf_on_non_PD",
        bool(np.isinf(ebic_bad)),
        f"ebic(-I) = {ebic_bad}",
    ))

    # Property 3: precision_to_partial_corr has unit diagonal and
    # produces values consistent with -Omega_ij / sqrt(Omega_ii Omega_jj).
    Pc = precision_to_partial_corr(Omega)
    diag_ok = bool(np.allclose(np.diag(Pc), 1.0, atol=1e-12))
    sym_ok = _is_symmetric(Pc, tol=1e-12)
    expected_off = -Omega[0, 1] / np.sqrt(Omega[0, 0] * Omega[1, 1])
    val_ok = bool(abs(Pc[0, 1] - expected_off) < 1e-10)
    out += [
        Check("property", "stage2", "precision_to_partial_corr_unit_diagonal",
              diag_ok, f"max |diag-1| = {np.max(np.abs(np.diag(Pc)-1)):.2e}"),
        Check("property", "stage2", "precision_to_partial_corr_symmetric",
              sym_ok, f"max |Pc-Pc^T| = {np.max(np.abs(Pc-Pc.T)):.2e}"),
        Check("property", "stage2", "precision_to_partial_corr_off_diag_formula",
              val_ok, f"Pc[0,1]={Pc[0,1]:.6f}, expected={expected_off:.6f}"),
    ]

    # Property 4: build_adjacency_matrix is symmetric, zero-diagonal,
    # and respects the precision-support mask.
    A = build_adjacency_matrix(Pc, Omega)
    sym_ok = _is_symmetric(A, tol=1e-12)
    diag_ok = bool(np.allclose(np.diag(A), 0.0, atol=1e-12))
    nonneg_ok = bool(np.all(A >= 0))
    out += [
        Check("property", "stage2", "build_adjacency_matrix_symmetric",
              sym_ok, f"max |A-A^T| = {np.max(np.abs(A-A.T)):.2e}"),
        Check("property", "stage2", "build_adjacency_matrix_zero_diagonal",
              diag_ok, f"max |diag| = {np.max(np.abs(np.diag(A))):.2e}"),
        Check("property", "stage2", "build_adjacency_matrix_nonnegative",
              nonneg_ok, f"min entry = {np.min(A):.2e}"),
    ]

    return out


def _property_tests_stage4() -> list[Check]:
    """Tests for :mod:`src.stage4_network.analysis` helpers."""
    out: list[Check] = []

    import networkx as nx
    from src.stage4_network.analysis import (
        _gini_coefficient, count_triadic_motifs, _dyad_preserving_rewire,
    )

    # Property 1: _gini_coefficient returns 0 on a constant vector.
    g_const = _gini_coefficient([1.0, 1.0, 1.0, 1.0])
    out.append(Check(
        "property", "stage4", "gini_zero_on_constant_input",
        bool(abs(g_const) < 1e-12),
        f"gini([1,1,1,1]) = {g_const:.4e}",
    ))

    # Property 2: gini of (0, 0, 0, 1) is 0.75 by the standard formula.
    g_max = _gini_coefficient([0.0, 0.0, 0.0, 1.0])
    out.append(Check(
        "property", "stage4", "gini_known_value_extreme",
        bool(abs(g_max - 0.75) < 1e-10),
        f"gini([0,0,0,1]) = {g_max:.6f}, expected 0.75",
    ))

    # Property 3: count_triadic_motifs on a feed-forward loop fixture
    # returns one FFL (030T) triad with no MR (111D) or SIM (021D).
    G_ffl = nx.DiGraph()
    G_ffl.add_edges_from([(0, 1), (1, 2), (0, 2)])
    counts_ffl = count_triadic_motifs(G_ffl)
    out.append(Check(
        "property", "stage4", "count_triadic_motifs_FFL_fixture",
        bool(counts_ffl["feed_forward_loop"] == 1
             and counts_ffl["mutual_regulation"] == 0
             and counts_ffl["single_input_module"] == 0),
        f"counts = {counts_ffl}",
    ))

    # Property 4: count_triadic_motifs on a SIM fixture (A->B, A->C) has
    # one SIM (021D) and no FFL.
    G_sim = nx.DiGraph()
    G_sim.add_nodes_from([0, 1, 2])
    G_sim.add_edges_from([(0, 1), (0, 2)])
    counts_sim = count_triadic_motifs(G_sim)
    out.append(Check(
        "property", "stage4", "count_triadic_motifs_SIM_fixture",
        bool(counts_sim["single_input_module"] == 1
             and counts_sim["feed_forward_loop"] == 0),
        f"counts = {counts_sim}",
    ))

    # Property 5: _dyad_preserving_rewire preserves the in/out-degree
    # sequences and the mutual-dyad count of the empirical graph. We
    # build a denser graph so rewiring has non-trivial work to do.
    rng_seed = np.random.RandomState(2026)
    G_emp = nx.gnp_random_graph(20, 0.3, directed=True, seed=2026)
    # Add a handful of mutual dyads explicitly so the mutual-preservation
    # invariant has something to verify.
    pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    for u, v in pairs:
        if not G_emp.has_edge(u, v):
            G_emp.add_edge(u, v)
        if not G_emp.has_edge(v, u):
            G_emp.add_edge(v, u)

    in_emp = sorted(d for _, d in G_emp.in_degree())
    out_emp = sorted(d for _, d in G_emp.out_degree())
    mut_emp = sum(1 for u, v in G_emp.edges() if G_emp.has_edge(v, u)) // 2
    n_edges_emp = G_emp.number_of_edges()

    G_rw = _dyad_preserving_rewire(G_emp, rng=rng_seed)

    in_rw = sorted(d for _, d in G_rw.in_degree())
    out_rw = sorted(d for _, d in G_rw.out_degree())
    mut_rw = sum(1 for u, v in G_rw.edges() if G_rw.has_edge(v, u)) // 2

    out += [
        Check("property", "stage4", "dyad_rewire_preserves_in_degree_sequence",
              bool(in_emp == in_rw),
              f"|empirical| = {len(in_emp)}, identical: {in_emp == in_rw}"),
        Check("property", "stage4", "dyad_rewire_preserves_out_degree_sequence",
              bool(out_emp == out_rw),
              f"|empirical| = {len(out_emp)}, identical: {out_emp == out_rw}"),
        Check("property", "stage4", "dyad_rewire_preserves_mutual_count",
              bool(mut_emp == mut_rw),
              f"mutual empirical = {mut_emp}, rewired = {mut_rw}"),
        Check("property", "stage4", "dyad_rewire_preserves_edge_count",
              bool(n_edges_emp == G_rw.number_of_edges()),
              f"edges empirical = {n_edges_emp}, rewired = {G_rw.number_of_edges()}"),
    ]

    return out


# =====================================================================
# 2. CACHE-LEVEL CHECKS
# =====================================================================

def check_stage1(stage1: dict) -> list[Check]:
    """Stage-1 invariants against a loaded ``stage1_results.pkl``."""
    out: list[Check] = []

    # --- A-DCC global parameters ---------------------------------------
    p = stage1.get("adcc_params", {})
    a, b, g = p.get("a", np.nan), p.get("b", np.nan), p.get("g", np.nan)
    s = a + b + g
    out += [
        Check("stage1", "<global>", "adcc_params_positive",
              bool(a > 0 and b > 0 and g > 0),
              f"a={a:.4f}, b={b:.4f}, g={g:.4f}"),
        Check("stage1", "<global>", "adcc_stationarity",
              bool(s < 1.0),
              f"a+b+g = {s:.4f} (< 1)"),
    ]

    # A-DCC intercept consistency: the subset Qbar and Nbar plus the
    # estimated (a, b, g) should reproduce a PSD intercept.
    Q_bar_sub = p.get("Q_bar_subset")
    N_bar_sub = p.get("N_bar_subset")
    if Q_bar_sub is not None and N_bar_sub is not None:
        from src.utils.dcc_core import adcc_intercept
        Omega = adcc_intercept(a, b, g, Q_bar_sub, N_bar_sub)
        lam_min_intercept = float(np.linalg.eigvalsh((Omega + Omega.T) / 2).min())
        out.append(Check(
            "stage1", "<global>", "adcc_intercept_PSD",
            bool(lam_min_intercept >= -1e-6),
            f"lambda_min(Omega) = {lam_min_intercept:.2e}",
        ))

    # --- Per-asset GARCH parameters ------------------------------------
    garch_results = stage1.get("garch_results", {}) or {}
    if garch_results:
        n_violations_intercept = 0
        n_violations_persistence = 0
        n_assets = 0
        for ticker, res in garch_results.items():
            if not isinstance(res, dict) or "params" not in res:
                continue
            params = res["params"]
            omega = params.get("omega", np.nan)
            alpha = params.get("alpha", np.nan)
            beta = params.get("beta", np.nan)
            if np.isfinite(omega) and omega <= 0:
                n_violations_intercept += 1
            if (np.isfinite(alpha) and np.isfinite(beta)
                    and alpha + beta >= 1.0):
                n_violations_persistence += 1
            n_assets += 1
        out += [
            Check("stage1", "<global>", "garch_omega_positive",
                  bool(n_violations_intercept == 0),
                  f"{n_violations_intercept}/{n_assets} tickers violate omega > 0"),
            Check("stage1", "<global>", "garch_alpha_plus_beta_lt_1",
                  bool(n_violations_persistence == 0),
                  f"{n_violations_persistence}/{n_assets} tickers violate "
                  f"alpha + beta < 1"),
        ]

    # --- Per-snapshot R_avg --------------------------------------------
    snaps = stage1.get("snapshot_correlations", {}) or {}
    for label, sc in snaps.items():
        R = sc.get("R_avg")
        if R is None:
            out.append(Check("stage1", label, "R_avg_present", False,
                             "R_avg missing"))
            continue

        finite = bool(np.all(np.isfinite(R)))
        diag_ok = bool(np.allclose(np.diag(R), 1.0, atol=TOL_DIAG, rtol=0.0))
        sym = _is_symmetric(R)
        offdiag = R[np.triu_indices_from(R, k=1)]
        bounded = bool(np.all(np.abs(offdiag) <= 1.0 + 1e-6))
        lam_min = float(np.linalg.eigvalsh((R + R.T) / 2).min())
        strict_psd = bool(lam_min >= TOL_STRICT_PSD)
        near_psd = bool(lam_min >= TOL_NEAR_PSD)

        out += [
            Check("stage1", label, "R_avg_finite", finite,
                  "" if finite else f"{(~np.isfinite(R)).sum()} non-finite entries"),
            Check("stage1", label, "R_avg_unit_diagonal", diag_ok,
                  f"max |diag-1| = {np.max(np.abs(np.diag(R)-1)):.2e}"),
            Check("stage1", label, "R_avg_symmetric", sym,
                  f"max |R - R^T| = {np.max(np.abs(R - R.T)):.2e}"),
            Check("stage1", label, "R_avg_offdiag_bounded", bounded,
                  f"max |off-diag| = {np.max(np.abs(offdiag)):.4f}"),
            Check("stage1", label, "R_avg_strict_PSD", strict_psd,
                  f"lambda_min = {lam_min:.2e} (>= -1e-6?)"),
            Check("stage1", label, "R_avg_near_PSD", near_psd,
                  f"lambda_min = {lam_min:.2e} (>= -5e-2?)"),
        ]

    return out


def check_stage2(stage2: dict) -> list[Check]:
    """Stage-2 invariants against a loaded ``stage2_results.pkl``."""
    out: list[Check] = []

    for label, data in stage2.items():
        Omega = data.get("precision")
        Pc = data.get("partial_corr")
        A = data.get("adjacency")
        lam = data.get("lambda_opt")
        n_edges_stored = data.get("n_edges")
        density_stored = data.get("density")
        tickers = data.get("tickers") or []
        if Omega is None or Pc is None or A is None:
            out.append(Check("stage2", label, "precision_present", False,
                             "precision/partial_corr/adjacency missing"))
            continue

        p = Omega.shape[0]
        # The n_days field for n/p estimation is not stored in the
        # stage2 cache itself; the Stage-1 snapshot_correlations have
        # it. We use len(tickers) as p but n_days has to come from
        # the joint check; here we report shape-level invariants only.

        # Precision PD + symmetric.
        sym_prec = _is_symmetric(Omega)
        # slogdet of an extremely sparse + heavily-scaled precision can
        # overflow; suppress numpy's RuntimeWarning since the sign output
        # is what we read here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            sign, logdet = np.linalg.slogdet(Omega)
        out += [
            Check("stage2", label, "precision_symmetric", sym_prec,
                  f"max |Omega-Omega^T| = {np.max(np.abs(Omega-Omega.T)):.2e}"),
            Check("stage2", label, "precision_PD",
                  bool(sign == 1 and np.isfinite(logdet)),
                  f"slogdet sign={sign}, logdet={logdet:.2f}"),
        ]

        # partial_corr unit diagonal + symmetric.
        out += [
            Check("stage2", label, "partial_corr_unit_diagonal",
                  bool(np.allclose(np.diag(Pc), 1.0, atol=TOL_DIAG, rtol=0.0)),
                  f"max |diag-1| = {np.max(np.abs(np.diag(Pc)-1)):.2e}"),
            Check("stage2", label, "partial_corr_symmetric",
                  _is_symmetric(Pc),
                  f"max |Pc-Pc^T| = {np.max(np.abs(Pc-Pc.T)):.2e}"),
        ]

        # Adjacency invariants.
        adj_diag_zero = bool(np.allclose(np.diag(A), 0.0, atol=TOL_DIAG))
        adj_nonneg = bool(np.all(A >= 0.0))
        out += [
            Check("stage2", label, "adjacency_symmetric", _is_symmetric(A),
                  f"max |A-A^T| = {np.max(np.abs(A-A.T)):.2e}"),
            Check("stage2", label, "adjacency_zero_diagonal", adj_diag_zero,
                  f"max |diag| = {np.max(np.abs(np.diag(A))):.2e}"),
            Check("stage2", label, "adjacency_nonnegative", adj_nonneg,
                  f"min entry = {np.min(A):.2e}"),
        ]

        # Edge count and density consistency.
        mask = np.abs(Omega) > 1e-10
        np.fill_diagonal(mask, False)
        k_recomp = int(mask.sum() // 2)
        density_recomp = k_recomp / (p * (p - 1) / 2)
        out += [
            Check("stage2", label, "n_edges_matches_precision",
                  bool(k_recomp == n_edges_stored),
                  f"stored={n_edges_stored}, recomputed={k_recomp}"),
            Check("stage2", label, "density_matches_n_edges",
                  bool(abs(density_recomp - density_stored) < 1e-10),
                  f"stored={density_stored:.6f}, recomputed={density_recomp:.6f}"),
        ]

        # Constrained-BIC floor invariant.
        out.append(Check(
            "stage2", label, "n_edges_meets_p_floor",
            bool(n_edges_stored >= p),
            f"n_edges={n_edges_stored}, p={p}, lambda*={lam:.4f}, "
            f"paper_labels_fallback={label in PAPER_FALLBACK_SNAPSHOTS}",
        ))

    return out


def check_stage2_against_stage1(stage1: dict, stage2: dict) -> list[Check]:
    """Joint checks requiring both Stage 1 and Stage 2 inputs."""
    out: list[Check] = []
    snaps = stage1.get("snapshot_correlations", {}) or {}

    for label, s2 in stage2.items():
        s1 = snaps.get(label)
        if s1 is None:
            continue
        p = s2["precision"].shape[0]
        n = s1["n_days"]
        ratio = n / p

        # EBIC tier: applied gamma must match the n/p tier rule. Falls
        # back to a stale-cache failure when applied_gamma is absent
        # (pre-2026-05-24 cache schema; re-run Stage 2 to populate).
        expected_g = _expected_gamma(ratio)
        applied_g = s2.get("applied_gamma")
        if applied_g is None:
            out.append(Check(
                "stage2", label, "ebic_gamma_tier_matches_npratio",
                False,
                f"applied_gamma absent (stale Stage-2 cache schema); "
                f"n/p = {ratio:.3f} -> expected gamma = {expected_g:.2f}",
            ))
        else:
            out.append(Check(
                "stage2", label, "ebic_gamma_tier_matches_npratio",
                bool(abs(applied_g - expected_g) < 1e-12),
                f"n/p = {ratio:.3f} -> expected gamma = {expected_g:.2f}, "
                f"applied = {applied_g:.2f}",
            ))

        # Shrinkage tier: applied delta must match the n/p tier rule.
        expected_delta = _expected_delta(ratio)
        applied_delta = s2.get("applied_delta")
        if applied_delta is None:
            out.append(Check(
                "stage2", label, "shrinkage_delta_tier_matches_npratio",
                False,
                f"applied_delta absent (stale Stage-2 cache schema); "
                f"n/p = {ratio:.3f} -> expected delta = {expected_delta:.3f}",
            ))
        else:
            out.append(Check(
                "stage2", label, "shrinkage_delta_tier_matches_npratio",
                bool(abs(applied_delta - expected_delta) < 1e-12),
                f"n/p = {ratio:.3f} -> expected delta = {expected_delta:.3f}, "
                f"applied = {applied_delta:.3f}",
            ))

    return out


def check_stage3(stage3: dict) -> list[Check]:
    """Stage-3 invariants against a loaded ``stage3_results.pkl``."""
    out: list[Check] = []

    for label, data in stage3.items():
        D = data.get("directed_adj")
        n_dir = data.get("n_directed")
        n_mut = data.get("n_bidirectional")
        n_drop = data.get("n_dropped")
        n_in = data.get("n_input_edges")
        if D is None:
            out.append(Check("stage3", label, "directed_adj_present", False,
                             "directed_adj missing"))
            continue

        out += [
            Check("stage3", label, "directed_adj_nonnegative",
                  bool(np.all(D >= 0)), f"min entry = {np.min(D):.2e}"),
            Check("stage3", label, "directed_adj_zero_diagonal",
                  bool(np.allclose(np.diag(D), 0.0, atol=TOL_DIAG)),
                  f"max |diag| = {np.max(np.abs(np.diag(D))):.2e}"),
        ]

        D_pos = D > 0
        oneway_mask = D_pos & ~D_pos.T
        n_directed_recomp = int(oneway_mask.sum())
        mutual_mask = D_pos & D_pos.T
        np.fill_diagonal(mutual_mask, False)
        n_mutual_recomp = int(mutual_mask.sum() // 2)

        out += [
            Check("stage3", label, "n_directed_consistent_with_directed_adj",
                  bool(n_dir == n_directed_recomp),
                  f"stored={n_dir}, recomputed={n_directed_recomp}"),
            Check("stage3", label, "n_bidirectional_consistent_with_directed_adj",
                  bool(n_mut == n_mutual_recomp),
                  f"stored={n_mut}, recomputed={n_mutual_recomp}"),
        ]

        s = n_dir + n_mut + n_drop
        out.append(Check(
            "stage3", label, "edge_count_conservation",
            bool(s == n_in),
            f"directed + mutual + dropped = {s}, input = {n_in}",
        ))

    return out


def check_stage4(stage4: dict) -> list[Check]:
    """Stage-4 invariants against a loaded ``stage4_results.pkl``."""
    out: list[Check] = []

    for label, data in stage4.items():
        pr = data.get("pagerank", {}) or {}
        comm = data.get("community", {}) or {}
        motifs = data.get("motifs")
        n_nodes = data.get("n_nodes")

        # PageRank stationary distribution sums to 1.
        pr_scores = pr.get("pagerank_scores")
        if isinstance(pr_scores, dict) and len(pr_scores) > 0:
            pr_sum = float(sum(pr_scores.values()))
            out.append(Check(
                "stage4", label, "pagerank_sums_to_one",
                bool(abs(pr_sum - 1.0) < TOL_PR_SUM),
                f"sum = {pr_sum:.8f}",
            ))

        # Range checks.
        gini = pr.get("gini")
        if gini is not None:
            out.append(Check("stage4", label, "gini_in_unit_interval",
                             bool(0.0 <= float(gini) <= 1.0),
                             f"gini = {gini:.4f}"))
        hhi = pr.get("hhi")
        if hhi is not None:
            out.append(Check("stage4", label, "hhi_in_unit_interval",
                             bool(0.0 < float(hhi) <= 1.0),
                             f"hhi = {hhi:.6f}"))
        Q = comm.get("modularity")
        if Q is not None:
            out.append(Check("stage4", label, "modularity_in_admissible_range",
                             bool(-0.5 <= float(Q) <= 1.0),
                             f"Q = {Q:.4f}"))
        purity = comm.get("purity")
        if purity is not None and not (isinstance(purity, float) and np.isnan(purity)):
            out.append(Check("stage4", label, "purity_in_unit_interval",
                             bool(0.0 <= float(purity) <= 1.0),
                             f"purity = {purity:.4f}"))

        # Community partition: sum of community sizes equals n_nodes.
        sizes = comm.get("community_sizes")
        if isinstance(sizes, dict) and n_nodes is not None:
            sum_sizes = sum(sizes.values())
            out.append(Check(
                "stage4", label, "community_partition_covers_nodes",
                bool(sum_sizes == n_nodes),
                f"sum of community sizes = {sum_sizes}, n_nodes = {n_nodes}",
            ))

        # Motif counts and Z-scores.
        if motifs is not None:
            ec = motifs.get("empirical_counts", {})
            if ec:
                all_int = all(isinstance(v, int) and v >= 0 for v in ec.values())
                out.append(Check(
                    "stage4", label, "motif_empirical_counts_non_negative_integers",
                    all_int, f"counts = {ec}",
                ))
            zs = motifs.get("z_scores", {})
            sp = motifs.get("significance_profile", {})
            z_finite = bool(all(np.isfinite(v) for v in zs.values())) if zs else False
            out.append(Check(
                "stage4", label, "motif_z_scores_finite",
                z_finite,
                f"Z = {dict((k, round(v, 2)) for k, v in zs.items())}",
            ))
            if sp:
                sp_vec = np.array(list(sp.values()), dtype=float)
                norm = float(np.linalg.norm(sp_vec))
                out.append(Check(
                    "stage4", label, "motif_significance_profile_unit_norm",
                    bool(abs(norm - 1.0) < 1e-6 or norm == 0.0),
                    f"|SP| = {norm:.6f}",
                ))

    return out


def check_stage5(stage5: dict) -> list[Check]:
    """Stage-5 invariants against a loaded ``stage5_results.pkl``."""
    out: list[Check] = []

    # Weight-sum invariants: each NSI weight configuration sums to 1.
    weights_4ch = (0.25, 0.20, 0.35, 0.20)
    weights_3ch = (0.35, 0.25, 0.40)
    weights_3ch_fb = (0.40, 0.25, 0.35)
    weights_roll = (0.35, 0.35, 0.30)
    for label, ws in [
        ("4-channel", weights_4ch),
        ("3-channel", weights_3ch),
        ("3-channel-fallback", weights_3ch_fb),
        ("rolling", weights_roll),
    ]:
        out.append(Check(
            "stage5", "<weights>", f"weights_sum_to_one_{label}",
            bool(abs(sum(ws) - 1.0) < 1e-12),
            f"sum({ws}) = {sum(ws):.6f}",
        ))

    snap_df = stage5.get("snapshot_nsi")
    if snap_df is None or len(snap_df) == 0:
        out.append(Check("stage5", "<all>", "snapshot_nsi_present", False,
                         "snapshot_nsi missing or empty"))
        return out

    for _, row in snap_df.iterrows():
        label = row["snapshot"]
        nsi = float(row["nsi"])
        out.append(Check(
            "stage5", label, "nsi_in_unit_interval",
            bool(0.0 <= nsi <= 1.0),
            f"NSI = {nsi:.4f}",
        ))

        if "motif_shift_norm" in row.index and not (
                isinstance(row.get("motif_shift_norm"), float)
                and np.isnan(row.get("motif_shift_norm"))):
            nsi_recomp = (
                0.25 * row["network_sparsity_norm"]
                + 0.20 * row["hhi_top10_norm"]
                + 0.35 * row["mean_corr_norm"]
                + 0.20 * row["motif_shift_norm"]
            )
            cfg = "4-channel"
        elif row.get("mean_corr_norm") is not None and not (
                isinstance(row.get("mean_corr_norm"), float)
                and np.isnan(row.get("mean_corr_norm"))):
            nsi_recomp = (
                0.35 * row["network_sparsity_norm"]
                + 0.25 * row["hhi_top10_norm"]
                + 0.40 * row["mean_corr_norm"]
            )
            cfg = "3-channel"
        else:
            nsi_recomp = (
                0.40 * row["network_sparsity_norm"]
                + 0.25 * row["hhi_top10_norm"]
                + 0.35 * row["density_norm"]
            )
            cfg = "3-channel-fallback"
        nsi_recomp = float(nsi_recomp)

        out.append(Check(
            "stage5", label, "nsi_recomputes_from_norm_columns",
            bool(abs(nsi_recomp - nsi) < TOL_NSI_RECOMPUTE),
            f"stored={nsi:.6f}, recomputed={nsi_recomp:.6f} ({cfg})",
        ))

    for col in ["network_sparsity_norm", "hhi_top10_norm",
                "mean_corr_norm", "motif_shift_norm", "density_norm"]:
        if col not in snap_df.columns:
            continue
        ser = pd.to_numeric(snap_df[col], errors="coerce")
        if ser.isna().all():
            continue
        lo, hi = float(ser.min()), float(ser.max())
        out.append(Check(
            "stage5", "<all>", f"{col}_in_unit_interval",
            bool(lo >= -1e-12 and hi <= 1.0 + 1e-12),
            f"range = [{lo:.4f}, {hi:.4f}]",
        ))

    return out


# =====================================================================
# 3. CROSS-STAGE CONSISTENCY
# =====================================================================

def cross_stage_checks(stage1: Optional[dict], stage2: Optional[dict],
                       stage3: Optional[dict], stage4: Optional[dict]
                       ) -> list[Check]:
    """Shape and ticker continuity across stages, per snapshot."""
    out: list[Check] = []
    if stage1 is None or stage2 is None:
        return out

    snaps = stage1.get("snapshot_correlations", {}) or {}

    for label, s2 in stage2.items():
        s1 = snaps.get(label)
        if s1 is None:
            out.append(Check("cross_stage", label, "stage1_snapshot_present",
                             False, "snapshot missing from Stage 1"))
            continue

        R_shape = s1["R_avg"].shape
        A_shape = s2["adjacency"].shape

        # 1. R_avg and adjacency shapes match.
        shape_ok = (R_shape == A_shape and R_shape[0] == R_shape[1])
        out.append(Check(
            "cross_stage", label, "stage1_R_avg_matches_stage2_adjacency_shape",
            shape_ok,
            f"R_avg = {R_shape}, adjacency = {A_shape}",
        ))

        # 2. Tickers match between Stage 1 and Stage 2.
        t1 = list(s1.get("tickers") or [])
        t2 = list(s2.get("tickers") or [])
        tickers_match = (t1 == t2)
        out.append(Check(
            "cross_stage", label, "stage1_tickers_match_stage2",
            tickers_match,
            f"|Stage1|={len(t1)}, |Stage2|={len(t2)}, "
            f"identical: {tickers_match}",
        ))

        # 3. Stage 3 shape continuity.
        if stage3 is not None and label in stage3:
            s3 = stage3[label]
            D_shape = s3["directed_adj"].shape
            d_shape_ok = (D_shape == A_shape)
            out.append(Check(
                "cross_stage", label,
                "stage2_adjacency_matches_stage3_directed_adj_shape",
                d_shape_ok,
                f"adjacency = {A_shape}, directed_adj = {D_shape}",
            ))
            t3 = list(s3.get("tickers") or [])
            out.append(Check(
                "cross_stage", label, "stage2_tickers_match_stage3",
                bool(t2 == t3),
                f"|Stage2|={len(t2)}, |Stage3|={len(t3)}, "
                f"identical: {t2 == t3}",
            ))

        # 4. Stage 4 node count continuity.
        if stage4 is not None and label in stage4:
            s4 = stage4[label]
            n4 = s4.get("n_nodes")
            out.append(Check(
                "cross_stage", label,
                "stage3_p_matches_stage4_n_nodes",
                bool(n4 == A_shape[0]),
                f"stage2 p = {A_shape[0]}, stage4 n_nodes = {n4}",
            ))
            t4 = list(s4.get("tickers") or [])
            out.append(Check(
                "cross_stage", label, "stage2_tickers_match_stage4",
                bool(t2 == t4),
                f"|Stage2|={len(t2)}, |Stage4|={len(t4)}, "
                f"identical: {t2 == t4}",
            ))

    return out


# =====================================================================
# 4. RUNNER
# =====================================================================

def _load(path: Path):
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def run_all(snapshots_dir: Path = SNAPSHOTS_DIR) -> list[Check]:
    """Load each stage's pickle, run property tests, then cache + cross-stage."""
    checks: list[Check] = []

    # 1. Property tests (independent of cache).
    checks += property_tests()

    # 2. Cache-level checks.
    stage1 = _load(snapshots_dir / "stage1_results.pkl")
    stage2 = _load(snapshots_dir / "stage2_results.pkl")
    stage3 = _load(snapshots_dir / "stage3_results.pkl")
    stage4 = _load(snapshots_dir / "stage4_results.pkl")
    stage5 = _load(snapshots_dir / "stage5_results.pkl")

    if stage1 is not None:
        checks += check_stage1(stage1)
    else:
        checks.append(Check("stage1", "<all>", "cache_present", False,
                            "stage1_results.pkl not found"))
    if stage2 is not None:
        checks += check_stage2(stage2)
    else:
        checks.append(Check("stage2", "<all>", "cache_present", False,
                            "stage2_results.pkl not found"))
    if stage1 is not None and stage2 is not None:
        checks += check_stage2_against_stage1(stage1, stage2)
    if stage3 is not None:
        checks += check_stage3(stage3)
    else:
        checks.append(Check("stage3", "<all>", "cache_present", False,
                            "stage3_results.pkl not found"))
    if stage4 is not None:
        checks += check_stage4(stage4)
    else:
        checks.append(Check("stage4", "<all>", "cache_present", False,
                            "stage4_results.pkl not found"))
    if stage5 is not None:
        checks += check_stage5(stage5)
    else:
        checks.append(Check("stage5", "<all>", "cache_present", False,
                            "stage5_results.pkl not found"))

    # 3. Cross-stage.
    checks += cross_stage_checks(stage1, stage2, stage3, stage4)

    return checks


def render_markdown_report(checks: list[Check], cache_label: str = "n=500") -> str:
    """Render a Markdown report grouped by category with a summary."""
    n_total = len(checks)
    n_failed = sum(1 for c in checks if not c.passed)
    n_passed = n_total - n_failed

    lines: list[str] = []
    lines.append("# Pipeline-Invariant Assertion Report")
    lines.append("")
    lines.append(f"**Cache:** {cache_label}")
    lines.append(f"**Total checks:** {n_total}  ·  "
                 f"**PASS:** {n_passed}  ·  **FAIL:** {n_failed}")
    lines.append("")

    by_stage: dict[str, list[Check]] = {}
    for c in checks:
        by_stage.setdefault(c.stage, []).append(c)

    lines.append("## Summary")
    lines.append("")
    lines.append("| Category | Pass | Fail |")
    lines.append("|---|---:|---:|")
    order = ("property", "stage1", "stage2", "stage3", "stage4", "stage5",
             "cross_stage")
    for stg in order:
        items = by_stage.get(stg, [])
        if not items:
            continue
        p = sum(1 for c in items if c.passed)
        f = sum(1 for c in items if not c.passed)
        lines.append(f"| {stg} | {p} | {f} |")
    lines.append("")

    if n_failed > 0:
        lines.append("## Failures")
        lines.append("")
        lines.append("| Category | Snapshot | Check | Detail |")
        lines.append("|---|---|---|---|")
        for c in checks:
            if not c.passed:
                lines.append(f"| {c.stage} | {c.snapshot} | {c.name} | {c.detail} |")
        lines.append("")

    for stg in order:
        items = by_stage.get(stg, [])
        if not items:
            continue
        lines.append(f"## {stg}")
        lines.append("")
        lines.append("| Snapshot | Check | Status | Detail |")
        lines.append("|---|---|---|---|")
        for c in items:
            status = "PASS" if c.passed else "**FAIL**"
            lines.append(f"| {c.snapshot} | {c.name} | {status} | {c.detail} |")
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    checks = run_all()
    report = render_markdown_report(checks)
    out_path = SNAPSHOTS_DIR / "invariants_report.md"
    out_path.write_text(report, encoding="utf-8")

    n_fail = sum(1 for c in checks if not c.passed)
    n_pass = len(checks) - n_fail
    print(f"Wrote {out_path}")
    print(f"Invariants: {n_pass}/{len(checks)} PASS, {n_fail} FAIL")
    if n_fail:
        print("Failing checks:")
        for c in checks:
            if not c.passed:
                print(f"  [{c.stage}/{c.snapshot}] {c.name}: {c.detail}")
