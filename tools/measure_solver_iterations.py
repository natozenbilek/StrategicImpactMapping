"""Measure per-snapshot GLASSO and PageRank iteration counts.

Reads the cached Stage 1/2/4 results from ``results/snapshots/`` and
reports the actual coordinate-descent and power-iteration counts that
the M-D complexity table currently reports as caps (``I_CD <= 500``,
``I_PR <= 100``). No cache is modified; the script only re-solves on
already-cached inputs.

Usage::

    python -m tools.measure_solver_iterations

Output goes to ``results/snapshots/solver_iterations.json`` and a
human-readable summary is printed to stdout. The JSON sidecar lets the
appendix M-D entries be regenerated from a single source of truth.

Why this exists: ``sklearn.covariance.graphical_lasso`` does expose
``return_n_iter=True`` but the production pipeline calls it without
that flag, so the iteration count is dropped on the floor. NetworkX
``pagerank`` does not return its iteration count under any flag; this
script reimplements the inner power-iteration loop in ten lines so the
count can be observed without depending on a NetworkX patch.

The numbers this script emits should be cited in
``paper/appendix.tex`` (M-D table) in place of the conservative caps;
run after every Stage 2 / Stage 4 invariant change.
"""
import json
import pickle
from pathlib import Path

import numpy as np
import networkx as nx
from sklearn.covariance import graphical_lasso

from src.config import SNAPSHOTS_DIR, PAGERANK_DAMPING
# Import the production safety-net so the matrix we feed graphical_lasso
# is bit-for-bit what fit_glasso_ebic fed during the original sweep —
# any private reimplementation here would risk numerically diverging at
# the boundary cases (n/p≈1 panels, low-delta shrinkage tiers).
from src.stage2_precision.glasso_filter import _apply_spectral_safety_net

# Stage-2 solver settings that the production pipeline pins; mirror
# the call in src.stage2_precision.glasso_filter so the iteration
# counts we measure are identical to what the live pipeline would see.
GLASSO_MAX_ITER = 500
GLASSO_TOL = 1e-4  # sklearn default

# Stage-4 PageRank settings: NetworkX defaults at the time the
# pipeline was written (NetworkX 3.2.1).
PAGERANK_MAX_ITER = 100
PAGERANK_TOL = 1e-6

OUTPUT_PATH = SNAPSHOTS_DIR / "solver_iterations.json"


def _prepare_corr_matrix(R, delta):
    """Re-apply the production pipeline's shrinkage + safety net.

    ``stage2_results.pkl`` caches the raw ``R_avg`` (the Stage-1
    window-averaged correlation, pre-shrinkage); ``applied_delta`` is
    also cached so we can reproduce the exact matrix that
    ``fit_glasso_ebic`` would have fed into ``graphical_lasso``.
    Ordering is identity-target Ledoit-Wolf shrinkage first, then the
    same uniform-shift safety net the production sweep uses. The
    safety net is imported from ``src.stage2_precision.glasso_filter``
    rather than re-implemented so the matrix is bit-for-bit identical
    on the boundary cases (low-delta tiers, n/p≈1).
    """
    p = R.shape[0]
    R_shrunk = (1.0 - delta) * R + delta * np.eye(p)
    return _apply_spectral_safety_net(R_shrunk)


def measure_glasso_iterations(stage2_cache):
    """Re-solve GLASSO at the cached lambda_opt of each snapshot.

    Returns ``{snapshot: {n_iter, lambda_opt, applied_delta, p,
    converged}}``. The applied_delta is recorded alongside the count
    so an auditor can verify the shrinkage path matches production.
    """
    out = {}
    for snap, d in stage2_cache.items():
        R = np.asarray(d["R_avg"], dtype=float)
        delta = float(d["applied_delta"])
        assert R.ndim == 2 and R.shape[0] == R.shape[1], \
            f"R_avg for {snap!r} not square: {R.shape}"
        corr = _prepare_corr_matrix(R, delta)
        lam = float(d["lambda_opt"])
        p = R.shape[0]

        # return_n_iter=True is exposed but the production pipeline
        # discards it; we keep the same numeric setup as
        # src.stage2_precision.glasso_filter to ensure the iteration
        # count is what the cached precision matrix actually cost.
        try:
            cov, prec, n_iter = graphical_lasso(
                corr, alpha=lam, mode="cd",
                max_iter=GLASSO_MAX_ITER, tol=GLASSO_TOL,
                return_n_iter=True,
            )
            converged = n_iter < GLASSO_MAX_ITER
        except Exception as exc:
            n_iter, converged = -1, False
            print(f"  [warn] {snap}: {type(exc).__name__}: {exc}")

        out[snap] = {
            "n_iter": int(n_iter),
            "lambda_opt": lam,
            "applied_delta": delta,
            "p": int(p),
            "converged": bool(converged),
        }
        print(f"  GLASSO {snap:30s}  p={p:3d}  delta={delta:.2f}  "
              f"lambda={lam:.4f}  n_iter={n_iter:3d}  "
              f"{'OK' if converged else 'CAP HIT'}")
    return out


def _pagerank_with_iter_count(G, alpha=PAGERANK_DAMPING,
                              tol=PAGERANK_TOL, max_iter=PAGERANK_MAX_ITER):
    """Power-iteration PageRank that also returns iteration count.

    Mirrors NetworkX 3.2.1's ``nx.pagerank`` numeric path: column-
    stochastic transition with uniform dangling redistribution, L1
    convergence with threshold ``N * tol``. We keep the same
    convergence criterion so the count is what production would see.
    """
    N = G.number_of_nodes()
    if N == 0:
        return 0
    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    A = np.zeros((N, N), dtype=float)
    for u, v in G.edges():
        A[idx[u], idx[v]] = 1.0
    out_deg = A.sum(axis=1)
    dangling = (out_deg == 0)
    # Column-stochastic of the transposed adjacency: A.T / col_sums.
    M = np.zeros_like(A)
    nonzero = ~dangling
    M[nonzero] = A[nonzero] / out_deg[nonzero, None]
    M = M.T  # now column-stochastic on M @ x
    teleport = np.ones(N, dtype=float) / N
    x = teleport.copy()
    for i in range(1, max_iter + 1):
        # Dangling redistribution: dangling mass goes uniform.
        dangling_mass = float(x[dangling].sum()) if dangling.any() else 0.0
        x_new = alpha * (M @ x + dangling_mass * teleport) \
                + (1.0 - alpha) * teleport
        err = float(np.abs(x_new - x).sum())
        x = x_new
        if err < N * tol:
            return i
    return max_iter  # did not converge within cap


def measure_pagerank_iterations(stage4_cache):
    """Power-iterate every cached Stage-4 graph; return ``{snap: n_iter}``."""
    out = {}
    for snap, d in stage4_cache.items():
        G = d["graph"]
        n_iter = _pagerank_with_iter_count(G)
        converged = n_iter < PAGERANK_MAX_ITER
        out[snap] = {
            "n_iter": int(n_iter),
            "n_nodes": int(G.number_of_nodes()),
            "n_edges": int(G.number_of_edges()),
            "converged": bool(converged),
        }
        print(f"  PageRank {snap:30s}  "
              f"n={G.number_of_nodes():3d}  m={G.number_of_edges():4d}  "
              f"n_iter={n_iter:3d}  {'OK' if converged else 'CAP HIT'}")
    return out


def main():
    s2_path = SNAPSHOTS_DIR / "stage2_results.pkl"
    s4_path = SNAPSHOTS_DIR / "stage4_results.pkl"
    assert s2_path.exists(), f"Stage 2 cache missing: {s2_path}"
    assert s4_path.exists(), f"Stage 4 cache missing: {s4_path}"

    with open(s2_path, "rb") as f:
        s2 = pickle.load(f)
    with open(s4_path, "rb") as f:
        s4 = pickle.load(f)

    print("=" * 64)
    print("GLASSO coordinate-descent iterations (one re-solve per snap)")
    print("=" * 64)
    glasso = measure_glasso_iterations(s2)

    print()
    print("=" * 64)
    print("PageRank power-iteration counts (one full pass per snap)")
    print("=" * 64)
    pagerank = measure_pagerank_iterations(s4)

    payload = {
        "config": {
            "glasso_max_iter": GLASSO_MAX_ITER,
            "glasso_tol": GLASSO_TOL,
            "pagerank_max_iter": PAGERANK_MAX_ITER,
            "pagerank_tol": PAGERANK_TOL,
            "pagerank_alpha": PAGERANK_DAMPING,
        },
        "glasso": glasso,
        "pagerank": pagerank,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {OUTPUT_PATH}")

    # Print summary stats so the appendix M-D entries can be updated
    # directly from stdout without opening the JSON.
    glasso_iters = [v["n_iter"] for v in glasso.values() if v["n_iter"] > 0]
    pr_iters = [v["n_iter"] for v in pagerank.values() if v["n_iter"] > 0]
    print()
    print("=" * 64)
    print("SUMMARY for appendix M-D")
    print("=" * 64)
    if glasso_iters:
        print(f"  I_CD measured: min={min(glasso_iters)} "
              f"max={max(glasso_iters)} median={int(np.median(glasso_iters))} "
              f"(cap was {GLASSO_MAX_ITER})")
    if pr_iters:
        print(f"  I_PR measured: min={min(pr_iters)} "
              f"max={max(pr_iters)} median={int(np.median(pr_iters))} "
              f"(cap was {PAGERANK_MAX_ITER})")


if __name__ == "__main__":
    main()
