"""A-DCC numerical primitives (Cappiello, Engle, Sheppard 2006).

    Q_t = (1 - a - b) Qbar - g Nbar
        + a z_{t-1} z_{t-1}'
        + b Q_{t-1}
        + g n_{t-1} n_{t-1}',
    R_t = diag(Q_t)^{-1/2} Q_t diag(Q_t)^{-1/2}.

n_t = min(z_t, 0). Stationarity: a, b, g >= 0 and a + b + g < 1.

Run under ``python -O`` to strip the inline ``assert`` invariants from
the hot loop once the pipeline is being benchmarked.
"""
import numpy as np
from scipy.linalg import solve_triangular


def adcc_intercept(a, b, g, Q_bar, N_bar):
    """Closed-form intercept Omega = (1 - a - b) Qbar - g Nbar."""
    assert a >= 0 and b >= 0 and g >= 0, (a, b, g)
    assert a + b + g < 1.0, f"non-stationary: a+b+g = {a + b + g}"
    assert Q_bar.shape == N_bar.shape, (Q_bar.shape, N_bar.shape)
    assert Q_bar.ndim == 2 and Q_bar.shape[0] == Q_bar.shape[1]
    return (1 - a - b) * Q_bar - g * N_bar


def adcc_qt_update(Q_prev, z_prev, const, a, b, g):
    """One A-DCC update step. ``const`` is the precomputed intercept."""
    assert Q_prev.shape == const.shape
    assert Q_prev.shape[0] == z_prev.shape[0]
    if z_prev.ndim == 1:
        z_prev = z_prev.reshape(-1, 1)
    n_prev = np.minimum(z_prev, 0.0)
    return (const
            + a * (z_prev @ z_prev.T)
            + b * Q_prev
            + g * (n_prev @ n_prev.T))


def normalize_qt_to_rt(Q_t):
    """R_t = D Q_t D with D = diag(Q_t)^{-1/2}.

    The 1e-8 floor on the inverse-square-root guards against degenerate
    diagonals during optimiser line-search; the diagonal is reset to
    unity after normalisation to absorb round-off.
    """
    assert Q_t.ndim == 2 and Q_t.shape[0] == Q_t.shape[1]
    d = np.sqrt(np.diag(Q_t))
    d_inv = 1.0 / np.maximum(d, 1e-8)
    R_t = Q_t * np.outer(d_inv, d_inv)
    np.fill_diagonal(R_t, 1.0)
    return R_t


def adcc_loglikelihood_contribution(R_t, z_t):
    """LL_t = -0.5 (log|R_t| + z' R_t^{-1} z - z' z).

    Cholesky path: log|R_t| = 2 sum log diag(L); z' R^{-1} z = ||L^{-1} z||^2.
    On Cholesky failure (non-PD line-search iterate), retries via slogdet
    and rejects non-positive determinants with None.
    """
    assert R_t.shape[0] == R_t.shape[1] == z_t.shape[0]
    z = z_t.ravel()
    try:
        L = np.linalg.cholesky(R_t)
        logdet = 2.0 * np.sum(np.log(np.diag(L)))
        y = solve_triangular(L, z, lower=True, check_finite=False)
        quad = float(y @ y)
        return float(-0.5 * (logdet + quad - float(z @ z)))
    except np.linalg.LinAlgError:
        try:
            sign, logdet = np.linalg.slogdet(R_t)
            if sign <= 0:
                return None
            quad = float(z @ np.linalg.solve(R_t, z))
            return float(-0.5 * (logdet + quad - float(z @ z)))
        except np.linalg.LinAlgError:
            return None
