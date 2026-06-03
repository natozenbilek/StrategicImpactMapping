"""Stage 5 - Network Stress Index aggregation (manuscript Sec. 3.5).

NSI = w_s tilde_s + w_h tilde_h + w_rho tilde_rho + w_m tilde_mu, with
weights (0.25, 0.20, 0.35, 0.20) on min/max-normalised channels:
sparsity, full-network PageRank HHI, A-DCC mean correlation, FFL motif
shift.

Modules:
* stress_index - snapshot and rolling NSI + VIX concordance backtest.
* volume_weighted_nsi - log-ADV reweighted variant (appendix F1 prototype).
"""
