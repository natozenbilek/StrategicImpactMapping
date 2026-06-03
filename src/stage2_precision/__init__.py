"""
Stage 2 - Graphical-LASSO precision-matrix filtering (manuscript Sec. 3.2).

The Stage-1 window-averaged correlation matrix is sparsified via
Graphical LASSO with an Extended Bayesian Information Criterion (EBIC)
penalty selected on a 25-point log-spaced lambda grid. Two manuscript-
specific numerical hardenings live here:

* an ``n/p``-tiered Ledoit-Wolf-style identity-target shrinkage that
  guarantees a positive-definite input to the sklearn coordinate
  descent solver on the short, high-volatility windows where the raw
  correlation matrix is rank-deficient (manuscript Sec. 3.2,
  Table tab:shrinkage-tiers);
* a constrained-BIC fallback rule that selects the BIC-optimal
  ``lambda`` subject to ``k(lambda) >= max(p, 10)`` when the
  unconstrained-BIC argmin yields a sub-degree graph (manuscript
  Sec. 3.2, eq:fallback) - on this panel the fallback fires on five
  of ten snapshots (both crisis peaks plus three other
  ``n/p < 0.30`` windows).

References
----------
Friedman, Hastie, Tibshirani (Biostatistics, 2008);
Foygel, Drton (NeurIPS, 2010);
Ledoit, Wolf (J. Empirical Finance, 2003).
"""
