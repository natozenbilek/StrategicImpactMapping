"""
Stage 4 - network-level metrics (manuscript Sec. 3.4).

For each Stage-3 directed snapshot, four pre-registered metrics are
computed:

================  =============================================
Metric            Hypothesis
================  =============================================
Q1: ``|Z_C|``     Empirical clustering deviation from an
                  Erdos-Renyi G(n, m) null - tests whether the
                  filtered partial-correlation graph is
                  distinguishable from random at all.
Q2: Gini + HHI    PageRank concentration on the directed graph -
                  tests whether a small number of nodes carry a
                  disproportionate share of systemic spillover.
Q3: ``Q``, purity Louvain modularity on the undirected
                  projection plus GICS sector purity - tests
                  whether community structure dissolves under
                  stress.
Q4: motif Z       Dyad-preserving (Maslov-Sneppen) rewire null
                  for the Davis-Leinhardt MAN triadic motifs
                  (FFL/030T, MR/111D, SIM/021D) - tests whether
                  reciprocity collapses under stress.
================  =============================================

Auxiliary modules:

* :mod:`src.stage4_network.crisis_signals` - density-invariant
  Watts-Strogatz sigma, mutual-dyad fraction, motif SP distance
  (used by the robustness section of the manuscript).
* :mod:`src.stage4_network.density_matched` - density-matched
  community analysis at a common top-k edge budget.

References
----------
Erdos, Renyi (Publ. Math. Debrecen, 1959);
Page, Brin, Motwani, Winograd (Stanford InfoLab, 1999);
Blondel et al. (J. Stat. Mech., 2008);
Davis, Leinhardt (Sociological Theories in Progress, 1972);
Milo et al. (Science, 2002);
Maslov, Sneppen (Science, 2002);
Watts, Strogatz (Nature, 1998);
Humphries, Gurney (PLoS ONE, 2008).
"""
