"""
Stage 3 - lead/follower edge orientation (manuscript Sec. 3.3).

Each undirected edge surviving Stage 2 is tested in both directions with
two complementary statistics:

* a lagged partial correlation between the source's previous-day return
  and the target's same-day return, with the top-five most-correlated
  assets (excluding the pair under test) held fixed as control variables,
* a Granger causality test at ``max_lag = 1`` on the same pair.

The two-test cascade assigns a per-direction confidence ``c_d`` in
``[0, 1]``: significant partial correlation at ``alpha = 0.05`` raises
``c_d`` to at least ``|rho_d|`` and significant Granger raises it to at
least ``0.5``. An edge with ``c_d = 0`` in both directions is dropped;
a single-positive direction wins; a ``1.5x`` dominance over the opposite
direction wins; ties with bidirectional evidence become mutual dyads.
The cascade compresses the mutual-edge fraction from the ``60-77%``
range observed under a single test alone to the ``5.3-19.2%`` reported
in Tab. tab:stage3 of the manuscript.

Implementation lives in :mod:`src.stage3_direction.lead_lag`.

References
----------
Granger (Econometrica, 1969);
Diks, Panchenko (J. Econ. Dyn. Control, 2006);
Kim, Kim, Park (Sustainability, 2020).
"""
