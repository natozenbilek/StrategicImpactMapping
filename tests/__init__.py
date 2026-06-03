"""
Unit-test package for the Strategic Impact Mapping pipeline.

Currently covers:

* :mod:`tests.test_dcc_core` - A-DCC recursion shape, symmetry,
  positive-semidefiniteness, and quasi-log-likelihood numerics.
* :mod:`tests.test_stage1_data` - Stage 0 data construction:
  ticker normalisation, cleanup cutoffs, contamination detector,
  volume scale guard, cache schema.
* :mod:`tests.test_stage2_precision` - Graphical-LASSO penalty
  grid, EBIC tier selection, identity-target shrinkage tiers,
  spectral safety net, constrained-BIC fallback, and end-to-end
  ``run_stage2`` invariants.

Run via ``pytest tests/`` from the repository root.
"""
