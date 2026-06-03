"""
Numerical core helpers shared across pipeline stages.

This sub-package collects routines used by more than one stage so they
do not have to be duplicated. The current contents are:

* :mod:`src.utils.dcc_core` - low-level A-DCC recursion (the
  ``(1 - a - b) * Qbar - g * Nbar + a z z' + b Q_{t-1} + g n n'``
  update), quasi-log-likelihood, and the stationarity guard
  ``a + b + g < 1`` used by both the parameter estimation in
  :mod:`src.stage1_data.dcc_garch` and the per-window re-application of
  the pooled parameters.
"""
