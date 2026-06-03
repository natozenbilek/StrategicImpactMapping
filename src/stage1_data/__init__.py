"""
Stage 1 - data ingest and A-DCC GARCH estimation (manuscript Sec. 3.1).

Two responsibilities:

* :mod:`src.stage1_data.download` pulls Yahoo Finance daily adjusted
  closes for the current S&P 500 plus a hand-picked set of crisis-era
  failures and assembles the unbalanced historical-constituent panel.
* :mod:`src.stage1_data.dcc_garch` fits univariate GARCH(1,1) with
  Student-t innovations per asset, then estimates a single A-DCC
  parameter triple ``(a, b, g)`` on a 100-asset balanced subset via
  multi-start L-BFGS-B and applies the optimum within each window to
  produce the window-averaged correlation matrix that feeds Stage 2.

References
----------
Bollerslev (J. Econometrics, 1986); Engle (J. Bus. Econ. Stat., 2002);
Cappiello, Engle, Sheppard (J. Financial Econometrics, 2006).
"""
