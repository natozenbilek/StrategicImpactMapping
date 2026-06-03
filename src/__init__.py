"""
Strategic Impact Mapping in Financial Markets - top-level package.

This package implements the five-stage pipeline accompanying the
manuscript "Strategic Impact Mapping in Financial Markets: A Dynamic
Directed Network Framework for S&P 500 Stress Analysis" (Toezenbilek,
2026). Each stage lives in a dedicated sub-package whose name mirrors
its role in the manuscript:

==================  ==========================================
Sub-package         Role in the manuscript
==================  ==========================================
``stage1_data``     Sec. 3.1 - Yahoo Finance ingest, GARCH(1,1)
                    plus A-DCC parameter estimation and the
                    per-snapshot correlation tensors.
``stage2_precision``Sec. 3.2 - Graphical-LASSO precision-matrix
                    filtering with EBIC selection, n/p-tiered
                    identity-target shrinkage, and the
                    constrained-BIC fallback rule (BIC-optimal
                    lambda subject to a k >= p edge-count floor).
``stage3_direction``Sec. 3.3 - Lead/follower edge orientation via
                    the lagged partial-correlation + Granger
                    two-test cascade.
``stage4_network``  Sec. 3.4 - Erdos-Renyi clustering deviation,
                    PageRank concentration, Louvain modularity
                    with GICS sector purity, and the dyad-
                    preserving MAN triadic motif census.
``stage5_nsi``      Sec. 3.5 - Composite Network Stress Index
                    (NSI) aggregation, rolling-window backtest,
                    and VIX concordance diagnostics.
``robustness``      Sec. 6 - Density-controlled and density-
                    invariant robustness corrections plus exact-
                    permutation and bootstrap inference helpers.
``utils``           Numerical core helpers shared across stages.
==================  ==========================================

The orchestration script ``run_pipeline.py`` at the repository root
chains the five stages end-to-end and caches intermediate artefacts
under ``results/snapshots/`` so downstream analysis (figures, tables,
inference) can re-load without re-running the multi-hour estimation
steps. Snapshot definitions, hyper-parameters, and random seeds live
in :mod:`src.config`.
"""
