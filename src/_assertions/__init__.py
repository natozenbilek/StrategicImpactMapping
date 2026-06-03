"""
Numerical-invariant assertion suite for the five-stage pipeline.

Each stage of the pipeline (Stages 1-5) writes a pickle cache to
``results/snapshots/``. The :mod:`src._assertions.invariants` module
loads each cache and verifies a fixed set of mathematical invariants
that should hold by construction (stationarity of A-DCC parameters,
positive-definiteness of GLASSO precision matrices, edge-count
conservation in Stage 3, PageRank stationary distribution summing to
unity, Network Stress Index in [0, 1], etc.). The suite is intended
as a sanity gate before any panel-size sweep or appendix recomputation.

Usage
-----
    python -m src._assertions.invariants

writes ``results/snapshots/invariants_report.md`` summarising the
PASS/FAIL status of each invariant per snapshot and prints a one-line
overall verdict to stdout.
"""
