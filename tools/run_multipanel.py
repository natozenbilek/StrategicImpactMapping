"""
Multi-panel sweep orchestrator.

Re-runs the five-stage pipeline at panel sizes
N in {50, 100, 150, 200, 250, 300, 350, 400, 450, 500} under seven
selection criteria:

* ``coverage`` --- top-N by sample-wide non-NaN observation count.
* ``adv``      --- top-N by mean dollar-volume over 2004-2025.
* ``bottom_adv`` / ``bottom_coverage`` --- bottom-N tail variants.
* ``decile1`` ... ``decile5`` --- ADV quintile (1=most liquid).

Per-panel artifacts go to ``results/multipanel/n{N}_{kind}/``:
``stage1_results.pkl`` ... ``stage5_results.pkl``,
``stage2_extended.pkl`` (per-snapshot unconstrained-BIC argmin alongside
the post-fallback selection so the fallback labelling is end-to-end
re-derivable), and ``invariants_report.md``.

Across all panels, ``results/multipanel/summary.json`` collects the
headline metrics per snapshot per panel for cross-panel comparison.

Isolation
---------
Each (panel, kind) runs in its own subprocess (multiprocessing.Pool
with ``maxtasksperchild=1``). The subprocess assigns
``src.config.SNAPSHOTS_DIR = panel_dir`` *before* importing any stage
module, so the ``from src.config import SNAPSHOTS_DIR`` bindings inside
the stage modules all capture the panel-specific cache directory at
import time. Sweep-2026-05-24 audit found the prior in-process
monkey-patch approach silently fell through for N>=150 panels (every
panel re-loaded the same 105-ticker subset cache), so the subprocess
boundary is the correctness guarantee.

Usage
-----
    python -m tools.run_multipanel --panel 100 --kind coverage
    python -m tools.run_multipanel --all --workers 8 --force
"""
from __future__ import annotations

import argparse
import datetime
import json
import multiprocessing as mp
import os
import pickle
import platform
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# Resolved once at import; the subprocess re-derives its own paths from
# the same constants so worker invocations are reproducible.
_THIS_FILE = Path(__file__).resolve()
ROOT_DIR = _THIS_FILE.parent.parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
MULTIPANEL_DIR = RESULTS_DIR / "multipanel"

DEFAULT_PANEL_SIZES = (50, 100, 150, 200, 250, 300, 350, 400, 450, 500)
SELECTION_KINDS = ("coverage", "adv", "bottom_adv", "bottom_coverage",
                   "decile1", "decile2", "decile3", "decile4", "decile5")


# ---------------------------------------------------------------------
# Ticker selection (parent-process helpers; subprocess re-imports them)
# ---------------------------------------------------------------------

def select_top_tickers_by_coverage(returns: pd.DataFrame, n: int) -> list[str]:
    """Top-n tickers by sample-wide non-NaN count."""
    coverage = returns.count().sort_values(ascending=False)
    keep = set(coverage.head(n).index)
    return [c for c in returns.columns if c in keep]


def select_top_tickers_by_adv(returns: pd.DataFrame, n: int,
                              adv_path: Path = DATA_DIR / "sp500_adv.parquet"
                              ) -> list[str]:
    """Top-n tickers by mean dollar-volume."""
    if not adv_path.exists():
        raise FileNotFoundError(
            f"ADV cache missing at {adv_path}. Run `python -m "
            f"tools.download_volume` first to fetch volume data.")
    adv = pd.read_parquet(adv_path).iloc[:, 0]
    candidates = list(set(returns.columns).intersection(adv.index))
    ranked = adv.loc[candidates].sort_values(ascending=False)
    keep = set(ranked.head(n).index)
    return [c for c in returns.columns if c in keep]


def select_bottom_tickers_by_adv(returns: pd.DataFrame, n: int,
                                 adv_path: Path = DATA_DIR / "sp500_adv.parquet"
                                 ) -> list[str]:
    """Bottom-n tickers by ADV (the speculative tail)."""
    if not adv_path.exists():
        raise FileNotFoundError(f"ADV cache missing at {adv_path}.")
    adv = pd.read_parquet(adv_path).iloc[:, 0]
    candidates = list(set(returns.columns).intersection(adv.index))
    ranked = adv.loc[candidates].sort_values(ascending=True)
    keep = set(ranked.head(n).index)
    return [c for c in returns.columns if c in keep]


def select_bottom_tickers_by_coverage(returns: pd.DataFrame, n: int) -> list[str]:
    """Bottom-n tickers by coverage."""
    coverage = returns.count().sort_values(ascending=True)
    keep = set(coverage.head(n).index)
    return [c for c in returns.columns if c in keep]


def select_decile_tickers_by_adv(returns: pd.DataFrame, n: int, decile: int,
                                 adv_path: Path = DATA_DIR / "sp500_adv.parquet"
                                 ) -> list[str]:
    """ADV quintile (1=most liquid, 5=least). n caps the quintile slice."""
    if not adv_path.exists():
        raise FileNotFoundError(f"ADV cache missing at {adv_path}.")
    assert decile in (1, 2, 3, 4, 5), f"decile must be 1..5; got {decile}"
    adv = pd.read_parquet(adv_path).iloc[:, 0]
    candidates = list(set(returns.columns).intersection(adv.index))
    ranked = adv.loc[candidates].sort_values(ascending=False)
    q_size = max(1, len(ranked) // 5)
    start = (decile - 1) * q_size
    end = start + q_size if decile < 5 else len(ranked)
    quintile_tickers = ranked.iloc[start:end].index
    keep = set(quintile_tickers[:n])
    return [c for c in returns.columns if c in keep]


def select_tickers(returns: pd.DataFrame, n: int, kind: str) -> list[str]:
    if kind == "coverage":
        return select_top_tickers_by_coverage(returns, n)
    if kind == "adv":
        return select_top_tickers_by_adv(returns, n)
    if kind == "bottom_adv":
        return select_bottom_tickers_by_adv(returns, n)
    if kind == "bottom_coverage":
        return select_bottom_tickers_by_coverage(returns, n)
    if kind.startswith("decile"):
        return select_decile_tickers_by_adv(returns, n, int(kind[-1]))
    raise ValueError(f"unknown kind {kind!r}; expected {SELECTION_KINDS}")


# ---------------------------------------------------------------------
# Per-panel subprocess worker
# ---------------------------------------------------------------------

def _worker(args: tuple[int, str, bool]) -> dict:
    """Subprocess entry: patches SNAPSHOTS_DIR, then imports stage modules.

    Runs in a fresh subprocess (Pool.maxtasksperchild=1) so the stage
    modules' ``from src.config import SNAPSHOTS_DIR`` bindings capture
    the panel-specific directory at import time. Any per-panel error
    is captured and returned as a record so one bad panel does not
    abort the sweep.
    """
    n, kind, force = args
    try:
        return _worker_impl(n, kind, force)
    except Exception as exc:
        import traceback
        return {
            "n": n,
            "kind": kind,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "snapshots": {},
            "tickers": [],
            "total_minutes": float("nan"),
        }


def _worker_impl(n: int, kind: str, force: bool) -> dict:
    panel_dir = MULTIPANEL_DIR / f"n{n}_{kind}"
    panel_dir.mkdir(parents=True, exist_ok=True)

    # CRITICAL: patch BEFORE any stage-module import. Stage modules do
    # ``from src.config import SNAPSHOTS_DIR`` at their own top level;
    # they capture whatever value sits on ``src.config`` *at the moment
    # of their first import*. So we mutate ``src.config`` first.
    import src.config as cfg
    cfg.SNAPSHOTS_DIR = panel_dir

    # Sanity check after import: every stage module must see the patched
    # value. If this assert fires, the subprocess boundary leaked.
    from src.stage1_data.dcc_garch import run_stage1, SNAPSHOTS_DIR as s1_dir
    from src.stage2_precision.glasso_filter import run_stage2, SNAPSHOTS_DIR as s2_dir
    from src.stage3_direction.lead_lag import run_stage3, SNAPSHOTS_DIR as s3_dir
    from src.stage4_network.analysis import run_stage4, SNAPSHOTS_DIR as s4_dir
    from src.stage5_nsi.stress_index import run_stage5, SNAPSHOTS_DIR as s5_dir
    for tag, observed in (("s1", s1_dir), ("s2", s2_dir), ("s3", s3_dir),
                          ("s4", s4_dir), ("s5", s5_dir)):
        assert observed == panel_dir, (
            f"{tag} captured SNAPSHOTS_DIR={observed} expected {panel_dir} "
            f"-- subprocess SNAPSHOTS_DIR patch leaked through to stage "
            f"modules' import-time bindings")

    returns = pd.read_parquet(DATA_DIR / "sp500_returns.parquet")
    info_path = DATA_DIR / "sp500_info.parquet"
    sp500_info = pd.read_parquet(info_path) if info_path.exists() else None

    chosen_tickers = select_tickers(returns, n, kind)
    assert len(chosen_tickers) <= n, (
        f"select_tickers returned {len(chosen_tickers)} tickers for n={n}")
    # For panel sizes within the available universe, expect exactly n.
    # decile5 can return fewer than n if the quintile is smaller.
    if not kind.startswith("decile"):
        assert len(chosen_tickers) == n, (
            f"select_tickers({n},{kind}) returned {len(chosen_tickers)} "
            f"tickers, expected {n}")
    returns_n = returns[chosen_tickers]
    assert returns_n.shape[1] == len(chosen_tickers)

    t_start = time.time()
    t1 = time.time()
    stage1 = run_stage1(returns_n, n_assets=None, force=force)
    print(f"  [Stage 1] {(time.time() - t1) / 60:.1f} min", flush=True)
    assert stage1["z_df"].shape[1] == len(chosen_tickers), (
        f"Stage1 z_df has {stage1['z_df'].shape[1]} cols, "
        f"expected {len(chosen_tickers)}")

    t2 = time.time()
    stage2 = run_stage2(stage1["snapshot_correlations"], force=force)
    print(f"  [Stage 2] {(time.time() - t2) / 60:.1f} min", flush=True)

    ext = {label: {k: v[k] for k in ("lambda_unconstr",
                                     "n_edges_unconstr",
                                     "ebic_unconstr",
                                     "lambda_opt",
                                     "n_edges",
                                     "density",
                                     "ebic",
                                     "fallback_fired")
                   if k in v}
           for label, v in stage2.items()}
    with open(panel_dir / "stage2_extended.pkl", "wb") as f:
        pickle.dump(ext, f)

    t3 = time.time()
    stage3 = run_stage3(stage2, returns_n, force=force)
    print(f"  [Stage 3] {(time.time() - t3) / 60:.1f} min", flush=True)

    t4 = time.time()
    stage4 = run_stage4(stage3, sp500_info=sp500_info, force=force)
    print(f"  [Stage 4] {(time.time() - t4) / 60:.1f} min", flush=True)

    t5 = time.time()
    stage5 = run_stage5(stage4_results=stage4, returns=returns_n,
                        snapshot_correlations=stage1["snapshot_correlations"],
                        force=force)
    print(f"  [Stage 5] {(time.time() - t5) / 60:.1f} min", flush=True)

    # Confirm every per-panel cache landed inside panel_dir.
    for fname in ("stage1_results.pkl", "stage2_results.pkl",
                  "stage3_results.pkl", "stage4_results.pkl",
                  "stage5_results.pkl"):
        assert (panel_dir / fname).exists(), (
            f"Expected {panel_dir / fname} after stage run "
            f"(panel cache may have leaked to global SNAPSHOTS_DIR)")

    total_min = (time.time() - t_start) / 60
    print(f"  TOTAL {n}/{kind}: {total_min:.1f} min", flush=True)

    snapshots_summary = {}
    if "snapshot_nsi" in stage5 and isinstance(stage5["snapshot_nsi"], pd.DataFrame):
        for _, row in stage5["snapshot_nsi"].iterrows():
            label = row["snapshot"]
            s2 = stage2.get(label, {})
            s3 = stage3.get(label, {})
            s4 = stage4.get(label, {})
            motifs = (s4.get("motifs") or {}).get("z_scores", {})
            n_dir = s3.get("n_directed", 0)
            n_mut = s3.get("n_bidirectional", 0)
            mutual_frac = (n_mut / (n_dir + n_mut)) if (n_dir + n_mut) > 0 else 0.0
            nsi_val = float(row["nsi"])
            assert -1e-9 <= nsi_val <= 1 + 1e-9, (
                f"NSI={nsi_val} out of [0,1] for {label} @ N={n} kind={kind}")
            snapshots_summary[label] = {
                "regime": row["regime"],
                "nsi": nsi_val,
                "mean_corr": float(row.get("mean_corr", np.nan)),
                "hhi": float((s4.get("pagerank") or {}).get("hhi", np.nan)),
                "gini": float((s4.get("pagerank") or {}).get("gini", np.nan)),
                "modularity": float((s4.get("community") or {}).get("modularity", np.nan)),
                "purity": float((s4.get("community") or {}).get("purity", np.nan)),
                "n_edges_undirected": int(s2.get("n_edges", 0)),
                "density": float(s2.get("density", np.nan)),
                "lambda_opt": float(s2.get("lambda_opt", np.nan)),
                "n_directed": int(n_dir),
                "n_mutual": int(n_mut),
                "mutual_dyad_fraction": float(mutual_frac),
                "z_ffl": float(motifs.get("feed_forward_loop", np.nan)),
                "z_mr": float(motifs.get("mutual_regulation", np.nan)),
                "z_sim": float(motifs.get("single_input_module", np.nan)),
            }

    # Run the invariant suite against the per-panel cache.
    try:
        from src._assertions.invariants import run_all, render_markdown_report
        checks = run_all(snapshots_dir=panel_dir)
        report = render_markdown_report(
            checks, cache_label=f"n={n}, kind={kind}")
        (panel_dir / "invariants_report.md").write_text(
            report, encoding="utf-8")
        n_fail = sum(1 for c in checks if not c.passed)
        print(f"  Invariants: {len(checks) - n_fail}/{len(checks)} PASS, "
              f"{n_fail} FAIL", flush=True)
    except Exception as exc:
        print(f"  WARNING: invariant sweep failed: {exc}", flush=True)

    return {
        "n": n,
        "kind": kind,
        "total_minutes": total_min,
        "tickers": chosen_tickers,
        "snapshots": snapshots_summary,
    }


# ---------------------------------------------------------------------
# Sweep driver
# ---------------------------------------------------------------------

def write_summary(records: list[dict], workers: int = 1) -> None:
    """Persist ``records`` and a provenance sidecar.

    The sidecar at ``summary_metadata.json`` records when the sweep
    last touched ``summary.json``, how many subprocesses were
    contending for CPU/memory at the time, the four BLAS thread-count
    environment variables, and the macOS/CPU identity. Per-subprocess
    wall-time (the ``total_minutes`` field on each record) is only
    interpretable in the context of that contention level; without
    the sidecar the appendix M-C reader would have no way to tell
    whether a 55-min n=500 coverage entry came from a sequential
    rerun or from an 8-worker pool. Treat the sidecar as the
    counterpart of ``data/cache_metadata.json``.
    """
    out_path = MULTIPANEL_DIR / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2, default=str)
    meta = {
        "produced_at": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "n_records": len(records),
        "worker_count": int(workers),
        "blas_env": {
            k: os.environ.get(k, "(unset)")
            for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                      "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
        },
        "cpu_brand": platform.processor() or platform.machine(),
        "platform": platform.platform(),
    }
    meta_path = MULTIPANEL_DIR / "summary_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {out_path} (+ {meta_path.name})", flush=True)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=int, default=None,
                        help="Panel size (positive int). With --all, "
                             "every size in --panel-sizes runs.")
    parser.add_argument("--kind", type=str, default=None,
                        choices=SELECTION_KINDS,
                        help="Selection rule; omit with --all to run all.")
    parser.add_argument("--all", action="store_true",
                        help="Run every (panel size, kind) combination.")
    parser.add_argument("--kinds", type=str, default=None,
                        help="Comma-separated subset of selection kinds "
                             "for --all, e.g. 'coverage,adv'. "
                             "Default: every kind.")
    parser.add_argument("--panel-sizes", type=str, default=None,
                        help="Comma-separated panel sizes for --all. "
                             "Default: "
                             + ",".join(str(s) for s in DEFAULT_PANEL_SIZES))
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel subprocesses (1 = serial single "
                             "subprocess). Each subprocess handles one "
                             "(panel, kind) and exits.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if a cache exists.")
    args = parser.parse_args(argv)

    if not args.all and (args.panel is None or args.kind is None):
        parser.error("Specify --all or both --panel and --kind.")

    if args.panel_sizes:
        panel_sizes = tuple(int(s.strip()) for s in args.panel_sizes.split(","))
        assert all(s > 0 for s in panel_sizes), "panel sizes must be positive"
    else:
        panel_sizes = DEFAULT_PANEL_SIZES

    if args.kinds:
        kinds = tuple(k.strip() for k in args.kinds.split(","))
        for k in kinds:
            assert k in SELECTION_KINDS, (
                f"unknown kind {k!r}; expected {SELECTION_KINDS}")
    else:
        kinds = SELECTION_KINDS

    if args.all:
        combos = [(n, k) for n in panel_sizes for k in kinds]
    else:
        combos = [(args.panel, args.kind)]

    summary_records: list[dict] = []
    summary_path = MULTIPANEL_DIR / "summary.json"
    if summary_path.exists() and not args.force:
        with open(summary_path) as f:
            summary_records = json.load(f)
    done_keys = {(r["n"], r["kind"]) for r in summary_records}

    pending = [(n, k, args.force) for (n, k) in combos
               if (n, k) not in done_keys or args.force]
    if not pending:
        print("All requested combinations already in summary.json; "
              "use --force to re-run.")
        return 0

    print(f"\nSweep: {len(pending)} jobs across {args.workers} workers "
          f"(each in a fresh subprocess)", flush=True)

    # Each subprocess handles exactly one job (maxtasksperchild=1) so
    # ``src.config.SNAPSHOTS_DIR`` is patched cleanly at subprocess
    # start, before any stage module captures it via
    # ``from src.config import SNAPSHOTS_DIR``.
    ctx = mp.get_context("spawn")
    n_ok = 0
    n_err = 0
    with ctx.Pool(processes=max(1, args.workers),
                  maxtasksperchild=1) as pool:
        for record in pool.imap_unordered(_worker, pending):
            n_k = (record["n"], record["kind"])
            summary_records = [r for r in summary_records
                               if (r["n"], r["kind"]) != n_k]
            summary_records.append(record)
            write_summary(summary_records, workers=args.workers)
            if "error" in record:
                n_err += 1
                print(f"  ERROR {n_k[0]}/{n_k[1]}: {record['error']}",
                      flush=True)
            else:
                n_ok += 1
                print(f"  DONE {n_k[0]}/{n_k[1]}: "
                      f"{record.get('total_minutes', float('nan')):.1f} min "
                      f"({len(record.get('snapshots', {}))} snapshots)",
                      flush=True)

    print(f"\nSweep complete: {n_ok} ok, {n_err} error "
          f"(out of {len(pending)} pending)", flush=True)
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
