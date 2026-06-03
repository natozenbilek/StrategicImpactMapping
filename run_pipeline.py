"""Pipeline orchestrator for the five-stage S&P 500 stress-network analysis.

Stage 1 : A-DCC GARCH (univariate GARCH(1,1)-t margins + asymmetric DCC)
Stage 2 : Graphical LASSO precision-matrix filter
Stage 3 : Lead-follower direction assignment
Stage 4 : Network analysis (Q1-Q4)
Stage 5 : Network Stress Index (opt-in via --extras)

Each stage caches its output to results/snapshots/stageN_results.pkl.
--force invalidates only the currently-running stage's cache; upstream
caches still load so e.g. `--stages 4 --force` keeps the cached
Stage-3 directed graph instead of falling back to Stage-2's undirected
form.
"""
import argparse
import time
import sys
import pickle
from pathlib import Path

# Force UTF-8 on Windows / Turkish locales so the box-drawing glyphs in
# the section banners do not crash the run when piped through tee.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, Exception):
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import SNAPSHOTS_DIR, RESULTS_DIR


def main():
    parser = argparse.ArgumentParser(
        description="Strategic Impact Mapping Pipeline (proposal-aligned)")
    parser.add_argument("--test", action="store_true", help="Test mode with 30 assets")
    parser.add_argument("--n-assets", type=int, default=None, help="Number of assets to use")
    parser.add_argument("--force", action="store_true", help="Force re-run, ignore cache")
    parser.add_argument("--stages", type=str, default=None,
                        help="Comma-separated stages to run (default: 1,2,3,4)")
    parser.add_argument("--extras", action="store_true",
                        help="[DEV] include Stage 5 (NSI).")
    parser.add_argument("--skip-leadlag", action="store_true",
                        help="[DEV] skip Stage 3; breaks direction-dependent Stage 4 metrics")
    parser.add_argument("--skip-motifs", action="store_true",
                        help="[DEV] skip Q4 motif analysis")
    parser.add_argument("--skip-rolling", action="store_true",
                        help="[DEV] skip rolling NSI inside Stage 5")
    parser.add_argument("--augmented", action="store_true",
                        help="Add MER (Wayback) + FNMA + FMCC (Yahoo pink-sheet) "
                             "to the panel. Cache writes to "
                             "results/snapshots/augmented/ so the baseline "
                             "cache is left intact. Implies --force.")
    args = parser.parse_args()

    if args.augmented:
        import src.config as _cfg
        _cfg.SNAPSHOTS_DIR = _cfg.RESULTS_DIR / "snapshots" / "augmented"
        _cfg.SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        # The augmented panel adds three tickers; baseline caches are
        # cardinality-incompatible, so force a fresh run rather than
        # silently loading a stale stage1 pickle keyed on 501 columns.
        args.force = True
        print(f"[AUGMENTED MODE] cache dir → {_cfg.SNAPSHOTS_DIR}")

    n_assets = args.n_assets or (500 if args.test else None)

    if args.stages:
        stages_to_run = set(int(s) for s in args.stages.split(","))
        invalid = stages_to_run - {1, 2, 3, 4, 5}
        if invalid:
            parser.error(f"Unknown stage(s): {sorted(invalid)}. Valid: 1-5.")
    else:
        stages_to_run = {1, 2, 3, 4}
        if args.extras:
            stages_to_run |= {5}

    if args.skip_leadlag:
        stages_to_run.discard(3)
    use_leadlag = 3 in stages_to_run

    print("=" * 60)
    print("  Strategic Impact Mapping in Financial Markets")
    print("=" * 60)
    print(f"\n  Mode: {'TEST' if args.test else 'FULL'}")
    print(f"  Assets: {n_assets or 'ALL'}")
    print(f"  Stages: {sorted(stages_to_run)}")
    print(f"  Lead-Lag (Stage 3): {'YES' if use_leadlag else 'SKIP (debug)'}")
    print(f"  Motif Analysis (Q4): {'YES' if not args.skip_motifs else 'SKIP (debug)'}")
    print(f"  Extras (Stage 5 NSI): {'YES' if args.extras else 'NO'}")
    print(f"  Force: {args.force}\n")

    total_start = time.time()
    stage_timings = {}
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Stage 1: data + A-DCC
    if 1 in stages_to_run:
        print("\n" + "=" * 60)
        print("STAGE 1: Data Download + A-DCC GARCH Estimation")
        print("=" * 60)
        t0 = time.time()
        from src.stage1_data.download import run_download
        from src.stage1_data.dcc_garch import run_stage1
        info, prices, returns = run_download(force=args.force, augmented=args.augmented)
        stage1 = run_stage1(returns, n_assets=n_assets, force=args.force)
        stage_timings["Stage 1: Data + A-DCC"] = time.time() - t0
        print(f"\n  Stage 1 completed in {stage_timings['Stage 1: Data + A-DCC']:.1f}s")
    else:
        with open(SNAPSHOTS_DIR / "stage1_results.pkl", "rb") as f:
            stage1 = pickle.load(f)
        from src.stage1_data.download import run_download
        info, prices, returns = run_download(augmented=args.augmented)

    # Stage 2: GLASSO
    if 2 in stages_to_run:
        print("\n" + "=" * 60)
        print("STAGE 2: GLASSO Precision Matrix Filtering")
        print("=" * 60)
        t0 = time.time()
        from src.stage2_precision.glasso_filter import run_stage2
        stage2 = run_stage2(stage1["snapshot_correlations"], force=args.force)
        stage_timings["Stage 2: GLASSO"] = time.time() - t0
        print(f"\n  Stage 2 completed in {stage_timings['Stage 2: GLASSO']:.1f}s")
    else:
        with open(SNAPSHOTS_DIR / "stage2_results.pkl", "rb") as f:
            stage2 = pickle.load(f)

    # Stage 3: lead-follower direction
    stage3 = None
    if use_leadlag:
        print("\n" + "=" * 60)
        print("STAGE 3: Lead-Follower Direction Assignment")
        print("=" * 60)
        t0 = time.time()
        from src.stage3_direction.lead_lag import run_stage3
        stage3 = run_stage3(stage2, returns, force=args.force)
        stage_timings["Stage 3: Lead-Lag"] = time.time() - t0
        print(f"\n  Stage 3 completed in {stage_timings['Stage 3: Lead-Lag']:.1f}s")
    else:
        # Reload Stage 3 cache (if any) so `--stages 4 --force` does not
        # silently drop the directed graph and break Stage 4 metrics.
        cache3 = SNAPSHOTS_DIR / "stage3_results.pkl"
        if cache3.exists():
            with open(cache3, "rb") as f:
                stage3 = pickle.load(f)

    # Stage 4: network analysis
    if 4 in stages_to_run:
        print("\n" + "=" * 60)
        print("STAGE 4: Network Analysis & Strategic Characterization")
        print("=" * 60)
        t0 = time.time()
        from src.stage4_network.analysis import run_stage4
        if stage3 is None:
            raise RuntimeError(
                "Stage 4 requires Stage 3 output; rerun without --skip-leadlag "
                "or include stage 3 in --stages.")
        stage4 = run_stage4(stage3, sp500_info=info,
                            run_motifs=not args.skip_motifs,
                            n_motif_rewires=50 if args.test else 100,
                            n_er_sims=100 if args.test else 200,
                            force=args.force)
        stage_timings["Stage 4: Network"] = time.time() - t0
        print(f"\n  Stage 4 completed in {stage_timings['Stage 4: Network']:.1f}s")
    else:
        with open(SNAPSHOTS_DIR / "stage4_results.pkl", "rb") as f:
            stage4 = pickle.load(f)

    # Stage 5: NSI (opt-in via --extras)
    stage5 = None
    if 5 in stages_to_run:
        print("\n" + "=" * 60)
        print("STAGE 5 [EXTRAS]: Network Stress Index (NSI)")
        print("=" * 60)
        t0 = time.time()
        from src.stage5_nsi.stress_index import run_stage5
        stage5 = run_stage5(stage4_results=stage4,
                            returns=returns if not args.skip_rolling else None,
                            snapshot_correlations=stage1["snapshot_correlations"],
                            force=args.force)
        stage_timings["Stage 5 [extras]: NSI"] = time.time() - t0

    # Summary
    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Total time: {total_time:.1f}s ({total_time / 60:.1f} min)")
    print(f"  Results cached in: {SNAPSHOTS_DIR}")

    if stage_timings:
        print(f"\n  {'Stage':<35} {'Duration':>10} {'% Total':>8}")
        print("  " + "-" * 55)
        for name, dur in stage_timings.items():
            pct = 100 * dur / total_time if total_time > 0 else 0
            dur_str = f"{dur / 60:.1f} min" if dur >= 60 else f"{dur:.1f}s"
            print(f"  {name:<35} {dur_str:>10} {pct:>7.1f}%")
        print("  " + "-" * 55)
        print(f"  {'TOTAL':<35} {total_time / 60:>9.1f}m {'100.0%':>8}")

    # Per-snapshot Q1-Q4 console summary
    if stage4:
        print(f"\n  {'Snapshot':<22} {'Regime':<9} {'Edges':>6} {'|Z_C|':>6} "
              f"{'HHI':>8} {'Q(mod)':>8} {'Purity':>7}")
        print("  " + "-" * 72)
        for label, data in stage4.items():
            zc = abs(data["erdos_renyi"]["z_scores"]["clustering"])
            Q = data["community"]["modularity"]
            purity = data["community"].get("purity", float("nan"))
            # `hhi_top10` is a legacy key; the stored value is the full-network HHI.
            hhi = data["pagerank"]["hhi_top10"]
            print(f"  {label:<22} {data['regime']:<9} {data['n_edges']:>6} "
                  f"{zc:>6.1f} {hhi:>8.4f} {Q:>8.4f} {purity:>7.4f}")

        if any(d.get("motifs") for d in stage4.values()):
            print(f"\n  Motif Significance Profile (SP):")
            print(f"  {'Snapshot':<22} {'Regime':<9} {'FFL':>7} {'MR':>7} {'SIM':>7}")
            print("  " + "-" * 55)
            for label, data in stage4.items():
                m = data.get("motifs")
                if not m:
                    continue
                sp = m["significance_profile"]
                print(f"  {label:<22} {data['regime']:<9} "
                      f"{sp.get('feed_forward_loop', 0):>7.3f} "
                      f"{sp.get('mutual_regulation', 0):>7.3f} "
                      f"{sp.get('single_input_module', 0):>7.3f}")

    if stage5 and "snapshot_nsi" in stage5:
        print(f"\n  [EXTRAS] NSI Summary:")
        for _, row in stage5["snapshot_nsi"].iterrows():
            print(f"    {row['snapshot']:<22} NSI={row['nsi']:.4f}  ({row['regime']})")


if __name__ == "__main__":
    main()
