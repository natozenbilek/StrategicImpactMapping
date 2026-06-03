"""Verify that only docstrings/comments have changed between two file states.

Usage:
    python _verify_codeonly.py snapshot   # write baseline
    python _verify_codeonly.py check      # compare current vs baseline
"""
import ast
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGETS = [
    "src/config.py",
    "src/utils/dcc_core.py",
    "src/stage1_data/dcc_garch.py",
    "src/stage1_data/download.py",
    "src/stage2_precision/glasso_filter.py",
    "src/stage2_precision/benchmarks.py",
    "src/stage3_direction/lead_lag.py",
    "src/stage4_network/analysis.py",
    "src/stage4_network/density_matched.py",
    "src/stage4_network/crisis_signals.py",
    "src/stage5_nsi/stress_index.py",
    "src/stage5_nsi/volume_weighted_nsi.py",
    "src/robustness/sensitivity.py",
    "run_pipeline.py",
    "paper/generate_figures.py",
    "paper/_inference.py",
    "tests/test_dcc_core.py",
]
BASELINE = ROOT / "paper" / "_codeonly_baseline.json"


def _strip_docstrings(tree: ast.AST) -> None:
    """Remove module/class/function docstrings in place."""
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef,
                                  ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = node.body
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:]


def _code_signature(path: Path) -> str:
    """Return SHA-256 of the docstring-stripped, normalised AST dump."""
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    _strip_docstrings(tree)
    dumped = ast.dump(tree, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(dumped.encode()).hexdigest()


def snapshot() -> None:
    """
    Record a docstring-stripped AST signature for every target file.

    Writes the per-file SHA-256 of the AST dump (after removing
    docstrings) to :data:`BASELINE` as JSON. The baseline is later
    consumed by :func:`check` to verify that nothing but documentation
    has drifted in the recorded files.

    Returns
    -------
    None
        Side effect: writes ``BASELINE`` to disk.
    """
    sigs = {}
    for rel in TARGETS:
        p = ROOT / rel
        if p.exists():
            sigs[rel] = _code_signature(p)
        else:
            sigs[rel] = None
    BASELINE.write_text(json.dumps(sigs, indent=2))
    print(f"Baseline written: {BASELINE} ({sum(1 for v in sigs.values() if v)} files)")


def check() -> int:
    """
    Compare current file signatures against the recorded baseline.

    For each file in :data:`TARGETS`, recomputes the AST hash with
    docstrings stripped (see :func:`_code_signature`) and compares it
    to the value recorded by a previous :func:`snapshot` call. Files
    whose hashes differ are reported on stdout.

    Returns
    -------
    int
        ``0`` when every file matches the baseline, ``1`` when at
        least one file has changed, ``2`` when the baseline file is
        missing entirely.

    Notes
    -----
    Pure documentation edits are not flagged because the AST hash is
    computed after the docstring strip.
    """
    if not BASELINE.exists():
        print(f"FAIL: baseline missing at {BASELINE}", file=sys.stderr)
        return 2
    expected = json.loads(BASELINE.read_text())
    bad = []
    missing_baseline = []
    for rel in TARGETS:
        p = ROOT / rel
        if not p.exists():
            print(f"  MISSING file: {rel}")
            continue
        try:
            current = _code_signature(p)
        except SyntaxError as e:
            print(f"  SYNTAX ERROR: {rel} : {e}")
            bad.append(rel)
            continue
        exp = expected.get(rel)
        if exp is None:
            missing_baseline.append(rel)
            continue
        if current != exp:
            bad.append(rel)
            print(f"  CHANGED: {rel}")
        else:
            print(f"  OK     : {rel}")
    if missing_baseline:
        print("\nNo baseline for:", missing_baseline)
    if bad:
        print(f"\nFAIL: {len(bad)} file(s) had non-comment changes")
        return 1
    print("\nPASS: all files match baseline (only docstrings/comments may differ).")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "snapshot":
        snapshot()
    elif cmd == "check":
        sys.exit(check())
    else:
        print(__doc__)
        sys.exit(2)
