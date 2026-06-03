"""Re-fit A-DCC MLE on the cached z_subset after the Dirichlet rescale.

Loads results/snapshots/stage1_results.pkl, re-runs estimate_adcc on the
same 100-ticker balanced sub-panel with the patched seed protocol, then
writes the updated adcc_params back into the same pickle. Backs up the
original to .bak_pre_dirichlet_fix before overwriting.

GARCH fits, snapshot R̄, and tickers are untouched. Snapshot R̄ would
only change if (a, b, g) shifted enough to perturb the recursion at the
4-decimal printed precision; we therefore re-build it iff the new
(a, b, g) differ from the cached values by > 1e-4 in any coordinate.
"""
import pickle
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.stage1_data.dcc_garch import estimate_adcc, extract_snapshot_correlations

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "results" / "snapshots" / "stage1_results.pkl"
BACKUP = ROOT / "results" / "snapshots" / "stage1_results.pkl.bak_pre_dirichlet_fix"

print(f"[refit] Loading {CACHE}")
with open(CACHE, "rb") as f:
    s1 = pickle.load(f)

z_df = s1["z_df"]
adcc_old = s1["adcc_params"]
subset_tickers = adcc_old["subset_tickers"]
print(f"[refit] subset_tickers = {len(subset_tickers)} tickers")

z_subset = z_df[subset_tickers].dropna(how="any").values
print(f"[refit] z_subset shape: {z_subset.shape}")
assert z_subset.shape[0] == adcc_old["subset_T"], (
    z_subset.shape[0], adcc_old["subset_T"])

print(f"[refit] Old fit: a={adcc_old['a']:.6f}, b={adcc_old['b']:.6f}, "
      f"g={adcc_old['g']:.6f}, "
      f"surviving={adcc_old['n_surviving_seeds']}/{adcc_old['n_attempted_seeds']}, "
      f"rejected_init={adcc_old['n_rejected_init']}")

print(f"[refit] Backup -> {BACKUP}")
shutil.copy2(CACHE, BACKUP)

adcc_new = estimate_adcc(z_subset)
adcc_new["Q_bar_subset"] = adcc_new["Q_bar"]
adcc_new["N_bar_subset"] = adcc_new["N_bar"]
adcc_new["subset_tickers"] = subset_tickers
adcc_new["subset_k"] = len(subset_tickers)
adcc_new["subset_T"] = int(z_subset.shape[0])
adcc_new["full_k"] = z_df.shape[1]

print(f"\n[refit] New fit: a={adcc_new['a']:.6f}, b={adcc_new['b']:.6f}, "
      f"g={adcc_new['g']:.6f}, "
      f"surviving={adcc_new['n_surviving_seeds']}/{adcc_new['n_attempted_seeds']}, "
      f"rejected_init={adcc_new['n_rejected_init']}")

delta = max(abs(adcc_new["a"] - adcc_old["a"]),
            abs(adcc_new["b"] - adcc_old["b"]),
            abs(adcc_new["g"] - adcc_old["g"]))
print(f"[refit] max |Δθ| vs old = {delta:.2e}")

s1["adcc_params"] = adcc_new
if delta > 1e-4:
    print(f"[refit] Δθ > 1e-4; rebuilding per-snapshot R̄ accumulator")
    s1["snapshot_correlations"] = extract_snapshot_correlations(z_df, adcc_new)
else:
    print(f"[refit] Δθ ≤ 1e-4; leaving snapshot_correlations untouched")

with open(CACHE, "wb") as f:
    pickle.dump(s1, f)
print(f"[refit] Wrote {CACHE}")
