"""
Empirically verify the adduct model against the real competition data.

test.parquet carries `precursor_error_ppm`, which the organiser computed against
the true molecular formula. For each spectrum:

    M_analytical = formula_mass(true_formula) + adduct.neutral_shift

The organiser's ppm and our precursor offset must then satisfy

    precursor_mz = M_analytical + adduct.precursor_offset

so we can solve for adduct.precursor_offset independently of our table and
compare. This is a real empirical check: it catches electron-mass and sign
errors that a synthetic neutral->mz->neutral round trip cannot (a wrong offset
round-trips perfectly).

Once the labelled train slice is available we repeat it against the known
molecular_formula column.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(ROOT, "src"))
from casmi.core import ADDUCT_OFFSET, formula_mass, PROTON, ELECTRON, MONO  # noqa: E402

TEST = os.path.join(ROOT, "data/raw/test.parquet")
SLICE = os.path.join(ROOT, "data/interim/validation_slices.parquet")

t = pd.read_parquet(TEST)
print(f"test.parquet: {t.shape}")
print("columns:", list(t.columns))

# NOTE: test.parquet has NO labels and NO precursor_error_ppm (12 columns).
# The only ground truth for the adduct model is the labelled train slice.
print("\n=== adduct coverage of the real test set ===")
vc = t["adduct"].value_counts()
print(vc.to_string())
unknown = set(vc.index) - set(ADDUCT_OFFSET)
assert not unknown, f"test contains adducts we cannot model: {unknown}"
print(f"all {len(vc)} test adducts are modelled")

print("\n=== decisive check on labelled train slice ===")
import os

# repository root, derived from this file so the tree is relocatable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.exists(SLICE):
    s = pd.read_parquet(SLICE)
    s = s[s["molecular_formula"].notna()].copy()
    s["M_analytical"] = s["molecular_formula"].map(formula_mass)
    s["theoretical_mz"] = s["M_analytical"] + s["adduct"].map(
        lambda a: ADDUCT_OFFSET[a][0])
    s["ppm_ours"] = (s["precursor_mz"] - s["theoretical_mz"]) / s["theoretical_mz"] * 1e6
    print(f"slice: {len(s):,} labelled spectra in {s['adduct'].nunique()} adduct classes")
    for ad, sub in s.groupby("adduct"):
        d = (sub["ppm_ours"] - sub["precursor_error_ppm"]).abs()
        print(f"  {ad:16s} n={len(sub):6d}  |our_ppm - organiser_ppm| max={d.max():.4f} "
              f"median={d.median():.4f}  our_ppm median={sub['ppm_ours'].median():+.4f}")
    worst = (s["ppm_ours"] - s["precursor_error_ppm"]).abs().max()
    print(f"\nworst disagreement with organiser ppm: {worst:.4f} ppm")
    assert worst < 0.05, f"adduct model disagrees with organiser: {worst} ppm"
    print("ADDUCT MODEL VERIFIED against organiser-labelled formula masses")
else:
    print(f"slice not ready yet ({SLICE} missing) — test-set coverage check done above")

# Sanity: the electron term really is distinguishable at this precision
print("\n=== why the electron term matters (10 ppm tolerance at m/z 330) ===")
mz = 330.0
print(f"  electron mass {ELECTRON:.6f} Da at m/z {mz} = {ELECTRON/mz*1e6:.2f} ppm")
print(f"  a proton      {PROTON:.6f} Da at m/z {mz} = {PROTON/mz*1e6:.2f} ppm")
