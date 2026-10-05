"""
Test the suggest's central claim on real data before rebuilding anything.

Claim under test (from docs/suggest_for_ChatGPT.md §1A, attributed to a public
"Lessons from ~30 submissions" discussion):
    when the correct molecule is in the candidate list but loses rank, the wrong
    winner has the SAME molecular formula ~98% of the time.
    => "mass windows are not the main lever anymore; structure-dependent evidence
       such as in-silico fragmentation is where gains appear."

If true, the right architecture is:
    formula-correct candidate set  ->  connectivity-aware re-scoring
and NOT wider mass windows or bigger pools.

This script measures, on the visible test (whose truth we can recover verbatim
from train):
  1. does the true structure have the SAME formula as the query? (must be ~100%
     if the claim holds)
  2. how many candidates survive an exact-formula filter vs a 40 ppm mass filter?
  3. of the mass-window candidates, what fraction share the query formula?

That tells us whether a formula filter is a free, huge precision win.
"""

from __future__ import annotations

import hashlib
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = r"D:\CASMI竞赛"
sys.path.insert(0, os.path.join(ROOT, "src"))
from casmi.core import formula_neutral_mass, neutral_mass_from_precursor  # noqa: E402

TEST = os.path.join(ROOT, "data", "raw", "test.parquet")
TRAIN = os.path.join(ROOT, "data", "raw", "train.parquet")
POOL = os.path.join(ROOT, "data", "processed", "candidate_pool.parquet")


def sig(mz):
    return hashlib.md5(np.round(np.asarray(mz, np.float64), 4).tobytes()).hexdigest()


def main():
    test = pd.read_parquet(TEST)
    tset = {sig(mz) for mz in test["ms2_mzs"]}

    # map every test spectrum to the structure train associates with it, and
    # record that structure's formula
    pf = pq.ParquetFile(TRAIN)
    sig2key, key2formula = {}, {}
    for b in pf.iter_batches(batch_size=300_000,
                             columns=["ms2_mzs", "inchikey14", "molecular_formula"]):
        df = b.to_pandas()
        for mz, k, f in zip(df["ms2_mzs"], df["inchikey14"], df["molecular_formula"]):
            h = sig(mz)
            if h in tset:
                if isinstance(k, str) and k:
                    sig2key.setdefault(h, k)
                if isinstance(f, str) and f:
                    key2formula.setdefault(k, f)
    # fill formulas for any key missing one
    for b in pf.iter_batches(batch_size=300_000,
                             columns=["inchikey14", "molecular_formula"]):
        df = b.to_pandas()
        for k, f in zip(df["inchikey14"], df["molecular_formula"]):
            if isinstance(k, str) and isinstance(f, str) and f:
                key2formula.setdefault(k, f)
    print(f"resolved {len(sig2key):,}/{len(tset):,} test spectra -> structure key")
    print(f"formulas known for {len(key2formula):,} keys", flush=True)

    # candidate pool, indexed by formula
    pool = pd.read_parquet(POOL, columns=["key", "mass", "sources"])
    pool = pool[np.isfinite(pool["mass"])]
    pool_keys = set(pool["key"])

    # formulas from the train structure table (the pool carries formula for some)
    tr = pd.read_parquet(os.path.join(ROOT, "data", "processed", "structures.parquet"),
                         columns=["key", "formula", "mass"])
    kf = dict(zip(tr["key"], tr["formula"]))
    for k, f in key2formula.items():
        kf.setdefault(k, f)

    # pool formula table: use whatever we know, else derive from the key's formula
    pool["formula"] = [kf.get(k) for k in pool["key"]]
    by_formula = defaultdict(list)
    for k, f in zip(pool["key"], pool["formula"]):
        if isinstance(f, str) and f:
            by_formula[f].append(k)
    print(f"pool formulas indexed: {len(by_formula):,} distinct\n", flush=True)

    mass = np.sort(pool["mass"].to_numpy(np.float64))

    n = 0
    same_formula = 0
    n_mass_win = []
    n_formula_win = []
    frac_formula_in_mass = []
    for mid, sub in test.groupby(test["molecule_id"].astype(str), sort=False):
        truth = {sig2key[sig(mz)] for mz in sub["ms2_mzs"] if sig(mz) in sig2key}
        if not truth:
            continue
        # query formula: from the spectra's own molecular_formula if present,
        # else from the recovered truth structure
        qf = None
        if "molecular_formula" in sub.columns:
            for f in sub["molecular_formula"]:
                if isinstance(f, str) and f:
                    qf = f
                    break
        if qf is None:
            for t in truth:
                if isinstance(kf.get(t), str):
                    qf = kf[t]
                    break
        if qf is None:
            continue
        n += 1
        if any(kf.get(t) == qf for t in truth):
            same_formula += 1

        # mass window candidates (40 ppm) using imidiate median neutral mass
        masses = []
        for r in sub.itertuples(index=False):
            f = formula_neutral_mass(getattr(r, "molecular_formula", None))
            m = f if f is not None else neutral_mass_from_precursor(
                r.precursor_mz, str(r.adduct))
            masses.append(m)
        qm = float(np.median(masses))
        tol = abs(qm) * 40e-6
        lo = np.searchsorted(mass, qm - tol, "left")
        hi = np.searchsorted(mass, qm + tol, "right")
        nm = int(hi - lo)
        nf = len(by_formula.get(qf, ()))
        n_mass_win.append(nm)
        n_formula_win.append(nf)
        if nm:
            frac_formula_in_mass.append(min(nf, nm) / nm)

    n_mass_win = np.asarray(n_mass_win)
    n_formula_win = np.asarray(n_formula_win)
    print(f"molecules analysed: {n}\n")
    print("=== claim 1: does the TRUE structure share the query formula? ===")
    print(f"  same formula: {same_formula}/{n} = {same_formula/n*100:.1f}%")
    print("\n=== claim 2: selectivity of each filter ===")
    print(f"  mass window (±40 ppm) median candidates : {int(np.median(n_mass_win)):,}")
    print(f"  exact formula median candidates         : {int(np.median(n_formula_win)):,}")
    f_ = np.asarray(frac_formula_in_mass)
    if f_.size:
        print(f"  fraction of mass-window candidates that share the query formula: "
              f"median {np.median(f_)*100:.1f}%")
    print("\n=> If the formula filter is much tighter AND keeps the truth, then")
    print("   same-formula grouping is the right frame for connectivity scoring.")


if __name__ == "__main__":
    main()
