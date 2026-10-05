"""
Is the isobar/isomer confusion a fundamental limit or a tuning problem?

Hypothesis under test
---------------------
A tight mass window is often assumed to solve isobaric interference. High-
resolution instruments reach 1-5 ppm, so a +/-10-40 ppm filter is considered
highly selective. But small isobaric substitutions have mass defects that are
LARGE compared with a 10-40 ppm window at these masses:

    CH2 -> N      : 12.000000 - 14.003074 = -3.074 mDa   (N is heavier)
    O   -> CH4    : 15.994915 - 16.031300 = -36.4 uDa
    C2H4-> CO     : 28.031300 - 27.994915 = +36.4 uDa
    CH2O-> C2H4   : 30.010565 - 28.031300 = +1979 uDa

So at m/z 400, +/-40 ppm is +/-16 mDa, which ADMITS several of these
substitutions. The consequence would be that isobaric (different-formula)
candidates cannot be removed by tightening the mass window within a wide range,
and the only clean filter is the formula itself.

This script measures, on the real candidate pool, how many DISTINCT formulas and
how many structures fall inside a given ppm window, and how that ratio behaves
across the ppm range. If the count of distinct formulas stays >1 even at very
tight windows, the isobar floor is instrumental, not an artifact of tuning.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = r"D:\CASMI竞赛"
POOL = os.path.join(ROOT, "data", "processed", "candidate_pool.parquet")


def main():
    df = pd.read_parquet(POOL, columns=["key", "mass", "formula"])
    df = df[np.isfinite(df["mass"])].copy()
    tr = pd.read_parquet(os.path.join(ROOT, "data", "processed", "structures.parquet"),
                         columns=["key", "formula"])
    kf = dict(zip(tr["key"], tr["formula"]))
    df["formula"] = [f if isinstance(f, str) and f and f.strip() else kf.get(k)
                     for k, f in zip(df["key"], df["formula"])]
    df = df[df["formula"].notna()]
    df["formula"] = df["formula"].astype(str)
    print(f"pool with formula: {len(df):,} structures, "
          f"{df['formula'].nunique():,} distinct formulas")

    masses = np.sort(df["mass"].to_numpy(np.float64))
    # formula -> its exact mass (use the median mass of its members, they agree)
    f_mass = df.groupby("formula")["mass"].median().to_dict()
    f_arr = np.array(list(f_mass.values()))
    f_names = list(f_mass.keys())
    order = np.argsort(f_arr)
    f_arr = f_arr[order]
    f_names = [f_names[i] for i in order]
    print(f"distinct formula masses: {len(f_arr):,}\n")

    rng = np.random.default_rng(0)
    # sample query masses from the actual pool (the region of interest)
    q_idx = rng.choice(len(masses), size=400, replace=False)
    q_masses = masses[q_idx]

    print(f"{'ppm':>6} {'structures median':>18} {'formulas median':>16} "
          f"{'frac formulas>1':>16} {'iso/isomer ratio':>17}")
    print("-" * 80)
    for ppm in (5.0, 10.0, 20.0, 40.0, 100.0, 200.0):
        n_struct, n_form = [], []
        for q in q_masses:
            tol = abs(q) * ppm * 1e-6
            lo = np.searchsorted(masses, q - tol, "left")
            hi = np.searchsorted(masses, q + tol, "right")
            n_struct.append(int(hi - lo))
            lo2 = np.searchsorted(f_arr, q - tol, "left")
            hi2 = np.searchsorted(f_arr, q + tol, "right")
            n_form.append(int(hi2 - lo2))
        n_struct = np.asarray(n_struct)
        n_form = np.asarray(n_form)
        ratio = np.median(n_struct / np.maximum(n_form, 1))
        print(f"{ppm:6.0f} {int(np.median(n_struct)):18,d} {int(np.median(n_form)):16,d} "
              f"{(n_form > 1).mean()*100:15.1f}% {ratio:17.1f}")

    print()
    print("Reading: 'iso/isomer ratio' is structures per distinct formula, i.e. how")
    print("many same-formula (true isomer) competitors sit inside the window per")
    print("formula. If 'formulas median' stays >1 even at 5-10 ppm, then isobaric")
    print("interference is NOT removable by tightening the mass window, and the")
    print("formula is the only clean discriminator.")

    # explicit substitution table for context
    print("\nIsobaric substitution mass defects (mDa), for reference:")
    subs = {"N -> CH2": 14.003074 - 12.0,
            "O -> CH4": 16.031300 - 15.994915,
            "CO -> C2H4": 28.031300 - 27.994915,
            "CH2 -> N": 12.0 - 14.003074,
            "NH2 -> O": 16.018724 - 15.994915,
            "S -> O2": 31.972071 - 31.989829}
    for k, v in subs.items():
        print(f"  {k:12s} {v*1000:+8.2f} mDa   "
              f"= {abs(v)/400*1e6:7.1f} ppm at m/z 400")


if __name__ == "__main__":
    main()
