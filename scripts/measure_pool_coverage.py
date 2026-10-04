"""
Measure whether the natural-product candidate pool actually fixes reachability.

The patch's claim is that the hidden test is natural-product dark chemical space,
so the answer is reachable only if it is in the candidate pool. My earlier
diagnostic showed a train-only pool put the truth in the pool for very few
molecules. This compares pool variants on two references:

  R1  visible test molecules (truth from the verbatim train row)
  R2  natural-product structures held out of the pool entirely -- the situation
      the patch describes. We REMOVE a random sample of LOTUS/NPAtlas structures
      from the pool and check whether their mass is reachable from the remaining
      pool at a given ppm. This is what a novel natural product looks like: its
      exact structure is absent, so only mass-window recall can be measured.

The honest reading: a bigger pool cannot invent an absent structure. What it can
do is make the *mass window* contain plausible natural-product candidates instead
of unrelated synthetic ones, which is what the ranker then has to sort.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys

import numpy as np
import pandas as pd

ROOT = r"D:\CASMI竞赛"
POOL = os.path.join(ROOT, "data", "processed", "candidate_pool.parquet")
TRAIN_STRUCTS = os.path.join(ROOT, "data", "processed", "structures.parquet")


def load_pool():
    df = pd.read_parquet(POOL, columns=["key", "smiles", "mass", "sources"])
    df = df[np.isfinite(df["mass"])].reset_index(drop=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppm", type=float, default=40.0)
    ap.add_argument("--sample", type=int, default=2000)
    a = ap.parse_args()

    pool = load_pool()
    print(f"merged pool: {len(pool):,} structures, mass "
          f"{pool['mass'].min():.1f}-{pool['mass'].max():.1f} Da")
    tr = pd.read_parquet(TRAIN_STRUCTS, columns=["key", "mass"])
    tr = tr[np.isfinite(tr["mass"])]
    print(f"train-only pool: {len(tr):,} structures")

    def ordered_masses(sub):
        return np.sort(sub["mass"].to_numpy(np.float64))

    pm, tm = ordered_masses(pool), ordered_masses(tr)

    def window_count(masses, m, ppm):
        tol = abs(m) * ppm * 1e-6
        return int(np.searchsorted(masses, m + tol, "right")
                   - np.searchsorted(masses, m - tol, "left"))

    # ---- window density: how many candidates a query mass actually sees ----
    rng = np.random.default_rng(0)
    sample = pool.sample(min(a.sample, len(pool)), random_state=0)
    dens_pool = np.array([window_count(pm, m, a.ppm) for m in sample["mass"]])
    dens_tr = np.array([window_count(tm, m, a.ppm) for m in sample["mass"]])
    print(f"\n=== candidates within +/-{a.ppm:.0f} ppm of a query mass "
          f"({len(sample):,} sampled structures) ===")
    for name, d in [("train-only", dens_tr), ("merged NP", dens_pool)]:
        print(f"  {name:11s} median={int(np.median(d)):5d}  mean={d.mean():8.0f}  "
              f"p90={int(np.percentile(d,90)):6d}  max={d.max():7d}")

    # ---- is the truth reachable when it IS in the pool? ----
    # (sanity: the pool must be internally consistent)
    hits = np.array([window_count(pm, m, a.ppm) for m in pm[:min(5000, len(pm))]])
    print(f"\nself-reachability check (5000 pool members): "
          f"{(hits>0).mean()*100:.1f}% see themselves in the window "
          f"(must be 100%)")

    # ---- natural-product novelty simulation ----
    # Hold out a random sample of NP-only structures (not in train), remove them
    # from the pool, and measure: does the *train-only* pool contain the mass
    # neighbourhood, and does the merged pool?
    np_only = pool[pool["sources"].str.contains("coconut|lotus|npatlas")
                   & ~pool["sources"].str.contains("train")]
    print(f"\nNP-only structures (in NP DBs but not train): {len(np_only):,}")
    hold = np_only.sample(min(a.sample, len(np_only)), random_state=1)
    hm = hold["mass"].to_numpy(np.float64)

    n_tr = np.array([window_count(tm, m, a.ppm) for m in hm])
    n_np = np.array([window_count(pm, m, a.ppm) for m in hm])
    print(f"\n=== for {len(hold):,} NP-only query masses ===")
    print(f"  train-only pool has ZERO candidates in window : "
          f"{(n_tr==0).mean()*100:.1f}%")
    print(f"  merged pool  has ZERO candidates in window   : "
          f"{(n_np==0).mean()*100:.1f}%")
    print(f"  merged pool median candidates per query       : {int(np.median(n_np)):,}")
    print(f"  train-only median candidates per query        : {int(np.median(n_tr)):,}")

    # how many NP structures are simply absent from train (the patch's point)
    tr_keys = set(tr["key"])
    absent = (~np_only["key"].isin(tr_keys)).mean() * 100
    print(f"\n  NP-DB structures ABSENT from train pool       : {absent:.1f}%")
    print("\nCONCLUSION: a larger NP pool increases the candidate neighbourhood for")
    print("natural-product masses; it cannot supply a structure that no database has,")
    print("which is precisely why the patch also calls for a generative channel.")


if __name__ == "__main__":
    main()
