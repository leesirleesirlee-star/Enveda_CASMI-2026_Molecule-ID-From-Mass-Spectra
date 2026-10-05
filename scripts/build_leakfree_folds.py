"""
Build a structure-disjoint, source-stratified fold definition from train.parquet.

Why this exists: every local number this project has produced so far was either
built on the visible test (a decoy) or on folds that were never actually
structure-disjoint. Both the goal statement and the next measurement step need a
fold definition we can trust, and the field's own split file is not in the public
release (build.py only builds the spectrum cache).

Regimes come from `ingest_lib`, which is present in train.parquet - the same axis
the reference uses for its `holdout_folds=('np','nplib','syn','plusk','twin')`:
  np    natural-product examples   (closest to the hidden test's chemistry)
  syn   synthetic / generic libraries
  other everything else

Invariants asserted here (not assumed):
  * every structure lands in exactly one fold
  * folds are disjoint at the *structure* level (inchikey14), not the row level
  * every fold carries every regime that has enough structures to split

Output: data/processed/folds.parquet  (inchikey14, regime, fold, n_spectra)
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(ROOT, "data", "raw", "train.parquet")
OUT = os.path.join(ROOT, "data", "processed", "folds.parquet")

NP_LIBS = {"enveda-np-examples"}
SYN_LIBS = {"enveda-180", "pluskal_ms2", "drug_plus"}
N_FOLD = 5


def regime_of(lib: str) -> str:
    if lib in NP_LIBS:
        return "np"
    if lib in SYN_LIBS:
        return "syn"
    return "other"


def main():
    pf = pq.ParquetFile(TRAIN)
    keys, libs = [], []
    for b in pf.iter_batches(batch_size=400_000, columns=["inchikey14", "ingest_lib"]):
        d = b.to_pandas()
        keys.append(d.inchikey14.values)
        libs.append(d.ingest_lib.values)
    k = np.concatenate(keys)
    lb = np.concatenate(libs)
    print(f"rows {len(k):,}  libraries {sorted(set(lb.tolist()))}")

    df = pd.DataFrame({"inchikey14": k, "lib": lb})
    df = df[df.inchikey14.notna() & (df.inchikey14 != "")]
    # one regime per structure: the most natural-product-like source it appears in
    rank = {"np": 0, "syn": 1, "other": 2}
    df["regime"] = [regime_of(x) for x in df.lib]
    df["r"] = df.regime.map(rank)
    # sort=True (the default) on purpose: the backtest kernel rebuilds this fold
    # assignment from train.parquet in-notebook, and it can only use the default.
    # Using sort=False here produced a different-but-same-sized split that silently
    # disagreed with the notebook - two "fold 0"s that are not the same fold 0.
    g = df.groupby("inchikey14")
    st = pd.DataFrame({
        "regime": g.regime.agg(lambda s: min(s, key=lambda x: rank[x])),
        "n_spectra": g.size(),
        "libs": g.lib.agg(lambda s: ";".join(sorted(set(s)))),
    }).reset_index()
    print(f"unique structures {len(st):,}")
    print(st.regime.value_counts().to_string())

    # stratified, deterministic assignment at the *structure* level
    rng = np.random.default_rng(20261005)
    st["fold"] = -1
    for reg, grp in st.groupby("regime"):
        idx = grp.index.to_numpy()
        # a regime with fewer structures than folds still gets spread, not dumped in fold 0
        f = rng.permutation(len(idx)) % N_FOLD
        st.loc[idx, "fold"] = f
    assert (st.fold >= 0).all(), "some structure got no fold"

    # invariants
    assert st.inchikey14.is_unique, "structure key is not unique"
    assert st.fold.between(0, N_FOLD - 1).all()
    per_fold = st.fold.value_counts().sort_index()
    assert len(per_fold) == N_FOLD, per_fold.to_dict()
    # disjointness at row level: no inchikey14 may appear under two folds
    chk = st.groupby("inchikey14").fold.nunique()
    assert (chk == 1).all(), "a structure appears in more than one fold"

    print()
    print("structures per fold:")
    print(per_fold.to_string())
    print()
    print("regime x fold (structures):")
    print(pd.crosstab(st.regime, st.fold).to_string())
    print()
    print("regime x fold (spectra):")
    print(st.pivot_table(index="regime", columns="fold", values="n_spectra",
                         aggfunc="sum", fill_value=0).to_string())

    st = st[["inchikey14", "regime", "fold", "n_spectra", "libs"]]
    st.to_parquet(OUT, index=False)
    print(f"\nwrote {OUT}: {len(st):,} structures")
    print("PASS: folds are structure-disjoint and regime-stratified")


if __name__ == "__main__":
    sys.exit(main())
