"""
Diagnose the analog channel's reach.

In V_A_hard the answer's spectra are absent from the library, so the direct
channel cannot fire and the analog channel is the only way the true structure can
be surfaced. It scores ~0.048, which suggests the analog search is not reaching
the compounds that should carry the annotation.

The analog window is currently +/- ANALOG_DM (2 Da) around the query mass. For
natural products, a real analogue is often a glycosylation (+162), a prenylation
(+68), a methylation (+14) or an oxidation (+16) away -- all well inside 2 Da for
the *small* modifications, but a query whose nearest library neighbour is 20 Da
away is currently invisible.

This script measures, for each V_A query, the mass distance and fingerprint
similarity to the most similar *available* library compound, so we can see
whether the window is the binding constraint or whether the information simply
is not there.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from baseline_v1 import StructureStore, STRUCTS, TRAIN        # noqa: E402
from casmi.folds_fast import build_folds_fast                  # noqa: E402
from casmi.matching_fast import bin_spectrum                   # noqa: E402

LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def tanimoto_rows(qfp, mat):
    inter = LUT[np.minimum(mat, qfp[None, :])].sum(axis=1).astype(np.float64)
    uni = LUT[mat].sum(axis=1).astype(np.float64) + LUT[qfp].sum() - inter
    out = np.zeros(len(mat))
    nz = uni > 0
    out[nz] = inter[nz] / uni[nz]
    return out


def main():
    store = StructureStore(STRUCTS)
    folds = build_folds_fast(TRAIN, vb_limit=0, vc_limit=0, verbose=False)
    f = folds["V_A_hard"]
    removed = set(f.exclude)          # answer structures NOT in this fold's library

    # library = everything else, with fingerprints available
    lib_mask = np.array([k not in removed for k in store.keys])
    lib_idx = np.flatnonzero(lib_mask)
    lib_fp = store.fp[lib_idx]
    lib_mass = store.mass[lib_idx]
    print(f"library: {len(lib_idx):,} structures with fingerprints "
          f"(excluded {len(removed)} answers)")

    rows = []
    qkeys = list(f.queries)
    rng = np.random.default_rng(0)
    for k in qkeys:
        qi = store._key_to_idx.get(k)
        if qi is None or not store.has_fp[qi]:
            continue
        qfp = store.fp[qi]
        qm = store.mass[qi]
        # restrict to a generous analogue range so we can see the true distance
        near = np.flatnonzero((lib_mass > qm - 200) & (lib_mass < qm + 200))
        if near.size == 0:
            continue
        sub = lib_idx[near]
        tan = tanimoto_rows(qfp, store.fp[sub])
        j = int(np.argmax(tan))
        best_i = sub[j]
        rows.append({
            "key": k,
            "qmass": qm,
            "best_tanimoto": float(tan[j]),
            "dm": float(qm - store.mass[best_i]),
            "n_within_2da": int(((lib_mass > qm - 2) & (lib_mass < qm + 2)).sum()),
            "n_within_20da": int(((lib_mass > qm - 20) & (lib_mass < qm + 20)).sum()),
            "n_within_200da": int(near.size),
        })
        if len(rows) % 50 == 0:
            print(f"  {len(rows)} queries done", flush=True)

    df = pd.DataFrame(rows)
    print(f"\nanalysed {len(df)} queries")
    print("\n=== mass distance to the most fingerprint-similar library compound ===")
    print(df["dm"].abs().describe().to_string())
    print("\n|dm| buckets:")
    for lo, hi in [(0, 0.5), (0.5, 2), (2, 10), (10, 50), (50, 200)]:
        n = int(((df.dm.abs() >= lo) & (df.dm.abs() < hi)).sum())
        print(f"  {lo:5.1f}-{hi:5.1f} Da: {n:4d}  ({n/len(df)*100:5.1f}%)")
    print("\n=== Tanimoto of that best analogue ===")
    print(df["best_tanimoto"].describe().to_string())
    print("\n=== how many compounds sit inside each window ===")
    for c in ["n_within_2da", "n_within_20da", "n_within_200da"]:
        print(f"  {c}: median {int(df[c].median())}, mean {df[c].mean():.0f}")
    df.to_parquet(os.path.join(ROOT, "data", "processed", "analog_reach.parquet"),
                  index=False)
    print("\nsaved -> data/processed/analog_reach.parquet")


if __name__ == "__main__":
    main()
