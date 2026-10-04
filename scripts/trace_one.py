"""Trace ONE visible-test molecule end to end to find where truth is lost."""
import hashlib
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = r"D:\CASMI竞赛"
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from baseline_v1 import StructureStore, SpectralIndex, STRUCTS, LIB_INDEX_DIR  # noqa: E402
from casmi.core import neutral_mass_from_precursor, smiles_to_inchikey14       # noqa: E402
from casmi.spectra import clean_peaks                                          # noqa: E402


def sig(mz):
    return hashlib.md5(np.round(np.asarray(mz, np.float64), 4).tobytes()).hexdigest()


test = pd.read_parquet(os.path.join(ROOT, "data", "raw", "test.parquet"))
target_mid = str(test["molecule_id"].iloc[0])
sub = test[test["molecule_id"].astype(str) == target_mid]
print(f"molecule {target_mid}: {len(sub)} spectra")
for r in sub.itertuples(index=False):
    m = neutral_mass_from_precursor(r.precursor_mz, str(r.adduct))
    print(f"  {r.spectrum_id}  adduct={r.adduct:16s} prec={r.precursor_mz:.4f} -> M={m:.4f}")

tset = {sig(mz) for mz in sub["ms2_mzs"]}
pf = pq.ParquetFile(os.path.join(ROOT, "data", "raw", "train.parquet"))
truth = set()
for b in pf.iter_batches(batch_size=300_000,
                         columns=["ms2_mzs", "inchikey14", "normalized_smiles"]):
    df = b.to_pandas()
    for mz, k, smi in zip(df["ms2_mzs"], df["inchikey14"], df["normalized_smiles"]):
        if sig(mz) in tset:
            kk = k if isinstance(k, str) and k else smiles_to_inchikey14(smi)
            if kk:
                truth.add(kk)
print(f"truth keys from train: {sorted(truth)}")

store = StructureStore(STRUCTS)
km = store._key_mass
for k in truth:
    i = store._key_to_idx.get(k)
    print(f"  {k}: in store={i is not None}, mass={km.get(k)}")

# candidate union exactly as score_molecule builds it
allkeys = set()
for r in sub.itertuples(index=False):
    m = neutral_mass_from_precursor(r.precursor_mz, str(r.adduct))
    for ppm in (15.0, 40.0):
        allkeys |= {store.keys[i] for i in store.window(m, ppm)}
    allkeys |= {store.keys[i] for i in store.between(m - 2.0, m + 2.0)}
print(f"candidate pool size: {len(allkeys):,}")
print(f"truth in candidate pool: {truth & allkeys}")
for k in truth:
    if k not in allkeys:
        i = store._key_to_idx.get(k)
        if i is None:
            print(f"  {k} NOT IN STORE at all")
            continue
        mt = store.mass[i]
        for r in sub.itertuples(index=False):
            m = neutral_mass_from_precursor(r.precursor_mz, str(r.adduct))
            ppm = (m - mt) / mt * 1e6
            print(f"  {k}: true_mass={mt:.4f} vs query M={m:.4f} -> {ppm:+.1f} ppm "
                  f"(window 15/40 ppm)")
