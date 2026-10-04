"""Instrumented run: per-molecule truth-in-pool and rank, no aggregation hiding bugs."""
import hashlib
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = r"D:\CASMI竞赛"
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import baseline_v1 as bv                                             # noqa: E402
from casmi.core import neutral_mass_from_precursor, smiles_to_inchikey14  # noqa: E402
from casmi.spectra import clean_peaks                                # noqa: E402


def sig(mz):
    return hashlib.md5(np.round(np.asarray(mz, np.float64), 4).tobytes()).hexdigest()


class S:
    __slots__ = ("mz", "intensity", "neutral_mass")

    def __init__(self, mz, i, m):
        self.mz, self.intensity, self.neutral_mass = mz, i, m


N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
test = pd.read_parquet(os.path.join(ROOT, "data", "raw", "test.parquet"))
tset = {sig(mz) for mz in test["ms2_mzs"]}
pf = pq.ParquetFile(os.path.join(ROOT, "data", "raw", "train.parquet"))
sig2key = {}
for b in pf.iter_batches(batch_size=300_000,
                         columns=["ms2_mzs", "inchikey14", "normalized_smiles"]):
    df = b.to_pandas()
    for mz, k, smi in zip(df["ms2_mzs"], df["inchikey14"], df["normalized_smiles"]):
        h = sig(mz)
        if h in tset:
            kk = k if isinstance(k, str) and k else smiles_to_inchikey14(smi)
            if kk:
                sig2key.setdefault(h, kk)
print(f"sig2key: {len(sig2key)}/{len(tset)}")

store = bv.StructureStore(bv.STRUCTS)
si = bv.SpectralIndex(bv.LIB_INDEX_DIR)

mids = [str(m) for m in test["molecule_id"].drop_duplicates()][:N]
ranks, in_pool = [], []
for mid in mids:
    sub = test[test["molecule_id"].astype(str) == mid]
    truth = {sig2key[sig(mz)] for mz in sub["ms2_mzs"] if sig(mz) in sig2key}
    sp = []
    for r in sub.itertuples(index=False):
        mz, it = clean_peaks(r.ms2_mzs, r.ms2_normalized_intensities)
        if mz.size < 3:
            continue
        sp.append(S(mz, it, neutral_mass_from_precursor(r.precursor_mz, str(r.adduct))))
    ranked, direct, analog, cand = bv.score_molecule(sp, store, si)
    keys = [r[0] for r in ranked[:25]]
    r = next((i for i, k in enumerate(keys, 1) if k in truth), 0)
    ranks.append(r)
    in_pool.append(1 if (truth & set(cand)) else 0)
    print(f"  {mid}: spectra={len(sp)} cand={len(cand):6d} truth_in_pool="
          f"{bool(truth & set(cand))} rank={r}")

ranks = np.asarray(ranks)
print()
print(f"molecules: {len(ranks)}")
print(f"truth in candidate pool : {sum(in_pool)}/{len(in_pool)} = {np.mean(in_pool)*100:.1f}%")
print(f"truth in top-25         : {(ranks>0).sum()}/{len(ranks)} = {(ranks>0).mean()*100:.1f}%")
print(f"MRR@25                  : {np.mean(np.where(ranks>0, 1.0/np.maximum(ranks,1), 0.0)):.4f}")
