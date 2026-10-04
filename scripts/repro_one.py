"""Reproduce the diagnostic's per-molecule scoring for a known-good molecule."""
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

store = bv.StructureStore(bv.STRUCTS)
si = bv.SpectralIndex(bv.LIB_INDEX_DIR)

targets = ["m_005e53"]
for mid in targets:
    sub = test[test["molecule_id"].astype(str) == mid]
    truth = {sig2key[sig(mz)] for mz in sub["ms2_mzs"] if sig(mz) in sig2key}
    print(f"\n=== {mid}: truth={truth} ===")
    sp = []
    for r in sub.itertuples(index=False):
        mz, it = clean_peaks(r.ms2_mzs, r.ms2_normalized_intensities)
        if mz.size < 3:
            continue
        sp.append(S(mz, it, neutral_mass_from_precursor(r.precursor_mz, str(r.adduct))))
    ranked, direct, analog, cand = bv.score_molecule(sp, store, si)
    keys = [r[0] for r in ranked[:25]]
    print(f"  spectra fed: {len(sp)}")
    print(f"  pools: cand={len(cand)}, ranked={len(ranked)}")
    print(f"  truth in cand: {truth & set(cand)}")
    print(f"  top-5 keys: {keys[:5]}")
    print(f"  truth in top-25: {truth & set(keys)}")
    for t in truth:
        pos = next((i for i, k in enumerate(keys, 1) if k == t), None)
        print(f"    {t}: rank={pos}  in_cand={t in cand}")
