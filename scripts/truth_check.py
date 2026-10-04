"""
Truth-recovery check on the visible test, time-bounded and instrumented.

The earlier parallel version reported "truth in candidate pool = 1.2%", which
contradicts a direct single-molecule trace (m_005e53: truth in pool, ranked #5).
Before trusting either number I need a run that prints per-molecule evidence and
stops on a wall-clock budget, so a partial result is still interpretable.

Ground truth: the visible test spectra are verbatim copies of train rows, so each
test spectrum can be mapped to the structure train associates with it.
"""

from __future__ import annotations

import hashlib
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = r"D:\CASMI竞赛"
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import baseline_v1 as bv                                             # noqa: E402
from casmi.core import formula_neutral_mass, neutral_mass_from_precursor  # noqa: E402
from casmi.spectra import clean_peaks                                # noqa: E402


def sig(mz):
    return hashlib.md5(np.round(np.asarray(mz, np.float64), 4).tobytes()).hexdigest()


class S:
    __slots__ = ("mz", "intensity", "neutral_mass")

    def __init__(self, mz, i, m):
        self.mz, self.intensity, self.neutral_mass = mz, i, m


def main():
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 900.0
    pool_path = sys.argv[2] if len(sys.argv) > 2 else bv.STRUCTS
    t_start = time.time()

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
                kk = k if isinstance(k, str) and k else None
                if kk:
                    sig2key.setdefault(h, kk)
    print(f"resolved {len(sig2key):,}/{len(tset):,} test spectra to train structures",
          flush=True)

    store = bv.StructureStore(pool_path)
    si = bv.SpectralIndex(bv.LIB_INDEX_DIR)

    mids = [str(m) for m in test["molecule_id"].drop_duplicates()]
    n = 0
    in_pool = 0
    hit25 = 0
    ranks = []
    for mid in mids:
        if time.time() - t_start > budget:
            print(f"\n=== time budget {budget:.0f}s reached after {n} molecules ===")
            break
        sub = test[test["molecule_id"].astype(str) == mid]
        truth = {sig2key[sig(mz)] for mz in sub["ms2_mzs"] if sig(mz) in sig2key}
        if not truth:
            continue
        sp = []
        for r in sub.itertuples(index=False):
            mz, it = clean_peaks(r.ms2_mzs, r.ms2_normalized_intensities)
            if mz.size < 3:
                continue
            f = formula_neutral_mass(getattr(r, "molecular_formula", None))
            m = f if f is not None else neutral_mass_from_precursor(
                r.precursor_mz, str(r.adduct))
            sp.append(S(mz, it, m))
        ranked, direct, analog, cand = bv.score_molecule(sp, store, si)
        keys = [r[0] for r in ranked[:25]]
        n += 1
        ip = bool(truth & set(cand))
        r = next((i for i, k in enumerate(keys, 1) if k in truth), 0)
        in_pool += ip
        hit25 += bool(r)
        ranks.append(r)
        print(f"  [{n:3d}] {mid}: cand={len(cand):7,} truth_in_pool={ip} rank={r}",
              flush=True)

    if n:
        ranks = np.asarray(ranks)
        print()
        print(f"molecules evaluated      : {n}")
        print(f"truth in candidate pool  : {in_pool}/{n} = {in_pool/n*100:.1f}%")
        print(f"truth within top-25      : {hit25}/{n} = {hit25/n*100:.1f}%")
        print(f"MRR@25 vs train labels   : "
              f"{np.mean(np.where(ranks>0, 1.0/np.maximum(ranks,1), 0.0)):.4f}")


if __name__ == "__main__":
    main()
