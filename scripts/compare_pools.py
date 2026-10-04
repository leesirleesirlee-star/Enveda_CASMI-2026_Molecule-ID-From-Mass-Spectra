"""
Does the merged NP pool improve truth recovery on the visible test?

Compares scoring with the train-only structures.parquet against the merged
candidate_pool.parquet, on the visible test molecules whose truth we recover from
the verbatim train rows. Parallelised because sequential scoring is ~4 s/molecule.
"""

from __future__ import annotations

import argparse
import hashlib
import multiprocessing as mp
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = r"D:\CASMI竞赛"
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from casmi.core import neutral_mass_from_precursor, smiles_to_inchikey14  # noqa: E402
from casmi.spectra import clean_peaks                                     # noqa: E402

TEST = os.path.join(ROOT, "data", "raw", "test.parquet")
TRAIN = os.path.join(ROOT, "data", "raw", "train.parquet")
POOL_TRAIN = os.path.join(ROOT, "data", "processed", "structures.parquet")
POOL_MERGED = os.path.join(ROOT, "data", "processed", "candidate_pool.parquet")
LIB = os.path.join(ROOT, "data", "processed", "library_spectra_sorted")

_STORE = None
_SI = None


def _sig(mz):
    return hashlib.md5(np.round(np.asarray(mz, np.float64), 4).tobytes()).hexdigest()


class _S:
    __slots__ = ("mz", "intensity", "neutral_mass")

    def __init__(self, mz, i, m):
        self.mz, self.intensity, self.neutral_mass = mz, i, m


def _init(pool_path):
    global _STORE, _SI
    import baseline_v1 as bv
    # StructureStore expects a 'fp' column; candidate_pool.parquet provides it
    _STORE = bv.StructureStore(pool_path)
    _SI = bv.SpectralIndex(LIB)


def _work(payload):
    mid, spectra, truths = payload
    import baseline_v1 as bv
    sp = [_S(*t) for t in spectra]
    ranked, direct, analog, cand = bv.score_molecule(sp, _STORE, _SI)
    keys = [r[0] for r in ranked[:25]]
    ts = set(truths)
    rank = next((i for i, k in enumerate(keys, 1) if k in ts), 0)
    return mid, len(cand), bool(ts & set(cand)), rank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--molecules", type=int, default=150)
    ap.add_argument("--procs", type=int, default=10)
    a = ap.parse_args()

    test = pd.read_parquet(TEST)
    tset = {_sig(mz) for mz in test["ms2_mzs"]}
    pf = pq.ParquetFile(TRAIN)
    sig2key = {}
    for b in pf.iter_batches(batch_size=300_000,
                             columns=["ms2_mzs", "inchikey14", "normalized_smiles"]):
        df = b.to_pandas()
        for mz, k, smi in zip(df["ms2_mzs"], df["inchikey14"], df["normalized_smiles"]):
            h = _sig(mz)
            if h in tset:
                kk = k if isinstance(k, str) and k else smiles_to_inchikey14(smi)
                if kk:
                    sig2key.setdefault(h, kk)
    print(f"resolved {len(sig2key):,}/{len(tset):,} test spectra to train structures",
          flush=True)

    mols, truths = {}, {}
    for mid, mz, in zip(test["molecule_id"], test["ms2_mzs"]):
        k = sig2key.get(_sig(mz))
        if k:
            truths.setdefault(str(mid), set()).add(k)
    for row in test.itertuples(index=False):
        mid = str(row.molecule_id)
        mz, it = clean_peaks(row.ms2_mzs, row.ms2_normalized_intensities)
        if mz.size < 3:
            continue
        mols.setdefault(mid, []).append(
            (mz, it, neutral_mass_from_precursor(row.precursor_mz, str(row.adduct))))

    mids = [m for m in mols if truths.get(m)][:a.molecules]
    payload = [(m, mols[m], sorted(truths[m])) for m in mids]
    print(f"evaluating {len(payload)} molecules\n", flush=True)

    for tag, pool in [("train-only", POOL_TRAIN), ("merged NP", POOL_MERGED)]:
        if not os.path.exists(pool):
            print(f"skip {tag}: {pool} missing")
            continue
        with mp.Pool(a.procs, initializer=_init, initargs=(pool,)) as p:
            res = p.map(_work, payload)
        cand_n = np.array([r[1] for r in res])
        in_pool = np.array([r[2] for r in res])
        ranks = np.array([r[3] for r in res])
        mrr = float(np.mean(np.where(ranks > 0, 1.0 / np.maximum(ranks, 1), 0.0)))
        print(f"=== pool = {tag} ===")
        print(f"  median candidate pool size : {int(np.median(cand_n)):,}")
        print(f"  truth in candidate pool    : {in_pool.sum()}/{len(res)} = "
              f"{in_pool.mean()*100:.1f}%")
        print(f"  truth within top-25        : {(ranks>0).sum()}/{len(res)} = "
              f"{(ranks>0).mean()*100:.1f}%")
        print(f"  MRR@25 vs train labels     : {mrr:.4f}\n", flush=True)


if __name__ == "__main__":
    main()
