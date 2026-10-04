"""
Parallel diagnosis + library ablation on the visible test set.

Two questions, in order of importance:

Q1 (correctness). For each visible test spectrum, train.parquet contains that
exact spectrum and therefore names the structure it belongs to. Does the
pipeline recover that structure? If not, the ranking code is broken and no
amount of tuning helps.

Q2 (which library). The test set is 100% timsTOF. Of my 2.37M library spectra
only 1,179 (250 compounds) are timsTOF; the rest are Orbitrap/QTOF/etc. enveda-180
alone contributes 1.14M timsTOF spectra from 182,893 compounds. Excluding it on
the grounds that it is "synthetic chemistry" may have thrown away the only
large instrument-matched reference set. This measures each library variant.

Runs molecules in parallel across processes: sequential scoring of 400 molecules
takes over 10 minutes per library variant, which is too slow to iterate on.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from collections import defaultdict

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
STRUCTS = os.path.join(ROOT, "data", "processed", "structures.parquet")
LIB_FULL = os.path.join(ROOT, "data", "processed", "library_spectra_sorted")
LIB_NP = os.path.join(ROOT, "artifacts", "kaggle_assets_np3", "library_spectra_sorted")

_STORE = None
_SI = None


def _sig(mz):
    return hashlib.md5(np.round(np.asarray(mz, np.float64), 4).tobytes()).hexdigest()


def _init(libdir):
    global _STORE, _SI
    import baseline_v1 as bv
    _STORE = bv.StructureStore(STRUCTS)
    _SI = bv.SpectralIndex(libdir)


class _S:
    __slots__ = ("mz", "intensity", "neutral_mass")

    def __init__(self, mz, intensity, neutral_mass):
        self.mz = mz
        self.intensity = intensity
        self.neutral_mass = neutral_mass


def _work(mid, spectra, truths, kw):
    import baseline_v1 as bv
    sp = [_S(*t) for t in spectra]
    ranked, direct, analog, cand = bv.score_molecule(sp, _STORE, _SI, **kw)
    keys = [r[0] for r in ranked[:25]]
    return mid, keys, sorted(truths)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--libs", nargs="*", default=["np", "full"])
    ap.add_argument("--analog-dm", type=float, default=2.0)
    ap.add_argument("--analog-top", type=int, default=60)
    ap.add_argument("--procs", type=int, default=10)
    a = ap.parse_args()

    test = pd.read_parquet(TEST)
    print(f"visible test: {len(test)} spectra, {test['molecule_id'].nunique()} molecules",
          flush=True)

    # --- Q1 prerequisite: which train structure does each test spectrum belong to
    tset = {_sig(mz) for mz in test["ms2_mzs"]}
    print(f"unique test spectrum fingerprints: {len(tset):,}", flush=True)
    pf = pq.ParquetFile(TRAIN)
    sig_to_key: dict[str, str] = {}
    for b in pf.iter_batches(batch_size=300_000,
                             columns=["ms2_mzs", "inchikey14", "normalized_smiles"]):
        df = b.to_pandas()
        for mz, k, smi in zip(df["ms2_mzs"], df["inchikey14"], df["normalized_smiles"]):
            h = _sig(mz)
            if h in tset:
                kk = k if isinstance(k, str) and k else smiles_to_inchikey14(smi)
                if kk:
                    sig_to_key.setdefault(h, kk)
    print(f"resolved {len(sig_to_key):,}/{len(tset):,} test spectra to a train structure",
          flush=True)

    mol_true: dict[str, set] = defaultdict(set)
    for mid, mz in zip(test["molecule_id"], test["ms2_mzs"]):
        k = sig_to_key.get(_sig(mz))
        if k:
            mol_true[str(mid)].add(k)
    n_with = sum(1 for m in test["molecule_id"].drop_duplicates() if mol_true.get(str(m)))
    print(f"molecules with >=1 train-associated structure: {n_with}/"
          f"{test['molecule_id'].nunique()}", flush=True)

    # spectra per molecule
    mols = defaultdict(list)
    for row in test.itertuples(index=False):
        mz, inten = clean_peaks(row.ms2_mzs, row.ms2_normalized_intensities)
        if mz.size < 3:
            continue
        mols[str(row.molecule_id)].append(
            (mz, inten, neutral_mass_from_precursor(row.precursor_mz, str(row.adduct))))

    import multiprocessing as mp
    grid = [("np", LIB_NP), ("full", LIB_FULL)]
    grid = [g for g in grid if g[0] in a.libs]

    for tag, libdir in grid:
        tasks = [(mid, mols[mid], mol_true.get(mid, set()))
                 for mid in mols if mol_true.get(mid)]
        kw = dict(analog_dm=a.analog_dm, analog_candidate_dm=2.0,
                  analog_top_hits=a.analog_top)
        print(f"\n=== library={tag} ({libdir})  molecules={len(tasks)} "
              f"analog_dm={a.analog_dm} ===", flush=True)
        ranks, cand_has, top1 = [], 0, 0
        with mp.Pool(a.procs, initializer=_init, initargs=(libdir,)) as pool:
            for i, (mid, keys, truth) in enumerate(
                    pool.starmap(_work, [(m, s, t, kw) for m, s, t in tasks]), 1):
                ts = set(truth)
                if set(keys) & ts:
                    cand_has += 1
                if keys and keys[0] in ts:
                    top1 += 1
                r = next((j for j, k in enumerate(keys, 1) if k in ts), 0)
                ranks.append(r)
                if i % 50 == 0:
                    print(f"   {i}/{len(tasks)}", flush=True)
        ranks = np.asarray(ranks)
        n = len(ranks)
        mrr = float(np.mean(np.where(ranks > 0, 1.0 / np.maximum(ranks, 1), 0.0)))
        print(f"  candidate pool contains truth : {cand_has}/{n} = {cand_has/n*100:.1f}%")
        print(f"  truth ranked #1               : {top1}/{n} = {top1/n*100:.1f}%")
        print(f"  truth within top-25           : {int((ranks>0).sum())}/{n} = "
              f"{(ranks>0).mean()*100:.1f}%")
        print(f"  MRR@25 vs train labels        : {mrr:.4f}", flush=True)


if __name__ == "__main__":
    main()
