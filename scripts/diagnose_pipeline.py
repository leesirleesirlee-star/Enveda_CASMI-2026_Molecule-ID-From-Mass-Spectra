"""
Systematic diagnosis: why does the pipeline score ~0.13 when the public
reproducible baseline is ~0.33?

The one ground truth we actually have locally is the VISIBLE test set, whose
spectra are provably verbatim copies of train rows (all 1,213 from enveda-180).
That makes it a leak-aware probe, not a leaderboard proxy -- but it answers a
much more basic question than MRR:

    Does the pipeline recover, for each test spectrum, the structure that
    train.parquet itself associates with that exact spectrum?

If it fails THAT, the pipeline is broken in a way no amount of tuning fixes. If
it succeeds, the gap to 0.33 is about the hidden set's composition (molecules
absent from train), not about the ranking code.

It also reports, at each stage, the fraction of molecules where the true
structure was even reachable -- recall vs ranking, again.
"""

from __future__ import annotations

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

from baseline_v1 import (StructureStore, SpectralIndex, score_molecule,  # noqa: E402
                         STRUCTS, LIB_INDEX_DIR, TRAIN, TEST)
from casmi.core import smiles_to_inchikey14, neutral_mass_from_precursor  # noqa: E402
from casmi.spectra import clean_peaks                                    # noqa: E402


def sig(mz, it):
    return hashlib.md5(np.round(np.asarray(mz, np.float64), 4).tobytes()).hexdigest()


def main():
    test = pd.read_parquet(TEST)
    print(f"visible test: {len(test)} spectra, {test['molecule_id'].nunique()} molecules")

    # 1. Which train structure does each visible test spectrum belong to?
    test_sig = {sig(mz, it): sid for mz, it, sid in
                zip(test["ms2_mzs"], test["ms2_normalized_intensities"], test["spectrum_id"])}
    print(f"unique test spectrum fingerprints: {len(test_sig):,}")

    pf = pq.ParquetFile(TRAIN)
    sig_to_key: dict[str, str] = {}
    null_key = 0
    for b in pf.iter_batches(batch_size=200_000,
                             columns=["ms2_mzs", "ms2_normalized_intensities",
                                      "inchikey14", "normalized_smiles"]):
        df = b.to_pandas()
        for mz, it, k, smi in zip(df["ms2_mzs"], df["ms2_normalized_intensities"],
                                  df["inchikey14"], df["normalized_smiles"]):
            h = sig(mz, it)
            if h not in test_sig:
                continue
            kk = k if isinstance(k, str) and k else smiles_to_inchikey14(smi)
            if kk is None:
                null_key += 1
                continue
            sig_to_key.setdefault(h, kk)
    print(f"test spectra matched to a train structure key: {len(sig_to_key):,}/{len(test_sig):,}"
          f"  (skipped {null_key} with no derivable key)")

    # 2. per molecule: the set of structures train associates with its spectra
    mol_true: dict[str, set[str]] = defaultdict(set)
    for mid, mz, it in zip(test["molecule_id"], test["ms2_mzs"],
                           test["ms2_normalized_intensities"]):
        k = sig_to_key.get(sig(mz, it))
        if k:
            mol_true[str(mid)].add(k)
    n_with = sum(1 for m in test["molecule_id"].unique() if mol_true.get(str(m)))
    print(f"molecules with >=1 train-associated structure: {n_with}/"
          f"{test['molecule_id'].nunique()}")

    # 3. run the real pipeline on the visible test set
    store = StructureStore(STRUCTS)
    si = SpectralIndex(LIB_INDEX_DIR)
    mols = defaultdict(list)
    for row in test.itertuples(index=False):
        ad = str(row.adduct)
        mz, inten = clean_peaks(row.ms2_mzs, row.ms2_normalized_intensities)
        if mz.size < 3:
            continue
        mols[str(row.molecule_id)].append({
            "mz": mz, "intensity": inten,
            "neutral_mass": neutral_mass_from_precursor(row.precursor_mz, ad)})

    # simple spectrum holder compatible with score_molecule
    class S:
        __slots__ = ("mz", "intensity", "neutral_mass")

        def __init__(self, d):
            self.mz, self.intensity, self.neutral_mass = d["mz"], d["intensity"], d["neutral_mass"]

    mids = [str(m) for m in test["molecule_id"].drop_duplicates()]
    n_eval = 0
    top1_hit = 0
    top25_hit = 0
    cand_has = 0
    ranks = []
    for mid in mids:
        truth = mol_true.get(mid)
        if not truth:
            continue
        n_eval += 1
        sp = [S(d) for d in mols[mid]]
        ranked, direct, analog, cand = score_molecule(sp, store, si)
        keys = [r[0] for r in ranked[:25]]
        if set(keys) & truth:
            cand_has += 1
        if keys and keys[0] in truth:
            top1_hit += 1
        r = 0
        for i, k in enumerate(keys, 1):
            if k in truth:
                r = i
                break
        ranks.append(r)
        if r:
            top25_hit += 1

    ranks = np.asarray(ranks)
    print()
    print(f"=== pipeline vs train-associated structure ({n_eval} molecules) ===")
    print(f"  true structure in candidate pool : {cand_has}/{n_eval} = {cand_has/n_eval*100:.1f}%")
    print(f"  true structure ranked #1         : {top1_hit}/{n_eval} = {top1_hit/n_eval*100:.1f}%")
    print(f"  true structure within top-25     : {top25_hit}/{n_eval} = {top25_hit/n_eval*100:.1f}%")
    mrr = float(np.mean(np.where(ranks > 0, 1.0/np.maximum(ranks, 1), 0.0)))
    print(f"  MRR@25 (against train labels)    : {mrr:.4f}")


if __name__ == "__main__":
    main()
