"""
Fast head-to-head: v1 vs v2 on a small molecule count, with the fragmentation
channel switchable.

The v1 path is slow because it builds candidates from a 730k pool by mass alone
(361 candidates x thousands of spectra). v2's whole point is that the formula
filter shrinks that to ~36, so v2 should also be much faster -- this measures both
quality and speed, which is what we need to decide the architecture today.
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

import baseline_v1 as bv                                              # noqa: E402
from casmi.core import formula_neutral_mass, neutral_mass_from_precursor  # noqa: E402
from casmi.pipeline_v2 import Query, _rank01, score_query_v2           # noqa: E402
from casmi.spectra import clean_peaks                                  # noqa: E402


def sig(mz):
    return hashlib.md5(np.round(np.asarray(mz, np.float64), 4).tobytes()).hexdigest()


class S:
    __slots__ = ("mz", "intensity", "neutral_mass")

    def __init__(self, mz, i, m):
        self.mz, self.intensity, self.neutral_mass = mz, i, m


def main():
    n_mol = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    use_frag = (sys.argv[2] != "0") if len(sys.argv) > 2 else True
    t0 = time.time()

    test = pd.read_parquet(os.path.join(ROOT, "data", "raw", "test.parquet"))
    tset = {sig(mz) for mz in test["ms2_mzs"]}
    pf = pq.ParquetFile(os.path.join(ROOT, "data", "raw", "train.parquet"))
    sig2key = {}
    for b in pf.iter_batches(batch_size=300_000,
                             columns=["ms2_mzs", "inchikey14"]):
        df = b.to_pandas()
        for mz, k in zip(df["ms2_mzs"], df["inchikey14"]):
            h = sig(mz)
            if h in tset and isinstance(k, str) and k:
                sig2key.setdefault(h, k)

    store = bv.StructureStore(os.path.join(ROOT, "data", "processed",
                                           "candidate_pool.parquet"))
    si = bv.SpectralIndex(os.path.join(ROOT, "data", "processed",
                                       "library_spectra_sorted"))

    mids = [str(m) for m in test["molecule_id"].drop_duplicates()][:n_mol]
    rows = []
    for mid in mids:
        sub = test[test["molecule_id"].astype(str) == mid]
        truth = {sig2key[sig(mz)] for mz in sub["ms2_mzs"] if sig(mz) in sig2key}
        if not truth:
            continue
        qf = None
        for t in truth:
            i = store._key_to_idx.get(t)
            if i is not None and isinstance(store.formula[i], str):
                qf = store.formula[i]
                break
        sp, qobj, positive = [], Query(mid), True
        for r in sub.itertuples(index=False):
            mz, it = clean_peaks(r.ms2_mzs, r.ms2_normalized_intensities)
            if mz.size < 3:
                continue
            f = formula_neutral_mass(getattr(r, "molecular_formula", None))
            m = f if f is not None else neutral_mass_from_precursor(
                r.precursor_mz, str(r.adduct))
            sp.append(S(mz, it, m))
            qobj.mz.append(mz)
            qobj.intensity.append(it)
            qobj.mass.append(m)
            if str(getattr(r, "ionization_mode", "")).lower().startswith("neg"):
                positive = False
        if not sp:
            continue
        qobj.positive = positive

        t1 = time.time()
        ranked1, _d, _a, c1 = bv.score_molecule(sp, store, si, analog_dm=2.0)
        dt1 = time.time() - t1
        r1 = next((i for i, x in enumerate([q[0] for q in ranked1[:25]], 1)
                   if x in truth), 0)

        t2 = time.time()
        ranked2, _ix, diag = score_query_v2(qobj, store, si, formula=qf,
                                           use_frag=use_frag)
        dt2 = time.time() - t2
        k2 = [q[0] for q in ranked2[:25]]
        r2 = next((i for i, x in enumerate(k2, 1) if x in truth), 0)

        rows.append((mid, qf, r1, r2, diag["n_candidates"], dt1, dt2))
        print(f"  {mid} f={qf:12s} v1_rank={r1:2d} v2_rank={r2:2d} "
              f"v2_cand={diag['n_candidates']:4d} t1={dt1:6.1f}s t2={dt2:6.1f}s",
              flush=True)

    if not rows:
        print("no molecules evaluated")
        return
    r1 = np.array([x[2] for x in rows])
    r2 = np.array([x[3] for x in rows])
    mrr = lambda r: float(np.mean(np.where(r > 0, 1.0 / np.maximum(r, 1), 0.0)))
    print()
    print(f"=== {len(rows)} molecules (frag={'on' if use_frag else 'off'}) ===")
    print(f"  v1: hit@25={int((r1>0).sum())}/{len(r1)}  MRR@25={mrr(r1):.4f}  "
          f"median {np.median([x[5] for x in rows]):.1f}s/mol")
    print(f"  v2: hit@25={int((r2>0).sum())}/{len(r2)}  MRR@25={mrr(r2):.4f}  "
          f"median {np.median([x[6] for x in rows]):.1f}s/mol")
    print(f"  v2 median candidates: {int(np.median([x[4] for x in rows]))}")


if __name__ == "__main__":
    main()
