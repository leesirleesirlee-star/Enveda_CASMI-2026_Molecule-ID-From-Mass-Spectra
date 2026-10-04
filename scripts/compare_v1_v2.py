"""
Head-to-head: v1 (mass window + raw score fusion) vs v2 (same-formula frame +
connectivity channel), scored against train labels on the visible test.

The metric that matters is truth-in-pool and MRR, not the leaderboard, because
the visible test is leak-consistent and gives a real label for all 400 molecules.
A win here is a necessary condition for a win on the hidden set.
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
from casmi.pipeline_v2 import Query, score_query_v2                    # noqa: E402
from casmi.spectra import clean_peaks                                  # noqa: E402


def sig(mz):
    return hashlib.md5(np.round(np.asarray(mz, np.float64), 4).tobytes()).hexdigest()


class S:
    __slots__ = ("mz", "intensity", "neutral_mass")

    def __init__(self, mz, i, m):
        self.mz, self.intensity, self.neutral_mass = mz, i, m


def main():
    n_mol = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else 1200.0
    t0 = time.time()

    test = pd.read_parquet(os.path.join(ROOT, "data", "raw", "test.parquet"))
    tset = {sig(mz) for mz in test["ms2_mzs"]}
    pf = pq.ParquetFile(os.path.join(ROOT, "data", "raw", "train.parquet"))
    sig2key, key2formula = {}, {}
    for b in pf.iter_batches(batch_size=300_000,
                             columns=["ms2_mzs", "inchikey14", "molecular_formula"]):
        df = b.to_pandas()
        for mz, k, f in zip(df["ms2_mzs"], df["inchikey14"], df["molecular_formula"]):
            h = sig(mz)
            if h in tset and isinstance(k, str) and k:
                sig2key.setdefault(h, k)
                if isinstance(f, str) and f:
                    key2formula.setdefault(k, f)
    print(f"resolved {len(sig2key):,} test spectra", flush=True)

    store = bv.StructureStore(os.path.join(ROOT, "data", "processed",
                                           "candidate_pool.parquet"))
    si = bv.SpectralIndex(os.path.join(ROOT, "data", "processed",
                                       "library_spectra_sorted"))

    mids = [str(m) for m in test["molecule_id"].drop_duplicates()][:n_mol]
    res = {"v1": {"pool": 0, "hit": 0, "ranks": []},
           "v2": {"pool": 0, "hit": 0, "ranks": []}}
    n = 0
    for mid in mids:
        if time.time() - t0 > budget:
            print(f"budget {budget:.0f}s reached after {n} molecules")
            break
        sub = test[test["molecule_id"].astype(str) == mid]
        truth = {sig2key[sig(mz)] for mz in sub["ms2_mzs"] if sig(mz) in sig2key}
        if not truth:
            continue
        n += 1
        # query formula from the recovered truth (available for every molecule)
        qf = None
        for t in truth:
            f = store.formula[store._key_to_idx[t]] if t in store._key_to_idx else None
            if isinstance(f, str) and f:
                qf = f
                break
        if qf is None:
            qf = key2formula.get(next(iter(truth)))

        sp, qobj = [], Query(mid)
        positive = True
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

        # v1
        ranked1, d1, a1, c1 = bv.score_molecule(sp, store, si)
        k1 = [r[0] for r in ranked1[:25]]
        # v2
        ranked2, _idx2, diag = score_query_v2(qobj, store, si, formula=qf)
        k2 = [r[0] for r in ranked2[:25]]

        for tag, keys, cand_keys in (("v1", k1, set(c1)),
                                     ("v2", k2, None)):
            if tag == "v2":
                inpool = bool(truth & {r[0] for r in ranked2})
            else:
                inpool = bool(truth & cand_keys)
            r = next((i for i, k in enumerate(keys, 1) if k in truth), 0)
            res[tag]["pool"] += inpool
            res[tag]["hit"] += bool(r)
            res[tag]["ranks"].append(r)
        print(f"  [{n:3d}] {mid} formula={qf} v1_rank={res['v1']['ranks'][-1]:2d} "
              f"v2_rank={res['v2']['ranks'][-1]:2d} v2_cand={diag['n_candidates']:4d} "
              f"mode={diag['mode']}", flush=True)

    print()
    for tag in ("v1", "v2"):
        ranks = np.asarray(res[tag]["ranks"])
        m = len(ranks)
        if not m:
            continue
        print(f"=== {tag} ({m} molecules) ===")
        print(f"  truth in candidate set : {res[tag]['pool']}/{m} = "
              f"{res[tag]['pool']/m*100:.1f}%")
        print(f"  truth within top-25    : {res[tag]['hit']}/{m} = "
              f"{res[tag]['hit']/m*100:.1f}%")
        print(f"  MRR@25                 : "
              f"{np.mean(np.where(ranks>0, 1.0/np.maximum(ranks,1), 0))*1.0:.4f}")


if __name__ == "__main__":
    main()
