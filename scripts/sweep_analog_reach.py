"""
Sweep the analog reach on the strict fold.

Diagnosis says a 2 Da scan cannot reach most true analogues (only 17% of V_A-hard
queries have their most similar library compound within 2 Da; 63% are 10-50 Da
away). This measures whether widening the scan actually helps MRR, rather than
assuming it must.

analog_dm      = how far the library-spectrum scan reaches (the analog channel)
analog_candidate_dm = how far structures enter the candidate pool
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

ROOT = r"D:\CASMI竞赛"
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from baseline_v1 import (StructureStore, SpectralIndex, score_molecule,   # noqa: E402
                         STRUCTS, LIB_INDEX_DIR, TRAIN)
from casmi.folds_fast import build_folds_fast, exclusion_set              # noqa: E402
from casmi.core import mrr25_breakdown                                    # noqa: E402


def evaluate(f, store, si, mids, **kw):
    preds = {}
    for mid in mids:
        ranked, direct, analog, cand = score_molecule(
            f.queries[mid].spectra, store, si, **kw)
        preds[mid] = [store.smiles[cand[r[0]]] for r in ranked[:25]]
    truths = {m: f.truths[m] for m in mids}
    return mrr25_breakdown(preds, truths)


def main():
    store = StructureStore(STRUCTS)
    si = SpectralIndex(LIB_INDEX_DIR)
    folds = build_folds_fast(TRAIN, vb_limit=0, vc_limit=0, verbose=False)
    f = folds["V_A_hard"]
    excl = exclusion_set(f, store.keys)
    mids = list(f.queries)[:60]
    print(f"V_A_hard strict fold, {len(mids)} queries, {len(excl)} excluded")
    print(f"baseline constants: ANALOG_DM={2.0}, top_hits=60\n")

    grid = [
        dict(analog_dm=2.0,  analog_candidate_dm=2.0,  analog_top_hits=60),
        dict(analog_dm=20.0, analog_candidate_dm=2.0,  analog_top_hits=60),
        dict(analog_dm=50.0, analog_candidate_dm=2.0,  analog_top_hits=60),
        dict(analog_dm=50.0, analog_candidate_dm=2.0,  analog_top_hits=400),
        dict(analog_dm=200.0, analog_candidate_dm=2.0, analog_top_hits=400),
    ]
    print(f"{'reach':>6} {'cand_dm':>8} {'top':>5} | {'MRR@25':>8} {'top1':>6} "
          f"{'hit@25':>7} | {'sec':>6}")
    for g in grid:
        t = time.time()
        r = evaluate(f, store, si, mids, exclude_keys=excl, **g)
        el = time.time() - t
        print(f"{g['analog_dm']:>6.0f} {g['analog_candidate_dm']:>8.1f} "
              f"{g['analog_top_hits']:>5d} | {r['mrr25']:>8.4f} {r['top1']:>6.3f} "
              f"{r['hit@25']:>7.3f} | {el:>6.0f}", flush=True)


if __name__ == "__main__":
    main()
