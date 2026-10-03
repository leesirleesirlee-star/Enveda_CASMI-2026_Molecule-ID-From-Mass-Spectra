"""Profile the baseline stages to find the real bottleneck, with timestamps."""
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

T0 = time.time()


def mark(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)


from baseline_v1 import (StructureStore, SpectralIndex, STRUCTS,   # noqa: E402
                         LIB_INDEX_DIR, TRAIN, score_molecule)
from casmi.validation import build_folds, V_A                          # noqa: E402

mark("imports done")

store = StructureStore(STRUCTS)
mark(f"StructureStore loaded ({len(store):,})")

spec_idx = SpectralIndex(LIB_INDEX_DIR)
mark("SpectralIndex loaded")

cols = ["ms2_mzs", "ms2_normalized_intensities", "precursor_mz", "adduct",
        "normalized_smiles", "inchikey14", "ingest_lib", "molecular_formula"]
pf = pq.ParquetFile(TRAIN)
frames = [b.to_pandas() for b in pf.iter_batches(batch_size=250_000, columns=cols)]
df = pd.concat(frames, ignore_index=True)
mark(f"train frame loaded {len(df):,}")

folds = build_folds(df)
mark(f"folds built: {[ (k,len(v)) for k,v in folds.items() ]}")

fold = folds[V_A]
mids = list(fold.queries)[:3]
for mid in mids:
    mol = fold.queries[mid]
    m = mol.consensus_neutral_mass()
    t = time.time()
    # how big is the candidate set and the library scan set?
    cand = set()
    for s in mol.spectra:
        for ppm in (15.0, 40.0):
            for i in store.window(s.neutral_mass, ppm):
                cand.add(store.keys[i])
        for i in store.between(s.neutral_mass - 2.0, s.neutral_mass + 2.0):
            cand.add(store.keys[i])
    n_cand = len(cand)
    lib_keys = set(cand)
    for s in mol.spectra:
        for i in store.between(s.neutral_mass - 2.0, s.neutral_mass + 2.0):
            lib_keys.add(store.keys[i])
    n_lib = len(lib_keys)
    mzs, ints, ks, ts = spec_idx.by_keys(lib_keys)
    t_pre = time.time() - t
    mark(f"  mol {mid}: {len(mol.spectra)} spectra, M={m:.2f}, "
         f"cand={n_cand:,}, lib_keys={n_lib:,}, lib_spectra={len(mzs):,}, "
         f"prep={t_pre:.2f}s")

    t = time.time()
    ranked, d, a, c = score_molecule(mol.spectra, store, spec_idx)
    t_score = time.time() - t
    mark(f"    score_molecule: {t_score:.1f}s  -> top score "
         f"{ranked[0][1]:.3f}" if ranked else "    no candidates")
