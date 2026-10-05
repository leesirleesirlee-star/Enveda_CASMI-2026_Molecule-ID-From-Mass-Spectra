"""
Build the backtest query set for Method A: a held-out fold, in the *competition's*
test.parquet format, plus the truth table to score against.

Why the format must match exactly: the backtest kernel is the V44 notebook with
COMP pointed at this file, so any column-name or -order difference fails at load
time, deep inside the pipeline, after a GPU run has already started.

Leak-free-ness is NOT arranged here - it is arranged at query time by the engine's
own simulation hooks, which the engine documents itself:
    Engine.run(spectra, target, exclude=..., exclude_sid=..., exclude_lib=..., drop_pid=...)
      exclude        boolean mask over library spectra to hide
      exclude_sid/lib  representative to hide for the analog channel
      drop_pid       remove this pool entry (class-3 simulation)
This script only produces the queries and the answer key.

Usage:
  python scripts/build_backtest_queries.py --fold 0 [--limit 2000] [--regime other]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = r"D:\CASMI竞赛"
TRAIN = os.path.join(ROOT, "data", "raw", "train.parquet")
TEST = os.path.join(ROOT, "data", "raw", "test.parquet")
FOLDS = os.path.join(ROOT, "data", "processed", "folds.parquet")
OUTDIR = os.path.join(ROOT, "data", "processed")

# the competition's test.parquet column order, read from the real file
TEST_COLS = ["molecule_id", "spectrum_id", "ms2_mzs", "ms2_normalized_intensities",
             "base_peak_intensity", "adduct", "ionization_mode", "instrument_type",
             "precursor_mz", "collision_energy_orig", "collision_energy_ev",
             "collision_energy_orig_units"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--regime", default="other")
    ap.add_argument("--limit", type=int, default=2000,
                    help="cap the number of query molecules (the engine is not cheap)")
    ap.add_argument("--seed", type=int, default=20261005)
    a = ap.parse_args()

    # the real test file is the schema authority - do not hardcode types
    ref_schema = pq.ParquetFile(TEST).schema_arrow
    assert [f.name for f in ref_schema] == TEST_COLS, [f.name for f in ref_schema]
    print("test.parquet schema confirmed:", len(TEST_COLS), "columns")

    folds = pd.read_parquet(FOLDS)
    sel = folds[(folds.fold == a.fold) & (folds.regime == a.regime)]
    keys = set(sel.inchikey14.tolist())
    print(f"fold {a.fold} regime {a.regime}: {len(keys):,} held-out structures")

    rng = np.random.default_rng(a.seed)
    if len(keys) > a.limit:
        keys = set(rng.choice(sorted(keys), size=a.limit, replace=False).tolist())
        print(f"subsampled to {len(keys):,} query molecules")

    # pull every spectrum belonging to those structures
    t0 = time.time()
    pf = pq.ParquetFile(TRAIN)
    cols = ["inchikey14", "ms2_mzs", "ms2_normalized_intensities", "base_peak_intensity",
            "adduct", "ionization_mode", "instrument_type", "precursor_mz",
            "collision_energy_orig", "collision_energy_ev", "collision_energy_orig_units"]
    parts = []
    for b in pf.iter_batches(batch_size=250_000, columns=cols):
        d = b.to_pandas()
        d = d[d.inchikey14.isin(keys)]
        if len(d):
            parts.append(d)
    q = pd.concat(parts, ignore_index=True)
    print(f"collected {len(q):,} spectra for {q.inchikey14.nunique():,} molecules "
          f"in {time.time()-t0:.0f}s")

    # assertions that protect the downstream run
    missing = keys - set(q.inchikey14.unique())
    assert not missing, f"{len(missing)} held-out structures have no spectrum"
    q = q[q.ms2_mzs.map(len) > 0].copy()
    assert len(q), "no spectra with peaks"
    assert q.inchikey14.nunique() == len(keys), "a held-out structure lost all its spectra"

    q = q.rename(columns={"inchikey14": "molecule_id"})
    q["spectrum_id"] = q.molecule_id + "_" + q.groupby("molecule_id").cumcount().astype(str)
    q = q[TEST_COLS]

    # cast to the reference schema so column *types* match too, not just names
    tbl = pa.Table.from_pandas(q, preserve_index=False).cast(ref_schema)
    out_q = os.path.join(OUTDIR, f"backtest_fold{a.fold}_{a.regime}.parquet")
    pq.write_table(tbl, out_q)
    print(f"wrote {out_q}  rows={tbl.num_rows}  molecules={q.molecule_id.nunique()}")

    truth = (q.groupby("molecule_id").size().rename("n_spectra").reset_index()
             .merge(sel[["inchikey14"]], left_on="molecule_id", right_on="inchikey14")
             [["molecule_id", "inchikey14", "n_spectra"]])
    assert truth.molecule_id.is_unique and truth.inchikey14.notna().all()
    out_t = os.path.join(OUTDIR, f"backtest_fold{a.fold}_{a.regime}_truth.parquet")
    truth.to_parquet(out_t, index=False)
    print(f"wrote {out_t}  molecules={len(truth)}")
    print(f"spectra per molecule: median {int(truth.n_spectra.median())}, "
          f"min {truth.n_spectra.min()}, max {truth.n_spectra.max()}")
    print("PASS: query set is schema-identical to test.parquet and every query has a truth key")


if __name__ == "__main__":
    sys.exit(main())
