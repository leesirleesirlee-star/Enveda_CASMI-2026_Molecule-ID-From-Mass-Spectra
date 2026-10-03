"""
Build a compact Kaggle asset bundle for an offline submission.

Rationale: the full library bundle is 335 MB and the link to Kaggle measured at
~16-25 kB/s, i.e. 4-6 hours. But most of that volume is low-value here:

  * enveda-180 is 1.14M spectra (48% of the library) and is *synthetic* small
    molecules on timsTOF. The competition test set is natural products, so it is
    the wrong chemical space, even though the visible placeholder test happened
    to be sampled from it.
  * The structure table plus the instrument-matched natural-product libraries
    (enveda-np-examples, gnps, riken, mona, massbank, spectraverse, msdial,
    drug_plus, masaryk) is what actually supports retrieval and analog search
    for a natural-product test set.

Outputs (artifacts/kaggle_assets_compact/):
  structures.parquet            - all 275,810 structures with fingerprints
  library_spectra_sorted/       - key-partitioned spectra (natural-product libs)
  dataset-metadata.json         - written BOM-free (a BOM makes the Kaggle CLI
                                  fail with an opaque JSONDecodeError)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DATA = r"D:\CASMI竞赛\data"
SRC_LIB = os.path.join(DATA, "processed", "library_spectra.parquet")
SRC_STRUCTS = os.path.join(DATA, "processed", "structures.parquet")
OUT_DIR = r"D:\CASMI竞赛\artifacts\kaggle_assets_compact"

# Libraries kept. enveda-180 is deliberately excluded: synthetic chemistry.
KEEP_LIBS = ["enveda-np-examples", "gnps", "riken", "mona", "massbank",
             "spectraverse", "msdial", "drug_plus", "masaryk", "pluskal_ms2"]
PART_ROWS = 100_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--np-only", action="store_true",
                    help="keep only instrument-matched np-examples spectra (tiny)")
    ap.add_argument("--drop-rich", action="store_true",
                    help="cap spectra per compound to shrink further")
    ap.add_argument("--max-per-compound", type=int, default=0)
    a = ap.parse_args()

    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)

    # --- structures (all of them: this is the candidate pool) ---
    dst_structs = os.path.join(a.out, "structures.parquet")
    shutil.copyfile(SRC_STRUCTS, dst_structs)
    print(f"structures: {os.path.getsize(dst_structs)/1e6:.1f} MB", flush=True)

    # --- library spectra ---
    keep = ["enveda-np-examples"] if a.np_only else KEEP_LIBS
    print(f"reading library (keeping {len(keep)} sources)...", flush=True)
    df = pd.read_parquet(SRC_LIB)
    n_all = len(df)
    df = df[df["ingest_lib"].isin(keep)].copy()
    print(f"  {n_all:,} -> {len(df):,} spectra, "
          f"{df['key'].nunique():,} compounds", flush=True)

    if a.max_per_compound > 0:
        before = len(df)
        df = (df.sort_values("n_peaks", ascending=False)
                .groupby("key", as_index=False)
                .head(a.max_per_compound))
        print(f"  capped to {a.max_per_compound}/compound: {before:,} -> {len(df):,}",
              flush=True)

    # sort by key and partition so a key never spans two partitions
    df = df.sort_values("key", kind="stable").reset_index(drop=True)
    keys = df["key"].to_numpy()
    n = len(df)
    bounds, start = [], 0
    while start < n:
        end = min(start + PART_ROWS, n)
        while end < n and keys[end] == keys[end - 1]:
            end += 1
        bounds.append((start, end))
        start = end

    out_lib = os.path.join(a.out, "library_spectra_sorted")
    if os.path.isdir(out_lib):
        shutil.rmtree(out_lib)
    os.makedirs(out_lib, exist_ok=True)

    cols = ["key", "ingest_lib", "precursor_mz", "adduct", "is_timstof",
            "mz", "intensity", "n_peaks"]
    index_rows = []
    for p, (s, e) in enumerate(bounds):
        sub = df.iloc[s:e][cols]
        tbl = pa.Table.from_pandas(sub, preserve_index=False)
        pq.write_table(tbl, os.path.join(out_lib, f"part_{p:04d}.parquet"),
                       compression="zstd")
        vc = sub["key"].value_counts()
        for k, c in vc.items():
            index_rows.append({"key": k, "partition": p, "n_spectra": int(c)})
        if p % 4 == 0 or p == len(bounds) - 1:
            print(f"  part {p}/{len(bounds)-1} ({e-s:,} rows, {time.time()-t0:.0f}s)",
                  flush=True)

    idx = pd.DataFrame(index_rows)
    idx.to_parquet(os.path.join(out_lib, "library_index.parquet"), index=False)
    dup = int(idx["key"].duplicated().sum())
    assert dup == 0, f"{dup} keys span multiple partitions"
    print(f"  {len(bounds)} partitions, {len(idx):,} key entries, no key split",
          flush=True)

    # --- metadata, written WITHOUT a BOM ---
    meta = {
        "title": "CASMI26 compact assets",
        "id": "nicholasnicklee/casmi26-assets-compact",
        "licenses": [{"name": "CC0-1.0"}],
        "description": (
            "Offline assets for the CASMI 2026 retrieval + analog pipeline. "
            "structures.parquet: 275,810 deduplicated structures (key, SMILES, "
            "monoisotopic mass, Morgan fingerprint). library_spectra_sorted/: "
            "cleaned MS2 peaks partitioned by structure key from the "
            "natural-product ingest libraries, excluding the synthetic "
            "enveda-180 source."),
    }
    with open(os.path.join(a.out, "dataset-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    total = sum(os.path.getsize(os.path.join(dp, f))
                for dp, _, fs in os.walk(a.out) for f in fs)
    print(f"\nwrote {a.out}")
    print(f"  total {total/1e6:.1f} MB in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
