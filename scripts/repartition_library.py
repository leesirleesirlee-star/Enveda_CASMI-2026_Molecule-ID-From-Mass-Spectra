"""
Repartition the library spectra parquet so it is sorted by structure key and
written in bounded partitions.

Why: the first baseline loaded all 2,367,034 spectra into Python objects, which
cost ~10 GB RSS and ~90 s per process just to start. But a single query only
ever needs the spectra of a few hundred compounds (its mass window plus the
analog window). Sorting by key and partitioning makes that a bounded read: the
in-memory index stays tiny (key -> partition), and peaks are fetched per query.

Output layout (data/processed/library_spectra_sorted/part_XXXX.parquet):
  within each partition, rows are sorted by key and a key never spans two
  partitions, so a query reads exactly the partitions containing its keys.

Also writes library_index.parquet: key, partition, n_spectra, so the runtime
never has to scan the peaks files.
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

SRC = r"D:\CASMI竞赛\data\processed\library_spectra.parquet"
OUT_DIR = r"D:\CASMI竞赛\data\processed\library_spectra_sorted"
PART_ROWS = 100_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--part-rows", type=int, default=PART_ROWS)
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    t0 = time.time()
    pf = pq.ParquetFile(a.src)
    print(f"reading {pf.metadata.num_rows:,} library spectra from {a.src}", flush=True)

    # Read only what we need to sort: the key plus payload. The payload is the
    # big part (peaks), so we stream it in chunks and keep it as Arrow tables.
    chunks = []
    for b in pf.iter_batches(batch_size=250_000):
        chunks.append(b)
    tbl = pa.Table.from_batches(chunks)
    print(f"  loaded in {time.time()-t0:.0f}s, "
          f"{tbl.num_rows:,} rows, {tbl.nbytes/1e9:.2f} GB in memory", flush=True)

    keys = tbl.column("key").to_pylist()
    order = np.argsort(np.asarray(keys, dtype=object), kind="stable")
    tbl = tbl.take(pa.array(order))
    print(f"  sorted by key", flush=True)

    keys_sorted = tbl.column("key").to_pylist()
    n = tbl.num_rows
    # assign partitions so that a key never spans two partitions
    bounds = []           # (start, end_exclusive, first_key)
    start = 0
    while start < n:
        end = min(start + a.part_rows, n)
        # extend to include the whole final key of this partition
        while end < n and keys_sorted[end] == keys_sorted[end - 1]:
            end += 1
        bounds.append((start, end, keys_sorted[start]))
        start = end

    print(f"  writing {len(bounds)} partitions of <= ~{a.part_rows:,} rows", flush=True)
    index_rows = []
    for p, (s, e, first_key) in enumerate(bounds):
        sub = tbl.slice(s, e - s)
        path = os.path.join(a.out_dir, f"part_{p:04d}.parquet")
        pq.write_table(sub, path, compression="zstd")
        # record which keys live here, with their row counts
        kc = pd.Series(sub.column("key").to_pylist()).value_counts()
        for k, c in kc.items():
            index_rows.append({"key": k, "partition": p, "n_spectra": int(c),
                               "row_start": int(s), "row_end": int(e)})
        if p % 5 == 0 or p == len(bounds) - 1:
            print(f"    part {p}/{len(bounds)-1}: {e-s:,} rows  ({time.time()-t0:.0f}s)",
                  flush=True)

    idx = pd.DataFrame(index_rows)
    idx_path = os.path.join(a.out_dir, "library_index.parquet")
    idx.to_parquet(idx_path, index=False)
    print(f"\nwrote {len(idx):,} key entries -> {idx_path}")
    print(f"partitions: {len(bounds)}, total rows: {n:,}, elapsed {time.time()-t0:.0f}s")

    # sanity: a key must not appear in two partitions
    dup = idx["key"].duplicated().sum()
    print(f"keys spanning multiple partitions: {dup} (must be 0)")
    assert dup == 0


if __name__ == "__main__":
    main()
