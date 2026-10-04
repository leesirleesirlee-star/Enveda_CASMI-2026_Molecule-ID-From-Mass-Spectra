"""
Build a lean, FULL-coverage spectral library asset.

Problem this solves: the deployed Kaggle library had only 94,329 compounds while
the local full library has 274,935, so 180,606 candidate structures had no
spectral evidence at all in the kernel. But the full 299 MB of already-zstd
parquet cannot be recompressed (measured ratio 0.99), and the link to Kaggle
runs at ~30-60 kB/s, so shipping all of it is impractical.

What actually shrinks it, in order of effect:
  1. Keep only the columns the runtime reads (the scorer loads key,
     precursor_mz, adduct, is_timstof, mz, intensity; ingest_lib and
     instrument_type were dead weight).
  2. Cap spectra per compound. The full library averages 8.6 spectra/compound;
     for a direct spectral match, a handful of diverse representatives carries
     nearly all the signal. This is tunable and its cost is measurable.
  3. Write with a higher zstd level.

Coverage is kept at ALL 274,935 compounds -- the goal is that every candidate
has *some* spectral evidence, which is what the earlier asset failed to do.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = r"D:\CASMI竞赛"
SRC = os.path.join(ROOT, "data", "processed", "library_spectra.parquet")
OUT = os.path.join(ROOT, "artifacts", "lib_lean")
KEEP = ["key", "precursor_mz", "adduct", "is_timstof", "mz", "intensity", "n_peaks"]
PART_ROWS = 100_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--max-per-compound", type=int, default=4,
                    help="0 = keep every spectrum")
    ap.add_argument("--level", type=int, default=19, help="zstd level")
    a = ap.parse_args()

    t0 = time.time()
    df = pd.read_parquet(SRC, columns=KEEP)
    raw_mb = os.path.getsize(SRC) / 1e6
    print(f"read {len(df):,} spectra ({raw_mb:.1f} MB zstd) in {time.time()-t0:.0f}s",
          flush=True)

    if a.max_per_compound > 0:
        before = len(df)
        # prefer spectra with more peaks: a richer spectrum is more informative
        df = (df.sort_values("n_peaks", ascending=False, kind="stable")
                .groupby("key", as_index=False, sort=False)
                .head(a.max_per_compound))
        print(f"  capped {before:,} -> {len(df):,} spectra "
              f"({a.max_per_compound}/compound)", flush=True)

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

    if os.path.isdir(a.out):
        shutil.rmtree(a.out)
    os.makedirs(a.out, exist_ok=True)

    index_rows = []
    for p, (s, e) in enumerate(bounds):
        sub = df.iloc[s:e][KEEP]
        tbl = pa.Table.from_pandas(sub, preserve_index=False)
        pq.write_table(tbl, os.path.join(a.out, f"part_{p:04d}.parquet"),
                       compression="zstd", compression_level=a.level)
        for k, c in sub["key"].value_counts().items():
            index_rows.append({"key": k, "partition": p, "n_spectra": int(c)})
        if p % 6 == 0 or p == len(bounds) - 1:
            print(f"  part {p}/{len(bounds)-1}  {time.time()-t0:.0f}s", flush=True)

    idx = pd.DataFrame(index_rows)
    idx.to_parquet(os.path.join(a.out, "library_index.parquet"), index=False)
    dup = int(idx["key"].duplicated().sum())
    assert dup == 0, f"{dup} keys span partitions"
    tot = sum(os.path.getsize(os.path.join(a.out, f))
              for f in os.listdir(a.out))
    print(f"\nwrote {a.out}")
    print(f"  compounds {len(idx):,}   spectra {int(idx['n_spectra'].sum()):,}")
    print(f"  size {tot/1e6:.1f} MB   (src {raw_mb:.1f} MB, ratio {tot/1e6/raw_mb:.2f})")
    print(f"  keys split across partitions: {dup}")


if __name__ == "__main__":
    main()
