"""
Build the offline assets needed for retrieval:
  * a labelled "spectral library" of target-like library spectra
  * a mass-indexed structure library with fingerprints

Why filter the library at all? train.parquet holds 2.54M spectra from 11 sources,
most of them acquired on instruments unlike the test set (which is 100% Bruker
timsTOF). Keeping everything both bloats the offline asset and dilutes retrieval
with non-transferable spectra. We keep the instrument-matched sources plus the
broad natural-product libraries that supply the structures, and record which
source supplied each answer so results can be attributed.

Outputs (all under data/processed/):
  library_spectra.parquet  - cleaned peaks + structure key per library spectrum
  structures.parquet       - one row per unique structure (key, smiles, mass, fp)
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from casmi.core import smiles_to_inchikey14           # noqa: E402
from casmi.retrieval import morgan_fp, exact_mass     # noqa: E402
from casmi.spectra import clean_peaks                 # noqa: E402

TRAIN = r"D:\CASMI竞赛\data\raw\train.parquet"
OUT_DIR = r"D:\CASMI竞赛\data\processed"

# Library sources kept for spectral retrieval. timsTOF/Enveda sources match the
# test instrument; gnps/riken/massbank/mona/spectraverse/msdial provide the bulk
# of natural-product structural diversity.
KEEP_LIBS = {
    "enveda-np-examples", "enveda-180", "gnps", "riken", "massbank",
    "mona", "spectraval", "spectraverse", "msdial", "pluskal_ms2", "masaryk",
    "drug_plus",
}
# Sources acquired on the same instrument family as the hidden test set. Only
# these can support a *direct* spectral match claim.
TIMSTOF_LIBS = {"enveda-180", "enveda-np-examples"}

SCALAR_COLS = ["ingest_lib", "normalized_smiles", "inchikey", "inchikey14",
               "molecular_formula", "ionization_mode", "instrument_type",
               "adduct", "precursor_mz", "precursor_error_ppm", "num_peaks",
               "base_peak_intensity", "collision_energy_orig",
               "collision_energy_orig_units"]


def build_structures(pf: pq.ParquetFile, out_path: str) -> dict:
    """One pass over the scalar columns only, deduped to unique structures."""
    acc: dict[str, dict] = {}
    t0 = time.time()
    n_rows = 0
    for batch in pf.iter_batches(batch_size=200_000, columns=SCALAR_COLS):
        df = batch.to_pandas()
        n_rows += len(df)
        for smi, k, formula, lib in zip(df["normalized_smiles"], df["inchikey14"],
                                        df["molecular_formula"], df["ingest_lib"]):
            if smi is None or (isinstance(smi, float) and np.isnan(smi)):
                continue
            key = k if isinstance(k, str) and k else smiles_to_inchikey14(smi)
            if not key:
                continue
            rec = acc.get(key)
            if rec is None:
                m = exact_mass(smi)
                if m is None:
                    continue
                rec = {"key": key, "smiles": smi, "mass": m,
                       "formula": formula, "libs": [], "n_spectra": 0}
                acc[key] = rec
            if lib not in rec["libs"]:
                rec["libs"].append(lib)
            rec["n_spectra"] += 1
        print(f"  scanned {n_rows:,} rows, {len(acc):,} unique structures "
              f"({time.time()-t0:.0f}s)", flush=True)

    keys = list(acc)
    print(f"  computing fingerprints for {len(keys):,} structures...", flush=True)
    fps = [morgan_fp(acc[k]["smiles"]) for k in keys]
    rows = []
    for k, fp in zip(keys, fps):
        r = acc[k]
        rows.append({
            "key": k, "smiles": r["smiles"], "mass": r["mass"],
            "formula": r["formula"], "libs": ";".join(sorted(r["libs"])),
            "n_spectra": r["n_spectra"],
            "fp": fp.tolist() if fp is not None else None,
        })
    out = pd.DataFrame(rows)
    out.to_parquet(out_path, index=False)
    print(f"  wrote {out_path}: {len(out):,} structures", flush=True)
    return {"n_structures": len(out), "n_rows_scanned": n_rows}


def build_library_spectra(pf: pq.ParquetFile, out_path: str,
                          batch_size: int = 100_000) -> dict:
    """Cleaned peaks for the kept library sources, streamed batch by batch."""
    import zlib

    writer = None
    n_in = n_out = 0
    t0 = time.time()
    cols = SCALAR_COLS + ["ms2_mzs", "ms2_normalized_intensities"]
    for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
        df = batch.to_pandas()
        n_in += len(df)
        keep = df["ingest_lib"].isin(KEEP_LIBS)
        df = df[keep]
        if len(df) == 0:
            continue
        recs = []
        for row in df.itertuples(index=False):
            mz, inten = clean_peaks(row.ms2_mzs, row.ms2_normalized_intensities)
            if mz.size < 3:
                continue
            key = row.inchikey14 if isinstance(row.inchikey14, str) else None
            if not key:
                key = smiles_to_inchikey14(row.normalized_smiles)
            if not key:
                continue
            recs.append({
                "key": key,
                "ingest_lib": row.ingest_lib,
                "instrument_type": row.instrument_type,
                "is_timstof": row.ingest_lib in TIMSTOF_LIBS,
                "adduct": row.adduct,
                "precursor_mz": float(row.precursor_mz),
                "mz": mz.tobytes(),
                "intensity": inten.tobytes(),
                "n_peaks": int(mz.size),
            })
        n_out += len(recs)
        if recs:
            tbl = pa.Table.from_pandas(pd.DataFrame(recs), preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out_path, tbl.schema, compression="zstd")
            writer.write_table(tbl)
        print(f"  library: {n_in:,} scanned -> {n_out:,} kept ({time.time()-t0:.0f}s)",
              flush=True)
    if writer is not None:
        writer.close()
    print(f"  wrote {out_path}: {n_out:,} library spectra", flush=True)
    return {"n_library_spectra": n_out, "n_scanned": n_in}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=TRAIN)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--skip-library", action="store_true")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    print(f"opening {a.train}")
    pf = pq.ParquetFile(a.train)
    print(f"  rows={pf.metadata.num_rows:,} row_groups={pf.num_row_groups}")

    stats = {}
    if not a.skip_library:
        print("\n[1/2] building library spectra (peaks)...")
        stats.update(build_library_spectra(
            pf, os.path.join(a.out_dir, "library_spectra.parquet")))
    print("\n[2/2] building structure library...")
    stats.update(build_structures(
        pf, os.path.join(a.out_dir, "structures.parquet")))

    print("\nsummary:", stats)
    import json
    with open(os.path.join(a.out_dir, "build_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
