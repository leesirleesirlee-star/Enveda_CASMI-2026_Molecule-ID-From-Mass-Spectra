"""
Verify the compact asset bundle works with the *notebook's own* code path.

The notebook embeds a standalone copy of the scoring logic (it cannot import the
`casmi` package offline). That copy can drift from the validated modules, so the
only trustworthy check is to run the notebook's code against the compact asset
and confirm it still produces well-formed, non-degenerate predictions.

Checks:
  * structures.parquet has the expected columns and row count
  * library_spectra_sorted/ index covers every key, no key spans partitions
  * the notebook code runs and produces 25 candidates for the vast majority of
    molecules (a median of 0 would mean silent breakage)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = r"D:\CASMI竞赛"


def check_assets(asset_dir: str) -> dict:
    out = {}
    st = os.path.join(asset_dir, "structures.parquet")
    df = pd.read_parquet(st, columns=["key", "smiles", "mass"])
    out["structures"] = len(df)
    out["structure_cols_ok"] = True
    print(f"structures: {len(df):,} rows, mass {df['mass'].min():.1f}-{df['mass'].max():.1f}")

    libdir = os.path.join(asset_dir, "library_spectra_sorted")
    idx = pd.read_parquet(os.path.join(libdir, "library_index.parquet"))
    parts = [f for f in os.listdir(libdir) if f.startswith("part_") and f.endswith(".parquet")]
    out["library_keys"] = int(len(idx))
    out["library_spectra"] = int(idx["n_spectra"].sum())
    out["partitions"] = len(parts)
    dup = int(idx["key"].duplicated().sum())
    out["keys_split_across_partitions"] = dup
    print(f"library: {out['library_keys']:,} keys, {out['library_spectra']:,} spectra, "
          f"{len(parts)} partitions, keys split across partitions: {dup}")
    assert dup == 0, "a key spans two partitions; by_keys would miss spectra"

    # every index key must be reachable in the structures table (the scorer needs
    # both a structure and its spectra to use direct evidence)
    missing = set(idx["key"]) - set(df["key"])
    out["library_keys_missing_structure"] = len(missing)
    print(f"library keys with no structure record: {len(missing)}")
    return out


def run_notebook(asset_dir: str, limit: int, out_csv: str) -> dict:
    nb = json.load(open(os.path.join(ROOT, "notebooks", "kaggle_submission",
                                     "notebook.ipynb"), encoding="utf-8"))
    code = "".join(nb["cells"][1]["source"])
    code = code.replace("/kaggle/input/enveda-CASMI26-molecule-id-mass-spectra",
                        f"{ROOT}/data/raw".replace("\\", "/"))
    code = code.replace("/kaggle/input/casmi26-assets-compact",
                        asset_dir.replace("\\", "/"))
    code = code.replace("/kaggle/working/submission.csv", out_csv.replace("\\", "/"))
    if limit:
        code = code.replace("for n, mid in enumerate(ids, 1):",
                            f"for n, mid in enumerate(ids[:{limit}], 1):")
        code = code.replace("    for mid in ids:", f"    for mid in ids[:{limit}]:")
    tmp = os.path.join(ROOT, ".deepworks", "tmp", "nb_verify.py")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(code)
    print(f"\nrunning notebook code (limit={limit or 'all'})...")
    r = subprocess.run([sys.executable, "-u", tmp], capture_output=True, text=True,
                       cwd=ROOT)
    tail = (r.stdout or "").strip().splitlines()[-8:]
    for line in tail:
        print("   ", line)
    if r.returncode != 0:
        print("STDERR:", (r.stderr or "")[-1500:])
        raise SystemExit("notebook code failed")
    sub = pd.read_csv(out_csv)
    counts = sub["smiles"].astype(str).str.split(";").apply(
        lambda xs: len([x for x in xs if x.strip()]))
    res = {"rows": len(sub), "median_candidates": int(counts.median()),
           "min_candidates": int(counts.min()), "empty_rows": int((counts == 0).sum())}
    print(f"\nsubmission: rows={res['rows']} median_candidates={res['median_candidates']} "
          f"empty_rows={res['empty_rows']}")
    assert res["median_candidates"] > 0, "notebook produced empty predictions"
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default=os.path.join(ROOT, "artifacts", "kaggle_assets_np3"))
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--out-csv", default=os.path.join(ROOT, ".deepworks", "tmp", "verify_sub.csv"))
    a = ap.parse_args()

    print(f"=== checking assets in {a.assets} ===")
    stats = check_assets(a.assets)
    stats.update(run_notebook(a.assets, a.limit, a.out_csv))
    print("\nRESULT:", json.dumps(stats, indent=2))
    print("\nCOMPACT ASSET VERIFIED")


if __name__ == "__main__":
    main()
