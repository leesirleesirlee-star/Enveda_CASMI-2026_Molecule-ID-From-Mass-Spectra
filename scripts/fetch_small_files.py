"""Fetch the small competition files via the official Kaggle API client."""
import os
from kaggle.api.kaggle_api_extended import KaggleApi

RAW = r"D:\CASMI竞赛\data\raw"
COMP = "enveda-CASMI26-molecule-id-mass-spectra"

api = KaggleApi()
api.authenticate()

for fn in ["test.parquet", "sample_submission.csv"]:
    dest = os.path.join(RAW, fn)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"skip (exists): {fn} {os.path.getsize(dest):,} bytes")
        continue
    print(f"downloading {fn} ...", flush=True)
    api.competition_download_file(COMP, fn, path=RAW, force=False, quiet=False)
    print(f"  -> {os.path.getsize(dest):,} bytes")

print("\nfinal:")
for fn in sorted(os.listdir(RAW)):
    p = os.path.join(RAW, fn)
    if os.path.isfile(p):
        print(f"  {fn:45s} {os.path.getsize(p):>15,} bytes")
