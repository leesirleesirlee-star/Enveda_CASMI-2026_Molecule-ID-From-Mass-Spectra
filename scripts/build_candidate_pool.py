"""
Build the merged natural-product candidate pool (docs/prd_patch.md §2.1 Layer C).

The patch's core correction: the hidden test is natural-product dark chemical
space, and the answer is only reachable if it is IN the candidate pool. My
diagnostic measured that with a train-only pool the true structure is a candidate
for just 1.2% of visible-test molecules -- and the visible test is the easy,
leaked case. So the pool is the binding constraint, not the ranking code.

Sources merged here, deduplicated by InChIKey14:
  * train.parquet structures      (275,810)  -- Layer A/B structural backbone
  * COCONUT 2.0 via two mirrors   (738,827 + 436,389 raw)
  * LOTUS natural products        (150,590)
  * NPAtlas                       (TSV)

Output: data/processed/candidate_pool.parquet
  columns: key, smiles, mass, formula, sources, fp (packed 2048-bit Morgan)
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
sys.path.insert(0, os.path.join(ROOT, "src"))
from casmi.retrieval import exact_mass, morgan_fp          # noqa: E402
from casmi.core import smiles_to_inchikey14                # noqa: E402

TRAIN_STRUCTS = os.path.join(ROOT, "data", "processed", "structures.parquet")
EXT = os.path.join(ROOT, "data", "external")
OUT = os.path.join(ROOT, "data", "processed", "candidate_pool.parquet")


def _norm_key(k, smi):
    if isinstance(k, str) and len(k) >= 14:
        return k[:14]
    if isinstance(k, str) and k:
        return k
    return smiles_to_inchikey14(smi) if smi else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--limit-per-source", type=int, default=0)
    a = ap.parse_args()

    acc: dict[str, dict] = {}
    t0 = time.time()

    def add(key, smiles, mass, formula, source):
        if not key or not smiles:
            return 0
        if key in acc:
            acc[key]["sources"].add(source)
            return 0
        acc[key] = {"key": key, "smiles": smiles,
                    "mass": float(mass) if mass is not None else np.nan,
                    "formula": formula, "sources": {source}}
        return 1

    # ---- 1. train structures (already has keys + masses) ----
    st = pd.read_parquet(TRAIN_STRUCTS, columns=["key", "smiles", "mass", "formula", "libs"])
    n = 0
    for k, s, m, f, libs in zip(st["key"], st["smiles"], st["mass"], st["formula"], st["libs"]):
        n += add(k, s, m, f, "train")
    print(f"train: +{n:,}  (total {len(acc):,})  {time.time()-t0:.0f}s", flush=True)

    # ---- 2. COCONUT (aidensong123 mirror) ----
    p = os.path.join(EXT, "coconut_aiden", "coconut_structures.parquet")
    if os.path.exists(p):
        df = pd.read_parquet(p, columns=["canonical_smiles", "exact_mass",
                                         "inchikey", "formula"])
        if a.limit_per_source:
            df = df.head(a.limit_per_source)
        n = 0
        for smi, m, ik, f in zip(df["canonical_smiles"], df["exact_mass"],
                                 df["inchikey"], df["formula"]):
            n += add(_norm_key(ik, smi), smi, m, f, "coconut")
        print(f"coconut(aiden): +{n:,}  (total {len(acc):,})  {time.time()-t0:.0f}s",
              flush=True)

    # ---- 3. COCONUT 2.0 (prvsiyan mirror: keys+smiles+mass) ----
    p = os.path.join(EXT, "coconut_prvsiyan", "coco_meta.pkl")
    if os.path.exists(p):
        import pickle
        with open(p, "rb") as fh:
            meta = pickle.load(fh)
        mass = np.load(os.path.join(EXT, "coconut_prvsiyan", "coco_mass.npy"))
        keys, smis = meta["keys"], meta["smiles"]
        lim = a.limit_per_source or len(keys)
        n = 0
        for i in range(min(lim, len(keys))):
            n += add(keys[i], smis[i], mass[i], None, "coconut2")
        print(f"coconut2(prvsiyan): +{n:,}  (total {len(acc):,})  {time.time()-t0:.0f}s",
              flush=True)

    # ---- 4. LOTUS ----
    p = os.path.join(EXT, "casmi26-lotus-pool", "lotus_pool.parquet")
    if os.path.exists(p):
        df = pd.read_parquet(p)
        if a.limit_per_source:
            df = df.head(a.limit_per_source)
        n = 0
        for k, smi, m, f in zip(df["ik14_std"], df["smiles"], df["mass"], df["formula"]):
            n += add(_norm_key(k, smi), smi, m, f, "lotus")
        print(f"lotus: +{n:,}  (total {len(acc):,})  {time.time()-t0:.0f}s", flush=True)

    # ---- 5. NPAtlas ----
    p = os.path.join(EXT, "npatlas", "NPAtlas_download_2024_09.tsv")
    if os.path.exists(p):
        df = pd.read_csv(p, sep="\t", low_memory=False,
                         usecols=lambda c: c in (
                             "compound_inchikey", "compound_smiles",
                             "compound_accurate_mass", "compound_molecular_formula"))
        if a.limit_per_source:
            df = df.head(a.limit_per_source)
        n = 0
        for ik, smi, m, f in zip(df["compound_inchikey"], df["compound_smiles"],
                                 df["compound_accurate_mass"],
                                 df["compound_molecular_formula"]):
            n += add(_norm_key(ik, smi), smi, m, f, "npatlas")
        print(f"npatlas: +{n:,}  (total {len(acc):,})  {time.time()-t0:.0f}s", flush=True)

    # ---- materialise, filling missing masses ----
    keys = list(acc)
    print(f"\ncomputing fingerprints / masses for {len(keys):,} structures...", flush=True)
    rows = []
    for i, k in enumerate(keys):
        r = acc[k]
        m = r["mass"]
        if not np.isfinite(m):
            m = exact_mass(r["smiles"])
            if m is None:
                continue
        fp = morgan_fp(r["smiles"])
        if fp is None:
            continue
        rows.append({"key": k, "smiles": r["smiles"], "mass": m,
                     "formula": r["formula"],
                     "sources": ";".join(sorted(r["sources"])),
                     "fp": fp.tolist()})
        if (i + 1) % 100_000 == 0:
            print(f"  {i+1:,}/{len(keys):,}  {time.time()-t0:.0f}s", flush=True)

    out = pd.DataFrame(rows)
    out.to_parquet(a.out, index=False)
    print(f"\nwrote {a.out}: {len(out):,} structures, "
          f"{os.path.getsize(a.out)/1e6:.1f} MB, {time.time()-t0:.0f}s")
    print("\nsource coverage (structures naming each source):")
    for s in ["train", "coconut", "coconut2", "lotus", "npatlas"]:
        n = int(out["sources"].str.contains(s).sum())
        print(f"  {s:10s} {n:,}")


if __name__ == "__main__":
    main()
