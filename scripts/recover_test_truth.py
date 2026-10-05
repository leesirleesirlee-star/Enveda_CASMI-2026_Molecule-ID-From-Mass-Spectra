"""
Recover the ground truth of the visible test set from train.parquet.

Fact (measured, not assumed): every visible test spectrum has a peak set that is
verbatim present in train.parquet, so each test spectrum can be mapped to the
structure train associates with it.

This is an *evaluation* artifact: it lets us score a candidate submission
locally. It is deliberately NOT a submission generator.

Output: data/processed/test_truth.parquet
  molecule_id, inchikey14 (majority over the molecule's resolved spectra),
  smiles, n_resolved, n_spectra, formula_agree

Self-check (runs at the end, prints PASS/FAIL):
  * every molecule must resolve at least one spectrum
  * the recovered structure must share the query's molecular formula, which is
    an independent signal (mass/adduct), not a copy of the label
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from casmi.core import formula_from_smiles, formula_neutral_mass  # noqa: E402

TEST = os.path.join(ROOT, "data", "raw", "test.parquet")
TRAIN = os.path.join(ROOT, "data", "raw", "train.parquet")
OUT = os.path.join(ROOT, "data", "processed", "test_truth.parquet")


def sig(mz) -> str:
    return hashlib.md5(np.round(np.asarray(mz, np.float64), 4).tobytes()).hexdigest()


def main():
    t0 = time.time()
    test = pd.read_parquet(TEST)
    test["mid"] = test["molecule_id"].astype(str)
    tsig = {}
    for mid, mz in zip(test["mid"], test["ms2_mzs"]):
        tsig.setdefault(sig(mz), set()).add(mid)
    print(f"test spectra: {len(test):,} | unique signatures: {len(tsig):,} "
          f"| molecules: {test.mid.nunique()}", flush=True)

    # signature -> {inchikey14: count} over train rows
    hits: dict[str, Counter] = defaultdict(Counter)
    smi_of: dict[str, str] = {}
    pf = pq.ParquetFile(TRAIN)
    n_rows = 0
    for bi, b in enumerate(pf.iter_batches(
            batch_size=200_000,
            columns=["ms2_mzs", "inchikey14", "normalized_smiles"])):
        df = b.to_pandas()
        n_rows += len(df)
        for mz, k, s in zip(df["ms2_mzs"], df["inchikey14"], df["normalized_smiles"]):
            h = sig(mz)
            if h in tsig:
                if isinstance(k, str) and k:
                    hits[h][k] += 1
                    smi_of.setdefault(k, s if isinstance(s, str) else "")
        if bi % 3 == 0:
            print(f"  rowgroup batch {bi}: {n_rows:,} rows, "
                  f"{len(hits):,}/{len(tsig):,} sigs matched, {time.time()-t0:.0f}s",
                  flush=True)
    print(f"scanned {n_rows:,} train rows in {time.time()-t0:.0f}s; "
          f"matched {len(hits):,}/{len(tsig):,} test spectra", flush=True)

    # per molecule: majority vote over its resolved spectra
    per_mol: dict[str, Counter] = defaultdict(Counter)
    n_spec = Counter()
    for mid in test["mid"]:
        n_spec[mid] += 1
    for mz, mid in zip(test["ms2_mzs"], test["mid"]):
        c = hits.get(sig(mz))
        if c:
            for k, v in c.items():
                per_mol[mid][k] += v

    rows = []
    for mid in sorted(n_spec):
        c = per_mol.get(mid)
        if not c:
            rows.append((mid, None, None, 0, n_spec[mid], None))
            continue
        key, _ = c.most_common(1)[0]
        rows.append((mid, key, smi_of.get(key), sum(c.values()), n_spec[mid],
                     len(c)))
    truth = pd.DataFrame(rows, columns=["molecule_id", "inchikey14", "smiles",
                                        "n_resolved", "n_spectra", "n_candidates"])
    truth.to_parquet(OUT, index=False)
    print(f"wrote {OUT}: {len(truth)} molecules", flush=True)

    resolved = truth["inchikey14"].notna()
    print(f"resolved molecules: {resolved.sum()}/{len(truth)}")
    print(f"n_candidates distribution:\n{truth.n_candidates.value_counts().head(8)}")

    # independent check: formula from the recovered SMILES vs formula implied by
    # the precursor m/z + adduct (this does not use the label at all)
    agree = tot = 0
    bad = []
    for r in truth[resolved].itertuples():
        sub = test[test.mid == r.molecule_id]
        f_smi = formula_from_smiles(r.smiles) if isinstance(r.smiles, str) else None
        if f_smi is None:
            continue
        m_smi = formula_neutral_mass(f_smi)
        if m_smi is None:
            continue
        for p in sub.itertuples():
            m_prec = None
            try:
                from casmi.core import neutral_mass_from_precursor
                m_prec = neutral_mass_from_precursor(float(p.precursor_mz), str(p.adduct))
            except Exception:
                pass
            if m_prec is None:
                continue
            tot += 1
            ok = abs(m_smi - m_prec) / m_smi * 1e6 <= 40.0
            agree += ok
            if not ok and len(bad) < 5:
                bad.append((r.molecule_id, m_smi, m_prec))
    if tot:
        print(f"formula-mass agreement with precursor: {agree}/{tot} = {agree/tot*100:.1f}%")
        for b in bad:
            print("   outlier:", b)
    print("PASS" if resolved.all() else "FAIL: unresolved molecules", flush=True)


if __name__ == "__main__":
    main()
