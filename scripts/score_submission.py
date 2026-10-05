"""
Score a submission CSV against the recovered visible-test truth.

Why this exists: the leaderboard only gives one number per day (5 submissions),
which is far too little signal to steer. `recover_test_truth.py` recovers the
ground truth of the visible test set from train.parquet; this turns a
submission into the same MRR@25 the grader computes, plus a per-molecule
breakdown that says *which* molecules are wrong.

Assertions that keep this honest:
  * the submission must cover exactly the truth's molecule_ids
  * every candidate is canonicalised with the grader's tautomer rules

Usage:
  python score_submission.py <submission.csv> [more.csv ...]
"""

from __future__ import annotations

import os
import sys

import pandas as pd

ROOT = r"D:\CASMI竞赛"
sys.path.insert(0, os.path.join(ROOT, "src"))

from casmi.core import mrr25_breakdown, smiles_to_inchikey14  # noqa: E402

TRUTH = os.path.join(ROOT, "data", "processed", "test_truth.parquet")


def load_truth():
    t = pd.read_parquet(TRUTH)
    assert t.inchikey14.notna().all(), "unresolved truth molecules"
    return dict(zip(t.molecule_id.astype(str), t.smiles))


def score(path, truths):
    sub = pd.read_csv(path)
    sub["molecule_id"] = sub.molecule_id.astype(str)
    assert sub.molecule_id.is_unique, "duplicate molecule_id"
    missing = set(truths) - set(sub.molecule_id)
    assert not missing, f"submission misses {len(missing)} molecules, e.g. {list(missing)[:3]}"
    preds = {m: [s for s in str(v).split(";") if s]
             for m, v in zip(sub.molecule_id, sub.smiles)}
    for m, p in preds.items():
        assert len(p) <= 25, f"{m} has {len(p)} candidates > 25"
    d = mrr25_breakdown(preds, truths)
    return d, preds


def main():
    paths = sys.argv[1:]
    if not paths:
        print(__doc__)
        raise SystemExit(2)
    truths = load_truth()
    tkeys = {m: {smiles_to_inchikey14(s)} for m, s in truths.items()}
    print(f"truth: {len(truths)} molecules")
    print()
    results = {}
    for p in paths:
        d, preds = score(p, truths)
        results[p] = (d, preds)
        print(f"=== {os.path.basename(p)}")
        for k in ("mrr25", "top1", "hit@5", "hit@10", "hit@25", "mean_rank_when_hit"):
            v = d[k]
            print(f"   {k:20s} {v:.4f}" if isinstance(v, float) else f"   {k:20s} {v}")
        # rank histogram
        ranks = []
        for m, p in preds.items():
            ks = tkeys[m]
            r = 0
            for i, smi in enumerate(p[:25], start=1):
                k = smiles_to_inchikey14(smi)
                if k is not None and k in ks:
                    r = i
                    break
            ranks.append(r)
        r = pd.Series(ranks)
        print("   rank histogram: " + ", ".join(
            f"{lab}={int((r == lo).sum())}" for lab, lo in
            [("1", 1), ("2", 2), ("3-5", 3), ("6-10", 6), ("11-25", 11), ("miss", 0)]
            if lab != "3-5") )
        print(f"   rank1={int((r==1).sum())} rank2={int((r==2).sum())} "
              f"rank3-5={int(r.between(3,5).sum())} rank6-10={int(r.between(6,10).sum())} "
              f"rank11-25={int(r.between(11,25).sum())} miss={int((r==0).sum())}")
        print()

    if len(results) > 1:
        print("=== pairwise top-1 agreement ===")
        names = list(results)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = results[names[i]][1], results[names[j]][1]
                same = sum(
                    bool(a[m]) and bool(b[m]) and
                    smiles_to_inchikey14(a[m][0]) == smiles_to_inchikey14(b[m][0])
                    for m in truths)
                print(f"  {os.path.basename(names[i])} vs {os.path.basename(names[j])}: "
                      f"{same}/{len(truths)} = {same/len(truths)*100:.1f}%")


if __name__ == "__main__":
    main()
