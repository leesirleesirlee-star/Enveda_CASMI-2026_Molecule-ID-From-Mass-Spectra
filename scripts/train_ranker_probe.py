"""
Probe: can we actually train and evaluate a ranker on the public ranker rows?

This is the modelling track's step-2 harness, minus the reference comparison.
It answers three concrete questions before we spend any GPU quota:

  1. do the 819 public query groups load and group correctly?
  2. do the 31 features carry ranking signal at all (group-wise CV MRR@25)?
  3. what is the recall ceiling on this data - i.e. for how many groups does the
     group even contain a positive? No ranker can beat that.

Group-wise CV, never row-wise: rows inside a group share a query, so a row split
would leak and report a fantasy score.

Run: python scripts/train_ranker_probe.py
"""

from __future__ import annotations

import sys

import numpy as np

NPZ = r"D:\CASMI竞赛\.deepworks\tmp\rankerfeat\rank_train.npz"
K = 25


def main():
    z = np.load(NPZ)
    X, Y, G = z["X"].astype(np.float32), z["Y"], z["G"]
    groups = np.unique(G)
    print(f"rows={len(Y):,}  features={X.shape[1]}  groups={len(groups):,}  "
          f"positives={int(Y.sum()):,} ({Y.mean()*100:.2f}%)")

    # recall ceiling: a group with no positive at all is unwinnable
    pos_per_group = np.array([Y[G == g].sum() for g in groups])
    print(f"groups containing >=1 positive: {(pos_per_group > 0).sum():,}/{len(groups):,} "
          f"= {(pos_per_group > 0).mean()*100:.1f}%   <-- recall ceiling")

    sizes = np.array([int((G == g).sum()) for g in groups])
    print(f"candidates/group: median {int(np.median(sizes))}, min {sizes.min()}, max {sizes.max()}")

    import lightgbm as lgb

    rng = np.random.default_rng(20261005)
    order = rng.permutation(groups)
    folds = np.array_split(order, 5)
    ranks, top1 = [], []
    for i, va_groups in enumerate(folds):
        va = np.isin(G, va_groups)
        tr = ~va
        # lightgbm needs rows grouped contiguously with a group-size vector
        tr_idx = np.argsort(G[tr], kind="stable")
        Xtr, Ytr, Gtr = X[tr][tr_idx], Y[tr][tr_idx], G[tr][tr_idx]
        _, counts = np.unique(Gtr, return_counts=True)
        model = lgb.LGBMRanker(
            objective="lambdarank", n_estimators=300, learning_rate=0.05,
            num_leaves=31, min_child_samples=20, subsample=1.0, colsample_bytree=1.0,
            label_gain=[0, 1], verbosity=-1,
        )
        model.fit(Xtr, Ytr.astype(int), group=counts)
        s = model.predict(X[va])
        for g in va_groups:
            m = G[va] == g
            if not m.any() or Y[va][m].sum() == 0:
                ranks.append(0)
                continue
            # rank of the first positive by descending score
            sc = s[m]
            lab = Y[va][m]
            o = np.argsort(-sc, kind="stable")
            r = int(np.argmax(lab[o] > 0)) + 1
            ranks.append(r if r <= K else 0)
            top1.append(int(r == 1))
        print(f"  fold {i}: {len(va_groups)} groups, "
              f"MRR@25={np.mean([1.0/r for r in ranks[-len(va_groups):] if r > 0] or [0]):.4f}")

    ranks = np.asarray(ranks, dtype=float)
    hit = ranks > 0
    print()
    print(f"group-wise CV over {len(ranks):,} groups")
    print(f"  MRR@25            {np.mean(np.where(hit, 1.0/np.maximum(ranks,1), 0.0)):.4f}")
    print(f"  top-1             {np.mean(ranks == 1):.4f}")
    print(f"  hit@25            {hit.mean():.4f}")
    print("PASS: ranker rows load, group, train and evaluate" if hit.any()
          else "FAIL: nothing hit at all - grouping or labels are wrong")
    return 0 if hit.any() else 1


if __name__ == "__main__":
    sys.exit(main())
