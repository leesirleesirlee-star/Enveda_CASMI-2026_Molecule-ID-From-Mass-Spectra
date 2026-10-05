"""
Differential test: our ported `merge`/`promote` must equal the reference's.

`scripts/test_claw_port.py` checks hand-written expectations. This goes further:
it loads the two implementations from their own sources - the reference
(bobthebot369 v17, the 0.417 release) and our generated notebook - and compares
them on randomised inputs. A silent porting difference is exactly the failure
that would waste a GPU run and a submission.

Run: python scripts/test_claw_diff.py
"""

from __future__ import annotations

import ast
import json
import os
import random
import sys

ROOT = r"D:\CASMI竞赛"
REF = os.path.join(ROOT, ".deepworks", "tmp", "v17", "fusion_core.py")
OURS = os.path.join(ROOT, "notebooks", "v45", "claw", "notebook.ipynb")


def defs_from_source(src: str, names: set[str]) -> dict:
    tree = ast.parse(src)
    keep = [n for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in keep} == names, f"missing {names - {n.name for n in keep}}"
    # one namespace only: exec(code, ns) makes ns the functions' __globals__,
    # so `promote` can resolve `merge`
    ns: dict = {"np": __import__("numpy")}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "<defs>", "exec"), ns)  # noqa: S102
    return ns


def main():
    ref = defs_from_source(open(REF, encoding="utf-8").read(), {"merge", "promote"})
    d = json.loads(open(OURS, encoding="utf-8").read())
    ours = defs_from_source("".join(d["cells"][19]["source"]), {"merge", "promote"})

    rng = random.Random(20261005)
    trials = 3000
    diff = 0
    for t in range(trials):
        nb = rng.randint(0, 30)
        npc = rng.randint(0, 30)
        base = [f"B{i}" for i in range(nb)]
        pc = [f"P{i}" for i in range(npc)]
        # deliberately overlap keys so the "already in base" filter is exercised
        if rng.random() < 0.5 and base:
            pc = [rng.choice(base) if rng.random() < 0.4 else p for p in pc]
        slots = rng.choice([[2, 4, 6, 8, 10], [4, 8, 12, 16, 20],
                            [2, 3, 4, 5, 6, 8, 10, 12, 14, 16], []])
        n = rng.choice([25, 25, 10, 5, 1])
        a = ref["promote"](base, list(base), pc, list(pc), slots, n=n)
        b = ours["promote"](base, list(base), pc, list(pc), slots, n=n)
        a2 = ref["merge"](base, list(base), pc, list(pc), slots, n=n)
        b2 = ours["merge"](base, list(base), pc, list(pc), slots, n=n)
        if a != b or a2 != b2:
            diff += 1
            if diff <= 3:
                print(f"MISMATCH trial {t}: base={len(base)} pc={pc[:6]} slots={slots} n={n}")
                print(f"   ref merge ={a2}\n   our merge ={b2}")
                print(f"   ref promo ={a}\n   our promo ={b}")
    print(f"{trials} randomised cases, {diff} mismatches")
    if diff:
        print("FAIL: our CLAW helpers differ from the reference")
        return 1
    print("PASS: merge/promote are behaviourally identical to bobthebot369 v17")
    return 0


if __name__ == "__main__":
    sys.exit(main())
