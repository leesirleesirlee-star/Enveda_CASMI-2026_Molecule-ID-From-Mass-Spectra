"""
Check the ported CLAW logic against the notebook that defines it.

Two things are being protected here:
  1. `promote` must be *slot-preserving except for rank 1* - that is the whole
     point of the rule (bobthebot369 v17: "every other entry keeps its slot").
  2. The gate must be exactly `S > S_TAU and pop_top >= POP_TAU`, inside the
     `lib_max < LIB_TAU` branch, and must fall back to plain `merge` otherwise.

The functions are lifted out of the *generated* notebook, so this test fails if
the builder ever stops emitting the code it claims to emit.

Run: python scripts/test_claw_port.py
"""

from __future__ import annotations

import ast
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB = os.path.join(ROOT, "notebooks", "v45", "claw", "notebook.ipynb")
REF = {"A", "B", "C", "D", "E", "F"}


def load_namespace():
    d = json.loads(open(NB, encoding="utf-8").read())
    src = "".join(d["cells"][19]["source"])
    # keep only the definitions; the driving loop needs the real pipeline
    tree = ast.parse(src)
    keep = [n for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.Import, ast.ImportFrom))]
    assert keep, "cell 19 has no function definitions"
    mod = ast.Module(body=keep, type_ignores=[])
    ns: dict = {}
    exec(compile(mod, "<cell19 defs>", "exec"), ns)  # noqa: S102 - our own generated code
    return ns


def main():
    ns = load_namespace()
    merge, promote = ns["merge"], ns["promote"]

    base = ["A", "B", "C", "D", "E"]
    bkeys = list(base)
    pc = ["F", "A", "G"]
    pkeys = list(pc)
    slots = [2, 4, 6, 8, 10]

    # merge: PubChem-only entries land in the fixed slots, base fills the rest.
    # pc=[F,A,G] with A already in base -> pcs=[F,G]; slots 2 and 4 take them.
    m = merge(base, bkeys, pc, pkeys, slots, n=25)
    assert m == ["A", "F", "B", "G", "C", "D", "E"], m

    # promote: pc[0] leads, everything else keeps its *relative* order
    p = promote(base, bkeys, pc, pkeys, slots, n=25)
    assert p[0] == "F", p
    assert p[1:] == [x for x in m if x != "F"], (p, m)
    assert len(p) == len(m), "promote must not change the list length"

    # promote with no PubChem candidates is exactly merge
    assert promote(base, bkeys, [], [], slots, n=25) == merge(base, bkeys, [], [], slots, n=25)

    # promote must not duplicate the promoted structure
    assert p.count("F") == 1, p

    # cap at n
    long_base = [f"S{i}" for i in range(40)]
    assert len(promote(long_base, list(long_base), ["X"], ["X"], slots, n=25)) == 25

    print("PASS promote/merge semantics")

    # --- gate structure: assert the emitted code still has the CLAW branch ---
    d = json.loads(open(NB, encoding="utf-8").read())
    src = "".join(d["cells"][19]["source"])
    assert "USE_PROMOTION and _S is not None and float(_S) > S_TAU and _pt >= POP_TAU" in src, \
        "gate expression changed"
    assert "elif lib_max >= LIB_TAU:" in src and src.index("elif lib_max >= LIB_TAU:") \
        < src.index("USE_PROMOTION and _S"), "promotion must sit inside the lib_max branch"
    src3 = "".join(d["cells"][3]["source"])
    assert "USE_PROMOTION, S_TAU, POP_TAU = True, 6.0, 5.0" in src3, "gate constants changed"
    print("PASS gate structure (constants 6.0 / 5.0, inside the lib_max<LIB_TAU branch)")
    print("all CLAW port checks passed")


if __name__ == "__main__":
    sys.exit(main())
