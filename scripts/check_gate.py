"""
Self-check for the environment gate in run_gated_variant.py.

The gate decides whether a variant may be submitted, so a wrong verdict is expensive in both
directions: a false PASS submits a run whose forward models are dead, and a false FAIL blocks a
good run. The second one actually happened -- the criterion demanded `ICE meta status == "ok"`,
but a healthy budget-limited ICE reports `"budget"`, so the pinned ctl run was rejected even
though it reproduced the reference exactly (71 molecules scored, 365 rows reordered).

Cases below are the shapes that matter, including the real log from the broken unpinned run.

Usage:  python scripts/check_gate.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "_gated", os.path.join(ROOT, "scripts", "run_gated_variant.py"))
_gated = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gated)

Q = '"'


def rec(data):
    """One Kaggle log record, with quotes escaped the way the real log escapes them."""
    return json.dumps({"stream_name": "stdout", "time": 1.0, "data": data}) + "\n"


ICE_OK = ("ICE meta {" + Q + "status" + Q + ": " + Q + "ok" + Q
          + ", " + Q + "n_mols_scored" + Q + ": 71}\n")
ICE_BUDGET = ("ICE meta {" + Q + "status" + Q + ": " + Q + "budget" + Q
              + ", " + Q + "n_mols_scored" + Q + ": 71, " + Q + "errors" + Q + ": []}\n")
ICE_ERR = ("ICE meta {" + Q + "status" + Q + ": " + Q + "error" + Q
           + ", " + Q + "errors" + Q + ": [" + Q + "RuntimeError: pip install failed" + Q + "]}\n")
RERANKED = "ICE rerank stats {'molecules': 71, 'changed_top25': 67, 'changed_top1': 0}\n"
RERANKED_ZERO = "ICE rerank stats {'molecules': 0, 'changed_top25': 0, 'changed_top1': 0}\n"

REAL_BAD_LOG = os.path.join(ROOT, ".deepworks", "tmp", "our_run.log")

CASES = [
    ("real log from the broken unpinned run",
     open(REAL_BAD_LOG, encoding="utf-8").read() if os.path.exists(REAL_BAD_LOG) else "", False),
    ("healthy, status ok", rec(ICE_OK) + rec(RERANKED), True),
    ("healthy, budget-limited (the case that was mis-judged)", rec(ICE_BUDGET) + rec(RERANKED), True),
    ("ok but nothing reranked", rec(ICE_OK) + rec(RERANKED_ZERO), False),
    ("still erroring", rec(ICE_ERR) + rec(RERANKED), False),
    ("empty log", "", False),
]


def main():
    fails = 0
    for name, log, expected in CASES:
        ok, why = _gated.gate(log)
        good = ok == expected
        fails += not good
        print(f"  {'ok  ' if good else 'FAIL'} {name:<56} gate={'PASS' if ok else 'FAIL'} "
              f"(want {'PASS' if expected else 'FAIL'})  {why[:60]}")
    print(f"\n  cases: {len(CASES)}   mismatches: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
