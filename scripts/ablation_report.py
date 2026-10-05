"""
Report the V45 ablation table: every submission, its score, and its delta vs the control.

The control matters because the leaderboard score is deterministic for a fixed
submission file, but *not* across reruns of the same notebook (GPU kernels,
adduct ties and argsort ties all vary). So a variant is only a signal if it
clears the control by more than the control's own spread.

Usage:
  python ablation_report.py [--prefix "V45"]
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request

TOK = open(r"D:\CASMI竞赛\.secrets\kaggle\access_token").read().strip()
COMP = "enveda-CASMI26-molecule-id-mass-spectra"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="V45")
    a = ap.parse_args()

    req = urllib.request.Request(
        f"https://www.kaggle.com/api/v1/competitions/submissions/list/{COMP}",
        headers={"Authorization": "Bearer " + TOK})
    with urllib.request.urlopen(req, timeout=90) as r:
        subs = json.loads(r.read().decode(errors="replace"))

    rows = []
    for s in subs:
        desc = s.get("description") or ""
        if desc.startswith(a.prefix):
            rows.append(dict(
                ref=s.get("ref"), desc=desc,
                score=float(s["publicScore"]) if s.get("publicScore") else None,
                status=s.get("status"), bytes=s.get("totalBytes"),
                date=(s.get("date") or "")[:16],
                err=s.get("errorDescription") or "",
            ))
    if not rows:
        print(f"no submissions with prefix {a.prefix!r} yet")
        return

    ctl = next((r["score"] for r in rows if "ctl" in r["desc"] and r["score"] is not None), None)
    print(f"{'variant':<52} {'ref':>9} {'score':>7} {'delta':>8} {'MB':>6}  status")
    print("-" * 100)
    for r in sorted(rows, key=lambda r: -(r["score"] or 0)):
        variant = r["desc"].replace(f"{a.prefix} ", "")[:50]
        d = f"{r['score']-ctl:+.4f}" if (ctl is not None and r["score"] is not None) else "   -"
        mb = f"{r['bytes']/1e6:.2f}" if r["bytes"] else "-"
        sc = f"{r['score']:.3f}" if r["score"] is not None else "-"
        flag = ""
        if ctl is not None and r["score"] is not None and abs(r["score"] - ctl) > 1e-9:
            flag = "  <<<" if r["score"] > ctl else "  (worse)"
        err = f"  ERR:{r['err'][:40]}" if r["err"] else ""
        print(f"{variant:<52} {str(r['ref']):>9} {sc:>7} {d:>8} {mb:>6}  {r['status']}{flag}{err}")
    print()
    if ctl is None:
        print("!! no control score yet - deltas are not interpretable")
    else:
        print(f"control (ctl) = {ctl:.3f}; treat |delta| <= 0.001 as noise")


if __name__ == "__main__":
    main()
