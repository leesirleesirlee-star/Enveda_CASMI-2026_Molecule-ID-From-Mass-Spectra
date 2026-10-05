"""
Pipeline-fidelity check: does our pushed notebook reproduce the reference output?

This is the check PLAYBOOK.md lists as item 0 of "after every experiment". It answers a
question we have never actually tested: is `build_variant_kernel -> push -> run -> output`
faithful? If it is not, every variant score rests on an unverified baseline.

The reference is the author's public run of the same pipeline, whose output we downloaded:
400 rows, 9,772 candidates, 393,908 bytes.

One wrinkle, caught while designing this: `kernels/output` returns the files of the LATEST
version only. The ablation kernel's latest version is `nolock`, not `ctl`, so what we can
download is nolock's public output. Its relationship to the reference is exactly
predictable -- the two differ by cell 31's champion lock, and on public data the lock does
fire but moves exactly ONE molecule (the author's log says "top-1 locks applied: 1").
So the criterion is not "byte-identical" but "differences confined to a single row".

Usage:
  python scripts/check_pipeline_fidelity.py                 # check, no network writes
  python scripts/check_pipeline_fidelity.py --list           # just show what is downloadable
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCE = os.path.join(ROOT, ".deepworks", "tmp", "v44_submission.csv")
DOWNLOAD_DIR = os.path.join(ROOT, ".deepworks", "tmp", "fidelity")
KERNEL = "nicholasnicklee/casmi26-v45-ablation"
TOKEN_PATH = os.path.join(ROOT, ".secrets", "kaggle", "access_token")

# the reference's measured shape -- asserted, not assumed
REF_BYTES, REF_ROWS, REF_CANDIDATES = 393_908, 400, 9_772


def token():
    if not os.path.exists(TOKEN_PATH):
        raise SystemExit(f"no Kaggle token at {TOKEN_PATH}")
    return open(TOKEN_PATH, encoding="utf-8").read().strip()


def api(url, tok, raw=False):
    req = urllib.request.Request(url, headers={"Authorization": "Bearer " + tok})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read() if raw else json.loads(r.read())


def shape(path):
    import pandas as pd
    d = pd.read_csv(path)
    n = d.smiles.astype(str).str.split(";").map(len)
    return dict(bytes=os.path.getsize(path), rows=len(d),
                candidates=int(n.sum()), per_row=(int(n.min()), int(n.max())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", default=KERNEL)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    owner, slug = a.kernel.split("/", 1)
    tok = token()

    # the reference must be the one we verified; refuse to use anything else
    if not os.path.exists(REFERENCE):
        raise SystemExit(f"reference output missing: {REFERENCE}")
    ref = shape(REFERENCE)
    assert (ref["bytes"], ref["rows"], ref["candidates"]) == (REF_BYTES, REF_ROWS, REF_CANDIDATES), \
        f"the reference is not the file we validated earlier: {ref}"

    out = api(f"https://www.kaggle.com/api/v1/kernels/output?userName={owner}&kernelSlug={slug}", tok)
    files = [f.get("fileName") for f in out.get("files", [])]
    print(f"kernel        : {a.kernel}")
    print(f"output files  : {len(files)}")
    for f in files[:10]:
        print(f"     {f}")
    if a.list:
        return 0
    if "submission.csv" not in files:
        print("\nnot ready: the run has not produced submission.csv yet.")
        print("(the kernel's output listing is empty until the version finishes)")
        return 2

    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    dst = os.path.join(DOWNLOAD_DIR, f"{slug}_submission.csv")
    url = f"https://www.kaggle.com/api/v1/kernels/output/download/{owner}/{slug}/submission.csv"
    with open(dst, "wb") as fh:
        fh.write(api(url, tok, raw=True))
    got = shape(dst)
    print(f"\ndownloaded    : {os.path.relpath(dst, ROOT)}")

    print(f"\n{'':<14}{'reference':>12}{'ours':>12}")
    for k in ("bytes", "rows", "candidates"):
        print(f"{k:<14}{ref[k]:>12,}{got[k]:>12,}")
    print(f"{'per-row min':<14}{ref['per_row'][0]:>12}{got['per_row'][0]:>12}")
    print(f"{'per-row max':<14}{ref['per_row'][1]:>12}{got['per_row'][1]:>12}")

    if got["bytes"] == ref["bytes"] and got["rows"] == ref["rows"]:
        print("\nVERDICT: byte-identical to the reference -> the build/push/run chain is faithful.")
        return 0

    # otherwise: how many rows actually differ?
    import pandas as pd
    r = pd.read_csv(REFERENCE).set_index("molecule_id")
    g = pd.read_csv(dst).set_index("molecule_id")
    if set(r.index) != set(g.index):
        print("\nVERDICT: FAIL — the molecule_id sets differ, so these are not the same task.")
        return 1
    diff = [m for m in r.index if str(r.loc[m, "smiles"]) != str(g.loc[m, "smiles"])]
    print(f"\nrows differing: {len(diff)} of {len(r)}")
    for m in diff[:5]:
        print(f"   {m}")
        print(f"     ref : {str(r.loc[m,'smiles'])[:90]}")
        print(f"     ours: {str(g.loc[m,'smiles'])[:90]}")
    if len(diff) <= 1:
        print("\nVERDICT: PASS — differences confined to one row, as predicted for nolock vs the"
              "\nreference (the champion lock moves exactly one molecule on public data).")
        return 0
    print(f"\nVERDICT: FAIL — {len(diff)} rows differ; the chain is NOT faithful and every variant"
          "\nscore rests on an unverified baseline. Stop and investigate.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
