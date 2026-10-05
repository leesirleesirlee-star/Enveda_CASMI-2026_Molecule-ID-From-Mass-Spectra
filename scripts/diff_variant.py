"""Exact diff between a built variant and the V44 base notebook it was derived from."""
import difflib
import json
import sys
import os

# repository root, derived from this file so the tree is relocatable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASE = os.path.join(ROOT, "notebooks/v44_base/notebook.ipynb")
VARIANT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "notebooks/v45/claw/notebook.ipynb")

b = json.load(open(BASE, encoding="utf-8"))
v = json.load(open(VARIANT, encoding="utf-8"))
assert len(b["cells"]) == len(v["cells"]), "cell counts differ"

print(f"base    : {BASE}")
print(f"variant : {VARIANT}")
print()
changed = []
for i, (cb, cv) in enumerate(zip(b["cells"], v["cells"])):
    sb = "".join(cb.get("source", []))
    sv = "".join(cv.get("source", []))
    if sb != sv:
        changed.append(i)
print(f"cells changed: {changed}  (of {len(b['cells'])}; types "
      f"{[v['cells'][i].get('cell_type', '?') for i in changed]})")
print()

for i in changed:
    sb = "".join(b["cells"][i].get("source", [])).splitlines()
    sv = "".join(v["cells"][i].get("source", [])).splitlines()
    print(f"########## CELL {i}  ({len(sb)} -> {len(sv)} lines)")
    d = list(difflib.unified_diff(sb, sv, lineterm="", n=1))
    shown = 0
    for ln in d[2:]:
        if ln.startswith(("---", "+++")):
            continue
        # the inlined patch is thousands of chars on one line; summarise it
        if len(ln) > 220:
            ln = ln[:220] + f"   ...[{len(ln)} chars total]"
        print(ln)
        shown += 1
        if shown > 60:
            print("   ... (truncated)")
            break
    print()
