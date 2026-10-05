"""Compile every code cell of each built variant: catch syntax errors before a 2 h GPU run."""
import glob
import json
import os
import sys

ROOT = r"D:\CASMI竞赛"
bad = 0
for nb in sorted(glob.glob(os.path.join(ROOT, "notebooks", "v45", "*", "notebook.ipynb"))):
    d = json.loads(open(nb, encoding="utf-8").read())
    errs = []
    for i, c in enumerate(d["cells"]):
        if c.get("cell_type") != "code":
            continue
        src = "".join(c.get("source", []))
        if not src.strip():
            continue
        # notebook magics are not valid python; none should be present here
        assert not src.lstrip().startswith("!"), f"{nb} cell {i} has a shell magic"
        try:
            compile(src, f"<cell {i}>", "exec")
        except SyntaxError as e:
            errs.append(f"cell {i}: line {e.lineno}: {e.msg}")
    name = os.path.basename(os.path.dirname(nb))
    if errs:
        bad += 1
        print(f"FAIL {name}")
        for e in errs:
            print("   ", e)
        continue

    # For CLAW variants: actually assemble CORE the way the notebook does and
    # compile it, so an inlined patch cannot silently produce broken probe code.
    src9 = "".join(d["cells"][9].get("source", []))
    if "_CLAW_PATCH" in src9:
        head, sep, _ = src9.partition("open('/kaggle/working/probe_core2.py'")
        assert sep, "cell 9 no longer writes probe_core2.py the way we expect"
        ns: dict = {}
        exec(compile(head, "<cell9-head>", "exec"), ns)   # string assignments only
        core = ns.get("CORE", "")
        try:
            compile(core, "<assembled probe_core2>", "exec")
        except SyntaxError as e:
            bad += 1
            print(f"FAIL {name}: assembled probe_core2 is not valid python: "
                  f"line {e.lineno}: {e.msg}")
            continue
        for need in ("def probe_one(", "top_pop", "fz_top", "_POP_UNION"):
            assert need in core, (name, need)
        assert core.count("def probe_one(") == 2, \
            f"{name}: expected the patch to add a second probe_one definition"
        src19 = "".join(d["cells"][19].get("source", []))
        assert "def promote(" in src19 and "USE_PROMOTION" in src19, name
        print(f"ok   {name}  (CLAW: probe_core2 re-assembled {len(core)} chars, validates)")
    else:
        print(f"ok   {name}")
print("variants with errors:", bad)
sys.exit(1 if bad else 0)
