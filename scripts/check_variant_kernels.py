"""Compile every code cell of each built variant: catch syntax errors before a 2 h GPU run."""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# A missing dataset_source costs a full ~5 h run before anyone notices, and it
# fails at import time deep inside the pipeline. Check the metadata too.
REQUIRED_DATASETS = {
    "prvsiyan/casmi26-fp-models-v2",
    "dmitriigluzdov/casmi26-pubchem-popularity-prior",
    "prvsiyan/casmi26-ranker-features",
    "megayak/casmi26-simulated-ranker-rows",
    "ahmedberatozer/casmi26-fpnet-full1",
    "ahmedberatozer/casmi26-glacier",
    "ahmedberatozer/casmi26-iceberg",
    "ahmedberatozer/casmi26-pubchem-tier",
    "ahmedberatozer/casmi26-v2-pool",
    "ahmedberatozer/casmi26-v3-models",
    "ahmedberatozer/casmi26-v4b-models",
    "prvsiyan/chebi-lipidmaps-casmi26",
    "prvsiyan/coconut-casmi26-candidates",
    "metric/rdkit-2026-3-3-wheel",
}
COMPETITION = "enveda-CASMI26-molecule-id-mass-spectra"

bad = 0
for nb in sorted(glob.glob(os.path.join(ROOT, "notebooks", "v45", "*", "notebook.ipynb"))):
    d = json.loads(open(nb, encoding="utf-8").read())
    name = os.path.basename(os.path.dirname(nb))

    meta_path = os.path.join(os.path.dirname(nb), "kernel-metadata.json")
    meta = json.loads(open(meta_path, encoding="utf-8").read())
    ds = set(meta.get("dataset_sources") or [])
    problems = []
    if REQUIRED_DATASETS - ds:
        problems.append(f"missing datasets {sorted(REQUIRED_DATASETS - ds)}")
    if ds - REQUIRED_DATASETS:
        problems.append(f"unexpected datasets {sorted(ds - REQUIRED_DATASETS)}")
    if not meta.get("enable_gpu"):
        problems.append("enable_gpu is false")
    if (meta.get("competition_sources") or []) != [COMPETITION]:
        problems.append(f"competition_sources={meta.get('competition_sources')}")
    if problems:
        bad += 1
        print(f"FAIL {name}: " + "; ".join(problems))
        continue

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
