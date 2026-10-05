"""
Build ablation variants of the reproduced V44 kernel.

Why: the whole public field sits at exactly 0.417 because everyone runs the same
stack. We only need a small, *real* gain to pass that tie cluster (rank 59 is
0.418), so the job is to test one controlled change at a time. Each variant is
pushed as its own kernel version, and one competition submission scores it.

Every substitution is asserted to match exactly once: a silent no-op edit would
make an ablation meaningless, which is the failure mode this script exists to
prevent.

Usage:
  python build_variant_kernel.py --variant icefull [--out notebooks/v45]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = r"D:\CASMI竞赛"
SRC = os.path.join(ROOT, "notebooks", "v44_base", "notebook.ipynb")
COMP = "enveda-CASMI26-molecule-id-mass-spectra"
DATASETS = [
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
]

# cell 7 embeds eng_runner.py inside a string literal; these two knobs live there
ENGINE_TOP1_SHIELD = "top = int(np.argmax(base)); score[top] = float(np.max(score)) + 1.0"
ENGINE_PAIR_W = "pair_weight = 0.15"
ICE_LINE = "TOPN, ICE_LAM, ICE_BUDGET, ICE_PC = 60, 1.0, 300, True"
SLOTS_LINE = "SLOTS_AGG, SLOTS_GENTLE = [2, 4, 6, 8, 10], [4, 8, 12, 16, 20]"
RELTH_LINE = "LIB_TAU, REL_TH = 0.9, 600.0"

NOLOCK_CELL = """# V45 ablation: champion top-1 lock DISABLED.
# The V44 lock dict is keyed on the *visible* test's molecule_ids; if the rerun
# test differs that lookup raises KeyError and the notebook dies before the final
# validation cell. Disabling it both removes that failure mode and measures how
# much the 0.411 V29 predictions actually contribute at rerun.
print('V45: champion top-1 lock disabled')
"""

VARIANTS: dict[str, list[tuple[int, str, str]]] = {
    "ctl": [],
    "nolock": [],
    "icefull": [(3, ICE_LINE, "TOPN, ICE_LAM, ICE_BUDGET, ICE_PC = 60, 1.0, 3000, True")],
    "unlock_engine": [
        (7, ENGINE_PAIR_W, "pair_weight = 0.35"),
        (7, ENGINE_TOP1_SHIELD, "pass  # V45: engine top-1 shield removed"),
    ],
    "pc_aggressive": [
        (3, SLOTS_LINE,
         "SLOTS_AGG, SLOTS_GENTLE = [2, 3, 4, 5, 6, 8, 10, 12, 14, 16], "
         "[3, 5, 7, 9, 11, 13, 15, 17, 19, 21]"),
        (3, RELTH_LINE, "LIB_TAU, REL_TH = 0.9, 200.0"),
    ],
}

# day-2 combos
VARIANTS["combo_a"] = VARIANTS["icefull"] + [
    (7, ENGINE_PAIR_W, "pair_weight = 0.35"),
    (7, ENGINE_TOP1_SHIELD, "pass  # V45: engine top-1 shield removed"),
]
VARIANTS["combo_b"] = VARIANTS["icefull"] + VARIANTS["pc_aggressive"]
VARIANTS["topn120"] = [
    (3, ICE_LINE, "TOPN, ICE_LAM, ICE_BUDGET, ICE_PC = 120, 1.0, 3600, True"),
]


def apply(nb: dict, variant: str) -> dict:
    edits = VARIANTS[variant]
    if variant == "nolock" or variant == "ctl":
        pass
    cells = nb["cells"]
    for idx, old, new in edits:
        src = "".join(cells[idx]["source"])
        n = src.count(old)
        assert n == 1, (f"variant {variant}: cell {idx} matched {n} times for {old[:60]!r} "
                        f"- a silent no-op edit would invalidate the ablation")
        src = src.replace(old, new)
        cells[idx]["source"] = src.splitlines(keepends=True)
        print(f"  cell {idx}: {old[:58]!r} -> {new[:58]!r}")
    if variant == "nolock":
        assert "_champion_top1" in "".join(cells[31]["source"])
        cells[31]["source"] = NOLOCK_CELL.splitlines(keepends=True)
        print("  cell 31: champion lock replaced with no-op")
    return nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    ap.add_argument("--out", default=os.path.join(ROOT, "notebooks", "v45"))
    ap.add_argument("--slug", default="casmi26-v45-ablation")
    ap.add_argument("--title", default="CASMI26 V45 Ablation")
    a = ap.parse_args()

    nb = json.loads(open(SRC, encoding="utf-8").read())
    print(f"variant {a.variant}: {len(VARIANTS[a.variant])} edits")
    nb = apply(nb, a.variant)
    nb.setdefault("metadata", {})["casmi_variant"] = f"v45-{a.variant}"

    out = os.path.join(a.out, a.variant)
    os.makedirs(out, exist_ok=True)
    nb_path = os.path.join(out, "notebook.ipynb")
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False)
    meta = {
        "id": f"nicholasnicklee/{a.slug}",
        "title": a.title,
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "dataset_sources": DATASETS,
        "competition_sources": [COMP],
        "kernel_sources": [],
        "model_sources": [],
    }
    meta_path = os.path.join(out, "kernel-metadata.json")
    with open(meta_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"wrote {nb_path} ({os.path.getsize(nb_path):,} bytes) and {meta_path}")

    # self-check: the variant really differs from the control in the intended way
    ctl = json.loads(open(SRC, encoding="utf-8").read())
    diff = sum(1 for i, (x, y) in enumerate(zip(ctl["cells"], nb["cells"]))
               if "".join(x["source"]) != "".join(y["source"]))
    print(f"self-check: {diff} cell(s) differ from the control")
    assert (diff > 0) == (a.variant != "ctl"), "variant is not distinguishable from control"


if __name__ == "__main__":
    main()
