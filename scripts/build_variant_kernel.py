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

CLAW_PATCH_PATH = os.path.join(ROOT, "notebooks", "v45", "_claw_patch.py")


def claw_patch() -> str:
    """The v17 popularity probe (CLAW's S / top_pop / fz_top producers)."""
    body = open(CLAW_PATCH_PATH, encoding="utf-8").read()
    assert "'''" not in body and '"""' not in body, \
        "patch contains a triple quote and would break the string literal it is inlined into"
    assert "\\" not in body, "patch contains a backslash; escape it before inlining"
    for need in ("def init_worker(", "def probe_one(", "fz_top", "top_pop"):
        assert need in body, f"CLAW patch lacks {need!r}"
    return body


# cell 9: persist the three CLAW quantities that pc_runner currently throws away.
# NOTE: this text lands *inside* a single-quoted string literal, so it must not
# contain a bare single quote - hence the double quotes around the dict keys.
CLAW_DIAG = (
    "res[mid] = dict(pc=smis, pc_fz=fzs, pc_keys=keys, **extra[mid])",
    'res[mid] = dict(pc=smis, pc_fz=fzs, pc_keys=keys, fz_top=d.get("fz_top"), '
    'S=d.get("S"), top_pop=d.get("top_pop", 0.0), **extra[mid])',
)
# cell 9: append the v17 probe so the gated quantities actually get computed
CLAW_APPEND = (
    "open('/kaggle/working/probe_core2.py', 'w').write(CORE)",
    "_CLAW_PATCH = '''" + claw_patch() + "'''\n"
    "CORE = CORE + _CLAW_PATCH\n"
    "open('/kaggle/working/probe_core2.py', 'w').write(CORE)",
)
# cell 3: the gate constants
CLAW_CONST = (
    "TOPN, ICE_LAM, ICE_BUDGET, ICE_PC = 60, 1.0, 300, True",
    "TOPN, ICE_LAM, ICE_BUDGET, ICE_PC = 60, 1.0, 300, True\n"
    "USE_PROMOTION, S_TAU, POP_TAU = True, 6.0, 5.0  # V45 CLAW (lehau007 v27 / bobthebot369 v17)",
)
# cell 19: the promotion helper
CLAW_PROMOTE = (
    "    return out\nrows, stats = [], dict(untouched=0, gentle=0, aggressive=0, no_pc=0)",
    "    return out\n\n\n"
    "def promote(base, base_keys, pc, pc_keys, slots, n=25):\n"
    "    \"\"\"CLAW b417: a confident PubChem-only proposal leads; every other entry keeps its slot.\"\"\"\n"
    "    usual = merge(base, base_keys, pc, pc_keys, slots, n=n)\n"
    "    if not pc:\n"
    "        return usual\n"
    "    top = pc[0]\n"
    "    return ([top] + [x for x in usual if x != top])[:n]\n\n\n"
    "rows, stats = [], dict(untouched=0, gentle=0, aggressive=0, no_pc=0)",
)
# cell 19: the gate itself (replaces the unconditional merge)
CLAW_GATE = (
    """        rel = p['pc_fz'][0] - p['best_pool_fz'] if np.isfinite(p['best_pool_fz']) else 1e9
        slots = SLOTS_AGG if rel > REL_TH else SLOTS_GENTLE
        stats['aggressive' if rel > REL_TH else 'gentle'] += 1
        final = merge(smis, keys, p['pc'], p['pc_keys'], slots)""",
    """        _bp = p.get('best_pool_fz')
        _tf = p.get('fz_top') if p.get('fz_top') is not None else (p['pc_fz'][0] if p.get('pc_fz') else None)
        rel = (_tf - _bp) if (_tf is not None and _bp is not None and np.isfinite(_bp)) else 1e9
        slots = SLOTS_AGG if rel > REL_TH else SLOTS_GENTLE
        stats['aggressive' if rel > REL_TH else 'gentle'] += 1
        _S = p.get('S'); _pt = float(p.get('top_pop', 0.0) or 0.0)
        if USE_PROMOTION and _S is not None and float(_S) > S_TAU and _pt >= POP_TAU:
            stats['promoted'] = stats.get('promoted', 0) + 1
            final = promote(smis, keys, p['pc'], p['pc_keys'], slots)
        else:
            final = merge(smis, keys, p['pc'], p['pc_keys'], slots)""",
)
POP_MU_LINE = "POP_LAM, POP_UNION, POOLPOP_MU = 0.25, 200, 0.15"

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

# --- CLAW: the field's only rank-1-moving lever (see docs/V45_消融手册.md) ---
CLAW_EDITS = [
    (3, CLAW_CONST[0], CLAW_CONST[1]),
    (9, CLAW_DIAG[0], CLAW_DIAG[1]),
    (9, CLAW_APPEND[0], CLAW_APPEND[1]),
    (19, CLAW_PROMOTE[0], CLAW_PROMOTE[1]),
    (19, CLAW_GATE[0], CLAW_GATE[1]),
]
VARIANTS["claw"] = CLAW_EDITS
VARIANTS["claw_pop30"] = CLAW_EDITS + [
    (3, POP_MU_LINE, "POP_LAM, POP_UNION, POOLPOP_MU = 0.25, 200, 0.30"),
]
VARIANTS["pop30"] = [
    (3, POP_MU_LINE, "POP_LAM, POP_UNION, POOLPOP_MU = 0.25, 200, 0.30"),
]

# --- gate bypass -------------------------------------------------------------
# The whole PubChem channel (2,674 s of GPU) sits behind `lib_max < LIB_TAU(0.9)`.
# V44's own engine comment says "V43's lib_max was 1.0 for every hidden query",
# which would disable the channel for *every* molecule - and with it CLAW, the
# only rank-1 lever we have. These variants test that directly: if the channel
# was already open they change nothing, if it was shut they open it.
FORCE_ICE_PC = (
    "            if p and p.get('pc') and BASE[m][2] < LIB_TAU:",
    "            if p and p.get('pc'):  # V45 force: lib_max gate disabled",
)
FORCE_MERGE = (
    "    elif lib_max >= LIB_TAU:",
    "    elif lib_max >= 9.9:  # V45 force: lib_max gate disabled (lib_max is <=1)",
)
FORCE_EDITS = [(17, FORCE_ICE_PC[0], FORCE_ICE_PC[1]),
               (19, FORCE_MERGE[0], FORCE_MERGE[1])]
VARIANTS["merge_force"] = FORCE_EDITS
VARIANTS["claw_force"] = CLAW_EDITS + FORCE_EDITS

# --- RRF weight: the only way the engine's rank-1 can ever win -----------------
# In cell 21 fuse2 gives the v4 list 1/(KRR+r) and the engine ALPHA/(KRR+r), then
# `sorted(sc, key=lambda k: -sc[k])`. At ALPHA=0.6 the v4 rank-1 scores 0.25 and
# the engine rank-1 scores 0.15, and because the sort is stable a tie still favours
# v4 (inserted first). So the engine's top-1 can only ever win for ALPHA > 1.0 -
# which is why v28's 0.6 -> 0.65 tweak tests nothing.
FUSE_LINE = "ALPHA, KRR = 0.6, 3.0"
VARIANTS["alpha12"] = [(21, FUSE_LINE, "ALPHA, KRR = 1.2, 3.0  # V45: engine rank-1 can outrank v4's")]

# --- use the runtime headroom -------------------------------------------------
# Every run of this stack uses ~2 h of the 9 h notebook limit, while the whole
# public field scores 0.417 with the same *shape* of stack. If the 0.42-0.47 teams
# differ mainly by how much forward-model scoring they do, the lever is depth:
# give ICEBERG/GLACIER a far deeper candidate list and a budget that actually
# covers it. topn120 is the mild version, this is the aggressive end.
GL_LINE = ("GL_SCORES, gl_stats, GL_LAM, GL_BUDGET = {}, "
           "dict(molecules=0, changed_vs_ice_top25=0, changed_vs_ice_top1=0), 1.0, 4000")
VARIANTS["bigcompute"] = [
    (3, ICE_LINE, "TOPN, ICE_LAM, ICE_BUDGET, ICE_PC = 200, 1.0, 5400, True"),
    (17, GL_LINE, GL_LINE.replace(", 1.0, 4000", ", 1.0, 6000")),
]

# --- recall expansion, tail-only ---------------------------------------------
# The 819-group probe (docs/V45_消融手册.md 8.6) says ~45% of hidden molecules
# never get the truth into the top-25 at all: the loss is half recall, and the
# structure pool cannot be widened (LOTUS/NPAtlas add 373 structures). PubChem is
# the only recall channel left, and today it may occupy just slots 2,4,6,8,10.
#
# Widening into the *head* risks demoting correct pool candidates, which is
# exactly what the field's gate exists to prevent. So this variant fills only
# slots 16-25: for the ~45% of molecules whose truth is absent those 10 slots are
# all wrong anyway, so displacing them costs nothing, while a hit at rank ~20 is
# worth +0.05 where the molecule currently scores 0.
TAIL_SLOTS = (
    "        slots = SLOTS_AGG if rel > REL_TH else SLOTS_GENTLE",
    "        slots = list(range(16, 26))  # V45: tail-only PubChem recall expansion",
)
VARIANTS["pc_tail10"] = [(19, TAIL_SLOTS[0], TAIL_SLOTS[1])]
VARIANTS["claw_tail10"] = CLAW_EDITS + [(19, TAIL_SLOTS[0], TAIL_SLOTS[1])]


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
