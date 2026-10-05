"""
Tidy the repository root: move loose top-level documents into docs/ and give the
non-ASCII names English ones.

Done in Python rather than PowerShell on purpose: these filenames are non-ASCII and
an earlier Get-Content/Set-Content round-trip double-encoded a file into mojibake.
subprocess passes the names through as bytes and git does the moving (so history is
preserved), instead of us re-writing files by hand.

Usage: python scripts/tidy_repo.py [--apply]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (old path relative to repo root, new path)
MOVES = [
    ("CASMI竞赛详细说明.md", "docs/competition_brief.md"),
    ("inpiration_1.md", "docs/inspiration_1.md"),          # also fixes the typo
    ("architecture_review.md", "docs/architecture_review.md"),
    ("prd patch.md", "docs/prd_patch.md"),
    ("CASMI_2026_PRD_v2.1.docx.docx", "docs/CASMI_2026_PRD_v2.1.docx"),  # double extension
]

# prose references that would otherwise point at a path that no longer exists
REFS = [
    ("scripts/verify_formula_claim.py", "architecture_review.md", "docs/architecture_review.md"),
    ("scripts/build_candidate_pool.py", "PRD patch §2.1", "docs/prd_patch.md §2.1"),
    ("scripts/build_notebook.py", "PRD patch:", "docs/prd_patch.md:"),
]


def run(args):
    # core.quotePath=false: without it git escapes non-ASCII paths as octal, and a
    # filename like CASMI竞赛详细说明.md never matches its real name.
    p = subprocess.run(["git", "-c", "core.quotePath=false"] + args,
                       cwd=ROOT, capture_output=True)
    return p.returncode, p.stdout.decode("utf-8", "replace"), p.stderr.decode("utf-8", "replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="actually move (default: dry run)")
    a = ap.parse_args()

    rc, out, _ = run(["ls-files"])
    tracked = set(out.splitlines())
    print(f"tracked files: {len(tracked)}")

    planned = []
    for old, new in MOVES:
        if old not in tracked:
            print(f"  SKIP  {old!r} is not tracked (already moved?)")
            continue
        if new in tracked:
            print(f"  SKIP  target {new!r} already tracked")
            continue
        planned.append((old, new))

    print(f"\nplanned moves ({len(planned)}):")
    for old, new in planned:
        print(f"  {old}  ->  {new}")

    hits = 0
    print(f"\nprose references to update ({len(REFS)}):")
    for path, old, new in REFS:
        full = os.path.join(ROOT, path)
        if not os.path.exists(full):
            print(f"  SKIP  {path} missing")
            continue
        s = open(full, encoding="utf-8").read()
        n = s.count(old)
        print(f"  {path}: {n} occurrence(s) of {old!r}")
        hits += n

    if not a.apply:
        print("\ndry run; re-run with --apply to execute")
        return 0

    print("\napplying...")
    for old, new in planned:
        rc, _, err = run(["mv", old, new])
        print(("  ok   " if rc == 0 else "  FAIL ") + f"{old} -> {new}" + (f"  {err.strip()}" if rc else ""))

    for path, old, new in REFS:
        full = os.path.join(ROOT, path)
        if not os.path.exists(full):
            continue
        s = open(full, encoding="utf-8").read()
        if old in s:
            open(full, "w", encoding="utf-8", newline="").write(s.replace(old, new))
            print(f"  updated refs in {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
