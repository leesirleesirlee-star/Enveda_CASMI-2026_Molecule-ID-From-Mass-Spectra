"""Freeze the v17 popularity patch (probe_core2 half of the CLAW rule) into the repo.

Extracted from bobthebot369/enveda-casmi-2026-v17-zenith-apex (which is the same
release as lehau007/casmi26-sota-v27-zenith-apex-0417, LB 0.417) so the port is a
copy rather than a retype. The patch redefines init_worker/probe_one at the end of
probe_core2, so appending it to our CORE is enough to activate it.
"""
import os

# repository root, derived from this file so the tree is relocatable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SRC = os.path.join(ROOT, ".deepworks/tmp/v17/pc__probe_core2.py")
DST = os.path.join(ROOT, "notebooks/v45/_claw_patch.py")

src = open(SRC, encoding="utf-8").read()
marker = "# ---- v17 patch:"
i = src.find(marker)
assert i > 0, "v17 patch marker not found"
patch = src[i:]

# it must be the *tail* of the module: appending it after our own definitions is
# what makes the redefinition take effect
for need in ("def init_worker(", "def probe_one(", "fz_top", "'S'", "top_pop",
             "_POP_UNION", "_POP_LAM"):
    assert need in patch, f"patch is missing {need!r}"
assert "def probe_one" in patch and patch.count("def probe_one") == 1

open(DST, "w", encoding="utf-8", newline="\n").write(patch)
print(f"wrote {DST}: {len(patch)} chars, {len(patch.splitlines())} lines")
print("--- first 3 lines ---")
print("\n".join(patch.splitlines()[:3]))
