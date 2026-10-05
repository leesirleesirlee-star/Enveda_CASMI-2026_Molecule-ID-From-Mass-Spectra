"""
Push a built kernel variant and report the new version number.

Pushing only *saves* a version (save_kernel); the competition rerun is what
actually executes it, so this costs no GPU quota.

Usage:
  python push_kernel.py notebooks/v45/icefull [more folders ...]
"""

from __future__ import annotations

import os
import sys

# repository root, derived from this file so the tree is relocatable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("KAGGLE_API_TOKEN",
                      open(os.path.join(ROOT, ".secrets/kaggle/access_token")).read().strip())

from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: E402

def main():
    folders = sys.argv[1:]
    if not folders:
        print(__doc__)
        raise SystemExit(2)
    api = KaggleApi()
    api.authenticate()
    for folder in folders:
        meta = os.path.join(folder, "kernel-metadata.json")
        assert os.path.exists(meta), f"missing {meta}"
        print(f"--- pushing {folder}")
        resp = api.kernels_push(folder)
        if resp is None:
            print("    FAILED (no response)")
            continue
        if getattr(resp, "error", None):
            print("    ERROR:", resp.error)
            continue
        print(f"    OK version={resp.versionNumber} url={getattr(resp, 'url', '')}")
        for attr in ("invalidDatasetSources", "invalidCompetitionSources",
                     "invalidKernelSources", "invalidTags"):
            v = getattr(resp, attr, None)
            if v:
                print(f"    !! {attr}: {v}")


if __name__ == "__main__":
    main()
