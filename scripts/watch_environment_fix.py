"""
Wait for the pinned-image verification run and report the verdict, so nobody has to poll.

The gated chain (run_gated_variant.py) already applies this test internally, but it only
finishes hours later -- after the variant's own run, its submission and the hidden rerun. This
watcher answers the one question that matters much sooner: did pinning the docker image bring
ICEBERG and GLACIER back to life?

It reuses run_gated_variant.gate, so the acceptance test has a single definition:
    ICE meta status == "ok"   AND   ICE rerank stats molecules > 0

Usage:
  python scripts/watch_environment_fix.py --kernel nicholasnicklee/casmi26-v45-ctl --timeout-h 6
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_spec = importlib.util.spec_from_file_location(
    "_gated", os.path.join(ROOT, "scripts", "run_gated_variant.py"))
_gated = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gated)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--timeout-h", type=float, default=6)
    a = ap.parse_args()

    end = time.time() + a.timeout_h * 3600
    last = None
    while time.time() < end:
        st = _gated.status(a.kernel)
        if st != last:
            print(f"[{time.strftime('%H:%M:%S')}] {a.kernel}: {st}", flush=True)
            last = st
        if st in ("complete", "error", "cancelled"):
            break
        time.sleep(60)

    data = _gated.outputs(a.kernel) or {}
    log = data.get("log") or ""
    print(f"log chars: {len(log)}  output files: {len(data.get('files', []))}", flush=True)
    ok, reason = _gated.gate(log)
    verdict = ("PASS" if ok else "FAIL") + ": " + reason
    print(f"\nENVIRONMENT GATE — {verdict}\n", flush=True)
    open(os.path.join(ROOT, ".deepworks", "tmp", "gate_verdict.txt"), "w",
         encoding="utf-8").write(verdict + "\n")
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
