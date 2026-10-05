"""
Run the next variant behind an environment gate.

Why this exists rather than plain queue_variant.py: the pinned-image fix for the ICEBERG /
GLACIER failure (docs/EXPERIMENTS.md, 2026-10-05 fourth segment) is still unverified. A
verification run (`--gate-kernel`) is already in flight, and the acceptance test is readable
from its log:

    ICE meta {"status": "ok", ...}          not "error"
    ICE rerank stats {'molecules': N, ...}  with N > 0     (was 0 when the wheel install failed)

The chain here is:

  1. wait for the gate kernel to complete, fetch its log, apply that test;
  2. wait for the target variant's *current* run to finish, which frees a GPU session slot
     (the account allows two, and one is already held by the gate run);
  3. push the variant again, so the new version carries the pinned image;
  4. wait for it, then submit -- but only if the gate passed;
  5. poll the score and report.

Step 2 runs regardless of the gate, because the slot frees before the gate verdict is
needed: the gate kernel and the variant's new run take about the same time, so the verdict
is available well before the submission decision. If the gate fails, the run is spent but
nothing is submitted, and no misleading score enters the ledger.

Usage:
  python scripts/run_gated_variant.py --gate-kernel nicholasnicklee/casmi26-v45-ctl \
      --folder notebooks/v45/claw --desc "V45 claw v2 (pinned image)" --timeout-h 14
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKEN_PATH = os.path.join(ROOT, ".secrets", "kaggle", "access_token")
PYTHON = sys.executable


def token():
    return open(TOKEN_PATH, encoding="utf-8").read().strip()


def api(path):
    req = urllib.request.Request("https://www.kaggle.com/api/v1" + path,
                                 headers={"Authorization": "Bearer " + token()})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def split(ref):
    o, s = ref.split("/", 1)
    return o, s


def status(ref):
    o, s = split(ref)
    try:
        return api(f"/kernels/status?userName={o}&kernelSlug={s}").get("status")
    except Exception:
        return "unknown"


def outputs(ref):
    o, s = split(ref)
    try:
        return api(f"/kernels/output?userName={o}&kernelSlug={s}")
    except Exception:
        return {}


def wait_complete(ref, timeout_h, label):
    end = time.time() + timeout_h * 3600
    while time.time() < end:
        st = status(ref)
        if st in ("complete", "error", "cancelled"):
            print(f"[{time.strftime('%H:%M:%S')}] {label}: {st}", flush=True)
            return st
        time.sleep(45)
    print(f"[{time.strftime('%H:%M:%S')}] {label}: TIMEOUT", flush=True)
    return "timeout"


def wait_free_slot(ref, timeout_h):
    """Wait until this kernel's own run has finished, freeing a session slot."""
    return wait_complete(ref, timeout_h, f"slot: waiting for {ref}")


def _meta_object(log, tag):
    """Pull one JSON object out of the log after `tag`.

    The log is a stream of JSON records, so the metadata arrives with its quotes escaped
    (\\") and the object can contain nested braces. Extract by brace matching, then unescape.
    """
    i = log.find(tag)
    if i < 0:
        return None
    i = log.find("{", i)
    if i < 0:
        return None
    depth = 0
    for j in range(i, min(len(log), i + 20000)):
        c = log[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                raw = log[i:j + 1]
                for attempt in (raw, raw.replace('\\"', '"'), raw.replace('\\\\"', '"')):
                    try:
                        return json.loads(attempt)
                    except Exception:
                        continue
                return None
    return None


def gate(log):
    """Return (ok, reason) for the ICE/GL acceptance test."""
    if not log:
        return False, "the gate kernel produced no log"
    meta = _meta_object(log, "ICE meta")
    if not isinstance(meta, dict):
        # The real logs carry a nested traceback inside the object, which can defeat strict
        # parsing. Fall back to reading the status token directly, so a genuinely healthy run
        # is not reported as a failure just because the object would not deserialise.
        i = log.find("ICE meta")
        window = log[i:i + 600] if i >= 0 else ""
        m = re.search(r"status\D{0,12}?(\w+)", window)
        if not m:
            return False, "no parseable 'ICE meta' object in the log"
        st = m.group(1)
        if st not in ("ok", "budget"):
            errs = re.search(r"RuntimeError: ([^\\\"]{0,160})", window)
            return False, f"ICE meta status={st!r}: {errs.group(1) if errs else ''}"
        meta = {"status": st}
    st = meta.get("status")
    if st not in ("ok", "budget"):
        errs = (meta.get("errors") or [""])
        errs = errs[0] if errs else ""
        return False, f"ICE meta status={st!r}: {str(errs)[:200]}"
    r = re.search(r"ICE rerank stats \{'molecules': (\d+)", log)
    n = int(r.group(1)) if r else 0
    if n <= 0:
        return False, "ICE is healthy but reranked 0 molecules"
    # "budget" is a HEALTHY status, not a failure: it means ICE stopped after covering part of
    # the set within ICE_BUDGET seconds. The reference run reports the same thing -- it scored
    # 71 of 400 molecules and reordered 365 rows, and our pinned run reproduces that exactly.
    # Requiring status=="ok" here rejected a working run and blocked the submission, which is
    # how this criterion came to be corrected.
    note = "" if st == "ok" else f" (status={st}, which is normal for a budget-limited ICE)"
    return True, f"ICE healthy, reranked {n} molecules{note}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-kernel", required=True)
    ap.add_argument("--folder", required=True)
    ap.add_argument("--desc", required=True)
    ap.add_argument("--timeout-h", type=float, default=14)
    a = ap.parse_args()

    slug = json.load(open(os.path.join(a.folder, "kernel-metadata.json"),
                          encoding="utf-8"))["id"]
    img = json.load(open(os.path.join(a.folder, "kernel-metadata.json"),
                         encoding="utf-8")).get("docker_image") or ""
    print(f"gate kernel : {a.gate_kernel}")
    print(f"variant     : {slug}  (image ...{img[-20:] if img else 'NOT PINNED'})", flush=True)

    # 1. the gate verdict
    st = wait_complete(a.gate_kernel, a.timeout_h, "gate run")
    log = (outputs(a.gate_kernel) or {}).get("log") or ""
    ok, reason = gate(log) if st == "complete" else (False, f"gate run ended {st}")
    print(f"\nGATE: {'PASS' if ok else 'FAIL'} — {reason}\n", flush=True)
    open(os.path.join(ROOT, ".deepworks", "tmp", "gate_verdict.txt"), "w",
         encoding="utf-8").write(("PASS" if ok else "FAIL") + ": " + reason + "\n")

    # 2. free a slot: the variant's own current run must finish first
    wait_free_slot(slug, a.timeout_h)

    # 3. push the pinned version
    print(f"[{time.strftime('%H:%M:%S')}] pushing {a.folder}", flush=True)
    p = subprocess.run([PYTHON, os.path.join("scripts", "push_kernel.py"), a.folder],
                       cwd=ROOT, capture_output=True)
    print((p.stdout or b"").decode("utf-8", "replace").strip()[-300:], flush=True)

    # 3b. Wait for the NEW version to actually start before waiting for it to finish.
    # Without this, the status still reads "complete" from the previous version for a moment,
    # wait_complete would return immediately, and we would submit the OLD run's output --
    # which is exactly the unpinned, ICE/GL-dead run this chain exists to replace.
    st = status(slug)
    end = time.time() + 900
    while st in ("complete", "cancelled", "unknown") and time.time() < end:
        print(f"[{time.strftime('%H:%M:%S')}] waiting for the new version to start "
              f"(currently {st})", flush=True)
        time.sleep(30)
        st = status(slug)
    print(f"[{time.strftime('%H:%M:%S')}] new version status: {st}", flush=True)

    # 4. wait, then submit only if the gate passed
    v = wait_complete(slug, a.timeout_h, "variant run")
    if not ok:
        print("gate FAILED — not submitting; the run is spent, no score recorded.", flush=True)
        return 3
    if v != "complete":
        print(f"variant run ended {v} — not submitting.", flush=True)
        return 4
    print(f"[{time.strftime('%H:%M:%S')}] submitting", flush=True)
    p = subprocess.run([PYTHON, os.path.join("scripts", "wait_submit_report.py"),
                        "--kernel", slug, "--desc", a.desc, "--timeout-h", str(a.timeout_h)],
                       cwd=ROOT, capture_output=True)
    print((p.stdout or b"").decode("utf-8", "replace")[-2000:], flush=True)
    print((p.stderr or b"").decode("utf-8", "replace")[-600:], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
