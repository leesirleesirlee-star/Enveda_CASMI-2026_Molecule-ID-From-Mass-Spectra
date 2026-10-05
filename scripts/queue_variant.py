"""
Queue one variant: wait for a free GPU slot, push it, wait for its run, submit, report the score.

Why this exists: every account gets only 2 concurrent GPU sessions, a pushed
version immediately occupies one, and a competition submission is refused until
that run is COMPLETE. So variants must be serialised behind whatever is already
running, and the waiting is pure latency.

Fail-closed: it submits only if the run reached COMPLETE *and* the declared
output file is present, and it submits at most once.

Usage:
  python queue_variant.py --folder notebooks/v45/claw --desc "V45 claw: ..." \
      [--after owner/slug] [--timeout-h 8]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

os.environ.setdefault("KAGGLE_API_TOKEN",
                      open(r"D:\CASMI竞赛\.secrets\kaggle\access_token").read().strip())
TOK = os.environ["KAGGLE_API_TOKEN"]
COMP = "enveda-CASMI26-molecule-id-mass-spectra"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def api(url, payload=None, method="GET"):
    req = urllib.request.Request(
        url, method=method,
        headers={"Authorization": "Bearer " + TOK, "Content-Type": "application/json"},
        data=json.dumps(payload).encode() if payload is not None else None)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, r.read().decode(errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")
    except Exception as e:
        return "ERR", f"{type(e).__name__}: {e}"


def status_of(owner, slug):
    st, body = api(f"https://www.kaggle.com/api/v1/kernels/status?"
                   f"userName={owner}&kernelSlug={slug}")
    if st != 200:
        return f"http{st}"
    try:
        return json.loads(body).get("status", "?")
    except Exception:
        return "?"


def wait_complete(owner, slug, deadline, poll, label):
    last = None
    while time.time() < deadline:
        s = status_of(owner, slug)
        if s != last:
            log(f"{label}: status={s}")
            last = s
        if "COMPLETE" in s.upper():
            return True
        if "ERROR" in s.upper() or "CANCEL" in s.upper():
            log(f"{label}: run ended as {s}")
            return False
        time.sleep(poll)
    log(f"{label}: timed out")
    return False


def output_files(owner, slug):
    st, body = api(f"https://www.kaggle.com/api/v1/kernels/output?"
                   f"userName={owner}&kernelSlug={slug}")
    if st != 200:
        return []
    try:
        return [f.get("fileName") for f in json.loads(body).get("files", [])]
    except Exception:
        return []


def score_for(desc, deadline, poll):
    while time.time() < deadline:
        st, body = api(f"https://www.kaggle.com/api/v1/competitions/submissions/list/{COMP}")
        if st == 200:
            try:
                me = next((s for s in json.loads(body) if s.get("description") == desc), None)
                if me and me.get("publicScore"):
                    return me
            except Exception:
                pass
        time.sleep(poll)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--desc", required=True)
    ap.add_argument("--after", default=None,
                    help="owner/slug to wait on before pushing (frees the GPU slot)")
    ap.add_argument("--file", default="submission.csv")
    ap.add_argument("--timeout-h", type=float, default=8.0)
    ap.add_argument("--poll-s", type=int, default=180)
    a = ap.parse_args()
    deadline = time.time() + a.timeout_h * 3600

    if a.after:
        o, s = a.after.split("/")
        log(f"waiting for a free slot on {a.after}")
        wait_complete(o, s, deadline, a.poll_s, "slot")

    meta = json.load(open(os.path.join(a.folder, "kernel-metadata.json"), encoding="utf-8"))
    owner, slug = meta["id"].split("/")

    from kaggle.api.kaggle_api_extended import KaggleApi
    kapi = KaggleApi()
    kapi.authenticate()
    log(f"pushing {a.folder}")
    resp = kapi.kernels_push(a.folder)
    if resp is None or getattr(resp, "error", None):
        log(f"ABORT: push failed: {getattr(resp, 'error', 'no response')}")
        raise SystemExit(3)
    ver = resp.versionNumber
    log(f"pushed {owner}/{slug} v{ver}")

    if not wait_complete(owner, slug, deadline, a.poll_s, f"v{ver}"):
        raise SystemExit(4)
    files = output_files(owner, slug)
    log(f"output files: {files[:8]}{' ...' if len(files) > 8 else ''}")
    if a.file not in files:
        log(f"ABORT: {a.file} missing, refusing to submit")
        raise SystemExit(5)

    st, body = api("https://api.kaggle.com/v1/competitions.CompetitionApiService/"
                   "CreateCodeSubmission",
                   {"fileName": a.file, "competitionName": COMP,
                    "kernelOwner": owner, "kernelSlug": slug,
                    "kernelVersion": ver, "submissionDescription": a.desc}, "POST")
    log(f"submit -> HTTP {st}: {body[:200]}")
    if st != 200:
        raise SystemExit(6)

    me = score_for(a.desc, deadline, a.poll_s)
    if me:
        log(f"SCORE [{a.desc}] public={me['publicScore']} bytes={me.get('totalBytes')} "
            f"status={me.get('status')} ref={me.get('ref')} err={me.get('errorDescription')!r}")
    else:
        log("score not available before the deadline")


if __name__ == "__main__":
    main()
