"""
Wait for a pushed kernel version to finish, submit it to the competition, then
report the score.

Why this shape: a Kaggle code-competition submission is rejected while the
notebook is still running ("Notebook is still running. Did not find provided
Notebook Output File."), and the submission itself then reruns the notebook on
the hidden test for ~2.5 h. Both waits are pure latency, so they are automated
here instead of blocking a session.

Fail-closed rules:
  * never submit unless the run reached COMPLETE *and* the declared output file
    exists in the kernel output listing
  * never retry a submission more than once (the daily quota is 5 per team)

Usage:
  python wait_submit_report.py --kernel owner/slug --version 2 \
      --desc "V45 nolock" [--file submission.csv] [--timeout-h 6]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

# repository root, derived from this file so the tree is relocatable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TOK = open(os.path.join(ROOT, ".secrets/kaggle/access_token")).read().strip()
COMP = "enveda-CASMI26-molecule-id-mass-spectra"
URL_SUBMIT = ("https://api.kaggle.com/v1/competitions.CompetitionApiService/"
              "CreateCodeSubmission")


def api(url, payload=None, method="GET", raw=False):
    req = urllib.request.Request(
        url, method=method,
        headers={"Authorization": "Bearer " + TOK, "Content-Type": "application/json"},
        data=json.dumps(payload).encode() if payload is not None else None)
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            body = r.read()
            return r.status, body if raw else body.decode(errors="replace")
    except urllib.error.HTTPError as e:
        body = e.read()
        return e.code, body if raw else body.decode(errors="replace")
    except Exception as e:
        return "ERR", b"" if raw else f"{type(e).__name__}: {e}"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def kernel_status(owner, slug):
    st, body = api(f"https://www.kaggle.com/api/v1/kernels/status?"
                   f"userName={owner}&kernelSlug={slug}")
    if st != 200:
        return f"http{st}"
    try:
        return json.loads(body).get("status", "?")
    except Exception:
        return "?"


def output_files(owner, slug):
    st, body = api(f"https://www.kaggle.com/api/v1/kernels/output?"
                   f"userName={owner}&kernelSlug={slug}")
    if st != 200:
        return []
    try:
        return [f.get("fileName") for f in json.loads(body).get("files", [])]
    except Exception:
        return []


def submissions():
    st, body = api(f"https://www.kaggle.com/api/v1/competitions/submissions/list/{COMP}")
    if st != 200:
        return []
    try:
        return json.loads(body)
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--version", type=int, required=True)
    ap.add_argument("--desc", required=True)
    ap.add_argument("--file", default="submission.csv")
    ap.add_argument("--timeout-h", type=float, default=6.0)
    ap.add_argument("--poll-s", type=int, default=180)
    a = ap.parse_args()
    owner, slug = a.kernel.split("/")
    deadline = time.time() + a.timeout_h * 3600

    log(f"waiting for {a.kernel} v{a.version} to finish (budget {a.timeout_h}h)")
    last = None
    while time.time() < deadline:
        s = kernel_status(owner, slug)
        if s != last:
            log(f"status={s}")
            last = s
        if "COMPLETE" in s.upper():
            break
        if "ERROR" in s.upper() or "CANCEL" in s.upper():
            log(f"ABORT: run ended as {s}")
            raise SystemExit(3)
        time.sleep(a.poll_s)
    else:
        log("ABORT: timed out waiting for the run")
        raise SystemExit(4)

    files = output_files(owner, slug)
    log(f"output files: {len(files)} listed, first: {files[:6]}{' ...' if len(files) > 6 else ''}")
    # Guard by *fetching* the file, not by looking for it in the listing: the listing is
    # capped (~500 entries) and a successful ICEBERG install writes thousands of package files
    # into /kaggle/working, which pushes submission.csv past the cap. Checking the listing then
    # refuses a perfectly valid submission -- observed on the pinned ctl run, whose submission.csv
    # downloaded fine (393,910 bytes) while the listing stopped at ice_site/*.
    if a.file in files:
        log(f"found {a.file} in the output listing")
    else:
        st, body = api(f"https://www.kaggle.com/api/v1/kernels/output/download/"
                       f"{owner}/{slug}/{a.file}", raw=True)
        if st != 200 or not body:
            log(f"ABORT: {a.file} is neither listed nor downloadable (HTTP {st}), refusing to submit")
            raise SystemExit(5)
        log(f"{a.file} is not in the (capped) listing but downloads: {len(body):,} bytes")

    payload = {"fileName": a.file, "competitionName": COMP,
               "kernelOwner": owner, "kernelSlug": slug,
               "kernelVersion": a.version, "submissionDescription": a.desc}
    st, body = api(URL_SUBMIT, payload, "POST")
    log(f"submit -> HTTP {st}: {body[:300]}")
    if st != 200:
        log("ABORT: submission rejected")
        raise SystemExit(6)

    # find our ref and poll the score
    ref = None
    for s in submissions():
        if s.get("description") == a.desc:
            ref = s.get("ref")
            break
    log(f"submission ref={ref}")
    while time.time() < deadline and ref:
        subs = submissions()
        me = next((s for s in subs if s.get("ref") == ref), None)
        if me and me.get("publicScore"):
            log(f"SCORE {a.desc}: public={me['publicScore']} status={me.get('status')} "
                f"bytes={me.get('totalBytes')} err={me.get('errorDescription')!r}")
            return
        time.sleep(a.poll_s)
    log("score not available before deadline (check the submissions page)")


if __name__ == "__main__":
    main()
