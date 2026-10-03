"""
Submit a kernel version to the CASMI 2026 code competition.

Why not the CLI: `kaggle competitions submit -k <kernel> -v <n>` returns a bare
400 with no reason. The service actually explains itself
("Your team has used its daily Submission allowance (5)..."), but the CLI hides
it. Calling CreateCodeSubmission directly also gets the field casing right:
camelCase is required; snake_case produces a misleading 403
("Permission 'kernelSessions.get' was denied"), which looks like an auth problem
but is not.

Usage:
  python submit_kernel.py --version 3 [--file submission.csv] [--message "..."]
"""

from __future__ import annotations

import argparse
import json
import urllib.request

TOKEN_PATH = r"D:\CASMI竞赛\.secrets\kaggle\access_token"
COMP = "enveda-CASMI26-molecule-id-mass-spectra"
OWNER = "nicholasnicklee"
SLUG = "casmi26-retrieval-analog"
URL = "https://api.kaggle.com/v1/competitions.CompetitionApiService/CreateCodeSubmission"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", type=int, required=True)
    ap.add_argument("--file", default="submission.csv",
                    help="output file the kernel writes (the competition requires "
                         "the name even for notebook submissions)")
    ap.add_argument("--message", default="retrieval + analog propagation")
    ap.add_argument("--kernel", default=f"{OWNER}/{SLUG}")
    a = ap.parse_args()

    owner, slug = a.kernel.split("/")
    payload = {
        "fileName": a.file,
        "competitionName": COMP,
        "kernelOwner": owner,
        "kernelSlug": slug,
        "kernelVersion": int(a.version),
        "submissionDescription": a.message,
    }
    tok = open(TOKEN_PATH).read().strip()
    req = urllib.request.Request(
        URL, method="POST",
        headers={"Authorization": "Bearer " + tok,
                 "Content-Type": "application/json"},
        data=json.dumps(payload).encode())
    print("submitting:", json.dumps(payload, indent=2))
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            body = r.read().decode()
            print(f"HTTP {r.status}")
            print(body[:1200])
    except Exception as e:
        body = ""
        if hasattr(e, "read"):
            try:
                body = e.read().decode(errors="replace")
            except Exception:
                pass
        print(f"HTTP {getattr(e, 'code', 'ERR')}: {e}")
        # surface the service's own explanation, which the CLI discards
        try:
            msg = json.loads(body)["error"]["message"]
            print("SERVICE SAYS:", msg)
        except Exception:
            print("BODY:", body[:800])
        raise SystemExit(1)


if __name__ == "__main__":
    main()
