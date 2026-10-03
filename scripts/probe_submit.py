"""Call CreateCodeSubmission directly to see the real 400 reason."""
import os
import json
import urllib.request

TOK = open(r"D:\CASMI竞赛\.secrets\kaggle\access_token").read().strip()
BASE = "https://api.kaggle.com/v1/competitions.CompetitionApiService/"


def call(method, payload=None):
    req = urllib.request.Request(
        BASE + method, method="POST",
        headers={"Authorization": "Bearer " + TOK,
                 "Content-Type": "application/json"},
        data=json.dumps(payload).encode() if payload is not None else None)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, r.read().decode()[:800]
    except Exception as e:
        body = ""
        if hasattr(e, "read"):
            try:
                body = e.read().decode(errors="replace")[:800]
            except Exception:
                pass
        return getattr(e, "code", "ERR"), f"{type(e).__name__}: {e} | body={body}"


cases = {
    "with file_name": {
        "fileName": "submission.csv",
        "competitionName": "enveda-CASMI26-molecule-id-mass-spectra",
        "kernelOwner": "nicholasnicklee",
        "kernelSlug": "casmi26-retrieval-analog",
        "kernelVersion": 3,
        "submissionDescription": "retrieval + analog",
    },
    "snake_case": {
        "file_name": "submission.csv",
        "competition_name": "enveda-CASMI26-molecule-id-mass-spectra",
        "_kernel_owner": "nicholasnicklee",
        "kernel_slug": "casmi26-retrieval-analog",
        "kernel_version": 3,
        "submission_description": "retrieval + analog",
    },
    "no version": {
        "fileName": "submission.csv",
        "competitionName": "enveda-CASMI26-molecule-id-mass-spectra",
        "kernelOwner": "nicholasnicklee",
        "kernelSlug": "casmi26-retrieval-analog",
        "submissionDescription": "retrieval + analog",
    },
}
for name, payload in cases.items():
    print(f"--- {name} ---")
    print(call("CreateCodeSubmission", payload))
    print()
