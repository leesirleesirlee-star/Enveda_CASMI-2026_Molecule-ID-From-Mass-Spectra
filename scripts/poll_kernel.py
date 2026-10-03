"""Poll a Kaggle kernel status with SSL retries, printing only the status line."""
import os
import subprocess
import sys
import time

SLUG = "nicholasnicklee/casmi26-retrieval-analog"
TOKEN = open(r"D:\CASMI竞赛\.secrets\kaggle\access_token").read().strip()
PY = r"D:\Coding\miniconda\envs\casmi2026\python.exe"

env = dict(os.environ, KAGGLE_API_TOKEN=TOKEN)

for i in range(60):
    try:
        r = subprocess.run([PY, "-m", "kaggle", "kernels", "status", SLUG],
                           capture_output=True, text=True, timeout=90, env=env)
        out = (r.stdout or "").strip()
        err = (r.stderr or "")
        if "status" in out:
            line = out.splitlines()[-1].strip()
            print(f"[{time.strftime('%H:%M:%S')}] {line}", flush=True)
            if any(k in line for k in ("COMPLETE", "ERROR", "CANCELLED")):
                print("TERMINAL STATE REACHED", flush=True)
                break
        else:
            # transient SSL/network failure; report compactly and retry
            first = (err or out).strip().splitlines()
            msg = first[0][:120] if first else "no output"
            print(f"[{time.strftime('%H:%M:%S')}] (retry) {msg}", flush=True)
    except subprocess.TimeoutExpired:
        print(f"[{time.strftime('%H:%M:%S')}] (retry) timeout", flush=True)
    time.sleep(30)
