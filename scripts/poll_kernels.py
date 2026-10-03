"""Poll several Kaggle kernels until each reaches a terminal state."""
import os
import subprocess
import sys
import time

PY = r"D:\Coding\miniconda\envs\casmi2026\python.exe"
TOKEN = open(r"D:\CASMI竞赛\.secrets\kaggle\access_token").read().strip()
ENV = dict(os.environ, KAGGLE_API_TOKEN=TOKEN)

KERNELS = sys.argv[1:] or [
    "nicholasnicklee/casmi26-retrieval-analog",
    "nicholasnicklee/casmi26-retrieval-analog-np",
]

TERMINAL = ("COMPLETE", "ERROR", "CANCELLED")
state = {k: "?" for k in KERNELS}

for _ in range(70):
    for k in KERNELS:
        if state[k] in TERMINAL:
            continue
        try:
            r = subprocess.run([PY, "-m", "kaggle", "kernels", "status", k],
                               capture_output=True, text=True, timeout=90, env=ENV)
            out = (r.stdout or "").strip()
            if "status" in out:
                line = out.splitlines()[-1]
                st = line.split('"')[-2].split(".")[-1] if '"' in line else "?"
                state[k] = st
        except subprocess.TimeoutExpired:
            pass
    print("[" + time.strftime("%H:%M:%S") + "] " +
          "  ".join(f"{k.split('/')[-1]}={v}" for k, v in state.items()), flush=True)
    if all(v in TERMINAL for v in state.values()):
        print("ALL TERMINAL", flush=True)
        break
    time.sleep(30)
