"""
Measure achievable throughput properly, to choose a real download strategy.

Compares: (a) single stream, (b) N concurrent streams, on the mirrors we have.
A per-connection cap and an aggregate cap need opposite responses, so this
distinguishes them instead of guessing.
"""
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

UA = {"User-Agent": "Mozilla/5.0"}

SOURCES = {
    "HF pradeepss007": ("https://huggingface.co/datasets/pradeepss007/Casmi26/resolve/main/"
                        "enveda-CASMI26-molecule-id-mass-spectra/train.parquet"),
    "HF lonelyforever": ("https://huggingface.co/datasets/lonelyforever/enveda.casmi/"
                         "resolve/main/train.parquet"),
}

CHUNK = 6 * 1024 * 1024
# Sample from deep inside the file so we are not measuring the same cached prefix
OFFSETS = [1_500_000_000, 1_600_000_000, 1_700_000_000, 1_800_000_000,
           1_900_000_000, 2_000_000_000, 2_100_000_000, 2_200_000_000]


def one(url, start, nbytes=CHUNK):
    end = start + nbytes - 1
    req = urllib.request.Request(url, headers={**UA, "Range": f"bytes={start}-{end}"})
    t0 = time.time()
    got = 0
    with urllib.request.urlopen(req, timeout=120) as r:
        while got < nbytes:
            c = r.read(1 << 20)
            if not c:
                break
            got += len(c)
    dt = time.time() - t0
    return got, dt


for name, url in SOURCES.items():
    print(f"\n===== {name} =====", flush=True)
    # single stream
    try:
        got, dt = one(url, OFFSETS[0])
        print(f"  1 stream : {got/1e6:5.2f} MB in {dt:5.1f}s -> {got/1e6/dt:5.3f} MB/s", flush=True)
    except Exception as e:
        print(f"  1 stream : FAILED {type(e).__name__}: {e}", flush=True)

    # 8 concurrent streams
    try:
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=8) as ex:
            res = list(ex.map(lambda o: one(url, o), OFFSETS))
        dt = time.time() - t0
        tot = sum(r[0] for r in res)
        print(f"  8 streams: {tot/1e6:5.2f} MB in {dt:5.1f}s -> {tot/1e6/dt:5.3f} MB/s "
              f"(per-stream {tot/1e6/8/dt:5.3f} MB/s)", flush=True)
    except Exception as e:
        print(f"  8 streams: FAILED {type(e).__name__}: {e}", flush=True)
