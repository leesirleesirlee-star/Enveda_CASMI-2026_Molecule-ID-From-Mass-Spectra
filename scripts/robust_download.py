"""
Resilient parallel chunked downloader (resumable, per-chunk retry).

The link to the data hosts breaks a single connection constantly (observed:
IncompleteRead errors, throughput collapsing from 500 kB/s to 30 kB/s).
Each chunk is an independent range request, so one broken connection only
costs that chunk. Completed chunks are cached on disk and skipped on rerun.

Usage:
  python robust_download.py --url <URL> --out <FILE> [--chunk-mb 16] [--workers 8]
"""
import argparse
import json
import os
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

UA = {"User-Agent": "Mozilla/5.0"}
_print_lock = threading.Lock()


def head_size(url):
    req = urllib.request.Request(url, headers={**UA, "Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        cr = r.headers.get("Content-Range")
        if cr and "/" in cr:
            return int(cr.split("/")[1])
        cl = r.headers.get("Content-Length")
        return int(cl) if cl else None


def fetch_range(url, a, b, tries=10):
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={**UA, "Range": f"bytes={a}-{b}"})
            with urllib.request.urlopen(req, timeout=300) as r:
                buf = bytearray()
                while True:
                    c = r.read(1 << 20)
                    if not c:
                        break
                    buf += c
            if len(buf) == b - a + 1:
                return bytes(buf)
            last = f"short {len(buf)}/{b - a + 1}"
        except Exception as e:
            last = f"{type(e).__name__}"
        time.sleep(min(1.5 * (i + 1), 20))
    raise RuntimeError(f"chunk {a}-{b} failed: {last}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk-mb", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    # Partition the chunk space so several instances (e.g. one per mirror) can
    # run concurrently without ever doing the same chunk. Total throughput here
    # is capped by the link (~1 MB/s), and a single connection gives ~0.18 MB/s,
    # so a handful of streams saturates it; more than that just thrashes.
    ap.add_argument("--part", type=int, default=0, help="this instance's part index")
    ap.add_argument("--parts", type=int, default=1, help="total number of instances")
    a = ap.parse_args()

    total = head_size(a.url)
    if total is None:
        sys.exit("could not determine remote size (no Content-Range/Content-Length)")
    print(f"remote size: {total:,} bytes ({total/1024**3:.2f} GB)", flush=True)

    csz = a.chunk_mb * 1024 * 1024
    nchunks = (total + csz - 1) // csz
    parts_dir = a.out + ".parts"
    os.makedirs(parts_dir, exist_ok=True)

    # Chunk files are named by index, and index i means bytes [i*csz, ...). If
    # the chunk size (or the remote size) changes between runs, every existing
    # file silently means different bytes and a resumed download produces a
    # corrupt output. Guard against that explicitly.
    manifest = os.path.join(parts_dir, "_manifest.json")
    want = {"chunk_size": csz, "total": total}
    if os.path.exists(manifest):
        with open(manifest) as f:
            have = json.load(f)
        if have != want:
            n_old = len([f for f in os.listdir(parts_dir) if f.endswith(".bin")])
            print(f"WARNING: chunk layout changed {have} -> {want}; "
                  f"discarding {n_old} stale chunks (they mean different bytes now).",
                  flush=True)
            for f in os.listdir(parts_dir):
                if f.endswith((".bin", ".tmp")):
                    os.remove(os.path.join(parts_dir, f))
    with open(manifest, "w") as f:
        json.dump(want, f)

    todo = []
    for i in range(nchunks):
        if a.parts > 1 and (i % a.parts) != a.part:
            continue          # owned by another instance
        s = i * csz
        e = min(s + csz, total) - 1
        pf = os.path.join(parts_dir, f"{i:05d}.bin")
        if os.path.exists(pf) and os.path.getsize(pf) == e - s + 1:
            continue
        todo.append((i, s, e, pf))

    done_bytes = (nchunks - len(todo)) * csz
    print(f"chunks: {nchunks} x {a.chunk_mb}MB | this instance: part {a.part}/{a.parts} | "
          f"already done: {nchunks-len(todo)} ({done_bytes/1024**3:.2f} GB) | "
          f"to download here: {len(todo)}", flush=True)

    t0 = time.time()
    base_done = done_bytes
    lock = threading.Lock()
    counter = {"n": 0}

    def work(item):
        i, s, e, pf = item
        data = fetch_range(a.url, s, e)
        tmp = pf + ".tmp"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, pf)
        with lock:
            counter["n"] += 1
            # report this instance's own contribution, not the whole chunk space
            own = counter["n"] * csz
            el = time.time() - t0
            rate = own / el / 1e6 if el > 0 else 0
            print(f"  [{counter['n']}/{len(todo)}] {i:05d} done | "
                  f"own {own/1e6:.0f} MB | {rate:.2f} MB/s | {el/60:.1f} min", flush=True)
        return i

    failed = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(work, it): it for it in todo}
        for f in as_completed(futs):
            try:
                f.result()
            except Exception as e:
                failed += 1
                print(f"  !! chunk {futs[f][0]:05d} FAILED: {e}", flush=True)

    if failed:
        print(f"\n{failed} chunks failed — rerun this script to resume.", flush=True)
        sys.exit(1)

    # Stitching must verify that the WHOLE chunk set is present, not just the
    # chunks this instance downloaded. With --parts > 1 each instance owns a
    # subset, so a naive stitch produces a silently truncated file. Never write
    # the output unless every chunk exists at its exact expected size.
    missing, wrong = [], []
    for i in range(nchunks):
        pf = os.path.join(parts_dir, f"{i:05d}.bin")
        s = i * csz
        e = min(s + csz, total) - 1
        exp = e - s + 1
        if not os.path.exists(pf):
            missing.append(i)
        elif os.path.getsize(pf) != exp:
            wrong.append((i, os.path.getsize(pf), exp))

    if missing or wrong:
        print(f"\nCANNOT STITCH: {len(missing)} chunks missing, {len(wrong)} wrong size.",
              flush=True)
        if missing[:10]:
            print(f"  missing (first 10): {missing[:10]}", flush=True)
        if wrong[:5]:
            print(f"  wrong size (first 5): {wrong[:5]}", flush=True)
        print("  Re-run the downloader (with the same --chunk-mb) to fetch the rest.",
              flush=True)
        sys.exit(2)

    print("\nstitching...", flush=True)
    with open(a.out, "wb") as out:
        for i in range(nchunks):
            with open(os.path.join(parts_dir, f"{i:05d}.bin"), "rb") as f:
                while True:
                    b = f.read(1 << 22)
                    if not b:
                        break
                    out.write(b)

    got = os.path.getsize(a.out)
    print(f"wrote {a.out}: {got:,} bytes", flush=True)
    if got != total:
        sys.exit(f"SIZE MISMATCH: {got} != {total}")
    print("SIZE OK", flush=True)


if __name__ == "__main__":
    main()
