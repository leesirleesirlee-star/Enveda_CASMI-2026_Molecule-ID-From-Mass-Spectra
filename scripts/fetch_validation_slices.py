"""
Fetch only the row groups needed for the three-fold validation harness,
using HTTP range reads with per-chunk retry/resume. Avoids the flaky 2.82GB download.

V-A : row group 20 = ingest_lib 'enveda-np-examples' (1184 rows, same instrument as test)
V-B : sampled row groups, answer RETAINED in library  -> retrieval ceiling
V-C : sampled row groups held out                     -> strict / extrapolation

Output: data/interim/validation_slices.parquet
"""
import io
import struct
import time
import urllib.request
import pyarrow as pa
import pyarrow.parquet as pq

URL = ("https://huggingface.co/datasets/pradeepss007/Casmi26/resolve/main/"
       "enveda-CASMI26-molecule-id-mass-spectra/train.parquet")
OUT = r"D:\CASMI竞赛\data\interim\validation_slices.parquet"
UA = {"User-Agent": "Mozilla/5.0"}

# columns we actually need (skip the huge list columns only where avoidable)
COLS = ["ingest_lib", "normalized_smiles", "inchikey", "inchikey14", "molecular_formula",
        "ionization_mode", "instrument_type", "adduct", "precursor_mz",
        "precursor_error_ppm", "ms2_mzs", "ms2_normalized_intensities",
        "num_peaks", "base_peak_intensity", "collision_energy_ev",
        "collision_energy_orig", "collision_energy_orig_units"]


def fetch(a, b, tries=8):
    """Range GET with exponential backoff; returns bytes."""
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(URL, headers={**UA, "Range": f"bytes={a}-{b}"})
            with urllib.request.urlopen(req, timeout=180) as r:
                buf = b""
                while True:
                    c = r.read(1 << 20)
                    if not c:
                        break
                    buf += c
            if len(buf) == b - a + 1:
                return buf
            last = f"short read {len(buf)}/{b - a + 1}"
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
        wait = min(2 ** i, 30)
        print(f"    retry {i+1}/{tries} in {wait}s ({last})", flush=True)
        time.sleep(wait)
    raise RuntimeError(f"range {a}-{b} failed after {tries} tries: {last}")


def rg_bounds(m, gi):
    rg = m.row_group(gi)
    offs = []
    for ci in range(rg.num_columns):
        c = rg.column(ci)
        for o in (c.data_page_offset, c.dictionary_page_offset):
            if o is not None and o > 0:
                offs.append(o)
    return min(offs), max(offs)


# --- read footer ---
print("reading footer...", flush=True)
req = urllib.request.Request(URL, headers={**UA, "Range": "bytes=0-0"})
with urllib.request.urlopen(req, timeout=60) as r:
    TOTAL = int(r.headers["Content-Range"].split("/")[1])
print(f"remote train.parquet: {TOTAL/1024**3:.2f} GB", flush=True)

TAIL = 262144
tail = fetch(TOTAL - TAIL, TOTAL - 1)
flen = struct.unpack("<I", tail[-8:-4])[0]
footer = tail[-(flen + 8):-8]
m = pq.ParquetFile(io.BytesIO(b"PAR1" + footer + struct.pack("<I", flen) + b"PAR1")).metadata
print(f"rows={m.num_rows:,} row_groups={m.num_row_groups}", flush=True)

# --- choose row groups ---
PLAN = {
    "V_A": [20],          # enveda-np-examples, same instrument as test
    "V_B": [0, 5],        # answer retained -> retrieval ceiling
    "V_C": [10, 15],      # held out -> strict
}

frames = []
for fold, groups in PLAN.items():
    for gi in groups:
        start, end = rg_bounds(m, gi)
        nbytes = end - start
        print(f"\n[{fold}] row group {gi}: fetching {nbytes/1e6:.1f} MB ...", flush=True)
        blob = fetch(start, end - 1)
        t = pq.ParquetFile(io.BytesIO(blob))
        tbl = t.read(columns=COLS)
        df = tbl.to_pandas()
        df["fold"] = fold
        df["_row_group"] = gi
        frames.append(df)
        print(f"[{fold}] rg{gi}: {len(df):,} rows, "
              f"libs={sorted(df['ingest_lib'].unique())}", flush=True)

out = pa.concat_tables([pa.Table.from_pandas(f, preserve_index=False) for f in frames])
pq.write_table(out, OUT, compression="zstd")
print(f"\nwrote {OUT}")
print("total rows:", out.num_rows)
