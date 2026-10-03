"""
Build the Kaggle submission notebook.

Constraints that shape this file:
  * A Code Competition: the notebook itself is submitted and re-run by Kaggle
    with internet DISABLED. There is no pip install and the image has no rdkit.
  * Therefore everything the run needs is either (a) preinstalled in the image
    (numpy, pandas, pyarrow, lightgbm/xgboost) or (b) attached as a Kaggle
    Dataset. RDKit is NOT in the image, so it must come from an attached wheel
    or a dataset; we therefore avoid RDKit at inference time entirely and use the
    precomputed InChIKey14 keys and Morgan fingerprints stored in our assets.
  * The notebook must READ the competition test set at run time and write
    /kaggle/working/submission.csv. A static uploaded CSV scores 0.

The generated notebook is standalone: the scoring code is embedded rather than
imported from the `casmi` package, because the package cannot be installed
offline and shipping a wheel just for that adds a failure mode.

Usage: python build_notebook.py --out notebooks/kaggle_submission
"""
from __future__ import annotations

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

NOTEBOOK_CODE = r'''# CASMI 2026 - Molecule ID From Mass Spectra
# Retrieval + analog-propagation submission notebook.
#
# Offline by design: no pip install, no network. Assets are attached as a Kaggle
# Dataset. The test set is read at run time from the competition mount.

import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

T_START = time.time()


def log(msg):
    print(f"[{time.time()-T_START:7.1f}s] {msg}", flush=True)


# ----------------------------------------------------------------- config
COMP_DIR = "/kaggle/input/enveda-CASMI26-molecule-id-mass-spectra"
if not os.path.isdir(COMP_DIR):
    # fall back to whichever competition mount exists
    for cand in ("/kaggle/input/competitions/enveda-CASMI26-molecule-id-mass-spectra",
                 "/kaggle/input/enveda-casmi26-molecule-id-mass-spectra"):
        if os.path.isdir(cand):
            COMP_DIR = cand
            break
TEST_PATH = os.path.join(COMP_DIR, "test.parquet")
SAMPLE_SUB = os.path.join(COMP_DIR, "sample_submission.csv")

# Assets we produced offline (see docs/外部资源清单.md). Overridden by env var so
# the notebook can be pointed at any attached dataset.
ASSET_DIR = os.environ.get("CASMI_ASSETS", "/kaggle/input/casmi26-assets")
STRUCTS = os.path.join(ASSET_DIR, "structures.parquet")
LIB_DIR = os.path.join(ASSET_DIR, "library_spectra_sorted")
OUT_PATH = "/kaggle/working/submission.csv"

# Scoring parameters (identical to the validated local baseline)
RETRIEVAL_PPM = 15.0
RETRIEVAL_PPM_WIDE = 40.0
ANALOG_DM = 2.0
ANALOG_DM_MAX = 200.0
ANALOG_WEIGHT = 1.5
ANALOG_TOP_HITS = 60
FP_SCALE = 2.0
TOP_K = 25
BIN_WIDTH = 0.01
BIN_LO, BIN_HI = 40.0, 1300.0
MIN_REL_INTENSITY = 0.01
MIN_MZ, MAX_MZ = 50.0, 1200.0
MERGE_TOL_DA = 0.01
MAX_PEAKS = 80

MONO = {"H": 1.0078250319, "C": 12.0, "N": 14.0030740052, "O": 15.9949146221,
        "S": 31.97207069, "P": 30.97376151, "F": 18.99840320, "Cl": 34.96885271,
        "Br": 78.9183376, "I": 126.904468, "Na": 22.98976928, "K": 38.96370649}
PROTON = 1.007276466621
ELECTRON = 0.000548579909
H2O = 2 * MONO["H"] + MONO["O"]
_M_NA, _M_K, _M_CL = MONO["Na"], MONO["K"], MONO["Cl"]
_M_HCOOH = MONO["C"] + 2 * MONO["O"] + 2 * MONO["H"]

ADDUCT_OFFSET = {
    "[M+H]+": PROTON,
    "[M+NH4]+": MONO["N"] + 4 * MONO["H"] + PROTON,
    "[M-H2O+H]+": PROTON - H2O,
    "[M-2H2O+H]+": PROTON - 2 * H2O,
    "[M+Na]+": _M_NA - ELECTRON,
    "[M+K]+": _M_K - ELECTRON,
    "[M-H]-": -MONO["H"] + ELECTRON,
    "[M-H2O-H]-": -MONO["H"] - H2O + ELECTRON,
    "[M+CH2O2-H]-": _M_HCOOH - MONO["H"] + ELECTRON,
    "[M+Cl]-": _M_CL + ELECTRON,
}


def neutral_mass_from_precursor(mz, adduct):
    return float(mz) - ADDUCT_OFFSET[adduct]


# ------------------------------------------------------------ peak handling
def clean_peaks(mz, intensity):
    mz = np.asarray(mz, dtype=np.float64)
    inten = np.asarray(intensity, dtype=np.float64)
    if mz.size == 0:
        return np.empty(0, np.float32), np.empty(0, np.float32)
    o = np.argsort(mz)
    mz, inten = mz[o], inten[o]
    keep = (mz >= MIN_MZ) & (mz <= MAX_MZ) & (inten > 0)
    mz, inten = mz[keep], inten[keep]
    if mz.size == 0:
        return np.empty(0, np.float32), np.empty(0, np.float32)
    rel = inten / inten.max()
    hit = rel >= MIN_REL_INTENSITY
    mz, inten = mz[hit], inten[hit]
    if mz.size == 0:
        return np.empty(0, np.float32), np.empty(0, np.float32)
    if mz.size > 1:
        gs = np.r_[0, np.where(np.diff(mz) > MERGE_TOL_DA)[0] + 1]
        nm, ni = [], []
        for i, s in enumerate(gs):
            e = gs[i + 1] if i + 1 < len(gs) else mz.size
            j = s + int(np.argmax(inten[s:e]))
            nm.append(mz[j]); ni.append(inten[j])
        mz, inten = np.asarray(nm), np.asarray(ni)
    if mz.size > MAX_PEAKS:
        idx = np.argsort(inten)[::-1][:MAX_PEAKS]
        idx.sort()
        mz, inten = mz[idx], inten[idx]
    return mz.astype(np.float32), inten.astype(np.float32)


def bin_spectrum(mz, intensity):
    if mz.size == 0:
        return np.empty(0, np.int32), np.empty(0, np.float32)
    w = np.sqrt(np.maximum(intensity, 0.0))
    b = np.rint((np.asarray(mz, np.float64) - BIN_LO) / BIN_WIDTH).astype(np.int64)
    ok = (b >= 0) & (b <= int((BIN_HI - BIN_LO) / BIN_WIDTH))
    b, w = b[ok], w[ok]
    if b.size == 0:
        return np.empty(0, np.int32), np.empty(0, np.float32)
    o = np.argsort(b, kind="stable")
    b, w = b[o], w[o]
    uniq, start = np.unique(b, return_index=True)
    if uniq.size != b.size:
        w = np.maximum.reduceat(w, start)
        b = uniq
    return b.astype(np.int32), w.astype(np.float32)


def shifted_cosine(q_mz, q_int, dm, lib_bins, lib_w, lib_spec, n_spec):
    out = np.zeros(n_spec, dtype=np.float64)
    if q_mz.size == 0 or lib_bins.size == 0:
        return out
    b, w = bin_spectrum(q_mz - dm, q_int)
    if b.size == 0:
        return out
    qn = float(np.sqrt((w.astype(np.float64) ** 2).sum()))
    if qn <= 0:
        return out
    qw = w / qn
    lo = np.searchsorted(lib_bins, b, side="left")
    hi = np.searchsorted(lib_bins, b, side="right")
    keep = np.flatnonzero(hi > lo)
    if keep.size == 0:
        return out
    parts = [np.arange(lo[t], hi[t]) for t in keep]
    ent = np.concatenate(parts)
    ent_t = np.concatenate([np.full(hi[t] - lo[t], t, dtype=np.int64) for t in keep])
    spec = lib_spec[ent]
    np.add.at(out, spec, qw[ent_t] * lib_w[ent])
    # normalise over the matched intersection only (a shift matches a subset)
    o = np.argsort(spec, kind="stable")
    spec_s, w_s = spec[o], lib_w[ent[o]]
    uniq, start = np.unique(spec_s, return_index=True)
    denom = np.sqrt(np.add.reduceat(w_s.astype(np.float64) ** 2, start))
    nz = denom > 0
    out[uniq[nz]] /= denom[nz]
    return out


# ------------------------------------------------------------------ assets
class StructureStore:
    def __init__(self, path):
        df = pd.read_parquet(path)
        self.smiles = df["smiles"].to_numpy()
        self.mass = df["mass"].to_numpy(np.float64)
        order = np.argsort(self.mass, kind="stable")
        self.keys = df["key"].to_numpy()[order]
        self.smiles = self.smiles[order]
        self.mass = self.mass[order]
        fp = df["fp"].to_numpy()
        dim = len(fp[0]) if len(fp) and fp[0] is not None else 0
        self.fp = np.zeros((len(df), dim), dtype=np.uint8)
        self.has_fp = np.zeros(len(df), dtype=bool)
        for i, v in enumerate(fp):
            if v is not None:
                self.fp[i] = np.asarray(v, dtype=np.uint8)
                self.has_fp[i] = True
        self.fp = self.fp[order]
        self.has_fp = self.has_fp[order]
        self._lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
        self._key_to_idx = {k: i for i, k in enumerate(self.keys)}
        self._key_mass = dict(zip(self.keys, self.mass))
        log(f"StructureStore: {len(self.keys):,} structures")

    def __len__(self):
        return len(self.keys)

    def window(self, m, ppm):
        tol = abs(m) * ppm * 1e-6
        return np.arange(np.searchsorted(self.mass, m - tol, side="left"),
                         np.searchsorted(self.mass, m + tol, side="right"))

    def between(self, lo_m, hi_m):
        return np.arange(np.searchsorted(self.mass, lo_m, side="left"),
                         np.searchsorted(self.mass, hi_m, side="right"))

    def tanimoto(self, qfp, idx):
        sub = self.fp[idx]
        inter = self._lut[np.minimum(sub, qfp[None, :])].sum(axis=1).astype(np.float64)
        uni = self._lut[sub].sum(axis=1).astype(np.float64) + self._lut[qfp].sum() - inter
        out = np.zeros(len(idx))
        nz = uni > 0
        out[nz] = inter[nz] / uni[nz]
        return out


class SpectralIndex:
    """Lazy, query-scoped view of the key-partitioned spectral library."""

    def __init__(self, dir_path):
        idx = pd.read_parquet(os.path.join(dir_path, "library_index.parquet"))
        self.part_of = dict(zip(idx["key"], idx["partition"]))
        self.dir = dir_path
        self._cache = {}
        self._group_cache = {}
        self._loaded = []
        log(f"SpectralIndex: {len(self.part_of):,} compounds")

    def _load(self, p):
        got = self._cache.get(p)
        if got is not None:
            return got
        df = pd.read_parquet(os.path.join(self.dir, f"part_{p:04d}.parquet"),
                             columns=["key", "mz", "intensity"])
        mz = np.empty(len(df), dtype=object)
        inten = np.empty(len(df), dtype=object)
        for i, (a, b) in enumerate(zip(df["mz"].to_numpy(), df["intensity"].to_numpy())):
            mz[i] = np.frombuffer(a, dtype=np.float32)
            inten[i] = np.frombuffer(b, dtype=np.float32)
        by_key = defaultdict(list)
        for i, k in enumerate(df["key"].to_numpy()):
            by_key[k].append(i)
        got = {"mz": mz, "inten": inten, "by_key": by_key,
               "keys": df["key"].to_numpy()}
        self._cache[p] = got
        if p not in self._loaded:
            self._loaded.append(p)
        return got

    def direct_cosines(self, q_mz, q_int, keys):
        if not keys:
            return np.empty(0, object), np.empty(0, np.float32)
        qb, qw = bin_spectrum(q_mz, q_int)
        if qb.size == 0:
            return np.empty(0, object), np.empty(0, np.float32)
        qn = float(np.sqrt((qw.astype(np.float64) ** 2).sum()))
        if qn <= 0:
            return np.empty(0, object), np.empty(0, np.float32)
        qwn = (qw / qn).astype(np.float32)
        best = {}
        by_part = defaultdict(list)
        for k in keys:
            p = self.part_of.get(k)
            if p is not None:
                by_part[p].append(k)
        for p, ks in by_part.items():
            blk = self._load(p)
            spec_ids, spec_key = [], []
            for k in ks:
                for si in blk["by_key"].get(k, ()):
                    spec_ids.append(si)
                    spec_key.append(k)
            if not spec_ids:
                continue
            all_b, all_w, all_owner = [], [], []
            for j, si in enumerate(spec_ids):
                b, w = bin_spectrum(blk["mz"][si], blk["inten"][si])
                if b.size == 0:
                    continue
                nrm = float(np.sqrt((w.astype(np.float64) ** 2).sum()))
                if nrm <= 0:
                    continue
                all_b.append(b); all_w.append((w / nrm).astype(np.float32))
                all_owner.append(np.full(b.size, j, dtype=np.int32))
            if not all_b:
                continue
            bins = np.concatenate(all_b); wts = np.concatenate(all_w)
            owner = np.concatenate(all_owner)
            o = np.argsort(bins, kind="stable")
            bins, wts, owner = bins[o], wts[o], owner[o]
            scores = np.zeros(len(spec_ids))
            lo = np.searchsorted(bins, qb, side="left")
            hi = np.searchsorted(bins, qb, side="right")
            for t in range(qb.size):
                if hi[t] <= lo[t]:
                    continue
                e = slice(lo[t], hi[t])
                np.add.at(scores, owner[e], qwn[t] * wts[e])
            for j, k in enumerate(spec_key):
                if scores[j] > best.get(k, 0.0):
                    best[k] = float(scores[j])
        if not best:
            return np.empty(0, object), np.empty(0, np.float32)
        ks = list(best)
        return np.asarray(ks, object), np.asarray([best[k] for k in ks], np.float32)

    def group_binned(self, parts):
        parts = sorted(set(parts))
        if not parts:
            return None
        # cache: rebuilding the binned view on every query dominated runtime
        ck = tuple(parts)
        got = self._group_cache.get(ck)
        if got is not None:
            return got
        bl, wl, sl, kl = [], [], [], []
        off = 0
        for p in parts:
            blk = self._load(p)
            all_b, all_w, all_s = [], [], []
            for si in range(len(blk["mz"])):
                b, w = bin_spectrum(blk["mz"][si], blk["inten"][si])
                if b.size == 0:
                    continue
                nrm = float(np.sqrt((w.astype(np.float64) ** 2).sum()))
                if nrm <= 0:
                    continue
                all_b.append(b); all_w.append((w / nrm).astype(np.float32))
                all_s.append(np.full(b.size, si, dtype=np.int64))
            if not all_b:
                continue
            b = np.concatenate(all_b); w = np.concatenate(all_w)
            s = np.concatenate(all_s)
            o = np.argsort(b, kind="stable")
            bl.append(b[o]); wl.append(w[o]); sl.append(s[o] + off)
            kl.append(blk["keys"])
            off += len(blk["mz"])
        if not bl:
            return None
        got = {"bins": np.concatenate(bl), "w": np.concatenate(wl),
               "spec": np.concatenate(sl), "keys": np.concatenate(kl), "n": off}
        self._group_cache[ck] = got
        return got

    def release(self, keep_last=3):
        keep = set(self._loaded[-keep_last:])
        for p in list(self._cache):
            if p not in keep:
                del self._cache[p]
        self._loaded = [p for p in self._loaded if p in keep]
        # the binned group view references the dropped partitions
        self._group_cache.clear()


# ----------------------------------------------------------------- scoring
def score_molecule(spectra, store, spec_idx, use_analog=True):
    cand = {}
    for s in spectra:
        m = s["neutral_mass"]
        if not np.isfinite(m):
            continue
        for ppm in (RETRIEVAL_PPM, RETRIEVAL_PPM_WIDE):
            for i in store.window(m, ppm):
                cand.setdefault(store.keys[i], int(i))
        if use_analog:
            for i in store.between(m - ANALOG_DM, m + ANALOG_DM):
                cand.setdefault(store.keys[i], int(i))
    if not cand:
        return [], {}
    cand_keys = list(cand)
    key_pos = {k: i for i, k in enumerate(cand_keys)}
    direct = np.zeros(len(cand_keys))
    q_masses = [s["neutral_mass"] for s in spectra if np.isfinite(s["neutral_mass"])]
    if not q_masses:
        return [], {}

    lib_direct = {}
    for s in spectra:
        hk, hs = spec_idx.direct_cosines(s["mz"], s["intensity"], cand_keys)
        for k, v in zip(hk, hs):
            if v > lib_direct.get(k, 0.0):
                lib_direct[k] = float(v)
    for k, v in lib_direct.items():
        p = key_pos.get(k)
        if p is not None and v > direct[p]:
            direct[p] = v

    if use_analog:
        km = store._key_mass
        lib_keys = set()
        for m in q_masses:
            for i in store.between(m - ANALOG_DM, m + ANALOG_DM):
                lib_keys.add(store.keys[i])
        lib_keys -= set(cand_keys)
        if lib_keys:
            parts = {spec_idx.part_of[k] for k in lib_keys if k in spec_idx.part_of}
            grp = spec_idx.group_binned(parts)
            if grp is not None and grp["n"] > 0:
                analog_raw = {}
                for s in spectra:
                    qm = s["neutral_mass"]
                    if not np.isfinite(qm):
                        continue
                    buckets = defaultdict(list)
                    for k in lib_keys:
                        ml = km.get(k)
                        if ml is None:
                            continue
                        dm = qm - ml
                        if abs(dm) > ANALOG_DM_MAX:
                            continue
                        buckets[round(dm, 3)].append(k)
                    for dm_val, ks in buckets.items():
                        if abs(dm_val) < 1e-3:
                            continue
                        sc = shifted_cosine(s["mz"], s["intensity"], float(dm_val),
                                            grp["bins"], grp["w"], grp["spec"], grp["n"])
                        if not np.any(sc > 0):
                            continue
                        kset = set(ks)
                        for si in np.flatnonzero(sc > 0):
                            k = grp["keys"][si]
                            if k in kset and sc[si] > analog_raw.get(k, 0.0):
                                analog_raw[k] = float(sc[si])
                top = sorted(analog_raw.items(), key=lambda kv: -kv[1])[:ANALOG_TOP_HITS]
                if top:
                    sub_keys = [k for k, _ in top]
                    sub_idx = np.array([store._key_to_idx[k] for k in sub_keys], np.int64)
                    valid = store.has_fp[sub_idx]
                    cand_arr = np.array([cand[k] for k in cand_keys], np.int64)
                    ok = store.has_fp[cand_arr]
                    pos_ok = np.flatnonzero(ok)
                    if pos_ok.size:
                        sub_cand = cand_arr[pos_ok]
                        analog = np.zeros(len(cand_keys))
                        for k, sc in top:
                            hi = store._key_to_idx[k]
                            if not store.has_fp[hi]:
                                continue
                            tan = store.tanimoto(store.fp[hi], sub_cand)
                            np.maximum.at(analog, pos_ok,
                                          (sc ** FP_SCALE) * tan * ANALOG_WEIGHT)
                    else:
                        analog = np.zeros(len(cand_keys))
                else:
                    analog = np.zeros(len(cand_keys))
            else:
                analog = np.zeros(len(cand_keys))
        else:
            analog = np.zeros(len(cand_keys))
    else:
        analog = np.zeros(len(cand_keys))

    final = direct + analog
    order = np.argsort(-final)
    ranked = [(cand_keys[i], float(final[i])) for i in order]
    return ranked, cand


# ---------------------------------------------------------------------- run
def main():
    log("reading test set")
    test = pd.read_parquet(TEST_PATH)
    log(f"  {len(test)} spectra, {test['molecule_id'].nunique()} molecules")

    store = StructureStore(STRUCTS)
    spec_idx = SpectralIndex(LIB_DIR)

    # aggregate spectra per molecule, computing the adduct-aware neutral mass
    molecules = defaultdict(list)
    for row in test.itertuples(index=False):
        ad = str(row.adduct)
        if ad not in ADDUCT_OFFSET:
            continue
        mz, inten = clean_peaks(row.ms2_mzs, row.ms2_normalized_intensities)
        if mz.size < 3:
            continue
        molecules[str(row.molecule_id)].append({
            "mz": mz, "intensity": inten,
            "neutral_mass": neutral_mass_from_precursor(row.precursor_mz, ad),
        })
    log(f"  usable molecules: {len(molecules)}")

    sample = pd.read_csv(SAMPLE_SUB)
    ids = list(sample["molecule_id"].astype(str))

    preds = {}
    t0 = time.time()
    for n, mid in enumerate(ids, 1):
        spectra = molecules.get(mid)
        if not spectra:
            preds[mid] = []
            continue
        ranked, cand = score_molecule(spectra, store, spec_idx)
        preds[mid] = [store.smiles[cand[k]] for k, _ in ranked[:TOP_K]]
        if n % 25 == 0 or n == len(ids):
            el = time.time() - t0
            log(f"  {n}/{len(ids)}  {el:.0f}s ({el/n:.2f}s/mol)")
        if n % 25 == 0:
            spec_idx.release(keep_last=3)

    # format: one row per molecule, ';'-joined, best first, dedup by string
    rows = []
    for mid in ids:
        seen, out = set(), []
        for smi in preds.get(mid, []):
            if smi and smi not in seen:
                seen.add(smi)
                out.append(smi)
            if len(out) >= TOP_K:
                break
        rows.append({"molecule_id": mid, "smiles": ";".join(out)})
    sub = pd.DataFrame(rows, columns=["molecule_id", "smiles"])
    sub.to_csv(OUT_PATH, index=False)
    log(f"wrote {OUT_PATH}")

    counts = sub["smiles"].astype(str).str.split(";").apply(
        lambda xs: len([x for x in xs if x.strip()]))
    log(f"  rows={len(sub)} candidates min={counts.min()} "
        f"median={int(counts.median())} max={counts.max()}")
    log("done")


main()
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "notebooks", "kaggle_submission"))
    ap.add_argument("--user", default="casmi-user")
    ap.add_argument("--slug", default="casmi26-retrieval-analog")
    ap.add_argument("--assets-dataset", default="casmi-user/casmi26-assets")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)

    # notebook in the .ipynb shape Kaggle expects
    nb = {
        "cells": [
            {"cell_type": "markdown", "metadata": {},
             "source": ["# CASMI 2026 retrieval + analog propagation\n",
                        "Reads the competition test set at run time and writes "
                        "`/kaggle/working/submission.csv`.\n"]},
            {"cell_type": "code", "execution_count": None, "metadata": {},
             "outputs": [], "source": NOTEBOOK_CODE.splitlines(keepends=True)},
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    nb_path = os.path.join(a.out, "notebook.ipynb")
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

    meta = {
        "id": f"{a.user}/{a.slug}",
        "title": "CASMI26 retrieval + analog",
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "dataset_sources": [a.assets_dataset],
        "competition_sources": ["enveda-CASMI26-molecule-id-mass-spectra"],
        "kernel_sources": [],
        "model_sources": [],
    }
    with open(os.path.join(a.out, "kernel-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"wrote {nb_path}")
    print(f"wrote {os.path.join(a.out, 'kernel-metadata.json')}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
