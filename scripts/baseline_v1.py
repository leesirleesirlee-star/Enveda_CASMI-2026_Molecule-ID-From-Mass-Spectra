"""
P1 baseline: mass-window retrieval + direct spectral match, with the analog
(dm-shift) channel, evaluated on the three-fold protocol.

The goal of this run is NOT a leaderboard score. It is to produce the first
trustworthy numbers on the strict protocol, and specifically to separate:

  * retrieval failure  - the true structure was never in the candidate set
  * ranking failure    - it was in the set but ordered too low

That split is reported as oracle recall@K next to MRR@25. Without it, both
failure modes look identical and effort gets spent in the wrong place.

Usage:
  python baseline_v1.py --fold V_A [--limit N] [--channels direct,analog]
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from casmi.core import smiles_to_inchikey14            # noqa: E402
from casmi.matching import (                           # noqa: E402
    DEFAULT_TOL, MAX_DM, modified_cosine, propagate_analog,
)
from casmi.spectra import clean_peaks, load_spectra, group_by_molecule  # noqa: E402
from casmi.validation import V_A, V_B, V_C, build_folds, evaluate       # noqa: E402

DATA = r"D:\CASMI竞赛\data"
TRAIN = os.path.join(DATA, "raw", "train.parquet")
STRUCTS = os.path.join(DATA, "processed", "structures.parquet")
LIBS = os.path.join(DATA, "processed", "library_spectra.parquet")
OUT = os.path.join(DATA, "processed")

# Retrieval windows (ppm) and the analog mass-difference window.
RETRIEVAL_PPM = 15.0
RETRIEVAL_PPM_WIDE = 40.0
ANALOG_DM = 2.0          # +/- Da around the query mass for the analog channel
ANALOG_DM_MAX = MAX_DM   # hard cap on how far a library compound may sit
TOP_CANDIDATES = 25


# --------------------------------------------------------------- library load

class StructureStore:
    """Mass-sorted structure table with fingerprints, loadable once."""

    def __init__(self, path: str):
        df = pd.read_parquet(path)
        self.keys = df["key"].to_numpy()
        self.smiles = df["smiles"].to_numpy()
        self.mass = df["mass"].to_numpy(dtype=np.float64)
        order = np.argsort(self.mass, kind="stable")
        self.keys = self.keys[order]
        self.smiles = self.smiles[order]
        self.mass = self.mass[order]
        fp = df["fp"].to_numpy()
        dim = len(fp[0]) if len(fp) and fp[0] is not None else 0
        self.fp_dim = dim
        self.fp = np.zeros((len(df), dim), dtype=np.uint8)
        self.has_fp = np.zeros(len(df), dtype=bool)
        for i, v in enumerate(fp):
            if v is not None:
                self.fp[i] = np.asarray(v, dtype=np.uint8)
                self.has_fp[i] = True
        self.fp = self.fp[order]
        self.has_fp = self.has_fp[order]
        self._lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
        # key -> sorted index, and key -> mass, built once
        self._key_to_idx = {k: i for i, k in enumerate(self.keys)}
        self._key_mass = dict(zip(self.keys, self.mass))
        print(f"StructureStore: {len(self):,} structures, fp_dim={dim}, "
              f"mass {self.mass.min():.1f}-{self.mass.max():.1f} Da")

    def __len__(self):
        return len(self.keys)

    def window(self, m: float, ppm: float) -> np.ndarray:
        tol = abs(m) * ppm * 1e-6
        lo = np.searchsorted(self.mass, m - tol, side="left")
        hi = np.searchsorted(self.mass, m + tol, side="right")
        return np.arange(lo, hi)

    def between(self, lo_m: float, hi_m: float) -> np.ndarray:
        lo = np.searchsorted(self.mass, lo_m, side="left")
        hi = np.searchsorted(self.mass, hi_m, side="right")
        return np.arange(lo, hi)

    def tanimoto(self, qfp: np.ndarray, idx: np.ndarray) -> np.ndarray:
        sub = self.fp[idx]
        inter = self._lut[np.minimum(sub, qfp[None, :])].sum(axis=1).astype(np.float64)
        uni = (self._lut[sub].sum(axis=1).astype(np.float64)
               + self._lut[qfp].sum() - inter)
        out = np.zeros(len(idx), dtype=np.float64)
        nz = uni > 0
        out[nz] = inter[nz] / uni[nz]
        return out


class SpectralIndex:
    """Library spectra indexed by compound key, plus a neutral-mass array."""

    def __init__(self, path: str):
        t0 = time.time()
        df = pd.read_parquet(path)
        self.keys = df["key"].to_numpy()
        self.prec = df["precursor_mz"].to_numpy(dtype=np.float64)
        self.adduct = df["adduct"].to_numpy()
        self.is_timstof = df["is_timstof"].to_numpy()
        self.lib = df["ingest_lib"].to_numpy()
        # store cleaned peaks as separate object arrays; 2.4M spectra is too many
        # for per-row Python lists to stay cheap, but numpy object arrays of
        # float32 arrays are acceptable at ~1.3 GB.
        self.mz = np.empty(len(df), dtype=object)
        self.inten = np.empty(len(df), dtype=object)
        for i, (mzb, ib) in enumerate(zip(df["mz"].to_numpy(), df["intensity"].to_numpy())):
            self.mz[i] = np.frombuffer(mzb, dtype=np.float32)
            self.inten[i] = np.frombuffer(ib, dtype=np.float32)
        # neutral mass per spectrum, derived from its compound's structure mass
        self.by_key: dict[str, list[int]] = defaultdict(list)
        for i, k in enumerate(self.keys):
            self.by_key[k].append(i)
        print(f"SpectralIndex: {len(df):,} spectra for {len(self.by_key):,} compounds "
              f"({time.time()-t0:.0f}s)")

    def __len__(self):
        return len(self.keys)


# --------------------------------------------------------------------- driver

def score_molecule(query_spectra, store: StructureStore, spec_idx: SpectralIndex,
                   use_analog: bool = True, tol: float = DEFAULT_TOL,
                   analog_weight: float = 1.5, fp_scale: float = 2.0):
    """
    Rank candidate structures for one molecule.

    Two channels, fused linearly (the published recipe; a GBDT on top is
    deliberately avoided because it is documented to collapse out of domain):

      direct(c)  = max over the molecule's spectra of the unshifted peak-match
                   cosine against c's own library spectra.
      analog(c)  = max over (query spectrum q, library compound a, candidate c):
                       SimMod(q, a)^2 * Tanimoto(fp_c, fp_a)
                   weighted by how confidently c is tied to a.

    The fingerprint term is what keeps analog propagation honest: a shifted
    spectral match only lifts candidates that are genuinely structurally like
    the library compound that produced the shift, so a coincidental shift does
    not drag in unrelated chemistry.

    Returns (ranked, direct_by_key, analog_by_key, cand).
    """
    # 1. candidate union: tight ppm window, wide ppm window, and analog window
    cand: dict[str, int] = {}
    for s in query_spectra:
        m = s.neutral_mass
        if not np.isfinite(m):
            continue
        for ppm in (RETRIEVAL_PPM, RETRIEVAL_PPM_WIDE):
            for i in store.window(m, ppm):
                cand.setdefault(store.keys[i], int(i))
        if use_analog:
            for i in store.between(m - ANALOG_DM, m + ANALOG_DM):
                cand.setdefault(store.keys[i], int(i))
    if not cand:
        return [], {}, {}, {}

    cand_keys = list(cand)
    cand_idx = np.array([cand[k] for k in cand_keys], dtype=np.int64)
    key_pos = {k: i for i, k in enumerate(cand_keys)}
    direct = np.zeros(len(cand_keys), dtype=np.float64)
    cand_key_set = set(cand_keys)

    q_masses = [s.neutral_mass for s in query_spectra if np.isfinite(s.neutral_mass)]
    if not q_masses:
        return [], {}, {}, {}

    # 2. Restrict the library scan.
    # Scanning every library spectrum whose compound is within +/-200 Da is far
    # too slow (tens of thousands of greedy peak matches per molecule). Two much
    # tighter, well-motivated sets are enough:
    #   (a) spectra of the candidates themselves  -> the direct channel, and the
    #       only way a true exact hit can ever be found;
    #   (b) spectra of compounds within +/-ANALOG_DM of a query mass -> the
    #       analog channel, i.e. plausible near-mass analogues.
    km = store._key_mass
    scan_ids: set[int] = set()
    for k in cand_key_set:
        scan_ids.update(spec_idx.by_key.get(k, ()))
    for m in q_masses:
        for i in store.between(m - ANALOG_DM, m + ANALOG_DM):
            scan_ids.update(spec_idx.by_key.get(store.keys[i], ()))
    if not scan_ids:
        return [], {}, {}, {}

    scan = np.fromiter(sorted(scan_ids), dtype=np.int64)

    # 3. per-library-compound spectral evidence
    lib_direct: dict[str, float] = {}
    # shifted matches: library key -> best gated SimMod**fp_scale
    lib_shift: dict[str, float] = {}
    for s in query_spectra:
        if not np.isfinite(s.neutral_mass):
            continue
        qm = s.neutral_mass
        for si in scan:
            k = spec_idx.keys[si]
            m_lib = km.get(k)
            if m_lib is None:
                continue
            dm = qm - m_lib
            if abs(dm) > ANALOG_DM_MAX:
                continue
            sc, nm, mode = modified_cosine(s.mz, s.intensity,
                                           spec_idx.mz[si], spec_idx.inten[si],
                                           dm, tol)
            if sc <= 0 or nm == 0:
                continue
            if mode == "direct":
                if sc > lib_direct.get(k, 0.0):
                    lib_direct[k] = sc
            elif use_analog:
                gated = sc ** fp_scale
                if gated > lib_shift.get(k, 0.0):
                    lib_shift[k] = gated
            # a shifted match still carries direct evidence at its own mass
            if sc > lib_direct.get(k, 0.0) and mode == "direct":
                lib_direct[k] = sc

    # direct evidence lands on the compound's own candidate entry
    for k, v in lib_direct.items():
        p = key_pos.get(k)
        if p is not None and v > direct[p]:
            direct[p] = v

    # 4. propagate shifted evidence to candidates via fingerprint similarity
    analog = np.zeros(len(cand_keys), dtype=np.float64)
    if use_analog and lib_shift:
        valid = store.has_fp[cand_idx]
        valid_pos = np.flatnonzero(valid)
        if valid_pos.size:
            sub = cand_idx[valid_pos]
            for hk, strength in lib_shift.items():
                hi = store._key_to_idx.get(hk)
                if hi is None or not store.has_fp[hi]:
                    continue
                tan = store.tanimoto(store.fp[hi], sub)      # vs each candidate
                contrib = strength * tan * analog_weight
                np.maximum.at(analog, valid_pos, contrib)

    final = direct + analog
    order = np.argsort(-final)
    ranked = [(cand_keys[i], float(final[i]), float(direct[i]), float(analog[i]))
              for i in order]
    return (ranked,
            dict(zip(cand_keys, direct.tolist())),
            dict(zip(cand_keys, analog.tolist())),
            cand)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default=V_A, choices=[V_A, V_B, V_C, "all"])
    ap.add_argument("--limit", type=int, default=0, help="limit molecules (debug)")
    ap.add_argument("--no-analog", action="store_true")
    ap.add_argument("--out", default=os.path.join(OUT, "baseline_v1.json"))
    a = ap.parse_args()

    print("loading structures...")
    store = StructureStore(STRUCTS)
    print("loading library spectra...")
    spec_idx = SpectralIndex(LIBS)

    print("\nbuilding validation folds from train...")
    cols = ["molecule_id", "spectrum_id", "ms2_mzs", "ms2_normalized_intensities",
            "precursor_mz", "adduct", "normalized_smiles", "inchikey14",
            "ingest_lib", "molecular_formula"]
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(TRAIN)
    frames = []
    for b in pf.iter_batches(batch_size=200_000, columns=cols):
        frames.append(b.to_pandas())
    train = pd.concat(frames, ignore_index=True)
    print(f"  train frame: {len(train):,} rows")
    folds = build_folds(train)
    for name, f in folds.items():
        print(f"  {name}: {len(f)} queries, library {len(f.library):,}")

    results = {}
    names = list(folds) if a.fold == "all" else [a.fold]
    for name in names:
        fold = folds[name]
        mids = list(fold.queries)
        if a.limit:
            mids = mids[:a.limit]
        print(f"\n=== {name}: scoring {len(mids)} molecules "
              f"(analog={'off' if a.no_analog else 'on'}) ===")
        preds, cand_keys_map = {}, {}
        t0 = time.time()
        for n, mid in enumerate(mids, 1):
            mol = fold.queries[mid]
            ranked, direct, analog, cand = score_molecule(
                mol.spectra, store, spec_idx, use_analog=not a.no_analog)
            preds[mid] = [store.smiles[cand[k]] for k, *_ in ranked[:TOP_CANDIDATES]]
            cand_keys_map[mid] = [k for k, *_ in ranked]
            if n % 10 == 0 or n == len(mids):
                el = time.time() - t0
                print(f"  {n}/{len(mids)}  {el:.0f}s  ({el/max(n,1):.2f}s/mol)", flush=True)
        truths = {m: fold.truths[m] for m in mids if m in fold.truths}
        res = evaluate(name, preds, truths, cand_keys_map, verbose=True)
        res["n_scored"] = len(mids)
        res["seconds"] = round(time.time() - t0, 1)
        res["analog"] = not a.no_analog
        results[name] = res

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nwrote {a.out}")
    for name, r in results.items():
        print(f"  {name}: MRR@25={r['mrr25']:.4f}  hit@25={r['hit@25']:.3f}")


if __name__ == "__main__":
    main()
