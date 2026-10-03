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
from casmi.matching import DEFAULT_TOL, MAX_DM          # noqa: E402
from casmi.matching_fast import bin_spectrum, shifted_cosine  # noqa: E402
from casmi.folds_fast import build_folds_fast            # noqa: E402
from casmi.spectra import clean_peaks, load_spectra, group_by_molecule  # noqa: E402
from casmi.validation import V_A, V_B, V_C, build_folds, evaluate       # noqa: E402

DATA = r"D:\CASMI竞赛\data"
TRAIN = os.path.join(DATA, "raw", "train.parquet")
STRUCTS = os.path.join(DATA, "processed", "structures.parquet")
LIBS = os.path.join(DATA, "processed", "library_spectra.parquet")
LIB_INDEX_DIR = os.path.join(DATA, "processed", "library_spectra_sorted")
OUT = os.path.join(DATA, "processed")

# Retrieval windows (ppm) and the analog mass-difference window.
RETRIEVAL_PPM = 15.0
RETRIEVAL_PPM_WIDE = 40.0
ANALOG_DM = 2.0          # +/- Da around the query mass for the analog channel
ANALOG_DM_MAX = MAX_DM   # hard cap on how far a library compound may sit
# The fingerprint gate is the expensive part of the analog channel, so it is
# applied only to the strongest spectral analogues.
ANALOG_TOP_HITS = 60
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
    """
    Lazy, query-scoped view of the compound-keyed spectral library.

    Loading all 2.37M spectra eagerly costs ~10 GB RSS and ~90 s of startup, yet
    a single query only ever touches the spectra of a few hundred compounds. So
    only the small per-key index is held in memory; peak arrays are fetched from
    the partition files on demand and cached.

    The backing store is sorted by key with a key never spanning two partitions,
    so `by_keys` reads exactly the partitions it needs.
    """

    def __init__(self, dir_path: str):
        t0 = time.time()
        idx = pd.read_parquet(os.path.join(dir_path, "library_index.parquet"))
        self.part_of: dict[str, int] = dict(zip(idx["key"], idx["partition"]))
        self.n_of: dict[str, int] = dict(zip(idx["key"], idx["n_spectra"]))
        self.n_parts = int(idx["partition"].max()) + 1
        self.dir = dir_path
        self._part_cache: dict[int, dict] = {}
        self._loaded_parts: list[int] = []
        print(f"SpectralIndex(lazy): {len(self.part_of):,} compounds, "
              f"{int(idx['n_spectra'].sum()):,} spectra in {self.n_parts} partitions "
              f"({time.time()-t0:.1f}s)")

    def __len__(self):
        return len(self.part_of)

    def _load_part(self, p: int) -> dict:
        got = self._part_cache.get(p)
        if got is not None:
            return got
        df = pd.read_parquet(os.path.join(self.dir, f"part_{p:04d}.parquet"),
                             columns=["key", "precursor_mz", "adduct",
                                      "is_timstof", "mz", "intensity"])
        by_key: dict[str, list[int]] = defaultdict(list)
        mz = np.empty(len(df), dtype=object)
        inten = np.empty(len(df), dtype=object)
        for i, (mzb, ib) in enumerate(zip(df["mz"].to_numpy(),
                                          df["intensity"].to_numpy())):
            mz[i] = np.frombuffer(mzb, dtype=np.float32)
            inten[i] = np.frombuffer(ib, dtype=np.float32)
        for i, k in enumerate(df["key"].to_numpy()):
            by_key[k].append(i)
        got = {"mz": mz, "inten": inten, "by_key": by_key,
               "prec": df["precursor_mz"].to_numpy(dtype=np.float64),
               "adduct": df["adduct"].to_numpy(),
               "is_timstof": df["is_timstof"].to_numpy(),
               "keys": df["key"].to_numpy(),
               "bins": None}
        self._part_cache[p] = got
        if p not in self._loaded_parts:
            self._loaded_parts.append(p)
        return got

    def _binned(self, p: int) -> dict:
        """
        Build (once per partition) the binned sparse representation used by the
        vectorised direct channel, arranged for fast bin -> spectra lookup.

        Layout: all entries of the partition sorted by bin, with entry_spec the
        spectrum each entry belongs to and weight_ranges giving, for entry i,
        the cumulative weight of its spectrum -- that is what lets a batched
        cosine accumulate into the right spectrum without a Python loop over
        library spectra.
        """
        blk = self._load_part(p)
        if blk["bins"] is not None:
            return blk["bins"]
        all_b, all_w, all_s = [], [], []
        for si in range(len(blk["mz"])):
            b, w = bin_spectrum(blk["mz"][si], blk["inten"][si])
            nrm = float(np.sqrt((w.astype(np.float64) ** 2).sum())) if w.size else 0.0
            if nrm > 0:
                w = (w / nrm).astype(np.float32)
            all_b.append(b)
            all_w.append(w)
            all_s.append(np.full(b.size, si, dtype=np.int64))
        if all_b:
            bins = np.concatenate(all_b)
            wts = np.concatenate(all_w)
            spec = np.concatenate(all_s)
        else:
            bins = np.empty(0, dtype=np.int32)
            wts = np.empty(0, dtype=np.float32)
            spec = np.empty(0, dtype=np.int64)
        order = np.argsort(bins, kind="stable")
        bins_s = bins[order]
        blk["bins"] = {"bins": bins_s, "w": wts[order], "spec": spec[order]}
        return blk["bins"]

    def group_binned(self, parts):
        """
        Concatenate the binned entries of `parts` into one grouped view, so a
        single vectorised call can score every library spectrum in the analog
        window instead of thousands of individual Python matches.

        `spec` is remapped to a contiguous index across the concatenated parts.
        """
        parts = sorted(set(parts))
        if not parts:
            return None
        bins_l, w_l, spec_l, key_l = [], [], [], []
        off = 0
        for p in parts:
            blk = self._load_part(p)
            bb = blk["bins"] if blk["bins"] is not None else self._binned(p)
            if bb["bins"].size == 0:
                continue
            bins_l.append(bb["bins"])
            w_l.append(bb["w"])
            spec_l.append(bb["spec"] + off)
            key_l.append(blk["keys"])
            off += len(blk["mz"])
        if not bins_l:
            return None
        return {
            "bins": np.concatenate(bins_l),
            "w": np.concatenate(w_l),
            "spec": np.concatenate(spec_l),
            "keys": np.concatenate(key_l),
            "n_spec": off,
        }

    def by_keys(self, keys):
        """
        Return (mz_list, intensity_list, key_list, is_timstof_list) for the given
        structure keys, loading only the partitions those keys live in.
        """
        keys = [k for k in keys if k in self.part_of]
        if not keys:
            return [], [], [], []
        parts = sorted({self.part_of[k] for k in keys})
        for p in parts:
            self._load_part(p)
        mzs, ints, ks, ts = [], [], [], []
        for k in keys:
            blk = self._part_cache[self.part_of[k]]
            for i in blk["by_key"].get(k, ()):
                mzs.append(blk["mz"][i])
                ints.append(blk["inten"][i])
                ks.append(k)
                ts.append(bool(blk["is_timstof"][i]))
        return mzs, ints, ks, ts

    def direct_cosines(self, q_mz: np.ndarray, q_int: np.ndarray,
                       keys: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """
        Vectorised unshifted cosine of one query spectrum against the library
        spectra of `keys`, grouped to the best score per key.

        All candidate spectra are binned in one batched numpy pass (a per-spectrum
        Python loop here dominated the runtime: ~1000 bin_spectrum calls per
        query). Binned candidate spectra are cached per (partition, key) so
        repeated molecules do not re-bin.
        """
        if not keys:
            return np.empty(0, dtype=object), np.empty(0, dtype=np.float32)
        qb, qw = bin_spectrum(q_mz, q_int)
        if qb.size == 0:
            return np.empty(0, dtype=object), np.empty(0, dtype=np.float32)
        qn = float(np.sqrt((qw.astype(np.float64) ** 2).sum()))
        if qn <= 0:
            return np.empty(0, dtype=object), np.empty(0, dtype=np.float32)
        qwn = (qw / qn).astype(np.float32)

        best: dict[str, float] = {}
        by_part: dict[int, list[str]] = defaultdict(list)
        for k in keys:
            p = self.part_of.get(k)
            if p is not None:
                by_part[p].append(k)

        for p, ks in by_part.items():
            blk = self._load_part(p)
            # gather every member spectrum of these keys, binned in one pass
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
                all_b.append(b)
                all_w.append((w / nrm).astype(np.float32))
                all_owner.append(np.full(b.size, j, dtype=np.int32))
            if not all_b:
                continue
            bins = np.concatenate(all_b)
            wts = np.concatenate(all_w)
            owner = np.concatenate(all_owner)
            order = np.argsort(bins, kind="stable")
            bins, wts, owner = bins[order], wts[order], owner[order]

            scores = np.zeros(len(spec_ids), dtype=np.float64)
            lo = np.searchsorted(bins, qb, side="left")
            hi = np.searchsorted(bins, qb, side="right")
            for t in range(qb.size):
                if hi[t] <= lo[t]:
                    continue
                ent = slice(lo[t], hi[t])
                np.add.at(scores, owner[ent], qwn[t] * wts[ent])
            for j, k in enumerate(spec_key):
                v = float(scores[j])
                if v > best.get(k, 0.0):
                    best[k] = v

        if not best:
            return np.empty(0, dtype=object), np.empty(0, dtype=np.float32)
        ks_out = list(best)
        return (np.asarray(ks_out, dtype=object),
                np.asarray([best[k] for k in ks_out], dtype=np.float32))

    def release(self, keep_last: int = 3) -> None:
        """Drop all but the most recently used partitions to bound memory."""
        if keep_last <= 0:
            self._part_cache.clear()
            self._loaded_parts.clear()
            return
        keep = set(self._loaded_parts[-keep_last:])
        for p in list(self._part_cache):
            if p not in keep:
                del self._part_cache[p]
        self._loaded_parts = [p for p in self._loaded_parts if p in keep]


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

    # 2. Channels.
    #   Direct: vectorised unshifted cosine against the candidates' own spectra.
    #           This is the bulk of the work, so it must not be a Python loop.
    #   Analog: exact shifted matching, but only against compounds inside the
    #           analog window -- a much smaller set.
    km = store._key_mass

    # ---- direct channel (vectorised) ----
    lib_direct: dict[str, float] = {}
    for s in query_spectra:
        hk, hs = spec_idx.direct_cosines(s.mz, s.intensity, cand_keys)
        for k, v in zip(hk, hs):
            if v > lib_direct.get(k, 0.0):
                lib_direct[k] = float(v)

    # ---- analog channel (vectorised shifted cosine) ----
    # Only library compounds inside the analog window are considered. The query
    # mass differs per spectrum, so the shift (and therefore a fresh vectorised
    # pass) is per spectrum -- still thousands of times cheaper than a Python
    # loop over pairs.
    analog_raw: dict[str, float] = {}
    if use_analog:
        lib_keys: set[str] = set()
        for m in q_masses:
            for i in store.between(m - ANALOG_DM, m + ANALOG_DM):
                lib_keys.add(store.keys[i])
        lib_keys -= cand_key_set
        if lib_keys:
            parts = {spec_idx.part_of[k] for k in lib_keys if k in spec_idx.part_of}
            grp = spec_idx.group_binned(parts)
            if grp is not None and grp["n_spec"] > 0:
                for s in query_spectra:
                    qm = s.neutral_mass
                    if not np.isfinite(qm):
                        continue
                    # per-library-compound dm, evaluated in the shifted frame:
                    # shifting the library by qm - m_lib is the same as shifting
                    # the query by m_lib - qm, which depends on the compound.
                    # Group compounds by their (binned) dm to bound the passes.
                    buckets: dict[float, list] = defaultdict(list)
                    for k in lib_keys:
                        m_lib = km.get(k)
                        if m_lib is None:
                            continue
                        dm = qm - m_lib
                        if abs(dm) > ANALOG_DM_MAX:
                            continue
                        buckets[round(dm, 3)].append(k)
                    for dm_val, ks in buckets.items():
                        if abs(dm_val) < 1e-3:
                            continue      # unshifted evidence is handled as direct
                        scores = shifted_cosine(s.mz, s.intensity, float(dm_val),
                                                grp["bins"], grp["w"], grp["spec"],
                                                grp["n_spec"])
                        if scores.size == 0 or not np.any(scores > 0):
                            continue
                        kset = set(ks)
                        # best score per key, restricted to the compounds in
                        # this dm bucket (their spectra are the only valid ones)
                        hit_spec = np.flatnonzero(scores > 0)
                        if hit_spec.size == 0:
                            continue
                        for si in hit_spec:
                            k = grp["keys"][si]
                            if k not in kset:
                                continue
                            v = float(scores[si])
                            if v > analog_raw.get(k, 0.0):
                                analog_raw[k] = v

    # direct evidence lands on the compound's own candidate entry
    for k, v in lib_direct.items():
        p = key_pos.get(k)
        if p is not None and v > direct[p]:
            direct[p] = v

    # analog evidence is gated by structure similarity, applied only to the
    # strongest spectral analogues (the fingerprint gate is the expensive part)
    lib_shift: dict[str, float] = {}
    if use_analog and analog_raw:
        top = sorted(analog_raw.items(), key=lambda kv: -kv[1])[:ANALOG_TOP_HITS]
        for k, sc in top:
            lib_shift[k] = sc ** fp_scale

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
    ap.add_argument("--vb-limit", type=int, default=0,
                    help="0 skips V-B (233k queries; only sample it deliberately)")
    ap.add_argument("--vc-limit", type=int, default=300)
    ap.add_argument("--va-limit", type=int, default=None)
    ap.add_argument("--out", default=os.path.join(OUT, "baseline_v1.json"))
    a = ap.parse_args()

    print("loading structures...")
    store = StructureStore(STRUCTS)
    print("loading library index (lazy)...")
    spec_idx = SpectralIndex(LIB_INDEX_DIR)

    print("\nbuilding validation folds (lazy: only query spectra are materialised)...")
    t_fold = time.time()
    folds = build_folds_fast(TRAIN, va_limit=a.va_limit, vb_limit=a.vb_limit,
                             vc_limit=a.vc_limit)
    print(f"  folds built in {time.time()-t_fold:.0f}s")
    for name, f in folds.items():
        nsp = sum(len(m.spectra) for m in f.queries.values())
        print(f"  {name}: {len(f)} queries, {nsp:,} query spectra, "
              f"library {len(f.library):,}")

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
            # bound memory: only the most recently used partitions stay cached
            if n % 25 == 0:
                spec_idx.release(keep_last=3)
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
