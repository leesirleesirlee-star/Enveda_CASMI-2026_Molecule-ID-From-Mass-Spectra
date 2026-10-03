"""
Fast fold construction for evaluation.

Why this exists: the generic `build_folds` materialises all 2,539,608 spectra
into Python `Spectrum` objects (~13 GB RSS, minutes of CPU) even when a run only
evaluates 250 queries. This module builds the same three folds but creates
Spectrum objects only for the molecules a run actually needs, and does the
expensive per-row peak cleaning once, streaming.

Fold definitions (identical to casmi.validation):
  V-A  structure seen in enveda-np-examples (timsTOF). Queries use ONLY those
       calibration spectra; the library keeps every other source -> an
       instrument-transfer test.
  V-B  same-source holdout, answers retained  -> retrieval ceiling
  V-C  identity-disjoint holdout, answers removed -> strict extrapolation
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from casmi.core import smiles_to_inchikey14          # noqa: E402
from casmi.spectra import Molecule, Spectrum, clean_peaks  # noqa: E402
from casmi.validation import Fold, V_A, V_B, V_C, CALIBRATION_LIBS  # noqa: E402

COLS = ["ms2_mzs", "ms2_normalized_intensities", "precursor_mz", "adduct",
        "normalized_smiles", "inchikey14", "ingest_lib", "molecular_formula",
        "instrument_type", "collision_energy_ev", "num_peaks", "base_peak_intensity"]


def scan_keys_and_libs(path: str, batch: int = 400_000,
                       verbose: bool = True) -> tuple[dict, dict, dict]:
    """
    Single cheap pass over the label columns only (no peaks, no Spectrum objects).

    Returns:
      key_to_smiles : structure key -> SMILES
      key_libs      : structure key -> set of ingest_lib it appears in
      calib_keys    : keys that occur under an instrument-matched source
    """
    key_to_smiles: dict[str, str] = {}
    key_libs: dict[str, set] = defaultdict(set)
    calib_keys: set[str] = set()
    pf = pq.ParquetFile(path)
    n = 0
    for b in pf.iter_batches(batch_size=batch, columns=["normalized_smiles",
                                                        "inchikey14", "ingest_lib"]):
        df = b.to_pandas()
        n += len(df)
        for smi, k, lib in zip(df["normalized_smiles"], df["inchikey14"],
                               df["ingest_lib"]):
            if smi is None or (isinstance(smi, float) and np.isnan(smi)):
                continue
            kk = k if isinstance(k, str) and k else smiles_to_inchikey14(smi)
            if not kk:
                continue
            key_to_smiles.setdefault(kk, smi)
            key_libs[kk].add(lib)
            if lib in CALIBRATION_LIBS:
                calib_keys.add(kk)
        if verbose:
            print(f"    label scan: {n:,} rows, {len(key_to_smiles):,} keys", flush=True)
    return key_to_smiles, dict(key_libs), calib_keys


def load_query_spectra(path: str, wanted: set[str], batch: int = 200_000,
                       verbose: bool = True) -> dict[str, Molecule]:
    """
    Build Molecule objects for `wanted` structure keys only.

    Peaks are cleaned here, once, for just the spectra that belong to a wanted
    structure -- so memory scales with the query set, not with the corpus.
    """
    mols: dict[str, Molecule] = {}
    pf = pq.ParquetFile(path)
    seen = 0
    for b in pf.iter_batches(batch_size=batch, columns=COLS):
        df = b.to_pandas()
        seen += len(df)
        # cheap pre-filter on the key column before touching peaks
        keys = []
        for smi, k in zip(df["normalized_smiles"], df["inchikey14"]):
            kk = k if isinstance(k, str) and k else smiles_to_inchikey14(smi)
            keys.append(kk)
        keys = np.asarray(keys, dtype=object)
        mask = np.array([k in wanted for k in keys])
        if not mask.any():
            continue
        sub = df[mask]
        sub_keys = keys[mask]
        for (_, row), kk in zip(sub.iterrows(), sub_keys):
            mz, inten = clean_peaks(row["ms2_mzs"], row["ms2_normalized_intensities"])
            if mz.size < 3:
                continue
            ce = row.get("collision_energy_ev", None)
            try:
                ce = () if ce is None else tuple(float(x) for x in ce)
            except (TypeError, ValueError):
                ce = ()
            sp = Spectrum(
                spectrum_id=f"{kk}_{len(mols.get(kk, Molecule(kk)).spectra)}",
                molecule_id=str(kk), mz=mz, intensity=inten,
                precursor_mz=float(row["precursor_mz"]),
                adduct=str(row["adduct"]),
                instrument_type=str(row.get("instrument_type", "") or ""),
                ingest_lib=str(row.get("ingest_lib", "") or ""),
                collision_energy=ce,
            )
            mols.setdefault(kk, Molecule(kk)).spectra.append(sp)
        if verbose:
            print(f"    query scan: {seen:,} rows, {len(mols):,} query structures built",
                  flush=True)
    return mols


def _make_lib(key_to_smiles: dict[str, str], remove: set[str]) -> dict[str, dict]:
    return {k: {"key": k, "smiles": s, "ingest_libs": {"all"}}
            for k, s in key_to_smiles.items() if k not in remove}


def build_folds_fast(path: str, vc_fraction: float = 0.15, seed: int = 42,
                     vb_limit: int | None = None, vc_limit: int | None = None,
                     va_limit: int | None = None,
                     verbose: bool = True) -> dict[str, Fold]:
    key_to_smiles, key_libs, calib_keys = scan_keys_and_libs(path, verbose=verbose)
    rng = np.random.default_rng(seed)

    # ---- V-A: calibration structures ----
    va_keys = sorted(calib_keys & set(key_to_smiles))
    if va_limit is not None:
        va_keys = va_keys[:max(0, va_limit)]
    mols_a = load_query_spectra(path, set(va_keys), verbose=verbose)
    # keep only the calibration-source spectra for the queries
    queries_a = {}
    for k, mol in mols_a.items():
        sp = [s for s in mol.spectra if s.ingest_lib in CALIBRATION_LIBS]
        if sp:
            queries_a[k] = Molecule(k, sp)
    truth_a = {k: key_to_smiles[k] for k in queries_a}
    folds: dict[str, Fold] = {}
    if queries_a:
        folds[V_A] = Fold(V_A, queries_a, truth_a, _make_lib(key_to_smiles, set()),
                          "instrument-matched natural products (enveda-np-examples, "
                          "timsTOF); queries use only calibration spectra, library "
                          "keeps all other sources -> instrument-transfer test")

    # ---- V-A-hard: same queries, own structure removed from the library ----
    # V-A scores a perfect 1.0 because every calibration structure also occurs
    # in other ingest libraries, so a same-structure spectrum is always present
    # and an exact hit is trivially available. That makes V-A a poor proxy for a
    # hidden test set of novel molecules. V-A-hard removes the query's own
    # structure from the candidate universe, which is the realistic setting:
    # the true answer is NOT an exact library hit, so only analog propagation
    # and the other evidence channels can surface it.
    if queries_a:
        lib_hard = _make_lib(key_to_smiles, set(queries_a))
        leaked = set(truth_a) & set(lib_hard)
        assert not leaked, f"V_A_hard library leaks {len(leaked)} answers"
        folds["V_A_hard"] = Fold(
            "V_A_hard", dict(queries_a), dict(truth_a), lib_hard,
            "instrument-matched queries with their OWN structure removed from the "
            "library -> exact hits impossible, analog/evidence channels required")

    # ---- V-B / V-C: split remaining structures ----
    remaining = sorted(set(key_to_smiles) - set(va_keys))
    perm = rng.permutation(len(remaining))
    n_c = max(1, int(len(remaining) * vc_fraction))
    vc_keys = [remaining[i] for i in perm[:n_c]]
    vb_keys = [remaining[i] for i in perm[n_c:]]
    # NB: limits are None-or-positive; 0 must mean "skip this fold entirely",
    # so test for None explicitly rather than relying on truthiness.
    if vc_limit is not None:
        vc_keys = vc_keys[:max(0, vc_limit)]
    if vb_limit is not None:
        vb_keys = vb_keys[:max(0, vb_limit)]

    for name, keys, desc, remove_truth in (
        (V_B, vb_keys, "same-source holdout, answer RETAINED in library "
                       "(retrieval ceiling)", False),
        (V_C, vc_keys, "identity-disjoint holdout, answer REMOVED from library "
                       "(strict extrapolation)", True),
    ):
        if not keys:
            continue
        mols = load_query_spectra(path, set(keys), verbose=verbose)
        truth = {k: key_to_smiles[k] for k in mols if k in key_to_smiles}
        if not truth:
            continue
        remove = {smiles_to_inchikey14(t) for t in truth.values()} if remove_truth else set()
        remove.discard(None)
        lib = _make_lib(key_to_smiles, remove)
        # Invariant: a "answers removed" fold must actually be missing its
        # answers. Assert it here, because a scorer that bypasses `fold.library`
        # silently turns the whole protocol into a no-op (this really happened:
        # V-C and V_A-hard both reported ~1.0 while claiming answers were gone).
        if remove_truth:
            leaked = set(truth) & set(lib)
            assert not leaked, f"{name} library leaks {len(leaked)} answers"
        folds[name] = Fold(name, mols, truth, lib, desc)

    return folds
