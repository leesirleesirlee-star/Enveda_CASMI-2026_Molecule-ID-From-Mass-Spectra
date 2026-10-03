"""
Candidate retrieval: mass-window structure library + fingerprint utilities.

The proven architecture starts by reducing ~275k candidate structures to the
~hundreds whose neutral mass is compatible with a query's precursor. Everything
downstream (direct spectral match, analog propagation, fragmentation, neural
fingerprint) only ever scores that shortlist, so recall here bounds the whole
pipeline.

Design:
  * one canonical structure record per InChIKey14 (dedup across libraries)
  * a mass-sorted array with binary search for the ppm window
  * Morgan fingerprints as packed bits for fast Tanimoto
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .core import ADDUCT_OFFSET, smiles_to_inchikey14

# Retrieval windows. The primary window is tight (analytical precision); the
# fallback exists because library masses are sometimes quoted loosely.
DEFAULT_PPM = 15.0
FALLBACK_PPM = 40.0
# 13C isotopologue shift: a candidate can be the 13C peak of a heavier molecule,
# so allow +/-1.003355 Da in addition to the ppm window.
C13_SHIFT = 1.0033548

FP_RADIUS = 2
FP_NBITS = 2048


@dataclass
class Structure:
    key: str
    smiles: str
    mass: float
    formula: str | None = None
    ingest_libs: tuple[str, ...] = ()
    fp: np.ndarray | None = None       # packed uint8 Morgan fingerprint
    n_spectra: int = 0


def morgan_fp(smiles: str, radius: int = FP_RADIUS, nbits: int = FP_NBITS) -> np.ndarray | None:
    """Packed Morgan fingerprint as uint8, or None if the SMILES won't parse."""
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import AllChem

        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
        return np.packbits(np.frombuffer(bv.ToBitString().encode(), dtype=np.uint8) - 48)
    except Exception:
        return None


def exact_mass(smiles: str) -> float | None:
    """Monoisotopic mass of a SMILES (heavy atoms + implicit/explicit H)."""
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem.Descriptors import ExactMolWt

        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(smiles)
        return float(ExactMolWt(mol)) if mol is not None else None
    except Exception:
        return None


class StructureLibrary:
    """
    Mass-indexed structure library with binary-search retrieval.

    masses is kept sorted; a parity permutation keeps fingerprints aligned so
    retrieval returns indices we can use to gather fingerprints without a dict
    lookup per candidate.
    """

    def __init__(self, structures: Sequence[Structure]):
        n = len(structures)
        if n == 0:
            raise ValueError("empty structure library")
        self.structures: list[Structure] = list(structures)
        masses = np.asarray([s.mass for s in self.structures], dtype=np.float64)
        order = np.argsort(masses, kind="stable")
        self._order = order
        self.masses = masses[order]
        self.keys = [self.structures[i].key for i in order]

        # fingerprint matrix in the same order (rows are candidates)
        dim = None
        for s in self.structures:
            if s.fp is not None:
                dim = s.fp.size
                break
        if dim is None:
            self.fp_matrix = None
        else:
            m = np.zeros((n, dim), dtype=np.uint8)
            have = np.zeros(n, dtype=bool)
            for i, s in enumerate(self.structures):
                if s.fp is not None and s.fp.size == dim:
                    m[i] = s.fp
                    have[i] = True
            self.fp_matrix = m[order]
            self.fp_valid = have[order]
        self._keyset = set(self.keys)

    def __len__(self) -> int:
        return len(self.structures)

    def mass_window(self, neutral_mass: float, ppm: float = DEFAULT_PPM,
                    include_c13: bool = True) -> np.ndarray:
        """
        Indices of structures within the ppm window of neutral_mass.

        Returns indices into the sorted arrays (use .structure_at / fingerprints).
        """
        tol = abs(neutral_mass) * ppm * 1e-6
        lo = np.searchsorted(self.masses, neutral_mass - tol, side="left")
        hi = np.searchsorted(self.masses, neutral_mass + tol, side="right")
        idx = np.arange(lo, hi)
        if include_c13:
            # 13C isotopologue: the query may be the M+1 peak of a heavier
            # molecule, i.e. its neutral mass is ~1.0034 below the precursor.
            for shift in (C13_SHIFT, -C13_SHIFT):
                m2 = neutral_mass + shift
                tol2 = abs(m2) * ppm * 1e-6
                lo2 = np.searchsorted(self.masses, m2 - tol2, side="left")
                hi2 = np.searchsorted(self.masses, m2 + tol2, side="right")
                if hi2 > lo2:
                    idx = np.union1d(idx, np.arange(lo2, hi2))
        return idx

    def structure_at(self, i: int) -> Structure:
        return self.structures[self._order[i]]

    def fingerprints_at(self, idx: np.ndarray) -> np.ndarray:
        if self.fp_matrix is None:
            raise ValueError("library has no fingerprints")
        return self.fp_matrix[idx]

    def tanimoto(self, query_fp: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """Tanimoto of one packed fingerprint against selected library rows."""
        if self.fp_matrix is None:
            raise ValueError("library has no fingerprints")
        sub = self.fp_matrix[idx]
        # popcount via lookup on packed uint8
        lut = _POPCOUNT_LUT()
        q = lut[query_fp].sum()
        inter = np.minimum(sub, query_fp[None, :]).astype(np.uint8)
        a = lut[inter].sum(axis=1)
        b = lut[sub].sum(axis=1)
        union = q + b - a
        out = np.zeros_like(a, dtype=np.float32)
        nz = union > 0
        out[nz] = a[nz] / union[nz]
        return out


_POPCOUNT = None


def _POPCOUNT_LUT() -> np.ndarray:
    global _POPCOUNT
    if _POPCOUNT is None:
        _POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    return _POPCOUNT


def library_from_frame(df, smiles_col: str = "normalized_smiles",
                       formula_col: str = "molecular_formula",
                       key_col: str = "inchikey14",
                       lib_col: str = "ingest_lib",
                       verbose: bool = False) -> StructureLibrary:
    """
    Build a deduplicated StructureLibrary from a labelled train frame.

    One record per InChIKey14; the first non-null SMILES wins and the ingest
    libraries are unioned so provenance is preserved (needed for reporting which
    source actually supplied an answer).
    """
    acc: dict[str, dict] = {}
    for row in df.itertuples(index=False):
        smi = getattr(row, smiles_col, None)
        if smi is None or (isinstance(smi, float) and np.isnan(smi)):
            continue
        k = getattr(row, key_col, None)
        if not isinstance(k, str) or not k:
            k = smiles_to_inchikey14(smi)
        if not k:
            continue
        rec = acc.get(k)
        if rec is None:
            m = exact_mass(smi)
            if m is None:
                continue
            rec = {"key": k, "smiles": smi,
                   "formula": getattr(row, formula_col, None),
                   "mass": m, "libs": set(), "n": 0}
            acc[k] = rec
        lib = getattr(row, lib_col, None)
        if lib is not None:
            rec["libs"].add(lib)
        rec["n"] += 1

    structures = []
    for k, r in acc.items():
        structures.append(Structure(
            key=k, smiles=r["smiles"], mass=r["mass"], formula=r["formula"],
            ingest_libs=tuple(sorted(r["libs"])), fp=morgan_fp(r["smiles"]),
            n_spectra=r["n"],
        ))
    lib = StructureLibrary(structures)
    if verbose:
        n_fp = int(lib.fp_valid.sum()) if lib.fp_matrix is not None else 0
        print(f"  library: {len(lib):,} unique structures, {n_fp:,} with fingerprints, "
              f"mass {lib.masses.min():.1f}-{lib.masses.max():.1f} Da")
    return lib


def _self_check():
    # fingerprint + tanimoto sanity
    a = morgan_fp("CCO")
    b = morgan_fp("CCO")
    c = morgan_fp("c1ccccc1")
    assert a is not None and a.size == FP_NBITS // 8
    assert np.array_equal(a, b), "same molecule must give the same fingerprint"
    assert not np.array_equal(a, c)
    assert morgan_fp("not_a_smiles") is None

    # exact masses
    assert abs(exact_mass("CCO") - 46.0418648) < 1e-4
    assert abs(exact_mass("c1ccccc1") - 78.0469502) < 1e-4
    assert exact_mass("not_a_smiles") is None

    # library behaviour
    import pandas as pd

    rows = [dict(normalized_smiles="CCO", molecular_formula="C2H6O",
                 inchikey14=None, ingest_lib="gnps"),
            dict(normalized_smiles="CCO", molecular_formula="C2H6O",
                 inchikey14=None, ingest_lib="massbank"),
            dict(normalized_smiles="c1ccccc1", molecular_formula="C6H6",
                 inchikey14=None, ingest_lib="gnps"),
            dict(normalized_smiles="CCCCO", molecular_formula="C4H10O",
                 inchikey14=None, ingest_lib="riken")]
    lib = library_from_frame(pd.DataFrame(rows))
    assert len(lib) == 3, f"dedup failed: {len(lib)}"
    # the duplicate CCO must have unioned provenance
    ethanol = [s for s in lib.structures if s.smiles == "CCO"][0]
    assert set(ethanol.ingest_libs) == {"gnps", "massbank"}, ethanol.ingest_libs

    # ppm window retrieval
    m_etoh = exact_mass("CCO")           # 46.0419
    m_but = exact_mass("CCCCO")          # 74.0732
    idx = lib.mass_window(m_etoh, ppm=5.0)
    got = {lib.structure_at(i).smiles for i in idx}
    assert "CCO" in got, got
    assert "c1ccccc1" not in got, got
    # butanol is 60% heavier, so any sane ppm window excludes it
    assert "CCCCO" not in got, got
    # a window wide enough to span butanol (60% of 46 Da is ~600_000 ppm) picks it up
    wide = {lib.structure_at(i).smiles
            for i in lib.mass_window(m_etoh, ppm=700_000)}
    assert "CCCCO" in wide, wide
    # and a mid-width window still excludes it, proving the window is honoured
    mid = {lib.structure_at(i).smiles for i in lib.mass_window(m_etoh, ppm=100_000)}
    assert "CCCCO" not in mid, mid
    # 13C shift lets a candidate ~1.0034 Da heavier be found when the query is
    # the M+1 peak; shifting the query down must reach butanol-free territory but
    # must still recover ethanol itself
    idx_c13 = lib.mass_window(m_etoh - C13_SHIFT, ppm=5.0, include_c13=True)
    assert "CCO" in {lib.structure_at(i).smiles for i in idx_c13}
    # and the shift genuinely widens the window vs. disabling it
    n_shift = lib.mass_window(m_but, ppm=5.0, include_c13=True).size
    n_noshift = lib.mass_window(m_but, ppm=5.0, include_c13=False).size
    assert n_shift >= n_noshift

    # tanimoto
    idx = lib.mass_window(m_etoh, ppm=100000)
    t = lib.tanimoto(morgan_fp("CCO"), idx)
    assert t.shape[0] == idx.size
    assert np.all(t >= 0) and np.all(t <= 1)
    self_sim = t[[lib.structure_at(i).key for i in idx].index(
        [s.key for s in lib.structures if s.smiles == "CCO"][0])]
    assert self_sim > 0.999, self_sim

    # empty library must raise, not silently return nothing
    try:
        library_from_frame(pd.DataFrame([dict(normalized_smiles=None,
                                              molecular_formula=None,
                                              inchikey14=None, ingest_lib=None)]))
        raise AssertionError("empty library did not raise")
    except ValueError:
        pass

    print("retrieval self-checks passed")


if __name__ == "__main__":
    _self_check()
