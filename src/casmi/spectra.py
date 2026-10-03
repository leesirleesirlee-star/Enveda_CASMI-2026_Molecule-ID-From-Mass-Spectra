"""
Spectrum handling: peak cleaning, quality gating, molecule-level aggregation.

Design notes
------------
* Peak arrays arrive as Arrow list columns, so they are materialised to float32
  numpy arrays once, at load time.
* Intensities in the competition data are already normalised to base peak = 1.
  We never re-normalise away the absolute scale wholesale, but a *relative*
  threshold (fraction of base peak) is the standard, robust noise cut.
* A molecule's spectra are aggregated explicitly: the competition scores
  molecules, not spectra, so every downstream channel returns per-molecule
  evidence combined across that molecule's spectra and collision energies.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from .core import ADDUCT_OFFSET, neutral_mass_from_precursor

# Cleaning defaults, chosen to sit inside the usual MS/MS practice:
# drop below 1% of base peak, outside 50-1200 m/z, merge near-duplicate peaks,
# then keep the most intense MAX_PEAKS.
MIN_REL_INTENSITY = 0.01
MIN_MZ = 50.0
MAX_MZ = 1200.0
MERGE_TOL_DA = 0.01
MAX_PEAKS = 80
MIN_PEAKS = 3


@dataclass
class Spectrum:
    spectrum_id: str
    molecule_id: str
    mz: np.ndarray
    intensity: np.ndarray
    precursor_mz: float
    adduct: str
    ionization_mode: str = ""
    instrument_type: str = ""
    collision_energy: tuple[float, ...] = ()
    neutral_mass: float = float("nan")
    n_peaks_raw: int = 0
    base_peak_intensity: float = float("nan")

    def __post_init__(self):
        if np.isnan(self.neutral_mass) and self.adduct in ADDUCT_OFFSET:
            self.neutral_mass = neutral_mass_from_precursor(self.precursor_mz, self.adduct)


@dataclass
class Molecule:
    """All spectra observed for one molecule_id, plus per-adduct mass evidence."""
    molecule_id: str
    spectra: list[Spectrum] = field(default_factory=list)

    def adducts(self) -> list[str]:
        return sorted({s.adduct for s in self.spectra})

    def neutral_mass_estimates(self) -> dict[str, float]:
        """adduct -> median implied neutral mass (robust to one bad spectrum)."""
        by_ad = defaultdict(list)
        for s in self.spectra:
            if not np.isnan(s.neutral_mass):
                by_ad[s.adduct].append(s.neutral_mass)
        return {ad: float(np.median(v)) for ad, v in by_ad.items()}

    def consensus_neutral_mass(self) -> float:
        """Median over spectra; the adduct-aware estimate."""
        m = [s.neutral_mass for s in self.spectra if not np.isnan(s.neutral_mass)]
        return float(np.median(m)) if m else float("nan")

    def mass_spread_ppm(self) -> float:
        """
        Spread of implied neutral mass across this molecule's spectra, in ppm.

        A tight spread is evidence the adduct model is self-consistent: spectra
        of one molecule taken under different adducts must imply one mass.
        """
        m = np.asarray([s.neutral_mass for s in self.spectra if not np.isnan(s.neutral_mass)])
        if m.size < 2:
            return float("nan")
        return float((m.max() - m.min()) / np.median(m) * 1e6)


def clean_peaks(mz: Sequence[float], intensity: Sequence[float],
                min_rel_intensity: float = MIN_REL_INTENSITY,
                merge_tol_da: float = MERGE_TOL_DA,
                max_peaks: int = MAX_PEAKS) -> tuple[np.ndarray, np.ndarray]:
    """
    Sort, threshold, merge near-duplicate peaks, and cap peak count.

    Merging matters because centroiding leaves split peaks: two peaks within
    merge_tol_da are one physical peak, and leaving both double-counts intensity
    in every cosine-style score.
    """
    mz = np.asarray(mz, dtype=np.float64)
    inten = np.asarray(intensity, dtype=np.float64)
    if mz.size == 0:
        return np.empty(0), np.empty(0)

    order = np.argsort(mz)
    mz, inten = mz[order], inten[order]

    keep = (mz >= MIN_MZ) & (mz <= MAX_MZ) & (inten > 0)
    mz, inten = mz[keep], inten[keep]
    if mz.size == 0:
        return np.empty(0), np.empty(0)

    rel = inten / inten.max()
    hit = rel >= min_rel_intensity
    mz, inten = mz[hit], inten[hit]
    if mz.size == 0:
        return np.empty(0), np.empty(0)

    # merge peaks closer than merge_tol_da, keeping the most intense of each group.
    # A new group starts at every index whose gap to the PREVIOUS peak exceeds the
    # tolerance, so index i joins the group of i-1 when the gap is small. (An
    # off-by-one here merges the wrong pair and silently corrupts every score.)
    if mz.size > 1:
        grp_start = np.r_[0, np.where(np.diff(mz) > merge_tol_da)[0] + 1]
        new_mz, new_in = [], []
        for i, s in enumerate(grp_start):
            e = grp_start[i + 1] if i + 1 < len(grp_start) else mz.size
            seg_i = inten[s:e]
            j = s + int(np.argmax(seg_i))
            new_mz.append(mz[j])
            new_in.append(inten[j])
        mz = np.asarray(new_mz)
        inten = np.asarray(new_in)

    if mz.size > max_peaks:
        idx = np.argsort(inten)[::-1][:max_peaks]
        idx.sort()
        mz, inten = mz[idx], inten[idx]

    return mz.astype(np.float32), inten.astype(np.float32)


def load_spectra(df, min_peaks: int = MIN_PEAKS, verbose: bool = False) -> list[Spectrum]:
    """
    Build Spectrum objects from a competition parquet frame.

    Spectra with fewer than min_peaks cleaned peaks carry almost no structural
    information and are dropped (their molecule may still be represented by its
    other spectra).
    """
    out, dropped = [], 0
    for row in df.itertuples(index=False):
        mz, inten = clean_peaks(getattr(row, "ms2_mzs"), getattr(row, "ms2_normalized_intensities"))
        if mz.size < min_peaks:
            dropped += 1
            continue
        ce = getattr(row, "collision_energy_ev", None)
        try:
            # Arrow list columns arrive as numpy arrays; `x or ()` is ambiguous on
            # arrays, so test for None/empty explicitly.
            ce = () if ce is None else tuple(float(x) for x in ce)
        except (TypeError, ValueError):
            ce = ()
        out.append(Spectrum(
            spectrum_id=str(getattr(row, "spectrum_id")),
            molecule_id=str(getattr(row, "molecule_id")),
            mz=mz,
            intensity=inten,
            precursor_mz=float(getattr(row, "precursor_mz")),
            adduct=str(getattr(row, "adduct")),
            ionization_mode=str(getattr(row, "ionization_mode", "") or ""),
            instrument_type=str(getattr(row, "instrument_type", "") or ""),
            collision_energy=ce,
            n_peaks_raw=int(getattr(row, "num_peaks", mz.size) or mz.size),
            base_peak_intensity=float(getattr(row, "base_peak_intensity", np.nan) or np.nan),
        ))
    if verbose:
        print(f"  load_spectra: kept {len(out):,}, dropped {dropped:,} (too few peaks)")
    return out


def group_by_molecule(spectra: Iterable[Spectrum]) -> dict[str, Molecule]:
    groups: dict[str, Molecule] = {}
    for s in spectra:
        groups.setdefault(s.molecule_id, Molecule(s.molecule_id)).spectra.append(s)
    return groups


def _self_check():
    # sorting, thresholding, merging
    mz, in_ = clean_peaks([200.0, 100.0, 100.005, 100.02, 50.0, 1100.0],
                          [0.5, 1.0, 0.4, 0.9, 0.0001, 0.2])
    assert np.all(np.diff(mz) > 0), "not sorted"
    # 100.0 and 100.005 are one physical peak; the more intense one (100.0, i=1.0)
    # survives, and 100.02 is >tol away so it stays separate.
    assert mz.size == 4, f"expected 4 peaks after merge, got {mz.size}: {mz}"
    assert abs(mz[0] - 100.0) < 1e-9, f"merge kept the wrong peak: {mz}"
    assert abs(mz[1] - 100.02) < 1e-9, f"unexpected second peak: {mz}"
    assert 50.0 not in mz, "below-threshold peak survived"
    # a peak whose neighbour is within tolerance but which is itself more intense
    # must be the survivor
    mz_a, _ = clean_peaks([100.0, 100.004], [0.2, 0.9])
    assert mz_a.size == 1 and abs(mz_a[0] - 100.004) < 1e-9, mz_a
    # out-of-range dropped
    mz2, _ = clean_peaks([20.0, 30.0], [1.0, 1.0])
    assert mz2.size == 0
    # empty input
    mz3, in3 = clean_peaks([], [])
    assert mz3.size == 0 and in3.size == 0
    # peak cap respects the most intense
    big_mz = np.arange(100.0, 100.0 + 300 * 0.5, 0.5)
    big_in = np.arange(big_mz.size, dtype=float)
    mzc, inc = clean_peaks(big_mz, big_in, max_peaks=10)
    assert mzc.size == 10
    assert inc.min() >= big_in.max() - 10  # kept the top-10 intensities

    # neutral mass / molecule aggregation
    s1 = Spectrum("s1", "m1", np.array([100.0]), np.array([1.0]), 181.0712, "[M+H]+")
    s2 = Spectrum("s2", "m1", np.array([100.0]), np.array([1.0]), 203.0531, "[M+Na]+")
    mol = group_by_molecule([s1, s2])["m1"]
    est = mol.neutral_mass_estimates()
    assert set(est) == {"[M+H]+", "[M+Na]+"}, est
    # the two adducts must imply the same neutral mass (glucose ~180.063)
    assert abs(est["[M+H]+"] - est["[M+Na]+"]) < 1e-3, est
    assert abs(est["[M+H]+"] - 180.0633881) < 1e-3, est

    print("spectra self-checks passed")


if __name__ == "__main__":
    _self_check()
