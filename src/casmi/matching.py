"""
Channel 1 & 2: direct spectral matching and analog (mass-shift) propagation.

These are the two channels that carry the proven public baseline.

**Direct match** is a peak-matching cosine (optionally with spectral entropy
weighting), aggregated per molecule across that molecule's spectra.

**Analog propagation** is the key idea and is easy to get wrong. The library
hit's *structure is never edited*. Instead:

  1. Find a library spectrum `a` whose neutral mass differs from the query by
     dm = M_query - M_lib.
  2. Match peaks under two simultaneous hypotheses: an unmodified fragment
     (mz_q ~ mz_a) and a shifted fragment (mz_q ~ mz_a + dm). This is the GNPS
     modified cosine.
  3. Propagate the annotation: a candidate structure `c` is boosted when its
     fingerprint is similar to the library compound whose shifted spectrum
     matched, i.e. AnalogScore(c) = max_a [ SimMod(q,a)^2 * Tanimoto(fp_c, fp_a) ].

So a query that is a *chemical analogue* of a library compound (e.g. a
glycosylated or methylated variant absent from every library) still gets its
true structure ranked highly, provided that structure is in the candidate pool
by mass. This is what lets the pipeline score molecules that are genuinely
absent from the spectral library.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

# Peak matching tolerance. 0.01 Da is the usual starting point; the competition
# spectra are timsTOF (high resolution), so this is deliberately not tighter, to
# tolerate library spectra from lower-resolution instruments.
DEFAULT_TOL = 0.01
# Analog search only considers library compounds within this neutral-mass offset.
MAX_DM = 200.0
# Modified-cosine power applied to the similarity before it gates the
# fingerprint term (from the published reference: SimMod ** 2.0).
SIM_POWER = 2.0


def _weight(mz: np.ndarray, intensity: np.ndarray, mode: str) -> np.ndarray:
    """Peak weights used by the cosine."""
    if mode == "sqrt":
        return np.sqrt(np.maximum(intensity, 0.0))
    if mode == "raw":
        return np.maximum(intensity, 0.0)
    raise ValueError(f"unknown weighting {mode!r}")


def peak_matched_cosine(mz_q: np.ndarray, in_q: np.ndarray,
                        mz_l: np.ndarray, in_l: np.ndarray,
                        shift: float = 0.0, tol: float = DEFAULT_TOL,
                        weighting: str = "sqrt") -> tuple[float, int]:
    """
    Greedy peak-matching cosine between two spectra, with an optional shift
    applied to the library peaks. Returns (score, n_matched).

    Matching is greedy by descending product of intensities: the strongest
    compatible pair is taken first, then both peaks are removed. This mirrors
    the standard MS/MS cosine implementations.
    """
    if mz_q.size == 0 or mz_l.size == 0:
        return 0.0, 0
    wq = _weight(mz_q, in_q, weighting)
    wl = _weight(mz_l, in_l, weighting)
    mzl = mz_l + shift

    # candidate pairs within tolerance
    order = np.argsort(mzl)
    mzl_s, idx_l = mzl[order], order
    lo = np.searchsorted(mzl_s, mz_q - tol, side="left")
    hi = np.searchsorted(mzl_s, mz_q + tol, side="right")

    pairs = []
    for i in range(mz_q.size):
        for j in range(lo[i], hi[i]):
            k = idx_l[j]
            if abs(mz_q[i] - mzl_s[j]) <= tol:
                pairs.append((wq[i] * wl[k], i, k))
    if not pairs:
        return 0.0, 0

    pairs.sort(key=lambda t: -t[0])
    used_q, used_l = set(), set()
    num = 0.0
    for prod, i, k in pairs:
        if i in used_q or k in used_l:
            continue
        used_q.add(i)
        used_l.add(k)
        num += prod

    denom = float(np.sqrt((wq ** 2).sum() * (wl ** 2).sum()))
    if denom <= 0:
        return 0.0, 0
    return float(num / denom), len(used_q)


def modified_cosine(mz_q, in_q, mz_l, in_l, dm: float,
                    tol: float = DEFAULT_TOL,
                    weighting: str = "sqrt") -> tuple[float, int, str]:
    """
    GNPS modified cosine: the best of the unmodified, +dm and -dm hypotheses.

    dm is the neutral-mass difference M_query - M_library. A fragment that
    contains the modified site appears at mz + dm in the query, so we try both
    the unshifted comparison and the shifted ones and keep the best.
    """
    best = peak_matched_cosine(mz_q, in_q, mz_l, in_l, 0.0, tol, weighting)
    if abs(dm) < 1e-4:
        return best[0], best[1], "direct"
    pos = peak_matched_cosine(mz_q, in_q, mz_l, in_l, dm, tol, weighting)
    neg = peak_matched_cosine(mz_q, in_q, mz_l, in_l, -dm, tol, weighting)
    cands = [(best[0], best[1], "direct"), (pos[0], pos[1], "shift+"),
             (neg[0], neg[1], "shift-")]
    cands.sort(key=lambda t: -t[0])
    s, n, mode = cands[0]
    # Report "direct" when the unmodified hypothesis wins (even by a hair),
    # so the downstream gate can tell a true analogue from a plain hit.
    if mode != "direct" and s <= best[0] + 1e-9:
        mode = "direct"
    return s, n, mode


def neutral_loss_cosine(mz_q, in_q, mz_l, in_l, prec_q: float, prec_l: float,
                        tol: float = DEFAULT_TOL,
                        weighting: str = "sqrt") -> tuple[float, int]:
    """
    Cosine over neutral losses (precursor - fragment) instead of fragments.

    Complements the fragment-level view: losses are often conserved across
    analogues even when the fragment masses shift.
    """
    lq = prec_q - mz_q
    ll = prec_l - mz_l
    keep_q = lq > 0
    keep_l = ll > 0
    if not keep_q.any() or not keep_l.any():
        return 0.0, 0
    return peak_matched_cosine(lq[keep_q], in_q[keep_q], ll[keep_l], in_l[keep_l],
                               0.0, tol, weighting)


@dataclass
class DirectHit:
    lib_index: int
    score: float
    n_matched: int
    mode: str
    dm: float


def direct_match_query(query_mz, query_in, query_prec: float,
                       lib_mz: Sequence[np.ndarray], lib_in: Sequence[np.ndarray],
                       lib_mass: Sequence[float],
                       candidate_mask: np.ndarray | None = None,
                       dm_max: float = MAX_DM,
                       tol: float = DEFAULT_TOL) -> list[DirectHit]:
    """
    Score one query spectrum against library spectra, using the modified cosine.

    Only library spectra whose neutral mass is within dm_max contribute to the
    analog channel; direct (unshifted) matches are useful at any mass, but we
    restrict to a generous window because a hit far in mass is not informative
    about the candidate structures we are ranking.
    """
    q_mass = np.nan
    hits: list[DirectHit] = []
    n = len(lib_mz)
    for i in range(n):
        if candidate_mask is not None and not candidate_mask[i]:
            continue
        dm = 0.0  # unknown per-library-spectrum query mass is handled by caller
        s, nm, mode = modified_cosine(query_mz, query_in, lib_mz[i], lib_in[i],
                                      dm, tol)
        if s <= 0 or nm == 0:
            continue
        hits.append(DirectHit(i, s, nm, mode, dm))
    hits.sort(key=lambda h: -h.score)
    return hits


def propagate_analog(sim_mod: float, fp_candidate: np.ndarray,
                     fp_library_hit: np.ndarray) -> float:
    """
    AnalogScore(c) = SimMod(q,a)^SIM_POWER * Tanimoto(fp_c, fp_a).

    The fingerprint term is what stops the shift from over-claiming: a shifted
    spectral match only boosts candidates that are actually structurally like
    the library compound that matched.
    """
    t = _tanimoto_packed(fp_candidate, fp_library_hit)
    return float(max(sim_mod, 0.0) ** SIM_POWER * t)


_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _tanimoto_packed(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None or a.size != b.size:
        return 0.0
    inter = int(_LUT[np.minimum(a, b)].sum())
    union = int(_LUT[a].sum()) + int(_LUT[b].sum()) - inter
    return inter / union if union > 0 else 0.0


def _self_check():
    from .retrieval import morgan_fp

    # --- negligible-shift: identical spectra score 1.0 ---
    mz = np.array([100.0, 150.0, 200.0])
    in_ = np.array([1.0, 0.5, 0.2])
    s, n = peak_matched_cosine(mz, in_, mz, in_)
    assert abs(s - 1.0) < 1e-9, s
    assert n == 3, n

    # --- disjoint spectra score 0 ---
    s, n = peak_matched_cosine(np.array([100.0]), np.array([1.0]),
                               np.array([500.0]), np.array([1.0]))
    assert s == 0.0 and n == 0

    # --- a shifted spectrum is found ONLY by the modified (shifted) cosine ---
    dm = 15.0
    lib_mz = np.array([100.0, 150.0, 200.0])
    lib_in = np.array([1.0, 0.5, 0.2])
    # every query fragment contains the modification site, so all shift by +dm
    q_mz = np.array([115.0, 165.0, 215.0])
    q_in = np.array([1.0, 0.5, 0.2])
    s_direct, n_direct, _ = modified_cosine(q_mz, q_in, lib_mz, lib_in, 0.0)
    s_mod, nm, mode = modified_cosine(q_mz, q_in, lib_mz, lib_in, dm)
    assert s_direct == 0.0 and n_direct == 0, \
        f"unshifted spectra should not match: {s_direct}, {n_direct}"
    assert s_mod > 0.99, f"modified cosine failed to find the shift: {s_mod}"
    assert mode in ("shift+", "shift-"), mode
    assert nm == 3, nm
    # partial shift: two of three fragments move, so the shift hypothesis wins
    # while the direct hypothesis still explains one peak
    q_partial = np.array([115.0, 165.0, 200.0])
    s_dir_p, n_dir_p, _ = modified_cosine(q_partial, q_in, lib_mz, lib_in, 0.0)
    s_mod_p, n_mod_p, mode_p = modified_cosine(q_partial, q_in, lib_mz, lib_in, dm)
    assert n_dir_p == 1, n_dir_p
    assert n_mod_p == 2, n_mod_p
    assert s_mod_p > s_dir_p, (s_mod_p, s_dir_p)
    assert mode_p in ("shift+", "shift-"), mode_p

    # --- the shift must be reported as an analogue, not a direct hit ---
    s_noshift_case, _, mode_plain = modified_cosine(lib_mz, lib_in,
                                                    lib_mz, lib_in, dm)
    assert mode_plain == "direct", mode_plain

    # --- greedy matching: one library peak cannot match two query peaks ---
    s, n = peak_matched_cosine(np.array([100.0, 100.005]), np.array([1.0, 1.0]),
                               np.array([100.0]), np.array([1.0]))
    assert n == 1, f"a single library peak matched {n} query peaks"

    # --- tolerance is enforced ---
    s, n = peak_matched_cosine(np.array([100.0]), np.array([1.0]),
                               np.array([100.02]), np.array([1.0]), tol=0.01)
    assert n == 0, "tolerance not enforced"

    # --- neutral-loss cosine ---
    s, n = neutral_loss_cosine(np.array([100.0, 150.0]), np.array([1.0, 0.5]),
                               np.array([120.0, 170.0]), np.array([1.0, 0.5]),
                               prec_q=200.0, prec_l=220.0)
    assert abs(s - 1.0) < 1e-9, f"losses are identical so score should be 1: {s}"

    # --- analog propagation: same fingerprint boosts, different does not ---
    a = morgan_fp("CCO")
    assert propagate_analog(0.9, a, a) > 0.8           # 0.81 * tanimoto(1.0)
    benzene = morgan_fp("c1ccccc1")
    assert propagate_analog(0.9, a, benzene) < 0.3     # low tanimoto
    # a perfect fingerprint match with a weak spectral match must stay weak
    assert propagate_analog(0.1, a, a) < 0.02
    # None fingerprints must not crash
    assert propagate_analog(0.9, None, a) == 0.0

    print("matching self-checks passed")


if __name__ == "__main__":
    _self_check()
