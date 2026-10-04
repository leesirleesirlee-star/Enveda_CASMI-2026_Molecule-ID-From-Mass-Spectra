"""
Vectorised spectral similarity for large-scale retrieval.

The Python greedy peak matcher in `matching.py` is correct and is the reference,
but it costs ~0.3 ms per spectrum pair, which is far too slow when a single query
must be compared against thousands of library spectra (measured: 23 s/molecule).

This module represents spectra as binned sparse vectors and computes cosine
similarity for a whole batch of library spectra at once with numpy. Binning at
0.01 Da reproduces the tolerance of the peak matcher: two peaks within one bin
align, and because each spectrum contributes at most one value per bin, the
"one library peak cannot match two query peaks" property of the greedy matcher
is preserved by construction.

The vectorised score is used for the *direct* channel (shift 0), where it is a
very close approximation. Shifted (analog) matching keeps the exact reference
implementation, since a shift moves peaks between bins and the correspondence
must be established properly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BIN_WIDTH = 0.01
BIN_LO = 40.0
BIN_HI = 1300.0


def bin_spectrum(mz: np.ndarray, intensity: np.ndarray,
                 weight: str = "sqrt") -> tuple[np.ndarray, np.ndarray]:
    """
    Map peaks onto integer bins with sqrt-intensity weights, one value per bin.

    Returns (bins, weights). Duplicate bins (two peaks inside one bin) are
    merged by keeping the larger weight, mirroring the greedy matcher's rule
    that one library peak matches at most one query peak.
    """
    if mz.size == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)
    w = np.sqrt(np.maximum(intensity, 0.0)) if weight == "sqrt" \
        else np.maximum(intensity, 0.0)
    b = np.rint((np.asarray(mz, dtype=np.float64) - BIN_LO) / BIN_WIDTH).astype(np.int64)
    ok = (b >= 0) & (b <= int((BIN_HI - BIN_LO) / BIN_WIDTH))
    b, w = b[ok], w[ok]
    if b.size == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)
    order = np.argsort(b, kind="stable")
    b, w = b[order], w[order]
    # merge duplicate bins, keeping the max weight
    uniq, start = np.unique(b, return_index=True)
    if uniq.size != b.size:
        merged = np.maximum.reduceat(w, start)
        b, w = uniq, merged
    return b.astype(np.int32), w.astype(np.float32)


def shifted_cosine(q_mz: np.ndarray, q_int: np.ndarray, dm: float,
                   lib_bins: np.ndarray, lib_w: np.ndarray,
                   lib_spec: np.ndarray, n_spec: int,
                   weight: str = "sqrt") -> np.ndarray:
    """
    Cosine of a query against many library spectra where the library peaks are
    shifted by +dm, computed in bin space.

    Matching peaks under the shifted hypothesis is equivalent to shifting the
    QUERY down by dm and comparing in the same frame, so the existing batched
    machinery applies.

    The denominator must NOT be the library spectrum's full norm. Under a shift,
    only library peaks whose shifted position has a query counterpart are
    matched; including the others inflates the denominator and, worse, a single
    matched peak can produce a score above 1. So the norm is taken over the
    intersection B' n L, which is exactly the set the reference greedy matcher
    normalises over.
    """
    out = np.zeros(n_spec, dtype=np.float64)
    if q_mz.size == 0 or lib_bins.size == 0:
        return out
    b, w = bin_spectrum(q_mz - dm, q_int, weight=weight)
    if b.size == 0:
        return out
    qn = float(np.sqrt((w.astype(np.float64) ** 2).sum()))
    if qn <= 0:
        return out
    qw = w / qn

    # locate every library entry that a shifted query peak can match
    lo = np.searchsorted(lib_bins, b, side="left")
    hi = np.searchsorted(lib_bins, b, side="right")
    sizes = hi - lo
    total = int(sizes.sum())
    if total == 0:
        return out

    # Accumulate with bincount over a bounded chunk instead of materialising the
    # full match set. With a 730k-structure pool the old version built a
    # 34M-element index array plus an argsort of it (262 MB per call, fatal at
    # 10 processes). bincount gives the same sums in O(total) memory-light work.
    CHUNK = 4_000_000
    nz_mask = sizes > 0
    nz_idx = np.flatnonzero(nz_mask)
    if total <= CHUNK:
        # Build both the entry list and its owner list from the SAME mask.
        # Deriving one from flatnonzero(sizes) and the other from sizes[sizes>0]
        # is a latent misalignment: they agree only while sizes has no negative
        # entries, and any such entry would silently pair the wrong query peak.
        ent = np.concatenate([np.arange(lo[t], hi[t]) for t in nz_idx])
        ent_t = np.repeat(nz_idx, sizes[nz_mask])
        spec = lib_spec[ent]
        np.add.at(out, spec, qw[ent_t] * lib_w[ent])
        sumsq = np.bincount(spec, weights=(lib_w[ent].astype(np.float64) ** 2),
                            minlength=n_spec)
        denom = np.sqrt(sumsq)
        nz = denom > 0
        out[nz] /= denom[nz]
        # A cosine cannot exceed 1. Clamping is a safety rail: if the weights are
        # not L2-normalised per spectrum (they are in group_binned, but not in an
        # arbitrary caller), the ratio can drift above 1. Better to saturate than
        # to feed an impossible score into the ranker.
        np.clip(out, 0.0, 1.0, out=out)
        return out

    # large case: walk query peaks, accumulating into per-spectrum arrays
    num = np.zeros(n_spec, dtype=np.float64)
    ssq = np.zeros(n_spec, dtype=np.float64)
    for t in nz_idx:
        a, bnd = int(lo[t]), int(hi[t])
        spec = lib_spec[a:bnd]
        wt = lib_w[a:bnd]
        num += np.bincount(spec, weights=qw[t] * wt, minlength=n_spec)
        ssq += np.bincount(spec, weights=wt.astype(np.float64) ** 2, minlength=n_spec)
    denom = np.sqrt(ssq)
    nz = denom > 0
    out[nz] = num[nz] / denom[nz]
    np.clip(out, 0.0, 1.0, out=out)
    return out


@dataclass
class BatchedSpectra:
    """Flattened binned spectra for fast batched cosine."""
    bins: np.ndarray        # int32, concatenated
    weights: np.ndarray     # float32, concatenated, already L2-normalised per spectrum
    offsets: np.ndarray     # int64, start index per spectrum (len n+1)
    keys: list[str]
    is_timstof: np.ndarray

    def __len__(self) -> int:
        return len(self.offsets) - 1


def batch_from_spectra(mz_list, inten_list, keys, is_timstof) -> BatchedSpectra:
    bins_all, w_all, offs = [], [], [0]
    for mz, it in zip(mz_list, inten_list):
        b, w = bin_spectrum(mz, it)
        nrm = float(np.sqrt((w.astype(np.float64) ** 2).sum())) if w.size else 0.0
        if nrm > 0:
            w = (w / nrm).astype(np.float32)
        bins_all.append(b)
        w_all.append(w)
        offs.append(offs[-1] + b.size)
    return BatchedSpectra(
        bins=np.concatenate(bins_all) if bins_all else np.empty(0, dtype=np.int32),
        weights=np.concatenate(w_all) if w_all else np.empty(0, dtype=np.float32),
        offsets=np.asarray(offs, dtype=np.int64),
        keys=list(keys),
        is_timstof=np.asarray(is_timstof, dtype=bool),
    )


def cosine_batch(q_bins: np.ndarray, q_weights: np.ndarray,
                 lib: BatchedSpectra, lib_idx: np.ndarray | None = None) -> np.ndarray:
    """
    Cosine similarity of one binned query against many binned library spectra.

    Uses np.searchsorted to locate, for every query peak, the library spectra
    (within the requested index set) that have a peak in the same bin -- so the
    work scales with actual bin collisions rather than with (#query peaks x
    #library spectra).
    """
    n_lib = len(lib)
    if lib_idx is None:
        lib_idx = np.arange(n_lib)
    out = np.zeros(n_lib, dtype=np.float32)
    if q_bins.size == 0 or lib.bins.size == 0 or lib_idx.size == 0:
        return out

    # query weights must be L2-normalised too
    qn = float(np.sqrt((q_weights.astype(np.float64) ** 2).sum()))
    if qn <= 0:
        return out
    qw = (q_weights / qn).astype(np.float32)

    # global bin -> library entry lookup, built once per call over the subset.
    # We map each query bin to the slice of library entries with that bin.
    sel_bins = lib.bins
    order = np.argsort(sel_bins, kind="stable")
    sorted_bins = sel_bins[order]

    lo = np.searchsorted(sorted_bins, q_bins, side="left")
    hi = np.searchsorted(sorted_bins, q_bins, side="right")
    for k in range(q_bins.size):
        if hi[k] <= lo[k]:
            continue
        entries = order[lo[k]:hi[k]]
        # which library spectrum does each entry belong to?
        spec = np.searchsorted(lib.offsets, entries, side="right") - 1
        np.add.at(out, spec, qw[k] * lib.weights[entries])
    return out


def _self_check():
    """Vectorised cosine must agree with the greedy reference on clear cases."""
    from .matching import peak_matched_cosine

    mz = np.array([100.0, 150.0, 200.0, 250.0])
    it = np.array([1.0, 0.5, 0.25, 0.1])

    # identical spectra -> 1.0
    lib = batch_from_spectra([mz], [it], ["a"], [True])
    s = cosine_batch(*bin_spectrum(mz, it), lib)
    assert abs(s[0] - 1.0) < 1e-6, s
    ref, n = peak_matched_cosine(mz, it, mz, it)
    assert abs(ref - 1.0) < 1e-9 and n == 4

    # disjoint spectra -> 0
    mz2 = np.array([500.0, 600.0])
    it2 = np.array([1.0, 1.0])
    lib2 = batch_from_spectra([mz2], [it2], ["b"], [True])
    s2 = cosine_batch(*bin_spectrum(mz, it), lib2)
    assert s2[0] == 0.0, s2

    # one shared peak out of four -> matches the reference magnitude
    mz3 = np.array([100.0, 700.0])
    it3 = np.array([1.0, 1.0])
    lib3 = batch_from_spectra([mz3], [it3], ["c"], [True])
    s3 = cosine_batch(*bin_spectrum(mz, it), lib3)
    ref3, n3 = peak_matched_cosine(mz, it, mz3, it3)
    assert n3 == 1
    assert abs(float(s3[0]) - ref3) < 1e-5, (s3, ref3)

    # two peaks inside one bin collapse to one (no double counting)
    b, w = bin_spectrum(np.array([100.0, 100.004]), np.array([1.0, 1.0]))
    assert b.size == 1, b
    # batch arithmetic: offsets are contiguous and cover every entry
    lib4 = batch_from_spectra([mz, mz2, mz3], [it, it2, it3], ["a", "b", "c"],
                              [True, False, True])
    assert len(lib4) == 3
    assert lib4.offsets[-1] == lib4.bins.size
    assert list(lib4.is_timstof) == [True, False, True]
    s4 = cosine_batch(*bin_spectrum(mz, it), lib4)
    assert s4.shape == (3,)
    assert abs(s4[0] - 1.0) < 1e-6 and abs(s4[1]) < 1e-9

    # ---- shifted cosine must agree with the reference modified cosine ----
    from .matching import modified_cosine

    # library = query shifted down by dm, i.e. the query is the +dm analogue
    dm = 15.0
    lib_mz5 = np.array([100.0, 150.0, 200.0])
    lib_it5 = np.array([1.0, 0.5, 0.2])
    q_mz5 = lib_mz5 + dm
    q_it5 = lib_it5
    batch5 = batch_from_spectra([lib_mz5], [lib_it5], ["x"], [True])
    bb = batch_from_spectra([lib_mz5], [lib_it5], ["x"], [True])
    import numpy as _np
    bins5, w5 = bin_spectrum(lib_mz5, lib_it5)
    spec5 = _np.zeros(bins5.size, dtype=_np.int64)
    sc = shifted_cosine(q_mz5, q_it5, dm, bins5, w5, spec5, 1)
    ref, nm, mode = modified_cosine(q_mz5, q_it5, lib_mz5, lib_it5, dm)
    assert abs(float(sc[0]) - ref) < 1e-5, (sc, ref)
    assert nm == 3 and mode in ("shift+", "shift-")
    # zero shift reduces to the plain cosine
    sc0 = shifted_cosine(lib_mz5, lib_it5, 0.0, bins5, w5, spec5, 1)
    assert abs(float(sc0[0]) - 1.0) < 1e-6, sc0
    # a wrong shift should not match
    sc_bad = shifted_cosine(q_mz5, q_it5, 40.0, bins5, w5, spec5, 1)
    assert float(sc_bad[0]) < 0.5, sc_bad

    print("fast matching self-checks passed")


if __name__ == "__main__":
    _self_check()
