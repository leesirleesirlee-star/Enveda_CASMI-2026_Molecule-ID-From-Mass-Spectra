"""
Three-fold validation protocol.

The visible test.parquet is a placeholder that Kaggle replaces with a hidden
test set at scoring time, so it cannot be used to tune anything. Validation is
therefore built from the labelled train data, in three deliberately different
regimes, because the real test set mixes them:

  V-A  instrument-calibration fold
       ingest_lib == 'enveda-np-examples': 250 natural products measured on the
       same instrument family as the hidden test set. This is the primary
       calibration signal and the closest proxy for the real leaderboard.

  V-B  same-source fold, answer RETAINED in the library
       Measures the retrieval ceiling: how well ranking works when the right
       structure is definitely reachable. If V-B is high and V-A is low, the
       problem is transfer across libraries/instruments, not recall.

  V-C  identity-disjoint fold, answer REMOVED from the library
       The strict extrapolation regime. Any gain that does not hold here is
       overfitting to library leakage.

Every evaluation also reports the oracle recall@K ceiling, so a poor MRR can be
attributed to retrieval (nothing good was retrieved) or to ranking (the answer
was in the candidate set but ordered too low). Without that split the two
failure modes look identical.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from .core import mrr25_breakdown, smiles_to_inchikey14
from .spectra import Molecule, Spectrum, group_by_molecule

V_A, V_B, V_C = "V_A", "V_B", "V_C"

# Which ingest_lib values constitute the instrument-matched calibration split.
CALIBRATION_LIBS = {"enveda-np-examples"}


@dataclass
class Fold:
    name: str
    queries: dict[str, Molecule]          # molecule_id -> its spectra
    truths: dict[str, str]                # molecule_id -> true SMILES
    library: dict[str, dict]              # key(ik14) -> structure record
    description: str = ""

    def __len__(self) -> int:
        return len(self.queries)


def _structures_from(df) -> dict[str, dict]:
    """ik14 -> {smiles, formula, ingest_lib, spectra:[...]}; the library unit."""
    out: dict[str, dict] = {}
    for row in df.itertuples(index=False):
        smi = getattr(row, "normalized_smiles", None)
        if smi is None or (isinstance(smi, float) and np.isnan(smi)):
            continue
        k = getattr(row, "inchikey14", None) or smiles_to_inchikey14(smi)
        if not k:
            continue
        rec = out.setdefault(k, {
            "key": k,
            "smiles": smi,
            "formula": getattr(row, "molecular_formula", None),
            "ingest_libs": set(),
            "spectra": [],
        })
        rec["ingest_libs"].add(getattr(row, "ingest_lib", None))
    return out


@dataclass
class ValidationData:
    """Everything the protocol needs, built once from the labelled train data."""
    mol_by_lib: dict[str, list] = field(default_factory=dict)   # ingest_lib -> rows
    all_keys: set[str] = field(default_factory=set)
    key_to_smiles: dict[str, str] = field(default_factory=dict)


def build_folds(df, vc_fraction: float = 0.15, seed: int = 42,
                calibration_libs: set[str] | None = None,
                mol_key: str = "inchikey14") -> dict[str, Fold]:
    """
    Build V-A / V-B / V-C from a labelled train frame.

    NOTE on identity: train.parquet has NO molecule_id (unlike test.parquet), so
    a "molecule" here is a structure, identified by mol_key (default inchikey14).
    All spectra carrying that key are the molecule's replicate spectra. This is
    exactly the aggregation the competition asks for: one ranked candidate list
    per molecule, informed by all of its spectra.

    df must carry: ms2_mzs, ms2_normalized_intensities, precursor_mz, adduct,
    normalized_smiles, inchikey14, ingest_lib.
    """
    from .spectra import load_spectra

    calibration_libs = calibration_libs or CALIBRATION_LIBS
    rng = np.random.default_rng(seed)
    lib_col = "ingest_lib" if "ingest_lib" in df.columns else None
    if mol_key not in df.columns:
        raise KeyError(f"train frame lacks {mol_key!r}: {list(df.columns)[:12]}")

    # normalise the key column: fill gaps from SMILES so nothing is lost.
    # The temporary name must NOT start with an underscore: itertuples() renames
    # such columns to positional _1/_2 and they become unreachable by name.
    keys = []
    for smi, k in zip(df["normalized_smiles"], df[mol_key]):
        kk = k if isinstance(k, str) and k else smiles_to_inchikey14(smi)
        keys.append(kk)
    df = df.assign(mkey=keys)
    df = df[df["mkey"].notna()]

    key_to_smiles: dict[str, str] = {}
    for k, smi in zip(df["mkey"], df["normalized_smiles"]):
        key_to_smiles.setdefault(k, smi)

    spectra = load_spectra(df, molecule_col="mkey")
    mols = group_by_molecule(spectra)
    # structure key -> its ingest library (first wins; used for reporting)
    mol_lib: dict[str, str] = {}
    if lib_col:
        for k, lib in zip(df["mkey"], df[lib_col]):
            mol_lib.setdefault(k, str(lib))

    folds: dict[str, Fold] = {}

    def make_lib(remove: set[str]) -> dict[str, dict]:
        return {k: {"key": k, "smiles": s, "ingest_libs": {"all"}}
                for k, s in key_to_smiles.items() if k not in remove}

    # ---- V-A: instrument-matched calibration split ----
    # Select on the ROW's ingest_lib, not on a per-structure "first library".
    # All 250 enveda-np-examples structures also occur in gnps/riken/mona/
    # massbank/drug_plus, and for most of them one of those is seen first, so a
    # setdefault-style pick silently yields an empty fold.
    calib_keys: set[str] = set()
    if lib_col:
        for k, lib in zip(df["mkey"], df[lib_col]):
            if lib in calibration_libs:
                calib_keys.add(k)
    if calib_keys:
        truth = {k: key_to_smiles[k] for k in calib_keys if k in key_to_smiles}
        # Queries use ONLY the calibration spectra (test-instrument family),
        # while the library keeps every other source for those same structures.
        # That makes V-A a genuine instrument-transfer test -- the exact
        # situation on the hidden test set, where the molecule is known to other
        # libraries from other instruments but not necessarily to timsTOF data.
        queries = {}
        for k in truth:
            sp = [s for s in mols[k].spectra if s.ingest_lib in calibration_libs]
            if sp:
                queries[k] = Molecule(k, sp)
        if queries:
            folds[V_A] = Fold(
                V_A, queries, truth, make_lib(set()),
                "instrument-matched natural products (enveda-np-examples, "
                "timsTOF): queries use only calibration spectra while the "
                "library keeps all other sources -- an instrument-transfer test")

    # ---- V-B / V-C: split the remaining structures ----
    used = set(folds[V_A].queries) if V_A in folds else set()
    remaining = sorted(set(mols) - used)
    if remaining:
        perm = rng.permutation(len(remaining))
        n_c = max(1, int(len(remaining) * vc_fraction))
        vc_ids = [remaining[i] for i in perm[:n_c]]
        vb_ids = [remaining[i] for i in perm[n_c:]]

        def mk(name, ids, desc, remove_truth):
            truth = {k: key_to_smiles[k] for k in ids if k in key_to_smiles}
            q = {k: mols[k] for k in truth}
            remove = {smiles_to_inchikey14(t) for t in truth.values()} if remove_truth else set()
            remove.discard(None)
            return Fold(name, q, truth, make_lib(remove), desc)

        folds[V_B] = mk(V_B, vb_ids,
                        "same-source holdout, answer RETAINED in library "
                        "(retrieval ceiling)", remove_truth=False)
        folds[V_C] = mk(V_C, vc_ids,
                        "identity-disjoint holdout, answer REMOVED from library "
                        "(strict extrapolation)", remove_truth=True)

    # store the full structure universe for oracle-recall diagnostics
    for f in folds.values():
        f.__dict__["all_keys"] = key_to_smiles
    return folds


# ------------------------------------------------------------------ scoring

def oracle_recall(queries: dict[str, Molecule],
                  candidate_keys: dict[str, Sequence[str]],
                  truths: dict[str, str],
                  ks: Sequence[int] = (1, 5, 10, 25, 50, 100)) -> dict:
    """
    Best achievable hit@k if ranking were perfect: is the true key even present
    in the retrieved candidate set?

    Separating this from MRR is what distinguishes "retrieval failed" from
    "ranking failed".
    """
    out = {}
    for k in ks:
        hits = 0
        n = 0
        for mid, truth in truths.items():
            tk = smiles_to_inchikey14(truth)
            if tk is None:
                continue
            n += 1
            cands = list(candidate_keys.get(mid, ()))[:k]
            if tk in cands:
                hits += 1
        out[f"recall@{k}"] = hits / n if n else 0.0
    return out


def evaluate(name: str, predictions: dict[str, Sequence[str]],
             truths: dict[str, str], candidate_keys: dict[str, Sequence[str]] | None = None,
             verbose: bool = True) -> dict:
    """MRR@25 breakdown for a fold, plus the oracle recall ceiling if available."""
    res = mrr25_breakdown(predictions, truths)
    res["fold"] = name
    if candidate_keys is not None:
        res.update(oracle_recall(
            {m: None for m in truths}, candidate_keys, truths))  # type: ignore[arg-type]
    if verbose:
        print(f"[{name}] MRR@25={res['mrr25']:.4f}  top1={res['top1']:.3f}  "
              f"hit@10={res['hit@10']:.3f}  hit@25={res['hit@25']:.3f}  "
              f"mean_rank(hit)={res['mean_rank_when_hit']:.1f}  n={res['n']}")
        if candidate_keys is not None:
            print(f"       oracle: " + "  ".join(
                f"{k}={v:.3f}" for k, v in res.items() if k.startswith("recall@")))
    return res


def _self_check():
    """Fold bookkeeping must not leak the answer into the V-C library."""
    import pandas as pd

    rows = []
    # 4 calibration molecules + 10 others, each with one spectrum
    for i in range(4):
        rows.append(dict(molecule_id=f"cal{i}", spectrum_id=f"cs{i}",
                         ms2_mzs=[100.0, 150.0, 200.0],
                         ms2_normalized_intensities=[1.0, 0.5, 0.2],
                         precursor_mz=300.0 + i, adduct="[M+H]+",
                         normalized_smiles=f"C{'C' * (i + 1)}O",
                         inchikey14=None, ingest_lib="enveda-np-examples",
                         molecular_formula="C2H6O"))
    for i in range(10):
        rows.append(dict(molecule_id=f"o{i}", spectrum_id=f"os{i}",
                         ms2_mzs=[100.0, 150.0, 200.0],
                         ms2_normalized_intensities=[1.0, 0.5, 0.2],
                         precursor_mz=400.0 + i, adduct="[M+H]+",
                         normalized_smiles=f"C{'C' * (i + 2)}O",
                         inchikey14=None, ingest_lib="gnps",
                         molecular_formula="C3H8O"))
    df = pd.DataFrame(rows)
    folds = build_folds(df)
    assert V_A in folds, folds.keys()
    assert folds[V_A].queries, "V-A empty"
    assert len(folds[V_A]) == 4, len(folds[V_A])
    assert V_B in folds and V_C in folds
    # V-C must not contain its own answers
    vc_truth_keys = {smiles_to_inchikey14(t) for t in folds[V_C].truths.values()}
    leaked = vc_truth_keys & set(folds[V_C].library)
    assert not leaked, f"V-C library leaks {len(leaked)} answers"
    # V-B must retain them
    vb_truth_keys = {smiles_to_inchikey14(t) for t in folds[V_B].truths.values()}
    assert vb_truth_keys & set(folds[V_B].library), "V-B should retain answers"
    # folds are disjoint
    assert not (set(folds[V_B].queries) & set(folds[V_C].queries))
    assert not (set(folds[V_A].queries) & set(folds[V_B].queries))
    assert not (set(folds[V_A].queries) & set(folds[V_C].queries))
    print("validation self-checks passed")


if __name__ == "__main__":
    _self_check()
