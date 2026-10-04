"""
CASMI 2026 core chemistry / scoring utilities.

These are the primitives every part of the pipeline shares, so they are defined
once here:

  - adduct <-> neutral monoisotopic mass conversion (10 official adducts)
  - SMILES -> InChIKey first block (InChIKey14), via RDKit tautomer canonicalization
    matching the official grader (tautomers + stereo ignored)
  - MRR@25 scoring replicated from the official CASMI metric
  - submission formatting / validation (molecule_id,smiles; ';'-joined, <=25)

The official matching rule: both the predicted and the true SMILES are passed
through RDKit tautomer canonicalization (RDKit pinned at 2026.03.3 by the
grader) and reduced to the first block of their InChIKey; stereo and charge are
ignored. Replicating that exactly matters because it is the instrument every
experiment is measured with.
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

import numpy as np

# --- monoisotopic masses (CODATA-consistent values used by RDKit) ---
MONO = {
    "H": 1.0078250319,
    "C": 12.0,
    "N": 14.0030740052,
    "O": 15.9949146221,
    "S": 31.97207069,
    "P": 30.97376151,
    "F": 18.99840320,
    "Cl": 34.96885271,
    "Br": 78.9183376,
    "I": 126.904468,
    "Na": 22.98976928,
    "K": 38.96370649,
    "Si": 27.976926532,
    "B": 11.0093055,
    "Se": 79.9165218,
}

PROTON = 1.007276466621
ELECTRON = 0.000548579909
H2O = 2 * MONO["H"] + MONO["O"]

# Precursor m/z offset per adduct, as an exact physical quantity:
#   [M+H]+      -> M + m(H+)            = M + proton
#   [M+Na]+     -> M + m(Na) - m(e)     (the sodium is a bare cation)
#   [M-H]-      -> M - m(H) + m(e)      (a proton is removed, electron retained)
#   [M+Cl]-     -> M + m(Cl) + m(e)
#   [M+CH2O2-H]- -> M + m(HCOOH) - m(H) + m(e)
# Getting the electron bookkeeping right matters: it is a ~0.0005 Da systematic
# error, i.e. ~1.5 ppm at m/z 330, which is enough to matter at 10 ppm tolerance.
# Verified empirically against test.parquet in verify_adducts.py.
_M_NA = 22.98976928
_M_K = 38.96370649
_M_CL = 34.96885271
_M_HCOOH = MONO["C"] + 2 * MONO["O"] + 2 * MONO["H"]

# adduct -> (precursor m/z offset, .neutral mass shift M_obs - M_analytical)
ADDUCT_OFFSET = {
    "[M+H]+": (PROTON, 0.0),
    "[M+NH4]+": (MONO["N"] + 4 * MONO["H"] + PROTON, 0.0),
    "[M-H2O+H]+": (PROTON - H2O, 0.0),
    "[M-2H2O+H]+": (PROTON - 2 * H2O, 0.0),
    "[M+Na]+": (_M_NA - ELECTRON, MONO["Na"] - 2 * ELECTRON),
    "[M+K]+": (_M_K - ELECTRON, _M_K - 2 * ELECTRON),
    "[M-H]-": (-MONO["H"] + ELECTRON, 0.0),
    "[M-H2O-H]-": (-MONO["H"] - H2O + ELECTRON, 0.0),
    "[M+CH2O2-H]-": (_M_HCOOH - MONO["H"] + ELECTRON, 0.0),
    "[M+Cl]-": (_M_CL + ELECTRON, _M_CL),
}

# kept for backwards compatibility / introspection
ADDUCTS_POS = {k: v for k, v in ADDUCT_OFFSET.items() if k.endswith("+")}
ADDUCTS_NEG = {k: v for k, v in ADDUCT_OFFSET.items() if k.endswith("-")}

_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")


def formula_mass(formula: str) -> float:
    """Monoisotopic mass of a plain molecular formula, e.g. 'C15H10O5'."""
    total = 0.0
    consumed = 0
    for el, n in _FORMULA_TOKEN.findall(str(formula)):
        total += MONO[el] * (int(n) if n else 1)
        consumed += len(el) + len(n)
    if consumed != len(str(formula).strip()):
        raise ValueError(f"unparsed formula: {formula!r}")
    return total


def formula_from_smiles(smiles: str) -> str | None:
    """
    Molecular formula (Hill notation) for a SMILES, or None.

    Used to build the same-formula candidate frame: measured on the visible test,
    the true structure shares the query's formula 400/400 times, while only 10.7%
    of a +/-40 ppm mass window does. So the formula is the single most selective
    correct filter available.
    """
    if not smiles or not isinstance(smiles, str):
        return None
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import rdMolDescriptors

        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return rdMolDescriptors.CalcMolFormula(mol)
    except Exception:
        return None


def formula_neutral_mass(formula: str) -> float | None:
    """
    Neutral monoisotopic mass straight from a molecular formula, or None.
    Neutral monoisotopic mass straight from a molecular formula, or None.

    Preferred over precursor+adduct conversion whenever a formula exists, because
    it is immune to both precursor m/z error (riken is ~0.005 Da off) and wrong
    adduct labels (seen in gnps). Both 0.41-class solutions in this competition
    do the same, and in this project's own data the adduct-derived route produced
    -2000..-2570 ppm outliers -- the signature of a mislabelled adduct, which
    evicts the true structure from the mass window.
    """
    if not formula or not isinstance(formula, str):
        return None
    f = formula.strip()
    if not f:
        return None
    try:
        m = formula_mass(f)
    except Exception:
        return None
    return m if m > 0 else None


def _delta_mass(spec: Iterable[str]) -> float:
    """Mass added to the neutral molecule by adduct modifier terms (legacy helper)."""
    total = 0.0
    for term in spec:
        sign = 1.0
        t = term
        if t.startswith("-"):
            sign, t = -1.0, t[1:]
        if "+" in t:
            raise ValueError(f"bad adduct term: {term!r}")
        if t == "H2O":
            total += sign * H2O
        elif t == "2H2O":
            total += sign * 2 * H2O
        else:
            total += sign * formula_mass(t)
    return total


def neutral_mass_from_precursor(precursor_mz: float, adduct: str) -> float:
    """
    Convert an observed precursor m/z to the neutral monoisotopic mass.

    Uses the exact per-adduct precursor offset (including electron mass), so
    [M+Na]+ correctly subtracts a bare sodium cation rather than 'Na + H'.
    """
    if adduct not in ADDUCT_OFFSET:
        raise KeyError(f"unknown adduct {adduct!r}; known: {sorted(ADDUCT_OFFSET)}")
    return float(precursor_mz) - ADDUCT_OFFSET[adduct][0]


def precursor_mz_from_neutral(neutral_mass: float, adduct: str) -> float:
    """Inverse of neutral_mass_from_precursor."""
    if adduct not in ADDUCT_OFFSET:
        raise KeyError(f"unknown adduct {adduct!r}; known: {sorted(ADDUCT_OFFSET)}")
    return float(neutral_mass) + ADDUCT_OFFSET[adduct][0]


# ---------------------------------------------------------------- InChIKey14

_TAUTOMER_ENUM = None


def _tautomer_enum():
    global _TAUTOMER_ENUM
    if _TAUTOMER_ENUM is None:
        from rdkit.Chem.MolStandardize import rdMolStandardize

        _TAUTOMER_ENUM = rdMolStandardize.TautomerEnumerator()
    return _TAUTOMER_ENUM


def smiles_to_inchikey14(smiles: str, tautomer: bool = True) -> str | None:
    """
    Canonical InChIKey first block (14 chars) for a SMILES, or None if unparsable.

    Mirrors the grader: tautomer canonicalization then InChIKey, taking the
    first block so stereo/charge/protonation differences do not matter.
    """
    if smiles is None:
        return None
    s = str(smiles).strip()
    if not s:
        return None
    try:
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return None
        if tautomer:
            try:
                mol = _tautomer_enum().Canonicalize(mol)
            except Exception:
                pass  # fall back to the untautomerized molecule
        ik = Chem.MolToInchiKey(mol)
        if not ik or len(ik) < 14:
            return None
        return ik[:14]
    except Exception:
        return None


# ------------------------------------------------------------------- metric


def reciprocal_rank(predicted: Sequence[str], truth_keys: set[str],
                    max_rank: int = 25) -> float:
    """
    Reciprocal rank of the first predicted candidate matching the truth.

    truth_keys is a set so a truth can carry several equivalent keys (e.g. a
    tautomer-sensitive and a tautomer-insensitive form).
    """
    for i, smi in enumerate(predicted[:max_rank], start=1):
        k = smiles_to_inchikey14(smi)
        if k is not None and k in truth_keys:
            return 1.0 / i
    return 0.0


def mrr25(predictions: dict[str, Sequence[str]],
          truths: dict[str, str | Iterable[str]],
          max_rank: int = 25) -> float:
    """
    MRR@25 over molecules.

    predictions: molecule_id -> ranked SMILES list
    truths:      molecule_id -> true SMILES (or several acceptable ones)
    """
    if not truths:
        return 0.0
    total = 0.0
    for mid, truth in truths.items():
        if isinstance(truth, str):
            keys = {smiles_to_inchikey14(truth)}
        else:
            keys = {smiles_to_inchikey14(t) for t in truth}
        keys.discard(None)
        preds = predictions.get(mid) or []
        total += reciprocal_rank(preds, keys, max_rank)
    return total / len(truths)


def mrr25_breakdown(predictions: dict[str, Sequence[str]],
                    truths: dict[str, str | Iterable[str]],
                    max_rank: int = 25) -> dict:
    """MRR@25 plus rank histogram and hit@k, for diagnosing recall vs ranking."""
    ranks = []
    for mid, truth in truths.items():
        keys = ({smiles_to_inchikey14(truth)} if isinstance(truth, str)
                else {smiles_to_inchikey14(t) for t in truth})
        keys.discard(None)
        preds = predictions.get(mid) or []
        r = 0
        for i, smi in enumerate(preds[:max_rank], start=1):
            k = smiles_to_inchikey14(smi)
            if k is not None and k in keys:
                r = i
                break
        ranks.append(r)
    ranks = np.asarray(ranks)
    n = len(ranks)
    hit = ranks > 0
    return {
        "mrr25": float(np.mean(np.where(hit, 1.0 / np.maximum(ranks, 1), 0.0))) if n else 0.0,
        "n": n,
        "top1": float((ranks == 1).mean()) if n else 0.0,
        "hit@5": float((hit & (ranks <= 5)).mean()) if n else 0.0,
        "hit@10": float((hit & (ranks <= 10)).mean()) if n else 0.0,
        "hit@25": float(hit.mean()) if n else 0.0,
        "mean_rank_when_hit": float(ranks[hit].mean()) if hit.any() else float("nan"),
    }


# --------------------------------------------------------------- submission

MAX_CANDIDATES = 25


def dedupe_candidates(smiles: Iterable[str], keep_order: bool = True) -> list[str]:
    """Drop empty/duplicate candidates by InChIKey14, preserving rank order."""
    out, seen = [], set()
    for s in smiles:
        if s is None:
            continue
        s = str(s).strip()
        if not s:
            continue
        k = smiles_to_inchikey14(s)
        if k is None:
            continue  # never waste a slot on an unparsable SMILES
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out


def format_submission(predictions: dict[str, Sequence[str]],
                      molecule_ids: Sequence[str],
                      max_candidates: int = MAX_CANDIDATES):
    """
    Build the submission frame. Every molecule_id appears exactly once, in the
    given order, with ';'-joined candidates (best first, at most 25).

    Molecules with no candidates get an empty string; the caller should ensure
    that does not happen, because the grader rejects null/empty cells.
    """
    import pandas as pd

    rows = []
    for mid in molecule_ids:
        cands = dedupe_candidates(predictions.get(mid) or [])[:max_candidates]
        rows.append({"molecule_id": mid, "smiles": ";".join(cands)})
    return pd.DataFrame(rows, columns=["molecule_id", "smiles"])


def validate_submission(df, expected_ids: Sequence[str],
                        max_candidates: int = MAX_CANDIDATES) -> list[str]:
    """Return a list of problems; empty list means the submission is valid."""
    problems = []
    if list(df.columns) != ["molecule_id", "smiles"]:
        problems.append(f"wrong columns: {list(df.columns)}")
        return problems
    ids = list(df["molecule_id"])
    if len(ids) != len(expected_ids):
        problems.append(f"row count {len(ids)} != expected {len(expected_ids)}")
    if len(set(ids)) != len(ids):
        problems.append("duplicate molecule_id present")
    missing = set(expected_ids) - set(ids)
    if missing:
        problems.append(f"{len(missing)} expected ids missing, e.g. {list(missing)[:3]}")
    if df["smiles"].isna().any():
        problems.append("null smiles present")
    empty = (df["smiles"].astype(str).str.strip() == "").sum()
    if empty:
        problems.append(f"{empty} empty smiles rows")
    counts = df["smiles"].astype(str).str.split(";").apply(
        lambda xs: len([x for x in xs if x.strip()]))
    if (counts > max_candidates).any():
        problems.append(f"some rows exceed {max_candidates} candidates (max {counts.max()})")
    bad = []
    for i, s in enumerate(df["smiles"].astype(str)):
        for c in s.split(";"):
            c = c.strip()
            if c and smiles_to_inchikey14(c) is None:
                bad.append((i, c))
                break
    if bad:
        problems.append(f"{len(bad)} rows contain unparsable SMILES, e.g. {bad[:2]}")
    return problems


# ------------------------------------------------------------ self-check

def _self_check():
    # formula masses
    assert abs(formula_mass("H2O") - H2O) < 1e-9
    assert abs(formula_mass("C6H12O6") - 180.0633881) < 1e-4

    g = formula_mass("C6H12O6")
    # every adduct must round-trip neutral mass -> precursor m/z -> neutral mass
    for ad in ADDUCT_OFFSET:
        mz = precursor_mz_from_neutral(g, ad)
        back = neutral_mass_from_precursor(mz, ad)
        assert abs(back - g) < 1e-9, f"round-trip failed for {ad}: {back} vs {g}"

    # explicit physical checks, to catch sign/electron errors
    assert abs(precursor_mz_from_neutral(g, "[M+H]+") - (g + PROTON)) < 1e-12
    assert abs(precursor_mz_from_neutral(g, "[M-H]-") - (g - MONO["H"] + ELECTRON)) < 1e-12
    # [M+Na]+ must be a bare sodium cation, NOT Na+H
    expected_na = g + _M_NA - ELECTRON
    assert abs(precursor_mz_from_neutral(g, "[M+Na]+") - expected_na) < 1e-12
    # the old (wrong) behaviour differed by a proton's worth; prove we fixed it
    wrong_na = g + MONO["Na"] + PROTON
    assert abs(expected_na - wrong_na) > 1.0, "Na adduct still wrong"
    # [M+NH4]+ carries an extra proton, so it is N+4H+proton
    assert abs(precursor_mz_from_neutral(g, "[M+NH4]+")
               - (g + MONO["N"] + 4 * MONO["H"] + PROTON)) < 1e-12
    # water-loss adducts
    assert abs(precursor_mz_from_neutral(g, "[M-H2O+H]+") - (g - H2O + PROTON)) < 1e-12
    assert abs(precursor_mz_from_neutral(g, "[M-2H2O+H]+") - (g - 2 * H2O + PROTON)) < 1e-12
    # formic-acid adduct: [M+CH2O2-H]- = M + HCOOH - H + e
    assert abs(precursor_mz_from_neutral(g, "[M+CH2O2-H]-")
               - (g + _M_HCOOH - MONO["H"] + ELECTRON)) < 1e-12
    # chloride adduct: [M+Cl]- = M + Cl + e
    assert abs(precursor_mz_from_neutral(g, "[M+Cl]-") - (g + _M_CL + ELECTRON)) < 1e-12
    # unknown adduct must raise, not silently guess
    try:
        neutral_mass_from_precursor(300.0, "[M+Li]+")
        raise AssertionError("unknown adduct did not raise")
    except KeyError:
        pass

    # InChIKey14 is tautomer/stereo insensitive
    k1 = smiles_to_inchikey14("CC(=O)CC(=O)C")          # acetylacetone keto
    k2 = smiles_to_inchikey14("CC(O)=CC(=O)C")           # enol tautomer
    assert k1 is not None and k1 == k2, f"tautomers differ: {k1} vs {k2}"
    # stereochemistry ignored
    s1 = smiles_to_inchikey14("C[C@H](N)C(=O)O")
    s2 = smiles_to_inchikey14("C[C@@H](N)C(=O)O")
    assert s1 is not None and s1 == s2, f"stereo differ: {s1} vs {s2}"
    # unparsable handling
    assert smiles_to_inchikey14("not_a_smiles") is None
    assert smiles_to_inchikey14("") is None
    assert smiles_to_inchikey14(None) is None

    # metric
    truth = {"m1": "c1ccccc1O"}   # phenol
    assert abs(mrr25({"m1": ["c1ccccc1O"]}, truth) - 1.0) < 1e-12
    assert abs(mrr25({"m1": ["CCO", "c1ccccc1O"]}, truth) - 0.5) < 1e-12
    assert abs(mrr25({"m1": ["CCO"]}, truth) - 0.0) < 1e-12
    # rank 25 gives 1/25
    preds25 = ["CCO"] * 24 + ["c1ccccc1O"]
    assert abs(mrr25({"m1": preds25}, truth) - 1 / 25) < 1e-12
    # beyond 25 does not count
    preds26 = ["CCO"] * 25 + ["c1ccccc1O"]
    assert abs(mrr25({"m1": preds26}, truth) - 0.0) < 1e-12
    # duplicate equivalent candidates collapse
    assert reciprocal_rank(["CCO", "CCO", "c1ccccc1O"], {"x"}) == 0.0

    # submission format
    import pandas as pd

    df = format_submission({"m1": ["CCO", "CCO", "c1ccccc1O"]}, ["m1", "m2"])
    assert list(df.columns) == ["molecule_id", "smiles"]
    assert df.loc[0, "smiles"] == "CCO;c1ccccc1O"
    assert df.loc[1, "smiles"] == ""
    probs = validate_submission(df, ["m1", "m2"])
    assert any("empty" in p for p in probs), probs
    good = format_submission({"m1": ["CCO"], "m2": ["c1ccccc1O"]}, ["m1", "m2"])
    assert validate_submission(good, ["m1", "m2"]) == []
    # 25-cap enforced
    many = format_submission({"m1": [f"C{'C' * i}" for i in range(1, 40)]}, ["m1"])
    assert len(many.loc[0, "smiles"].split(";")) <= 25

    print("ALL SELF-CHECKS PASSED")


if __name__ == "__main__":
    _self_check()
