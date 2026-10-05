# Neutral-mass inference error in spectral-library metadata, and a lower bound on isobaric ambiguity

**Subtitle**: two independently verifiable metabolomics measurements, independent of the competition

**Document type**: research report (method and measurements)
**Date**: 2026-10-04
**Data**: Enveda CASMI 2026 `train.parquet` (2,539,608 rows / 18 columns), whose `ingest_lib`
field records which public spectral library each row came from
**Methodological stance**: this report states only results that are **directly measurable and
reproducible from independent data**. Anything related to competition ranking is excluded from
the conclusions.

---

## Abstract

On 2.17 million MS/MS records carrying a library-source label, we quantify the disagreement
between the standard practice of **inferring neutral mass from precursor m/z plus an adduct
label** (the *adduct route*) and **computing neutral mass directly from the molecular formula**
(the *formula route*). Two conclusions of general relevance follow:

1. **The adduct route's error is far larger than instrumental mass accuracy.** In 3.34% of
   records globally the two routes disagree by more than 1000 ppm, i.e. a neutral-mass deviation
   of about 0.5 Da (at m/z 500). Modern high-resolution instruments determine precursor m/z to
   1–5 ppm. The error exceeds instrumental precision by **100–1000×**, so its source cannot be
   mass measurement — it can only be **adduct misassignment**. The rate is extremely uneven
   across libraries (spectraverse 35.88%, enveda-180 0.57%), which makes this a
   **library-specific, attributable, fixable infrastructure problem**.

2. **Isobaric ambiguity has a *lower bound* set by small mass-defect substitutions, and
   tightening the mass window cannot remove it.** On a real candidate pool, even narrowing the
   window to **3 ppm** leaves **50.2%** of queries with more than one molecular formula in range;
   at **1 ppm** it is still **12.5%**. The reason is that substitutions like O↔CH₄ and CO↔C₂H₄
   have a mass defect of ±36.4 mDa (equivalent to **±91 ppm** at m/z 400), NH₂↔O is 23.8 mDa
   (59.5 ppm), and S↔O₂ is 17.8 mDa (44.4 ppm). These defects are **larger** than a typical
   search window, so **the molecular formula is the only clean discriminator — the mass window is
   not**.

Both results have direct implications for metabolomics retrieval pipelines and benchmark design,
and both can be reproduced independently from the public libraries' original files.

---

## 1. Background and motivation

The standard untargeted-metabolomics pipeline assumes: given precursor m/z and adduct type, infer
the neutral mass and retrieve candidates from a structure library by mass window. Two
long-standing difficulties are usually treated as "tuning problems":

- **Adduct assignment**: the same m/z can correspond to different neutral molecules under
  different adducts;
- **Isobaric interference**: different molecular formulas can have nearly identical exact mass.

Literature and engineering practice usually respond in two ways: raise instrument resolution
(shrink the ppm window), and enlarge the candidate library. This work **measures the magnitude of
both difficulties** and tests whether "tighten the mass window" is an effective remedy.

The data we use has one favourable property: every row carries a **library-source label**
(`ingest_lib`), so the error can be **attributed per library** rather than reported only as a
single aggregate rate. That is the key to the actionable conclusions here.

---

## 2. Data and method

### 2.1 Data

`train.parquet`: 2,539,608 rows, 18 columns, 2.82 GB. Key fields:

| Field | Meaning |
|---|---|
| `ingest_lib` | which public library the spectrum came from (11 classes) |
| `molecular_formula` | molecular formula (the truth, used by the formula route) |
| `precursor_mz` | observed precursor m/z |
| `adduct` | adduct label (e.g. `[M+H]+`, `[M-H]-`, `[M+CH2O2-H]-`) |
| `ms2_mzs` / `ms2_normalized_intensities` | peak lists |

### 2.2 The two neutral-mass routes

**Formula route** (treated here as the reference):
$$M_{\text{formula}} = \sum_{e \in \text{formula}} n_e \, m_e$$
i.e. the sum of the monoisotopic masses of the formula's elements.

**Adduct route** (current standard practice):
$$M_{\text{adduct}} = \text{precursor\_mz} - \Delta_{\text{adduct}}$$
where $\Delta_{\text{adduct}}$ is that adduct's exact mass shift. We use a shift table that
includes the **electron-mass correction** (non-trivial: at m/z 330 the electron mass equals
1.66 ppm, not negligible against a 10 ppm tolerance):

| Adduct | $\Delta$ (Da) |
|---|---|
| `[M+H]+` | +1.007276 |
| `[M+Na]+` | +22.989221 |
| `[M+K]+` | +38.963158 |
| `[M+NH4]+` | +18.033823 |
| `[M-H2O+H]+` | −17.003289 |
| `[M-H]-` | −1.007825 |
| `[M+Cl]-` | +34.969402 |
| `[M+CH2O2-H]-` | +44.998203 |

**Disagreement metric**:
$$\delta = \frac{|M_{\text{formula}} - M_{\text{adduct}}|}{M_{\text{formula}}} \times 10^6 \ \text{(ppm)}$$

Both routes are computed only where a formula is available.

### 2.3 Measuring the isobaric lower bound

On a 730,161-structure candidate pool (729,761 with a formula, 125,343 distinct formulas), we
sample 400 real masses as queries and count, within a given ppm window:

- the number of **structures** in range (including isomers);
- the number of **distinct molecular formulas** in range (the isobar count).

Their ratio is "how many true isomer competitors each formula has on average".

### 2.4 Reproduction

```bash
python scripts/isobar_floor.py          # the tables in section 3
# per-library attribution in section 3.1: same computation as scripts/verify_formula_claim.py
```
The only dependencies are `numpy`/`pandas`/`pyarrow`; no external downloads.

---

## 3. Results

### 3.1 Per-library attribution: the adduct route's error is extremely uneven

On 2,171,648 records carrying a formula:

| Library | Records | \|δ\| > 10 ppm | **\|δ\| > 1000 ppm** |
|---|---:|---:|---:|
| **spectraverse** | 46,857 | 35.88% | **35.88%** |
| mona | 90,614 | 13.36% | **8.86%** |
| enveda-np-examples | 1,184 | 8.45% | **7.43%** |
| gnps | 212,187 | 10.04% | **6.89%** |
| msdial | 37,045 | 5.71% | 5.18% |
| drug_plus | 2,441 | 5.24% | 5.20% |
| riken | 338,478 | 28.09% | 4.85% |
| massbank | 97,118 | 10.26% | 1.89% |
| pluskal_ms2 | 523,112 | 1.65% | 1.52% |
| enveda-180 | 821,960 | 0.57% | 0.57% |
| masaryk | 652 | 0.15% | 0.00% |
| **Total** | **2,171,648** | **7.87%** | **3.34%** |

**Key observations**:

- Globally **3.34%** of records disagree by >1000 ppm. At m/z 500, 1000 ppm ≈ 0.5 Da.
- Modern high-resolution instruments (Orbitrap, timsTOF) determine precursor mass to **1–5 ppm**.
  The error exceeds instrumental precision by **200–1000×** and **cannot be explained by mass
  measurement error** — it can only be adduct misassignment or an inconsistent precursor
  convention.
- **For spectraverse the >10 ppm and >1000 ppm rates are identical (35.88%)**, which suggests
  this is not random noise but a **systematic convention difference** (a whole batch whose
  adduct/precursor definitions differ from the standard).
- Extremely uneven: the highest (35.88%) and lowest (0.15%) differ by **240×**. That makes the
  error **attributable and fixable**, not intrinsic noise of the field.

### 3.2 The isobaric ambiguity lower bound

| ppm window | Accuracy class | Distinct **formulas** in range (median) | Queries with >1 formula |
|---:|---|---:|---:|
| 0.5 | top-end Orbitrap | 1 | 3.8% |
| 1.0 | Orbitrap / timsTOF | 1 | 12.5% |
| 2.0 | routine high-res | 1 | 30.2% |
| **3.0** | routine high-res | **2** | **50.2%** |
| 5.0 | low-end high-res | 2 | 67.2% |
| 10.0 | near unit resolution | 5 | 82.8% |
| 40.0 | common loose window | 18 | 95.8% |

**Even at a 1 ppm window, 12.5% of queries face more than one molecular formula.**

### 3.3 Why the bound exists: small mass-defect substitutions

| Substitution | Mass defect (mDa) | Equivalent at m/z 400 |
|---|---:|---:|
| O ↔ CH₄ | 36.39 | **91.0 ppm** |
| CO ↔ C₂H₄ | 36.39 | **91.0 ppm** |
| NH₂ ↔ O | 23.81 | 59.5 ppm |
| S ↔ O₂ | −17.76 | 44.4 ppm |
| N ↔ CH₂ | 2003.07 | 5007.7 ppm |

O↔CH₄, CO↔C₂H₄ and NH₂↔O are precisely the **most common heteroatom substitutions** in natural
products and metabolites. Their mass defects (17.8–36.4 mDa) are **larger** than a 1–10 ppm search
window (0.4–4 mDa at m/z 400).

**Conclusion**: over the common mass range, "tightening the mass window" **cannot in principle**
separate these formulas, because separating O↔CH₄ would need <0.5 ppm (i.e. >2,000,000 resolving
power), and such substitutions are ubiquitous in real molecules.

### 3.4 The molecular formula is a far more efficient selector than the mass window

Measurements under two **different query populations** (both medians):

| Query population | Candidates within ±40 ppm | Exact-formula candidates |
|---|---:|---:|
| The candidate pool's own masses (400 sampled at random) | **167** | — |
| 400 test molecules with known truth (truth recovered verbatim from train) | **361** | **36** |

For the test molecules, **only 10.7% of the ±40 ppm candidates have the correct formula**. That is,
about **89% of candidates are wrong-formula isobars** and mathematically cannot be the answer.

The difference between the two rows is itself informative: the test molecules' window candidate
count (361) is **2.2×** the pool's random sample (167), consistent with §4.3 — the test set sits in
a region of chemical space with **denser candidates**.

> **Note on populations**: 361/36/10.7% come from `scripts/verify_formula_claim.py` (queries = test
> molecules); 167 comes from random sampling of the pool itself. This report does not mix the two.

---

## 4. Discussion

### 4.1 What conclusion 1 means

Current pipelines treat "precursor + adduct → neutral mass" as reliable input. Our measurement
shows that within one **internally consistent** merged dataset, 3.34% of records deviate from the
truth by more than 1000 ppm, and **that rate is determined by library source** (0.15%–35.88%).

Two implications:

1. **The dominant error term is misidentified.** The field's pursuit of "high resolution" targets
   mass-measurement precision (ppm scale). This report shows that for a substantial share of
   library records the dominant error is **adduct assignment** (10³ ppm scale). Further resolution
   gains are **ineffective** against that class of error.
2. **It is a fixable infrastructure item.** The error concentrates in a few libraries, and the
   worst (spectraverse) shows the systematic signature of ">10 ppm and >1000 ppm at the same
   rate" — pointing at **a whole batch's convention being inconsistent** rather than per-record
   random error. It can therefore be fixed by **re-calibrating each library's adduct convention**,
   at an engineering cost rather than a scientific one.

**Actionable recommendation for downstream users**: neutral mass should preferentially be computed
from the formula (when available), falling back to the adduct route only when the formula is
missing. Two independent top competition solutions already do this, but no quantitative
justification had been published; this report supplies one, with its magnitude.

### 4.2 What conclusion 2 means

"Raising resolution solves isobaric interference" is a common default. This report gives a
counterexample **and the mechanism** (§3.3, small mass-defect substitutions).

The quantitative upper bound: at 3 ppm (today's routine high-end), **half** of all queries face
multiple formulas; only at 1 ppm does it fall to 12.5%, and <0.5 ppm is not practically
achievable. Therefore:

- **The mass dimension is close to saturated.** Tightening ppm further inside the common working
  window yields diminishing returns and cannot remove O↔CH₄-type ambiguity.
- **Real discrimination must come from orthogonal information**: MS/MS fragmentation patterns
  (structure-relevant evidence), isotopic fine structure, chromatographic retention time. This
  measurement provides a **quantitative argument** for why structure-relevant evidence such as
  in-silico fragmentation must be introduced, rather than a methodological preference.

### 4.3 An incidental observation: the "commonness" of a formula is highly skewed

| Structures sharing the formula | Pool overall (729,761 structures) | Test truth (400 molecules) |
|---|---:|---:|
| Median | **1** | **43** |
| p25 / p75 | — | 12 / 125 |
| Share with exactly one structure | **52.6%** | **3.8%** |

Across the structure library as a whole, **52.6% of formulas correspond to exactly one
structure** — i.e. "fix the formula first" almost directly yields the answer. But on the test set
that share falls to 3.8%, with formulas generally in "popular" regions (median 43 isomers).

**Implication**: the difficulty of the formula→structure mapping **is not a constant; it depends
strongly on how crowded that formula is in the library**. This suggests:

- any benchmark design that estimates difficulty from "random formulas" will **systematically
  underestimate** real difficulty;
- for machine learning, this is a **usable prior feature** (the formula's in-library frequency),
  and it is **independent of the spectrum** and computable directly from the structure library.

---

## 5. Limitations and counter-arguments

1. **A single merged dataset.** All measurements come from CASMI 2026's `train.parquet`. Although
   its 11 `ingest_lib` sources are independent public libraries, the merge and re-annotation were
   done by the organisers and may introduce a **shared processing bias**. The "inter-library
   variance" conclusion therefore needs **independent reproduction from each library's original
   files** to be confirmed.
2. **The formula route is treated as the reference, but it is not error-free.** Formulas
   themselves come from library records; if a row's formula is wrong, this report attributes that
   error to the adduct route, which would **overestimate** the opposing error rate. Mitigating
   evidence: the disagreement is strongly library-clustered (rather than globally scattered),
   consistent with "a specific library's adduct convention is inconsistent" rather than "formulas
   are broadly wrong".
3. **Samples are not independent.** Multiple spectra of the same record share mass and adduct, so
   the effective sample size is smaller than the row count; the confidence intervals on these
   rates are wider than the row count suggests.
4. **§3.4's ratio (10.7%) depends on the candidate pool's composition**, and that pool is our own
   merge (train ∪ COCONUT ∪ LOTUS ∪ NPAtlas). Other structure libraries would change the absolute
   numbers; but the **direction of the conclusion** (the formula is a far stronger selector than
   the mass window) rests on the physical mechanism in §3.3 and does not depend on the pool.
5. **Downstream impact was not measured.** This report measures the error distribution and
   candidate-set selectivity, but **not** how many identifications these errors actually cause to
   fail. That is the most important follow-up experiment (see §6).
6. **A logical self-reminder**: in an evaluation with known truth, "the truth shares the query's
   formula" is a tautology of the evaluation definition and is not a finding. This report uses only
   the **selectivity ratio** (10.7%, 36 out of 361) as a result, not the tautological part.
7. **These "truths" come from the visible test set, not the evaluation set.** Every measurement
   here that uses "test molecules" as queries (§3.4 second row, §4.3 right column) uses truth
   recovered from the training set by verbatim peak-set matching, and **holds only for the visible
   test set**. Independent verification (see §7.1) shows the evaluation set is **different**, so
   these numbers **cannot** be extrapolated to it. They describe an **internally consistent proxy
   set usable for regression detection**, whose value lies in method comparison (relative
   differences on the same proxy set), not in absolute difficulty estimates.

> **This affects §4.3 most**: the right column's "median 43 isomers in the test set" describes the
> proxy set, not the evaluation set. Hence §4.3's inference that "benchmark design underestimates
> real difficulty" **lacks sufficient evidence** and should be downgraded to a hypothesis awaiting
> verification (see §7.1).

---

## 6. Follow-up work (by priority)

1. **Independently reproduce the inter-library variance** (most critical). Download the original
   releases of MassBank / GNPS / MoNA / RIKEN / spectraverse directly, without third-party
   re-annotation, and recompute the $\delta$ distribution. If values like 35.88% reproduce, §4.1
   is promoted to an established conclusion.
2. **Decompose the attribution.** For high-error libraries, separate three causes: (a) wrong
   adduct label; (b) a different precursor convention (e.g. an isotope peak taken, or [M+Na]
   reported as [M+H]); (c) a wrong formula. Method: enumerate candidate adducts to drive
   $\delta \to 0$ and look at the distribution of the optimal adduct — if it concentrates on a few
   shifts, that is a systematic convention problem.
3. **Quantify downstream impact.** Under the standard ±10 ppm retrieval window, compute the share
   of cases where a metadata error makes the correct candidate unretrievable. This is the key step
   from an error rate to an **identification failure rate**.
4. **Benchmark audit.** Check whether public benchmarks (MassSpecGym, EnvedaDark, …) are affected
   by the same source — especially whether they reuse the libraries above.
5. **Transferability.** The isobar lower-bound computation uses only formulas and masses and no
   spectra, so it can be computed for **any** compound library (ChEMBL, PubChem, COCONUT) to give
   that library's formula-ambiguity spectrum at a given ppm. This could become a general tool:
   given a library and an instrument's precision, predict the **achievable maximum formula-level
   discrimination rate**.

---

## 7. A claim that was refuted, and the methodological principle it produced

### 7.1 Refuted: you cannot infer the evaluation set from the visible test set

During development a seemingly strong inference appeared: all 1,213 spectra of the visible
`test.parquet` exist **byte for byte** in the training set (peak-set MD5 matches exactly, all from
enveda-180), and a public solution **hard-coded the top-1 answers for 400 test `molecule_id`s** in
its notebook while claiming 0.411 — from which it was inferred that "the public test set is the
evaluation set and the leaderboard leaks".

**That inference has been refuted by measurement**, as follows:

| Evidence | Value |
|---|---|
| The same notebook's output file on the **visible test set** | 393,908 bytes |
| The **evaluation-set** output Kaggle recorded for the same submitted code | **530,716 bytes** |
| The same code's score against the proxy truth on the visible test set | **0.9902** |
| The same code's actual score on the evaluation set | **0.417** |

The same code on the same input cannot produce files differing by 35% ⇒ **the evaluation set's
input differs from the visible test set**. If the evaluation set equalled the visible test set with
training labels as the answers, that submission would have scored ≈1.0; it scored 0.417. **The
score settles it**; the byte counts are only corroboration.

### 7.2 The methodological principle (the transferable part)

The refutation itself yields an observation of general relevance to ML benchmark design:

> **A high score on a visible/placeholder test set can be completely decoupled from real evaluation
> performance, and that decoupling cannot be discovered by inspecting the test set itself.**

Specifically:

1. The visible test set's spectra are **byte-for-byte from the training set**, so 400/400 "truths"
   can be recovered from it and a high score of **0.9902** obtained — a score that **looks** like
   an excellent model.
2. The same model scores **0.417** on the real evaluation set. **The gap is 0.57.**
3. The crucial point: **inspecting the test set is not enough to expose the problem.** The fact
   that "the spectra exist entirely in the training set" can be read either as "the test set is a
   placeholder" (correct) or as "the leaderboard leaks" (wrong). Telling them apart requires **the
   evaluation set's score or its input size**, neither of which is normally available during a
   competition.

**Practical implications**: when developing against any benchmark with a placeholder test set,
(a) the placeholder may be used only for **regression detection** (relative differences on the same
set) and **data understanding**;
(b) it must **not** be used to estimate absolute difficulty, tune parameters, or claim
generalisation;
(c) unless independent evidence (such as the evaluation score) shows the two are identically
distributed.

This is consistent with the caveat in §5.7: every measurement in this report that relies on
"recovered truth" must be read accordingly.

### 7.3 Impact on this report's conclusions

| Conclusion | Affected? |
|---|---|
| §3.1 per-library attribution of the adduct-route error | **unaffected** (uses only the training set's own formula/precursor/adduct fields) |
| §3.2 / §3.3 isobaric lower bound | **unaffected** (uses only the candidate pool's formulas and masses) |
| §3.4 first row (pool random sample, 167) | **unaffected** (queries drawn from the pool itself) |
| §3.4 second row + §4.3 (queries = test molecules) | **qualified**: describes the visible test set; not extrapolable |
| §4.3 "benchmarks underestimate real difficulty" | **downgraded to a hypothesis awaiting verification** |

**So this report's two main conclusions (§4.1, §4.2) depend on no test set at all** — they use only
training-set annotations and the structure library itself. That is precisely the value of
decoupling the research question from the competition.

---

## 8. Conclusions

1. **The dominant error in neutral-mass inference is not mass measurement but adduct assignment.**
   Among 2.17 million public-library records, 3.34% have an adduct-derived mass deviating from the
   formula-derived value by more than 1000 ppm — about 200–1000× instrumental mass precision. The
   error varies 240× across libraries (0.15%–35.88%) and shows systematic rather than random
   characteristics, making it an attributable, fixable infrastructure problem.

2. **Isobaric ambiguity has a physical lower bound that tightening the mass window cannot remove.**
   Because common heteroatom substitutions (O↔CH₄, CO↔C₂H₄, NH₂↔O) have mass defects of
   17.8–36.4 mDa — larger than a ppm-scale search window — even at 3 ppm 50.2% of queries face
   multiple formulas (12.5% at 1 ppm). **The molecular formula is the only clean selector**
   (361 → 36 candidates, retaining the truth); within a 40 ppm window, 89% of candidates have the
   wrong formula.

3. **Together they point to one methodological conclusion**: over the common mass range,
   **precursor-mass-based discrimination is close to saturated**, and further gains must come from
   (a) correcting adduct/metadata conventions, and (b) introducing structurally relevant
   orthogonal evidence (MS/MS fragment interpretation, isotopic fine structure, retention time).
   This report supplies quantitative support for that conclusion rather than a statement of
   preference.

4. **(Methodological, independent of the measurements above) A high score on a placeholder test
   set can be completely decoupled from real evaluation performance, and this cannot be discovered
   by inspecting the test set itself.** The same code scores **0.9902** on the visible test set
   (whose spectra are byte-for-byte from training) and **0.417** on the real evaluation set — a gap
   of 0.57; reading "the spectra exist entirely in the training set" as "the leaderboard leaks" is
   a **seemingly strong but refuted** inference (§7). Practically: a placeholder set may be used
   only for regression detection and data understanding, never to estimate absolute difficulty or
   claim generalisation.

**Conclusions 1 and 2 depend on no test set** — they use only training-set annotations and the
structure library itself. That, too, is the value of decoupling the research question from the
competition: the independently reproducible part is the publishable part.

---

## Appendix: measurement scripts and reproducibility

### A. Scripts

| Script | Output |
|---|---|
| `scripts/isobar_floor.py` | the ppm–formula ambiguity table in §3.2 and the substitution mass-defect table in §3.3 |
| `scripts/build_candidate_pool.py` | builds the 730,161-structure candidate pool used in §3.2/§3.4 |
| `scripts/verify_formula_claim.py` | the candidate-set selectivity in §3.4 (361 → 36, 10.7%) |
| `src/casmi/core.py` | the electron-mass-corrected adduct shift table; formula mass and formula parsing |

### B. Numerical self-verification log

Every key number in this report was recomputed and checked by script:

| Report location | Value | How verified | Result |
|---|---|---|---|
| §3.1 total | 2,171,648 rows / 7.87% / 3.34% | recomputed with per-library attribution | ✅ matches |
| §3.2 | 1 ppm → 12.5%; 3 ppm → 50.2%; 40 ppm → 95.8% | recomputed the same way in `isobar_floor` | ✅ matches |
| §3.2 | 5 ppm → 67.2%; 10 ppm → 82.8% | `isobar_floor.py` output | ✅ matches |
| §3.3 | O↔CH₄ = 36.39 mDa = 91.0 ppm @400 | computed directly from monoisotopic masses | ✅ matches |
| §3.4 | pool random sample ±40 ppm → 167 | recomputed from the pool's own masses | ✅ matches |
| §3.4 | test molecules ±40 ppm → 361; formula → 36; 10.7% | `verify_formula_claim.py` | ✅ matches |
| §4.3 | pool overall median 1 structure per formula (52.6% unique); test median 43 (3.8% unique) | binned statistics | ✅ matches |

**One imprecision corrected**: an earlier draft listed "±40 ppm candidate count 164–361" for the
pool random sample and the test molecules together, obscuring that they come from **different query
populations**. They have been separated and explained.

### C. Data and code availability

All measurements here are reproducible from the scripts above on the CASMI 2026 public training set
(CC BY-NC 4.0). The original library files needed for §6.1 are public resources (MassBank, GNPS,
MoNA, RIKEN, spectraverse) and **do not depend on competition data** — which is the key to whether
these conclusions generalise, and therefore the first follow-up task.

**Measurement environment**: Python 3.11.16, RDKit 2026.03.6, numpy 2.2.6, pandas 3.0.6.
