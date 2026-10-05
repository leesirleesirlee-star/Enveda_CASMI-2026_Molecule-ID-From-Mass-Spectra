# CASMI 2026 — full technical brief

**Document version**: 1.0
**Date**: 2026-10-03
**Audience**: the project's execution team
**Purpose**: to convey the competition's shape, submission flow, rule constraints and technical
background as a basis for project decisions

> Where our own measurements later differed from the figures quoted here (which came from the
> competition description and forum posts), the measured value is noted in the margin. The text is
> otherwise kept as written.

## 1. Competition overview

### 1.1 Basics

| Item | Content |
|---|---|
| **Name** | Enveda CASMI 2026 — Molecule ID From Mass Spectra |
| **Platform** | Kaggle |
| **Type** | Featured Code Competition |
| **Task** | predict a small molecule's 2-D chemical structure (SMILES) from LC-MS/MS spectra |
| **Host** | Enveda |
| **Prize pool** | $50,000 USD ($16,000 for first place) |
| **Start** | 2026-09-14 |
| **Registration / team merge deadline** | 2026-12-07 23:59 UTC |
| **Final submission deadline** | 2026-12-14 23:59 UTC |
| **Duration** | 90 days |

### 1.2 Background

CASMI (Critical Assessment of Small Molecule Identification) is a blind challenge founded in 2012,
inspired by CASP (the protein-structure prediction competition that produced AlphaFold). CASMI 2026
is the first challenge since 2022 and the first designed for the broader machine-learning
community.

Modern mass spectrometry can detect thousands of chemical signals in a complex biological sample
(blood, plant extract), yet in a typical untargeted study only about **10%** of signals match a
known molecule — the other **90%** are measured and recorded but never identified, leaving a large
amount of biological information and potential new drugs unexplored.

Enveda generated the CASMI 2026 test set with its drug-discovery platform: about **400 molecules**
corresponding to about **2,500 spectra**, **none of which has been published**. These molecules are
real or potential natural products and their analogs.

> **Measured later**: the public `test.parquet` holds **1,213 spectra / 400 molecules**.

### 1.3 Task definition

**Input**: LC-MS/MS data — fragment peak lists (m/z, intensity) plus precursor ion information.

**Output**: up to 25 candidate SMILES strings per `molecule_id`, ordered by confidence.

**Core challenge**: the test molecules are **genuinely novel**; the retrieval library may not
contain the right answer, so pure retrieval has a score ceiling. Competitors must handle spectral
library retrieval, candidate structure recall and de-novo structure prediction together.

**Prediction unit**: the unit is the **molecule**, not a single spectrum. One molecule may produce
several spectra at different collision energies or adducts, and the model must combine them into a
single candidate list.

## 2. Datasets

### 2.1 Official files

| File | Content |
|---|---|
| `train.parquet` | training set, about **2.5 million** MS/MS spectra covering about **275,000** unique structures |
| `test.parquet` | test set, about **1,500** spectra covering **400** molecules |
| `sample_submission.csv` | submission format template |

> **Measured later**: `train.parquet` is **2,539,608 rows / 275,810 structures**;
> `test.parquet` is **1,213 spectra**.

### 2.2 Training set detail

The training set integrates several public spectral libraries, including Enveda-180 (Enveda's
recently released open dataset, about **1.16 million** spectra from **184,330** synthetic small
molecules). Each row is one spectrum, carrying peak data and the corresponding structure labels
(`normalized_smiles`, `inchikey14`, `molecular_formula`).

### 2.3 Test set detail

- **1 to 16** spectra per molecule, median **3**
- every test spectrum was acquired on a **Bruker timsTOF**
- test molecule monoisotopic masses **157 to 1,159 Da**, median about **348 Da**
- **Important**: competitors found that the test spectra (the enveda-180 rows) are **byte-for-byte
  identical** to training-set data, meaning some test spectra already have labels in the training
  set

### 2.4 External data rules

Publicly and freely available external data — including pretrained models — is allowed, subject to:

1. **Public and free**: the data or weights must be freely available to all competitors
2. **Compliant**: a pretrained model's release must comply with its training data's licence. The
   hosts stated explicitly: "we cannot blanket-approve all open-weight models. As long as the
   open-weight model's release is compliant with its training data licence, it should be fine."
3. **Disclosed**: a winning solution must disclose every external model and dataset used and
   confirm compliance
4. **Training data**: weights trained on `train.parquet` are allowed, whether trained by your own
   team or obtained from another competitor's public release

## 3. Metric: MRR@25

### 3.1 Definition

Submissions are evaluated by **Mean Reciprocal Rank @ 25 (MRR@25)**:

$$
\text{MRR@25} = \frac{1}{U} \sum_{u=1}^{U} \frac{1}{\text{rank}_u}
$$

where $U$ is the number of molecules and $\text{rank}_u$ the position of the first correct
structure in the candidate list.

### 3.2 Scoring rules

- each molecule has exactly one correct structure, and **only the first correct guess counts**
- a correct guess at position **1** scores **1.0**
- at position **2**, **0.5**
- at position **25**, **0.04**
- no correct answer among the first 25 candidates scores **0**
- the final score is the mean over all molecules

### 3.3 Matching rule

A prediction is correct when the predicted SMILES describes **the same atom connectivity** as the
answer. Both sides are canonicalised for tautomers with **RDKit 2026.03.3**, reduced to the **first
block of the InChIKey (InChIKey14)**, and compared.

## 4. Submission format

### 4.1 CSV requirements

| Column | Description |
|---|---|
| `molecule_id` | molecule identifier |
| `smiles` | candidate SMILES strings |

### 4.2 Ordering rules

- multiple candidate SMILES are joined with **semicolons (`;`)**
- **the best candidate must come first**
- at most **25** candidates per molecule
- fewer than 25 is allowed; more is not

### 4.3 Example

```
molecule_id,smiles
m_0014ef,CC1=CC(=O)C=CC1=O;OC(=O)c1ccccc1O;C1CCNCC1
```

### 4.4 File location

```
/kaggle/working/submission.csv
```

### 4.5 Invalid SMILES

Competitors reported that submissions padded with filler SMILES score 0.000. In grader v13 an
unparseable guess still consumes a rank position, so **do not include invalid SMILES** — it wastes
candidate slots.

## 5. Code competition constraints

### 5.1 Runtime

| Environment | Maximum runtime |
|---|---|
| CPU notebook | **≤ 9 hours** |
| GPU notebook | **≤ 9 hours** |

### 5.2 Code requirements

- a **single-file, self-contained Python program**
- **no** `pip install` or `conda install` of new packages
- **no internet access**
- no part of the code may be skipped; no unexecuted cells

### 5.3 Available packages

Preinstalled in the Kaggle environment: `pandas`, `numpy`, `torch`, `xgboost`, `scikit-learn`,
`transformers`, `lightgbm`, `torch-geometric` and others.

> **Measured later**: the image does **not** ship `rdkit`, so it must be packaged as a dataset and
> installed offline ([`EXTERNAL_RESOURCES.md`](EXTERNAL_RESOURCES.md) section B).

### 5.4 External assets

Every model weight and data asset must be **uploaded as a Kaggle Dataset in advance**; the notebook
only performs inference. Assets may include:

- pretrained weights (e.g. DreaMS, about 1.2 GB)
- spectral embedding stores
- reranker model files
- precomputed candidate pools

A Kaggle notebook's disk is about **20 GB**, so the total asset size must be controlled.

## 6. Timeline

| Event | Date |
|---|---|
| Start | 2026-09-14 |
| Registration / team merge deadline | 2026-12-07 23:59 UTC |
| Final submission deadline | 2026-12-14 23:59 UTC |

## 7. Technical background

### 7.1 MS/MS principles

MS/MS provides two kinds of core information for structure prediction:

1. **Precursor ion mass (m/z)**: the instrument measures the precursor's mass-to-charge ratio
   precisely, effectively constraining molecular mass and possible formula. For small molecules z
   is usually ±1.
2. **Fragment ion pattern**: the selected precursor is fragmented by collision with a neutral gas,
   and the instrument records each fragment's m/z and intensity. The resulting intensity histogram
   is the molecule's spectrum.

**Key terms**:
- **base peak**: the most intense fragment ion
- **precursor peak**: the intact unfragmented ion's m/z; may or may not be present depending on the
  degree of fragmentation
- **adduct**: the charged species the molecule acquired, e.g. [M+H]⁺, [M+NH₄]⁺, [M-H]⁻

### 7.2 The state of the art

The strongest public solutions' main engines are all **retrieval + reranking**, not end-to-end
generation. Retrieval matches against a large spectral library; the reranker fuses multi-dimensional
evidence (spectral similarity, fragment explanation, fingerprint similarity, …) to produce the
final order.

### 7.3 Suggested stack

| Component | Suggestion | Purpose |
|---|---|---|
| Spectral embedding | DreaMS (1024-d) | map spectra into a chemically meaningful embedding space |
| GPU similarity | SimMS (1000× speedup) | fast similarity over a large retrieval library |
| Fragment scoring | a pure-Python RDKit implementation | check whether a candidate explains the observed fragments |
| Reranker | XGBoost / LightGBM | fuse multi-channel evidence into a final order |
| Vector index | FAISS | approximate nearest-neighbour search |

> **Measured later**: DreaMS and SimMS were both evaluated and **deliberately left out** of the main
> line ([`EXPERIMENTS.md`](EXPERIMENTS.md), 2026-10-04).

## 8. Rule highlights

### 8.1 Data use

- competition data may not be published or shared outside your official Kaggle team
- weights trained on `train.parquet` are allowed

### 8.2 Winning requirements

- the winning method will be **open-sourced**
- the test set will become a **lasting benchmark** for the field

### 8.3 Compliance checklist

- [ ] every external dataset/model is public and free
- [ ] pretrained weights' release complies with their training data's licence
- [ ] every external resource is recorded
- [ ] the code is single-file and self-contained
- [ ] the notebook runs in under 9 hours
- [ ] the submission format matches `sample_submission.csv`
- [ ] MRR@25 is printed explicitly inside the notebook

> **Measured later**: a 30 h/week GPU quota, not the 9 h per run, is the binding constraint
> ([`PLAYBOOK.md`](PLAYBOOK.md) §3).

## 9. Execution instructions for the agent

1. **Understand the task**: it is a "spectrum → structure" prediction task evaluated by MRR@25. The
   correct answer must land inside the top 25 to score; higher ranks score more.

2. **Follow the submission format**: a CSV with `molecule_id` and `smiles`, candidates joined by
   semicolons, best first, at most 25, written to `/kaggle/working/submission.csv`.

3. **Follow the code constraints**: a single-file Python program, no `pip install`, no internet,
   runtime under 9 hours.

4. **External asset strategy**: upload every weight and data asset as a Kaggle Dataset in advance
   and only run inference in the notebook.

5. **Validation strategy**: use scaffold-disjoint validation sets, dynamically remove the correct
   answer from the retrieval library at evaluation time, and monitor MRR@25, top-1, top-10 recall
   and top-25 recall.

6. **External data compliance**: before using any external data or model, confirm it is public and
   licence-compliant, and record every external resource.

7. **Time management**: watch the registration deadline (12-07) and the submission deadline
   (12-14), and test the full inference path well in advance.
