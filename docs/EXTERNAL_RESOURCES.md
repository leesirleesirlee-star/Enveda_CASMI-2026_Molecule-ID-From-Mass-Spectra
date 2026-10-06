# External resource compliance registry

Competition rules require a winning solution to **disclose every external dataset and model
used**, with confirmed rights of use. Every resource is registered here **at the moment it is
introduced**; licence fields are taken from the Kaggle API (`datasets/view`), not relayed
from prose.

**Two hard constraints that shape licence choices**

1. Competition data is **CC BY-NC 4.0** (non-commercial), and a winning solution must be
   open-sourced.
2. Kaggle Notebooks are graded **offline**, so every dependency and weight must be
   pre-packaged as a Kaggle Dataset.

---

## A. Official competition data

| Resource | Source | Licence | Use |
|---|---|---|---|
| `train.parquet` (2,539,608 spectra / 275,810 structures / 3,033,286,496 bytes) | Kaggle competition page | CC BY-NC 4.0 | Training, retrieval library, candidate-pool base |
| `test.parquet` (1,213 spectra / 400 molecules / 4,848,729 bytes) | Kaggle competition page | CC BY-NC 4.0 | **Format and pipeline checks only** |
| `sample_submission.csv` (400 rows) | Kaggle competition page | CC BY-NC 4.0 | Submission format |

> ⚠️ **The visible `test.parquet` is a decoy** (settled — see [PLAYBOOK.md](PLAYBOOK.md) §0.7):
> all 1,213 of its spectra are verbatim rows of `train`, and the hidden evaluation set uses
> different `molecule_id`s. **Never tune on it**; any "truth" recovered from it is unrelated
> to the leaderboard.

## B. Software dependencies

| Resource | Version | Licence | Use | Offline packaging |
|---|---|---|---|---|
| RDKit | 2026.03.3 (grader-pinned; 2026.03.6 locally) | BSD-3-Clause | Canonicalisation / InChIKey / fingerprints / fragments | **Required** (the image lacks rdkit — via `metric/rdkit-2026-3-3-wheel`) |
| PyTorch | image 2.10.0+cu128 | BSD-3-Clause | FPNet / ICEBERG / GLACIER inference | Included in the image |
| LightGBM | image version | MIT | v4b ranker (4 boosters) + engine ranker | Included in the image |
| numba | image version | BSD-2-Clause | Entropy similarity / shift-search kernels | Included in the image |
| NumPy / pandas / PyArrow / scikit-learn | image versions | BSD-3 / Apache-2.0 | Foundations | Included in the image |

## C. External models and weights — **actually used by the current main line**

| Resource | Licence (API-verified) | Size | Role in the pipeline |
|---|---|---|---|
| `ahmedberatozer/casmi26-v4b-models` | Other (in description) | 1,682.6 MB | **Main engine**: v1 engine code + `fe_v4` feature families + FPNet A/B + **DreamsFP D** + LightGBM ranker (160 features) + `MANIFEST.json` (SHA256 for 64 files) |
| `ahmedberatozer/casmi26-v3-models` | Other (in description) | 411.9 MB | FPNet A+B ensemble for the PubChem-only channel |
| `ahmedberatozer/casmi26-fpnet-full1` | Other (in description) | 198.8 MB | Engine FPNet replacement (**trained on all train folds**, including np-examples) |
| `ahmedberatozer/casmi26-iceberg` | Other (in description) | 66.9 MB | **ICEBERG 2.1** (ms-pred, MassSpecGym `msg_all` checkpoint) forward fragment reranking |
| `ahmedberatozer/casmi26-glacier` | Other (in description), upstream **MIT** | 61.1 MB | **GLACIER** (ms-pred, MassSpecGym checkpoint) forward fragment reranking |
| `prvsiyan/casmi26-fp-models-v2` | **CC0** | 288.2 MB | Fingerprint models for the second engine (two-ranker) |
| `metric/rdkit-2026-3-3-wheel` | **Unknown** ⚠️ | 148.9 MB | Offline install of RDKit 2026.3.3 (matching the grader) |

**Licence risks (must be disclosed)**

- The `Source note` of v4b/v3 states explicitly that it *contains data/models derived from
  the Enveda CASMI26 competition training data (which includes CC BY-NC components) and from
  COCONUT (CC BY 4.0)*. Its dataset licence is registered as "Other", and v4b ships a
  `LICENSE-NOTICE.txt` — **a winning disclosure should itemise these and attach that notice**.
- `metric/rdkit-2026-3-3-wheel` has licence field **Unknown**. RDKit itself is BSD-3-Clause,
  so redistributing the wheel should be compliant, but **the dataset does not declare it** —
  this needs to be stated proactively.
- GLACIER's upstream is marked MIT; ICEBERG declares no upstream licence and needs checking
  against the ms-pred repository.

## D. External structure / spectral libraries — **actually used by the current main line**

| Resource | Licence (API-verified) | Size | Use |
|---|---|---|---|
| `ahmedberatozer/casmi26-v2-pool` | Other (in description) | 1,684.6 MB | **Candidate pool**: ~710k–775k structures (train ∪ COCONUT ∪ ChEBI/LIPID MAPS) |
| `ahmedberatozer/casmi26-pubchem-tier` | Other (in description) | **7,214.1 MB** | PubChem structure tier (CID-SMILES + CID-Mass, public NCBI data; stereo stripped; organic CHNOPS+halogens; 150–1250 Da) |
| `prvsiyan/coconut-casmi26-candidates` | **CC BY 4.0** | 423.3 MB | COCONUT 2.0 candidates + fingerprints |
| `prvsiyan/chebi-lipidmaps-casmi26` | **CC BY-NC-SA 4.0** ⚠️ | 60.4 MB | ChEBI + LIPID MAPS candidates (**carries a ShareAlike obligation**) |
| `megayak/casmi26-simulated-ranker-rows` | **CC0** | 366.0 MB | Ranker training rows (simulated) |
| `prvsiyan/casmi26-ranker-features` | **CC0** | 9.6 MB | Ranker training features |
| `dmitriigluzdov/casmi26-pubchem-popularity-prior` | query timed out — **to be filled in** | 2,754.0 MB | PubChem SID/PMID popularity arrays (both pool and pc sets) |

**Also held locally (not in the main line; used only for coverage measurement)**

| Resource | Source | Licence | Result |
|---|---|---|---|
| COCONUT (aidensong123 mirror) | Kaggle | as upstream COCONUT | 480,118 structures |
| COCONUT 2.0 (prvsiyan mirror) | Kaggle | CC BY 4.0 | 436,389 structures |
| LOTUS | Kaggle (`casmi26-lotus-pool`) | as upstream LOTUS | 150,590 structures |
| NPAtlas | npatlas.org (TSV) | as upstream | 33,497 structures |

> 🔴 **Route closed by measurement**: merging these four libraries into the candidate pool
> adds only **373 structures (0.051%)**, and the public pool is already larger than our merged
> one. **Do not invest further here.** (See [PLAYBOOK.md](PLAYBOOK.md) §3.6, item 2.)

## E. Public notebook lineage (code provenance and attribution) — must be disclosed with a winning solution

Every part of the current main line originates from public notebooks. Authors (following
`nazarmohammed`'s citation chain plus our own tracing):

| Author | Contribution |
|---|---|
| `prvsiyan` | Analog Propagation baseline (library / analog / MetFrag-lite / FPNet) |
| `haideptry` | 4-channel retrieval + neural fingerprint ranker |
| `megayak` | Two Rankers One Engine; simulated ranker rows; the **negative** result on DreaMS |
| `llccqq624` (Jiachen Li) | Next Direct W088; the `w_pv=0.88` sweep and a review of three fatal traps |
| `dmitriigluzdov` | Retrieval + Class-3 generation; 160-feature / 4-booster LightGBM; the isolated PubChem channel; ICEBERG + GLACIER |
| `ahmedberatozer` | The v1→v4q engines and all model/pool assets; `fpnet_full1` |
| `seyitkaangunes` | Dual-engine fusion + BIO (ChEBI/LIPID MAPS) + AFIX; union + double forward rerank |
| `bobthebot369` | Popularity prior (μ=0.15); the public version of the **CLAW Promotion Rule** |
| `matterhorn3838` | GLACIER `[M+H]+` filtering (+0.004) |
| `lehau007` | V26/V27/V28 ensemble releases (V27 = the public best 0.417) |
| `nazarmohammed` | Curated three-engine fusion + forward cascade (the direct base of our V44) |
| **`imranarif536`** | **Author of `casmi26-v44-pairtail-locked-top1` — the notebook whose *code* our V45 line reproduces.** Source: https://www.kaggle.com/code/imranarif536/casmi26-v44-pairtail-locked-top1 |
| `nursrijan`, `ozertuu`, `analyticaobscura` | Variants and ablation records |

**On the V44/V45 naming, because it matters for attribution.** The notebook we reproduce is
`imranarif536`'s; the kernel we pulled it from (`xiaoyuzhoux120/casmi26-v44-pairtail-locked-top1`) sits
on one of our own team accounts and carries the same slug, which is what a Kaggle fork looks like. So
credit for the V44 pipeline belongs to `imranarif536`, and submission `56839982` is *our score of
their code*, not our design.

The distinction we do claim: the **V45** kernels in `notebooks/v45/` are built, verified and pushed by
this repository, with the docker image pinned to the reference's, and `V45 ctl` is the run that
reproduced 0.417 on our own environment. Our substantive additions to the pipeline are the CLAW gate
ported verbatim from v17, the `lib_max` gate analysis, the `pc_adaptive` slot policy, and the
fidelity/verification tooling under `scripts/`.

**Our own changes** (relative to that direct base): porting the CLAW gate and `promote()`
(copied verbatim from v17 rather than re-implemented), the `lib_max` gate analysis, and all
ablation variants under `notebooks/v45/`.

## F. TODO

- [ ] Fill in the licence field for `dmitriigluzdov/casmi26-pubchem-popularity-prior` (the
      first query timed out).
- [ ] Verify ICEBERG's upstream (ms-pred) licence; GLACIER is already marked MIT.
- [ ] Determine why `metric/rdkit-2026-3-3-wheel` is registered as Unknown and explain it in
      the disclosure.
- [ ] Assess whether the **ShareAlike** term of `chebi-lipidmaps-casmi26` requires our derived
      pool to be released under CC BY-NC-SA as well.
- [ ] Winning-disclosure template: itemise sections A–E and attach v4b's `LICENSE-NOTICE.txt`.

---

## Changelog

- 2026-10-03: registry created; official data and the local dependency stack registered;
  recorded that DreaMS/SimMS are not in the main line.
- 2026-10-05: **rewritten around the actual dependencies of the V44 reproduction line.**
  Section C changed from "the main line uses no external models" to listing the 7 models and
  weights actually used; section D registers 7 external libraries with API-verified licences;
  section E added (public notebook lineage and attribution, required for a winning
  disclosure); 3 licence risks and 5 TODOs flagged.
