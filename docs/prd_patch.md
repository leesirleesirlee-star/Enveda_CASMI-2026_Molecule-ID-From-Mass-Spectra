# PRD v2.1 patch: targeted correction toward natural-product dark chemical space

**Patch version**: v1.0
**Date**: 2026-10-04
**Applies to**: CASMI 2026 local lightweight edition v2.1
**Triggered by**: Kaggle discussion 745029 (hengck23) and the surrounding thread
**Core correction**: the test set's real target may be **natural-product dark chemical space**,
not Enveda-180's drug-like molecules. The retrieval library, candidate pool, validation sets and
reranker training data must all be adjusted accordingly.

> **Historical input document.** This is a design patch written before we had measured anything.
> Margin notes below record where later measurement confirmed, qualified, or refuted each claim.
> The text itself is kept as written.

## 1. Key updates to our understanding

1. **"natural product structures" in the competition description is the central clue.** Enveda
   cares about **natural metabolites not present in existing databases** — "dark chemical space".
2. **The public `test.parquet` is an enveda-180 sample, byte-identical to the training set**, but
   enveda-180 is **drug-like screening chemistry**, not the natural products that decide the award.
   > **Confirmed later**, and it is the single most consequential fact in the project: the visible
   > test set is a decoy ([`PLAYBOOK.md`](PLAYBOOK.md) §2).
3. **Class 3 is in neither PubChem nor COCONUT**, implying the genuinely novel molecules are
   natural-product analogs or entirely new natural-product scaffolds.
   > **Confirmed and central**: it became cause C of our recall-gap decomposition
   > ([`PLAYBOOK.md`](PLAYBOOK.md) §10).
4. **The timsTOF instrument** (used to acquire the test set) is designed for complex natural-product
   analysis, further supporting this reading.
   > **Confirmed** against the data.
5. **Recommended natural-product databases**: COCONUT, LOTUS, NPAtlas.
   > **Partly closed by measurement**: LOTUS ∪ NPAtlas adds only **373 structures (0.051%)** over
   > train ∪ COCONUT, so this route has almost no headroom
   > ([`PLAYBOOK.md`](PLAYBOOK.md) §13 item 2).
6. **Recommended reference work**: MS2Mol (de-novo prediction of natural metabolites), EnvedaDark
   (a 226-compound dark-chemical-space natural-product benchmark).

## 2. Specific changes to each PRD section

### 2.1 Section 3, system architecture: tiered retrieval library

**Original PRD**: the retrieval library comes from training-set spectra with no source distinction.

**Patch**:

- Build the retrieval library **in tiers by source**:
  - **Layer A**: Enveda-180 synthetic-molecule spectra (the bulk of the current training set).
  - **Layer B**: public natural-product spectra (the natural-product subset of MassBank, GNPS, MoNA).
  - **Layer C**: natural-product structure databases (COCONUT, LOTUS, NPAtlas), indexed by formula,
    serving as a spectrum-less candidate source.
- Retrieve from each **separately**, keeping source labels so the reranker can distinguish them.
- Do **not** mix Layer A's synthetic molecules directly into the natural-product candidate pool —
  that causes "candidate dilution".
  > **Qualified later**: candidate dilution turned out to be a much smaller effect than feared —
  > the pool window holds only ~65 candidates per molecule, so `TOPN=60` barely cuts anything
  > ([`PLAYBOOK.md`](PLAYBOOK.md) §10, cause A).

### 2.2 Section 3.3, multi-channel evidence scoring: new natural-product features

**Patch**: add the following features to the evidence score:

| Feature | Description |
|---|---|
| **Source label** | whether the candidate comes from a synthetic library / natural-product spectral library / natural-product structure library |
| **Natural-product scaffold match** | whether the candidate contains common natural-product scaffolds (terpenoids, alkaloids, flavonoids, …) |
| **COCONUT/LOTUS/NPAtlas hit** | whether the candidate appears in those databases |
| **Dark-chemical-space tendency** | whether the candidate is absent from PubChem (the Class 3 signal) |

### 2.3 Section 3.4, learned reranking: adjusting negative-sample sources

**Original PRD**: negatives are other candidates sharing the molecular formula.

**Patch**:

- Negatives must **include natural-product analogs**, not only synthetic molecules.
- Build the training set stratified by source, so the reranker learns to distinguish
  "natural product vs synthetic molecule".
- When validating, **report MRR@25 stratified by source**, not only the overall score.

### 2.4 Section 4, validation strategy: add natural-product-stratified validation

**Patch**: on top of the existing scaffold-disjoint validation, add:

- **Source-stratified validation**: split the validation set into a "synthetic subset" and a
  "natural-product subset", and report MRR@25 for each.
- **Dark-chemical-space validation set**: following EnvedaDark (226 natural products), build an
  internal Class 3 validation set to test the generative module.
- **DreaMS probe experiment version 4**: add COCONUT / LOTUS / NPAtlas to the reference datasets
  and compare coverage against MassSpecGym, Enveda-180 and non-Enveda-180.
  > **Superseded**: this probe line was replaced by the leak-free backtest
  > ([`PLAYBOOK.md`](PLAYBOOK.md) §20).

### 2.5 Section 5, implementation phases: add natural-product data preparation

**New in phase 0**:

- [ ] Download COCONUT, LOTUS, NPAtlas structures and index them by formula.
- [ ] Separate natural-product-source spectra (MassBank, GNPS, MoNA) out of the training set.
- [ ] Run DreaMS probe experiment versions 1–4 to quantify each reference dataset's coverage of the
      test set.
- [ ] Study the MS2Mol architecture and the EnvedaDark benchmark.

**New in phase 2**:

- [ ] Tiered retrieval library construction: Layer A / B / C.
- [ ] Add source labels and natural-product features to the reranker.
- [ ] Evaluate reranking stratified by source.

**New in phase 3**:

- [ ] Aim the generative module at natural-product dark chemical space first, following MS2Mol's
      de-novo approach.
- [ ] Use EnvedaDark as the internal Class 3 validation set.
- [ ] Apply confidence gating: allow generated candidates into high ranks only when the retrieval
      candidates are insufficient.
  > **Partly confirmed later**: the public V44 has no generative module; the engine does have a
  > Class-3 generation channel with 6 dedicated features
  > ([`PLAYBOOK.md`](PLAYBOOK.md) §17).

### 2.6 Section 6, technology stack: add natural-product tooling

**Patch**: add to the stack:

| Tool | Purpose |
|---|---|
| **COCONUT / LOTUS / NPAtlas** | natural-product structure databases, supplementing the candidate pool |
| **MS2Mol** | reference architecture for de-novo prediction of natural metabolites |
| **EnvedaDark** | dark-chemical-space natural-product benchmark, for internal validation |
| **msbuddy / MIST** | molecular-formula prediction, assisting natural-product candidate filtering |

## 3. New experiment: DreaMS probe version 4

On top of the three originally planned versions, add:

**Version 4**: reference datasets = COCONUT + LOTUS + NPAtlas (natural-product structure libraries)

- If version 4's coverage is **clearly higher than version 2 (Enveda-180)**, the test set is
  confirmed to favour natural-product dark chemical space.
- If version 4's coverage is comparable to version 2's, the test set may mix synthetic molecules
  and natural products, requiring stratified handling.
- If version 4's coverage is low, the hypothesis needs re-evaluation — the test set may still be
  predominantly Enveda-180.

**Output**: coverage curves for each version at similarity thresholds 0.9 / 0.85 / 0.8 / 0.7, plus
the nearest-neighbour source distribution of the test spectra.

## 4. Updated instructions for the agent

1. **Run DreaMS probe version 4 immediately**, compare against versions 1–3, and produce a coverage
   report.
2. **Download COCONUT, LOTUS, NPAtlas**, index by formula, and build the Layer C candidate pool.
3. **Separate natural-product-source spectra from the training set** to build Layer B.
4. **Add source stratification to the reranker's training data**, with negatives including
   natural-product analogs.
5. **Report MRR@25 stratified by source in validation**, focusing on the natural-product subset.
6. **Aim the generative module at dark chemical space first**, following MS2Mol and using
   EnvedaDark as the internal Class 3 validation.
7. **Apply confidence gating**: generated candidates enter high ranks only when retrieval
   candidates are insufficient.
8. **Do not retrieve Enveda-180 synthetic molecules and natural-product candidates together**, to
   avoid candidate dilution.

---

**Patch summary**: this patch reframes the PRD's goal from "generic small-molecule retrieval" to
"**natural-product dark-chemical-space-targeted retrieval + generation**". The tiered retrieval
library, reranking features, validation strategy and generative module all need corresponding
adjustment. The agent should prioritise DreaMS probe version 4 and natural-product database
preparation before moving on to main-engine optimisation.
