# Notes on two high-scoring solutions

> **Historical input document.** These are working notes written while studying two public
> solutions, before we had measured anything ourselves. They are kept as written — several
> interpretations here were later corrected by measurement (see
> [`PLAYBOOK.md`](PLAYBOOK.md) and [`EXPERIMENTS.md`](EXPERIMENTS.md)). Where a claim below
> turned out wrong, that is noted in the margin rather than edited into the text.

The core idea of both high-scoring solutions points at the same insight: **breaking through the
0.3 score barrier requires moving from "pure retrieval" to a hybrid "retrieval + controlled
generation" architecture**, with extremely fine control over **the ranking precision of the
candidate pool**.

---

## 🏔️ Solution 1: CASMI26 Fusion GLACIER MH (LB 0.413)

This solution (gengsr) scores a public **0.413**, the highest known in the discussion forum at the
time. Its input datasets outline the architecture:

| Input asset | Inferred purpose |
|---|---|
| CASMI26 fingerprint models | spectrum → molecular fingerprint prediction |
| CASMI26 PubChem Popularity Prior | a "popularity" prior over candidates (structures common in PubChem score higher) |
| CASMI26 ranker training features | reranker training features (public) |
| CASMI26 Simulated Ranker Rows | simulated reranker training data |

**What "Fusion" means**: this is a **multi-channel evidence fusion** ranking system. It likely
fuses:
1. **Predicted-fingerprint similarity** (predict a fingerprint from the spectrum, compare against
   the candidate's true fingerprint)
2. **Spectral retrieval similarity** (modified cosine / neutral losses)
3. **A PubChem popularity prior** (a weak prior: the more common a structure is in PubChem, the
   more likely it is the answer)
4. **Simulated rerank rows** (augmenting reranker training with simulated data)

**"GLACIER MH"** most likely refers to some variant or ensemble of **GLACIER** (a known mass-spectra
retrieval/reranking tool). Overall this is a **heavily engineered fusion ranking system**, not a
single model.

**Transferable points**:
- Reranker training data can be augmented by **simulation** rather than relying entirely on real
  training-set spectra. That addresses the shortage of spectrum–structure pairs.
- Introducing a **PubChem popularity prior** as a weak feature is a simple, effective trick — given
  equal evidence, more common structures are more likely to be the test set's choice.
- "Fusion" means **weighted fusion of multi-channel evidence**, with weights possibly learned by
  the reranker itself.

## 🔒 Solution 2: CASMI26 V44 Pairtail Locked Top1

The notebook could not be opened directly, but **"Pairtail Locked"** and **"Top1"** in the title
hint at the core strategy.

**"Pairtail"** most likely refers to **the tail of candidate pairs** — i.e. lower-ranked
candidates. **"Locked"** may mean some kind of **locking mechanism** applied to tail candidates.

Combined with discussion-forum threads on **"Confidence-Gated Analog Generation"** and
**"Protected Bio-DB Tail"**, the logic is likely:

> **Apply a "protective lock" to the top-N candidates from the retrieval engine, apply generative
> augmentation or analog substitution only to tail candidates, and allow generated candidates into
> the leading ranks only at high confidence.**

This matches the **confidence gating** strategy discussed earlier: **the shape of the MRR score
means that a wrong generated candidate displacing a correct one costs dearly. So the generative
module must be confined to "tail slots", and only strong evidence may let it "move up".**

**"V44"** probably means the 44th iteration, implying heavy experimental tuning.

> **Later correction.** The actual V44 contains no generative module at all. "Pairtail" refers to a
> PairTail LambdaRank model (0.15 weight, tail-only) inside the second engine, and "Locked Top1" to
> a hard-coded top-1 lock that we later proved **never fires** on the hidden set
> ([`PLAYBOOK.md`](PLAYBOOK.md) §15). The confidence-gating intuition here was right in spirit but
> attributed to the wrong mechanism.

## 🧬 The shared underlying logic

Despite different implementation details, both solutions share these strategies:

### Retrieval first, generation second (but generation tightly controlled)

The strongest public solutions' main engine is **still retrieval + reranking**, but the generative
module is **strictly confined to tail slots** and needs **confidence gating** to reach a high rank.
This agrees with the direction of PRD v2.1 ("retrieval first, generation second") but executes it
far more carefully.

### Reranker feature engineering is where the points come from

Both solutions invest heavily in the reranker:
- **Multi-channel evidence** (predicted fingerprints, spectral similarity, fragment explanation,
  popularity prior)
- **Simulated training data** (augmenting reranker training with simulated spectra)
- **Tiered candidate pools** (candidates from different sources carry different prior weights)

### A candidate pool's "quality" matters more than its "quantity"

The **"candidate dilution"** problem repeatedly raised in the forum is taken seriously in both.
Neither simply dumps every possible structure into the pool; both **manage and rank in tiers**.

### Targeted optimisation for natural-product dark chemical space

Together with hengck23's earlier post, both solutions most likely **tiered their candidate pools
specifically for natural products**, rather than concentrating all resources on Enveda-180's
synthetic molecules.

## 📚 Specific knowledge to acquire

| Area | Content | Link to these solutions |
|---|---|---|
| **Multi-channel evidence fusion** | how to design, weight and train a reranker over many weak features | the core of Fusion GLACIER |
| **Confidence gating** | using model output probability to decide whether generated candidates may rank highly | the core of Pairtail Locked |
| **Simulated data augmentation** | training a reranker on simulated spectra / simulated candidate pairs | "Simulated Ranker Rows" |
| **Popularity priors** | compound frequency in databases such as PubChem as a feature | "PubChem Popularity Prior" |
| **Tiered candidate pools** | managing candidates in tiers by source (synthetic / natural product / dark chemical space) | implied by both |
| **Reranker feature engineering** | building features such as fingerprint similarity, fragment coverage, neutral-loss match counts | the rerankers' inputs in both |

## 🎯 Direct implications for our project

1. **Reranker training data can be augmented by simulation**: no need to rely entirely on real
   training-set spectra. In-silico fragmentation of known structures can generate large numbers of
   training samples.

2. **Introduce a PubChem popularity prior**: a low-cost, plausibly effective weak feature. Given
   equal evidence, more common structures are more likely to be the test set's choice.

3. **Apply a "protective lock" to tail candidates**: do not let generated candidates freely enter
   the leading ranks. Allow them into tail slots only when the retrieval candidates are clearly
   insufficient (a high-confidence signal).

4. **Manage the candidate pool in tiers**: do not rank candidates from all sources together. Tier
   by source (Enveda-180, natural-product spectral libraries, natural-product structure libraries),
   rank within tiers, then fuse.

5. **Pursue the "simulated rerank rows" idea**: pretrain the reranker on simulated
   spectrum–candidate pairs, then fine-tune on real data. This may be the key to the shortage of
   paired data.

> **Later correction.** Points 2, 3 and 5 are the ones that survived contact with measurement.
> Point 1 became the basis of our leak-free backtest, and point 4 was closed by measurement: pool
> expansion adds only 0.051% ([`PLAYBOOK.md`](PLAYBOOK.md) §13).
