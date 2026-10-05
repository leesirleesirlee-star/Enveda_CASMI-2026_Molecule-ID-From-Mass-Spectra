# CASMI 2026 — Molecular Structure Identification from MS/MS Spectra

Kaggle *CASMI 2026 (Enveda)*: given a molecule's tandem mass spectra (1–16 spectra per
molecule, timsTOF), rank 25 candidate structures. The metric is **MRR@25** over
tautomer-canonical InChIKey14 (first block).

This repository reproduces the strongest public pipeline
(`xiaoyuzhoux120/casmi26-v44-pairtail-locked-top1`) at **0.417**, documents *why the public
design space is saturated at exactly that score*, and builds the measurement
infrastructure needed to go beyond it.

📓 **[CHANGELOG.md](CHANGELOG.md)** — the full iteration log (what changed, why, and what
was refuted). 📚 **[docs/](docs/)** — detailed write-ups.

---

## Results at a glance

| Item | Value |
|---|---|
| Task | CASMI 2026 (Enveda) — spectra → structure, 25 candidates per molecule |
| Metric | **MRR@25** (tautomer-canonical InChIKey14 first block) |
| **Our best** | **0.417** — public LB rank 111 of ≥800 teams |
| Rank 59 (the real bar) | 0.418 |
| Rank 1 | 0.471 |
| Best public notebook | **0.417** (i.e. where we already are) |
| Baseline we reproduce | `xiaoyuzhoux120/casmi26-v44-pairtail-locked-top1`, submission ref `56839982` |

**The headline finding:** the public design space saturates at **0.417**. We scanned all
**250 public notebooks — none advertises ≥ 0.418**. And two implementations that differ
substantially (one with CLAW promotion + a popularity patch + an ICE budget of 5400, one
with none of those and an ICE budget of 300) **land on the same score**. Crossing that
line therefore requires something the public implementations do not do.

---

## Four things you need to know before reading the code

1. **The visible `test.parquet` is a decoy, not the evaluation set.**
   All 1,213 of its spectra are **verbatim rows of `train.parquet`** (the `enveda-180`
   library). Any "improvement" tuned on it is meaningless — it is a placeholder that the
   grader replaces with a hidden set. Corroboration: the graded submission is 530,716 bytes
   versus 393,908 bytes for the public run's output, and 0.417 ≠ 0.98.

2. **There is no trustworthy local fold out of the box.**
   The public pack does not contain the authors' `split.parquet` (`build.py` only builds the
   spectrum cache), so their hold-out protocol cannot be reproduced. This repository builds
   its own structure-disjoint, source-stratified folds (see *Backtest* below).

3. **0.417 decomposes into recall × ranking.**
   Grouped CV on the public ranker's training rows: with recall forced to 1.0, ranking only
   reaches **0.765**. Back-solving the real 0.417 gives **recall@25 ≈ 0.545** — i.e. **about
   45% of hidden molecules never get their truth into the top 25**. The ranking half is
   heavily optimised by the public models; **the recall half has never been an explicit
   target**.

4. **Public implementations differ less than they look.**
   Line-by-line, v17 / v27 / V44 share the same engines, dual ranker, ICEBERG, GLACIER and
   RRF fusion; the differences concentrate in a handful of knobs
   ([docs/PLAYBOOK.md](docs/PLAYBOOK.md) §3.7).

---

## The main pipeline (V44)

```
                    ┌── fingerprint model FPNet (A+B) ──┐
MS/MS spectra ──►   ├── dual-ranker engine (160 feats)  ├──► RRF fusion ──► 25 slots ──► submission
 (1–16 per mol)     ├── ICEBERG / GLACIER forward score ┤        ▲
                    └── PubChem tier (~100M structures) ┘   fixed tail slots / CLAW promotion
        ▲
   candidate pool (~775k structures, ±10 ppm)
```

- **Candidate pool**: ~775k structures (train ∪ COCONUT ∪ ChEBI/LIPIDMAPS), ±10 ppm window
- **Ranking**: 160 features × 4 LightGBM LambdaRank boosters, 600 rounds
- **Forward models**: ICEBERG / GLACIER rerank only within a **fixed wall-clock budget**
  (note: this budget does *not* scale with molecule count — see the backtest section)
- **A second candidate universe**: the PubChem tier (~100M structures, 7.2 GB) is allowed
  **only 5 tail slots** and **has never had a scorer trained for it** — the structural
  source of the recall gap

---

## Repository layout

```
.
├── README.md                     ← you are here
├── CHANGELOG.md                  ← iteration log: what changed, why, what was refuted
├── docs/
│   ├── PLAYBOOK.md               ← main runbook: metric anatomy, routes, pre-registration
│   ├── EXPERIMENTS.md            ← experiment ledger (submission refs and LB scores)
│   ├── LEADERBOARD.md            ← all submissions + external reference points
│   ├── EXTERNAL_RESOURCES.md     ← compliance registry: data / software / models / licences
│   └── research_*.md             ← focused research reports
├── scripts/                      ← variant builder, checkers, queue, reports, backtest
├── notebooks/
│   ├── v44_base/                 ← the vendored baseline everything derives from
│   └── backtest/                 ← leak-free backtest kernel + calibration configs
├── experiments/variants/         ← generated variant notebooks (see its README index)
└── src/                          ← our own earlier pipeline (retrieval + analog propagation)
```

> `data/`, `outputs/`, `artifacts/` and `.deepworks/` are **not** in git (large files and
> scratch). `.secrets/` is ignored too and **must never be committed**.

---

## Reproducing a run

```bash
# 1) environment
conda create -n casmi2026 python=3.11 && conda activate casmi2026
pip install -r requirements.txt

# 2) build a variant kernel (CLAW shown)
python scripts/build_variant_kernel.py --variant claw \
    --slug casmi26-v45-claw --title "CASMI26 V45 Claw"

# 3) verify it: compiles, CLAW reassembly, slot semantics, 3000-case diff vs the
#    reference implementation, and kernel metadata
python scripts/check_variant_kernels.py

# 4) push -> wait -> submit -> report, fully automated
python scripts/queue_variant.py --folder notebooks/v45/claw \
    --desc "V45 claw" --timeout-h 12
```

> Step 4 needs Kaggle credentials at `.secrets/kaggle/access_token` (never committed).
> **The real bottleneck is the ~30 h/week GPU quota ≈ 6 variants per week** — not the daily
> submission limit.

---

## Measurement: a leak-free backtest

Rather than spending 5 GPU-hours per hypothesis, we built a backtest that hides a held-out
structure from all three channels that could leak it. The engine turned out to support this
natively:

```python
Engine.run(spectra, target,
           exclude=...,       # bool mask over LIBRARY SPECTRA (len == L.sid)
           exclude_sid=...,   # structure id, hides its representative from the analog channel
           drop_pid=...)      # removes the pool entry (class-3 simulation)
```

Two constraints were established by measurement, not assumption:

| Constraint | Consequence |
|---|---|
| ~18 s per molecule end-to-end | the 9 h ceiling allows ~1,800 queries, **not** the whole 9,477-structure fold |
| `ICE_BUDGET` / `GL_BUDGET` are **fixed** wall-clock caps | raising the query count silently *starves* ICE/GL ⇒ faithfulness and statistical power cannot both be had |

Hence two tiers:

| Tier | Config | Queries | Paired s.e. | Use |
|---|---|---|---|---|
| **Tier 1** | ICE/GL off | 2,000 | **±0.0021** | policy screening (off by construction, so no starvation) |
| **Tier 2** | ICE/GL on | 400 | ±0.0047 | only for policies that depend on the forward models |

Calibration configs (`bt`, `bt_top1`, `bt_no_ice`) and their **falsification conditions** are
pre-registered in the runbook.

---

## Method and discipline

These rules are not ceremony — each one corresponds to a real mistake made here:

| Rule | What it prevents |
|---|---|
| Never validate on the visible test set | it is a decoy; tuning on it gives systematically inflated numbers |
| Report both calibrated and strict identity-disjoint folds | one number alone means nothing |
| `\|Δ\| ≤ 0.001` is noise; `0.001–0.003` needs replication; `>0.003` still deserves it | the target gap is +0.001–0.002, the same size as the noise |
| **Pre-register** predictions *and falsification conditions before results arrive* | prevents post-hoc rationalisation |
| **Verify that a diagnostic can actually discriminate** | a counter that is always zero is worse than no counter |
| **Verify the verifiers** (negative tests, re-check the yardstick) | checkers, diagnostics and reference numbers all lie quietly |
| Treat silent failures as more dangerous than crashes | all five real bugs found here were silent |

---

## Data and compliance

- Competition data: `enveda-CASMI26-molecule-id-mass-spectra` (Kaggle, competition rules apply)
- External models and libraries: 7 models + 7 libraries, each logged with licence and purpose
  → [docs/EXTERNAL_RESOURCES.md](docs/EXTERNAL_RESOURCES.md)
- Public notebook lineage: 12 authors, all attributed
- **Credentials never enter the repository**; `.gitignore` starts with `.secrets/`

---

## Status and next step

| Item | Status |
|---|---|
| V44 baseline reproduced | ✅ 0.417 (ref `56839982`) |
| Public frontier scan (250 notebooks) | ✅ none ≥ 0.418 |
| Recall / ranking decomposition | ✅ recall ≈ 0.545 |
| Three routes closed by measurement | ✅ pool expansion (+0.051%), adduct expansion, hardcoded answers |
| Leak-free backtest | 🔧 built, calibration pending |
| Candidate experiments (variants) | 20 built, **only ~6/week can be tested** |

**The next step is not another variant — it is measurement capability.** The backtest lets
several policies be compared **within a single run** (paired, ±0.002) instead of one
hypothesis per 5 GPU-hours.

Full evidence and reasoning: [docs/PLAYBOOK.md](docs/PLAYBOOK.md) and [CHANGELOG.md](CHANGELOG.md).
