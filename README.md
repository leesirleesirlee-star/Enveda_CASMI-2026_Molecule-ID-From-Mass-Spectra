# CASMI 2026 — Molecular Structure Identification from MS/MS Spectra

Kaggle *CASMI 2026 (Enveda)*: given a molecule's tandem mass spectra (1–16 spectra per
molecule, timsTOF), rank 25 candidate structures. The metric is **MRR@25** over
tautomer-canonical InChIKey14 (first block).

This repository reproduces the strongest public pipeline — **`imranarif536/casmi26-v44-pairtail-locked-top1`**
([source](https://www.kaggle.com/code/imranarif536/casmi26-v44-pairtail-locked-top1)) — at **0.417**,
documents *why the public design space is saturated at exactly that score*, and builds the measurement
infrastructure needed to go beyond it. Attribution for this and every other public notebook we build
on is in **[Lineage and attribution](#lineage-and-attribution)** below and in
[docs/EXTERNAL_RESOURCES.md](docs/EXTERNAL_RESOURCES.md) §E.

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
| **Anchor on our own environment** | **V45 `ctl`** — **0.417**, ref `56854098`, through this repository's build→verify→push chain |

### ✅ Reproduction confirmed on our own environment (2026-10-06)

**V45 `ctl`** — kernel `nicholasnicklee/casmi26-v45-ctl`, submission ref **`56854098`** — scored
**0.417**, the same as the reference to three decimals.

Its *code* is byte-identical to the V44 baseline (`self-check: 0 cell(s) differ from the control`);
what "V45" names is the run we built, pushed and scored ourselves, which is what makes it an anchor
rather than a re-report of someone else's number. Two runs, one score:

| | V44 baseline (ref `56839982`) | **V45 `ctl`** (ref `56854098`) |
|---|---|---|
| code | V44 `notebook.ipynb` | V44 `notebook.ipynb`, 0 cells differ |
| built and pushed by | author's public kernel | **this repository's chain** |
| docker image | default for that run | **pinned to the reference's** |
| score | 0.417 | **0.417** |

That is what makes every later number interpretable. Without an anchor, a variant scoring 0.415 has
two readings — "the idea does not help" or "our whole environment runs 0.002 low" — and nothing
distinguishes them.

### What the anchor also settled: ICEBERG/GLACIER are worth ≈0 here

Two submissions, identical code, differing in exactly one respect:

| ref | ICE / GL | score |
|---|---|---:|
| `56839982` | **dead** — the kernel had no image pin, so the bundled cp312 RDKit wheel was refused on Python 3.13 and both runners aborted **silently** | 0.417 |
| `56854098` | **working** — 71 molecules scored, 365 rows reordered, GL ok on 366 | 0.417 |

Forward models off vs on, same score. So on this task the two forward models contribute less than
the metric can resolve — which also means raising their compute budget is not a route to a better
score, and that the silent failure, while real and worth fixing, never cost us points. It did
invalidate *diagnostics*, and finding it is what the fidelity check in `scripts/` exists for.

**The headline finding:** the public design space saturates at **0.417**. We scanned all
**250 public notebooks — none advertises ≥ 0.418**. And two implementations that differ
substantially (one with CLAW promotion + a popularity patch + an ICE budget of 5400, one
with none of those and an ICE budget of 300) **land on the same score**. Crossing that
line therefore requires something the public implementations do not do.

---

## How this project was built

**This is an AI-assisted project, and we would rather say so than have you infer it.** A large
share of the work here was done by an AI agent operating inside this repository, with a human
setting direction and making the calls.

Roughly: the **human** chose what to pursue and what to abandon, supplied the domain reading of the
task and the competition's constraints, wrote the original design inputs
([`prd_patch.md`](docs/prd_patch.md), [`inspiration_1.md`](docs/inspiration_1.md)), and reviewed
every conclusion. The **agent** read the public notebooks and the engine's source, wrote and
reorganised the scripts, designed the experiments and pre-registered their predictions, ran the
tooling, and drafted the documentation.

Two consequences you will notice, both deliberate:

- **The docs record mistakes, not just results.** [`CHANGELOG.md`](CHANGELOG.md) carries a table of
  ten conclusions that were *refuted*, and [`PLAYBOOK.md`](docs/PLAYBOOK.md) keeps corrections
  inline where the wrong claim was made. Some of those errors were the agent's
  (three silent failures in a row, a mis-computed resolution, a metric arithmetic slip); keeping
  them was a choice, because a record that only shows what worked cannot be audited.
- **Claims carry their evidence.** Every measured number names the script that produced it, and
  where a conclusion rests on an inference rather than a measurement, it says so.

The agent's own configuration and rule files are intentionally **not** part of this repository.

What we would ask you to check independently: the two headline claims — that the visible
`test.parquet` is a decoy, and that 0.417 decomposes into recall × ranking. Both are reproduced by
scripts in [`scripts/`](scripts/), and both can be re-derived from the public data without taking
our word for it.

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
├── requirements.txt              ← local tooling deps (the Kaggle side is separate)
├── docs/
│   ├── PLAYBOOK.md                    ← the runbook: metric anatomy, budget, routes, evidence
│   ├── EXPERIMENTS.md                 ← the experiment ledger (submission refs and LB scores)
│   ├── LEADERBOARD.md                 ← all submissions plus external reference points
│   ├── EXTERNAL_RESOURCES.md          ← compliance registry: data / software / models / licences
│   ├── competition_brief.md           ← the task, the metric, the rules, the timeline
│   ├── research_metadata_errors_and_isobar_floor.md
│   │                                  ← a standalone research report (two measurements)
│   ├── inspiration_1.md               ← notes on two public solutions, with later corrections
│   └── prd_patch.md                   ← the natural-product dark-space design patch
├── scripts/                      ← builders, checkers, the queue, reports, the backtest
├── notebooks/
│   ├── v44_base/                 ← the vendored baseline everything derives from
│   ├── v45/                      ← _claw_patch.py + the variant currently in flight
│   └── backtest/                 ← the leak-free backtest kernel and its calibration configs
└── src/                          ← our own earlier pipeline (retrieval + analog propagation)
```

**Variant notebooks are generated, not stored.** Everything under `notebooks/v45/<name>/` is built
from `notebooks/v44_base/notebook.ipynb` by `scripts/build_variant_kernel.py --variant <name>`, so
only the one currently running is committed. Regenerating any variant takes seconds and the
builder asserts that every replacement hits exactly once.

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

## Lineage and attribution

**We did not write the V44 pipeline, and we do not claim it.** It is
**`imranarif536/casmi26-v44-pairtail-locked-top1`**
([Kaggle source](https://www.kaggle.com/code/imranarif536/casmi26-v44-pairtail-locked-top1)) —
*"don't let a new ranker destroy a proven top-1; use LambdaRank to improve the tail."*

One naming confusion is worth clearing up, because it is credit that would otherwise be misassigned.
The kernel we pulled the code from sits at `xiaoyuzhoux120/casmi26-v44-pairtail-locked-top1`, which is
**one of our own team accounts** and carries the same slug — that is what a Kaggle *fork* looks like.
So:

| | Who | What it is |
|---|---|---|
| **V44** | `imranarif536` | the original notebook; **author of the pipeline** |
| V44 fork | `xiaoyuzhoux120` (our team) | the copy we pulled from; submission `56839982` is **our score of their code** |
| **V45** | this repository | our build → verify → push chain, image pinned; `V45 ctl` reproduced 0.417 on our own environment |

**What is ours in V45:** the pinned-image environment repair, the fidelity check that found
ICEBERG/GLACIER silently failing, the CLAW gate ported verbatim from v17, the `lib_max` gate
analysis, the `pc_adaptive` slot policy, and the verification tooling in `scripts/`.

**Everything else we build on** — 12 further public notebooks, 7 external models, 7 libraries, each
with its licence and purpose — is itemised in
[docs/EXTERNAL_RESOURCES.md](docs/EXTERNAL_RESOURCES.md), which exists so that a winning solution can
disclose its provenance as the competition rules require. If you are one of those authors and want a
correction or a removal, open an issue and we will act on it.

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

## Licence

The **code** here is MIT-licensed — see [LICENSE](LICENSE).

**Data**: the competition data and anything derived from it stay under the competition's terms
(**CC BY-NC 4.0**, non-commercial), and every bundled or referenced third-party model and library
keeps its own licence, itemised in [docs/EXTERNAL_RESOURCES.md](docs/EXTERNAL_RESOURCES.md).

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
| **Reproduced on our own environment** | ✅ **V45 `ctl`, 0.417 (ref `56854098`)** — build→verify→push chain validated |
| ICE/GL contribution measured | ✅ **≈0** — two submissions, forward models off vs on, same score |
| Public frontier scan (250 notebooks) | ✅ none ≥ 0.418 |
| Recall / ranking decomposition | ✅ recall ≈ 0.545 |
| Three routes closed by measurement | ✅ pool expansion (+0.051%), adduct expansion, hardcoded answers |
| Leak-free backtest | 🔧 built, calibration pending |
| Candidate experiments (variants) | 20 built, **only ~6/week can be tested** |

**The next step is not another variant — it is measurement capability.** The backtest lets
several policies be compared **within a single run** (paired, ±0.002) instead of one
hypothesis per 5 GPU-hours.

Full evidence and reasoning: [docs/PLAYBOOK.md](docs/PLAYBOOK.md) and [CHANGELOG.md](CHANGELOG.md).
