# Changelog

Every substantive iteration, in chronological order: what changed, why, and what came out
of it — **including conclusions that were later refuted**, which are often more useful than
the successes.

- Each entry corresponds to one or a group of commits (hashes included; `git show <hash>`)
- LB noise is roughly ±0.006, but **re-scoring the same submitted file is deterministic**,
  so deltas *within* one experiment are trustworthy
- Reading rule: `|Δ| ≤ 0.001` is noise; `0.001–0.003` needs replication; `>0.003` may be
  believed but should still be replicated before it drives a decision

---

## 2026-10-03 · Infrastructure, and one decisive discovery (P0/P1)

| Date | Commit | What |
|---|---|---|
| 10-03 | `7811daa` | Project skeleton, coding discipline, agent rules, global plan |
| 10-03 | `d8f9dd7` | Chemistry/scoring core (tautomer-canonical InChIKey14), three-fold validation protocol, mass-window retrieval, parallel downloader |
| 10-03 | `562854f` | Direct spectral matching + **analog Δm shift propagation** (our own core channel) |
| 10-03 | `a001792` | Recorded a data-acquisition incident and its final verification |
| 10-03 | `844529a` | Three-fold validation sets built on real data; retrieval baseline runs end to end |
| 10-03 | `1a41173` | Performance: lazy folding + binned vectorised scoring (**11× faster**); fixed the shift-cosine denominator |
| 10-03 | `2192544` | **Decisive finding: all 1,213 spectra of the visible `test.parquet` are verbatim rows of `train.parquet` (`enveda-180`)** ⇒ it is a decoy, and local fold scores are systematically inflated |
| 10-03 | `6753574` | Self-contained Kaggle submission notebook + first valid `submission.csv` |
| 10-03 | `28dc05d` `11e45d3` | Kaggle write-permission blocker (KGAT token 403), traced to a BOM issue |
| 10-03 | `b13e2c3` | **Fix: the strict fold never actually excluded the answer** — `score_molecule` bypassed `fold.library` |
| 10-03 | `6921c3a` | Offline notebook runs for real on Kaggle |
| 10-03 | `f607d20` | Diagnosis: the analog channel's quality window is too narrow (top improvement item) |

**Lessons from this phase**

1. **The validation set itself has to be validated.** We used the visible test as our
   validation set for several steps before discovering it is a decoy.
2. **A fold that looks like it works may never have excluded anything.** That leak was silent.

---

## 2026-10-04 · Candidate pool expansion

| Date | Commit | What |
|---|---|---|
| 10-04 | `c0b2fbd` | Built a natural-product candidate pool of **730,161 structures (2.6×)**, materially better coverage |
| 10-04 | `f3f91f9` | Architectural diagnosis: **100% of truths share the query's molecular formula; 89% of candidates have the wrong formula** |

**Outcome:** pool expansion worked (coverage improved) — but see 10-05: **it has already hit
its ceiling**.

---

## 2026-10-05 · The V45 ablation line (the main body of work)

### (1) Pivot to reproducing a proven strong baseline

**`a14f737`** — built the V44 ablation framework, pinned the leaderboard standing, and
**corrected** the earlier "visible test = hidden test" conclusion.

**Why pivot:** our own pipeline was stuck at 0.176 while the public V44 baseline sat at
0.417. **Reproduction before invention** — get to 0.417, then talk about beating it.

**Result:** exact V44 reproduction = **0.417** (submission ref `56839982`), matching the
author's reported score. An independent reproduction (gengsr V3) scored 0.413, matching its
author's claim **digit for digit** ⇒ **reproduction here is deterministic**.

### (2) CLAW port — the only lever in this domain that can touch rank 1

| Commit | What |
|---|---|
| `8141982` | Ported the CLAW promotion rule (from v17/v27): when `S>6 ∧ pop≥5`, promote the PubChem candidate to rank 1 |
| `9c01cc6` | **Differential test against the reference implementation: 3,000 random cases, 0 mismatches** |
| `bc6b986` | **Corrected our own "tail ceiling is 0.007" arithmetic** (the true tail ceiling is 0.5) |
| `8905c58` | **Ruled out three scenarios and concluded the champion lock never fires** |

**Why it is worth porting:** the merge stage is the **only** place in the whole pipeline that
can change rank 1 (every other channel is gated by `lib_max < 0.9`). And crossing 0.418
needs only **one molecule moving to rank 1 ≈ +0.0025**.

### (3) Automating the infrastructure

| Commit | What |
|---|---|
| `b815fa4` | **Variant queue**: wait for a slot → push → wait → submit → report, end to end; plus a submission byte-size survey |
| `cbcc1fa` | Session-duration cap on pushes, so weekly GPU quota is not over-reserved |
| `6cf7f67` | **Automatic batch-log harvesting with diagnostic lines highlighted** |
| `74ce346` | **Measured the real bottleneck: a 30 h/week GPU quota ≈ 6 variants per week** (not the daily submission limit) |

### (4) Routes closed by measurement (negative results count)

| Commit | Route closed | Evidence |
|---|---|---|
| `e699c34` | **Pool expansion** | LOTUS ∪ NPAtlas adds only **373 structures = 0.051%** over the existing pool |
| `e699c34` | **Adduct-hypothesis expansion** | formula–mass agreement is **1209/1213 = 99.7%**, already near perfect |
| `0b771eb` | **`lib_max` gate is closed on hidden data** | read the similarity kernel from a public v4q notebook: range `[0,1]`, empty spectrum → 0 ⇒ the gate is **open** on hidden data |
| `bdacedc` | **"Copy a public implementation" as a path forward** | scanned all **250 public notebooks: none advertises ≥ 0.418**; the public frontier is **exactly 0.417** |

### (5) Decomposing 0.417 — the most important positive result of this phase

| Commit | What |
|---|---|
| `4857d82` | **First decomposition of 0.417 into recall × ranking: recall ≈ 0.545** |
| `8c90131` | Separated three causes of missing recall: A (TOPN truncation — largely ruled out) / B (answer is in the PubChem tier) / C (answer is in no public library) |
| `7dcdbe8` | Quantified the modelling business case: the public ranker was trained on only **819 query groups** |

**How it was computed:** grouped CV on the public ranker's training rows (142,762 rows ×
31 features, 819 groups, recall forced to 1.0) gives **MRR@25 = 0.765, top-1 = 0.689**.
With recall at its maximum, ranking only reaches 0.765; the real score is 0.417 ⇒
**recall@25 ≈ 0.545**.

**Why this matters most:** it showed for the first time that **the ranking half is already
heavily optimised by the public models while the recall half (~0.19 of the score) has never
been an explicit target** — which re-prioritised everything that followed.

### (6) Three real "fails without erroring" bugs of the same family

| Commit | Bug | Consequence |
|---|---|---|
| `40c5939` | `CASMI_POP_DIR` anchored on `MANIFEST.json`, while the reference anchors on `pc_lsid.npy` | wrong path ⇒ worker pool died ⇒ **the entire PubChem channel silently switched off** ⇒ CLAW looked inert |
| `3dfd2ed` | The row-order alignment between the popularity arrays and the tier is an **external contract** nobody checked | misalignment ⇒ the gate quantity is garbage ⇒ results cannot be interpreted |
| `81956e8` | `except Exception: pass` is the notebook's **only fully silent failure path** | prior failed to load ⇒ the variant equals the control, invisible in the score |

**Shared lesson:** in experiments whose expected result may legitimately be zero,
**"the result cannot be interpreted" is more dangerous than "the result is bad"** — it makes
us retire a correct route on the strength of broken evidence.

### (7) Pre-registration, and a flaw found in the pre-registration itself

**`9c70831`** — wrote down each variant's prediction and **what would count as falsification**.

**`70db174`** — self-audit found the pre-registration was flawed: the plan was to use the
`promoted` counter to distinguish "the gate was too strict to fire" from "the mechanism does
not work". But **that branch never executes on public data** (`lib_max ≈ 1.0` there), so
`promoted` **is necessarily 0** — the diagnostic has no discriminating power.

**`b34aeb7` `fd99906`** — remedy: use the **submission byte count** as a free discriminator
(`totalBytes` ≈ 530,716 ⇒ the branch never executed; clearly different ⇒ it did), with its two
preconditions and reading band (530–533 KB is indistinguishable) written down.

> **Lesson: a diagnostic must itself be validated.** A counter that is always zero is worse
> than no counter, because it makes the result look interpretable.

### (8) Reading the source, turning inference into fact

| Commit | What |
|---|---|
| `fede8db` | Confirmed `lib_max`'s definition **from the engine source** (best library-spectrum similarity among candidates in the window); obtained the **160-feature list** |
| `37b56aa` | Read the training recipe (`ranker.py` / `traindata.py`): **both FPNet and the ranker are trained and evaluated only under "the truth is inside the ±10 ppm window"** |
| `1c0d4ab` | Hence the modelling line's first project: **unify the "pool universe" and the "tier universe"** |

**Correction:** I had said the generative channel is "the weakest part of the public field" —
**that was wrong**. The engine not only generates, it carries 6 dedicated features
(`is_gen`, `gen_parent_sim`, `gen_steps`, …).

### (9) Freezing variant creation, switching to strict serial

**`903d03a`** — 20 variants were built while only **~6 per week** can be tested. The marginal
value of building more is near zero, and it is speculative work. Switched to: **advance one
at a time, and let the previous result decide the next step.** The cost (half a GPU slot
idle) is accepted deliberately.

---

## 2026-10-05 · Method A: the leak-free backtest

**Motivation:** the real bottleneck is not a shortage of ideas but that we can measure
**6 things a week, each as a three-decimal scalar**. Any method that needs a threshold sweep
is impossible at that throughput.

| Commit | What |
|---|---|
| `d059dd8` | Built **structure-disjoint, source-stratified** folds (275,810 structures / 5 folds) with three asserted invariants |
| `5110155` | **Feasibility confirmed: the engine ships leak-hiding hooks** (`exclude` / `exclude_sid` / `drop_pid`) ⇒ **no pool or library rebuild needed** |
| `726528c` | Component 1: backtest query set (**cast column-by-column to `test.parquet`'s reference schema**, not hand-written names) |
| `d1e711e` | Component 2: verified the three hooks' **index spaces** and caching behaviour |
| `a88ff82` | Component 3: the backtest kernel (self-contained; patches the **class**, so it does not depend on `V1FE` forwarding kwargs) |
| `83e748e` | Component 4: calibration design and **pre-registered falsification conditions** |

**Four self-corrections in this phase** (all before spending GPU time):

| Commit | What I had said | What is true |
|---|---|---|
| `5ed4178` | resolution ±0.004 | I had mistaken the **per-fold** s.e. for the resolution; pooled + paired it is ±0.0006–0.0018 |
| `b99c371` `0d05d97` | Tier 1 uses the whole 9,500-structure fold | **cannot finish**: ~18 s/molecule puts the 9 h ceiling near 1,800 |
| `842b502` | the query count can simply be raised | it **silently starves ICE/GL**: their budgets are fixed wall-clock caps that do not scale |
| `1ba4d52` | the fold definition is trustworthy | there were **two same-sized but different "fold 0"s** (`sort=False` vs `sort=True`) |

**Three real bugs caught by reviewing untested code** (none of which would have errored):

| Commit | Bug | Symptom |
|---|---|---|
| `c4455a6` | truth supplied as InChIKey14 instead of SMILES | score **systematically underestimated** |
| `1ba4d52` | inconsistent fold definition | cross-experiment comparisons **silently misaligned** |
| `92269a2` | scorer ran **before** cell 32 | number did not match the artefact actually graded |

**`580ca24`** — audited the synthetic `COMP`'s coverage (including the easily missed
`sample_submission.csv`), and recorded honestly that **my own checker first raised a false
alarm** (it ignored `ast.Tuple` targets).

---

## 2026-10-05 · Published

The repository went public at
[github.com/leesirleesirlee-star/Enveda_CASMI-2026_Molecule-ID-From-Mass-Spectra](https://github.com/leesirleesirlee-star/Enveda_CASMI-2026_Molecule-ID-From-Mass-Spectra).

Preparing it meant pruning and correcting more than publishing: 142 tracked files became 68, the
documentation was translated into English (including reordering the runbook, whose sections had been
numbered in the order they were written rather than the order they are read), 36 hardcoded absolute
paths were replaced with a root derived from each script's own location, the AI-agent configuration
was kept out of the repository while the AI assistance itself was disclosed in the README, and an
MIT licence was added — which the repository had lacked despite a compliance registry that itemises
everyone else's licences.

Three checks were added that had not existed: all relative links resolve; no leaked tokens, user
paths or machine-specific paths appear in any tracked file; and the tracked tree runs from a
different directory, so a reader who clones it is not relying on files only we have.

---

## Refuted conclusions (kept for the record)

| Original conclusion | What refuted it | Commit |
|---|---|---|
| visible test = hidden test | byte counts 530,716 vs 393,908; and 0.417 ≠ 0.98 | `a14f737` |
| tail ceiling is 0.007 | arithmetic error; the correct ceiling is **0.5** | `bc6b986` |
| the champion lock does fire | all three scenarios ruled out by contradiction | `8905c58` |
| the pool can still be expanded | LOTUS ∪ NPAtlas adds only **0.051%** | `e699c34` |
| the ICE budget marks the plateau | refuted; `icefull` prior lowered accordingly | `c8e8bb9` |
| the `lib_max` gate is closed on hidden data | excluded by the similarity kernel's range and empty-spectrum behaviour | `0b771eb` |
| the generative channel is the field's weakest link | the engine has it, with 6 features | `fede8db` |
| CLAW is worth +0.024–0.043 | v27 (with CLAW) and V44 (without) both score 0.417 | `c8e8bb9` |
| the backtest resolves ±0.001 | after four corrections it is **±0.002**, and faithfulness and power cannot both be had | `842b502` |
| the V_A_hard fold is "impossible" | the engine ships the simulation hooks; the correct method was in the source all along | `5110155` |

---

## Open items

| Item | Status |
|---|---|
| Backtest calibration (component 5) | ⏳ waiting on GPU (~2 h per run) |
| `claw` variant's LB score and byte-count discriminator | ⏳ waiting on GPU |
| Unify the pool and tier universes (modelling line) | 📋 project defined, weeks of work, **gated on a cheap experiment first showing the tier contains real answers** |
