# CASMI 2026 global plan (v1, for review)

**Date**: 2026-10-03
**Status**: awaiting confirmation (work not yet started)
**Basis**: the competition brief, PRD v2.1, plus this round's direct checks of the competition API,
the data, and the local machine

> **Historical planning document.** Written before any code was run. Where later measurement
> corrected a claim here, the correction is noted in the margin; the text is otherwise kept as
> written.

## 0. One-sentence conclusion

**The direction is right, but two foundations of the PRD are broken — the validation method and the
main-engine choice — and the foundations must be fixed before building on them.**

Three pieces of good news: the competition is real and still running (**72.7 days left**); the
highest publicly reproducible artifact has already been written down, and copying it reaches
**0.33–0.34**; and the local GPU is entirely adequate (measured), so the real constraint is Kaggle's
9 hours and compute quota, not this laptop.

## 1. Verified facts (differences from the PRD marked ⚠️)

| Item | Verified result | Source |
|---|---|---|
| Competition type | **Code Competition (notebooks only)**; a static CSV submission necessarily scores 0 | Kaggle API `isKernelsSubmissionsOnly: true` |
| Daily submissions | **5** (not the usual 10) | Kaggle API `maxDailySubmissions: 5` |
| Deadlines | final **2026-12-14 23:59 UTC**; team merge **2026-12-07** | Kaggle API |
| Current size | **2,242 teams** (only 498 on 9/17 — growing fast) | Kaggle API `teamCount` |
| Metric | MRR@25 over InChIKey's first 14 characters, RDKit **pinned at 2026.03.3** tautomer canonicalisation → **stereochemistry and tautomers are ignored** | official metric notebook |
| Training data | **2,539,608 rows × 18 columns**, 2.82 GB, 21 row groups | read the remote parquet metadata directly |
| Test data | the locally visible `test.parquet` = **1,213 spectra / 400 molecules**, 12 columns | read directly |
| Sources | 11 libraries in `ingest_lib`: enveda-180 (~1.15M spectra), pluskal_ms2, riken, gnps, massbank, mona, spectraverse, msdial, drug_plus, masaryk, **enveda-np-examples** | row-group 21 statistics |
| ⚠️ **The visible test set is a placeholder** | at grading time Kaggle substitutes a **hidden test set** (different molecule IDs). **Tuning on the visible test is meaningless** | several competitors' notes plus 0.000 counterexamples |
| ⚠️ No RDKit in the Kaggle image | **no rdkit/matchms/faiss**, and grading is **offline**. Every dependency must be packaged as a Kaggle Dataset in advance | the Kaggle docker-python repo |
| ⚠️ P100 retired | retired 2026-09-15; only **T4×2** remains, GPU quota about 30 h/week | Kaggle announcement |
| Public scores | public SOTA **0.409–0.425** (9/25–10/2); mainstream reproducible band **0.33–0.34**; pure retrieval 0.143 | several competitors' repos |

### Where the leaderboard sits (for setting a target)

```
0.44  ── leader band (0.440 appeared on 9/29)
0.409 ─ current public SOTA (seyitkaangunes, four-channel fusion + popularity prior)
0.336 ─ beraterolelk "Analog Ranker" notebook
0.328 ─ reproduction of the public four-channel analog baseline
0.30  ─ the PRD's "final target"   <- already far exceeded by public solutions
0.25  ─ the PRD's "minimum acceptable"
0.176 ─ the PRD's recorded "existing retrieval baseline" (I can find no source for this ⚠️)
0.143 ─ a measured pure-retrieval baseline
0.000 ─ naive approaches (indexing only Enveda's two libraries / submitting a CSV)
```

**Conclusion: the PRD's 0.30 target is below the publicly reproducible level. A sensible target is:
stabilise at 0.32–0.34 first, then push for 0.36+.**

> **Measured later**: the public frontier turned out to be **exactly 0.417**, saturated — no public
> notebook advertises more ([`PLAYBOOK.md`](PLAYBOOK.md) §11). We reached 0.417.

## 2. The PRD's three hard defects (the most important output of this review)

### 2.1 The validation strategy is broken — the real cause of 0.873 local vs 0.14 on Kaggle

PRD chapter 7 requires **fully disjoint Murcko scaffolds plus dynamically removing the correct answer
from the retrieval library at evaluation time**.

That design mathematically kills retrieval: if the answer is not in the library, retrieval can never
recall it and only generation can. But the public fact is that **a substantial share of test
molecules have their reference spectra in train** (strongest form: "many test molecules share
identical or near-identical experimental spectra in `train.parquet`"). The 0.145 → 0.335 jump came
precisely from adduct-mass-shift library matching.

So the pipeline was trapped in a contradiction:
- validation (answers absent from the library) → the odd 0.873/0.14 gulf
- the real leaderboard (answers mostly present) → retrieval is the main engine, but the validation
  set punishes retrieval

**The corrected (three-fold) validation protocol**:

| Fold | Construction | Purpose | Maps to |
|---|---|---|---|
| **V-A instrument calibration** | `ingest_lib == 'enveda-np-examples'` (**confirmed: the 21st row group, 1,184 spectra / 250 natural products**, same instrument as the test set) | main calibration metric, directly comparable to the public LB | Class-1/2 bulk |
| **V-B same-source** | random hold-out from the whole library, **answers kept** | measures the pure-retrieval ceiling | the leaked part |
| **V-C strict** | identity/scaffold-disjoint, **answers removed** | measures true extrapolation | the Class-3 tail |

Also emit a **recall-ceiling curve (oracle recall@K)** to tell whether the bottleneck is recall or
ranking. **Every improvement must be reported on both V-A and V-C**; reading only V-A repeats the
"overfitting to leakage" mistake.

> **Measured later — this fold design was right in spirit but V-A turned out trivial**: all 250
> V-A structures also appear in other libraries, so its answers are always retrievable and it scores
> MRR **1.0000**. It had to be demoted to a sanity check, and V-A_hard (answers genuinely removed)
> was added ([`EXPERIMENTS.md`](EXPERIMENTS.md)). Later still, the whole fold set was rebuilt as
> structure-disjoint, source-stratified folds ([`PLAYBOOK.md`](PLAYBOOK.md) §20).

### 2.2 The main-engine choice was wrong — DreaMS + SimMS is not the winning path

PRD chapter 3 treats DreaMS (1024-d embeddings) + SimMS (GPU similarity) as the core engine. After
checking:

| PRD assumption | Reality |
|---|---|
| DreaMS checkpoint ~1.2 GB | actually **two checkpoints, about 2.6 GB** (embedding 1.24 GB + ssl 1.39 GB) |
| batch-infer 2.5M spectra on the local GPU | nobody has published throughput; extremely heavy dependencies (~20 packages incl. pyopenms); **not feasible inside Kaggle's free quota** |
| SimMS installable via `pip install simms` | **the PyPI `simms` is an unrelated radio-interferometry package**. Real SimMS needs `pip install git+…`, **impossible with grading offline** |
| Build a FAISS index | the Kaggle image **has no faiss** |

**And none of the public 0.33–0.34 solutions uses DreaMS.** They use:

> **mass-window recall → four-channel evidence → linear fusion → top-25 output**

The four channels:
1. **direct spectral matching** (modified cosine / spectral entropy),
2. **analog propagation** (the key innovation: **not** editing structures, but **shifting the matched
   library peaks by Δm** and propagating the annotation to candidates via fingerprint similarity —
   `AnalogScore(c)=max_a[Sim_mod(q,a)²·Tanimoto(fp_c,fp_a)]`),
3. **virtual fragments** (in-silico fragmentation; a pure-RDKit implementation suffices),
4. **neural fingerprints** (FPNet) — optional, a bonus.

**And fusion must be linear — do not put a GBDT reranker on top**: in the public data LightGBM
LambdaMART scores 0.7517 in-domain and **collapses to 0.2301 out-of-domain**. That is exactly the
trap in PRD §3.4's "XGBoost/LightGBM reranking".

> **Measured later — this became project rule #1, but with an important qualification.** The rule was
> written for *our own* pipeline. The winning public line does use a LightGBM LambdaRank reranker
> (160 features × 4 boosters) and scores 0.417, because it was trained group-wise on the right
> objective. The durable form of the lesson is "a learned model must be validated on a
> domain-shifted fold before it may drive decisions", not "never use GBDTs".

### 2.3 "9 hours" and local compute: right direction, wrong constraint

- The 9-hour limit is **only corroborated by competitors' notes** — the official Rules page is not
  readable (Kaggle renders it with JS; I tried three sources). Plan for 9 hours but **design for 6**.
- Local RTX 5060: **measured working** (below). The PRD is right that 8 GB is limiting for large
  models, but the conclusion should be "**no large models; retrieval plus lightweight scoring**",
  not "downgrade the generative model".

## 3. Measured local environment (working; can proceed)

| Item | Measured | Assessment |
|---|---|---|
| GPU | RTX 5060 Laptop, sm_120, 26 SM, 7.93 GB VRAM | ✅ usable |
| GPU throughput | fp32+TF32 about **14–15 TFLOPS** | ⚠️ low, power-capped |
| Power limit | currently 60 W (default 50 W, **cap 85 W**), full-load SM 2145 MHz / max 3090 MHz | ⚠️ **unlockable to 85 W on mains power, about +40%** |
| Driver / CUDA | 616.92 / CUDA 13.4; PyTorch 2.15.0+cu130 | ✅ matched |
| CPU / RAM / disk | 16 logical cores / ~32 GB / C: 238 GB free, D: **529 GB free** | ✅ ample |
| Existing conda env | `Digital_Resin` (py3.10.21): torch ✅, **rdkit 2026.03.6 ✅** (newer than the grader's 2026.03.3), numpy/pandas/sklearn ✅; matchms/xgboost/lightgbm/faiss ❌ | half usable; needs packages |
| Network | GitHub / HuggingFace / PyPI **all reachable**; Kaggle site reachable, API needs auth | ✅ data obtainable |

### Two important environment findings

1. **The sandbox blocks the GPU**: under the restricted mode `torch.cuda.is_available()` returns
   `False`; unrestricted it works. **All GPU commands need elevated execution.** (Confirmed not a
   hardware fault.)
2. **Data is obtainable without a Kaggle account**: HuggingFace has full mirrors of the competition
   data, verified:
   - `pradeepss007/Casmi26`: `train.parquet` **2.82 GB**, `test.parquet` 4.6 MB, `sample_submission.csv`
   - `lonelyforever/enveda.casmi`: the same three files (backup mirror)
   - another mirror (`cuonguyenphu/...`) contains other solutions' intermediates (`fp_model_v2.pt`,
     `library.pkl`, `ranker_*.parquet`)
   - `test.parquet` + `sample_submission.csv` + a complete public solution's code (70 KB) were
     actually downloaded to `.deepworks/tmp/`

   **Recommendation: still register a Kaggle account** (submission requires one), but development
   need not be blocked on it.

## 4. Proposed architecture (replacing PRD chapter 3)

```
                    ┌─────────────── offline (local, one-off) ───────────────┐
 train.parquet ──▶  filter "test-like" spectra (timsTOF + enveda-np-examples …)
 (2.82 GB/2.54M rows)  clean peaks -> compact library.pkl
                    build the candidate structure catalogue (train + COCONUT 2.0 + ChEBI/LIPID MAPS)
                    precompute: fingerprint matrix / mass index / shift index
                    └──────────────────────┬────────────────────────────────┘
                                           ▼  package and upload as a Kaggle Dataset (offline-usable)
                    ┌─────────────── online (Kaggle notebook, <6 h) ────────┐
 test.parquet ──▶   molecule-level aggregation (several spectra, several collision energies)
                    ① mass-window recall (±ppm + ¹³C shift, >99.9% precursor recall)
                    ② four-channel evidence:
                       direct match │ analog shift propagation │ virtual fragments │ neural fingerprint
                    ③ linear fusion + per-query z-score standardisation
                    ④ 25 slots: spread scaffold diversity; do not waste them on stereoisomers
                    ──▶ /kaggle/working/submission.csv
                    └──────────────────────────────────────────────────────┘
```

**Key design decisions (where they differ from the PRD)**:
- ❌ no DreaMS / SimMS as the main engine (kept as a later optional experiment, not main line)
- ❌ no GBDT reranker on top (unless V-C shows it does not collapse)
- ✅ use matchms' `ModifiedCosineGreedy` / `NeutralLossesCosine` (Apache-2.0, on PyPI)
- ✅ in-silico fragments implemented in RDKit (the PRD's correction about MetFrag is **right and
  important**; keep it)
- ✅ use the 25 slots for **scaffold diversity** — the metric ignores stereo/tautomers, so piling up
  stereoisomers is pure waste

## 5. Phased plan (10.4 weeks, aligned to the 12-14 deadline)

| Phase | Week | Deliverable | Acceptance |
|---|---|---|---|
| **P0 foundations** | 1 | three-fold validation (V-A/V-B/V-C) + oracle recall curve; environment completed (matchms/xgboost/lightgbm/pyarrow); data on disk | **first measure "the public solution's real baseline on this machine"** — a trustworthy baseline number |
| **P1 retrieval backbone** | 2–3 | mass-window recall + direct spectral matching + molecule-level aggregation; end-to-end run producing `submission.csv` | V-A MRR@25 ≥ 0.25; submission format 100% valid (first non-zero LB score) |
| **P2 analog channel** | 3–5 | Δm peak shifting + fingerprint-gated propagation | V-A ≥ **0.32–0.34**; **Kaggle ≥ 0.30** |
| **P3 fragments + fingerprints** | 6–8 | virtual-fragment channel; FPNet fingerprint channel (optional); linear-fusion tuning | V-A ≥ 0.36; Kaggle > 0.33 |
| **P4 push and consolidate** | 9–10.4 | end-to-end notebook < 6 h; technical report; external-data compliance registry | stable submissions; Kaggle 0.36+ |

**Weekly routine**: report both V-A and V-C, plus one Kaggle submission (5/day is ample).

> **Measured later**: the binding budget is not 5 submissions/day but a **30 h/week GPU quota ≈ 6
> variants per week** ([`PLAYBOOK.md`](PLAYBOOK.md) §3) — which restructured the whole experiment
> plan.

## 6. Risk register (new/corrected)

| Risk | Impact | Mitigation |
|---|---|---|
| **The visible test is a placeholder, misused for tuning** | severe misdirection, weeks wasted | trust only V-A/V-C; use test for format checks only |
| **Offline + no RDKit makes the grading rerun fail silently** | a 0-score submission | complete an offline packaging rehearsal before P1 ends (wheels + models all in the Dataset) |
| GPU power capped at 60 W | local precomputation ~40% slower | mains power + unlock to 85 W (needs your confirmation) |
| Domain shift collapsing the reranker (0.75 → 0.23) | a collapsing LB score | insist on V-C; prefer linear fusion |
| Kaggle GPU quota 30 h/week | large-model plans are infeasible | the main line does not depend on the GPU; the GPU is for local precomputation only |
| Licences of COCONUT/ChEBI and other external structure libraries | a winning solution must be open and compliant | register every external resource's licence now (the PRD already has the list; keep it) |
| `enveda-np-examples` has only 250 molecules | the main calibration fold has high variance | use it with V-B/V-C; note LB noise is about ±0.006 |

## 7. Three decisions needed

1. **Validation protocol**: accept "three-fold validation, retiring the PRD's pure
   scaffold-disjoint design"? (This is the most important one.)
2. **Main engine**: accept "retrieval + four-channel linear fusion" as the main line, demoting
   DreaMS/SimMS to an optional post-P3 experiment?
3. **Data acquisition**: start from the public HuggingFace mirror (I can download 2.82 GB
   immediately), or do you provide a Kaggle API token so we go through the official channel?
   (Submission needs an account either way.)

Two smaller ones to confirm too:
4. **Ponytail**: identified as [DietrichGebert/ponytail](https://github.com/DietrichGebert/ponytail)
   (MIT, about 54% less code / 20% fewer tokens). But note — **its purpose is "don't over-engineer
   while writing code", whereas this project's bottleneck is ML experiment design, not code volume**.
   I suggest installing it (harmless, saves tokens) but **not expecting it to address the project's
   main risk**. OpenCode is also installed here (`.opencode/` + `opencode.jsonc`); ponytail has an
   official OpenCode adapter, and DSH can use it as a skill. Shall I install it, and where?
5. **GPU power unlock**: raise the RTX 5060 from 60 W to 85 W? (Reversible; needs mains power.)

## 8. Decision record (confirmed 2026-10-03)

| # | Decision | Your choice | Status |
|---|---|---|---|
| 1 | Validation protocol | **switch to three-fold** (V-A/V-B/V-C), retire the PRD's pure scaffold-disjoint design | to implement |
| 2 | Main engine | **retrieval + four-channel linear fusion** as the main line; DreaMS/SimMS demoted to optional | planned accordingly |
| 3 | Data source | **official Kaggle API** (KGAT token provided) | ✅ working, downloading |
| 4 | Ponytail | **install** (OpenCode plugin + DSH skill) | ✅ done |
| 5 | GPU power | **do not unlock yet**; revisit when it becomes the bottleneck | recorded |

### Incidents found and handled during execution

- **⚠️ A Kaggle token nearly entered the repository**: `Kaggle_Token.txt` (37 bytes, containing a real
  `KGAT_` token) was staged by `git add` at the repo root.
  Handled: removed from the index → archived to `.secrets/kaggle/` → the root copy deleted →
  `.gitignore` fallback rules extended → a full-repo scan confirmed no residue.
  **Recommendation: revoke that token in Kaggle settings after the competition** (it appeared in
  plain text in conversation).
- **The local shell was completely unusable at one point**: the workspace directory's Windows
  permissions lacked the current user's WRITE_OWNER, so every command failed with
  `SetNamedSecurityInfoW failed`. Fixed (backup and rollback scripts in a recovery directory
  beside the workspace).
- **The sandbox blocks the GPU and some writes**: `torch.cuda.is_available()` is `False` in the
  restricted mode and `pip` unpacking fails. GPU commands, and writes into `.opencode/`, need
  elevated execution.

## Appendix: what this review actually did

- Repaired the workspace directory's Windows file permissions (the missing right was breaking
  every shell command); backup and rollback scripts in a recovery directory beside the workspace
- Read only the tail of the remote parquet over HTTP Range to obtain the full 2,539,608 × 18 schema,
  **without downloading the 2.82 GB file**
- Used row-group column statistics to confirm `enveda-np-examples` sits in the 21st row group
  (1,184 rows) → precisely locating the main calibration fold
- Measured GPU availability and throughput (blocked inside the sandbox, fine outside)
- Downloaded and inspected a public solution's code (`.deepworks/tmp/solution.py`, a 3,318-line
  LightGBM LambdaRank pipeline)
