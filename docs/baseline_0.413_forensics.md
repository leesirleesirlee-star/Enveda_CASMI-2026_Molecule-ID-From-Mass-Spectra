# CASMI26 Fusion + GLACIER + [M+H]+ (LB 0.413) — an implementable specification

**Subject**: Kaggle notebook `gengsr/casmi26-fusion-glacier-mh-lb-0-413`, by Gengsr, version 3,
last run 2026-10-01, T4 GPU, internet **off**.

**How the evidence was obtained (all first-hand code, no speculation)**: the unauthenticated
Kaggle kernel-pull API returned the complete notebook JSON (21 cells, 10 of them executable code):

- `https://www.kaggle.com/api/v1/kernels/pull/gengsr/casmi26-fusion-glacier-mh-lb-0-413`
  → **VERIFIED** (web_fetch / curl, no auth needed)
- Modules embedded in the notebook (written to `/kaggle/working/ours/`, no extra datasets):
  `fusion_core.py`, `frag_rescore.py`, `eng/casmi_engine.py`, `eng/eng_runner.py`, `eng/pv.py`,
  `eng/pv_fp.py`, `pc/pc_runner.py`, `pc/probe_core2.py` → **all obtained verbatim**
- The ICEBERG/GLACIER runners live in datasets rather than the notebook; the zips were downloaded
  and `gl_fuse.py` / `gl_runner.py` / `fuse.py` / `README.md` / `MANIFEST.json` /
  `LICENSE-NOTICE.txt` obtained verbatim → **VERIFIED**

Local copies: `.deepworks/tmp/notebook.json`, `.deepworks/tmp/nb/` (cells split out),
`.deepworks/tmp/embed/` (embedded modules), `.deepworks/tmp/glacier/`, `.deepworks/tmp/iceberg/`.

The notebook header states it is an executable reproduction of the author's scored submission
**56792011**, with the score verified on **2026-10-03**, and that it adapts public work from:
Huseyin Emre Aksoy (`casmi26-sub-v4b-0-409-on-the-public`) → Seyit Kaan Gunes
(`casmi26-v4n-engine-fusion-union-lb-0-399`) → Wang Penghua (`casmi26-v29-fusion-popglmh`), plus
data and models from Ahmed Berat Ozer / prvsiyan / megayak / bobthebot369 / Dmitrii Gluzdov.

> **Context from later work.** We reproduced this notebook's line at **0.413** (submission ref
> `56835199`, matching the author's claim digit for digit). Several items in §7 below that were
> unverifiable then have since been resolved — margin notes mark them.

## 0. Overall structure (10 code cells)

| cell | Role |
|---|---|
| 2 | `EMBED` dict: source strings for 8 embedded modules |
| 4 | the `CFG` knobs plus a **build-time override line** (see §4.0) |
| 6 | offline RDKit wheel install (2026.03.3), input discovery, MANIFEST sha256 checks, write `ours/` |
| 8 | SMOKE / VALIDATION / RERUN switches; `build_validation()` |
| 10 | **engine 2** subprocess → `ENG` |
| 12 | **PubChem-only channel** subprocess → `PC` |
| 14 | **main engine**: library + pool + FPNet bank + fe_v4 families + LightGBM ranker |
| 16 | all molecules → the `BASE` list (top 60) |
| 18 | **ICEBERG + GLACIER + FRAG scoring** (scores only, no reordering) |
| 20 | `fusion_core.build_submission` → `submission.csv` |

> Key design point: cell 18 only produces score dicts — **all ordering lives in `fusion_core`**;
> `DUMP_STAGES=True` dumps `base/pc/eng/ice/gl/frag/meta` to JSON, so fusion can be re-run and
> re-tuned offline on CPU without touching the GPU stage. That is a genuinely good engineering
> idea to copy.

## 1. Candidate generation

### 1.1 Main-engine candidate pool (engine 1)

- **Pool size 710,701 structures** (`ahmedberatozer/casmi26-v2-pool`, 1.68 GB). **VERIFIED**
  (dataset description; `dmitriigluzdov/casmi26-pubchem-popularity-prior`'s description states
  `casmi26-v2-pool (710,701 structures)`).
  Composition, per the notebook/dataset: **train structures ∪ COCONUT (CC BY 4.0)**; engine 2
  additionally unions **ChEBI + LIPID MAPS** (`prvsiyan/chebi-lipidmaps-casmi26`, 60 MB,
  CC BY-NC-SA 4.0).
- **Mass window**, `eng/pv.py` → `class CFG`:

```python
class CFG:
    PPM_WIN = 10.0
    PPM_FALLBACK = 30.0
    INT_FLOOR = 0.002
    MAX_PEAKS = 256
    MZ_TOL = 0.01
    ...
    ANALOG_WIN = 200.0
    N_ANALOG = 80
```

  Usage (`eng/casmi_engine.py::compute_channels`):

```python
cand = pool_window(target, CFG.PPM_WIN)
if len(cand) == 0: cand = pool_window(target, CFG.PPM_FALLBACK)
```

  `pool_window` binary-searches a mass-sorted `mass` array at `target*(1±ppm/1e6)`.
  → **a 10 ppm neutral-mass window, falling back to 30 ppm when empty**. **VERIFIED**
- **target = the median neutral mass over the molecule's spectra**, converted via a 30-entry adduct
  table (`pv.ADDUCTS`, including `[M+H]+ [M+NH4]+ [M+Na]+ [M+K]+ [M-H2O+H]+ [M-2H2O+H]+ [M+2H]2+ [M-H]- [M-H2O-H]- [M+CH2O2-H]- [M+C2H4O2-H]- [M+Cl]- [2M+H]+ [2M+Na]+ [2M-H]- [3M+H]+ …`).
- **No formula filter on this channel** — mass window only. The formula is used later, for
  same-formula grouping before reranking.
- One correction worth copying (`formula_mass`, a prvsiyan fix):

```python
def formula_mass(f):
    """ADDED: neutral monoisotopic mass by arithmetic on molecular_formula (prvsiyan fix; immune to
    riken's 0.005 Da precursor precision and gnps adduct mislabels). NaN if an element is untabulated."""
```

  → **when `molecular_formula` is available, use it for neutral mass in preference to the adduct
  conversion**; otherwise fall back to the adduct. Cell 16 also uses this to compensate for cases
  like "the ppm window drops 121,805 spectra". **VERIFIED (code)**

### 1.2 Ranking and truncation

- Main-engine LightGBM ranker (`casmi26-v4b-models`' `ranker`, multi-booster average):
  `rank_score(np.concatenate([X, F], 1), list(FEATURES) + list(names))`, where `X` = engine features
  and `F` = fe_v4 family features. **VERIFIED (call form, cell 16); the feature-name list lives in
  the dataset and was not obtained → UNVERIFIED** *(resolved later: `engine.py`'s `FEATURES` list
  was obtained and documented in [`PLAYBOOK.md`](PLAYBOOK.md) §17.)*
- Deduplicate by tautomer-canonical **InChIKey14**, take **`TOPN=60`**, and store
  `BASE[mid] = [smiles, keys, lib_max, ranker_scores, formulas, pool_ids]`, where `pid < 0` marks a
  **generated** candidate.
- `lib_max` = the molecule's best entropy similarity against the **experimental spectral library**
  (`train.parquet`, multi-source). `LIB_TAU = 0.9` → `lib_max ≥ 0.9` marks a **library hit, and the
  whole list is then left untouched** downstream.

### 1.3 PubChem-only channel (a big pool: 105,875,489 structures)

- Dataset `ahmedberatozer/casmi26-pubchem-tier`: **105,875,489 structures**, 7.21 GB, mass-sorted,
  `CID-SMILES + CID-Mass`, **stereo stripped, organic CHNOPS+halogens only, 150–1250 Da**.
  **VERIFIED** (dataset description)
- Two-pass scoring (`pc/probe_core2.py`, verbatim):

```python
PC_N1 = int(os.environ.get('CASMI_PC_N1', 5000))   # ours: overridable from CFG['PC_N1']
PPM = 10.0
K = 25
...
#   3. pass 1 (src/casmi/pubchem.py tier): partial f.z on the ECFP4 block of the selected bits, keep top PC_N1 = 1000.
#   4. pass 2: full selected-bit fingerprints, full f.z, sort descending.
#   5. walk down the f.z order, compute the metric score key (tautomer-canonical InChIKey14); keep a structure only if
#      its key is NOT in the candidate pool (train U COCONUT) and not already listed; stop at 25.
```

  → **no sampling inside the ±10 ppm window**; pass 1 scores approximately on the ECFP4 block and
  keeps the top `PC_N1` (CFG sets 5000); pass 2 scores exactly on the full selected bits and walks
  down the order, **excluding keys already in the pool**, emitting at most **25 PubChem-only
  structures** (`K=25`). `f.z` is also recorded for later gating.
- CFG: `USE_PC=True, PC_N1=5000, PC_WORKERS=4`. **VERIFIED**

### 1.4 Engine 2 (a second, independent candidate pool and ranking)

- `eng/casmi_engine.py::build_pool`: `COCONUT prvsiyan fingerprints` ∪ `bio_fp.npy`
  (ChEBI+LIPID MAPS, deduplicated by key) ∪ fingerprints computed on the fly from `train.parquet`
  structures. Fingerprints = Morgan r2/r3 (4096) + RDKitFP (2048) + MACCS, selecting **6,930 bits**
  per `fp_bits.npy`, then `np.packbits`. **VERIFIED**
- The same `pool_window(target, 10 ppm → 30 ppm fallback)`.
- Two rankers (`W_A=(0.35,0.55)` × `SEEDS=(0,1)`, HistGradientBoosting, 4 GBMs) plus prvsiyan's
  ranker (`W1_PRIORS=(0.30,0.60)` × 4 seeds, 8 GBMs more), fused by **rank blend**
  `blend_scores(a,b,wa=0.88)`; then `ORDER_K=80` → deduplicate → **`TOPK=40`**.
  **VERIFIED** (`eng/eng_runner.py` verbatim)
- Channels include analog propagation (`ANALOG_WIN=200.0` Da, `N_ANALOG=80/200`, `SIM_POWER=3.0`)
  and a MetFrag-lite `explain_score`.

### 1.5 Final candidate count per query

| Stage | Count |
|---|---|
| BASE (after engine-1 ranking) | **top 60** (`TOPN=60`) |
| candidates actually scored by ICE/GL | those among the 60 that **share a formula with ≥2 members in the group** (`ice_candidates`) |
| + engine-2 union | `ICE_UNION_K=40` (engine-2 top-40 whose keys are absent from BASE) |
| + PubChem-only (when the gate passes) | at most **25** |
| engine-2 list | top **40** (`ENG_TOPK=40`) |
| submission | at most **25** SMILES per molecule, `;`-joined |

**VERIFIED** (`ice_candidates` verbatim):

```python
def ice_candidates(smis, formulas, top_n=40):
    """SMILES worth predicting for one molecule: members of same-formula groups with >= 2 members inside the top_n
    (the only candidates rerank can move). Keeps ranked order; duplicates removed."""
```

## 2. GLACIER

### 2.1 What it is

**GLACIER** = the Coley group's `ms-pred` **single-stage (DETR-style) MS/MS forward-prediction
network**: a Graphormer backbone plus learnable object queries for "cleavage-site detection" and an
intensity head — Graph Learning of Atomic Component Instances via End-to-End Recognition.
Paper arXiv:2606.29161; MassSpecGym top-1 retrieval 70.0% (69.7% without CF), roughly **8× faster
inference** than the two-stage baseline. **VERIFIED**
(`https://ar5iv.labs.arxiv.org/html/2606.29161`, `https://github.com/coleygroup/ms-pred`)

**How GLACIER is used in this pipeline = forward fragment-spectrum prediction plus a spectral
similarity score** (not a likelihood, and not candidate generation): for each candidate molecule,
predict an MS/MS spectrum, compute **entropy similarity** against the observed spectrum, and use the
scalar `gl_ex_mean` as a **third z-score term inside the same-formula group**.

### 2.2 Packaging and offline-ability (a substantial amount of engineering worth copying)

Kaggle dataset **`ahmedberatozer/casmi26-glacier`**, 61,108,325 bytes (58.3 MB):

- `ckpt/glacier.pt` — 60,531,887 bytes; **264 tensors / 15,108,154 params**; re-saved from the
  official "GLACIER_checkpoint.zip → best.ckpt" (sha256 `5a47cecca707d3ab…`) keeping only
  `{hyper_parameters, state_dict}`, with tensors **bit-identical fp32** (optimizer/trainer/scheduler
  dropped).
- `gl_runner.py` (39,880 B) — a fork of `ice_runner.py` with **identical CLI / input format / job
  construction (adduct / instrument token / CE bucket) / scoring definition / chunking / budget /
  checkpoint-resume / meta JSON / failure behaviour**, changing only model loading and prediction.
- `gl_fuse.py` (7,533 B) — notebook-side glue: `run_gl` (spawns a subprocess, never raises, returns
  `{}` on failure) and `rerank_multi` (the multi-re-scorer form of `rerank`).
- `src/ms_pred/` — the ms-pred inference subset at **commit `708148c2a8eb`**, **MIT**
  (`Copyright (c) 2023 Samuel Goldman`), keeping only the modules inference needs.
- `shim/dgl`, `shim/torch_scatter`, `shim/LinSATNet` — **pure-torch stand-ins** (LinSATNet rewritten
  as an equality-constrained Sinkhorn top-k projection, used only for cleavage-site selection).
  → it runs on the Kaggle image without `dgl` / `torch_scatter`, which is the key to "offline-usable".
- **No wheels of its own**: it reuses ICEBERG's dataset `wheels/*.whl.ice` (which includes
  **RDKit 2025.3.6**). Deliberately named `*.whl.ice` because the notebook globs
  `/kaggle/input/**/rdkit-*.whl` to install RDKit 2026.03.3 for the main process — which must not be
  contaminated. **VERIFIED (LICENSE-NOTICE.txt / MANIFEST.json)**

**Licence**: the GLACIER weight download page carries no separate licence, so it is used under the
ms-pred repo's **MIT**; the dataset as a whole is marked "Other (specified in description)".
**VERIFIED (LICENSE-NOTICE.txt verbatim)**. The official weight source is the ms-pred README's
Dropbox link, "checkpoint of GLACIER trained on the MassSpecGym dataset". **VERIFIED (ms-pred README)**

### 2.3 Scoring definition (`gl_runner.py` head docstring, verbatim)

```
Score = the A1 panel definition `gl_ex_mean` (... = the ice_ex_mean definition):
  covered spectrum : mode +1 and adduct in {[M+H]+, [M+Na]+}
  instrument token : 'Orbitrap' if instr_family(instrument_type) == 1 else 'QTOF'
  CE               : units == 'NCE' -> mean of the numbers in collision_energy_orig, else mean(collision_energy_ev)
                     (NaN if empty); bucket = np.round(ce / 5) * 5, min 5; NaN -> 'sum' prediction (CE 20+40+60 merged)
  molecule         : predict_smis_joint.prepare_entry canonicalisation (valid atoms only, RemoveHs, <= 100 heavy
                     atoms, MolToSmiles); rejected -> null
  prediction       : TreeProcessor.featurize_tree -> IntenDataset.collate_fn -> JointModel.predict_inten_frag_batch
                     (collision_engs = CE bucket, instrument = token, adduct as given, precursor 0) -> drop rows with
                     intensity or m/z <= 0 -> merge identical m/z (round 4) by MAX -> top 100 -> sort by m/z ->
                     max 1 (float32)
  query            : raw peaks -> clean_peaks(prec, floor 0.001, top 512, <= prec + 2) (library cache, float32) ->
                     clean_peaks again (score.py) -> prep_query(floor 0.002, top 256, power 1, entropy weighting)
  similarity       : engine entropy_sim, 0.01 Da / 20 ppm;   gl_ex_mean = mean over the molecule's covered spectra
                     that have a prediction (NaN sims skipped).
```

Constants (verbatim): `COV = ('[M+H]+', '[M+Na]+')`, `SUM_CE = (20.0, 40.0, 60.0)`,
`FLOOR, TOPK, POWER, ENTW, TOL, PPM = 0.002, 256, 1.0, True, 0.01, 20.0`, `RDKIT_REQ = '2025.03'`.

Batch size: `batch_size_for(N) = max(1, min(bs_max, budget // N²))`, `budget = 40*40*32`, `bs_max=32`
(`ATOM_BUDGET_PER_BS = 40 * 40`). Failure guard: `FAILFAST_N = 64`.

### 2.4 Model size / runtime

- **15.1 M params / 58 MB fp32**, running on a T4 over 400 molecules' matched candidates.
- The default `--budget` is set to **`GL_BUDGET=4000` s** in this notebook; the ICEBERG reference
  docs give ICE at about **25–30 min** on a T4 with N=60 and 400 molecules
  (`~25–32 predictions/s`), and the paper claims GLACIER is about **8× faster**.
  **VERIFIED (the numbers) + LIKELY (the conversion)**

## 3. What "MH" is

**MH = `[M+H]+` (the protonated positive ion) — not molecular formula, not MetFrag, not MSHub.**

**VERIFIED** by three independent pieces of evidence:

1. the notebook's first markdown line, verbatim: `# CASMI26 Fusion + GLACIER [M+H]+`
2. the notebook's own description, verbatim:
   > Our v8 change restricts GLACIER to **[M+H]+** spectra; that idea also appears in
   > Wang Penghua's public work.
3. cell 18, verbatim (this is the entire implementation of "MH" — two lines):

```python
# v8 knob (v29/v27, public LB +0.006): GLACIER scores [M+H]+ spectra only
gl_items = [dict(it, spectra=[s for s in it['spectra'] if s.get('adduct') == '[M+H]+']) for it in items]
gl_items = [it for it in gl_items if it['spectra']]
```

That is: the GLACIER runner itself supports both `[M+H]+` and `[M+Na]+` as covered spectra, and
**this notebook filters out every `[M+Na]+` spectrum before handing items to GLACIER**, so only
molecules with `[M+H]+` get a GLACIER score (no score = `{}` = that term's z is taken as 0, falling
back to ICEBERG). The code comment puts the public-LB gain at **+0.006** — the same order as the
±0.006 LB noise. Attribution: Wang Penghua's `casmi26-v29-fusion-popglmh`.

> Do not confuse this with the MetFrag-style term that also exists in the pipeline: that one is
> **`frag` / `frag_rescore.py`**, not MH. Its docstring describes it as the "241-structure
> same-formula panel **+0.065 MRR**" item found by forum ablation, used with
> `FRAG_LAM=0.5, FRAG_MODE='gap'`.

## 4. Fusion / ranking

### 4.0 The CFG that actually took effect in this notebook (**the most important passage**)

Cell 4 defines `CFG`, and its last line is a build-time override — **this is the actual
configuration behind the 0.413**:

```python
CFG.update({'VERSION': 'v4b-libgate-pop025', 'POP_MU': 0.25, 'FRAG_LAM': 0.5, 'FRAG_MODE': 'gap',
            'ICE_LAM_LIB': 0.0, 'GL_LAM_LIB': 0.0})   # --set overrides from build_notebook.py
```

The defaults are `POP_MU=0.0, FRAG_LAM=0.0`; they are overridden to **`POP_MU=0.25`**,
**`FRAG_LAM=0.5` used only on molecules ICE cannot cover (`gap`)**, and **library hits' ICE/GL weights
forced to 0**. **VERIFIED (verbatim)**

Other relevant defaults (not overridden):

```python
TOPN=60, FP_BANK='full1',
LIB_TAU=0.9, REL_TH=600.0, SLOTS_AGG=[2,4,6,8,10], SLOTS_GENTLE=[4,8,12,16,20],
USE_PC=True, PC_N1=5000, PC_WORKERS=4,
USE_ICE=True, ICE_LAM=1.0, ICE_BUDGET=5400, ICE_PC=True, ICE_UNION=True, ICE_UNION_K=40,
USE_GL=True, GL_LAM=1.0, GL_BUDGET=4000,
USE_ENG=True, ENG_W_PV=0.88, ENG_W_A=[0.35,0.55], ENG_SEEDS=[0,1], ENG_N_ANALOG=200,
ENG_ORDER_K=80, ENG_TOPK=40,
FUSE_ALPHA=0.6, FUSE_KRR=3.0, FUSE_N=40, FUSE_ENG_K=40,
POST_ICE_LAM=None, POST_GL_LAM=None, FUSE_SKIP_LIB=None, FILL_25=False,
POP_LIB_OFF=False, FRAG_LIB_OFF=False,
SMOKE_N=12, SMOKE_ICE_BUDGET=300, FULL_ON_COMMIT=False, DUMP_STAGES=True, STAGE_TIMEOUT_S=4*3600,
```

On the `fusion_core.DEFAULTS` side: `LIB_TAU=0.9, REL_TH=600.0, TOPN=60, ICE_LAM=1.0, GL_LAM=1.0,
ICE_PC=True, FUSE_ALPHA=0.6, FUSE_KRR=3.0, FUSE_N=40, FALLBACK='CCO', POP_MU=0.0, FUSE_ENG_K=40,
FRAG_LAM=0.0, FRAG_MODE='all'`. **VERIFIED**

> ⚠️ Note this notebook uses **`ICE_LAM=GL_LAM=1.0`** (for non-library-hits), whereas the two
> datasets' READMEs recommend **0.5 / 0.5**. This is a knob the scored configuration changed.

### 4.1 The three stages

`fusion_core.build_submission`'s docstring, verbatim:

```
  stage A  z(ranker) + ICE_LAM z(ICEBERG) + GL_LAM z(GLACIER) inside same-formula groups (base + PubChem lists)
  stage B  gated PubChem merge: untouched if lib_max >= LIB_TAU, else PubChem-only structures into fixed slots
  stage C  weighted reciprocal-rank fusion with engine-2 lists, then the ICE/GL re-rank again on the fused list
```

**Stage A** (linear z-fusion inside same-formula groups, verbatim core):

```python
def rerank_multi(smis, keys, scores, formulas, score_dicts, lams, top_n=60, min_covered=2, tie_eps=1e-9):
    ...
    fused = _z([scores[i] for i in idx])
    n_act = 0
    for v, lam, c in cols:
        if c >= 2:                                   # n < 2 -> z 0 (fuse._z); skip the no-op addition
            zi = _z(v)
            fused = [a + lam * b for a, b in zip(fused, zi)]
            n_act += 1
    if n_act >= 2 and tie_eps > 0:
        fused = [round(x / tie_eps) for x in fused]  # monotone: never reverses an order, only merges float noise
    new = [idx[k] for k in sorted(range(len(idx)), key=lambda k: (-fused[k], k))]
    for slot, src in zip(idx, new):                  # idx is ascending = the group's slots
        order[slot] = src
```

The definition of z (pandas `groupby.transform` semantics, verbatim):

```python
def _z(vals):
    """= fuse._z: pandas groupby-transform z: (x - mean) / std(ddof=1) over non-missing; missing / std 0 / n < 2 -> 0."""
```

**Three details that must be copied correctly**:

1. **Reordering happens only inside same-formula groups within `top_n`, and is slot-preserving**
   (members swap only among their own original positions) — so a weak candidate is never lifted to
   the front wholesale.
2. **A group moves only if at least 2 of its members are covered** by that re-scorer
   (`min_covered=2`).
3. With several re-scorers, `tie_eps` rounding avoids cases like "ICE and GL both oppose the ranker,
   giving an exact `±0.7071*(1-0.5-0.5)=0` tie" being **broken randomly by float noise**.
4. `lam=0` with coverage leaves the group alone → this is how **a library hit (`lib_max≥0.9`) is
   frozen using `ICE_LAM_LIB=GL_LAM_LIB=0.0`**.

Ordering (cell 18 + `fusion_core`): `popularity_reorder` (if enabled) → `ICE rerank` →
`GL rerank_multi` → `frag_rerank` (gap mode only where ICE has no coverage).

**Stage B — the gated PubChem merge** (verbatim):

```python
bp = p.get('best_pool_fz')
rel = p['pc_fz'][0] - bp if bp is not None and np.isfinite(bp) else 1e9
aggressive = rel > c['REL_TH']
ms['aggressive' if aggressive else 'gentle'] += 1
final = merge(smis, keys, p['pc'], p['pc_keys'], c['SLOTS_AGG'] if aggressive else c['SLOTS_GENTLE'])
```

```python
def merge(base, base_keys, pc, pc_keys, slots, n=25):
    """PubChem-only structures (not already in base) go into `slots` (1-based); base fills the rest."""
```

- `lib_max ≥ 0.9` → **left completely untouched** (`untouched`).
- Otherwise PubChem-only structures are **inserted into fixed slots**: if `f.z` clearly exceeds the
  best in-pool value (`rel > REL_TH=600`) use the aggressive slots `[2,4,6,8,10]`, else the gentle
  slots `[4,8,12,16,20]`. Remaining positions are filled from base order.
- **The gate uses the pre-rerank `pc_fz[0]`** (code comment: `the gate keeps the ORIGINAL top f.z,
  so gate decisions do not move`) — the gate is decoupled from the rerank, avoiding a feedback loop.
- The PubChem channel is nearly always wrong, so it gets only 5 slots rather than replacing the list.

**Stage C — weighted reciprocal-rank fusion (RRF) plus a re-rank** (verbatim):

```python
def fuse_rrf(v_smis, e_smis, e_keys, score_key, alpha, krr, n):
    """sum 1/(krr + r) over the v4 list plus alpha/(krr + r) over the engine list, keyed by the metric key."""
```

`alpha = FUSE_ALPHA = 0.6`, `krr = FUSE_KRR = 3.0`, `n = FUSE_N = 40`, engine contributes only its
top `FUSE_ENG_K = 40`; after fusion **ICE / GL re-rank runs once more** (`POST_ICE_LAM/POST_GL_LAM`
default to `ICE_LAM/GL_LAM`). Finally `FILL_25=False` → lists are not padded to 25; a short list
stays short.

### 4.2 Channels and features used in fusion

| Channel | Signal | Use |
|---|---|---|
| LightGBM ranker (engine 1) | `rank_score([X,F])` | the main z-fusion term (within each same-formula group) |
| Experimental library match | `lib_max` (entropy sim) | **hard gate** `LIB_TAU=0.9` freezes; not part of z |
| FPNet (spectrum → fingerprint) bank | `FP_BANK='full1'` (v4m, trained on all folds); the PubChem channel uses A+B | produces the gate score `f.z` and `best_pool_fz` |
| ICEBERG 2.1 | `ice_ex_mean` | z term, `ICE_LAM=1.0` / 0.0 on library hits |
| **GLACIER** | `gl_ex_mean` (**`[M+H]+` only**) | z term, `GL_LAM=1.0` / 0.0 on library hits |
| MetFrag-parsimony (`frag_rescore.py`) | weighted share of ≤2-cleavage fragment explanation | `FRAG_LAM=0.5`, **only where ICE has no coverage** (`gap`) |
| PubChem popularity prior | `log1p(SIDs)+log1p(PMIDs)` | `z(ranker) + 0.25 * pop[pid]`, see below |
| engine 2 | ranks | RRF with `alpha=0.6` |
| analog propagation | Tanimoto × sim^3/^6 etc. | into ranker features (engine 1's `_analog_feats` gives 5 dims) |

The popularity prior (verbatim — note it is **not** a weighted sum into z, but a reorder among pool
candidates only):

```python
def popularity_reorder(smis, keys, scs, forms, pids, pool_pop, mu):
    """bobthebot369 v10 prior: f = z(ranker) + mu * pop[pid] for pool candidates; they are re-sorted among the pool
    slots, generated candidates (pid < 0) keep their slots and get a '|gen' formula tag (own same-formula group)."""
    s = np.asarray(scs, np.float64); z = (s - s.mean()) / (s.std() + 1e-9)
    pid = np.asarray(pids); isg = pid < 0
    f = z + mu * np.where(isg, 0.0, pool_pop[np.maximum(pid, 0)])
```

**"GBDT or linear?"** — at the candidate level it is **GBDT** (4 `HistGradientBoostingClassifier`s
× 500 trees each, or the main engine's LightGBM ranker); **the channel fusion itself is a linear z
sum plus RRF**, with no further GBDT at the fusion layer. That is consistent with this project's
discipline of "prefer linear fusion; a GBDT must pass V-C".

## 5. What 0.413 is actually made of (relative to a ~0.13 pipeline)

> ⚠️ I could not locate any specific 0.13-scoring notebook, so this section is **not a measured
> ablation against a 0.13 baseline**. It identifies, item by item from this 0.413 code, which
> channels are *additional*, plus the gains the notebook itself annotates. The implementation
> evidence for each item is VERIFIED; the causal claim "this is what a 0.13 pipeline lacks" is
> marked **LIKELY / inference**.

**(a) Experimental library matching with a hard gate (the biggest item).** The first thing the
pipeline does is an entropy-similarity search against `train.parquet`'s multi-source experimental
library (all of a molecule's spectra, with `search_shift_rows` — shifting reference spectra by
`query_prec - ref_prec` so the same compound measured under a different adduct still matches), and
when `lib_max ≥ 0.9` it **bypasses the rest of the chain entirely**. A pipeline that ranks
fingerprints only within COCONUT/PubChem and never uses the experimental library loses this whole
class of molecules. `LIKELY` (code VERIFIED, causation inferred).

**(b) A dual-engine candidate pool, independent of PubChem.** The 710,701-structure pool (train
structures ∪ COCONUT, plus ChEBI+LIPID MAPS in engine 2) is far more precise than taking
same-mass/same-formula candidates straight from PubChem; PubChem's 105.9M structures serve only as a
**gated supplementary channel** (5 fixed slots), not the main force. `LIKELY`

**(c) z-fusion inside same-formula groups, not global linear weighting.** Reordering happens only
within groups that share a formula and have ≥2 members, and is **slot-preserving**. That solves
"score scales are incomparable across formulas" while ensuring **a candidate ranked 30th cannot
jump to 3rd** — the key to out-of-domain robustness. A globally weighted score (or weights tuned on
the visible test) is exactly what collapses here. `LIKELY`

**(d) Two forward models rather than one.** ICEBERG 2.1 (`msg_all`, MassSpecGym) and GLACIER
(single-stage, MassSpecGym) predict **independently** and are z-scored separately. Their failure
modes differ, and `gl_fuse.rerank_multi` weights each z term separately with any term missing
(`None`) automatically counting as 0 rather than raising. `LIKELY` (`ICE_LAM=GL_LAM=1.0` with
separately tunable `POST_*` shows the author really did weight them separately.)

**(e) The `[M+H]+` restriction (= MH), +0.006.** Explicitly annotated by the notebook.
`VERIFIED (annotated value)`. Mechanism (my inference): GLACIER's `[M+Na]+` coverage/accuracy on
MassSpecGym is poor, and feeding a noisy term into the z-fusion contaminates groups ICEBERG already
got right. `UNVERIFIED (mechanism)`

**(f) MetFrag-parsimony only where ICE has no coverage (`FRAG_MODE='gap'`), +0.065 (panel).**
`frag_rescore.py` docstring, verbatim:

```
Defaults follow the forum ablation on a 241-structure same-formula isomer panel (+0.065 MRR over the common
baseline): w2=0.6, linear intensity, tol 0.005 Da, H shifts -2..+3, at most 2 cleavages.
BASELINE_CFG is the common/pv.py-style setting.
```

The configuration uses `gap` mode with `FRAG_LIB_OFF=False`, i.e. **this term is added only for
molecules ICEBERG does not cover** (negative mode, `[M+NH4]+`, …). That captures the panel gain
without touching the main path. `VERIFIED (documented) / LIKELY (attribution)`

**(g) PubChem popularity prior, `POP_MU=0.25`** (bobthebot369 v10 used 0.15 at LB 0.401). It acts
only on **pool candidates** (`pid ≥ 0`); generated candidates (`pid < 0`) keep their positions and
are tagged `|gen` in the formula field (their own group, excluded from the in-pool reorder).
`VERIFIED (code + comment)`

**(h) Engineering: every stage degrades silently on failure.** `USE_ENG/USE_PC/USE_ICE/USE_GL/FRAG`
are all inside `try/except`; an exception in any stage blanks that channel, keeps the upstream order,
and still yields a valid `submission.csv` (`fusion_core.validate` asserts at the end). In a
competition where a submission must succeed, this **pins the worst case to the rollback baseline**
rather than to zero. `VERIFIED`

**(i) Temperature/split discipline.** Cell 8's `build_validation()` constructs an
**identity-disjoint hold-out**: take 250 structures from `enveda-np-examples` and delete **all**
their train rows (`purge='all'`, simulating class 2 "the library does not know it at all") or only
that library's rows (`purge='lib'`, simulating class 1 "the same compound was measured by another
library"), then compute MRR@25 under the metric's convention (tautomer-canonical InChIKey14). This
corresponds exactly to this project's **V-C / V-A** fold idea. `VERIFIED`

## 6. Runtime and offline packaging

### 6.1 Attached Kaggle datasets (14, from the kernel metadata's `datasetDataSources`)

| # | Dataset | Size | Licence |
|---|---|---|---|
| 1 | `prvsiyan/casmi26-fp-models-v2` | 288 MB | CC0 |
| 2 | `dmitriigluzdov/casmi26-pubchem-popularity-prior` | 2.75 GB | Other |
| 3 | `prvsiyan/casmi26-ranker-features` | 9.5 MB | CC0 |
| 4 | `megayak/casmi26-simulated-ranker-rows` | 366 MB | CC0 |
| 5 | `ahmedberatozer/casmi26-fpnet-full1` | 199 MB | Other (contains CC BY-NC components) |
| 6 | `ahmedberatozer/casmi26-glacier` | 61 MB | Other (contains MIT ms-pred) |
| 7 | `ahmedberatozer/casmi26-iceberg` | 67 MB | Other (contains MIT ms-pred) |
| 8 | `ahmedberatozer/casmi26-pubchem-tier` | 7.21 GB | Other (public NCBI PubChem data) |
| 9 | `ahmedberatozer/casmi26-v2-pool` | 1.68 GB | Other (contains CC BY-NC components) |
| 10 | `ahmedberatozer/casmi26-v3-models` | 412 MB | Other |
| 11 | `ahmedberatozer/casmi26-v4b-models` | 1.68 GB | Other |
| 12 | `prvsiyan/chebi-lipidmaps-casmi26` | 60 MB | **CC BY-NC-SA 4.0** |
| 13 | `prvsiyan/coconut-casmi26-candidates` | 423 MB | **CC BY 4.0** |
| 14 | `metric/rdkit-2026-3-3-wheel` | 149 MB | Unknown |

Plus the competition data `enveda-CASMI26-molecule-id-mass-spectra`. About **15.2 GB** of input in
total. **VERIFIED** (the `datasetDataSources` field plus each dataset's metadata `totalBytes`)

### 6.2 Environment

- `enableGpu=true`, `machineShape=NvidiaTeslaT4`, **`enableInternet=false`**
- docker image `gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461`
- RDKit: cell 6 installs offline from the dataset via `pip install --no-index --no-deps rdkit-*.whl`
  (the wheel must match the `cp{major}{minor}` tag), then **asserts
  `rdkit.__version__ == '2026.03.3'`** (the metric convention pins it, since the key is the
  tautomer-canonical InChIKey14).
- The ICE/GL runners install **RDKit 2025.3.6** into **their own site directory**
  (`wheels/*.whl.ice`, `pip --no-index --no-deps --target`), and the runner refuses to run unless it
  sees `2025.03` (verbatim reason: `2026.03 kekulizes differently and changes predictions`).
  → **Two RDKit versions coexist in one run**, isolated by subprocess plus `--target`. One of the
  most valuable things to copy.
- `casmi26-v4b-models` is verified file-by-file with `MANIFEST.json`'s sha256, failing by `assert`;
  the `derivation` family has a `self_test()` asserting RDKit build consistency (`assert n_bad == 0`).
- `NUMBA_CACHE_DIR` points at `/kaggle/working/numba_cache` (a writable numba compilation cache).

### 6.3 Time budgets and runtime

| Stage | Budget |
|---|---|
| ICEBERG | `ICE_BUDGET = 5400 s` (`SMOKE_ICE_BUDGET = 300` on smoke runs) |
| GLACIER | `GL_BUDGET = 4000 s` |
| engine 2 / PubChem channel subprocesses | `STAGE_TIMEOUT_S = 4 * 3600` (each) |
| ICE/GL subprocess hard kill | `budget + grace_s`, `grace_s = 300` |

- ICEBERG reference docs: T4, 400 molecules, N=60 → **about 25–30 min**, `~25–32 predictions/s`
  (a measured 79/s on an RTX 5090 laptop). **VERIFIED**
- GLACIER paper: about **8× faster inference** than the two-stage baseline. **VERIFIED**
- **The notebook's actual total wall clock: UNVERIFIED** — every cell's `outputs` is empty in the
  pull API's JSON (`cells with outputs: 0`), so no run log is available. Given the budget caps and
  the rates above, **LIKELY about 2–3 hours** (T4).
- Checkpoint-resume design: the ICE/GL runners **write an all-null `out.json` before starting**, then
  overwrite it per chunk, stopping cleanly when the budget runs out — the process exits 0 as long as
  it has written `out.json` at all. → **exhausting the budget cannot destroy the submission**; those
  molecules simply fall back to the previous ordering.
- `DUMP_STAGES=True` → `/kaggle/working/stage_cache/{base,pc,eng,ice,gl,frag,meta}.json`, which lets
  a local `tools/refuse.py` **re-run and re-tune the fusion on CPU** without touching the GPU again.
- Cleanup: `shutil.rmtree('/kaggle/working/ice_site')` — the wheel site holds thousands of files, so
  removing it keeps the version's output listing manageable. **VERIFIED**

### 6.4 Submission format

```python
v4 = [(mid, rows.get(mid, c['FALLBACK'])) for mid in sample_ids]   # FALLBACK = 'CCO'
# rows[mid] = ';'.join(final[:25])
```

Semicolon-separated, in `sample_submission.csv`'s `molecule_id` order, at most 25 entries;
`fusion_core.validate` asserts matching row counts and uniqueness, non-empty `smiles`, at least 1 and
at most 25 entries.

## 7. WHAT I COULD NOT VERIFY

1. **Total runtime / per-stage measured durations.** The kernel-pull JSON has all `outputs` empty —
   no `[ 123s] …` log lines, no `run_manifest.json`. Only the code's budget caps and a third-party
   rate estimate for ICE. §6.3's 2–3 h is an **extrapolation**.
2. **Whether the 0.413 score corresponds exactly to this published CFG.** The notebook says "original
   submission 56792011, score verified on 2026-10-03", while this page's `lastRunTime` is
   2026-10-01, and Kaggle explicitly notes "competition-score badge requires a scored submission
   from this new page". I read the `CFG.update(...)` line verbatim, but **cannot prove it is the
   hyperparameter set used for the scored run**.
3. **The main engine v1's internals** (`casmi26-v4b-models`' `casmi` package): 1.68 GB, not
   downloaded. So `lib_max`'s exact computation, `FEATURES` (the ranker's feature names and count),
   the `fe_v4` families, `EngineCfg(generate=True)`'s generation logic and `fpnet.ModelBank`'s
   structure are all **inferred from call sites**, not confirmed by reading source.
   → **Resolved later** for the most part: `casmi26-v3-models` (411.9 MB) was downloaded and its
   `engine.py` / `ranker.py` / `traindata.py` read, confirming `lib_max`'s definition and giving the
   160-feature list ([`PLAYBOOK.md`](PLAYBOOK.md) §16–17).
4. **The internals of `casmi26-fp-models-v2` / `casmi26-v3-models` / `DreamsFP D` /
   `casmi26-ranker-features` / `casmi26-simulated-ranker-rows`**: not downloaded; only their purposes
   and byte counts are known. For engine 2's `BLOCKS` (feature blocks) I saw only the call form, not
   the concrete composition of the 31+ dimensions.
   → **Partly resolved later**: `casmi26-ranker-features` was downloaded and its 31 features ×
   819 groups characterised ([`PLAYBOOK.md`](PLAYBOOK.md) Part V).
5. **What the "0.13 pipeline" actually is**: no specific 0.13 notebook or public record of that score
   was located, so §5's comparison is **structural reasoning** ("here is what this one has extra"),
   not a measured ablation against 0.13.
6. **Why the `[M+H]+` restriction is worth +0.006.** The code comment gives the number, not an
   ablation table.
7. **Whether the PubChem channel's pass-1 `PC_N1` is 5000 or 1000.** `probe_core2.py`'s docstring says
   `keep top PC_N1 = 1000`, but the module constant is `PC_N1 = 5000` and CFG also sets 5000. The
   effective value is the environment variable `CASMI_PC_N1` (injected from CFG) = **5000**; the
   docstring is a stale comment.
8. **Whether `casmi26-pubchem-tier`'s 105,875,489 structures match what the notebook actually loads** —
   the dataset description and the popularity-prior description agree, but I did not download the
   7.21 GB to check `pc_mass.npy`'s length.
9. **Commercial viability of the licences.** Several datasets are marked "Other (specified in
   description)" and their own descriptions mention CC BY-NC components (Enveda competition training
   data; ChEBI/LIPID MAPS is CC BY-NC-SA 4.0). GLACIER/ICEBERG's code is MIT, but **the weights and
   the training-data chain are NC-constrained**.
