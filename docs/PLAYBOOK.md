# V45 ablation runbook

> **Goal**: move the public LB from **0.417 (rank 111)** to **> 0.417**, crossing a clone
> cluster of 61 tied teams (rank 59 = 0.418). The gap is about **+0.001**, so the strategy is
> controlled ablation: **one variable at a time.**

This document was reorganized for readability — it had grown by accretion, with section numbers
in the order they were written rather than the order they are read. Content is unchanged; the
original ordering is recoverable from git history.

---

# Part I — Operating rules

## 1. Metric anatomy (internalize this before running anything)

$$\text{MRR@25}=\frac{1}{400}\sum_{m}\frac{1}{\text{rank}_m}$$

**❌ A mistake I made and corrected**: I once computed the "ceiling of the tail (ranks 2–25)" as
$\frac{H_{25}-1}{400}=0.00704$. That is the "one molecule per rank" arithmetic, **not a
ceiling**. The correct ceiling: if all 400 molecules' truths ranked 2nd, the tail would
contribute $400\times\frac{1}{2}/400=\mathbf{0.5}$. **The tail is not a ceiling; it is the same
order of magnitude as the head.**

The score has two parts:

| Part | Contribution | Note |
|---|---|---|
| Head | $p$ (fraction of molecules at rank 1) | each extra molecule = **+0.0025** |
| Tail | $0.417-p$ | each molecule contributes $1/\text{rank}\in(0,0.5]$ |

Back-solving $p$:
- If $p=0.30$, the tail must contribute $0.117$, i.e. average $1/\text{rank}\approx0.167$
  (about rank 6) — plausible
- If $p=0.40$, the tail has only $0.017/0.60\approx0.028$ (about rank 35, i.e. most molecules
  outside the top 25) — implausible

⇒ **$p\approx0.30$–$0.35$; head ≈0.30–0.35, tail ≈0.07–0.12. Both are worth investing in.**

**Consistent with the public ledger**: `ICE_LAM=GL_LAM=1.0` is a pure tail change (ICE/GL never
alter top-1) and the public record shows **+0.007** — exactly "a tail of about 0.1, of which
forward reranking eats ~10%".

**Equivalent conditions for crossing 0.417 → 0.418** (any one):

- **1** more molecule at rank 1 (+0.0025)
- about **4** molecules moving from rank 5 to rank 3 (+0.00067 each)
- about **10** molecules moving from rank 10 to rank 5 (+0.00010 each)

## 2. Three facts to internalize first (or you will design the wrong experiment)

1. **The visible `test.parquet` is not the evaluation set.** Measured: one submission's file size
   as recorded by Kaggle was 530,716 bytes, while the same kernel's output on public data was
   393,908 bytes. The same code cannot differ by 35% on the same input ⇒ the hidden rerun's
   input differs. **Therefore any local score obtained by "recovering truth from train" cannot
   predict the LB.**
2. **The public LB is a deterministic oracle.** Re-scoring the same submitted file gives the same
   score, so deltas *within* one experiment are interpretable. Cross-version retraining noise
   (about ±0.006) does not apply here.
3. **Each variant costs two GPU runs.** Pushing a kernel version triggers a public-data batch run
   (~2 h), and submission is only allowed once that finishes; the submission itself triggers a
   hidden rerun (~2.5 h). Each account may hold at most **2 concurrent GPU sessions**; the
   submission allowance is **5/day per team**.

## 3. The real budget is a 30 h/week GPU quota, not "5 submissions/day"

Measured on this account via the `api.quota_view()` endpoint behind `kaggle quota`:

```
totalTimeAllowed = 1 day + 21600s = 30.0 h / week      <- the real budget
timeUsed         = 2637 s  (0.73 h)
timeReserved     = 41885 s (11.63 h)  <- two running sessions reserved at maximum duration
```

**This means I had mis-estimated experiment throughput.** The cost per variant is:

| Stage | Duration | Avoidable? |
|---|---|---|
| Public-data batch run triggered by the push | 2.0–2.8 h | **no** (submission is refused until it finishes) |
| Hidden rerun triggered by the submission | ~2.5 h | no |
| **Total** | **~5 h per variant** | |

⇒ **30 h/week ÷ 5 h ≈ 6 variants per week.** That is the real ceiling (a 5/day submission
allowance is 35/week and not a bottleneck at all).

**The discipline this implies**: every variant must answer "will this change a decision?" On that
basis I have already killed three things — pool expansion (ceiling 0.05%), a 1.15 GB code
download (the answer was obtained elsewhere), and `merge_force` (code analysis downgraded its
expectation from "possibly large" to "expected no change") — and gave the freed slot to `pop30`,
which has documentary support.

**One small fix made**: pushes now carry a session duration cap (`--session-timeout-s`, default
4 h). Without a cap, sessions are reserved at maximum duration (~5.8 h each), which limits how
many variants can be queued concurrently. Three variants × ~5.8 h still fits inside 30 h, so
this is headroom optimisation rather than a rescue — **the binding constraint is the 30 h/week
total.**

**The conclusion from this section alone**: spend carefully. Of ~6 weekly slots, only 1–2 should
bet on a new mechanism; the rest go to verification and combinations.

## 4. A variant's full lifecycle

```powershell
# 1) build the variant (assertion-based: a replacement that misses raises immediately,
#    so a "silent no-op experiment" is impossible)
python scripts\build_variant_kernel.py --variant icefull

# 2) push (= triggers the public-data run; consumes no submission allowance)
python scripts\push_kernel.py notebooks\v45\icefull

# 3) wait for COMPLETE -> auto-submit -> poll the score (run in the background, do not block)
python scripts\wait_submit_report.py --kernel nicholasnicklee/casmi26-v45-ablation `
    --version 3 --desc "V45 icefull: ICE_BUDGET 300->3000" --timeout-h 14
```

**The run must be COMPLETE first**, or submission is refused:
`Submission not allowed: Notebook is still running. Did not find provided Notebook Output File.`

## 5. Decision rules

| Observation | Conclusion |
|---|---|
| variant > ctl + 0.001 | real improvement; adopt it and next build its combination |
| variant ≈ ctl (±0.001) | that knob has no effect in this configuration; stop spending quota on it |
| variant < ctl − 0.001 | that knob is harmful; record it and stay away |

`ctl` is a verbatim copy of V44. **It does not need its own submission**: the author's code,
**character-for-character identical** to ctl, already scored 0.417, and gengsr's notebook claims
0.413 with our independent reproduction also **0.413, digit for digit** — reproduction is
deterministic. So ctl's score is **0.417**. (A ctl submission was once queued and then cancelled;
see the "do not submit ctl separately" section of `docs/EXPERIMENTS.md`.)

## 6. Variant list (one kernel slug per variant, so a push has zero state ambiguity)

| Variant | Kernel slug | Where it changes | Hypothesis | Status |
|---|---|---|---|---|
| `ctl` | `…-v45-ablation` v1 | nothing | control; score known = 0.417 | run complete |
| `nolock` | (retired) | delete the cell-31 lock | **ruled out by contradiction** (§14): the lock never fires, so it would just repeat 0.417 | cancelled |
| **`claw`** | `…-v45-claw` | cell 3 gate constants + cell 9 ports the v17 popularity probe + cell 19 `promote()` and gate | **the only mechanism in the field that can change rank 1** (§17) | **queued** |
| **`icefull`** | `…-v45-icefull` | cell 3 `ICE_BUDGET 300 → 3000` | the log shows ICE covered only 71/371 molecules; hidden molecules are larger ⇒ more candidates ⇒ starved harder | held |
| `unlock_engine` | `…-v45-unlock` | cell 7 removes `score[top] = max+1`, `pair_weight 0.15 → 0.35` | PairTail LambdaRank (`eval_at=(1,5,25)`) currently can never change rank 1 | built |
| `topn120` | `…-v45-topn120` | cell 3 `TOPN 60 → 120`, ICE budget 3600 | larger molecules have more candidates; top-60 may cut the truth | built |
| `pop30` | `…-v45-pop30` | cell 3 `POOLPOP_MU 0.15 → 0.30` | public ledger says +0.003~0.005; reranks BASE and can reach rank 1 | held |
| `claw_pop30` | `…-v45-clawpop30` | `claw` + `POOLPOP_MU 0.30` | combination | built |
| `pc_aggressive` | `…-v45-pcagg` | cell 3 doubles tail slots, `REL_TH 600 → 200` | let more out-of-pool PubChem candidates into the tail | built |
| `combo_a` / `combo_b` | `…-v45-comboa` / `…-comboB` | icefull+unlock / icefull+pcagg | combinations | built |

Pushing and queueing (`--after` points at the kernel currently holding the GPU sessions):

```powershell
python scripts\queue_variant.py --folder notebooks\v45\claw `
    --desc "V45 claw: v17 CLAW promotion" --after nicholasnicklee/casmi26-v45-ablation
```

## 7. Decision tree (follow this once a score arrives)

```
once both score(claw) and score(icefull) are in
├─ either > 0.417
│   ├─ immediately push its combination (claw -> claw_pop30; icefull -> combo_a/combo_b)
│   └─ freeze the winner as current best and write it into docs/LEADERBOARD.md
└─ both ≈ 0.417 (±0.001)
    ├─ push unlock_engine (a rank-1 lever, the only untried head knob)
    ├─ push topn120 (candidate depth; aimed at the confirmed distribution shift
    │   "hidden molecules are larger")
    └─ if still nothing ⇒ shallow knobs are exhausted, move to content routes:
        · merge LOTUS (150,590) + NPAtlas (33,497) into the candidate pool
          (the public 775k pool lacks both libraries)   [since measured: +373 structures, dead]
        · or verify "adduct-hypothesis expansion" (the research report measures a 3.34%
          adduct-assignment error rate in the library; if the hidden set has such errors,
          affected molecules' mass windows shift wholesale ⇒ guaranteed miss)
```

## 8. Variant creation is frozen; switch to strict serial (decided 2026-10-05)

**Decision**: build no more variants. The 20 already built are **assets, not a to-do list**.

**Why**: the pipeline can test about 6 variants a week (30 h GPU ÷ 5 h per variant) and I had
built 20. **The marginal value of building more is near zero, and it is speculative work that
project discipline explicitly rejects** ("no unrequested abstractions", "deletion over
addition"). Worse, I wrote "stopping here" and then built one more — which shows **self-restraint
alone is not enough; the rule has to go into the document.**

**What replaces it**:

```
advance one variant at a time
  -> take its score (+ log diagnostics)
  -> read it with the tiered rules in §18
  -> then decide whether the next is "its combination", "another independent lever",
     or "change direction"
```

**Queued but deliberately halted**: `icefull`, `pop30` (both queue jobs terminated having pushed
nothing — no loss). Not because they are bad, but because **whether they deserve 5 h depends on
`claw`'s result**: if `claw` works, that 5 h belongs to `claw_pop30`; if `claw` is inert because
its gate is too strict, it belongs to sweeping `S_TAU`. **Pre-committing slots to two independent
levers gives up the right to let the previous result choose the next step.**

**Cost**: one GPU slot sits idle for `claw`'s ~4.5 h. That is a **knowingly accepted** cost —
idling 4.5 h out of a 30 h weekly budget in exchange for the next direction being chosen by data
rather than guessed in advance.

---

# Part II — Where the score comes from

## 9. Decomposing 0.417 into recall × ranking (the first real anchor)

A **query-grouped** 5-fold cross-validation on the public ranker's training rows
(`prvsiyan/casmi26-ranker-features`, CC0) via `scripts/train_ranker_probe.py`, pure local CPU,
no GPU quota:

```
rows=142,762  features=31  groups=819  positives=1,638 (1.15%)
groups containing >=1 positive: 819/819 = 100.0%      <-- truth is always in the candidate set here
candidates/group: median 104, min 2, max 1352
group-wise CV:  MRR@25 = 0.7649   top-1 = 0.6886   hit@25 = 0.9512
```

**The key point: 100% of groups contain a positive in this simulated data**, i.e. **recall is
artificially fixed at 1.0**. Under that condition ranking reaches only **MRR@25 = 0.765**, while
the real LB is **0.417**.

Hence the first trustworthy decomposition (back-solving with
`score ≈ recall@25 × E[1/rank | hit]`):

| Quantity | Estimate | Meaning |
|---|---|---|
| `E[1/rank \| truth in top 25]` | ≈ 0.765 | conditional ranking quality from the table above |
| `recall@25` (real hidden set) | ≈ **0.545** | back-solved: 0.417 / 0.765 |
| ⇒ **about 45% of hidden molecules never get their truth into the top 25** | | |
| ⇒ total position | | the two halves are roughly equal |

**Two conclusions**:

1. **The bottleneck is not only ranking; recall is half of it.** I had been fixated on rank 1
   following "the head is 0.30–0.35", which covers only the ranking half; **the recall half
   (~0.19 of the score) had never been an explicit target.**
2. **The structure-pool recall route is closed by measurement** (§11 item 2: LOTUS/NPAtlas add
   only 373 structures). So **the only channel that can still expand recall is the PubChem tier**
   (7.2 GB, 150–1250 Da, far beyond the pool's 775k structures), and it is currently allowed only
   5 tail slots.

⇒ **This promotes `pc_aggressive` (double the tail slots) and "loosen the CLAW gate" from tail
tweaks to main-line recall work.** It is the first time I can give a quantitative argument for
"how large a share the PubChem channel should get".

**⚠️ Limitation that must be remembered**: 0.765 comes from **simulated** rows in which recall is
set to 1.0 throughout, which differs from the real evaluation distribution. `nazarmohammed`'s
header warns explicitly: *"simulated remove-the-truth-from-the-database novelty experiments can
dramatically overestimate real novel-molecule recovery"*. So **the 0.545 recall estimate is
soft** — it depends on assuming that the simulated rows' conditional ranking quality transfers to
real candidates. It is worth treating as a working hypothesis, not a conclusion.

## 10. Why is that 45% lost? — this decides which channel to fix

"Missing recall" has three causes, and **the available remedies are completely different**, so
they must be separated first:

| Cause | Symptom | Available remedy | Our assessment |
|---|---|---|---|
| **A. Truth is in the pool but cut by TOPN** (ranked below 60) | the engine scored it, it just never reached downstream | raise `TOPN` so ICE/GL can lift it | candidates per pool window are only ~65/molecule (ICE input 26,155/400), so **TOPN=60 barely cuts** ⇒ largely not the case |
| **B. Truth is not in the pool but is in the PubChem tier** | absent from the 775k pool | let PubChem candidates take more positions (`pc_tail10` / `pc_aggressive`) | the tier is ~100× the pool — **the only database channel that can still expand** |
| **C. Truth is in no public structure library** (a novel Enveda analog) | no retrieval can find it | **only analog propagation / Class-3 generation** (the engine's `generate=True` channel) | if the hidden set really contains many "potential" natural products, **this is the bulk** |

**Key inference**: if the loss is mostly **C**, giving PubChem more slots does not help — the
molecule is missing there too. `pc_tail10`'s benefit depends entirely on the B:C ratio, and **we
currently cannot separate B from C from the LB score alone**.

⇒ So we take the **lowest-asymmetry version**: `pc_tail10` touches only slots 16–25, so it cannot
harm existing correct candidates under A, B or C. It is a conservative "expand if you can" bet,
not an aggressive "B must be the bulk" bet.

⇒ Also recorded is a direction **no public solution has seriously addressed**: **C can only be
reached by analog/generative propagation**, which is both the field's weakest line and one we
have partly built ourselves in `src/casmi/fragments.py` and `pipeline_v2.py`. It belongs to the
modelling line, not the tuning line.

## 11. The public frontier is exactly 0.417 — we cannot copy our way past it

Scanned the titles/subtitles of **all 250 public notebooks** in this competition for anything
claiming ≥ 0.418:

```
total public notebooks seen: 250
=== notebooks advertising >= 0.418 ===
  none
```

The public best is **0.417** (`lehau007/casmi26-sota-v27-zenith-apex-0417`) — precisely where we
are. **The 59 teams between 0.418 and 0.471 have released no public code.**

### What this means for strategy

1. **The public design space is saturated near 0.417.** v17/v27 (with CLAW, with the popularity
   patch, ICE 5400) and our V44 (no CLAW, no patch, ICE 300) **both score 0.417** — direct
   evidence of saturation.
2. Therefore **the "within-public-space" knobs in the queue (including `claw`) will most likely
   land back on 0.417** — they are cheap lottery tickets, worth buying but **not to be counted on
   to cross**. Combining several sub-levers might edge above the single-lever saturation point;
   that is the only hope.
3. **To genuinely cross, we need something absent from public code.** Only three candidate
   classes exist: a better model, a different candidate source, a different algorithm. The second
   is closed by measurement (pool expansion ceiling 0.05%).

## 12. Component diff: our V44 vs the public 0.417 reference (v17 / v27)

Both score **0.417** but are composed differently. This table answers "what are we missing, and
what is it worth".

| Component | v17 / v27 (0.417) | Our V44 (0.417) | Note |
|---|---|---|---|
| v4b engine + `fpnet_full1` | ✓ | ✓ | |
| Dual-ranker engine (prvsiyan + megayak) | ✓ | ✓ **+ PairTail LambdaRank** | V44 adds a pair model (weight 0.15, tail only) |
| ICEBERG | ✓ **budget 5400** | ✓ **budget 300** | ⚠️ see below |
| GLACIER `[M+H]+` | ✓ budget 4000 | ✓ budget 4000 | same |
| Zero weight on library hits (`ICE_LAM_LIB=0`) | ✓ | ✗ | tail lever |
| **PubChem-only channel** | ✓ **popularity-patched** | ✗ unpatched v16 | produces `S`/`top_pop`/`fz_top` |
| **PubChem popularity prior** | ✓ `POP_LAM=0.25` | ✗ dataset mounted but unread | source of CLAW's gate quantities |
| **CLAW promotion rule** | ✓ `S_TAU=6, POP_TAU=5` | ✗ (ported this round) | the only mechanism that can change rank 1 |
| In-pool popularity prior | `POP_MU=0.25` | `POOLPOP_MU=0.15` | public ledger: +0.002 per 0.15 |
| MetFrag-style fragment forward term | ✓ `FRAG_LAM=0.5`, pool mode, **only for molecules ICEBERG cannot score** | ✗ (exists only as one engine-ranker feature) | **complements** ICE coverage |
| `FILL_25` / `FUSE_ALPHA=0.65` | v28 only | ✗ / 0.6 | |

### ⚠️ The most important contradiction in that table

**v17/v27 get 0.417 with a 5400 s ICE budget; our V44 gets 0.417 with only 300 s.** Two notebooks
differing ~5× in ICE coverage score the same ⇒ **the ICE budget is not this plateau's dividing
line.**

This directly **lowers `icefull`'s prior** (it was queued, but I no longer treat it as the main
bet). It also suggests v17's design of "use the fragment forward term only for molecules ICEBERG
cannot score" is likely patching a gap that **is not critical in the first place**.

⇒ **Conclusion: the dividing line is more likely on the "PubChem channel + CLAW" axis** (the two
rows v17 has and we entirely lack), not on ICE/GL coverage or weights. This agrees with item 4 of
the route list below.

## 13. Route list (what is closed, what is open, all with evidence)

The most expensive thing in experimentation is not compute — it is **time spent on a dead route**.
Each item below carries its basis.

| # | Route | Verdict | Basis |
|---|---|---|---|
| 1 | Hard-coding / memorising answers | ❌ **closed** | §14 by contradiction: the lock never fires; hidden `molecule_id`s differ from the visible ones |
| 2 | Candidate-pool expansion (LOTUS/NPAtlas) | ❌ **closed** | measured locally: in a 730,161-structure pool, **LOTUS ∪ NPAtlas adds only 373 structures (0.051%)** over train ∪ COCONUT. And the public pool (train 275,810 + COCONUT 499,133 + ChEBI/LIPIDMAPS 62,744) is already **larger** than our merged pool ⇒ no headroom at all |
| 3 | Adduct-hypothesis expansion | ❌ **low yield** | self-measured: on the visible test, "formula mass vs precursor+adduct mass" agrees **1209/1213 = 99.7%**; only 4 spectra deviate by ~1 Da. The organisers' adduct labels are self-consistent (the research report's 3.34% error rate refers to the **training library**, not this test set) |
| 4 | Get the PubChem channel into rank 1 (**CLAW**) | ✅ **open, best** | the public pool is saturated ⇒ out-of-pool answers can only be in the PubChem tier; CLAW is precisely the only mechanism that "promotes it to rank 1 only when confidence is extreme". **Queued** |
| 5 | Tail levers (ICE coverage / TOPN / tail slots) | ✅ open | the tail contributes ~0.07–0.12 (§1); public ledger `ICE_LAM=GL_LAM=1.0` = +0.007 |
| 6 | rank-1 knobs (`unlock_engine`, `POOLPOP_MU`) | ✅ open | variants built |
| 7 | Training a better ranker / FPNet | ✅ open but expensive | needs a trustworthy hold-out; the visible test is a decoy, so a "not in the library" hold-out must be built from train |
| 8 | Time/mass window tuning | ❌ near saturation | research report §3.3: the O↔CH₄ substitution deficit (36.4 mDa) exceeds any ppm-scale window, so tightening the window cannot in principle separate isobars |

**The measurement behind item 2** (runs in ~10 seconds):

```python
df = pd.read_parquet(r'data\processed\candidate_pool.parquet', columns=['key','sources'])
s = df.sources.astype(str)
marg = (s.str.contains('lotus') | s.str.contains('npatlas')) \
       & ~s.str.contains('coconut') & ~s.str.contains('train')
print(marg.sum())        # -> 373
```

**Conclusion**: the pool route is exhausted. This **reinforces item 4** — since structural
coverage is saturated, score can only be squeezed out of "how to get out-of-pool PubChem evidence
promoted at the right moment", which is exactly what CLAW does.

## 14. Known bottlenecks read off the run log

The public-data run log, in full:

```
ICE meta {"status":"budget","n_mols":400,"n_mols_covered":371,"n_mols_scored":71}
ICE rerank stats {'molecules': 71, 'changed_top25': 67, 'changed_top1': 0}
GL rerank stats  {'molecules': 69, 'changed_vs_ice_top25': 66, 'changed_vs_ice_top1': 0}
merge stats {'untouched': 400, 'gentle': 0, 'aggressive': 0, 'no_pc': 0}
fused (+ICE) submission written; top-1 changed in 6 | ICE reordered 365
V29 final top-1 locks applied: 1
total 7,370 s (ceiling 9 h)
```

Key points:
- **The forward fragment models (ICEBERG/GLACIER) never change rank 1**, only reordering ranks
  2–25 → the tail is the only place they can improve.
- `merge stats untouched=400` shows the PubChem channel was gated off wholesale by
  `lib_max ≥ 0.9`. Normal on public data (the answers *are* in the library); unknown on the hidden
  set.
- The engine ships a top-1 shield: `gate_stats {'top1_locked_pair_tail': 400}`.
- Only 2.05 h of 9 h used — about 7 h of compute headroom for more channels.

---

# Part III — Evidence dossier

## 15. Settled: the champion lock never fired on the hidden rerun (so `nolock` was redundant)

This matters because it kept resurfacing. Use **proof by contradiction** to eliminate every
scenario in which the lock could have fired:

| Scenario | Hypothesis | Predicted score | Actual | Verdict |
|---|---|---|---|---|
| X | hidden `molecule_id`s match the visible ones **and the molecules too** | answer key = visible-test recovered truth; our visible-test output scores **0.9902** against it (393/400 top-1 hits) ⇒ the submission should score ≈0.98 | **0.417** | ❌ eliminated |
| Y | hidden `molecule_id`s match but the **molecules differ** (ID reuse) | the lock force-inserts "the structure computed for a decoy molecule" at rank 1 ⇒ rank 1 always wrong, every correct candidate shifts down one. If unlocked rank-1 hit rate is p≈0.30–0.35, the locked score is ≈ (0.417−p) + 0.5p ≈ **0.25** | **0.417** | ❌ eliminated |
| Z | hidden IDs differ from the visible ones | `_champion_top1[str(mid)]` raises **KeyError**, cell 31 crashes; the `submission.csv` cell 21 already wrote is still graded (our run had status=complete, which is the evidence) | 0.417 | ✅ the only self-consistent one |

⇒ **The hidden set's `molecule_id`s differ from the visible test's; the champion lock never fired;
our effective submission is cell 21's fused output.** The hard-coded/memorised-answer route is
closed for good, and a `nolock` variant would only reproduce the same score (its sole real value
is **removing a crash** so cell 32's final validation can run — worth keeping as a robustness
change, not worth a submission slot).

**Side effect (important)**: because the notebook crashes at cell 31, **cell 32 never runs**, and
what gets graded is cell 21's raw output. So any "remove the lock" variant would incidentally
change the final file (one more dedup/truncation pass), **which would contaminate the ablation** —
hence `claw` and friends all **keep** cell 31 as-is, sharing the baseline's crash path.

## 16. Is `lib_max` trustworthy? (resolved — first read from a notebook, then confirmed from source)

This decided whether `claw` is a live lever at all, since CLAW's entire gate sits inside the
`lib_max < 0.9` branch. A 1.15 GB engine dataset kept dropping the connection, **but it was not
needed**: `ahmedberatozer/casmi26-v4q-inference` (a public notebook) embeds the engine's
similarity kernel.

```python
def entropy_sim(qmz, qp, cmz, cp, tol):
    ...
    if tot <= 0: return 0.0                      # <- empty spectrum returns 0, not 1
    return 1.0 - (2.0 * SAB - SA - SB) / np.log(4.0)   # standard entropy similarity, range [0,1]

def entropy_sim_shift(qmz, qp, cmz, cp, tol, shift):
    a = entropy_sim(...); b = entropy_sim(q, ref+shift)   # modified-cosine idea: same molecule,
    return a if a > b else b                              # different adduct keeps neutral losses
```

**The preprocessing is gentle** (from the same notebook's `CFG`): `INT_FLOOR = 0.002` (keep peaks
above 0.2% of base peak), `MAX_PEAKS = 256`, `INT_POWER = 1.0`, `ENT_WEIGHT = True`. That is, a
spectrum is **not** "cleaned" down to a few peaks.

⇒ **Inference (supported by both the code and the score)**:

1. `entropy_sim` has range `[0,1]` and returns 1.0 **if and only if** the (cleaned, optionally
   shifted) peak distributions are identical; an empty spectrum returns 0. **There is no
   "degenerate return 1.0" branch.**
2. Gentle preprocessing ⇒ different molecules being cleaned into the same spectrum is very
   unlikely.
3. So `lib_max = 1.0` means **the library contains an exact spectrum of that hidden molecule**.
4. If that held for all 400 hidden queries, the answers would be exactly retrievable from train ⇒
   the score should approach 1.0. **Measured: 0.417.**

⇒ **`lib_max` cannot be identically 1.0 on the hidden set; the `lib_max < 0.9` merge gate is open
there.**
⇒ **The PubChem channel fires and CLAW fires — the queued `claw` is a live lever, not a dead end.**

(`merge_force` was kept in the queue as a behavioural cross-check: if the channel was in fact
always open, it would not change the score.)

**And this terminated a piece of work**: the 1.15 GB `casmi26-v4b-models` download was cancelled —
it existed only to read how `engine.run` assembles `info['lib_max']`, and that answer was already
in hand and would no longer change any decision.

### Confirmed from engine source

After obtaining `code/casmi/engine.py`, its assembly is directly readable:

```python
lib_max = np.zeros(nc)                    # nc = number of candidates  <- note: per-candidate
...
    if v > lib_max[ci]: lib_max[ci] = v   # assign each library hit's similarity to its candidate
...
lmax = float(lib_max.max())               # the molecule-level scalar, i.e. info['lib_max']
X[:, col['lib_max']] = lib_max            # also one of the 160 features
```

⇒ **`lib_max` = the best library-spectrum similarity among candidates *inside the window*** (not
the library-wide maximum). So `lib_max ≥ 0.9` means: **some candidate in the window has a library
spectrum with similarity ≥ 0.9 to the query** — the library genuinely knows this compound. This
matches the public gate's design intent exactly.

Combined with the kernel's range/empty-spectrum behaviour and the score contradiction:

> **`lib_max` cannot be identically 1.0 on the hidden set; the merge gate is open there; CLAW
> fires.** This is now **source-confirmed**, not inferred.

## 17. The engine's 160 features (reference material; the modelling line will need it)

`engine.py`'s `FEATURES`, grouped by family (v4b appends the `fe_v4` families on top, totalling
160):

| Family | Features | Meaning |
|---|---|---|
| `lib_*` (10) | `lib_max/mean/top3/nspec/cos/max_rank/gap/grp_max/hit/same_adduct` | library-spectrum evidence |
| `ap_* / tan_* / top_sim*` (16) | `ap, a1, best_tan, top_tan, mean_tan, ap_rank, ap_gap, ap_grp_max, top_sim, ap_np, best_tan_np, top_sim_np, ap_np_gap, ap_sum, n_analog_hi, tan_zero_shift` | **analog propagation** |
| `fz*` (14) | `fz, fz_z, fz_rank, fz_gap, fz_norm, fz_top, fz_softmax, fz_grp_ent, fz_grp_gap, fz_single, fz_single_gap, fz_merged, fz_merged_gap` | FPNet fingerprint score |
| `fp_*` (5) | `fp_cos, fp_tan, fp_ll, fp_ll_gap` | fingerprint geometry / likelihood |
| `el_*` (5) | `el_c, el_n, el_o, el_sum, el_rank` | element counts (from the pool's formulas) |
| `fr_* / sf_*` (16) | `fr_int, fr_int_mean, fr_cnt, fr_top10, fr_strict, fr_rank, fr_gap, fr_z, fr_n, sf_*` | **fragment explanation** (the MetFrag-style term is a *feature* here, not a separate channel) |
| `formula_* / mass_* / n_*` (10) | `formula_share, n_formula, formula_is_top, mass_ppm, mass_ppm_rank, n_heavy, fp_bits, n_cand, n_query, n_pos, n_neg, target_mass` | formula and mass |
| Interactions (6) | `lib_x_fz, ap_x_fz, agree_lib, agree_ap, fr_x_fz, q_npeaks` | cross-channel agreement |
| **`gen_*` (6)** | **`is_gen, gen_parent_sim, gen_steps, gen_parent_tan, n_gen, gen_rank_fz`** | **Class-3 generated candidates** (parent-based edits) |
| `xs_*` (8) | `xs_max, xs_mean, xs_merged, xs_gap, xs_rank, xs_z, xs_top, xs_grp_gap` | cross-encoder |

**Two points worth remembering**:

1. **The generative channel already exists, with 6 features** (`is_gen` etc.) — so "we need
   generation" is already partly implemented in the public solution; the question is not whether
   it exists but how much it contributes.
2. **Fragment explanation (`fr_*`/`sf_*`) already exists as features** — which explains why my
   earlier assessment that "porting v17's standalone fragment term" had limited upside: **the
   same signal is already in the ranker's input**, just in a different place.

## 18. CLAW Promotion Rule (the rank-1 lever we entirely lacked)

Source: `bobthebot369/enveda-casmi-2026-v17-zenith-apex`, whose `CFG.update` says
`VERSION: 'v27-zenith-apex-0417'` — the same release as
`lehau007/casmi26-sota-v27-zenith-apex-0417` (LB 0.417). We **copied** the whole implementation
from v17 rather than re-implementing it.

```python
def promote(base, base_keys, pc, pc_keys, slots, n=25):
    """Confident PubChem proposal leads the usual list; every other entry keeps its slot (CLAW b417)."""
    usual = merge(base, base_keys, pc, pc_keys, slots, n=n)
    if not pc:
        return usual
    top = pc[0]
    return ([top] + [x for x in usual if x != top])[:n]

# the gate (inside the lib_max < LIB_TAU branch)
if USE_PROMOTION and S is not None and float(S) > S_TAU and pop_top >= POP_TAU:   # 6.0 / 5.0
    final = promote(smis, keys, p['pc'], p['pc_keys'], slots)
```

**What the two gate quantities mean** (from v17's `probe_core2` patch):

| Quantity | Definition | Meaning |
|---|---|---|
| `S` | the candidate's `z(f.z) + POP_LAM·pop` **within its own ±10 ppm window** | `> 6.0` = a **6σ outlier** in that window: "no second structure in this window is anywhere near the spectrum" |
| `pop_top` | `log1p(PubChem SID count) + log1p(PMID count)` | `≥ 5` = a compound that genuinely exists and is heavily documented |

**Why it must be gated**: promoting a PubChem candidate from slot 2 to rank 1 gains +0.5 if right
and loses −0.5 if wrong — **EV-neutral at exactly 50% precision**; from slot 4, break-even is 43%.
All the value comes from acting **only on extreme outliers**.

**What we were missing was two halves of code, not a constant**:
- V44's PubChem probe is an **unpatched v16** and does not produce `S`/`top_pop`/`fz_top`
- the runner discards those three quantities (`res[mid] = dict(pc=..., pc_keys=..., **extra[mid])`)
- cell 19's merge has no promotion branch
- yet `casmi26-pubchem-popularity-prior` **was already mounted** and `CASMI_POP_DIR/LAM/UNION`
  **were already in the environment** — with no code reading them

Port self-check: `python scripts/test_claw_port.py` (extracts `merge`/`promote` **from the
generated notebook** and executes them, asserting slot semantics and gate structure).

**⚠️ Unknown that must be settled from logs**: the whole gate sits **inside** the
`lib_max < 0.9` branch. The public-data run log says `merge stats {'untouched': 400}` — the
channel is gated off entirely, and V44's code comment even says *"V43's lib_max was 1.0 for every
hidden query"*. If the hidden set behaves the same, **CLAW never fires**. So every run's log must
be checked for the `promoted / aggressive / gentle / untouched` counts. (Superseded by §16, which
settles it from source: the gate is open on hidden data.)

---

# Part IV — Measurement

## 19. Pre-registration: predictions and falsification conditions, written before results arrive

| Variant | Prediction | Basis | If the result is zero | **What counts as falsification** |
|---|---|---|---|---|
| `claw` | **+0.000 ~ +0.005** | the public ledger claims +0.024~0.043; but v27 (with CLAW) and V44 (without) both score 0.417 | the gate is too strict (`S>6 ∧ pop≥5` rarely fires) — **not** evidence that CLAW is worthless | if `promoted` ≫ 0 while the score does not move ⇒ **the CLAW mechanism itself is inert** (that would be the negative evidence) |
| `icefull` | **0.000 ± 0.002** | v17/v27 get 0.417 with ICE 5400; we get 0.417 with 300 ⇒ ICE coverage does not explain the difference | ICE contributes nothing material to the tail | if the score **drops** ≥0.002 ⇒ ICE reranking is harmfully perturbing the tail |
| `pop30` | **+0.000 ~ +0.004** | the author lists "+0.002 per 0.15", i.e. +0.003~0.005 | the prior has no discriminative power on real candidates | if the score drops ⇒ pool popularity anti-correlates with being the answer (unlikely) |

**The key methodological point**: `claw`'s zero result has **two completely different meanings**,
distinguishable only by diagnostics:

```
promoted == 0             => gate too strict, CLAW was never tested -> sweep S_TAU before judging
promoted >> 0, score flat => CLAW really is inert                  -> retire the line entirely
```

### ⚠️ Correction found by self-audit before results arrived: `promoted` is **necessarily 0** in a batch run

I had planned to read `promoted` from the batch run's log. **That was wrong**:

- the batch run uses the **public test**, and all 1,213 of its spectra are verbatim from train
  ⇒ every molecule's `lib_max ≈ 1.0` (the original V44 public run measured
  `merge stats {'untouched': 400}`);
- `promote()` and the entire merge branch sit **inside the `lib_max < LIB_TAU(0.9)` else-branch**;
- ⇒ on public data that branch **never executes**, so `promoted` is identically 0 **regardless of
  whether CLAW works**.

So of the diagnostic chain, **only the first item is valid**:

| Diagnostic | Valid? | Why |
|---|---|---|
| `V45 pop-align: pc_lsid=N pc_mass=M OK\|MISALIGNED` | ✅ | it lives in `init_worker`, **independent of the gate**, so it always runs |
| `merge stats`' `promoted / aggressive / gentle` | ❌ | the gate is always "skip" on public data, so they are necessarily all zero |

**Consequence**: if `claw` scores 0.417, we **cannot** distinguish "gate too strict (never tested)"
from "mechanism inert" from the log. That distinction needs another variant: **`claw_force`
(already built) bypasses the `lib_max` gate**, so the merge/promotion branch executes on public
data too and produces non-zero `aggressive/gentle/promoted` in the batch log — that is **wiring
verification**, not effect verification.

**This does not affect `claw`'s score itself**: the gate is open on the hidden rerun (§16,
source-confirmed), so the score remains valid; only the **interpretability of a zero** is affected.

### ✅ Remedy: use the submission **byte count** to tell whether the merge branch actually ran

We **cannot download** the hidden rerun's submission file (`submissions.download` returns 403),
but **the submission record carries `totalBytes`**. And ctl's hidden byte count is known — it is
the author's 0.417 run: **530,716 bytes**.

That gives a free binary discriminator:

| `claw`'s `totalBytes` | Conclusion |
|---|---|
| **≈ 530,716** (matching) | the merge branch **almost certainly never ran** ⇒ `lib_max ≥ 0.9` holds on the hidden set too ⇒ **§16's reasoning is wrong**, the PubChem channel is dead weight for everyone, and the whole CLAW line is void |
| **clearly different** | the branch did run and the output did change ⇒ if the score is still 0.417, *that* is the actionable conclusion "the change took effect but did not help" |

**Why it works**: the patch changes the tier candidates' **ordering** (`z(f.z)+0.25·popularity`)
and the pass-1 union, and the merge branch places those 5 candidates into slots — different
candidates ⇒ different SMILES lengths ⇒ different byte count. The gate always skips on public
data, so **the batch run cannot show this signal; only the hidden submission's `totalBytes` can**.

**What it cannot tell us**: how many molecules were promoted (the exact `promoted` value is still
unavailable). But it turns the most important binary question — **was the gate open?** — from
"undecidable" into "read one number".

⇒ **Once `claw`'s result arrives, the first thing to do is compare `totalBytes`, not look at the
score.**

### The discriminator's two preconditions (written down to avoid over-reading later)

**Precondition 1: the "unchanged" reference byte count has to be borrowed.** We **never submitted
ctl** (cancelled), so the reference is the author's 0.417 run at **530,716 bytes**. Strictly, I
cannot prove that `scriptVersionId=355245043` is **character-identical** to the content I pulled
(that ID cannot be pulled directly). ⇒ the reference carries **undigested uncertainty**.

**Precondition 2: natural variation has a second data point.** The gengsr reproduction (giaok246,
**a different notebook**) produced **532,444 bytes** on the same hidden set. Two different
implementations differ by **1,728 bytes**.

⇒ So the reading cannot be "any difference means the gate was open":

| `claw`'s `totalBytes` | Reading |
|---|---|
| within **530–533 KB** | indistinguishable from "unchanged" ⇒ the gate most likely did not open |
| **clearly outside** that band (say ±10 KB) | the branch really ran ⇒ only then does the score mean "took effect but did not help" |

Order-of-magnitude check: if the merge branch fires for most molecules, each replaces up to 5
tier candidates (400 molecules ⇒ up to ~2,000 candidates swapped, ~50 bytes each) ⇒ **~100 KB**,
far above the 1.7 KB natural variation. **So the discriminator is mechanically sensitive enough —
precondition 1 is its weak point.**

**Lesson (same family as the three silent bugs)**: in pre-registration I treated "the diagnostic
distinguishes the two meanings" as a given, **without checking whether that diagnostic could even
take a non-zero value on the batch run's data**. A diagnostic must itself be validated —
otherwise it merely makes the result look interpretable.

This is exactly why I added the `stats['promoted']` counter and the `pop-align` hook to `claw` —
**without them the experiment could not reach a conclusion whatever the outcome**. And whether
`promoted` can be read at all depends on "batch-run logs are harvestable" (now covered by
automation).

Likewise, `icefull`'s and `pop30`'s zero results each have two meanings (genuinely no effect vs
silently broken), and both diagnostics are in their run logs (`ICE meta` /
`V45 POOL_POP prior DISABLED`).

**Conclusion: the design goal of these three experiments is not to raise the hit rate but to
guarantee a conclusion whatever the outcome.**

### ⚠️ Correction: `claw` is not a clean single-variable experiment (diff verified line by line)

Using `scripts/diff_variant.py` to compare `notebooks/v45/claw` against `notebooks/v44_base`:
**3 of 33 cells changed (cells 3 / 9 / 19), all code cells; 30 are byte-identical.** The
architectural spine (v4b engine + `fpnet_full1`, dual-ranker engine + PairTail, ICEBERG/GLACIER,
RRF, the champion lock, the 25-slot structure, all 14 datasets) is **untouched**.

But the three changes **all act on the same branch**, so a result cannot be attributed to CLAW
itself:

| # | Change | Effect |
|---|---|---|
| 1 | the v17 popularity patch | **changes the tier candidate list** (`z(f.z)+0.25·popularity` ordering + a popularity-expanded pass-1 union) |
| 2 | `rel` from `pc_fz[0]` to `fz_top` | the **aggressive/gentle slot choice can flip** (necessary — the patch changed `pc_fz`'s meaning — but it is a second behavioural change) |
| 3 | the `promote()` gate | promotes a tier candidate to rank 1 at high confidence |

⇒ **Even if the score moves, we cannot say "CLAW works".** My pre-registration treated `claw` as
"single-variable + instrumented"; that assumption was wrong (the instrumentation half was
corrected above).

**For a clean single variable**, the correct order is to apply only the patch first (candidate
list changes, rank 1 does not), then layer the promotion on top. **That needs a new variant, which
the "variant creation frozen" decision has paused** until `claw`'s result is in.

**Also note**: **on the public-data batch run, `claw`'s output should be byte-identical to ctl's**
(the gate always skips), so that run verifies "the plumbing works" but **cannot** verify the patch
or CLAW.

## 20. Method A: the leak-free backtest

**Motivation**: the real bottleneck is not a shortage of ideas but that we can measure **6 things
a week, each as a three-decimal scalar**. Any method needing a threshold sweep (CLAW's gate, tier
admission) is impossible at that throughput. So build measurement capability first, then talk
about methods.

**Step one, done**: `scripts/build_leakfree_folds.py` builds **structure-disjoint,
source-stratified** folds from `train.parquet` (`data/processed/folds.parquet`) with three
asserted invariants (each structure in exactly one fold; folds disjoint at the **inchikey14**
level rather than the row level; every fold carries every stratum):

```
rows 2,539,608   unique structures 275,810   fold sizes 55,161–55,163
regime: syn 228,179 | other 47,381 | np 250
```

### ✅ Feasibility confirmed: the engine ships leak-hiding hooks, so **no pool or library rebuild is needed**

I had assumed A's main cost was "removing the held-out molecule's structure and spectra from the
pool and library" — which would mean rebuilding the structures + spectra + fragments +
fingerprints quartet. **Reading `engine.py` showed it is unnecessary**: the author built the
simulation hooks in.

```python
def run(self, spectra, target, exclude=None, exclude_sid=-1, exclude_lib=-1, drop_pid=-1):
    """...
    exclude: boolean mask over library spectra to hide (simulation); exclude_sid/exclude_lib:
    representatives to hide for the analog channel; drop_pid: remove this pool entry (class-3 sim).
    """
```

| Hook | Effect | Leak it closes |
|---|---|---|
| `exclude` | boolean mask **over library spectra** (set the structure's own rows True among 2.54M) | the library-retrieval channel cannot see the answer |
| `exclude_sid` / `exclude_lib` | hide that structure's representative from the **analog channel** | analog propagation cannot see the answer |
| `drop_pid` | remove the entry from the pool window (author's comment: *class-3 simulation: the hidden truth may be generated*) | the pool has no answer, but **the generative channel can still produce it** |

⇒ **Each of the three leak surfaces (library / analog / pool) has a dedicated switch, and it is a
per-call argument, not a dataset to rebuild. A's cost drops from "rebuild assets" to "change one
call".**

**This also explains an old item**: the team's earlier V_A_hard (0.0476, judged "impossible")
**rebuilt the library by hand** to remove answers, which is not the same as the engine's built-in
simulation path. The author clearly anticipated the need and our early implementation bypassed it.
It also means this document's earlier line "V_A_hard is impossible" should be retired: **the
correct method was in the engine all along.**

### A's landing design (cost now clear)

The backtest kernel is the V44 notebook with three changes:

1. **`COMP` points at a synthesised "test set"**: take the structures of one fold with
   `regime == 'other'` from `folds.parquet` and assemble a `test.parquet` from their spectra in
   `train.parquet` (column names matching the competition);
2. **each query calls `E.run(..., exclude=mask, exclude_sid=sid, exclude_lib=lib, drop_pid=pid)`**,
   closing all three leak surfaces at once;
3. **compute MRR@25 at the end** (rank of the first candidate whose key equals the held-out
   structure's key), broken down by fold and by source stratum — the same diagnostic convention as
   `ranker.py`'s `mrr_at(k=25)`.

### ⚖️ Component 1 done, and it exposed a conflict that must be resolved: **statistics vs runtime**

`scripts/build_backtest_queries.py` built the query set (fold 0 / regime `other`,
`data/processed/backtest_fold0_other.parquet`): **2,000 molecules / 16,311 spectra**, with columns
**cast one by one to `test.parquet`'s reference schema** (not hand-written names), asserting
"every held-out structure has spectra; no structure was lost to peak cleaning". The truth table is
stored separately.

**But computing runtime against resolution shows they conflict**:

| Queries | vs the real run (400) | Estimated runtime* | Paired s.e. (10% of queries affected, \|Δ\|≈0.3) |
|---:|---:|---:|---:|
| 400 | 1× | ~2 h | ±0.0063 |
| 2,000 | 5× | **~10 h ✗ over the cap** | ±0.0028 |
| 9,500 (whole fold) | 24× | infeasible | **±0.00097** |

\* the engine's channel stage is about 2,012 s for 400 molecules; ICE/GL budgets are fixed but
feature computation grows linearly with molecules.

⇒ **Reaching ±0.001 needs ~9,500 queries, and 9,500 queries cannot finish.** That is A's real
constraint — not insufficient sample size, but **insufficient compute at the sample size that
would suffice**.

### ✅ Wiring verified: the three hooks' index spaces (read line by line in `library.py` / `engine.py`)

The easiest thing to get wrong in the backtest is **which index space each hook lives in**; a
mistake does not raise, it just silently fails to exclude:

| Hook | Index space | Length / source | Rebuilt per call? |
|---|---|---|---|
| `exclude` | **library spectra** (not structures, not representatives) | `len(L.sid)` = 2,539,608 | ❌ one boolean array per call, reusable by editing a few entries |
| `exclude_sid` / `exclude_lib` | **structure id** + **library ordinal** | values of `L.sid` / `L.lib` | ❌ two integers |
| `drop_pid` | **candidate pool** entry | `struct_key[sid]` → index into `pool.key` | ❌ one integer |

**Two implementation facts that decide whether it is cheap**:

1. `Library.window(..., exclude, ...)` does `c = c[~exclude[c]]` where `c` comes from
   `self.order`, i.e. **library-spectrum indices** ⇒ `exclude` is a boolean array of length
   2,539,608. **Mistaking it for a per-structure mask (275,810) raises a length mismatch** —
   fortunately that one does not fail silently.
2. `Engine.analogs` uses `rep = self.rep` — **the representative set is built once at engine
   construction and cached**; each call only does `keep &= ~exclude[ridx]` on the windowed slice
   ⇒ **per-query exclusion does not trigger a rebuild**. If it rebuilt the representative set per
   query (a 2.5M-row lexsort), 9,500 queries could never finish.

⇒ **Per-query exclusion is computationally cheap; the Tier 1 plan stands at the implementation
level.**

### Calibration design for component 4 (fixed **before** running, or "the backtest is trustworthy" becomes a post-hoc judgement)

A's value rests entirely on one thing: **can it rank configurations correctly?** So before using it
to screen any policy, it must be calibrated against configurations with **known magnitude
differences**. Three configurations are built (`--degrade`):

| Configuration | Construction | Expected | Role |
|---|---|---|---|
| `bt` | V44 as-is | baseline | reference |
| `bt_top1` | candidate list truncated to **1** | **large drop** | **structural check**: with only rank 1 left, MRR should collapse to top-1 accuracy |
| `bt_no_ice` | `ICE_LAM=0, ICE_BUDGET=0, ICE_PC=False` | slight drop | **magnitude check**: the public ledger records the ICE/GL weights as worth +0.007 |

**Falsification conditions (fixed)**:

```
if bt_top1 >= bt            => the backtest cannot even detect "one candidate" => A is void immediately
if bt_no_ice > bt           => the forward models contribute negatively, contradicting the public
                               ledger => the backtest is untrustworthy
if bt_no_ice ≈ bt (±0.002)  => undecidable => treat as "failed magnitude calibration";
                               A degrades to structural-only use
```

**A weakness that must be admitted**: the three configurations are **three separate notebooks**, so
they need three Kaggle runs, and **cross-run comparisons are unpaired** ⇒ a single 400-query run
has s.e. ≈ 0.0063, **larger than the 0.007 `no_ice` is looking for**. So `no_ice` can only be a
**sign check**, not a magnitude check.

**Comparisons between policies (A's actual purpose) are paired within a single notebook** — they
share one engine run and differ only in post-processing ⇒ paired s.e. ≈ ±0.001. **This is exactly
where A's use differs from its calibration, and the two must not be conflated.**

⇒ Component 4's criterion: **`bt_top1` must be clearly below `bt` (structural), and `no_ice` must
not be positive (sign).** If both hold, A can do paired policy comparisons; if not, stop.

### ⚠️ A class of leakage A cannot remove: **model-level leakage** (must appear in any conclusion)

The engine's three data surfaces (library / analog / pool) can be closed with `exclude` /
`exclude_sid` / `drop_pid`, **but the public models were trained on train, and our held-out
structures are in train too**:

| Component | Has seen the held-out structure? | Consequence |
|---|---|---|
| FPNet (v4b / v3's A+B) | **yes** | it may "recognise" the held-out structure ⇒ optimistic score |
| ranker (160 features × 4 GBM) | **yes** (training rows come from train) | same |
| ICEBERG / GLACIER | **yes** (forward models trained on train) | same; **Tier 1 disables them, which mitigates it** |
| PubChem tier channel | indirectly (the structure may be in the tier) | the tier has no spectra, so spectral retrieval is impossible ⇒ weak leakage |

**⇒ We cannot "delete" the held-out structure from the public models** (that would need retraining,
which in turn needs the author's `split.parquet`, absent from the public pack). Therefore:

> **A's absolute MRR is optimistic and must not be read as "what our pipeline scores on genuinely
> novel molecules".**

**How much this affects its use**:
- **Paired policy comparisons are affected less** — two policies share the same (equally leaked)
  models, so the difference comes mainly from the post-processing rule itself;
- but **if a policy specifically exploits "remembered" information** (e.g. CLAW promoting a
  PubChem candidate to rank 1, or any gate relying on model confidence), leakage makes it look
  better than it is.

**Handling**: treat A as a **ranking tool** (which policy is better), not a **level tool** (what
score we can get). Any policy that wins in A **must still be confirmed on the LB** — which is
precisely why the tiered reading rules exist.

### ✅ Fold definitions now aligned (there were once "two different fold 0"s)

Self-audit found `build_leakfree_folds.py` used `groupby(..., sort=False)` (first-appearance
order) while the backtest kernel inside a notebook can only use the default `sort=True` ⇒ **two
"fold 0"s of identical size (both 9,477) with different membership**, and no error.

⇒ The fold builder was switched to the default ordering, `folds.parquet` regenerated, and **all 15
(fold × stratum) cells checked one by one**:

```
all 15 fold x regime cells identical: True
```

**Why alignment is mandatory**: otherwise the "fold 0" written in the docs and the "fold 0" the
backtest actually uses are different sets, and any cross-experiment comparison silently
misaligns — while both sides have the same size, so nothing looks wrong in the numbers.

### ✅ Synthetic `COMP` coverage audited (not eyeballed)

Every place the notebook reads a file from `COMP` was listed and checked against the synthetic
directory:

| Read site | File | Present in the synthetic dir? |
|---|---|---|
| cell 5 | `train.parquet` | ✅ symlinked to the competition data |
| cell 5 | `test.parquet` (reading the real one for column names) | ✅ `COMP` has not been redirected yet at that point |
| cell 7 | **`sample_submission.csv`** | ✅ synthesised — **the easiest one to miss**: cell 7 reads it *after* `COMP=SM`, and without it the run dies an hour in |

The remaining `/kaggle/input/**` globs look for datasets (the rdkit wheel, `fpnet_full1`,
`eng_runner`'s `ROOTS`), not `COMP`, so they are unaffected.

**Also audited: does the replaced cell 5 leave any dangling global?** The original defined
`_sig / _te / _full / _keep / SMOKE_N / IS_RERUN / ICE_BUDGET`; my replacement keeps only the last
two. Checking each for use by later cells ⇒ **no dangling references** (`ICE_BUDGET` is used only
in cell 17, and it is kept).

⚠️ **My checker first raised a false alarm**: `ICE_BUDGET` is assigned by **tuple unpacking**
(`IS_RERUN, ICE_BUDGET = True, 300`), and my first version collected only `ast.Name` targets,
missing `ast.Tuple`, so it reported "defined" as "dangling". **Only after fixing the checker did
the conclusion above hold** — the same lesson again: **the tool used for verification must itself
be verified**, or time gets spent "fixing" a non-problem.

### 🔴 Correction 4: Tier 1's "9,500 queries" **cannot finish** — I had only counted the channel stage

The table above wrote Tier 1 as "whole fold, 9,500 queries ⇒ ±0.001". **That was wrong**: I
extrapolated from the **channel stage** alone (`pubchem channel: 2,012 s / 400 molecules`) and
forgot the channel is only part of the pipeline. Recomputing from **measured end-to-end wall
clock**:

| Quantity | Value | Basis |
|---|---|---|
| Wall clock of one 400-molecule run | **~7,200 s (2 h)** | the observed range for this repo's `ctl` batch runs (2.0–2.8 h) |
| Per molecule | **~18 s** | 7,200 / 400 |
| Molecules inside a 9 h cap | **≈1,800** | 32,400 / 18 |

⇒ **The whole fold (9,500) needs about 47 hours, more than 5× over the cap.** Turning ICE/GL off
saves about 900 s per run (300 s ICE + 600 s GL), i.e. ~15.7 s/molecule ⇒ a ceiling near **2,060**
molecules.

**Corrected feasible configurations**:

| Tier | Configuration | Queries | Paired s.e. (f=10%) | Note |
|---|---|---|---|---|
| Tier 1 | ICE/GL off | **~2,000** | **±0.0021** | ceiling ~2,060; 2,000 leaves margin |
| Tier 2 | ICE/GL on | **~1,200** | ±0.0027 | ceiling ~1,800; 1,200 leaves margin |

⇒ **A's real resolution is ±0.002, not the ±0.001 I claimed.** That is the same order as the LB's
three decimals, **but no longer better**. It is still far better than "one policy per 5 hours" (a
single run can compare several policies in a paired way); I simply have to lower the expectation
from "better than the LB" to "comparable to the LB".

**Cause of the error (same family as before)**: I extrapolated the **whole pipeline's** capacity
from a **single stage's** duration instead of using the measured end-to-end wall clock. **A
component's cost is not the system's cost.**

### 🔴 Correction 5: a larger query set **silently starves ICE/GL** — the tiers cannot be split by on/off alone

Continuing the arithmetic exposed a subtler problem: **`ICE_BUDGET` / `GL_BUDGET` are fixed
wall-clock caps that do not scale with molecule count.**

| Queries | ICE budget | Actual coverage | vs the real run (400 molecules / 71 covered) |
|---:|---:|---|---|
| 400 | 300 s | ~71 molecules | **identical** ✅ |
| 2,000 | 300 s | still only ~71 | **coverage drops to 1/5** ❌ it measures a different pipeline |

⇒ **Whenever ICE/GL are on, the query count must stay at 400**; otherwise I am not measuring "the
same system on more molecules" but "a system whose forward models are starved". **This error does
not raise — it just makes the baseline number uninterpretable.**

**Scaling the budgets proportionally does not rescue it either**: 2,000 queries would need
`ICE_BUDGET≈1,500` + `GL_BUDGET≈20,000` to hold coverage ⇒ those two alone are 6 h, and with the
rest about 12 h, **over the 9 h cap**.

**⇒ Faithfulness and statistical power cannot both be had.** So the tiers are not "off/on" but:

| Purpose | Configuration | Queries | Paired s.e. | Why |
|---|---|---|---|---|
| **Calibration** (component 4) | `bt` / `bt_top1` / `bt_no_ice` | **400** | ±0.0047 | must match the real run's stage behaviour, or the calibration is meaningless |
| **Tier 1 policy screening** (the workhorse) | `--degrade no_ice` | **2,000** | **±0.0021** | ICE/GL are **off by construction**, so no starvation |
| **Tier 2** (ICE/GL-dependent policies) | `bt` | **400** | ±0.0047 | can only afford 400 queries; weak power but faithful behaviour |

**The default `--limit` stays at 400**; Tier 1 is requested explicitly with
`--limit 2000 --degrade no_ice`.

---

# Part V — The modelling line

The largest single lever in the public ledger is **a better FPNet (+0.013)**, and that is precisely
"something absent from public code" — `fpnet_full1` is already the public best, so going further
means training our own.

**The quantified business case (this is the decisive part)**: pulling the public ranker training
features `prvsiyan/casmi26-ranker-features` (CC0, 9.6 MB) and measuring them:

```
rank_train.npz: X (142,762 × 31) float32 | Y {0,1} mean=0.0115 (1,638 positives)
                M {0,1} | G groups 0..818  ->  only 819 query groups, ~174 candidates/group
```

**The entire second engine's ranker was trained on 819 queries**, with only ~2 positives per group.
We hold **275,810 structures / 2,539,608 spectra** (and can use all of train's labels for free).

⇒ **Going from 819 queries to tens of thousands is the most concrete original space on this line.**
It needs no new data and no new architecture — only getting the training-row **generation** right
(remove the held-out molecule's structure and spectra from both library and pool, run the engine to
produce candidates and features, label by "is this candidate the held-out structure"). That is
exactly what the V_A_hard fold in `docs/EXPERIMENTS.md` tried to do, except it was used for
evaluation rather than training.

### The training recipe, now readable

`casmi26-v3-models` (411.9 MB) downloaded successfully, unpacking `code/casmi/` into 13 modules
including **`engine.py` (26.4 KB)**, **`ranker.py`**, **`traindata.py`**, `library.py`, `pool.py`,
`fpnet.py`, `frag.py`, `spectra.py`, `derive.py`, plus two 189.6 MB `fpnet_*.pt` files and
`ranker_0.pkl` (13.5 MB).

**① The ranker (`ranker.py`) — our metric is its metric**

```python
DEFAULT_PARAMS = dict(objective='lambdarank', learning_rate=0.03, num_leaves=63,
    min_data_in_leaf=40, feature_fraction=0.7, bagging_fraction=0.8, lambda_l2=1.0,
    lambdarank_truncation_level=30, label_gain=[0,1])   # rounds=600, seeds=(0,1,2,3) -> 4 boosters
def mrr_at(pred, y, g, k=25):   # per-group reciprocal rank -- identical to MRR@25
def cv_report(...):             # "Structure-grouped CV ... MRRs per (regime, fold)
                                #  plus a calibrated LB estimate"
```

⇒ The author uses **structure-grouped** CV, reports MRR by **`regime`**, and **has already
calibrated local CV to the LB score**. The `regime`/`fold` values come from `traindata.py`'s
`holdout_folds=('np','nplib','syn','plusk','twin')` — i.e. **stratified by source** (natural
products / NP libraries / synthetic / pluskal_ms2). **If the hidden set is dominated by the `np`
stratum, training weights should lean that way.**

**② FPNet (`traindata.py`) — the training objective is "rank first inside the window"**

```python
# positive = the true structure's index in the pool; negatives = K=63 sampled from the same ±10 ppm window
def negatives(self, pos): ...
# when the window is small (<8), mix in random decoys "so the softmax is not trivial"
def window_mrr(model, D, ...):  # "MRR@K of the true structure inside its ±10ppm pool window"
# augmentation: peak dropout, intensity jitter, ppm jitter (aug=True)
```

⇒ **Both FPNet and the ranker are optimised and evaluated only under "the truth is inside the
±10 ppm window".** This matches the §9 decomposition exactly: **these two models own the ranking
half; the recall half (the truth not being in the pool) was never their objective.**

**③ The row-generation code is not in the pack** — `code/` holds only inference-side modules;
`rank_train.npz` / `sim_rank_rows_*.npz` are pre-generated artifacts. So **we must write the row
generator ourselves**, but the recipe is now clear:

```
for each held-out molecule (held out by structure, using their fold scheme):
    C, X, F, names, info = V.run(spectra, target)     # the engine returns the 160 features directly
    label[c] = 1 if C.key[c] == the held-out structure's key else 0
=> (X, y, group=molecule) trained with DEFAULT_PARAMS as LambdaRank, evaluated with mrr_at
```

**Key judgement**: this route's ceiling depends on **how much "in-window ranking" can still
improve** — and the public ranker already trained 600 rounds × 4 seeds × 63 leaves on exactly that
objective. So **more queries will not necessarily improve in-window ranking**; the real space is
more likely in §9's **recall half** (getting the truth into the window), which depends on
FPNet/pool/analog, not on the ranker. ⇒ **The modelling line's first cut should be on the recall
side, not the ranker side.**

### The first project: **unify the two candidate universes**

The pipeline actually contains **two mutually disconnected structure universes** with grossly
unequal capability:

| | Pool universe | Tier universe (PubChem) |
|---|---|---|
| Scale | 710k–775k | ~100M (7.2 GB, 150–1250 Da) |
| Scorer | **v4b ranker: 160 features × 4 GBM × 600 rounds** | **only `f.z` (fingerprint dot product) + popularity** |
| Path into the submission | the bulk of the 25 slots | only 5 tail slots (or a CLAW promotion to rank 1) |
| Was a model trained for it? | yes (`mrr_at` is literally this task's metric) | **no** |

**This is the direct source of §9's 45% recall gap**: molecules outside the pool depend on a
channel for which **no model was ever trained**, and which is confined to 5 tail slots.

⇒ **Project definition**: extend the ranker's feature extraction to tier candidates (the feasible
subset: library similarity, FPNet `f.z`, formula match, popularity, **ICE/GL forward scores** —
ICE/GL can score any SMILES), then **train a unified ranker over the union of both universes**, so
tier candidates compete on evidence rather than occupying fixed slots.

**Why this may be the only direction with real position**:
- it is on the **recall** side (~0.19 of the score), not the ranking side;
- it targets a channel **the public models never optimised** (both public models are trained only
  under "the truth is inside the pool window");
- pool expansion is closed by measurement (0.051%), and the tier is the only structure source left
  that can expand.

**Cost and risk (honest estimate)**:
- it needs Kaggle runs for "held-out molecule → tier probe → features → labels" row generation,
  **each consuming weekly quota**;
- feature extraction requires modifying engine-side code — weeks of engineering;
- **ceiling unknown**: if most of the missing 45% are novel analogs present in *no* public library
  (§10 cause C), this project cannot save them either — the only route then is analog/generative
  propagation.

**So its status is "a large bet awaiting validation", not "the next thing to do".** Let the current
cheap variants return first: if cheap versions of "let the tier take more positions" (`claw`,
`pc_tail10`) work, then the tier really does contain answers and this project is worth the
investment.

**Discipline unchanged**: every modelling-line step must also be verifiable on the LB; until there
is a reproducible ≥ +0.002 shift, it is a hypothesis like any other.

---

# Part VI — Reference

## 21. Tools

| Script | Purpose |
|---|---|
| `scripts/build_variant_kernel.py` | build a variant from `notebooks/v44_base/notebook.ipynb`; every replacement must hit exactly once |
| `scripts/push_kernel.py` | push = save a version and trigger the batch run |
| `scripts/wait_submit_report.py` | wait for COMPLETE → verify the output file exists → submit → poll the score |
| `scripts/queue_variant.py` | queue a variant: wait for a free GPU session → push → wait → submit → report |
| `scripts/check_variant_kernels.py` | compile every variant cell, reassemble + compile CLAW's CORE, and validate kernel metadata |
| `scripts/diff_variant.py` | cell-by-cell, line-by-line diff of any variant against `notebooks/v44_base` |
| `scripts/recover_test_truth.py` | recover **visible**-test truth (regression detection only; unrelated to the LB) |
| `scripts/score_submission.py` | score any submission with the grader's tautomer InChIKey14 → MRR@25 + rank histogram |
| `scripts/build_leakfree_folds.py` | build the structure-disjoint, source-stratified folds |
| `scripts/build_backtest_queries.py` | build the backtest query set in the competition's exact schema |
| `scripts/build_backtest_kernel.py` | build the backtest kernel and its calibration configurations |

## 22. What not to do

- Do not tune on statistics from the visible test — it is not the evaluation set.
- Do not push several versions of the same kernel concurrently and then read `kernels/status`:
  that endpoint is **per kernel** (latest version), so v1 and v2 misreport each other. One kernel
  slug per variant is the only safe pattern.
- Do not treat the champion dict as an "answer" to extend — it is the product of another public
  reading, and its keys are the visible test's.
