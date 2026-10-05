# Leaderboard record

> LB noise is roughly ±0.006 across models, but **re-scoring the same submitted file is
> deterministic**, so deltas measured *within* one experiment are trustworthy. See the
> tiered reading rules in [PLAYBOOK.md](PLAYBOOK.md) §9.

## Current standing (measured 2026-10-05, not relayed)

| Item | Value |
|---|---|
| Team | **Spectral_Forge** (accounts: nicholasnicklee / xiaoyuzhoux120 / giaok246) |
| Our best | **0.417** (rank **111**) |
| Rank-100 cut | **0.417** (about 61 teams tied on this score) |
| Rank 59 | **0.418** ← the real bar to cross |
| Rank 1 | 0.471 |
| Leaderboard size | ≥800 teams |

Public score distribution: 0.471 / 0.466 / 0.441 / 0.440 / 0.435 … 0.418 (rank 59) →
**0.417 (ranks 60–120, 61 teams tied)** → 0.413 → 0.408 → 0.397 → 0.362.

## All of our submissions (23, by score)

| ref | Score | Notes |
|---|---|---|
| **56839982** | **0.417** | **Exact reproduction of V44 PairTail Locked Top1** (`xiaoyuzhoux120/casmi26-v44-pairtail-locked-top1`) — current best |
| 56835199 | 0.413 | gengsr V3 reproduction (`giaok246`) — matches its author's reported 0.413 **digit for digit** ⇒ reproduction here is deterministic |
| 56818216 | 0.132 | NP candidate pool 730k + formula mass |
| 56815919 | 0.176 | V1 B0 inference (the account's pre-existing generative line) |
| 56813793 | 0.130 | Retrieval + analog, full NP library (untruncated spectra) |
| 56813791 | 0.096 | Retrieval + analog, compact library (3 spectra per compound) |
| 56787177 … 56667967 | 0.169–0.176 | The account's pre-existing generative pipeline (10 submissions) |
| 56606429, 56535378 | 0.162, 0.176 | Early retrieval / COCONUT analog |
| 56534532, 56533793 | 0.141, 0.140 | Earliest mass-filtered retrieval baselines |

## External reference points (2026-10-05)

| Approach | Score |
|---|---|
| Public LB rank 1 | 0.471 |
| Best public notebook (`lehau007/…v27-zenith-apex-0417`) | **0.417** (= where we are) |
| Ours (V44 reproduction) | 0.417 |
| Best of the account's pre-existing generative pipeline | 0.176 |
| Community pure-retrieval reference | 0.143 |

> **All 250 public notebooks were scanned: none advertises ≥ 0.418.**
> ⇒ The 59 teams between 0.418 and 0.471 have released **no public code**; copying public
> work cannot cross the line.

## Three conclusions drawn from this

1. **0.417 is not "our level", it is "the public design space's level".** v17/v27 (with CLAW,
   with the popularity patch, ICE budget 5400) and our V44 (without CLAW, without the patch,
   ICE budget 300) **score exactly the same** — two configurations that differ this much
   landing on one score means the space is saturated there.
2. **Crossing needs only +0.001–0.002** (rank 59 = 0.418). By the decomposition in
   [PLAYBOOK.md](PLAYBOOK.md) §8.6, that is roughly **one molecule moving to rank 1**
   (+0.0025), or four molecules moving from rank 5 to rank 3.
3. **But "easy to cross" and "easy to be fooled by noise" are two sides of one fact.** At
   this effect size, results must be read with the tiered rule (below 0.001 no signal;
   0.001–0.003 must be replicated; above 0.003 may be believed first).

## Current experiment ledger (in progress)

| Variant | Kernel slug | ref | LB score | Pre-registered prediction |
|---|---|---|---|---|
| `claw` | `…-v45-claw` | pending | pending | +0.000 … +0.005 |
| `icefull` | `…-v45-icefull` | held | — | 0.000 ± 0.002 |
| `pop30` | `…-v45-pop30` | held | — | +0.000 … +0.004 |

Diagnostics that decide what a zero result means: the `promoted` counter, `V45 pop-align`,
`V45 POOL_POP loaded|DISABLED`, `ICE meta` — and, for the hidden run, the submission byte
count. See [PLAYBOOK.md](PLAYBOOK.md) §9.
