I dug into the current CASMI discussion landscape, the public solution lineage, and the V44 notebook code. There is a very important conclusion:

The 0.41 systems are not winning because they suddenly found a magical molecule generator. They are winning because they built a very aggressive candidate-union + rank-fusion + forward-fragmentation verification + protection pipeline.

And that changes what I would recommend for your team.

One caveat first: Kaggle is blocking direct retrieval of the exact page you singled out, discussion 745029, in this environment, and the exact gengsr/casmi26-fusion-glacier-mh-lb-0-413 notebook page is likewise not directly retrievable. I therefore won’t pretend I saw those pages verbatim. I was able to inspect the surrounding Kaggle discussions and a public GitHub mirror of the V44 notebook, plus the public code lineage behind the ~0.41 fusion solution.

⸻

1. What the Kaggle discussions are telling us

The most useful discussion I could access is “Lessons from ~30 submissions: where the MRR actually goes”. Its findings line up almost perfectly with what we’re seeing in the 0.4+ solutions. 

The important points are:

A. Stop treating precursor mass as the main problem

Their analysis says that when the correct molecule is already in the candidate list but loses rank, the wrong winner has the same molecular formula ~98% of the time on their holdout. In other words:

once you have the formula right, the real problem is which connectivity/isomer is correct.

They explicitly say mass windows are not the main lever anymore, and that structure-dependent evidence such as in-silico fragmentation is where gains appear. 

This is a huge clue for your team.

Your current system has spent a lot of effort improving retrieval around the correct mass. That’s necessary, but once candidates have approximately the right formula, cosine similarity alone has plateaued.

⸻

B. Candidate coverage is still the ceiling

The public solution lineage says something even more dramatic: their analysis estimated that standard candidate pools were missing roughly 40% of the test molecules, creating a ceiling for a single-pool reranker. 

So the winning strategy is:

Engine A candidates
        +
Engine B candidates
        ↓
     UNION
        ↓
   RRF fusion
        ↓
ICEBERG / GLACIER
        ↓
top-25

That’s fundamentally different from:

one candidate database
        ↓
better ranker
        ↓
top-25

And this is probably the most important difference between your current ~0.13–0.17 system and the 0.4-class systems.

⸻

C. Generation is much less valuable than it sounds

The discussion explicitly reports that adding generated candidates had not paid off on the leaderboard, and warns that simulated “remove-the-truth-from-the-database” novelty experiments can dramatically overestimate real novel-molecule recovery. 

That is extremely relevant to our previous GAN discussion.

Your teammate’s instinct that “we need something that can discover a new molecule” is scientifically reasonable.

But competitively:

the leaderboard evidence currently says retrieval + candidate expansion + structure-aware ranking is a much higher-return investment than jumping straight to a GAN.

⸻

2. The public solution architecture that reached ~0.4

The public development history is unusually revealing.

One public implementation records roughly this progression: 

Stage	Main idea	Public LB
Neural FPNet	spectrum → fingerprint	0.145
Analog propagation	retrieval + adduct shift + MetFrag	0.335
Two-ranker engine	multiple rankers + BIO candidates	0.358
v4n retrieval	stronger features + FPNet + PubChem	0.384
Engine fusion + ICE/GL	candidate union + forward MS/MS	0.399
Popularity + library gating	protect strong matches	0.404–0.409

That table is basically the entire story.

The jump from 0.145 → 0.335 is retrieval.

The jump from 0.358 → 0.399 is diverse engine fusion + forward fragmentation models.

The final 0.399 → 0.409 is mostly careful calibration and protection, not a radically larger model. 

⸻

3. Dissecting the gengsr 0.413 fusion notebook

Again, I could not directly retrieve the Kaggle page itself, but the public code lineage around this notebook makes the architecture fairly clear.

The central recipe is approximately:

               ┌── v4n retrieval engine ──┐
MS/MS ─────────┤                          ├── UNION
               └── second retrieval engine┘
                                      ↓
                              Reciprocal Rank Fusion
                                      ↓
                          ICEBERG + GLACIER
                                      ↓
                                top-25

Engine 1: v4n-style retrieval

This is the sophisticated descendant of the analog-propagation pipeline.

It has:

* spectral library retrieval
* mass/adduct handling
* analog propagation
* FPNet fingerprint evidence
* fragmentation-derived features
* structure/analog relationships
* PubChem candidate channel
* LightGBM ranking

The public history says v4n itself reached around 0.384 before the second-engine fusion. 

⸻

4. Engine 2 is not just “another ranker”

This is subtle.

The second engine is deliberately different.

The public recipe uses another retrieval/ranking architecture based on the earlier analog-propagation work:

* a separate candidate pool
* ChEBI/LIPID MAPS expansion
* timsTOF-aware retrieval
* another fingerprint model
* another ranker
* another analog propagation strategy

That matters because if Engine A and Engine B make identical mistakes, ensembling does almost nothing.

Instead:

Engine A says:
1 A
2 B
3 C
4 D
Engine B says:
1 B
2 E
3 A
4 F

Then RRF might produce:

A
B
E
C
F
...

So even if neither engine alone is perfect, their candidate sets complement each other.

This is exactly the sort of thing your current system is missing.

⸻

5. Reciprocal Rank Fusion is doing something very clever

The public recipe uses approximately:

RRF(s)
=
\frac{1}{3+r_A}
+
0.6\frac{1}{3+r_B}

where r_A,r_B are candidate ranks in the two engines. 

Notice what this does.

It doesn’t try to compare raw scores from two unrelated models.

That’s important because:

Engine A cosine = 0.82
Engine B score  = 7.3

Those numbers are meaningless against one another.

RRF throws away the incompatible score scales and asks:

“How strongly did each independent system rank this molecule?”

That is incredibly robust.

This is something you can steal immediately.

For your team:

your_library_ranker
        +
your_coconut_analog_ranker
        +
another public retrieval engine
        ↓
RRF

could be more useful than spending another week tuning one LightGBM.

⸻

6. The real breakthrough: union → forward prediction

This is the part I think you need to pay the most attention to.

Most ordinary systems do this:

retrieve top 25
      ↓
predict whether they are good

The strong CASMI system instead does:

Engine A top 60
       +
Engine B top 40
       ↓
      UNION
       ↓
~100 candidates
       ↓
predict their fragmentation spectra
       ↓
compare predicted fragmentation to observed spectrum
       ↓
rerank

The public solution explicitly identifies this union forward rescoring as the breakthrough from ~0.384 to ~0.399. 

That distinction matters enormously.

Suppose:

Engine A:
true molecule = rank 43
Engine B:
true molecule = rank 18

If you only use Engine A’s top 25:

true molecule disappears.

But if you union:

A top60 ∪ B top40

the true molecule enters the forward-model stage.

Now ICEBERG/GLACIER get the opportunity to say:

“Actually, this structure’s predicted fragmentation matches the observed spectrum much better.”

That’s how retrieval and generation-like structure reasoning cooperate.

⸻

7. What ICEBERG / GLACIER are actually doing

This is another point where the notebook names can make the system look more mysterious than it is.

They are forward models.

The flow is:

\text{candidate structure}
\rightarrow
\text{predicted MS/MS spectrum}

Then compare that predicted spectrum with the observed spectrum.

So the system is asking:

“Could this molecule chemically produce the peaks I observed?”

rather than merely:

“Does this molecule look statistically similar to the spectrum?”

That’s exactly the connectivity information the discussion says is missing from ordinary similarity methods. 

⸻

8. GLACIER’s [M+H]+ trick is extremely important

The Fusion Glacier MH title is not cosmetic.

The public solution discovered that GLACIER was trained for a particular precursor/adduct regime, so using it indiscriminately on every spectrum injects noise.

The implementation therefore does approximately:

gl_items = [
    spectra where adduct == "[M+H]+"
]

and only then runs GLACIER.

The related public solution reports that switching to an [M+H]+ filter improved the public score. 

This is a classic example of:

model applicability > model size

A weaker model used on the right domain can beat a stronger model used on the wrong domain.

⸻

9. Why the public systems then started “protecting” their library hits

This is another extremely important discovery.

The public experiments found that when the system already has a very strong experimental library match, a simulated forward model can actually make things worse.

Their solution introduced a library gate:

if library similarity >= 0.90:
    trust experimental evidence
    ↓
    reduce/disable ICEBERG
    reduce/disable GLACIER

The reported reasoning was that forward models had demoted some exact library hits, whereas the measured spectrum is direct experimental evidence. 

This is basically Bayesian thinking:

experimental spectrum from same/similar molecule
       >
simulated spectrum predicted by a model

when the experimental match is already extremely strong.

⸻

10. The PubChem channel is also much more conservative than I initially expected

This is another thing you should notice.

They did not do:

PubChem gives 10000 new candidates
↓
put them all into ranking

Instead:

trusted engine candidates
        +
a small, gated PubChem-only tail

The implementation inserts PubChem candidates into predefined positions such as:

2,4,6,8,10

or

4,8,12,16,20

depending on confidence.

This is brilliant for MRR@25.

Why?

Because suppose:

True molecule currently rank 1.

and you suddenly inject 20 uncertain PubChem guesses in front of it.

You’ve destroyed your MRR.

Instead they essentially say:

“I’ll reserve a few slots for the possibility that the retrieval system missed something, without sacrificing the strongest candidates.”

That idea is directly relevant to your team’s earlier failed PubChem expansion.

Your experiment already showed the problem:

more candidates → more coverage, but potentially worse MRR.

The public 0.4 system solved exactly that tension through gated tail insertion.

⸻

11. Now the fascinating one: V44 PairTail

I inspected the public mirror of the exact V44 notebook code.

And honestly:

“PairTail” sounds much more revolutionary than what it actually is.

Its core mechanism is quite simple.

It has another ranking model:

LambdaRank
objective="lambdarank"
eval_at=(1,5,25)
lambdarank_truncation_level=25

so the model is explicitly trained around ranking quality near the top of the list.

That is good.

But then the notebook does something even more important.

It calculates something like:

S =
0.85\,R_{\text{base}}
+
0.15\,R_{\text{pair}}

and then:

forces the existing baseline top-1 back to rank 1.

Conceptually:

top = argmax(base)
score[top] = max(score) + 1

So the PairTail model is effectively saying:

“You can rearrange the rest of the lineup, but don’t screw with my proven number-one answer.”

That is why I would call it:

tail reranking with a top-1 shield.

⸻

12. The “locked top-1” is probably the biggest trick in V44

The final V44 notebook goes even further.

It contains a hard-coded dictionary of the previous champion’s top-1 prediction for the 400 molecules and finally does:

wanted = champion_top1[molecule_id]
new_candidates = [
    wanted,
    everything_else
]

So regardless of what the downstream rerankers do:

the previous champion’s top-1 remains #1.

This explains the phrase:

PairTail Locked Top1

It is not just a prettier LambdaRank.

It’s a risk-management strategy.

The new system is allowed to improve ranks 2–25 while refusing to sacrifice a previously validated rank-1 prediction.

⸻

13. This is especially clever for MRR@25

Remember:

MRR =
\begin{cases}
1/r,&\text{truth at rank }r\\
0,&\text{otherwise}
\end{cases}

So moving:

rank 25 → rank 10

is useful.

But moving:

rank 1 → rank 4

is catastrophic.

Therefore:

protect rank 1
optimize ranks 2–25

is a very rational MRR strategy.

That is what the V44 architecture is exploiting.

⸻

14. But there’s a huge caveat about hard-coded top-1

I would not copy the hard-coded 400-molecule dictionary into your final competition solution blindly.

Why?

Because the competition’s visible test is a placeholder, and the organizers explicitly warn that the visible test.parquet is not the true scoring distribution; the submitted notebook is rerun on a hidden file. Another competitor independently reported losing submissions by tuning on the visible test. 

So this:

champion_top1["m_123"] = "..."

is essentially leaderboard-specific memorization.

It may be fantastic for the public leaderboard.

It is not a generalizable scientific model.

For your team, I’d replace:

hard-coded top1

with:

confidence-based top1 protection

For example:

if library similarity > 0.95
and strong experimental evidence:
       freeze top1
else:
       allow fusion/reranking

That gives you the same philosophy without hardcoding the public test.

⸻

15. Putting the two notebooks side by side

Here’s the important comparison.

Component	~0.413 Fusion/GLACIER family	V44 PairTail
Retrieval engine	v4n	v4n + additional engine
Second engine	Yes	Yes
Candidate union	Yes	Yes
RRF	Yes	Yes
FPNet	Yes	Yes
PubChem channel	Yes	Yes
ICEBERG	Yes	Yes
GLACIER	Yes	Yes
GL [M+H]+ filtering	Yes	Yes
Popularity prior	Yes	Yes
LambdaRank	indirectly / engine-level	Yes
Tail reranking	Yes	explicitly PairTail
Top-1 protection	gated	hard lock
Main philosophy	maximize complementary evidence	improve tail without sacrificing champion

So these are not actually two fundamentally different approaches.

They’re two iterations of the same emerging architecture.

⸻

16. The real architecture behind the 0.41 frontier

After stripping away all the notebook names, I would draw the winning system like this:

                         MS/MS spectrum
                               │
              ┌────────────────┼─────────────────┐
              │                │                 │
              ▼                ▼                 ▼
        Library search    Analog search     Neural FP search
              │                │                 │
              └────────────┬───┴─────────────────┘
                           │
                    Engine A candidates
                           │
                           │       Engine B
                           │          │
                           │          ▼
                           │     independent
                           │      candidates
                           │          │
                           └─────── UNION
                                  │
                                  ▼
                             RRF fusion
                                  │
                        ┌─────────┴──────────┐
                        │                    │
                        ▼                    ▼
                    ICEBERG              GLACIER
                        │                    │
                        └─────────┬──────────┘
                                  ▼
                        connectivity-aware
                             reranking
                                  │
                   ┌──────────────┴───────────────┐
                   │                              │
            strong library hit              uncertain case
                   │                              │
              protect top-1              PubChem/generative
                   │                     tail candidates
                   │                              │
                   └──────────────┬───────────────┘
                                  ▼
                               Top 25

That is the system you should be thinking about.

Not:

"Let's make our Transformer bigger."

And not yet:

"Let's train a GAN."

⸻

17. Now compare this against your team’s 0.13 system

This is where I think things become much less depressing.

Your current architecture is missing several structural layers that these public systems added.

Your previous system was roughly:

spectral library
      ↓
cosine
      ↓
candidate ranking
      ↓
COCONUT expansion
      ↓
top25

The ~0.41 family is more like:

Engine A
+
Engine B
+
PubChem
+
BIO
+
analog propagation
        ↓
candidate union
        ↓
RRF
        ↓
ICEBERG
+
GLACIER
        ↓
same-formula structural reranking
        ↓
gated external candidates
        ↓
top1 protection
        ↓
tail optimization
        ↓
top25

So your 0.13 is not evidence that your core idea is hopeless.

It’s evidence that your pipeline is currently several architectural layers behind the public frontier.

⸻

18. And here’s the really important part for your team

I would not copy everything.

The 0.41 notebooks contain a lot of accumulated complexity and Kaggle-specific engineering.

For your team, I’d extract the highest-value pieces in this order:

Phase 1 — do this first

Build:

Your current engine
        +
second candidate engine
        ↓
candidate UNION
        ↓
RRF

This is the easiest conceptual improvement.

⸻

Phase 2

Add:

ICEBERG
GLACIER

but only after candidate union.

Don’t use them as the candidate generator.

They should be the:

“Is this chemically compatible with the observed fragmentation?”

stage.

⸻

Phase 3

Implement:

GLACIER only when [M+H]+

Don’t throw GLACIER at every ionization/adduct regime.

⸻

Phase 4

Implement a real:

top-1 shield

but not the hard-coded 400-ID dictionary.

Use:

strong experimental/library evidence
        ↓
freeze current winner

and otherwise allow reranking.

⸻

Phase 5

Add PubChem/generative candidates only into controlled tail slots.

Something like:

rank 1 = trusted system
rank 2–20 = fused candidates
rank 21–25 = exploratory candidates

or dynamically gated positions.

This is much safer than dumping hundreds of PubChem structures into the main ranker.

⸻

Phase 6

Only then work on PairTail/LambdaRank.

Because PairTail is a ranking refinement.

It doesn’t solve your biggest problem if the correct candidate isn’t present.

⸻

19. What I would NOT do right now

I would not:

❌ build a GAN
❌ make your Transformer much larger
❌ expand candidate databases indiscriminately
❌ blindly widen mass windows
❌ push more PubChem structures into rank 1–10
❌ tune on visible test.parquet

The discussion evidence is remarkably consistent on these points. The visible test is especially dangerous because it is not representative of the actual hidden scoring distribution. 

⸻

20. The biggest insight from all of this

Earlier I told you that your main bottleneck was candidate recall.

After looking at these 0.41 systems, I would sharpen that diagnosis:

Your bottleneck is not merely candidate recall. It is candidate diversity.

That’s slightly different.

Suppose:

Engine A:
correct molecule rank 40
Engine B:
correct molecule rank 12
Your current system:
only Engine A

Improving Engine A from rank 40 → 25 is nice.

But:

Engine A ∪ Engine B

immediately gives you access to the correct molecule.

Then the forward fragmentation model can push it to rank 1–5.

That is why ensemble diversity + union + chemistry-aware reranking is so powerful here.

⸻

21. One more thing from the discussion that I think is tailor-made for you

The discussion says the remaining database-route errors are mainly connectivity/isomer errors, and that methods that actually depend on the molecular structure — especially in-silico fragmentation — are where gains remain. 

That perfectly matches the problem you’ve been circling around in our previous conversations.

You were asking:

“How can my model dare to find something new rather than only choosing from an existing list?”

The current leaderboard evidence suggests a more immediate answer:

Before teaching the system to invent new molecules, teach it to distinguish two existing molecules with the same formula but different connectivity.

That’s currently a much more valuable capability.

⸻

My verdict on the two notebooks

gengsr/casmi26-fusion-glacier-mh-lb-0-413

The important innovation is:

diverse retrieval engines → candidate union → RRF → forward fragmentation reranking

This is the architecture I would steal.

imranarif536/casmi26-v44-pairtail-locked-top1

The important innovation is:

don’t let a new ranker destroy a proven top-1; use LambdaRank to improve the tail

This is a second-order optimization, not the fundamental breakthrough.

And the hard-coded champion top-1 dictionary is very leaderboard-specific, so I would treat it as an engineering trick rather than a scientifically general method.

The broader public solution history corroborates exactly this hierarchy: retrieval/analog propagation produced the large gains, engine fusion + union forward models produced the major ~0.4 breakthrough, and later popularity/library gating provided smaller refinements. 

The next architecture I’d build for your team

YOUR CURRENT RETRIEVAL
        +
SECOND DIVERSE RETRIEVAL
        ↓
       UNION
        ↓
       RRF
        ↓
same-formula candidate grouping
        ↓
ICEBERG
+
GLACIER ([M+H]+ only)
        ↓
confidence-aware top1 shield
        ↓
gated PubChem / novel tail
        ↓
LambdaRank tail reranking
        ↓
TOP 25

That, in my view, is a much more realistic path from your current 0.13 toward the 0.3–0.4 regime than the GAN route.

One limitation remains: I could not responsibly summarize discussion 745029 itself because Kaggle would not expose its content here. I won’t fabricate what was written there. The accessible discussion around it, however, strongly reinforces the architecture above. 

Open the CASMI discussion page⁠ 
Open the V44 PairTail notebook you linked⁠ 
Open the Fusion + GLACIER notebook you linked⁠ 