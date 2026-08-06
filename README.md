# Learned Query-Variant Selection for RAG via Evidence Relevance Density

All numbers regenerated and verified 2026-08-05, including a from-scratch re-run of the method
that reproduced the committed predictions byte-identically. Nothing here is provisional.
Authoritative lab record: `FINDINGS.md`.

## 1. Problem

In RAG, query rewriting produces several phrasings of the user's question; some retrieve far
better evidence than others. Each topic has **31 candidate variants**, and exactly one must be
chosen **before generating anything** — retrieval has run (cheap), generation has not (expensive).

Picking the best every time would raise TREC N_All from **0.282 → 0.627**. But identifying it
requires generating and scoring all 31 answers: 31× the cost, and impossible at deployment where
no ground truth exists.

## 2. Prior work, and the hypothesis we falsified

Classical **Query Performance Prediction (QPP)** estimates query difficulty without relevance
judgments — *pre-retrieval* from the query's terms (IDF, SCQ, SCS, ICTF, QL), *post-retrieval*
from the retrieval score distribution (NQC, WIG, SMV, sigma, Clarity, RSD).

The work we extend **speculated** that the driver of answer quality is the **coverage/diversity**
of the evidence set. We pre-registered and tested that. Mean within-topic Pearson *r* vs. true
answer quality:

| feature | TREC | in-domain | train | verdict |
|---|---|---|---|---|
| `div − red` (the speculated headline) | −0.014 | −0.065 | −0.085 | **no signal, slightly backwards** |
| `div`, `spread` | −0.014 / +0.004 | −0.086 / −0.082 | −0.098 / −0.091 | negative |
| **`cov`** (how *on-topic* the passages are) | **+0.212** | **+0.230** | **+0.254** | the surviving signal |

As a selector `div − red` ranked **38th–48th of 48** and fell *below* doing nothing on all four
in-domain metrics. **The driver is not diversity but evidence relevance density** — how strongly
the retrieved passages match the actual information need. That reframing motivates everything
below.

## 3. Data

**One shared corpus** (`msmarco_v2.1_doc_segmented`) underlies both collections — so the
train→test gap is *query-style* transfer, not corpus transfer. This is what makes §4 ④ possible.

| | MS MARCO QnA (train + in-domain test) | TREC-RAG 2024 (zero-shot test) |
|---|---|---|
| topics | 5,000 (`marco_train_*`) | 56 |
| query source | raw Bing logs — short, noisy | NIST-curated |
| **mean query length** | **5.40 words** | **8.55 words** |
| variants/topic | 31 = original + 6 methods × 5 trials | same 31 |
| variant generator | Qwen2.5-7B-Instruct | QPP-4-RAG's files, **used verbatim** |
| answer generator (for labels) | Qwen2.5-32B-Instruct-AWQ | ours |
| labels | auto-nugget: `value`→N_All, `n_strict`→N_Strict | ours (4 nugget + 3 retrieval metrics) |
| scale | 155,000 (topic, variant) pairs | 1,736 |

The 6 rewriting methods: `genqr`, `genqr_ensemble`, `mugi`, `qa_expand`, `query2doc`, `query2e`.
The **31 TREC query files are the only imported artifact** — BM25 runs, QPP predictors, RAG
answers, nuggets and true scores are all computed by us (per-(topic,variant) true scores were
never released).

### How the 5,000 MS MARCO topics are partitioned

| slice | topics | mean q-len | role |
|---|---|---|---|
| fitting pool | 2,000 | 4.37 | 1,500 fit / 500 dev |
| └ dev = **longest-query quartile** | 500 | **8.13** | simulates the TREC shift |
| untouched remainder | 3,000 | — | never used to fit anything |
| └ **1,000-topic test** (sampled, seed 42) | 1,000 | **5.53** | primary in-domain result |
| older 100-topic test | 100 | — | disjoint from all 5,000 |

The dev slice at 8.13 words sits beside TREC's 8.55; the 1k test at 5.53 confirms it is genuinely
in-domain. Disjointness is verified programmatically and the build **aborts** on violation.

### Leakage in the variants — measured, flagged, never filtered

**34.9% of MS MARCO variants trip the answer-token-overlap detector** (mean overlap 0.36),
distributed very unevenly:

| method | flagged | | method | flagged |
|---|---|---|---|---|
| `mugi` | 53.6% | | `query2e` | 25.8% |
| `qa_expand` | 45.9% | | `genqr` | 17.9% |
| `query2doc` | 44.6% | | `original` | **11.1%** ← detector's false-positive floor |
| `genqr_ensemble` | 26.1% | | | |

The three worst offenders are exactly the methods that generate a pseudo-document or
pseudo-answer and prepend it, inheriting answer vocabulary. Only the excess above the 11.1%
floor is meaningful.

**Why it matters:** Fusion scores evidence against the **original question**, never the variant
text, so leaky text never enters the model. But it can inflate the *labels* — a variant carrying
answer terms retrieves answer-bearing passages and scores higher. Worth stating before a
reviewer asks.

## 4. The proposed method: `Fusion`

**Input** (per topic): the original question *q*, 31 variants, and each variant's already-retrieved
top-5 passages. **Output**: one variant id.

```
                                              ┌─ RELEVANCE MODEL (17 feat) ─→ s_rel ─┐
original question q + top-5 passages/variant ─┤                                      │  z-scored
                                              │                                   within topic
variant text + retrieval score distribution ──┴─ QPP MODEL (22 feat) ────────→ s_qpp ┤
                                                                                     ▼
                                       score = α·z(s_rel) + (1−α)·z(s_qpp)  →  argmax
```

### Features (17 + 22 = 39)

**Relevance model, 17.** Sees *q* + the top-5 passages. Twelve features = two frozen scorers ×
six list summaries:

| | mean k3 | sum k3 | mean k5 | sum k5 | max k5 | dcg k5 |
|---|---|---|---|---|---|---|
| **e5** (`e5-base-v2`, `query:`/`passage:` prefixes → asymmetric) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **ce** (`ms-marco-MiniLM-L-6-v2` cross-encoder) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

**mean** = typical quality, **sum** = total evidence mass, **max** = at least one strong hit,
**dcg** = rank-discounted. The two scorers are complementary: e5 is the stronger *single* feature
on both targets, yet `ce_mean_k3` carries the **largest weight for N_All (+0.52)**.
Plus 5 coverage features (`div`, `red`, `red_mean`, `spread`, `cov`). Note `div_minus_red` — the
falsified §2 headline — is computed but **deliberately excluded** from the model.

**QPP model, 22.** Sees the *variant text* and its BM25 score distribution; never a passage
embedding. 12 pre-retrieval (`pre_IDF-{avg,max,sum,std}`, `pre_SCQ-{avg,max,sum}`, `pre_SCS-1/2`,
`pre_avgICTF`, `pre_ql`) + 10 post-retrieval (`nqc`, `wig`, `smv` × norm/k variants, `sigma-max`,
`sigma-x0.5`, `RSD`).

### The four design decisions

**① Relevance is scored against the ORIGINAL question, never the variant text.** A rewrite that
drifts off-topic retrieves passages matching *itself* superbly — a naive score would reward it —
while being useless for the real need. (BERTQPP does the opposite by design: it predicts the
performance *of a query*.)

**② Two models blended at score level, not one model over all 39 features.** Merging the two
families inside a single model failed four separate times in this project.

**③ Every feature normalized WITHIN topic**, across that topic's own 31 siblings — the model
never sees an absolute scale, only "better or worse than its siblings." This is what makes
MS MARCO weights meaningful on TREC.

**④ Hyperparameters chosen on a SIMULATED QUERY SHIFT** ← *the methodological contribution*.
Since only query style shifts, we tune on the **longest-query quartile** of MS MARCO instead of a
random slice. The audit shows this is not cosmetic:

| target | dev protocol | dev q-len | chosen α |
|---|---|---|---|
| N_All | **shift-aware** | 8.1 | **0.75** (retains 25% QPP) |
| N_All | random | 5.5 | 1.00 (**discards QPP entirely**) |

A random in-domain split throws away precisely the component that transfers.

### Training — fitting 39 weights + α (the only training in the method)

e5 and the cross-encoder are **frozen**: no gradient reaches them, no checkpoint is saved, their
weights stay byte-identical to the public ones. They convert (question, passage) into 12 numbers
and are done. The BERTQPP training in §5 is a separate *baseline*, not part of Fusion.

```
for target ∈ {N_All, N_Strict}:                      # two independent runs
    for block ∈ {relevance(17), QPP(22)}:            # fit separately, never merged
        for norm ∈ {rank, z} × τ ∈ {.05,.1,.15,.2,.4}:
            w ← gradient ascent (400 iters, lr 0.5, L2 1e-3, w init 0)
        keep the (norm, τ, w) with best dev pick-utility
    α ← argmax over {0, .05, …, 1.00} on dev
```

**Objective.** Hard argmax has no gradient, so we maximize a soft surrogate: `p = softmax(Xw/τ)`
over a topic's 31 variants, objective `E[U] = Σ p·U` (expected true label of the soft pick). The
gradient `Xᵀ(p ⊙ (U − E[U]))/(T·τ)` is advantage-weighted — push toward variants scoring above
the topic's current expected utility, away from those below. **τ and norm are selected, not
learned**: each of the 10 settings is fit, then scored on dev by *actual hard-argmax*
pick-utility. The soft objective is only a training device.

**Fitted:** `Fusion[N_all]` → α=0.75, rank-norm, τ=0.1. `Fusion[N_strict]` → α=1.00, z-norm.
That **α=1.00 is a useful negative control**: on N_Strict the blend collapses to relevance alone
and QPP contributes nothing — fusion helps only on N_All, the metric it was introduced to fix.

**Cost.** Training: CPU-seconds (40 numbers). Inference: 155 e5 + 155 cross-encoder passes per
topic (31 × 5), then a 39-dim dot product — far more than a classical QPP predictor, far less
than the generate-all-31 alternative selection exists to avoid.

### Why this is not a neural method in the usual sense

No fine-tuning, no training run to reproduce, no chance the encoder memorized MS MARCO. **39
parameters fit on 1,500 topics** is an extremely favorable ratio and a large part of why the
model survives the transfer at all — a fine-tuned cross-encoder in the same slot failed
repeatedly. Reproducible without a GPU-days budget.

## 5. Results

`Fusion[N_all]` and `Fusion[N_strict]` are **one method, two training targets**. The
pre-registered rule reads N_All from the `N_all` row and N_Strict from the `N_strict` row;
every table follows it.

### 5a. TREC-RAG 2024 — 56 topics, zero-shot

| metric | Fusion | rank | best baseline | do-nothing floor |
|---|---|---|---|---|
| N_All | 0.3680 | **1 / 32** | sigma_0.5 (0.3617) | 0.2820 |
| N_Strict | 0.4073 | **1 / 32** | SCQ_avg (0.3459) | 0.2492 |
| N_Strict_all | 0.2296 | **1 / 32** | WIG_norm (0.2282) | 0.1769 |
| N_Vital | 0.5527 | **1 / 32** | WIG_norm (0.4893) | 0.3553 |
| **nDCG@5** | **0.5296** | **1 / 32** | BERTQPP_cross_k5 (0.3953) | 0.2790 |
| nDCG@10 | 0.4558 | **1 / 32** | BERTQPP_cross_k5 (0.3644) | 0.2569 |
| R@100 | 0.2580 | **1 / 32** | SMV_norm (0.2398) | 0.1648 |

**Rank 1 of 32 on all seven metrics**, against every classical QPP predictor plus a BERTQPP we
trained ourselves. **nDCG@5 is the largest single margin in the project** (0.5296 vs 0.3953,
oracle 0.6452): relevance density predicts *retrieval* quality far better than any QPP predictor.

### 5b. MS MARCO in-domain — 1000 held-out topics ⭐ primary in-domain result

| Method | N_All | N_Strict |
|---|---|---|
| `original` | 0.4151 | 0.5272 |
| `QL` (best baseline) | 0.4170 | 0.5468 |
| `BERTQPP_cross` | 0.4165 | 0.5284 |
| `BERTQPP_cross_k5` | 0.4079 | 0.5261 |
| `BERTQPP_bi` | 0.3843 | 0.5109 |
| **`Fusion`** | **0.4480** | **0.5869** |
| Oracle (upper bound) | 0.6710 | 0.8790 |

**Rank 1 of 28 on both metrics.** Paired bootstrap, 10k resamples:

| metric | comparison | delta | 95% CI | p |
|---|---|---|---|---|
| N_All | vs `original` | **+0.0330** | [+0.0179, +0.0479] | **<.0001** ✱ |
| N_All | vs best baseline (**test-selected**) | **+0.0310** | [+0.0183, +0.0434] | **<.0001** ✱ |
| N_Strict | vs `original` | **+0.0597** | [+0.0357, +0.0836] | **<.0001** ✱ |
| N_Strict | vs best baseline (**test-selected**) | **+0.0402** | [+0.0135, +0.0667] | **.0013** ✱ |

**Note what rows 2 and 4 are:** the bar is the best predictor *chosen on the test set* — a
winner's-curse oracle, the hardest possible baseline. On TREC (n=56) Fusion reached only parity
with it (p=.44); in-domain at n=1000 it **clears it significantly on both metrics**. This is the
strongest statistical result in the project. At n=1000 the bootstrap SE falls to ≈0.007–0.009
(from ≈0.02–0.03 at n=100), which is what makes margins of this size decisive.

Two limits: only **N_All and N_Strict** exist for this slice (the 5k pool never had
`strict_all_score`/`vital_score` computed), and six weak baselines are absent (`QSD_pre`,
`Clarity`, `Clarity_k10`, `DM`, `DM_e5`, `QSD_post`) because the 5k QPP file carries 22
predictors, not 31. All four BERTQPP variants — the real competition — *are* included.

### 5c. The 100-topic in-domain set (superseded)

Its only advantage is carrying all four metrics: N_All 0.4375 (rank 1), N_Strict 0.5975 (rank 2),
N_Strict_all 0.3182 (rank 3), N_Vital 0.7005 (rank 3). See §6.1–6.2 for why its two apparent
losses do not survive at n=1000.

### 5d. Headline claim

> **Matching model selection to the expected deployment condition buys +0.046 out-of-domain
> N_All (p = .023) at ~0.02 in-domain cost.**

Against the deployable baseline, TREC N_All improves **+0.0704 (p = .011)**. In-domain at n=1000
the method beats even the test-selected bar significantly, so that claim no longer rests on the
weaker "beats the deployable baseline" framing.

## 6. Caveats — raise these before you are asked

**6.1 The in-domain BERTQPP loss was a small-sample artifact.** At 100 topics BERTQPP's
bi-encoder led on N_Strict (0.6235 vs 0.5975). At 1000 topics it **reverses decisively: 0.5869 vs
0.5109**, with the bi-encoder falling below `original`. Given SE ≈0.02–0.03 at n=100, the original
result was inside noise.

**6.2 The evidence-breadth objection also dissolves at n=1000.** BERTQPP sees only the rank-1
document while we aggregate top-5, so a reviewer will ask how much of the margin is simply *more
evidence*. At 100 topics top-5 BERTQPP nearly tied us (0.4288 vs 0.4375, p=.33 n.s.). At 1000
topics the effect **reverses** — top-5 is worse than rank-1 for both encoders, both below
`original` — and Fusion leads the best k5 variant by **+0.040**.

**6.3 On TREC, the N_All lead over the inflated bar is parity, not a win** (+0.0063, **p=.44**).
That bar is a test-selected oracle and not deployable, but we cannot claim to significantly
exceed it there. The significant TREC gains are over the *honest* baselines.

**6.4 Statistical resolution at n=56 remains limited.** TREC SE ≈0.03, so gaps under ~0.05 are
unresolvable and no head-to-head vs BERTQPP reaches p<.05 either way *on that collection*.
Supportable TREC phrasing: *"ahead on most metrics, statistically indistinguishable on the
closest ones."*

**6.5 Much of the oracle gap is unreachable in principle.** Top variants are genuinely
near-equivalent (margin ≈0.02 vs per-variant noise σ≈0.046); multi-sample relabelling was
implemented, measured, and did not help. *"Avoid the bad variants"* is more honest than *"find
the best variant."*

**6.6 Doing nothing is a strong in-domain baseline** — only 3 of 24 baselines beat `original` on
both metrics at n=1000. This makes Fusion's margin more meaningful, not less.

**6.7 One configuration is reported throughout, on both collections.** An alternative tuned on a
random in-domain split scores better in-domain; reporting whichever wins per dataset would be
exactly the winner's curse this work criticizes.

**6.8 Label leakage is present and unfiltered** (§3): 34.9% of variants flagged, concentrated in
the pseudo-document methods. It cannot reach the model, but it can inflate labels.

## 7. Not yet run

Risk-averse / abstention selection (fall back to the original when the fused margin is small — the
natural consequence of §6.5); a relevance-estimator strength ladder; an independent second
labelling for a noise-corrected oracle (`labels.matched.B.jsonl` is **not** independent — values
identical to the primary labels on all 242 shared topics, verified); N_Strict_all and N_Vital at
1000 topics (needs a nuggetizer run over 31,000 pairs); the remaining 2,000 untouched topics,
which would scale the in-domain test to 3,000 at ~3× GPU cost.

## 8. Reproduction

| artifact | path |
|---|---|
| method | `scripts/75_fuse.py` |
| TREC + 100-topic tables | `outputs/table1_baselines_paper.{md,json}`, `..._trec.csv`, `..._marco.csv` |
| 1000-topic table | `outputs/table1_marco1k.{md,json}`, `outputs/table1_marco1k_marco1k.csv` |
| 1000-topic slice builder (contains the leakage gate) | `scripts/65_make_marco1k.py` |
| fitted configuration | `outputs/marco/fuse_selection.json` |
| significance | `outputs/repro/honest_eval.md` (TREC), `outputs/marco_test1k/bootstrap.md` (1k) |
| authoritative lab record | `FINDINGS.md` |
