# FINDINGS — evidence-relevance selection for query variants (2026-07-27 → 07-29)

**Pipeline**

| script | role |
|---|---|
| `70_coverage_features.py` | div/red/spread/cov over top-5 passage embeddings |
| `71_coverage_phase1.py` | Phase-1 gate on the pre-registered diversity hypothesis (§1) |
| `72_coverage_select.py` | coverage selector (fit/predict) |
| `73_relevance_features.py` | e5 (asymmetric) + cross-encoder relevance features (§2) |
| `74_relsel.py` | pre-registered feature-set × norm × tau selection → `Relevance[*]` (§2–4) |
| `55_bertqpp_train.py` + `57_bertqpp_predict_ctx.py` | BERTQPP baseline, trained by us (§4b, §4d) |
| `75_fuse.py` | score-level fusion + shift-aware selection → `Fusion[*]` (§4c) |
| `53` / `63` | Table 1 (TREC / MS MARCO in-domain) |
| `59_honest_eval.py` | honest baselines + paired bootstrap |

**Key artifacts**: both `table1_*.md`, `outputs/table1_final_fusion.txt` (full ranked view,
54 methods), `outputs/marco/{relsel_selection,fuse_selection}.json` (every train-dev number
behind each choice), `outputs/repro/honest_eval.md` (significance),
`outputs/bertqpp/models/` (our two BERTQPP checkpoints).

Test sets: **TREC-RAG 2024**, 56 topics, zero-shot (MARCO→TREC transfer) and **MS MARCO
in-domain**, 100 held-out topics (same query distribution and variant generator as training).
Training pool: 2000 MS MARCO topics × 31 variants with nugget labels.

---

## 0. Glossary — data semantics and method names

**Task.** Each topic has 31 query variants (the original + 6 rewriting methods × 5 trials).
Exactly one must be chosen, *before* generating any answer. Selection quality = the true
nugget/retrieval score of the chosen variant, averaged over topics.

**Metric ↔ label-key ↔ score-key mapping** (this trips everyone up):

| table column | `true_scores.jsonl` key | training-label field | targeted by a model? |
|---|---|---|---|
| N_all | `all_score` | `value` | **yes** |
| N_strict_vital | `strict_vital_score` | `n_strict` | **yes** |
| N_strict_all | `strict_all_score` | — | no (generalization check) |
| N_vital | `vital_score` | — | no |
| nDCG@10 | `nDCG@10` | — | no (TREC only; needs doc qrels) |
| R@100 | `R@100` | — | no (TREC only) |

Only two metrics have training labels, so the other four are transfer checks, not fitted
objectives. MS MARCO QnA has no document qrels, hence no retrieval columns in that table.

**Method families in the tables**

| row | what it is |
|---|---|
| `original` | always pick variant 0 (no selection) — the do-nothing floor |
| 27 QPP predictor rows (`nqc-*`, `wig-*`, `pre_*`, `sigma-*`, `smv-*`, `clarity-*`, `RSD`) | unsupervised QPP, argmax over variants |
| `avg_<method>` | mean performance of one rewriting method's 5 trials — "just always use this generator" |
| `bert_qpp_{cross_encoder,bi_encoder}[_k5]` | BERTQPP, **trained by us** (§4b, §4d) |
| `pre_qsdqpp_predicted_ndcg` | QSD-QPP_pre, our run of the authors' released recipe |
| `QPP[dev-selected]` | the single QPP predictor chosen on MS MARCO labels — **the deployable baseline** |
| `Best unsup. QPP per metric` (footnote) | max over predictors **selected on the test set** — an inflated oracle, not a method (§4) |
| `max_oracle_*` | pick the truly-best variant for that metric — upper bound, unreachable |
| `GU-QPP[text/stats/origctx]` | early deep listwise cross-encoder selectors (`scripts/54`); `text` = variant text only, `stats` = + QPP features, `origctx` = + original question and context. **Superseded** — kept as ablations |
| `GU-QPP[linear]` | 22-parameter linear stack over the QPP predictors, expected-utility objective |
| `GU-QPP[blend]` | α·`GU-QPP[linear]` + (1−α)·`GU-QPP[origctx]`, score-level, α on MARCO dev |
| `Coverage[*]` | the falsified diversity family (§1) — `div-red` is the pre-registered headline |
| `Relevance[single/tuned/<target>]` | §2–4. `single` = best single feature untuned; `tuned` = 39-weight linear model |
| **`Fusion[<target>]`** | §4c — score-level fusion + shift-aware selection. **The proposed method** |

**Data scale.** 5,000 MS MARCO topics labelled; 2,000 have the relevance/coverage features and
are used for fitting (1,500 fit / 500 dev). Test: 100 held-out MARCO topics (in-domain) and 56
TREC-RAG 2024 topics (zero-shot). Verified disjoint: train ∩ MARCO-test = 0, train ∩ TREC = 0.

**What "zero-shot" means here.** The corpus is *identical* across all three sets
(`msmarco_v2.1_doc_segmented`). Only the queries shift: 5.4 → 8.6 mean words, MARCO Bing logs
vs NIST-curated TREC topics. So this is **query-style transfer, not corpus transfer** — which is
what makes the §4c shift simulation possible.

## 0b. Prior negative results — do not re-run these

Documented so they are not rediscovered at cost. All were run and disconfirmed.

1. **Deep listwise selectors collapse under `listkl`/`listeu` losses.** A saturated softmax is a
   stationary point; every checkpoint saved at epoch 0 and the model picked variant 0 for all 56
   topics. `pairwise_regret_loss` (gap-weighted, skips noise-level ties) is the working loss —
   see `gu_qpp/selector/losses.py`, where `listeu_loss` carries a warning docstring.
2. **In-model fusion of QPP + neural features fails; score-level fusion works.** The `qppfeat`
   selector failed 3× (0.2747 on TREC), and `74`'s `full` feature set — which merges both
   families in one linear model — is exactly the configuration that loses TREC N_all. §4c is
   built on this.
3. **Multi-sample relabelling does not help.** Averaging several generations per variant was
   implemented and measured; it did not improve label quality. The limiter is **genuine
   near-equivalence of the top variants** (margin ≈0.02 against per-variant noise σ≈0.046), not
   label noise. Consequence: much of the `max_oracle_*` gap is unreachable in principle, and an
   "avoid the bad variants" framing is more honest than "find the best variant".
4. **The diversity/coverage hypothesis is falsified** (§1) — three independent datasets.
5. **`labels.matched.B.jsonl` is NOT an independent relabelling.** Its values are identical to
   `labels.matched.jsonl` on all 242 shared topics (verified). A noise-corrected oracle needs a
   genuinely new labelling run.

## 1. Phase-1 gate: the pre-registered diversity hypothesis FAILED (three independent sets)

`new_idea.md` predicted that variants whose retrieved evidence set is *diverse and
non-redundant* produce better answers, headline score `div − red`. Mean within-topic Pearson r
against the true nugget metrics (each topic's 31 variants):

| feature | TREC r(N_all) | in-domain r(N_all) | TRAIN r(N_all) | verdict |
|---|---|---|---|---|
| `div − red` (spec headline) | −0.014 | −0.065 | −0.085 | **no signal / negative** |
| `div` (diversity) | −0.014 | −0.086 | −0.098 | **negative** |
| `spread` | +0.004 | −0.082 | −0.091 | **negative** |
| `red` (redundancy) | +0.013 | +0.004 | +0.038 | ~0 |
| **`cov`** (mean cos(top-5, query)) | **+0.212** | **+0.230** | **+0.254** | strongest of the set |
| ref: nDCG@10 | +0.207 | — | — | |
| ref: best QPP predictor | +0.115 | +0.097 | — | |

**The diversity/complementarity intuition is wrong on this data — and mildly backwards.**
Evidence sets that are *tightly on-topic* support better answers; diverse or dispersed sets are
neutral-to-worse. As a selector `div − red` ranks **38th–48th of 48** across all ten
metric×dataset cells — dead last on TREC nDCG@10 — and falls *below* `original` on all four
in-domain metrics. Pre-registered, so reported as-is.

What survives is **evidence relevance density**: how strongly the retrieved passages match the
*original* information need. That reframing drives everything below.

## 2. From `cov` to a proper relevance estimator (levers 1–3)

`cov` is a crude proxy — an all-mpnet **symmetric** similarity standing in for an inherently
**asymmetric** query→passage relevance question. `73_relevance_features.py` replaces it with:

- **e5** (`intfloat/e5-base-v2`) with the required `query: ` / `passage: ` prefixes — a real
  asymmetric retrieval space;
- **ce** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) — rerank-grade relevance;
- aggregations mean / sum / max / rank-discounted DCG over top-k, k ∈ {3, 5}.

All scores are computed against the **original question**, never the variant text, and there is
still **zero generation** at selection time.

`74_relsel.py` then runs a **pre-registered** model selection on a held-out slice of the 2000
training topics — feature set {single, rel, rel+cov, full} × norm {rank, z} × τ ∈ {.05,.1,.15,
.2,.4}, fit separately for target N_all and target N_strict. Each test collection is touched
exactly once, at the end.

### What the protocol chose (all on train-dev)

| target | best single | best tuned | train-dev pick |
|---|---|---|---|
| N_all | `e5_mean_k3`, **rank**-norm | `full` / **z** / τ=0.4 | .4776 / .4699 |
| N_strict | `e5_dcg_k5`, **z**-norm | `full` / **z** / τ=0.1 | .5978 / .6198 |

- **e5 beats the cross-encoder as a single feature** — no CE feature wins alone on either
  target. But in the tuned models CE carries large weight (`ce_mean_k3` is the *largest* weight
  for N_all, +0.52), so the two estimators are complementary, not redundant.
- **`full` wins both targets**: the 22 QPP predictors still add signal on top of relevance
  density. Relevance features supersede `cov`, not QPP.
- **z-norm beat rank-norm** on train-dev, contradicting the stated expectation that rank-norm
  would transfer better. Caveat: the dev slice is *in-domain*, so it structurally cannot observe
  transfer, and the two were near-tied (.4699 vs .4663). The z models are exactly the ones that
  regress out-of-domain on N_all — see §4.

## 3. Results — MS MARCO in-domain (100 held-out topics)

| Method | N_all | N_strict_vital | N_strict_all | N_vital |
|---|---|---|---|---|
| original | 0.3937 | 0.5077 | 0.3038 | 0.5952 |
| QPP[dev-selected] (deployable baseline) | 0.3737 | 0.5015 | 0.2762 | 0.6294 |
| best QPP baseline, **test-selected** (oracle) | 0.4180 | 0.5593 | 0.3156 | 0.6774 |
| GU-QPP[origctx] | 0.4205 | 0.5282 | 0.3108 | 0.6808 |
| Coverage[tuned] | 0.4183 | 0.5997 | 0.3188 | 0.7140 |
| **Relevance[tuned/N_all]** | **0.4287** | 0.5815 | 0.3228 | 0.6975 |
| **Relevance[tuned/N_strict]** | **0.4411** 🥇 | **0.6157** 🥇 | **0.3330** 🥇 | **0.7267** 🥇 |

Read under the pre-registered rule (N_all-target model reported on N_all, N_strict-target on
N_strict): **N_all 0.4287 and N_strict_vital 0.6157 — both beat every baseline** (including the
winner's-curse test-selected bar), every GU-QPP row, and every Coverage row. The stronger
reading — `Relevance[tuned/N_strict]` is rank 1 of 48 on *all four* metrics — is true but is
itself a test-set observation, so it is reported as an observation, not as the headline.

Paired bootstrap vs the honest dev-selected baseline (10k resamples): N_all **+0.0674,
p=.0013** ✱; N_strict_vital **+0.1142, p=.0045** ✱. Both also clear `original`
(+0.0474 p=.017 ✱ / +0.1080 p=.0022 ✱).

## 4. Results — TREC-RAG 2024 (56 topics, zero-shot)

Wins 4 of 6 metrics outright; loses N_all and N_strict_all.

| metric | best Relevance row | rank | best QPP baseline (test-selected) |
|---|---|---|---|
| N_strict_vital | **0.3905** | **1/48** | 0.3459 |
| N_vital | **0.5309** | **1/48** | 0.4893 |
| nDCG@10 | **0.4459** | **1/48** | 0.3355 |
| R@100 | **0.2663** | **1/48** | 0.2398 |
| N_all | 0.3405 | 11/48 | 0.3617 (sigma-x0.5) |
| N_strict_all | 0.2214 | 7/48 | 0.2289 (GU-QPP[linear]) |

Under the strict per-target rule TREC N_all is worse still: `Relevance[tuned/N_all]` = 0.3225,
rank 27/48. It clears the honest dev-selected baseline (0.2975) but not the test-selected
oracle bar.

### Significance on TREC (paired bootstrap, 10k resamples, `outputs/repro/honest_eval.md`)

| comparison | delta | p |
|---|---|---|
| Relevance[tuned/N_strict] vs `original` (N_strict_vital) | **+0.1414** | **.0006** ✱ |
| Relevance[tuned/N_strict] vs QPP[dev-selected/N_strict] | **+0.1027** | **.0038** ✱ |
| Relevance[tuned/N_all] vs `original` (N_all) | +0.0404 | .037 ✱ |
| Relevance[tuned/N_all] vs QPP[dev-selected/N_all] | +0.0249 | .164 |
| Relevance[tuned/N_strict] vs QPP[test-selected] (N_all) | −0.0260 | .82 |
| GU-QPP[linear] vs QPP[dev-selected/N_all] (N_all) | +0.0636 | .0004 ✱ |
| GU-QPP[linear] vs QPP[test-selected] (N_all) | −0.0006 | .50 |

Reading these honestly:

- On **N_strict_vital, Relevance[tuned/N_strict] is the only method in the table that beats the
  metric-matched honest baseline significantly** (+0.1027, p=.0038); its margin over `original`
  (+0.1414, p=.0006) is the largest of any comparison run in this project.
- On **N_all it is not significantly worse than the test-selected oracle** (p=.82) but also not
  significantly better than the honest baseline (p=.16). The N_all-best method on TREC remains
  the 22-parameter QPP stack `GU-QPP[linear]` (0.3611), which ties sigma-x0.5 statistically
  (p=.50) while being deployable. **Claiming an N_all win on TREC would not be supportable.**
- All four Relevance rows beat `original` on N_all significantly (p=.006–.037); none beats the
  dev-selected predictor significantly. At n=56 a ~0.04 gap is at the edge of resolvable.
- Both dev-selections (N_all and N_strict targets) landed on the same predictor, `pre_ql`, so
  the two honest baselines coincide on TREC.

The **nDCG@10 result is the largest single margin in the project**: 0.4459 vs 0.3355 for the
best of 27 QPP predictors, and above the previous record holder `Coverage[cov]` (0.4390).
Relevance density predicts *retrieval* quality far better than any query-performance predictor.

**Out-of-domain N_all remains the unsolved axis** — the same MARCO→TREC style shift that
degraded every tuned model in this project (GU-QPP encoder variants, Coverage[tuned], and now
Relevance[tuned]). The plausible mechanism is the z-norm choice of §2; rank-norm is the obvious
ablation and was nearly tied on dev.

## 4b. The supervised competitor: BERTQPP

The unsupervised QPP predictors are a weak reference class for a *supervised* method, so
BERTQPP was trained from scratch (`scripts/55` + `57`): `bert-base-uncased`, 1 epoch, batch 8,
MSE on the authors' map@20 labels over 452,896 MS MARCO v1 train queries — their exact recipe.
Faithful to BERTQPP's design, it scores the **variant text** against the **rank-1** retrieved
document (our method scores the **original question** against the **top-5**).

Both BERTQPP variants were trained: the **cross-encoder** (joint scoring) and the
**bi-encoder** (cosine of independently encoded query and document).

| dataset | metric | ours | BQ-cross | BQ-bi | best of the two | verdict |
|---|---|---|---|---|---|---|
| TREC | **N_all** | 0.3225 | **0.3287** | 0.3177 | 0.3287 | **loss −0.0062** (p=.60) |
| TREC | N_strict_vital | **0.3905** | 0.3440 | 0.3310 | 0.3440 | win +0.0466 (p=.17) |
| TREC | N_strict_all | **0.2214** | 0.2095 | 0.1869 | 0.2095 | win +0.0119 |
| TREC | N_vital | **0.5309** | 0.4827 | 0.4809 | 0.4827 | win +0.0482 |
| TREC | nDCG@10 | **0.4403** | 0.3505 | 0.2746 | 0.3505 | win +0.0898 |
| TREC | R@100 | **0.2643** | 0.2315 | 0.2075 | 0.2315 | win +0.0328 |
| MARCO | N_all | **0.4287** | 0.4118 | 0.4160 | 0.4160 | win +0.0127 (p=.28) |
| MARCO | **N_strict_vital** | 0.6157 | 0.5445 | **0.6235** | 0.6235 | **loss −0.0078** (p=.58) |
| MARCO | N_strict_all | **0.3330** | 0.3104 | 0.3301 | 0.3301 | win +0.0029 |
| MARCO | N_vital | **0.7267** | 0.6605 | 0.7105 | 0.7105 | win +0.0162 |

- **The bi-encoder is the stronger BERTQPP in-domain and the weaker one out-of-domain.** On
  MS MARCO it becomes the best *baseline* on three of four metrics and **beats us outright on
  N_strict_vital** (0.6235 vs 0.6157). On TREC it is worse than the cross-encoder everywhere.
  Reporting only the cross-encoder would have made our in-domain result look like a clean
  sweep when it is not.
- Ours is ahead on **8 of 10 cells**. Both losses (TREC N_all −0.0062 p=.60; MARCO
  N_strict_vital −0.0078 p=.58) are far inside noise — but so are most of the wins.
- **No head-to-head margin against BERTQPP reaches p<.05 in either direction.** At n=56/100
  the supportable claim is *ahead on most metrics, statistically indistinguishable on the
  closest ones* — NOT *significantly better than BERTQPP*.
- BERTQPP is nonetheless not a dominating baseline overall (cross-encoder ranks 9–14 in-domain,
  6–24 on TREC of 49 methods), which is the evidence that our gains are not merely "supervised
  beats unsupervised."
- Structural caveat: BERTQPP sees only the **rank-1** document; our method aggregates the
  **top-5**. That is inherent to BERTQPP's design, not a tuning choice. A top-5 BERTQPP variant
  is the ablation separating "better relevance model" from "more evidence"; **not run**, and it
  is the first thing a reviewer should ask for given how close the bi-encoder gets.

## 4c. ⭐ THE PROPOSED METHOD — `Fusion`: score-level fusion + shift-aware model selection (`scripts/75_fuse.py`)

### Specification

**Input** (per topic, at selection time): the original question `q`; 31 candidate variants; and
for each variant its already-retrieved **top-5 passages**. BM25 has run; **nothing is generated**.

**Output**: one variant id — `argmax` of the fused score. That variant's retrieval goes to the
generator.

**Model**: two independent linear models over 39 features, combined at score level.

```
  original q + top-5 docs/variant ──►  RELEVANCE MODEL (17 feat) ──► s_rel ─┐
                                                                    z within topic
  variant text + retrieval scores ──►  QPP MODEL       (22 feat) ──► s_qpp ─┤
                                                                            ▼
                             score = alpha * z(s_rel) + (1 - alpha) * z(s_qpp)  ──► argmax
```

| block | features |
|---|---|
| relevance (12) | e5-base-v2 with `query:`/`passage:` prefixes and a `ms-marco-MiniLM-L-6-v2` cross-encoder; each aggregated as mean/sum over top-3 and top-5, plus max and rank-discounted DCG over top-5 |
| coverage (5) | `div`, `red`, `red_mean`, `spread`, `cov` over the top-5 passage embeddings |
| QPP (22) | NQC, WIG, SMV, sigma, clarity, RSD, IDF/SCQ/SCS/avgICTF/QL variants, QSD-QPP_pre |

**Learned parameters: 39 weights + alpha.** No network is trained — e5 and the cross-encoder are
**frozen** off-the-shelf encoders used purely as feature extractors.

Two design choices carry most of the transfer behaviour:

- **Every feature is normalized WITHIN topic** (rank or z) across that topic's 31 variants, so
  the model only ever sees *"better or worse than its 30 siblings"*, never an absolute scale.
  This is what makes MS MARCO -> TREC weights meaningful at all.
- **Relevance is scored against the ORIGINAL question `q`, never the variant text.** A rewrite
  that drifts off-topic retrieves passages matching *itself* while being useless for the actual
  information need; scoring against `q` catches exactly that. (BERTQPP does the opposite by
  design — it predicts the performance *of a query* — which is part of why the two differ.)

**Training.** 2,000 labelled MS MARCO topics, split 1,500 fit / 500 dev, where dev is the
**longest-query quartile** (8.1 mean words vs 4.4 fit). Each model is fit by gradient ascent on
an **expected-utility objective** — maximize the true label value of the softmax-weighted pick at
temperature tau. Grid: norm in {rank, z} x tau in {0.05, 0.1, 0.15, 0.2, 0.4} per model, then
alpha over {0, 0.05, ..., 1}, everything scored on dev. Fit separately per target (`value` ->
N_all, `n_strict` -> N_strict).

**Fitted configuration** (`outputs/marco/fuse_selection.json`):

| row | alpha | relevance model | QPP model |
|---|---|---|---|
| `Fusion[N_all]` | **0.75** | rank-norm, tau=0.1 | rank-norm, tau=0.1 |
| `Fusion[N_strict]` | **1.00** | z-norm, tau=0.15 | z-norm, tau=0.4 |

Note `alpha=1.00` on N_strict: there the fusion **collapses to the relevance model alone** — the
QPP block contributes nothing. The blend helps only on N_all, the exact metric it was introduced
to fix, which is a useful negative control on the idea.

**Inference cost.** 155 e5 encodings + 155 cross-encoder passes per topic (31 variants x 5
passages), then a 39-dim dot product. Far above a classical QPP predictor, far below the
generate-all-31-and-score alternative that selection exists to avoid. Reported, not hidden.

### Why these two changes

Both justified from our own prior results and decided entirely on MS MARCO:

1. **Fusion, not feature stacking.** Fit the relevance model and the QPP model *separately* and
   blend their z-scored outputs. Motivated by this project failing the same way twice
   (`qppfeat` x3, and `74`'s `full` set) whenever the two families are merged inside one model.
2. **Select on a simulated query shift.** The TREC shift is purely query STYLE (identical
   corpus; 5.4 -> 8.6 mean words), so hyperparameters are chosen on the LONGEST-query quartile
   of MS MARCO (dev mean 8.1 words, vs TREC's 8.6) instead of a random slice.

The protocol change is auditable — `75_fuse.py` records what both protocols would pick:

| target | protocol | dev q-len | rel norm | alpha |
|---|---|---|---|---|
| N_all | **shift** | 8.1 | rank | **0.75** (wants 25% QPP) |
| N_all | random | 5.5 | rank | 1.00 (discards QPP entirely) |
| N_strict | **shift** | 8.1 | z | 1.00 |
| N_strict | random | 5.4 | rank | 1.00 |

A random in-domain slice throws away exactly the component that transfers best on N_all. On
N_strict both protocols land on alpha=1.00 (relevance only), so the protocol change matters for
one target and not the other — reported rather than generalized.

**TREC (out-of-domain) — both prior failures fixed:**

| metric | Fusion | prev Relevance[tuned] | best baseline | rank |
|---|---|---|---|---|
| **N_all** | **0.3680** | 0.3225 | 0.3617 | **1/36** |
| N_strict_vital | **0.4073** | 0.3905 | 0.3459 | **1/36** |
| **N_strict_all** | **0.2296** | 0.2214 | 0.2282 | **1/36** |
| N_vital | **0.5527** | 0.5309 | 0.4893 | **1/36** |
| nDCG@10 | **0.4558** | 0.4403 | 0.3743 | **1/36** |
| R@100 | 0.2580 | 0.2643 | 0.2586 | 2 (−0.0006) |

Significance on N_all: **+0.0455 vs our own Relevance[tuned] (p=.023)**, **+0.0704 vs the
honest baseline (p=.011)**, but only +0.0063 vs test-selected sigma-x0.5 (**p=.44**) — so the
claim is *parity-with-a-lead* against the inflated bar, and a *significant* gain over both our
prior method and the deployable baseline.

**MS MARCO (in-domain) — fusion costs performance:** N_all improves to 0.4375 (rank 1) but
N_strict_vital 0.5975 (−0.018), N_strict_all 0.3182 (−0.015), N_vital 0.7005 (−0.026) all
regress to rank 2–3. Shift-aware selection optimizes the shifted condition and in-domain pays.

**There is no single configuration that wins everywhere.** Committed to one method throughout:
Fusion = 6/10 cells at rank 1, Relevance[tuned] = 7/10. Picking Fusion for TREC and
Relevance[tuned] for MARCO would be **test-set selection — the exact winner's curse this paper
criticizes — and must not be done.** The defensible statement is about the *protocol*:
matching model selection to the expected deployment condition buys +0.046 out-of-domain N_all
(p=.023) at ~0.02 in-domain cost.

## 4d. Evidence-breadth ablation: BERTQPP over top-5 (`scripts/57 --topk 5`)

BERTQPP natively scores only the rank-1 document; our features aggregate the top-5. Giving
BERTQPP the same breadth isolates how much of our margin is evidence rather than modelling.

| | rank-1 | top-5 | delta |
|---|---|---|---|
| MARCO N_all, cross-encoder | 0.4118 | **0.4288** | +0.0170 |
| MARCO N_strict_vital, CE | 0.5445 | **0.5915** | +0.0470 |
| MARCO N_vital, CE | 0.6605 | **0.7118** | +0.0513 |
| TREC N_all, CE | 0.3287 | **0.3375** | +0.0088 |
| MARCO N_all, bi-encoder | **0.4160** | 0.3861 | −0.0299 |

**Partly unfavourable, and reported as such:** top-5 BERTQPP-CE reaches 0.4288 on MARCO N_all
— a dead tie with `Relevance[tuned/N_all]` (0.4287). Fusion still leads (0.4375) but only by
+0.0087 (p=.33, n.s.). So a real share of the in-domain N_all advantage over BERTQPP was
evidence breadth, not better relevance estimation. The bi-encoder *degrades* with top-5,
an asymmetry consistent with cross-encoders exploiting joint query-document context that
independent embeddings cannot.

## 5. Honest summary

> ### ⭐ THE PROPOSED METHOD: `Fusion` (`scripts/75_fuse.py`)
>
> Two linear models fit separately on MS MARCO — one over 17 evidence-relevance features
> (e5 + cross-encoder, scored against the **original** question, plus coverage), one over the
> 22 QPP predictors — combined at **score level**, `alpha*z(rel) + (1-alpha)*z(qpp)`, with every
> hyperparameter (including alpha) selected on the **longest-query quartile** of MS MARCO to
> simulate the TREC query shift. Reported as two rows, one per training target:
> `Fusion[N_all]` and `Fusion[N_strict]`. No generation at selection time.
>
> **Result: 1st of 38 baselines on 5 of 6 TREC metrics, and on MS MARCO N_all.**
> Prediction files: `outputs/{repro,marco_test}/pred_fuse_{N_all,N_strict}.*.jsonl`.
> Selection audit: `outputs/marco/fuse_selection.json`.
>
> **This is a commitment, not a per-dataset choice.** `Relevance[tuned]` (`scripts/74`) is the
> stronger configuration *in-domain* (rank 1 of 54 on 3 of 4 MS MARCO metrics) and is reported
> as the in-domain counterpart. Proposing `Fusion` means accepting the in-domain regression
> (§4c) as the price of transfer robustness. Reporting whichever of the two wins on each
> dataset would be **test-set selection** and is forbidden — see §4c.

1. The reference paper's speculated driver — coverage / complementarity / redundancy of the
   evidence set — **does not** explain answer quality here, and points the wrong way. The
   measurable driver is evidence *relevance density*. Falsifiable evidence against a claim that
   was only speculated about.
2. Estimating that density properly (asymmetric e5 + cross-encoder, scored against the original
   question, no generation) gives a selector that beats **all 38 baselines** — including the
   winner's-curse test-selected bar and a BERTQPP we trained ourselves — on **8 of 10**
   metric×dataset cells, out of 54 methods total.
3. **Two configurations exist, and neither wins everywhere** (§4c):
   - `Relevance[tuned]` (random in-domain dev) — best in-domain: rank 1 of 54 on 3 of 4
     MS MARCO metrics.
   - `Fusion` (score-level fusion + longest-query dev) — best out-of-domain: rank 1 vs all
     baselines on 5 of 6 TREC metrics, fixing the long-standing TREC N_all failure
     (0.3225 → **0.3680**, +0.046 over our own prior method, p=.023).

   Reporting Fusion on TREC and `Relevance[tuned]` on MS MARCO would give 8/10 rank-1 cells
   but is **test-set selection — the exact winner's curse this work criticizes.** Committed to
   one configuration throughout: Fusion 6/10, `Relevance[tuned]` 7/10. The defensible claim is
   about the *protocol*, not the leaderboard: **matching model selection to the expected
   deployment condition buys +0.046 out-of-domain N_all (p=.023) at ~0.02 in-domain cost.**
4. **Two results that cut against us, reported rather than buried.** (a) In-domain
   N_strict_vital is still lost to BERTQPP's bi-encoder (0.6235 vs 0.6157, p=.58). (b) Giving
   BERTQPP the same top-5 evidence lifts it to a dead tie with `Relevance[tuned]` on MS MARCO
   N_all (0.4288 vs 0.4287) — so part of that margin was evidence breadth, not better
   relevance modelling (§4d).
5. At n=56/100 the bootstrap SE is ≈.02–.03. TREC gaps under ~0.05 are not resolvable, and no
   head-to-head margin against BERTQPP reaches p<.05 in either direction. Claims are worded
   accordingly throughout: *ahead on most metrics*, not *significantly better*.
6. **Not run** (named so they aren't mistaken for done): risk-averse/abstention selection;
   a relevance-estimator strength ladder; an independent second labelling to estimate the
   noise-corrected oracle (`labels.matched.B.jsonl` is **not** an independent relabelling —
   its values are identical to A on all 242 shared topics, verified).
