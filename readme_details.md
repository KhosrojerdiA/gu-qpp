# `Fusion` explained in plain language

A ground-up walkthrough of the proposed method, assuming no background. For the formal
specification, results tables and caveats, see [README.md](README.md). For the authoritative
lab record, see [FINDINGS.md](FINDINGS.md).

---

## The situation

You have a question:

> *"what was the immediate impact of the success of the manhattan project?"*

An LLM rewrote it 31 different ways. Each rewrite was fed to a search engine, and each got back
5 documents. So you are holding 31 little bundles:

```
rewrite 1  →  [doc, doc, doc, doc, doc]
rewrite 2  →  [doc, doc, doc, doc, doc]
...
rewrite 31 →  [doc, doc, doc, doc, doc]
```

**You must pick exactly one bundle.** Its 5 documents get handed to the AI, which writes the
final answer. Pick a good bundle → good answer. Pick a bad one → bad answer.

You cannot try them all — writing 31 answers is the expensive thing you are trying to avoid. So
you must guess, using only what you can see right now.

**`Fusion` is the guesser.**

---

## The core idea: two judges

`Fusion` asks **two completely different questions** about each rewrite, then combines the
verdicts.

> **Judge A — the relevance judge.** *"Look at the 5 documents this rewrite found. Do they
> actually answer the original question?"*
>
> **Judge B — the QPP judge.** *"Ignore the documents' content. Just look at the rewrite's
> wording and the search engine's confidence numbers. Does this smell like a good search query?"*

Each judge gives every rewrite a score. Then:

```
final score = 0.75 × (Judge A) + 0.25 × (Judge B)
```

Pick the rewrite with the highest final score. **That is the whole method.** Everything below is
detail about how each judge computes its number, and how the two are combined fairly.

---

## Judge A: does the evidence answer the question?

Judge A looks at **17 measurements** per rewrite.

**12 of them come from two AI scorers.** Both do the same job — read a question and a document,
output "how well does this document answer this question?" — but differently:

- **e5** turns the question into a list of numbers and each document into a list of numbers,
  then measures the angle between them.
- **the cross-encoder** reads the question and document *together* and gives a verdict. Slower,
  more accurate.

A real row from `outputs/marco/rel.train.jsonl` (topic `marco_train_0`, variant 0):

```
e5_mean_k3 = 0.8188   ← average match over the top 3 documents
e5_mean_k5 = 0.8161   ← average over top 5 (lower → docs 4–5 are weaker)
e5_max_k5  = 0.8209   ← the single best document
ce_mean_k3 = 1.4349   ← same idea, cross-encoder (different scale)
...
```

Why 6 flavours from each scorer? Because *"are these documents good?"* has several meanings: the
**average** one, the **best** one, the **total** amount of good evidence, and a version that
cares more about document #1 than document #5.

**5 more measurements** describe the shape of the document set — are the 5 documents
near-identical or scattered? These are leftovers from the hypothesis the project disproved; only
one of them (`cov`) turned out useful.

Judge A multiplies each of its 17 measurements by a weight and adds them up. One number per
rewrite.

### The single most important trick in the whole method

Judge A always compares documents against the **original question** — *never* against the
rewrite.

Here is why that matters enormously. Suppose one rewrite drifts off-topic:

> original: *"immediate impact of the manhattan project"*
> bad rewrite: *"history of nuclear physics research in Germany"*

That rewrite searches and gets back 5 documents about German nuclear physics. Those documents are
a **perfect** match — *for the rewrite*. If you scored them against the rewrite, this bad variant
would look like your best candidate.

But they do not answer what the user asked. Scoring against the original question exposes it
instantly: those 5 documents match "manhattan project impact" poorly, so Judge A scores it low.

*(This is precisely backwards from BERTQPP, the competing method, which scores documents against
the query being evaluated. That is a large part of why the two behave differently.)*

---

## Judge B: does this look like a good query?

Judge B never looks at what the documents *say*. It sees only two things:

1. **The rewrite's own words.** Are they rare and specific, or vague and common?
   *"manhattan project"* is specific; *"what was the thing"* is not. Classic measurements: IDF,
   SCQ, SCS, ICTF, QL. **(12 measurements)**
2. **The search engine's score numbers.** When BM25 ranked 100 documents, did #1 score far above
   the rest — meaning the engine locked on confidently — or were all 100 scores mushed together,
   meaning it was guessing? Measurements: NQC, WIG, SMV, sigma, RSD. **(10 measurements)**

Same procedure: 22 measurements × 22 weights, summed. One number per rewrite.

---

## Why two judges instead of one big model?

The obvious thing is to throw all 39 measurements into a single model. **That was tried four
separate times in this project, and it failed every time.**

Intuitively: the two families speak different languages and live on different scales. Merged into
one model they interfere — one family swamps the other, and the combination generalizes badly to
new data.

Kept apart, each judge becomes good at its own job, and only their *final verdicts* are combined.
That separation is a finding from the project's own failed experiments, not a stylistic
preference.

---

## Two normalizations (easy to confuse — they are different)

### Normalization #1 — on the inputs, within each topic

For one topic, take one measurement (say `e5_mean_k3`) across all 31 rewrites, and replace the
raw values with **ranks**.

So the model never sees `0.8188`. It sees *"7th best out of 31 for this topic."*

**Why this is essential:** some questions are just easy — every rewrite scores ~0.9. Others are
hard — every rewrite scores ~0.3. If the model saw raw numbers, it would learn *"0.9 means good"*
and then be hopeless on hard topics where 0.4 is genuinely excellent.

By ranking within the topic, the model only ever learns **"is this rewrite better or worse than
its 30 siblings?"** — a question that means the same thing on every topic, in every dataset.
**This is the single reason weights learned on MS MARCO still work on TREC.**

### Normalization #2 — on the two judges' outputs

Judge A's scores might land in the range 0–10. Judge B's might land in −3 to +3. If you simply
added them, Judge A would dominate no matter what number you put in front of it.

So before blending, each judge's scores are converted to **z-scores** within the topic
(recentred to average 0, rescaled to spread 1). Now they are on equal footing, and the blend
weight genuinely controls the mix.

---

## α: how much to trust each judge

```
score = α × z(Judge A) + (1 − α) × z(Judge B)
```

α = 0.75 means *"trust Judge A 75%, Judge B 25%."*

α was not guessed — every value from 0 to 1 in steps of 0.05 was tried, and the best kept.

For the N_Strict version α came out **1.00**, meaning Judge B contributed *nothing* there. That
is an honest negative result, reported rather than hidden: the two-judge idea helps on one metric
and not the other.

---

## How the weights were learned

Judge A needs 17 weights, Judge B needs 22. Where do they come from?

There are **1,500 practice topics where the answer is already known** — the full expensive
pipeline was run on all 31 rewrites and the resulting answer quality measured. For the example
topic, those true scores were:

```
[0.50, 0.45, 0.50, 0.40, ..., 0.70, ...]   ← rewrite #10 was best, at 0.70
```

Training is then simple:

1. Start with all weights at **zero**.
2. Score the 31 rewrites with the current weights.
3. Compare against the truth: **nudge weights up** for measurements that were high on
   genuinely-good rewrites, **down** for measurements that were high on bad ones.
4. Repeat 400 times.

Important: the model is **not** trying to predict "0.70". It only needs its highest-scoring
rewrite to be a genuinely good one. It can be wildly wrong about the actual numbers as long as it
puts a good rewrite on top.

**Total learned: 17 + 22 + 1 = 40 numbers.**

The two AI scorers (e5, cross-encoder) are **never trained**. They are used like a thermometer —
an instrument that reports a reading. Nobody adjusts the thermometer.

---

## The last piece: choosing settings for the *right* conditions

A few settings had to be chosen — α, and how sharply to nudge. Normally you would hold out a
random slice of training data and test settings on it.

**The problem:** the training questions are short, messy Bing queries (5.4 words average). The
real test, TREC, has long, carefully-written questions (8.6 words). A random slice of training
data is *also* short and messy — so it cannot tell you what will work on long questions.

**The fix:** use the **25% longest training questions** as the validation slice. Those average
8.1 words — almost exactly TREC's 8.6. Now the validation set resembles the real deployment
condition.

It changed the outcome:

| validation slice | verdict |
|---|---|
| random (5.5 words) | α = 1.00 → **throw Judge B away entirely** |
| longest 25% (8.1 words) | α = 0.75 → **keep Judge B** |

And on the actual TREC test, keeping Judge B was right — this is what fixed the one metric that
had been failing all along. **That is the methodological contribution.**

---

## Putting it all together — one new question, start to finish

1. Question arrives. 31 rewrites are generated, each already searched, each holding 5 documents.
2. Run e5 and the cross-encoder over 31 × 5 = 155 question-document pairs, twice →
   **310 forward passes**.
3. Compute 39 measurements per rewrite.
4. **Rank each measurement within this topic** (normalization #1).
5. Judge A: 17 numbers × 17 weights → one score. Judge B: 22 × 22 → one score.
6. **z-score both judges' outputs** (normalization #2).
7. Blend: `0.75 × A + 0.25 × B`.
8. Take the highest. Send that rewrite's 5 documents to the generator.

**No answer was written during any of this.** That is the entire point — 310 small forward passes
instead of 31 full LLM generations.

---

## The three-sentence version

> `Fusion` scores each rewrite two ways — *"do its documents actually answer the original
> question?"* and *"does it look like a good query?"* — then blends the two verdicts 75/25 and
> takes the best. Everything is measured relative to the rewrite's 30 siblings rather than on an
> absolute scale, which is why weights learned on one dataset survive on another. It is 40
> learned numbers on top of two frozen off-the-shelf scorers, and it never generates an answer.

---

## Where to go next

| you want | read |
|---|---|
| formal spec, results, significance, caveats | [README.md](README.md) |
| every experiment, including the failed ones | [FINDINGS.md](FINDINGS.md) |
| the code | `scripts/75_fuse.py` |
