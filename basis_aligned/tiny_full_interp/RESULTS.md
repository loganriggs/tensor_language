# RESULTS — local box (tiny full interpretation)

Newest first. Every number here is reproducible from the JSONs named beside it;
nothing is quoted from a transcript. Registered predictions are written into
each results JSON *before* the rung that tests them runs, and the ones that
were **refuted** are marked as such rather than quietly dropped.

---

## 2026-08-08 — FINDING 12 (RUNG 5, THE COMPRESSION FRONTIER): a description 5.7× shorter than the model exists, but it is the model's own weights coded better — every description built out of an *interpretation* is dominated, and merging tokens is the worst code we measured

Files: `tf_compress.py` (coders + the swappable depth-1 decoder),
`tf_compress_run.py` (sections A–M), `tf_compress_frontier.py` (Pareto + figure),
`tf_compress_tables.py` (every table below is printed by this script from the
JSONs, so nothing here is transcribed by hand),
`tf_vanilla_d1_w128_b8192_s0_compress.json` (+ `_s1_` for the confirmation cell),
`tf_vanilla_d1_w128_b8192_s0_compress_frontier.json`,
`fig_tf_compression_frontier.png`.

**Why this exists.** Rung 5 asked for an explicit description reproducing the
model, and the only weights-free artifact the ladder produced was the model's
own token-pair table: 8192×8192 = 67.1M entries against a 1.34M-parameter
model, at KL 0.657. The "explanation" was 50× larger than the thing explained.
Rather than argue about whether that counts, this finding replaces the argument
with a plot: **description length in bits on x, KL from the true model on y,
with the model's own length marked**, so "does any description beat the model
itself?" is a measurement.

**The accounting, stated once and applied everywhere.** A description is a bit
string that a fixed decoder (`tf_compress.D1Desc.forward`, the depth-1 vanilla
forward with every table swappable) turns into a next-token predictor.
Everything the decoder needs that is not source code is charged: tables,
codebooks, cluster indices, per-row scales, bit-allocation maps, entropy-coder
histograms. fp32 = 32 bits; an index into k things = ⌈log₂k⌉ bits; an
arithmetic-coded symbol stream = its empirical entropy plus its histogram at
fp16. Tables are fitted on `est`; every number is scored on `held` (64
sequences × 256 tokens = 16 384 tokens). Two conventions are declared because
they are generous to the *structural* schemes and therefore conservative for
the negative result: the token surface strings and the estimation split itself
are free to the decoder (they are part of the corpus specification, shared by
the model and by every description), so a scheme may condition on a token's
spelling or on its corpus co-occurrence statistics without paying for them.

**Positive control.** The decoder with all tables at their trained values
reproduces the model to `rel_logit_diff` 4.5e-6 and KL 1.5e-6 — that 1.5e-6 is
the measurement floor and no KL below it is meaningful.

### The headline

| | |
|---|---|
| the model | 1 343 616 parameters (78.0% embedding, 22.0% body), **42.996 Mbit** at fp32, held CE 4.7114 |
| shortest description at KL ≤ 0.005 | **7.59 Mbit, 5.7× smaller** |
| shortest at KL ≤ 0.025 | **5.79 Mbit, 7.4× smaller** |
| shortest at KL ≤ 0.11 | **4.09 Mbit, 10.5× smaller** |
| shortest at KL ≤ 0.45 | **2.68 Mbit, 16.0× smaller** |
| the extreme point (a 1-bit embedding) | **2.43 Mbit, 17.7× smaller**, KL 0.829 |
| lossless to the measurement floor | 16.45 Mbit, 2.6× smaller |
| the rung-5 weights-free table | **2147 Mbit at fp32 (50× the model), 537 Mbit even at int8**, at KL 0.657 |

For scale on the KL axis: the model's whole advantage over the unigram floor is
CE 7.2845 → 4.7114, i.e. **2.573 nats**. So KL 0.005 is 0.2% of everything the
model knows, KL 0.023 is 0.9%, and KL 0.41 is 16%.

**Every point on that frontier is the model's own weights, coded better** —
with exactly one partial exception, which is the honest positive of this
finding: conditioning the embedding code on the token's estimation-split
co-occurrence statistics lands on the frontier at four of its twenty-five
points and is worth 7–14% of the bits. Nothing else assembled out of an
interpretation — prototypes, named feature groups, low-rank factors, CP terms,
exact anchor rows for important tokens — comes within a factor of two of the
frontier at any KL.

![compression frontier](fig_tf_compression_frontier.png)

### 1. Memorisation vs structure, measured three ways — and the first two registered predictions are REFUTED

Logan's framing was that some of the model is memorisation (token-specific
facts) and some is structure (rules that generalise across tokens), that the
structure should compress a lot and the memorisation somewhat. The embedding is
78% of the parameters, so "the memorisation" is concretely the 8192×128 token
table, in its two roles: the **read** role (`rms(Wte[t])`, the layer-0 module
input — how a token steers the computation) and the **write** role (the tied
unembedding — the token's identity as an answer). The description may compress
the two separately, paying for both tables.

**(a) Merging tokens — the obvious attack — is the worst code we measured.**
Cluster the token axis into k behavioural prototypes (frequency-weighted Lloyd
on the trained rows), coarsen one role, leave the other exact, score. At k=512
— the 16× cut Logan's intuition points at — the read role leaves **KL 1.184**
and the write role **0.868**. For calibration, *deleting the MLP entirely*
costs 4.70 and deleting all past attention costs 0.29. A prototype dictionary
that keeps 512 of 8192 tokens distinct is worse than throwing away the model's
attention several times over. Pushing to k=4096 (a 2× cut) still leaves KL
0.435 for the read role and costs 16.9 Mbit — while plain 4-bit scalar
quantisation of the *whole* table costs 4.03 Mbit at KL 0.028. **Clustering is
4× the bits at 15× the KL.** Learned clusters do beat a random grouping at
matched k (1.184 vs 1.974 at k=512), so the clustering is working; it is the
*idea* that fails.

**(b) Precision per role: the two roles are nearly symmetric, and P1 is
refuted.** We registered that the read role would be far more compressible than
the write role (< 0.10 vs > 1.0 at 512 clusters). Measured, in the currency
that actually decides the frontier — bits per weight, not number of prototypes
— the two roles cost almost the same, and the *write* role is marginally
cheaper: at 3 bits, read-only KL 0.0736 vs write-only 0.0544; at 4 bits, 0.0157
vs 0.0119. The prediction is refuted, and in the opposite direction. There is
no cheap role.

**(c) Features vs identity: the spelling pays for 1% of the table and the
corpus statistics for 7–14%.** Two conditional codes were built, both
with their conditioning information given away free:

* **orthography + frequency** (log unigram band, orthographic class, length,
  hashed character 1- and 2-grams: 272 features): regression R² **0.256** on
  the embedding. Coding only the residual saves **1.1, 1.3, 1.2 and 0.7%** of
  bits at matched KL for 2-, 3-, 4- and 5-bit residuals — i.e. essentially
  nothing.
* **corpus co-occurrence** (PPMI of the est-split bigram table, 128 left + 128
  right singular directions): R² **0.405**. Saves **14.0 / 10.7 / 7.9 / 7.1%**
  of bits at matched KL for 2-/3-/4-/5-bit residuals — small, but real enough that with a quantised body it reaches the Pareto
  frontier at four points (5.05 Mbit at KL 0.061, 6.17 at 0.021, 6.76 at 0.012,
  7.84 at 0.0031). If the 8192×257 projection basis is *charged* rather than
  regenerated from the corpus, it loses by 4×; the frontier claim depends on
  the declared convention that the corpus is free.

So a token's embedding row is 26% predictable from its spelling and 41% from
its corpus statistics — and knowing the spelling buys 1% of its description
length, knowing the corpus statistics 7–14%. What the model stores is precisely the part that is *not* predictable
from the token's surface or its data statistics. Registered P2 (features
recover < 30% of what learned clusters recover) is **refuted as stated**: as a
*grouping* at ~440 groups, features recover 60% of the learned clusters' KL —
but that is only because grouping at all is so weak a baseline that a
size-matched random grouping recovers 67%, i.e. more. The corrected verdict is
sharper than the registered one: **surface features are worth about as much as
chance when used to merge tokens and about 1% when used to predict-and-code;
corpus co-occurrence statistics, which are a fairer reading of "structure", are
worth 7–14%.**

**The split, in bits.** At the 5.77 Mbit / KL 0.023 point the bill is 4.03 Mbit
embedding (70%) and 1.74 Mbit body (30%) — so at the knee the memorisation is
**492 bits ≈ 62 bytes per token** and the structure is 1.74 Mbit for all of it.
The memorisation *is* compressible, 8.3× from fp32, but only by dropping
precision, not by finding types. Registered P3 (< 60% of the bits in the
embedding at the knee) is **refuted**: 70%.

### 2. What resisted compression — the negative deliverable

Each of these is a well-measured "no", with the bits.

**The MLP is not low CP rank, in the neuron basis or out of it.** The bilinear
MLP *is* a rank-512 symmetric CP decomposition, so truncating hidden units is a
genuine CP truncation — and the program's standing lesson is that the neuron
basis is a gauge, so we also **refitted** the CP decomposition by ALS directly
on the folded tensor (never materialising the 128³ object; the ALS normal
equations only need Gram matrices). The refit is a real improvement at every
rank — 384 terms: KL 0.270 refitted vs 0.316 truncated; 128 terms: 1.361 vs
1.695; 32 terms: 2.360 vs 2.789 — which confirms the gauge lesson. It does not
matter: **plain 3-bit quantisation of Left/Right/Down costs 3.78 Mbit at KL
0.116, while the best 384-term refit costs 7.87 Mbit at KL 0.270.** Registered
P4 (no structural MLP scheme beats scalar quantisation at matched bits) is
**CONFIRMED**, and it is confirmed by a wide margin. The tensor genuinely needs
all 512 of its terms; what it does not need is 32-bit coefficients.

**The embedding has no low-dimensional token manifold.** Rank 96 of 128 costs
25.6 Mbit and still leaves KL 0.800. Every low-rank point is off the frontier
by more than 5×.

**Rotating the coding basis buys nothing.** Transform coding with per-column
reverse-water-filling bit allocation is the best embedding coder we found — but
the gain is entirely the *allocation*, not the *rotation*: at ~4.0 Mbit the
identity basis gives KL 0.0150, Hadamard 0.0154, PCA 0.0168. For a program
built around basis alignment this is worth stating plainly: **the embedding's
trained coordinate basis is already as good a coding basis as any orthogonal
alternative**, including its own principal axes.

**Exact anchor rows do not port from `../qk_mdl`.** The parent program's
frontier-dominating hybrid — exact fp32 rows for the top-B tokens by
attribution plus a compressed remainder — is *dominated* here at every B and
every tail coder: `anchor256 + 4-bit tail` costs 5.37 Mbit at KL 0.017 while
plain 5-bit costs 5.11 Mbit at KL 0.0065. Registered P5 is **REFUTED**. The
weakened form survives: *graded* precision (6 bits for the top 2048 tokens, 4
for the tail) does beat uniform precision by about 30% in bits at matched KL
(4.59 Mbit at KL 0.0093 against ~5.2 Mbit interpolated). So "spend more bits on
frequent tokens" is right; "spend infinite bits on a few tokens" is wrong. The
difference from the parent program is that there the compressed object was a
V×V score table with wildly heterogeneous row importance, whereas here every
row of the unembedding sits in the softmax denominator of *every* prediction —
that is a plausible explanation, and it is not measured here.

**Product quantisation loses to scalar quantisation.** At ~4.3 Mbit,
`pq_m128_b4` gives KL 0.081 and 4-bit scalar with per-row scales gives 0.028.
Per-row scales beat per-subspace codebooks: the row norms of the embedding vary
enough that removing that one degree of freedom per token is worth more than a
learned 256-word codebook per 8 dimensions.

### 3. What *did* work, and the one thing that surprised us

* **Entropy-coding the quantised symbols** (histogram charged) saves a flat
  ~10% at every bit depth. Free and honest.
* **Per-column bit allocation** (reverse water-filling on the column variances)
  is worth about 1.5× in bits at matched KL over uniform per-row quantisation.
* **Graded precision by token frequency** ≈ another 30%.
* **Distilling the quantised description on `est`** (straight-through
  quantiser in the loop, best iterate selected on a disjoint `est` slice,
  nothing fitted on held) is the only technique that changes the *shape* of the
  frontier rather than shifting it: it is worth almost nothing at 5–8 bits and
  it is worth **an order of magnitude of KL** at 1–3 bits. A **1-bit embedding**
  — every one of the 1 048 576 embedding weights reduced to one of two values
  per row — post-hoc gives KL 6.07 and distilled gives **0.83, at 2.43 Mbit,
  17.7× smaller than the model.** That the model survives a binary embedding at
  all is the most surprising number in this finding.
* **Conditioning on corpus statistics** is worth 7–14% of the embedding's bits
  and, uniquely among the structural schemes, makes the frontier.
* The body is the *precision-sensitive* part, not the embedding: at 4 bits the
  attention matrices alone cost KL 0.221 while the whole embedding costs 0.028.
  98k of the 1.34M parameters carry most of the precision requirement.

### 4. The rung-5 reframe: "weights-free" is not a meaningful MDL constraint

Rung 5 as written asks for a description "with no weights". Charging bits
dissolves that distinction: an 8192×128 table called *the embedding* and an
8192×8192 table called *the model's bigram table* are both just tables, and the
second is 64× bigger. The weights-free artifact the ladder produced costs 2147
Mbit at fp32 — **50× the model it explains** — and 537 Mbit even at int8, at KL
0.657. It is off the frontier by two and a half orders of magnitude and it is
not close to being a competitive description of anything.

The honest restatement, and the one this finding answers: **is there a
description shorter than the model that reproduces it?** Yes — 5.7× shorter at
KL 0.004, 7.4× at 0.023, 16× at 0.41. And the answer that matters for
interpretability: **all of them are the model's own weights coded better, and
no description built out of an interpretation is anywhere near the frontier.**
Rung 5 is therefore *passed* in the MDL sense and *failed* in the sense it was
meant: at this size, the model's per-token content is not compressible into
types, features, prototypes, factors or exceptions — only into fewer bits per
number.

### 5. Registered predictions and their verdicts

All six of P1–P6 were written into the results JSON before section A ran; P7
was registered after the self-red-team demanded the corpus-statistic
experiment, and before that experiment ran. Three of seven survive.

| | prediction | verdict |
|---|---|---|
| P1 | the read role is far more compressible than the write role (< 0.10 vs > 1.0 at 512 clusters) | **REFUTED** — 1.184 vs 0.868 at 512 clusters, and 0.074 vs 0.054 at 3 bits: nearly symmetric, write marginally cheaper |
| P2 | feature groupings recover < 30% of what learned clusters recover | **REFUTED as stated** (60%) — but only because a size-matched *random* grouping recovers 67%; the intended claim survives in stronger form |
| P3 | at the knee the embedding is < 60% of the bits, total ≥ 6× below fp32, KL < 0.05 | **PARTLY REFUTED** — 7.4× below fp32 at KL 0.023, but the embedding is **70%** of the bits, not < 60% |
| P4 | no structural MLP scheme beats scalar quantisation at matched bits | **CONFIRMED**, by 2× in bits and 2.3× in KL |
| P5 | exact anchor rows + compressed tail dominate a pure scheme by ≥ 1.5× | **REFUTED** — dominated everywhere; only the graded (non-exact) form helps, by ~30% |
| P6 | no weights-free table lands near the frontier | **CONFIRMED** — 12–50× the model's own bits at KL 0.657 |
| P7 | corpus co-occurrence statistics reach R² 0.40–0.65 but save < 25% of bits | **CONFIRMED** — R² 0.405, saves 7–14% |

### 6. Confirmation on a second cell

The battery (sections A, B, C, E, F, G, J, K, L) was re-run on
`tf_vanilla_d1_w128_b8192_s1` — independent seed, same cell. See T11. What
replicates tightly: the frontier's *position* (`embT640+body8` 6.467 Mbit at KL
0.0156 vs 6.464 at 0.0164; `embT768+body8` 7.594 at 0.0042 vs 7.593 at 0.0051),
the per-role near-symmetry at 3 bits (0.0736/0.0544 vs 0.0821/0.0584), the
clustering failure (k=512 read 1.184 vs 1.224) and both regression R² values
(0.256 vs 0.260, 0.405 vs 0.405). What does **not** replicate tightly is the
absolute KL of *aggressive uniform quantisation of the whole model*: 4-bit is
0.288 on seed 0 and 0.849 on seed 1, 6-bit is 0.0112 vs 0.0327. Seed 1 is
simply a more quantisation-brittle model at low precision, so the frontier's
low-bit tail should be read as a shape, not as a per-model constant. Every
claim in §1–§4 is about orderings and ratios, all of which hold on both seeds.

### 7. Adversarial review, round 1 (self-red-team)

**O1 — "quantisation is not an interpretation, so this finding is vacuous."**
Accepted as a description of the result and rejected as a criticism of it. The
implicit hope in rung 5 was that a *structured* description would be short. The
measurement says it is not, and it says so against seven structural families
with honest bit accounting. A negative frontier result is exactly the kind of
thing this program said it wanted.

**O2 — "you searched a finite family of schemes."** True, and the two
strongest candidates we could think of were added *because of this objection*
(section K, orthography; section L, corpus co-occurrence), both with their
conditioning given away free. Orthography loses; corpus co-occurrence is the
one structural scheme that does *not*, and it is reported as a positive. Still untried and worth a future
tick: tensor-train / hierarchical Tucker of the MLP tensor; learned rotations
per PQ subspace; weight sharing under a learned permutation; magnitude pruning
plus sparse coding; and coding the embedding conditional on a *trained* small
model's embedding rather than on raw statistics.

**O3 — "16 384 held tokens is not many."** Sequence-clustered standard errors
are attached to every distilled point (64 sequences as the independent unit);
they are 0.8–1.0% of the KL at every distilled point, far below the effect
sizes quoted. The
frontier's ordering is not within noise anywhere it is used.

**O4 — "a distilled description is a different model, not a description of
this one."** It is scored by KL to *this* model on held text, which is
precisely the rung-5 criterion, and the tables it stores are decoded by exactly
the same decoder as the post-hoc points. The distinction that matters is
declared: distilled points are fitted on `est` against the true model's own
outputs, post-hoc points are not fitted at all, and both are scored on `held`.

**O5 — "entropy coding assumes an arithmetic coder you did not charge for."**
The coder is code, like the rest of the decoder; the standing convention charges
data, not the program. The histogram *is* charged (2^b × 16 bits). Removing
entropy coding entirely moves every affected point right by ~10% and changes no
ordering.

**O6 — "features lose to random grouping only because feature groups are
unbalanced."** Caught by this objection and fixed: a size-matched random
control (same group-size histogram, random membership) is in section J. It
changes the verdict — features beat size-matched random for the write role at
all three group counts and for the read role at the two smaller ones — and the
corrected statement is in §1(c).

**O7 — "the frequency ordering used by stratification and anchors is fitted on
est and not charged."** Declared. It is free under the same convention that
makes the token strings free (the decoder can recompute it from the corpus). If
you reject that convention, delete the stratified and anchor families: they are
not the frontier winners, so the headline is unchanged.

**O8 — "the KL direction."** All numbers are KL(model ‖ description), matching
the existing rung-5 ladder in `tf_interp.ladder`, so this finding's KLs are
directly comparable to Table C and FINDING 3.

**What we could not compress, in one sentence.** The MLP tensor's CP rank, the
embedding's row space, and the per-token content of the embedding are all
incompressible in every basis we tried: 512 CP terms, 128 dimensions and 8192
distinct tokens are all *needed*, and the only thing that turned out to be
surplus was precision — about 27 of the 32 bits on every number.

### 8. The tables (printed by `tf_compress_tables.py` from the JSONs)

model: 1343616 params, fp32 42.996 Mbit, held CE 4.7114, KL floor 1.47e-06

### T1 the model as its own description (uniform b-bit weights)

| bits/weight | description length | x smaller than fp32 | KL |
|---|---|---|---|
| 2 | 3.015 Mbit | 14.3x | 4.03130 |
| 3 | 4.358 Mbit | 9.9x | 1.60630 |
| 4 | 5.702 Mbit | 7.5x | 0.28790 |
| 5 | 7.045 Mbit | 6.1x | 0.05808 |
| 6 | 8.389 Mbit | 5.1x | 0.01118 |
| 8 | 11.076 Mbit | 3.9x | 0.00064 |
| 12 | 16.450 Mbit | 2.6x | 0.00000 |
| 16 | 21.823 Mbit | 2.0x | 0.00000 |
| 32 | 42.996 Mbit | 1.0x | 0.00000 |

### T2 the Pareto frontier (all families, everything charged)

| description length | x smaller | KL | scheme |
|---|---|---|---|
| 2.431 Mbit | 17.7x | 0.82907 | `distilled_emb1_body4` |
| 2.680 Mbit | 16.0x | 0.40897 | `distilled_emb2_body3` |
| 3.613 Mbit | 11.9x | 0.38761 | `distilled_emb2_body6` |
| 3.766 Mbit | 11.4x | 0.13230 | `distilled_emb3_body3` |
| 4.089 Mbit | 10.5x | 0.10487 | `distilled_emb3_body4` |
| 4.702 Mbit | 9.1x | 0.09298 | `distilled_emb3_body6` |
| 4.771 Mbit | 9.0x | 0.07519 | `embT512+body6` |
| 5.054 Mbit | 8.5x | 0.06073 | `corpusstat_res_q3+body6` |
| 5.167 Mbit | 8.3x | 0.03455 | `distilled_emb4_body4` |
| 5.785 Mbit | 7.4x | 0.02387 | `distilled_emb4_body6` |
| 6.171 Mbit | 7.0x | 0.02094 | `corpusstat_res_q4+body6` |
| 6.426 Mbit | 6.7x | 0.01913 | `embS6_4_2048+body6` |
| 6.467 Mbit | 6.6x | 0.01556 | `embT640+body8` |
| 6.760 Mbit | 6.4x | 0.01171 | `corpusstat_res_q4+body8` |
| 7.016 Mbit | 6.1x | 0.00987 | `embS6_4_2048+body8` |
| 7.464 Mbit | 5.8x | 0.00567 | `distilled_emb5_body8` |
| 7.594 Mbit | 5.7x | 0.00421 | `embT768+body8` |
| 7.742 Mbit | 5.6x | 0.00400 | `embS8_5_512+body8` |
| 7.842 Mbit | 5.5x | 0.00305 | `corpusstat_res_q5+body8` |
| 8.516 Mbit | 5.0x | 0.00152 | `distilled_emb6_body8` |
| 11.076 Mbit | 3.9x | 0.00064 | `uniform_8bit` |
| 15.917 Mbit | 2.7x | 0.00062 | `corpusstat_residual_q6` |
| 16.450 Mbit | 2.6x | 0.00000 | `uniform_12bit` |
| 21.823 Mbit | 2.0x | 0.00000 | `uniform_16bit` |
| 42.996 Mbit | 1.0x | 0.00000 | `uniform_32bit` |

### T3 coarsening the token axis: merging tokens is a bad code

| groups k | read-role KL | write-role KL | read, random grouping | write, random grouping |
|---|---|---|---|---|
| 1 | 5.053 | 3.859 | 5.053 | 3.859 |
| 2 | 5.318 | 2.772 | 4.851 | 3.866 |
| 4 | 3.885 | 2.342 | 4.863 | 3.897 |
| 8 | 3.281 | 2.071 | 4.703 | 3.919 |
| 16 | 2.885 | 1.872 | 4.415 | 3.915 |
| 32 | 2.470 | 1.673 | 3.959 | 4.018 |
| 64 | 1.882 | 1.454 | 3.437 | 4.026 |
| 128 | 1.690 | 1.281 | 2.903 | 3.964 |
| 256 | 1.489 | 1.073 | 2.387 | 3.578 |
| 512 | 1.184 | 0.868 | 1.974 | 3.019 |
| 1024 | 1.144 | 0.879 | 1.523 | 2.465 |
| 2048 | 0.789 | 0.503 | 1.108 | 1.896 |
| 4096 | 0.435 | 0.354 | 0.721 | 1.254 |

### T4 per-role PRECISION (the currency that works)

| bits/weight | read role coarsened only | write role only | both (tied) |
|---|---|---|---|
| 1 | 1.08676 | 3.12100 | 4.88301 |
| 2 | 0.44255 | 0.29603 | 0.74989 |
| 3 | 0.07362 | 0.05442 | 0.12924 |
| 4 | 0.01573 | 0.01187 | 0.02775 |
| 5 | 0.00364 | 0.00283 | 0.00646 |
| 6 | 0.00089 | 0.00069 | 0.00158 |
| 8 | 0.00005 | 0.00004 | 0.00010 |

### T5 embedding schemes at ~matched bits

| scheme | embedding bits | KL (body exact fp32) |
|---|---|---|
| `pq_m16_b4` | 0.590 Mbit | 2.56078 |
| `transform_hadamard_256bpr` | 1.039 Mbit | 1.85946 |
| `pq_m16_b6` | 1.049 Mbit | 1.76136 |
| `transform_none_256bpr` | 1.061 Mbit | 1.64974 |
| `lowrank_r4` | 1.065 Mbit | 2.41059 |
| `pq_m32_b4` | 1.114 Mbit | 1.98253 |
| `cluster_k256` | 1.114 Mbit | 1.93467 |
| `transform_pca_256bpr` | 1.160 Mbit | 2.89988 |
| `lowrank_r16_q8` | 1.328 Mbit | 2.05873 |
| `pq_m8_b8` | 1.573 Mbit | 1.68460 |
| `transform_hadamard_384bpr` | 1.810 Mbit | 0.33136 |
| `pq_m32_b6` | 1.835 Mbit | 0.90531 |
| `transform_pca_384bpr` | 1.835 Mbit | 0.33906 |
| `transform_none_384bpr` | 1.856 Mbit | 0.31344 |
| `pq_m16_b8` | 2.097 Mbit | 1.11556 |
| `lowrank_r8` | 2.130 Mbit | 2.27137 |
| `pq_m64_b4` | 2.163 Mbit | 0.78471 |
| `cluster_k512` | 2.171 Mbit | 1.76309 |
| `scalar_q2` | 2.359 Mbit | 0.74989 |
| `lowrank_r32_q8` | 2.393 Mbit | 1.82815 |
| `transform_hadamard_512bpr` | 2.884 Mbit | 0.06896 |
| `transform_pca_512bpr` | 2.885 Mbit | 0.08010 |
| `scalar_q3_entropy` | 2.913 Mbit | 0.12924 |
| `transform_none_512bpr` | 2.936 Mbit | 0.06484 |
| `pq_m32_b8` | 3.146 Mbit | 0.31320 |
| `strat_hi8_lo3_n512` | 3.257 Mbit | 0.06798 |
| `strat_hi6_lo3_n1024` | 3.334 Mbit | 0.05549 |
| `scalar_q3` | 3.408 Mbit | 0.12924 |
| `strat_hi8_lo3_n1024` | 3.598 Mbit | 0.05466 |
| `transform_pca_640bpr` | 3.976 Mbit | 0.01683 |
| `transform_hadamard_640bpr` | 3.990 Mbit | 0.01538 |
| `scalar_q4_entropy` | 4.029 Mbit | 0.02775 |
| `transform_none_640bpr` | 4.042 Mbit | 0.01496 |
| `lowrank_r16` | 4.260 Mbit | 2.05867 |
| `pq_m128_b4` | 4.260 Mbit | 0.08100 |
| `cluster_k1024` | 4.276 Mbit | 1.73407 |
| `strat_hi8_lo4_n512` | 4.304 Mbit | 0.01467 |
| `strat_hi6_lo4_n1024` | 4.311 Mbit | 0.01258 |
| `strat_hi5_lo4_n2048` | 4.326 Mbit | 0.01280 |
| `scalar_q4` | 4.456 Mbit | 0.02775 |
| `lowrank_r64_q8` | 4.524 Mbit | 1.38990 |
| `strat_hi8_lo4_n1024` | 4.574 Mbit | 0.01175 |
| `strat_hi6_lo4_n2048` | 4.591 Mbit | 0.00930 |
| `transform_pca_768bpr` | 5.104 Mbit | 0.00421 |
| `scalar_q5_entropy` | 5.110 Mbit | 0.00646 |
| `transform_hadamard_768bpr` | 5.118 Mbit | 0.00386 |
| `transform_none_768bpr` | 5.170 Mbit | 0.00366 |
| `pq_m64_b8` | 5.243 Mbit | 0.02426 |
| `strat_hi6_lo5_n1024` | 5.256 Mbit | 0.00364 |
| `strat_hi8_lo5_n512` | 5.317 Mbit | 0.00346 |
| `scalar_q5` | 5.505 Mbit | 0.00646 |
| `scalar_q6_entropy` | 6.169 Mbit | 0.00158 |
| `scalar_q6` | 6.554 Mbit | 0.00158 |
| `scalar_q8_entropy` | 8.258 Mbit | 0.00010 |
| `cluster_k2048` | 8.479 Mbit | 1.18735 |
| `lowrank_r32` | 8.520 Mbit | 1.82926 |
| `scalar_q8` | 8.651 Mbit | 0.00010 |
| `pq_m128_b8` | 9.437 Mbit | 0.00013 |
| `lowrank_r48` | 12.780 Mbit | 1.62227 |
| `cluster_k4096` | 16.876 Mbit | 0.86074 |
| `lowrank_r64` | 17.039 Mbit | 1.39058 |
| `lowrank_r96` | 25.559 Mbit | 0.80044 |

### T6 the body: structure vs precision

| scheme | body bits | KL (embedding exact fp32) |
|---|---|---|
| `mlp_trunc_units32` | 3.543 Mbit | 2.78943 |
| `mlp_trunc_units64` | 3.936 Mbit | 2.28594 |
| `mlp_trunc_units128` | 4.723 Mbit | 1.69459 |
| `mlp_trunc_units256` | 6.296 Mbit | 0.92326 |
| `mlp_trunc_units384` | 7.868 Mbit | 0.31645 |
| `mlp_cp_refit32` | 3.543 Mbit | 2.36033 |
| `mlp_cp_refit64` | 3.936 Mbit | 1.92732 |
| `mlp_cp_refit128` | 4.723 Mbit | 1.36052 |
| `mlp_cp_refit256` | 6.296 Mbit | 0.74690 |
| `mlp_cp_refit384` | 7.868 Mbit | 0.27000 |
| `mlp_q2` | 3.580 Mbit | 0.83826 |
| `mlp_q3` | 3.777 Mbit | 0.11634 |
| `mlp_q4` | 3.973 Mbit | 0.02485 |
| `mlp_q6` | 4.366 Mbit | 0.00142 |
| `mlp_q8` | 4.760 Mbit | 0.00009 |
| `mlp_cp128_q4` | 3.359 Mbit | 1.40792 |
| `mlp_cp128_q6` | 3.457 Mbit | 1.37247 |
| `mlp_cp128_q8` | 3.555 Mbit | 1.36127 |
| `mlp_cp256_q4` | 3.564 Mbit | 0.78773 |
| `mlp_cp256_q6` | 3.760 Mbit | 0.74614 |
| `mlp_cp256_q8` | 3.957 Mbit | 0.74760 |
| `mlp_cp384_q4` | 3.768 Mbit | 0.32008 |
| `mlp_cp384_q6` | 4.063 Mbit | 0.26945 |
| `mlp_cp384_q8` | 4.358 Mbit | 0.27037 |
| `attn_q2` | 6.517 Mbit | 3.05268 |
| `attn_q3` | 6.615 Mbit | 1.19544 |
| `attn_q4` | 6.713 Mbit | 0.22056 |
| `attn_q6` | 6.910 Mbit | 0.00807 |
| `attn_q8` | 7.107 Mbit | 0.00046 |
| `body_cp256_q3` | 0.639 Mbit | 2.13498 |
| `body_cp256_q4` | 0.836 Mbit | 1.00773 |
| `body_cp256_q6` | 1.229 Mbit | 0.75744 |

### T7 anchor rows + compressed tail (ported from ../qk_mdl)

| scheme | embedding bits | KL |
|---|---|---|
| `anchor0_freq_tail_pq_m16_b8` | 2.097 Mbit | 1.11556 |
| `anchor0_freq_tail_pq_m8_b8` | 1.573 Mbit | 1.68460 |
| `anchor0_freq_tail_q4` | 4.456 Mbit | 0.02775 |
| `anchor0_freq_tail_q3` | 3.408 Mbit | 0.12924 |
| `anchor0_freq_tail_q2` | 2.359 Mbit | 0.74989 |
| `anchor0_freq_tail_cluster_k512` | 2.171 Mbit | 2.22267 |
| `anchor64_freq_tail_pq_m16_b8` | 2.352 Mbit | 0.72332 |
| `anchor64_freq_tail_pq_m8_b8` | 1.832 Mbit | 1.19938 |
| `anchor64_freq_tail_q4` | 4.685 Mbit | 0.02074 |
| `anchor64_freq_tail_q3` | 3.644 Mbit | 0.09572 |
| `anchor64_freq_tail_q2` | 2.604 Mbit | 0.56472 |
| `anchor64_freq_tail_cluster_k512` | 2.433 Mbit | 1.50119 |
| `anchor256_freq_tail_pq_m16_b8` | 3.116 Mbit | 0.53937 |
| `anchor256_freq_tail_pq_m8_b8` | 2.608 Mbit | 0.93209 |
| `anchor256_freq_tail_q4` | 5.369 Mbit | 0.01698 |
| `anchor256_freq_tail_q3` | 4.353 Mbit | 0.07890 |
| `anchor256_freq_tail_q2` | 3.337 Mbit | 0.47091 |
| `anchor256_freq_tail_cluster_k512` | 3.220 Mbit | 1.25262 |
| `anchor512_freq_tail_pq_m16_b8` | 4.135 Mbit | 0.41397 |
| `anchor512_freq_tail_pq_m8_b8` | 3.644 Mbit | 0.75779 |
| `anchor512_freq_tail_q4` | 6.282 Mbit | 0.01462 |
| `anchor512_freq_tail_q3` | 5.299 Mbit | 0.06793 |
| `anchor512_freq_tail_q2` | 4.316 Mbit | 0.40779 |
| `anchor512_freq_tail_cluster_k512` | 4.270 Mbit | 1.02398 |
| `anchor1024_freq_tail_pq_m16_b8` | 6.174 Mbit | 0.30352 |
| `anchor1024_freq_tail_pq_m8_b8` | 5.715 Mbit | 0.56890 |
| `anchor1024_freq_tail_q4` | 8.107 Mbit | 0.01169 |
| `anchor1024_freq_tail_q3` | 7.190 Mbit | 0.05461 |
| `anchor1024_freq_tail_q2` | 6.272 Mbit | 0.33200 |
| `anchor1024_freq_tail_cluster_k512` | 6.369 Mbit | 0.81203 |
| `anchor2048_freq_tail_pq_m16_b8` | 10.250 Mbit | 0.19167 |
| `anchor2048_freq_tail_pq_m8_b8` | 9.857 Mbit | 0.37050 |
| `anchor2048_freq_tail_q4` | 11.758 Mbit | 0.00821 |
| `anchor2048_freq_tail_q3` | 10.971 Mbit | 0.03885 |
| `anchor2048_freq_tail_q2` | 10.185 Mbit | 0.24148 |
| `anchor2048_freq_tail_cluster_k512` | 10.568 Mbit | 0.55753 |
| `anchor256_q8_tail_pq_m16_b8` | 2.338 Mbit | 0.53993 |
| `anchor1024_q8_tail_pq_m16_b8` | 3.061 Mbit | 0.30358 |

### T8 distillation vs post-hoc quantisation at the same bits

| budget (emb/body bits) | distilled bits | distilled KL | post-hoc bits | post-hoc KL |
|---|---|---|---|---|
| 1/4 | 2.431 Mbit | 0.82907 ± 0.00836 | 2.429 Mbit | 6.07152 |
| 2/3 | 2.680 Mbit | 0.40897 ± 0.00383 | 2.591 Mbit | 2.33225 |
| 2/4 | 2.999 Mbit | 0.41099 ± 0.00357 | 2.906 Mbit | 1.15662 |
| 2/6 | 3.613 Mbit | 0.38761 ± 0.00320 | 3.515 Mbit | 0.76806 |
| 3/3 | 3.766 Mbit | 0.13230 ± 0.00129 | 3.719 Mbit | 1.60630 |
| 3/4 | 4.089 Mbit | 0.10487 ± 0.00096 | 4.034 Mbit | 0.42086 |
| 3/6 | 4.702 Mbit | 0.09298 ± 0.00079 | 4.643 Mbit | 0.14160 |
| 4/4 | 5.167 Mbit | 0.03455 ± 0.00034 | 5.149 Mbit | 0.28790 |
| 4/6 | 5.785 Mbit | 0.02387 ± 0.00019 | 5.759 Mbit | 0.03784 |
| 4/8 | 6.401 Mbit | 0.02294 ± 0.00020 | 6.373 Mbit | 0.02828 |
| 5/8 | 7.464 Mbit | 0.00567 ± 0.00005 | 7.454 Mbit | 0.00697 |
| 6/8 | 8.516 Mbit | 0.00152 ± 0.00001 | 8.513 Mbit | 0.00213 |

### T9 does the token SPELLING pay for its row?

feature regression R^2 on the embedding = 0.2560 (272 features)

| residual bits | feature-conditional bits | its KL | plain bits at the SAME KL | bits saved |
|---|---|---|---|---|
| 2 | 2.097 Mbit | 0.44513 | 2.120 Mbit | +1.1% |
| 3 | 3.237 Mbit | 0.07795 | 3.280 Mbit | +1.3% |
| 4 | 4.353 Mbit | 0.01670 | 4.406 Mbit | +1.2% |
| 5 | 5.435 Mbit | 0.00398 | 5.475 Mbit | +0.7% |
| 6 | 6.494 Mbit | 0.00095 | — | — |

### T9b does the CORPUS CO-OCCURRENCE statistic pay for it?

PPMI-SVD regression R^2 on the embedding = 0.4049 (257 features)

| residual bits | conditional bits (statistic free) | its KL | plain bits at the SAME KL | bits saved |
|---|---|---|---|---|
| 2 | 2.080 Mbit | 0.27926 | 2.419 Mbit | +14.0% |
| 3 | 3.219 Mbit | 0.04976 | 3.606 Mbit | +10.7% |
| 4 | 4.336 Mbit | 0.01109 | 4.709 Mbit | +7.9% |
| 5 | 5.417 Mbit | 0.00249 | 5.828 Mbit | +7.1% |
| 6 | 6.476 Mbit | 0.00062 | — | — |

### T10 the weights-free artifacts, priced

| artifact | description length | x the fp32 model | KL |
|---|---|---|---|
| `weightsfree_VxV_bigram_table_fp32` | 2147.484 Mbit | 49.9x | 0.6573 |
| `weightsfree_VxV_bigram_table_fp16` | 1073.742 Mbit | 25.0x | 0.6573 |
| `weightsfree_VxV_bigram_table_int8` | 536.871 Mbit | 12.5x | 0.6573 |
| `factored_r0_plus_WU_fp32` | 67.109 Mbit | 1.6x | 0.6573 |

### T11 CONFIRMATION on seed 1 (same cell, independent run)

| quantity | seed 0 | seed 1 |
|---|---|---|
| held CE | 4.7114 | 4.7094 |
| KL, uniform_4bit | 0.28790 | 0.84900 |
| KL, uniform_6bit | 0.01118 | 0.03272 |
| KL, uniform_8bit | 0.00064 | 0.00169 |
| embT640+body8: bits / KL | 6.467 Mbit / 0.01556 | 6.464 Mbit / 0.01638 |
| embT768+body8: bits / KL | 7.594 Mbit / 0.00421 | 7.593 Mbit / 0.00509 |
| feature regression R^2 | 0.2560 | 0.2598 |
| read-only / write-only KL at 3 bits | 0.07362 / 0.05442 | 0.08214 / 0.05839 |
| read clustering k=512 KL | 1.184 | 1.224 |

---

## 2026-08-08 — FINDING 11 (PHASE V1, the six-architecture slice): the interpretable architectures compute something GENUINELY DIFFERENT — they USE a residual route the plain model leaves empty, and they get induction at half the width

**The question this slice was built to answer** is not which variant wins on
loss — at 1.6M parameters that is nearly meaningless — but whether architectures
that claim to be more interpretable *compute the same thing by different means,
or something different*. With four modules and an exact fold, that is decidable.

**Verdict: DIFFERENT, and by a margin far outside the seed spread.** All five
non-vanilla variants *use* a residual route that carries essentially nothing in
the plain model, and four of the five acquire an algorithm (induction) that the
plain model needs twice the width to build, at all three seeds — the fifth
(`codebook`) at two of three. One of them (`predicate`) also *beats* the plain
model on loss while being the most legible of the six, though its induction is
handed to it by the architecture rather than learned.

**Corrected by the independent round-2 review (§8):** the route difference is a
**magnitude** difference, not a weight-space one. The plain model's layer 1 is,
per unit of read displacement, the *most* sensitive of the six to layer-0
attention's direction; it transmits nothing only because it renormalises its own
first attention write down to 0.3% of that read. The earlier phrasing — that the
variants *open* a route the plain model leaves *shut* — is withdrawn.

Cell: depth 2, width 128, vocab 8192 trained byte-level BPE, three seeds, Muon
0.02 matched across every arm, 15,000 steps × batch 16, single epoch, identical
data order. Files: `tf_*_d2_w128_b8192_s{0,1,2}_interp3.json`, comparison in
`tf_variant_compare.json` / `.txt` and `tf_consolidated_table.md`, registered
predictions in `tf_variant_predictions.json` (written before the first training
step), independent review in `tf_reviewer_round_2.json` with its raw numbers in
`tf_round2_measurements.json`.

> **STATUS: COMPLETE.** All 37 arms (six architectures x three seeds, plus
> nineteen control and robustness arms) were force-reanalysed through ONE
> revision of `tf_interp3.py` in a single pass (`tf_interp3_final.log`), so no
> number below came from an older code path —
> `tf_variant_compare.json`'s `dropped_because_produced_by_an_older_analysis_revision`
> list is **empty**, the `--control` gate passes at 1.9e-6, and all 37 fold
> gates pass. The learning-rate falsifier is **closed** (the plain model is null
> at Muon 0.01, 0.02 and 0.04). An **independent round-2 review** by an agent
> that did not produce these results has been through the slice and changed
> several claims — read **§8** before citing §1, and `tf_reviewer_round_2.json`
> for the full record.

### THE CONSOLIDATED COMPARISON TABLE

Cell: depth 2, width 128, vocab 8192 trained byte-level BPE, Muon 0.02 matched
across every arm (no per-arm sweep), AdamW 0.004 on the embedding, 15,000 steps
x batch 16, single epoch, identical data order. Generated by
`tf_consolidated_table.py` from `tf_variant_compare.json`; the standalone copy
is `tf_consolidated_table.md`. **Significance yardstick:** the vanilla seed
spread measured at this exact cell before the slice existed — CE 0.0074 nats,
induction 0.0086. Nothing smaller than that is called a difference.

#### Table A — per architecture, per seed (the six primary arms)

| architecture | seed | held CE (T512) | bits/byte | induction (± probe floor 3 SE) | natural swap (excess over own depth-1 null) | A0→layer-1 read deleted, KL [zero, resample] | content/null | selection/null |
|---|---|---|---|---|---|---|---|---|
| vanilla | 0 | 4.6512 | 1.7872 | -0.0138 ± 0.0075 (-1.8×) **below floor** | +0.1000 (t=5.8) (+0.0318) | [2.42e-05, 5.48e-06] | 0.980 | 0.362 |
| vanilla | 1 | 4.6501 | 1.7868 | -0.0022 ± 0.0092 (-0.2×) **below floor** | +0.0925 (t=5.8) (+0.0243) | [9.80e-06, 4.02e-06] | 0.976 | 0.300 |
| vanilla | 2 | 4.6377 | 1.7820 | +0.0059 ± 0.0142 (+0.4×) **below floor** | +0.1170 (t=6.8) (+0.0489) | [4.66e-06, 3.47e-06] | 0.980 | 0.378 |
| slots | 0 | 4.7418 | 1.8220 | +0.1129 ± 0.0128 (+8.8×) | +0.1688 (t=9.7) (+0.0831) | [0.574, 0.123] | 0.998 | 0.285 |
| slots | 1 | 4.7356 | 1.8196 | +0.1133 ± 0.0145 (+7.8×) | +0.1880 (t=10.1) (+0.1024) | [0.511, 0.127] | 1.000 | 0.270 |
| slots | 2 | 4.7468 | 1.8239 | +0.0654 ± 0.0119 (+5.5×) | +0.1515 (t=9.3) (+0.0659) | [0.568, 0.122] | 1.000 | 0.273 |
| bandwidth | 0 | 4.6263 | 1.7776 | +0.0965 ± 0.0098 (+9.8×) | +0.1980 (t=9.9) (+0.1002) | [0.600, 0.150] | 0.999 | 0.266 |
| bandwidth | 1 | 4.6253 | 1.7773 | +0.1789 ± 0.0161 (+11.1×) | +0.2148 (t=10.1) (+0.1170) | [0.493, 0.149] | 1.000 | 0.206 |
| bandwidth | 2 | 4.6321 | 1.7799 | +0.0817 ± 0.0103 (+7.9×) | +0.1967 (t=9.6) (+0.0988) | [0.521, 0.133] | 1.001 | 0.206 |
| predicate | 0 | 4.3843 | 1.6846 | +2.5934 ± 0.0304 (+85.3×) | +1.5029 (t=34.1) (-0.0327) | [0.352, 0.071] | 0.997 | 0.268 |
| predicate | 1 | 4.3883 | 1.6862 | +2.6378 ± 0.0239 (+110.1×) | +1.4756 (t=32.8) (-0.0601) | [0.401, 0.074] | 0.999 | 0.268 |
| predicate | 2 | 4.3858 | 1.6852 | +2.6895 ± 0.0195 (+137.9×) | +1.4878 (t=32.6) (-0.0478) | [0.306, 0.065] | 1.000 | 0.254 |
| codebook | 0 | 4.7480 | 1.8244 | +0.0540 ± 0.0086 (+6.3×) | +0.1682 (t=8.7) (+0.0821) | [0.113, 0.108] | 0.995 | 0.269 |
| codebook | 1 | 4.7571 | 1.8279 | +0.0358 ± 0.0085 (+4.2×) | +0.1420 (t=8.3) (+0.0559) | [0.105, 0.097] | 0.996 | 0.286 |
| codebook | 2 | 4.7576 | 1.8281 | +0.0228 ± 0.0249 (+0.9×) **below floor** | +0.1404 (t=8.5) (+0.0543) | [0.138, 0.083] | 0.997 | 0.261 |
| shrink | 0 | 4.7357 | 1.8197 | +0.0510 ± 0.0146 (+3.5×) | +0.1248 (t=7.6) (+0.0415) | [0.301, 0.148] | 0.998 | 0.192 |
| shrink | 1 | 4.7199 | 1.8136 | +0.1032 ± 0.0154 (+6.7×) | +0.1562 (t=8.3) (+0.0730) | [0.170, 0.134] | 0.997 | 0.186 |
| shrink | 2 | 4.7172 | 1.8126 | +0.1037 ± 0.0200 (+5.2×) | +0.1680 (t=9.2) (+0.0847) | [0.216, 0.149] | 0.997 | 0.218 |

> `predicate`'s induction is **supplied by the architecture, not discovered**: `MATCH_prev[i,j] = 1[tok_{j-1} == tok_i]` is a complete induction head handed over as one scalar per head, zeroing those 16 scalars removes 98.7–99.1% of the score, and its depth-1 cell already scores +1.536 on the natural-text swap — which is why its *excess over its own depth-1 null* is negative. Do not read its column as a learned circuit.

#### Table B — the same, aggregated over the three seeds (mean ± sd)

| architecture | nominal params (body / embed) | effective params | stream width | held CE | bits/byte | induction | routing KL zero | routing KL resample | content/null | selection/null |
|---|---|---|---|---|---|---|---|---|---|---|
| **vanilla** (n=3) | 1,638,656 (590,080 / 1,048,576) | 1638656 | 128 | 4.6463 ± 0.0075 | 1.7853 ± 0.0029 | -0.0034 ± 0.0099 | 1.29e-05 ± 1.01e-05 | 4.32e-06 ± 1.04e-06 | 0.978 ± 0.002 | 0.347 ± 0.041 |
| **slots** (n=3) | 1,638,656 (590,080 / 1,048,576) | 1638656 | 128 | 4.7414 ± 0.0056 | 1.8219 ± 0.0022 | +0.0972 ± 0.0275 | 5.51e-01 ± 3.47e-02 | 1.24e-01 ± 2.97e-03 | 0.999 ± 0.001 | 0.276 ± 0.008 |
| **bandwidth** (n=3) | 1,894,480 (583,760 / 1,310,720) | 1894480 | 160 | 4.6279 ± 0.0037 | 1.7783 ± 0.0014 | +0.1190 ± 0.0524 | 5.38e-01 ± 5.55e-02 | 1.44e-01 ± 9.40e-03 | 1.000 ± 0.001 | 0.226 ± 0.034 |
| **predicate** (n=3) | 1,902,704 (591,984 / 1,310,720) | 1902704 | 160 | 4.3861 ± 0.0020 | 1.6854 ± 0.0008 | +2.6402 ± 0.0481 | 3.53e-01 ± 4.74e-02 | 7.01e-02 ± 4.79e-03 | 0.998 ± 0.002 | 0.264 ± 0.008 |
| **codebook** (n=3) | 1,894,480 (583,760 / 1,310,720) | 1935440 *(+40960 buffers)* | 160 | 4.7542 ± 0.0054 | 1.8268 ± 0.0021 | +0.0375 ± 0.0157 | 1.19e-01 ± 1.68e-02 | 9.60e-02 ± 1.28e-02 | 0.996 ± 0.001 | 0.272 ± 0.013 |
| **shrink** (n=3) | 1,650,944 (602,368 / 1,048,576) | 1650944 | 128 | 4.7243 ± 0.0100 | 1.8153 ± 0.0038 | +0.0860 ± 0.0303 | 2.29e-01 ± 6.67e-02 | 1.43e-01 ± 8.22e-03 | 0.997 ± 0.001 | 0.199 ± 0.017 |

#### Table C — the rung-5 reconstruction ladder, KL from the model (nats), mean ± sd over the three seeds

| ladder stage | vanilla | slots | bandwidth | predicate | codebook | shrink |
|---|---|---|---|---|---|---|
| `embed_only` | 16.309 ± 0.104 | 16.052 ± 0.097 | 17.061 ± 0.240 | 17.123 ± 0.155 | 15.840 ± 0.210 | 2.888 ± 0.045 |
| `plus_self_attn` | 4.762 ± 0.152 | 4.040 ± 0.026 | 4.071 ± 0.071 | 3.283 ± 0.560 | 3.622 ± 0.011 | 2.814 ± 0.092 |
| `model_bigram` | 0.815 ± 0.007 | 0.911 ± 0.013 | 0.990 ± 0.012 | 1.100 ± 0.019 | 0.867 ± 0.016 | 0.900 ± 0.019 |
| `no_attention_at_all` | 0.922 ± 0.007 | 1.811 ± 0.033 | 1.943 ± 0.084 | 2.739 ± 0.114 | 0.967 ± 0.031 | 1.233 ± 0.050 |
| `past_attn_mean_ablated` | 0.889 ± 0.003 | 1.163 ± 0.033 | 1.298 ± 0.088 | 1.380 ± 0.053 | 0.898 ± 0.008 | 1.031 ± 0.038 |
| `no_mlp` | 4.696 ± 0.446 | 2.262 ± 0.087 | 2.333 ± 0.180 | 2.177 ± 0.098 | 2.021 ± 0.083 | 1.903 ± 0.094 |
| `no_attn_layer0` | 0.559 ± 0.013 | 1.701 ± 0.003 | 1.757 ± 0.174 | 2.271 ± 0.154 | 0.535 ± 0.029 | 0.849 ± 0.023 |
| `no_attn_layer1` | 0.510 ± 0.032 | 0.547 ± 0.043 | 0.715 ± 0.096 | 0.656 ± 0.049 | 0.412 ± 0.019 | 0.651 ± 0.056 |
| `no_mlp_layer0` | 3.883 ± 0.056 | 2.441 ± 0.032 | 2.903 ± 0.349 | 2.153 ± 0.129 | 1.309 ± 0.096 | 1.271 ± 0.119 |
| `no_mlp_layer1` | 0.966 ± 0.054 | 0.502 ± 0.029 | 0.563 ± 0.050 | 0.623 ± 0.074 | 0.411 ± 0.054 | 0.493 ± 0.032 |
| `l1_reads_embedding` | 1.361 ± 0.355 | 1.558 ± 0.485 | 0.825 ± 0.033 | 1.750 ± 0.446 | 0.219 ± 0.032 | 0.367 ± 0.029 |
| `l1_reads_e_plus_attn0` | 1.680 ± 0.136 | 0.373 ± 0.032 | 0.372 ± 0.069 | 0.360 ± 0.027 | 0.113 ± 0.005 | 0.183 ± 0.020 |
| `l1_reads_e_plus_mlp0` | 0.000 ± 0.000 | 0.541 ± 0.039 | 0.533 ± 0.046 | 0.349 ± 0.052 | 0.117 ± 0.018 | 0.229 ± 0.066 |
| `trunc_delta1_only` | 0.470 ± 0.003 | 0.574 ± 0.002 | 0.606 ± 0.005 | 0.504 ± 0.005 | 0.540 ± 0.008 | 0.540 ± 0.008 |
| `trunc_delta_le4` | 0.264 ± 0.003 | 0.334 ± 0.004 | 0.351 ± 0.004 | 0.330 ± 0.002 | 0.325 ± 0.008 | 0.304 ± 0.008 |
| `positional_only_pattern` | 0.318 ± 0.010 | 0.346 ± 0.018 | 0.414 ± 0.036 | 0.273 ± 0.005 | 0.356 ± 0.029 | 0.315 ± 0.018 |
| `no_rotary_pattern` | 3.499 ± 0.102 | 1.354 ± 0.064 | 1.444 ± 0.079 | 0.556 ± 0.048 | 1.347 ± 0.044 | 1.453 ± 0.038 |

#### Table D — effective rank, selection vs content, against the same-shape random null

| architecture | content entropy rank | its random-factored null | ratio | selection entropy rank | its random-table null | ratio |
|---|---|---|---|---|---|---|
| vanilla | 120.57 ± 0.29 | 123.22 ± 0.00 | 0.978 ± 0.002 | 5.55 ± 0.66 | 15.99 ± 0.00 | 0.347 ± 0.041 |
| slots | 31.69 ± 0.03 | 31.71 ± 0.00 | 0.999 ± 0.001 | 4.41 ± 0.13 | 15.99 ± 0.00 | 0.276 ± 0.008 |
| bandwidth | 39.57 ± 0.03 | 39.58 ± 0.00 | 1.000 ± 0.001 | 3.61 ± 0.55 | 15.99 ± 0.00 | 0.226 ± 0.034 |
| predicate | 39.52 ± 0.07 | 39.58 ± 0.00 | 0.998 ± 0.002 | 4.21 ± 0.13 | 15.99 ± 0.00 | 0.264 ± 0.008 |
| codebook | 39.42 ± 0.02 | 39.58 ± 0.00 | 0.996 ± 0.001 | 4.35 ± 0.21 | 15.99 ± 0.00 | 0.272 ± 0.013 |
| shrink | 31.62 ± 0.02 | 31.71 ± 0.00 | 0.997 ± 0.001 | 3.18 ± 0.27 | 15.99 ± 0.00 | 0.199 ± 0.017 |

#### Table E — control and robustness arms

| arm | what it controls | held CE | induction (± probe floor 3 SE) | routing KL [zero, resample] | live slots / read |
|---|---|---|---|---|---|
| `bandwidth_slot32_d2_s0` | embedding pinned to vanilla (stream 128, not 160) | 4.7424 | +0.0962 ± 0.0184 | [0.452, 0.114] | 4.00 |
| `bandwidth_slot32_d2_s1` | embedding pinned to vanilla (stream 128, not 160) | 4.7460 | +0.0797 ± 0.0147 | [0.495, 0.117] | 4.00 |
| `predicate_slot32_d2_s0` | embedding pinned to vanilla (stream 128, not 160) | 4.4830 | +2.4597 ± 0.0188 | [0.335, 0.067] | 4.00 |
| `predicate_slot32_d2_s1` | embedding pinned to vanilla (stream 128, not 160) | 4.4813 | +2.5189 ± 0.0276 | [0.272, 0.074] | 4.00 |
| `slots_gc3e-2_d2_s0` | group-lasso coefficient x1000 | 5.2195 | -0.0163 ± 0.0128 | [0.657, 0.067] | 4.00 |
| `slots_gc3e-3_d2_s0` | group-lasso coefficient x100 | 4.9633 | -0.0324 ± 0.0100 | [0.333, 0.085] | 4.00 |
| `slots_gc3e-4_d2_s0` | group-lasso coefficient x10 | 4.7273 | +0.1424 ± 0.0171 | [0.591, 0.127] | 4.00 |
| `slots_lr0.01_d2_s0` | learning-rate falsifier (Muon 0.01) | 4.7467 | +0.0802 ± 0.0107 | [0.499, 0.116] | 4.00 |
| `slots_lr0.04_d2_s0` | learning-rate falsifier (Muon 0.04) | 4.7498 | +0.0833 ± 0.0103 | [0.548, 0.122] | 4.00 |
| `slots_nolasso_d2_s0` | partition + per-slot norm without the group lasso | 4.7607 | +0.0836 ± 0.0117 | [0.483, 0.112] | 4.00 |
| `slots_nolasso_d2_s1` | partition + per-slot norm without the group lasso | 4.7520 | +0.0999 ± 0.0095 | [0.469, 0.118] | 4.00 |
| `slots_nolasso_d2_s2` | partition + per-slot norm without the group lasso | 4.7696 | +0.0442 ± 0.0059 | [0.474, 0.118] | 4.00 |
| `slots_writeinit_only_d2_s0` | the nonzero decoder init ALONE (n_slots 1, no lasso) | 4.6576 | -0.0095 ± 0.0089 | [3.53e-06, 3.09e-06] | 1.00 |
| `slots_writeinit_only_d2_s1` | the nonzero decoder init ALONE (n_slots 1, no lasso) | 4.6453 | -0.0117 ± 0.0081 | [5.61e-06, 3.04e-06] | 1.00 |
| `slots_writeinit_only_d2_s2` | the nonzero decoder init ALONE (n_slots 1, no lasso) | 4.6571 | -0.0025 ± 0.0071 | [3.37e-06, 2.86e-06] | 1.00 |
| `vanilla_lr0.01_d2_s0` | learning-rate falsifier (Muon 0.01) | 4.6518 | -0.0180 ± 0.0065 | [9.54e-04, 3.18e-04] | 1.00 |
| `vanilla_lr0.01_d2_s1` | learning-rate falsifier (Muon 0.01) | 4.6571 | -0.0013 ± 0.0101 | [6.70e-04, 2.85e-04] | 1.00 |
| `vanilla_lr0.04_d2_s0` | learning-rate falsifier (Muon 0.04) | 4.6718 | -0.0142 ± 0.0108 | [4.38e-08, 4.68e-08] | 1.00 |
| `vanilla_lr0.04_d2_s1` | learning-rate falsifier (Muon 0.04) | 4.6729 | -0.0117 ± 0.0071 | [1.07e-07, 7.38e-08] | 1.00 |

**How to read the routing column.** The `[zero, resample]` pair is the KL from
the true model when layer-0 attention's write is deleted from layer 1's Q/K/V
read *only* (residual untouched, everything downstream recomputed), by zeroing
and by substituting the write the same module produced on a *different*
sequence. It is a fact about what each trained model transmits. It is **not**
evidence that the plain model's weights ignore that channel — see §8, R1.

### 1. The attention-to-attention path: nothing flows through it in the plain model — but the reason is MAGNITUDE, not a closed channel

> **CORRECTED BY THE INDEPENDENT ROUND-2 REVIEW (`tf_reviewer_round_2.json`,
> objection R1).** The causal numbers below are unchanged and replicate on
> three seeds. What is **retracted** is the mechanistic gloss that the plain
> model's layer 1 is *blind* to layer-0 attention, or that the variants
> *opened* something in weight space. Under a matched-displacement probe the
> plain model turns out to be the **most** sensitive of the six to layer-0
> attention's direction. See §8.

FINDING 8 measured that layer 1's read is 99.9% the first MLP's write, and
concluded the attention→attention path induction needs is numerically closed.
**That conclusion survives, but the metric it was quoted from does not.** Norm
share is not invariant to a change of normalisation convention, and the slot
variants change exactly that. So the verdict is re-derived from an intervention:
delete each upstream write from layer 1's Q/K/V **read only**, leave the
residual stream untouched, recompute everything downstream, and score the KL
from the true model. Both flavours are reported — zeroing (the lower bound) and
**resampling** (substitute the write the same module produced on a *different*
sequence, so the substituted vector is on-distribution by construction and the
slot is just as full as before).

| deleting layer-0 attention from layer-1's read | zero | resample |
|---|---|---|
| vanilla seed 0 / 1 / 2 | 2.4e−5 / 9.8e−6 / 4.7e−6 | 5.5e−6 / 4.0e−6 / 3.5e−6 |
| slots | 0.574 | 0.123 |
| bandwidth | 0.600 | 0.150 |
| predicate | 0.352 | 0.071 |
| codebook | 0.113 | 0.108 |
| shrink | 0.301 | 0.148 |

Even the *harshest* vanilla number is 2×10⁴ times smaller than the *gentlest*
variant number. For scale, deleting the first MLP's write from the same read
costs vanilla 1.796 nats — so in the plain model layer 1 reads one upstream
module and, to five decimal places, not the other.

**The normalisation confound, tested rather than argued** (`norm_confound_control`,
demanded in review). Impose a 4-way slot norm on the *plain* model at analysis
time — same weights, no retraining — and recompute every composition statistic.
Its pattern sensitivity to layer-0 attention moves from 0.00424 to 0.00434, a 2%
change, **not** to the slot variant's 1.27. So the sensitivity is not a
normalisation artifact. Two statistics are withdrawn as evidence and kept only
as context: the *post*-norm share is forced to 1/G by construction, and the
*pre*-norm share cannot move under the control at all (it is a statistic about
stream magnitudes, which is precisely the thing the training pressure the slot
norm removes was shaping).

### 2. It is not merely used — the algorithm RUNS on it

A used route that carries nothing in particular would be a weak result. The
route decomposition that overturned this program's first induction-circuit claim
is applied unchanged: remove layer-0 attention from layer-1's read only, from
MLP-0's input only, from both, and outright, then re-run the induction battery.
Fractions of the induction score removed, **all three seeds**:

| | via layer-1's read | via MLP-0's input | write deleted |
|---|---|---|---|
| **vanilla, width 256** (has induction, +0.084 / +0.097 / +0.101) | **−0.005 / −0.001 / −0.002** | 1.44 / 1.34 / 1.44 | 1.44 / 1.34 / 1.44 |
| slots | **1.17 / 0.90 / 1.27** | 0.24 / 0.28 / 0.17 | 0.96 / 0.81 / 0.87 |
| bandwidth | **1.11 / 1.08 / 1.27** | 0.37 / 0.30 / 0.28 | 1.00 / 1.04 / 1.14 |
| shrink | **1.53 / 1.31 / 1.32** | 0.17 / 0.23 / 0.24 | 1.46 / 1.27 / 1.34 |
| codebook | 1.24 / 1.53 / 1.87 | 1.23 / 1.53 / 1.87 | 0.95 / 1.14 / 0.91 |
| predicate | 0.15 / 0.12 / 0.05 | 0.20 / 0.18 / 0.20 | 0.58 / 0.57 / 0.59 |

(The plain model's own width-128 fractions are **undefined** — its induction
score, the denominator, is null — which is why the control row is width 256.)

**Specificity control, demanded by the round-2 review** (§8 R1): a 60% read
perturbation could kill everything indiscriminately. It does not. The same
intervention removes 117% of the *induction* score in `slots` but only **15%** of
the order-free **bag** score; `shrink` 153% vs 5%; `codebook` 124% vs 40%;
`bandwidth` 111% vs 54%.

(Fractions above 1 mean the intervention drives the score below zero, not that
more than all of it was removed.) The plain model and the slot variants are
**mirror images**: the plain model's induction signal reaches layer-1 attention
entirely through the feed-forward block and not at all through the read; in
slots, bandwidth and shrink it is the opposite. Codebook's two routes are not
separable and that is recorded as a limitation, not resolved. Predicate does not
use either route because it does not need them (see 3).

**The decisive control, now at three seeds:** the plain model at width 256 *does*
have induction (+0.0841 / +0.0965 / +0.1007, 4.9–16.6× its own power floor) and
its attention-to-attention path still carries nothing (2.3e−5 / 3.6e−5 / 4.7e−5
zeroed, 5.4e−6 / 6.4e−6 / 7.4e−6 resampled), with the signal travelling through
MLP-0 (route fraction 1.34–1.44) and not through the read (−0.005 to −0.001). So
"the route carries something" and "the model inducts" are independent
properties, and the variants change both.

### 3. Predicate: induction from SIXTEEN NAMED SCALARS, and the positional work moves off the rotary

`predicate` scores +2.593 ± 0.027 on the synthetic probe — 85× its power floor,
31× the largest induction this program has ever measured (vanilla width 256,
+0.084) — and +1.503 (t = 34) on the natural-text bag-preserving swap. It is
also the *cheapest* model in the slice, 0.267 nats below vanilla.

All of it is one named term. `MATCH_prev[i,j] = 1[tok_{j-1} == tok_i]` attends
from the current token to the position *after* an earlier copy of it: a complete
induction head in one layer, handed over as one scalar per head. Zeroing those
16 scalars (`pred_b`, 2 layers × 8 heads) in place:

| named terms zeroed | induction |
|---|---|
| none | +2.5934 |
| previous-token match `b` | **+0.0330** (98.7% removed) |
| same-token match `c` | +2.4977 (3.7%) |
| positional profile | +3.1224 (*negative* removal) |
| all three | **−0.0028** — exactly vanilla's null |

The learned bilinear branches contribute **no** induction on their own. No
single head carries it either — every one-head knockout leaves 2.26–2.61, so it
is strongly sub-additive, as registered. A second legible consequence: removing
the rotary costs vanilla 3.429 nats of KL and predicate only 0.532, because the
named positional profile has absorbed the positional work the rotary branches
were doing.

### 4. WHICH mechanism? The write partition and per-slot norm — not the lasso, and not the write init

`slots` changes four things at once versus vanilla, and one of them is a
confound with nothing to do with interpretability: vanilla **zero-inits** its
decoders while every variant inits them nonzero. The reduction gate proves
`slots(n_slots=1, lasso 0, zero writes)` is *bit-exact* vanilla, so an arm with
n_slots 1 and no lasso differs from vanilla by the init alone.

**All four arms now at three seeds** (the round-2 review retrained the two
mechanism arms at seeds 1 and 2; they were seed 0 only before):

| arm | nonzero write init | partition + per-slot norm | group lasso | CE (s0/s1/s2) | induction (s0/s1/s2) | A0 out of l1 read [z, r], s0 |
|---|---|---|---|---|---|---|
| vanilla | — | — | — | 4.6512 / 4.6501 / 4.6377 | −0.0138 / −0.0022 / +0.0059 *(all below floor)* | [2.4e−5, 5.5e−6] |
| write-init only | ✓ | — | — | 4.6576 / 4.6453 / 4.6571 | −0.0095 / −0.0117 / −0.0025 *(all below floor)* | [3.5e−6, 3.1e−6] |
| slots, no lasso | ✓ | ✓ | — | 4.7607 / 4.7520 / 4.7696 | **+0.0836 / +0.0999 / +0.0442** *(all above floor)* | [0.483, 0.112] |
| slots | ✓ | ✓ | ✓ | 4.7418 / 4.7356 / 4.7468 | **+0.1129 / +0.1133 / +0.0654** *(all above floor)* | [0.574, 0.123] |

**The nonzero write init explains none of it, at three seeds** (CE inside the
0.0074 seed spread, induction below its own power floor at every seed, route
still carrying 3.4–5.6e−6). **The write partition plus per-slot RMSNorm is the
whole mechanism**, also at three seeds. The in-loss group lasso adds +0.02 to
+0.03 of induction on top and is not necessary; §9 shows it is also not doing
what it was added to do.

The natural reading is not that the partition *enables* a route, but that it
**removes the plain model's option to collapse one**. In vanilla the first
attention block writes with norm 9.4 into a stream whose last write has norm
6931 — a factor of 740 — so its contribution is renormalised into
invisibility both at layer 1's read and at the readout (logit share 0.0002).
Give each module a private slot that is separately renormalised and that
collapse is no longer available; the model that results uses the channel and
gets an algorithm out of it. The plain model at width 256 shows it is not a
capacity limit: with the shared stream it still routes around the channel even
when it *has* induction to route.

### 5. Same-or-different, question by question

1. **Rung-5 ladder.** The gross shape is preserved in all six (bare embedding
   worst, then self-attention-only, then the model's own bigram table) but the
   levels move a lot and **there is one genuine reordering**: in `predicate`,
   deleting all attention costs 2.608 nats against 2.174 for deleting all MLPs —
   attention is worth more than the feed-forward, which is true in no other cell
   in this program. Registered prediction P1 said no variant would reorder the
   ladder: **REFUTED by predicate.** The MLP-over-attention ratio falls from 4.6
   (vanilla) to 1.26–2.1 (slots/bandwidth/codebook/shrink) and inverts to 0.83
   (predicate).
2. **Composition budget.** Answered above: the path is shut in vanilla to five
   decimals and open in all five variants, causally and normalisation-invariantly.
3. **Induction.** Present at width 128 in `slots`, `bandwidth`, `predicate` and
   `shrink` at **all three seeds** (each above its own probe power floor) and in
   `codebook` at **two of three** (seed 2 reads +0.0228 against a floor of
   0.0249 — found by the round-2 review, §8 R4; `codebook`'s natural-text swap
   excess is positive at all three seeds). Absent in vanilla at that width
   across three seeds and three learning rates. Registered prediction P3 said absent in
   A/B/C/E/F and present only in D: **the D half was right and for the right
   mechanism; the A/B/C/E/F half was wrong on B, C, E and F.**
4. **Selection vs content.** Selection stays low rank everywhere (0.19–0.36 of
   its random-table null; predicate and shrink lowest, as registered). **No
   variant moves content off its null** — 0.98 to 1.00 of the same-shape
   random-factored null in all six. P4 held. *A first draft of this number said
   slots and shrink sat at 0.42 and 0.40; that was the masking, not the content
   — see the correction below.* **Detection threshold, added by the round-2
   review (R5):** this is a null result, so the detector was calibrated by
   planting content confined to an *r*-dimensional input subspace. It reads
   0.02–0.09 of the null at *r* = 2, 0.07–0.30 at *r* = 4, 0.27–0.83 at *r* = 8,
   0.80–0.96 at *r* = 16 and 0.95–0.99 at *r* = 32 — indistinguishable from the
   models' own 0.98–1.00. So the supported claim is **"content is not confined
   to fewer than roughly 8–16 of the stream's 128–160 input directions"**, not
   "content has no structure".
5. **Ablation ranges.** Every knockout is quoted as [zero, resample]. P5 said
   resample would be harsher everywhere: **REFUTED.** For the layer-0 attention
   *write* in `slots` the order inverts (zero 1.72, resample 0.75) because a
   zeroed slot under per-slot RMSNorm is far more off-distribution than a zeroed
   contribution to a shared stream. The distribution-shift share of the layer-0
   knockout is 0.12 in vanilla and 0.56 in slots.

### 6. What the mechanisms actually delivered — including where they did not

The variants are supposed to buy legibility, not only a different computation.
Measured (`mechanism` block of each cell JSON), three of the four promises are
only half kept:

* **The in-loss group lasso does not prune.** Its whole objective is to empty
  slots out of each read matrix, and at coefficient 3e−5 it empties none: every
  one of the 28 read matrices in every slot variant keeps all four slot groups
  above 1% of its mass, and the shares sit near 0.25. `mean_live_slots_per_read`
  is 4.00 of 4 in all five. The promised sparse wiring diagram is **not**
  delivered at this size.
* **But the wiring table is still informative, by its tilt rather than its
  zeros — and the tilt agrees with the causal answer.** Layer 1's *queries and
  keys* put their largest share on slot 0, the slot layer-0 attention writes
  into (0.40–0.47 in slots/bandwidth/shrink against ~0.20 on the MLP's slot),
  while layer 1's *values* put only 0.11–0.14 there. That is a
  read-it-off-the-weights version of the causal result in §1–2, and vanilla has
  no such table at all because it has no partition. Caveat that must travel with
  it: the token remnant is full width, so a slot contains *an embedding chunk
  plus one module's write*, and shares on slots not yet written are reading the
  embedding.
* **The codebook's discreteness buys no legibility here.** All 256 atoms are
  used in all four quantised modules, usage entropy is 5.32–5.43 nats against a
  maximum of 5.545, and it takes 182–204 atoms to cover 90% of assignments. The
  dictionary is as flat as a random one. Registered prediction P4's sub-clause
  ("fewer than half the atoms carry 90%") is **REFUTED.**
  **Round-2 review (R3b), two corrections, both against our own design.**
  (i) *The flatness is in the data, not the mechanism, and that half is now
  stronger:* a k-means dictionary of the same 256 atoms, fit on those very
  activations and free to be as unequal as it likes, is **flatter still** —
  usage entropy 0.98 of maximum against the trained codebook's 0.83, 202 atoms
  for 90% against 111. (ii) *The published error figure is retracted.* "22% at
  block 0, 39% at block 1" is the residual over the **full** module input, of
  which only one or two of four slots are ever quantised; **on the slots
  actually quantised the relative error is 0.77–0.85.** Worse, the dictionary
  is under-trained: with the model's own matching pursuit at the model's own
  *k*, k-means at the same 256 atoms reaches **0.43–0.51**, and the trained
  codebook (0.846) is **no better than the random unit dictionary it was
  initialised from** (0.816). A same-budget PCA to *k* dimensions also beats it
  (0.70–0.77), so the mechanism fails the README's "beat a same-parameter-count
  alternative" test. The honest sentence is that the codebook arm pays 0.097
  nats for a dictionary that is both flat *and* barely trained at this size.
* **The shrinking channel: half of this claim was wrong, and a spectral
  statistic is why.** The original reading was "the remnant projections are near
  full rank for the width they are given (entropy rank 62.2 of 64 at block 1,
  30.9 of 32 at the readout), so the floor is doing the compressing". Entropy
  rank is not causal, so the round-2 review (R3c) truncated each projection to
  its top-*r* singular directions and scored the held KL, against a random
  subspace of the same rank, on three seeds:
  * **block-1 remnant (64 of 128): the claim SURVIVES, causally.** Rank 32 still
    costs 0.28–0.33 nats, rank 48 costs 0.12–0.16, only the full 64 reaches 0.
    And the *particular* subspace is doing work — a **random** 64-dimensional
    subspace costs 0.45–0.50 nats where the model's own costs 0 — so the model
    has selected a specific 64-dimensional summary without making it low rank.
  * **readout remnant (32): the claim is RETRACTED.** Truncating it to **rank
    one** costs 0.022–0.023 nats, and a random rank-1 subspace costs 0.023–0.026
    — statistically the same. Thirty-one of its thirty-two directions are
    causally worthless and the whole projection is worth ≤0.023 nats. Entropy
    rank 30.9 of 32 was describing a spectrum that carries almost nothing.
* **The predicate profiles are genuinely readable.** Each layer-0 head's named
  positional term peaks at a specific relative distance — heads 1–4 at distance
  0, heads 5–7 at distance 1, head 0 at distance 2 — which is a per-head distance
  kernel you can print, and it is why removing the rotary costs this variant
  0.532 nats against vanilla's 3.429. (Only the peak *location* is quoted: the
  profile is one factor in a product, so its sign is gauge and is not
  interpreted. 40–52% of the absolute mass sits beyond distance 16, so the
  kernel is peaked but not local.)

### 7. Per-head values are ranges, and the ranking does not survive the harsher ablation

Every head is ablated both ways. In the plain model resampling is harsher at 14
of 16 heads, and — the part that matters for anyone reading a head ranking off a
zeroing experiment — **the head ordering obtained by zeroing does not survive
the switch at either layer** (the top head does). The single-head costs also sum
to *less* than the whole-layer cost (0.52 at layer 1, 0.69 at layer 0), so the
heads are complementary rather than redundant — the opposite of the registered
head-compensation prediction, which expected the sum to over-count.

### 8. INDEPENDENT ROUND-2 REVIEW — what an outside reviewer changed

Full record with every measurement: **`tf_reviewer_round_2.json`**, raw numbers
in `tf_round2_measurements.json`, new arms in `tf_round2_train.log` /
`tf_round2_trainb.log`. Round 1 (`tf_variant_reviewer_round_1.json`) was the
analyst's self-red-team; the README requires a second round by someone who did
not produce the results, and this is it. Seven objections, each answered by a
measurement that was run.

**R1 — the routing measurement is not architecture-fair, and the mechanism is
RETRACTED (the biggest change).** The deletion measurement records the
displacement it causes: removing layer-0 attention from layer-1's read moves
that read by **0.2–0.3% in vanilla and 60–68% in every variant**. KL is
quadratic in a small displacement, so a 210× displacement ratio predicts a ~4×10⁴
KL ratio *by itself*. Replacement measurement: a **matched-displacement
directional probe** in the post-norm read space — displace layer 1's read by
exactly 5/10/20% of its own norm along the direction each upstream module
contributes, identical in every arm, recompute everything downstream, score KL.
Result, at a 10% displacement along layer-0 attention's (orthogonalised)
direction:

| | vanilla | slots | bandwidth | predicate | codebook | shrink |
|---|---|---|---|---|---|---|
| KL, seeds 0/1/2 | **0.0270 / 0.0172 / 0.0163** | 0.0203 / 0.0140 / 0.0160 | 0.0124 / 0.0114 / 0.0189 | 0.0132 / 0.0146 / 0.0125 | 0.0121 / 0.0127 / 0.0138 | 0.0108 / 0.0099 / 0.0096 |
| × a random direction | **5.8–8.1** | 3.7–5.2 | 3.0–5.0 | 5.3–5.6 | 1.3–1.5 | 2.5–3.0 |

**The plain model is the *most* sensitive of the six** to layer-0 attention's
direction, per unit of read displacement — and in vanilla the probe is
uncontaminated by magnitude (|cos(direction, read)| = 0.001, against 0.33–0.50
in the variants). Consistency check: extrapolating vanilla's local gain
quadratically down to its actual 0.18–0.32% displacement predicts **2.73e−5 /
1.06e−5 / 5.01e−6** against the measured **2.42e−5 / 9.80e−6 / 4.66e−6** — within
8–13%. So the plain model's ~zero routing KL is precisely what an ordinary,
fully open sensitivity produces at a vanishing displacement.

*What stands:* the causal transmission numbers, on three seeds. *What is
withdrawn:* "the plain model shuts the path" as a statement about weight space.
The supported statement is a **magnitude** statement — the plain model
renormalises its own first attention write down to 0.3% of layer 1's read, so a
fully sensitive channel carries nothing; per-slot RMSNorm removes that option
and forces the write to a quarter of the read, at which point the *same*
sensitivity delivers 0.1 nats. This is §4's "removes the option to collapse"
framing, and it is now the only framing the data supports.

*A second-order confirmation, from a probe that turned out NOT to be fair:*
injecting the same fraction of the **pre**-norm stream norm costs vanilla 0.0034
nats along a random direction and the variants 0.13–0.62, because the variants'
raw slot norms are wildly unequal and per-slot RMSNorm only equalises them
afterwards. That flavour is recorded and explicitly not quoted; it is the same
confound showing up in a second place.

*Specificity control for §2, which the same objection threatens* ("a 60%
perturbation kills everything"): the intervention removes 117% of the induction
score in `slots` but only **15%** of the order-free **bag** score; `shrink`
153% vs 5%; `codebook` 124% vs 40%; `bandwidth` 111% vs 54%. Induction dies
while the bag effect largely survives, so the route is content-specific. §2
stands, with bandwidth quoted as the weakest case.

**R2 — what else could produce induction at half the width, now that the
learning rate is dead.** Four alternatives, all measured per parameter block by
re-instantiating each model at its own seed (the init is bit-reproducible):
*initialisation scale* — vanilla, slots, shrink and the write-init-only arm have
**bit-identical** init RMS in every read matrix; the only difference is the
decoder (0 vs 0.0172), which the write-init-only arm isolates and which is null.
*Effective learning rate per block* — Muon's update is orthogonalised, so step
size is set by the learning rate and not by gradient scale; and measured,
vanilla travels **further** from init at layer-0 attention than slots does
(relative distance 12.9 vs 6.9 at `c_q`, 14.7 vs 5.9 at `c_k`), so the plain
model moves its first attention block *more* and still does not induct.
*Embedding capacity* — slots has vanilla's embedding exactly, and the effective
rank of the embedding update is 0.81–0.84 of maximum in every arm.
*Trainable directions used* — effective rank of the update differs by a few
percent and flips sign block to block (layer-0 `c_q` 0.44 vanilla vs 0.55 slots;
layer-0 `c_v` 0.72 vs 0.62). **None of the four separates the arms; two point the
wrong way.**

**R3 — the three unflattering findings, re-examined because they are about our
own designs.** The codebook and shrink outcomes are in §6 above (one half of
each survived, one half was retracted or restated). The lasso arm is in §9.

**R4 — what still rests on one seed.** Now three seeds and replicating: the
routing KL both flavours, induction, the natural-text swap, the route split, the
**predicate ladder reordering** (attention knockout 2.608 / 2.805 / 2.805 against
MLP 2.174 / 2.080 / 2.277 — round 1's C11 is closed), the predicate named-term
ablation (98.7% / 99.1% / 98.8% removed by the 16 previous-token scalars), and
the flat bag score. The **decisive width-256 control** was re-analysed at seeds
1 and 2 through the same pipeline: induction +0.084 / +0.097 / +0.101 with the
path still shut (2.3–4.7e−5 zero, 5.4–7.4e−6 resample) and the signal travelling
through MLP-0 (route fraction 1.34–1.44) and not the read (−0.005 to −0.001).
**Found by this review and not previously reported: `codebook` fails its own
power floor at seed 2** (+0.0228 against a 3-SE floor of 0.0249, 0.9×; seeds 0
and 1 are 6.3× and 4.2×). So "all five variants acquire induction at width 128"
is 3/3 for slots, bandwidth, predicate and shrink and **2/3 for codebook** on
the synthetic probe — though 3/3 on the natural-text swap (+0.168 / +0.142 /
+0.140 at t = 8.3–8.7 against vanilla's +0.100 / +0.092 / +0.117 at t = 5.8–6.8).

**R5 — the content null result had an uncalibrated detector.** Handled in §5
item 4: the statistic is blind above a planted input rank of about 16 of 128, so
the claim is re-quoted with that threshold.

**R7 — predicate's +2.64 is a handover, not a discovery.** Already stated in §3;
the reviewer's addition is that the *table* must carry the flag, because tables
travel without their paragraphs. Done in Table A/B.

### 9. The group lasso, re-based on a coefficient sweep instead of one setting

§6 reported "the in-loss group lasso empties no slot" from a single coefficient
(3e−5) inherited from another program at another scale. That is exactly the kind
of unflattering claim about our own design that deserves the harshest test, so
the round-2 review retrained `slots` at 10×, 100× and 1000× that coefficient and
put all four through the same analysis:

| group-lasso coefficient | total group norm (14 read matrices) | mean live slots / read | smallest single group share | held CE | induction (floor) |
|---|---|---|---|---|---|
| 0 (no lasso) | 2706.0 | 4.00 / 4 | 0.145 | 4.7607 | +0.0836 (0.0117) |
| **3e−5** (the reported arm) | 1682.8 | 4.00 / 4 | 0.126 | 4.7418 | +0.1129 (0.0128) |
| 3e−4 | 375.7 | 4.00 / 4 | 0.099 | **4.7273** | **+0.1424** (0.0171) |
| 3e−3 | 37.6 | 4.00 / 4 | 0.078 | 4.9633 | −0.0324 (0.0100) |
| 3e−2 | **2.94** | 4.00 / 4 | 0.017 | 5.2195 | −0.0163 (0.0128) |

**The penalty works; it just does not select.** Driving the total group norm down
by a factor of **920** empties nothing: all 56 slot groups stay above 1% of their
matrix's mass at every coefficient, and the smallest share only falls from 0.145
to 0.017. What breaks first is the model — CE degrades by 0.22–0.48 nats and the
induction the architecture buys is destroyed (at or below its own power floor) at
3e−3 and 3e−2, *before* a single slot is emptied. At this size a group lasso
shrinks all groups proportionally instead of choosing among them; the promised
sparse wiring diagram is not available at any coefficient that leaves a working
model. **Claim SURVIVES and is strengthened.**

Two corrections that fall out of the sweep: the write-up said "28 read matrices",
but the measured object is **14** per model (7 per block × 2 blocks) carrying 56
slot groups; and **3e−5 was not the best coefficient** — 3e−4 is better on both CE
and induction, so the primary `slots` arm is quoted at a slightly suboptimal
setting and its 0.091-nat cost against vanilla is an overestimate (0.076 at 3e−4).

### 10. DOCUMENTED LIMITATIONS after the fix round — what is still not settled, and why

Nothing below is a to-do; each is a limitation with its reason. The README
forbids leaving anything in a "we will check later" state, so this is the
complete residue after the round-2 review and its fix round.

* **`codebook`'s route attribution is not separable.** Its induction dies at
  1.24–1.87 through layer-1's read *and* 1.23–1.87 through MLP-0's input, at all
  three seeds. No route claim is made for that arm. Its transmission numbers
  stand on their own.
* **`codebook`'s synthetic induction is 2 of 3 seeds above its own power floor**
  (seed 2: +0.0228 against 0.0249). The natural-text swap excess is positive at
  all three (+0.054 to +0.082 over its depth-1 null), so the arm is not null —
  but the synthetic headline is quoted as 2 of 3 and not averaged into "all
  five".
* **The content-spectrum null result has a detection threshold.** The statistic
  cannot see structure above a planted input rank of about 16 of 128 (§5.4).
  Anything the models do with content in a 16-to-128-dimensional subspace is
  invisible to it. Closing that needs a different detector, not more seeds.
* **The matched-embedding (`_slot32`) arms and the learning-rate arms are at two
  seeds, not three** (0 and 1). Both are one-directional controls — they exist
  to show an effect is *not* explained by embedding size or learning rate, and
  both agree at both seeds — so a third seed would strengthen, not decide.
* **The gain-normalisation in §8 R1 is conservative toward the variants, not
  toward vanilla.** A 60% read displacement is past the quadratic regime, so
  extrapolating the variants' local gain over-predicts their deletion KL by
  1.2–3× while vanilla's is accurate to 8–13%. The conclusion (vanilla's
  sensitivity is at least as large) is therefore a lower bound on the size of
  the correction, not an upper one.
* **`bandwidth` is the weakest case for the content-specificity of the route.**
  Its read deletion removes 111% of the induction score but also 54% of the bag
  score, against 15% (slots) and 5% (shrink). Quoted with that figure attached.
* **The partition dose-response was not obtained.** The `--n-slots 2` arm is
  invalid (§8 R8: it silently muted the entire second block) and has been
  discarded to `discarded_arms/`. A real dose-response needs a mechanism that
  varies the partition without changing which modules can write — that is a new
  design, not a rerun, and it is not claimed here.
* **The trained codebook is under-trained, so the codebook arm does not test
  discreteness at its best.** k-means at the same size halves its error (§6). Any
  statement about what discrete codes buy at this scale is a statement about
  *this* EMA codebook, not about discreteness.
* **`slots` is quoted at a suboptimal lasso coefficient** (3e−5; 3e−4 is better on
  both CE and induction), so its CE cost against vanilla is an overestimate. The
  primary arms were deliberately not per-arm-tuned — matched optimisers are the
  protocol — and re-tuning every arm would break the matching, so this is
  recorded rather than fixed.

### Arithmetic dressed as a finding, caught before it was reported

The MLP content spectrum for the **masked-decoder** variants (`slots`, `shrink`)
was initially measured over all 128 output rows of the folded tensor. But
`write_out` discards every row outside the module's own slot, so 96 of 128 rows
never receive a gradient and sit at their init — measured row norms are 100.5
inside slot 1 and 4.7 outside. The all-rows spectrum duly reported "entropy rank
51 against a null of 123", which is 32/128 and nothing else. Restricted to the
live rows with a shape-matched null, every variant lands at 0.98–1.00 of its
null. Small decoders are physically slot-sized, so nothing changed for them.

### What this cost, stated in the same breath

Four of the five variants are *worse* on loss: +0.085 to +0.097 nats, i.e. 11–13
vanilla seed standard deviations. `bandwidth` is 0.025 better and `predicate`
0.267 better. So the honest sentence is: **the slot architectures buy a
different computation for about a tenth of a nat, and the predicate architecture
buys a different computation and a quarter-nat of loss at the same time.**

---

## 2026-08-08 — FINDING 7 (DEPTH 2): "attention is inert" was a property of the LADDER, not of the model

**Verdict: the depth-1 headline does not survive its own adversarial test, at
either depth.** The claim came from one increment — the gap between the
bigram-only reconstruction and the no-attention-at-all knockout — and that
increment is not attention's marginal value. It is the gap between two
*different* reduced models, one of which (the bigram) still contains the
self-attention term and has already frozen the context away.

The same two components, added in both orders, on held text, KL from the true
model (`*_order.json`, `tf_interp2.ladder_order`; mean ± sd over 3 seeds):

| cell | attention added FIRST | attention added LAST | ratio | MLP first | MLP last |
|---|---|---|---|---|---|
| depth 1, w32 | 2.030 ± 0.028 | 0.290 ± 0.027 | **7.0** | 8.549 | 6.810 |
| depth 1, w64 | 3.460 ± 0.231 | 0.475 ± 0.017 | **7.3** | 11.779 | 8.794 |
| depth 1, w128 | 4.659 ± 0.151 | 0.707 ± 0.009 | **6.6** | 15.158 | 11.206 |
| depth 1, w256 | 4.074 | 0.939 | 4.3 | 17.923 | 14.787 |
| depth 2, w32 | 4.224 ± 0.050 | 0.371 ± 0.012 | **11.4** | 8.033 | 4.180 |
| depth 2, w64 | 7.670 ± 0.522 | 0.617 ± 0.006 | **12.4** | 11.564 | 4.510 |
| depth 2, w128 | 11.633 ± 0.473 | 0.941 ± 0.007 | **12.4** | 15.351 | 4.659 |
| depth 2, w256 | 15.561 | 1.229 | 12.7 | 18.541 | 4.208 |

Readings, including the ones that cost us a headline:

* **No single number is "what attention is worth".** It ranges over a factor of
  4–13 depending only on where in the ladder it is added. The order-free
  Shapley average is the honest scalar; the depth-1 mailbox number (0.04 nats)
  is neither marginal — it is smaller than *both*.
* **What actually changes with depth is attention's STANDALONE capability, not
  its necessity.** Attention-with-no-MLPs goes from KL 8.88 (depth 1, w64) to
  4.55 (depth 2, w64): two attention layers compose into something twice as
  good on their own. Its marginal on top of the MLPs barely moves
  (0.47 → 0.61). The MLPs still do the same job, so the second attention layer
  is mostly *redundant capability*, not new function.
* **Under an on-distribution (resample) ablation attention is worth 2-3x more
  than the zeroing says** — 1.12/1.44 nats at depth 1 widths 128/256 and
  1.51/2.01 at depth 2 — so every "attention is cheap" number in this program,
  including the ones above, is a LOWER bound. See FINDING 8.
* **The depth-1-style increment reproduces at depth 2 and is still small**
  (no-attention-at-all minus bigram: 0.032 / 0.049 / 0.107 / 0.131 at widths
  32/64/128/256). Registered prediction `d2_attention_not_inert` is **half
  right**: the two framings do continue to disagree, as predicted, but the
  absolute knockout cost at width 64 is 0.61, not the ">1.0 nats" registered.
  **Refuted on the number, confirmed on the mechanism.**

The full depth-2 ladder (KL from the model, held text, mean ± sd over 3 seeds;
width 256 is one seed):

| stage | d2 w32 | d2 w64 | d2 w128 | d2 w256 |
|---|---|---|---|---|
| embed only | 8.44 | 12.21 | 16.31 | 19.77 |
| model's own bigram (weights-only table) | 0.333 ± 0.008 | 0.559 ± 0.001 | 0.815 ± 0.007 | 1.058 |
| no attention at all | 0.366 ± 0.012 | 0.608 ± 0.007 | 0.922 ± 0.007 | 1.189 |
| past attention mean-ablated | 0.357 ± 0.015 | 0.588 ± 0.003 | 0.889 ± 0.003 | 1.202 |
| no MLP (both) | 4.19 ± 0.05 | 4.55 ± 0.50 | 4.70 ± 0.45 | 4.18 |
| pattern replaced by its distance profile | 0.248 | 0.268 | 0.318 | 0.418 |
| rotary removed | 1.81 | 3.00 | 3.50 | 3.71 |

CE and bits/byte (BPE V=8192, 3.755 bytes/token): depth 2 reaches 5.3166 /
4.9124 / 4.5503 / 4.2446 nats at widths 32–256 (2.043 / 1.888 / 1.748 / 1.631
bits per byte), against depth 1's 5.4130 / 5.0477 / 4.7234 / 4.4613. **A second
layer buys 0.10–0.22 nats — less than one width doubling buys** (0.37).

---

## 2026-08-08 — FINDING 8 (DEPTH 2): layer 1 reads the MLP, not the attention — the composition channel is 0.1–0.4% wide

The two attention layers, deleted separately (KL from the model; the deletion
is a full re-run of the folded pipeline, so everything downstream responds):

| cell | delete layer-0 attention | delete layer-1 attention | delete both | sum of the two |
|---|---|---|---|---|
| w32 | 0.123 ± 0.016 | **0.232 ± 0.017** | 0.366 | 0.355 |
| w64 | 0.253 ± 0.020 | **0.334 ± 0.014** | 0.608 | 0.587 |
| w128 | **0.559 ± 0.013** | 0.510 ± 0.032 | 0.922 | 1.069 |
| w256 | **0.889** | 0.621 | 1.189 | 1.510 |

**Registered prediction `d2_layer_split` REFUTED**: we registered that layer 0
dominates at every width. Layer *1* dominates at widths 32 and 64, and under
the zero-ablation the ordering appears to flip at 128. And the two deletions
are *super*-additive at 32–64 (joint > sum: the layers back each other up) and
*sub*-additive at 128–256.

**But the flip is an artifact of the ablation, and the reviewer round caught
it.** A zeroed write is off distribution, so a **resample ablation** was added
(`resample_ablation`): replace the layer's attention write with the write that
same layer produced on a *different* sequence — a real output of that module,
on distribution by construction.

| cell | layer 0: zero → resample | layer 1: zero → resample | both: zero → resample |
|---|---|---|---|
| d1 w128 | 0.703 → **1.118** | — | 0.703 → **1.118** |
| d1 w256 | 0.939 → **1.435** | — | 0.939 → **1.435** |
| d2 w32 | 0.129 → 0.215 | 0.232 → **0.473** | 0.371 → **0.667** |
| d2 w64 | 0.260 → 0.376 | 0.336 → **0.594** | 0.617 → **1.007** |
| d2 w128 | 0.566 → 0.535 | 0.520 → **0.861** | 0.941 → **1.510** |
| d2 w256 | 0.905 → 0.782 | 0.644 → **1.075** | 1.229 → **2.013** |

Two consequences, both against our own earlier statements:

* **Zeroing was the GENTLER intervention almost everywhere.** The resample cost
  exceeds the zero cost at 13 of 14 layer-cells, so the knockout numbers quoted
  above (and at depth 1) *understate* attention's value rather than inflating
  it with distribution shift. The only exceptions are layer 0 at widths 128–256,
  where 12–14% of the zeroing cost is distribution shift.
* **The layer ordering does NOT flip.** Under the on-distribution ablation,
  layer-1 attention costs more than layer-0 attention at **every** width. The
  flip at 128 was a property of the zeroing, and the honest statement is
  "layer 1 carries more, and the zero-ablation understates that at large
  widths."

**What layer 1 reads** (`composition_budget`, held text). Layer 1's module
input is `rms(e + A0 + M0)`; the shares of that vector's norm, and the relative
change in layer 1's own attention pattern when each write is deleted **from the
read only** (the residual is untouched, so nothing else moves):

| cell | share of read: e | share: layer-0 attention | share: MLP-0 | pattern change without layer-0 attention | without MLP-0 |
|---|---|---|---|---|---|
| w32 | 0.37% | **0.075%** | 99.98% | **0.14%** | 145% |
| w64 | 0.31% | **0.114%** | 99.98% | **0.19%** | 124% |
| w128 | 0.31% | **0.227%** | 99.96% | **0.33%** | 126% |
| w256 | 0.31% | **0.416%** | 99.91% | **0.60%** | 121% |

And the causal version, in the ladder: substituting `rms(e + M0)` for layer 1's
read — i.e. deleting layer-0's attention write from what layer 1 sees —
reproduces the model at **KL 0.0000 at every width and seed**. Substituting
`rms(e)` costs 0.80–1.46, which is *worse* than deleting layer-1 attention
outright, and substituting `rms(e + A0)` costs 0.86–1.68.

So: **the attention→attention path — the one the textbook induction circuit
runs on — is numerically closed in these models.** Layer 1's selection is a
function of the layer-0 MLP's write and essentially nothing else. The channel
does widen monotonically with width (0.075% → 0.416%), which is the only
structural quantity we have found that moves in the direction of composition.

---

## 2026-08-08 — FINDING 9 (DEPTH 2): induction APPEARS, at width 256, and it does not use the residual-stream composition path

**Registered prediction `d2_induction` REFUTED at width 256, held at 128.** We
registered, before measuring the unmeasured cells, that the induction score
would stay within ±0.05 nats and under 3 standard errors at depths 2, widths
128 and 256.

| cell | induction score | bag score | detectable-effect floor (3 SE) |
|---|---|---|---|
| depth 1, w32 / w64 / w128 / w256 (3 seeds each) | −0.006 / −0.012 / −0.026 / −0.035 | +0.015 / +0.031 / +0.060 / +0.081 | — |
| depth 2, w32 (3 seeds) | −0.008 ± 0.002 | +0.020 | 0.008 |
| depth 2, w64 (3 seeds) | −0.014 ± 0.002 | +0.045 | 0.011 |
| depth 2, w128 (3 seeds) | −0.003 ± 0.010 | +0.086 | 0.010 |
| **depth 2, w256 (3 seeds)** | **+0.0938 ± 0.0086** | +0.133 | 0.006–0.017 |

(depth-2 width-256 per seed: +0.0841, +0.0965, +0.1007, each 5–17× its own
floor; the depth-1 width-256 matched cells are −0.0354 ± 0.0015 over 3 seeds,
so the flip is between depths at fixed width, not a width effect on its own.)

The width-256 score is **five times the battery's own detectable-effect floor**
and the first positive value anywhere in the program. It is corroborated by an
independent probe on **real held text**: destroying the induction evidence with
a **bag-preserving swap** (exchange the token that followed the earlier
occurrence with another prefix token — a permutation, so the prefix multiset is
identical and only the adjacency changes) costs the model 0.244 nats on the
induction target. Because a *depth-1* model — which structurally cannot compose
— also scores positive on that probe (its distance kernel notices the swap),
the depth-1 cell at the same width is used as the **matched null**:

| width | depth-1 null | depth 2 | excess | t |
|---|---|---|---|---|
| 32 | +0.023 | +0.026 | +0.003 | 0.2 |
| 64 | +0.041 | +0.055 | +0.015 | 0.9 |
| 128 | +0.067 | +0.103 | +0.036 | 1.7 |
| **256** | +0.085 | **+0.241** | **+0.155** | **5.5** |

(width 256 is 3 depth-1 seeds against 3 depth-2 seeds; the other widths are
3 against 3 as well.)

### The circuit, and why it is not the textbook one

Located by ablation (`tf_induction_circuit.py`,
`tf_vanilla_d2_w256_b8192_s0_induction_circuit.json`):

| intervention | induction score | KL cost |
|---|---|---|
| none | 0.0841 ± 0.0065 | 0 |
| drop **layer-0 head 1** | **0.0083 ± 0.0051** | 0.186 |
| drop layer-1 head 15 | 0.0353 ± 0.0064 | 0.016 |
| drop both | −0.0025 ± 0.0035 | 0.189 |
| delete layer-0 head 1 **from layer 1's Q/K/V read** | **0.0841** | — |
| delete layer-0 head 1 **from MLP-1's input** | 0.0841 | — |
| delete layer-0 head 1 **from MLP-0's input** | **0.0083** | — |
| control: delete a *different* layer-0 head from layer 1's read | 0.0841 | — |

Layer-0 head 1 is one of the two heads with the most distance-1 attention mass
(11.0% and 11.9%, against 0.8–8% for the other fourteen), and layer-1 head 15
has the most in its layer (10.8%). So the *participants* are the ones the
standard story names. **The wiring is not.** Deleting head 1's write from what
layer 1's queries and keys read changes the induction score by 0.0000; deleting
it from what the layer-0 **MLP** squares reproduces the entire effect. The
previous-token signal reaches layer-1 attention **through the MLP**, which is
exactly what FINDING 8's composition budget predicts, since layer 1's read is
99.9% MLP-0's write and 0.4% layer-0 attention.

### Replicated on three seeds, including the route decomposition

`tf_w256_seeds_chain.sh` trained depth-2 width-256 seeds 1 and 2 (and depth-1
width-256 seeds 1 and 2 for the matched null). Everything holds:

| seed | induction | natural-text swap | the head that carries it | its distance-1 share (rank in layer 0) |
|---|---|---|---|---|
| 0 | +0.0841 | +0.244 | layer-0 head 1 | 0.110 (2nd of 16) |
| 1 | +0.0965 | +0.236 | layer-0 head 6 | 0.114 (1st of 16) |
| 2 | +0.1007 | +0.242 | layer-0 head 5 | 0.119 (2nd of 16) |

and the route decomposition is the same in all three — deleting the head's
write from layer 1's read leaves the score at 0.0841 / 0.0965 / 0.1007
(unchanged to 4 decimals), deleting it from MLP-0's input gives 0.0083 /
0.0131 / −0.0318 (the whole effect, and at seed 2 an overshoot past zero).
The head index is arbitrary across seeds; what replicates is that it is one of
the two heads with the most distance-1 attention mass, and that its route is
the MLP.

**Selection-effect control:** the heads were chosen on probe seeds 0–4, so the
entire decomposition was re-scored on **disjoint probe seeds 100–104** and
reproduces to within 0.001 at every cell.

---

## 2026-08-08 — FINDING 10 (ADVERSARIAL REVIEW): the rung-4 composed table does not predict what its head causally does — FINDING 6 is corrected

The standing rule is "compose to the logits **and confirm causally**". FINDING 6
did the first half. Doing the second half breaks it.

For every head, the agreement between the rung-4 object
`p_h · (OV_h W_Uᵀ)` — the head's **direct** route to the logit — and the head's
actual causal effect `logits(full) − logits(drop h)` on held text:

| cell | direct-route Pearson (per head) | through-MLP Pearson |
|---|---|---|
| depth 1, w32 | 0.17–0.39 | 0.63–0.83 |
| depth 1, w64 | 0.03–0.42 | 0.69–0.91 |
| depth 1, w128 | 0.00–0.43 | 0.77–0.95 |
| depth 1, w256 | −0.01–0.19 | 0.87–0.98 |
| depth 2, layer 0 | **0.002–0.02** | 0.93–0.96 |
| depth 2, layer 1 | 0.51–0.70 | 0.94–0.98 |

This is FINDING 2 biting back: the direct route is dead, so an object built out
of the direct route describes nothing. The correct composition — propagating
the head's write through the MLPs, which is *exact* here because the MLP is
bilinear — tracks the causal effect at 0.63–0.98 with 92–95% sign agreement.

**What that costs FINDING 6.** Its headline was "the heads are not copy heads:
the median rank of the attended token among the tokens it boosts is ≈5600 of
8192, i.e. attending to a token pushes its own logit *down*". Re-derived
causally — build the two-token context `[u, t]`, drop the head, and rank the
attended token `u` by how much the head's presence pushes it:

| cell | causal median rank of the attended token (of 8192), per head |
|---|---|
| depth 1, w32 | 1003, 1902 |
| depth 1, w64 | 286, 2834, 2867, 3310 |
| depth 1, w128 | 296, 694, 2190, 2316, 2563, 3144, 3689, 3752 |
| depth 1, w256 | 425, 508, 795, 2234 … 4880 |
| depth 2, w64 | layer 0: 3526, 3582, 3670, 5231; layer 1: **572**, 1084, 2255, 5222 |

**Retraction:** "≈5600 of 8192, pushed down" is a statement about the direct
composed table, not about the heads. Causally the median is 286–4880, several
heads put the attended token in the top 4–6% of the vocabulary, and no head is
anywhere near the "pushes its own token down" description. The *weaker* claim
survives: no head is a copy head in the strict sense (rank 0), the effect is
diffuse, and identity pairs are not specially favoured.

Everything else in the reviewer round is in `tf_reviewer_round_1_depth2.json`.

---

## 2026-08-08 — FINDING 1: the fold gate failures were PRECISION, and fixing the dtype made three independent controls sharper

**Verdict: precision, not a bug — and the corrected gate is strictly stronger
than the one it replaces.**

The old criterion mixed units: three relative algebraic checks at 1e-6 and one
**absolute** end-to-end logit check at 1e-5. Logits here live on
`30·tanh(·/30)` and reach 15–20, where one fp32 ulp is already ~1e-6, so the
absolute budget was about eight ulps for a forward pass that accumulates
thousands of roundings. **Four of the six trained cells failed on that clause
alone**, while every algebraic identity passed at 2–5e-7. (Naively making the
logit clause relative at 1e-6 does not help: it fails five of six, because the
fp32 algebraic tolerance of 1e-6 is itself at the rounding floor — the width-128
truncated-tokenizer cell sits at 1.05e-6 in fp32 and 7.7e-16 in fp64.)

Three pieces of evidence, not one:

1. **fp64 collapse.** Making `fold_forward`, `fold_mlp`, `fold_layer0_qk` and
   `rot_matrix` dtype-clean (they hard-cast to float32 and crashed on a
   `.double()`d model) lets the same comparison run in fp64. The end-to-end
   residual drops from 1.5e-5–2.7e-5 to **1.3e-14–4.4e-14 absolute**, i.e. about
   ten fp64 ulps at logit magnitude 15. The algebraic identities go to
   5e-16–1.5e-15 relative.
2. **The forward disagrees with itself by more.** The gate now measures
   `max|forward_fp32 − forward_fp64| / max|logit|`, the reference's own fp32
   noise: 6e-7 to 2.9e-6. At width 128 that is **larger** than the
   fold-vs-forward gap (1.7e-6). The fold agrees with the forward better than
   the forward agrees with itself.
3. **A negative control proves the new gate is not a loosening.**
   `tf_model.gate_negative_control` corrupts the MLP tensor by a factor
   `1+1e-7` and rolls the value factors by one head. The 1e-7 corruption
   produces an fp32 absolute logit difference of **1.19e-7** — the superseded
   absolute-1e-5 gate would have **passed** it; the new fp64 tier fails it
   (9.9e-9 > 1e-9). Both corruptions are caught; the clean model passes.

A dtype bug *was* found and fixed, just not in the fold algebra: `rot_matrix`
built its inverse-frequency vector in fp64 and rounded to fp32 while
`rope_tables_exact` built it in fp32, putting a one-ulp wedge between the
folded and the forward rotation. With both at the same dtype:

| control | before | after |
|---|---|---|
| planted known-answer table, δ=3 | 5.79e-9 | **1.59e-14** |
| fp64 attention-table identity | (could not run) | **7e-16** |

### The corrected gate (two tiers)

* **fp32 sanity band** — every identity, *relative*, < 1e-5 (sized by
  `sqrt(N)·eps_fp32`; the two paths do the same ~1e3–1e4 multiply-accumulates in
  a different order).
* **fp64 exactness** — algebraic identities < 1e-12 relative, end-to-end
  < 1e-9 absolute, **and** the fold-vs-forward gap ≤ 10× the forward's own
  fp32-vs-fp64 self-noise. The last clause is what would catch a small genuine
  bug hiding under a fixed threshold.

### Identity table (all local checkpoints; `tf_identity_table.json`)

fp32 columns are relative except the logit-abs column; fp64 columns are the
real exactness gate.

| stem | pass | fp32 mlp T | fp32 gauge | fp32 attn | fp32 logit abs | fp32 logit rel | fp64 mlp T | fp64 attn | fp64 logit abs | fwd self-noise | gap/noise | planted δ=3 | neg ctl |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| d1_w128·bpe_s0 | ✓ | 4.5e-07 | 5.2e-07 | 9.8e-07 | 2.7e-05 | 1.73e-06 | 1.4e-15 | 6.8e-16 | 4.4e-14 | 2.9e-06 | 0.59 | 1.6e-14 | ✓ |
| d1_w128·bpe_s1 | ✓ | 5.5e-07 | 4.5e-07 | 6.4e-07 | 3.8e-05 | 2.47e-06 | 1.7e-15 | 7.1e-16 | 3.0e-14 | 3.4e-06 | 0.72 | 1.6e-14 | ✓ |
| d1_w128·bpe_s2 | ✓ | 4.2e-07 | 4.8e-07 | 1.6e-06 | 3.8e-05 | 2.47e-06 | 1.3e-15 | 1.2e-15 | 4.4e-14 | 3.7e-06 | 0.66 | 1.6e-14 | ✓ |
| d1_w128·trunc_s0 | ✓ | 2.9e-07 | 4.6e-07 | 1.1e-06 | 2.4e-05 | 1.73e-06 | 1.5e-15 | 6.6e-16 | 2.9e-14 | 1.7e-06 | 0.99 | 1.6e-14 | ✓ |
| d1_w256·bpe_s0 | ✓ | 5.3e-07 | 5.2e-07 | 1.5e-06 | 4.7e-05 | 2.90e-06 | 1.8e-15 | 8.5e-16 | 5.9e-14 | 4.3e-06 | 0.68 | 1.6e-14 | ✓ |
| d1_w32·bpe_s0 | ✓ | 2.8e-07 | 3.4e-07 | 4.6e-07 | 2.1e-05 | 1.40e-06 | 8.5e-16 | 7.2e-16 | 3.7e-14 | 6.2e-07 | 2.28 | 1.6e-14 | ✓ |
| d1_w32·bpe_s1 | ✓ | 1.9e-07 | 3.7e-07 | 3.8e-07 | 1.7e-05 | 1.22e-06 | 6.9e-16 | 6.6e-16 | 3.6e-14 | 1.5e-06 | 0.79 | 1.6e-14 | ✓ |
| d1_w32·bpe_s2 | ✓ | 2.6e-07 | 3.3e-07 | 2.9e-07 | 1.3e-05 | 9.29e-07 | 8.6e-16 | 7.8e-16 | 4.7e-14 | 9.3e-07 | 1.00 | 1.6e-14 | ✓ |
| d1_w32·trunc_s0 | ✓ | 3.1e-07 | 3.9e-07 | 9.2e-07 | 7.2e-06 | 5.63e-07 | 4.7e-16 | 1.5e-15 | 2.1e-14 | 1.2e-06 | 0.49 | 1.6e-14 | ✓ |
| d1_w64·bpe_s0 | ✓ | 2.5e-07 | 3.5e-07 | 4.5e-07 | 1.5e-05 | 9.21e-07 | 1.0e-15 | 6.5e-16 | 2.6e-14 | 1.9e-06 | 0.49 | 1.6e-14 | ✓ |
| d1_w64·bpe_s1 | ✓ | 4.2e-07 | 4.0e-07 | 2.0e-07 | 1.1e-05 | 8.05e-07 | 1.1e-15 | 4.6e-16 | 2.4e-14 | 1.6e-06 | 0.52 | 1.6e-14 | ✓ |
| d1_w64·bpe_s2 | ✓ | 3.4e-07 | 3.6e-07 | 4.5e-07 | 1.2e-05 | 8.13e-07 | 9.5e-16 | 9.4e-16 | 2.9e-14 | 2.2e-06 | 0.37 | 1.6e-14 | ✓ |
| d1_w64·trunc_s0 | ✓ | 2.3e-07 | 2.5e-07 | 3.3e-07 | 6.7e-06 | 4.86e-07 | 8.1e-16 | 8.5e-16 | 1.3e-14 | 5.8e-07 | 0.83 | 1.6e-14 | ✓ |
| d2_w32·bpe_s0 | ✓ | 2.2e-07 | 3.1e-07 | 9.4e-07 | 1.5e-05 | 1.11e-06 | 7.3e-16 | 2.1e-15 | 4.0e-14 | 1.9e-06 | 0.59 | 1.6e-14 | ✓ |
| d2_w64·bpe_s0 | ✓ | 2.5e-07 | 3.6e-07 | 4.9e-07 | 1.0e-05 | 7.18e-07 | 1.2e-15 | 1.1e-15 | 2.3e-14 | 3.6e-06 | 0.20 | 1.6e-14 | ✓ |
| d2_w64·bpe_s1 | ✓ | 3.7e-07 | 3.8e-07 | 3.2e-07 | 1.2e-05 | 9.43e-07 | 1.1e-15 | 6.0e-16 | 2.1e-14 | 2.8e-06 | 0.34 | 1.6e-14 | ✓ |

All **16** local checkpoints pass, depth-2 cells included.

Also green in `tf_identity_table.json` but omitted above for width: the
all-heads factor-indexed attention identity (a V×V-free recomputation of the
same quantity, added because materializing 16 heads × 4 distances in fp64 at
V=8192 is 68 GB and OOM-killed the width-256 fold), the QR-vs-eigenvalue
spectrum control (2e-13 to 2e-10 relative) and the factor-vs-dense-SVD control
(2e-16 to 1e-14).

**The six width-256 cells from the scale box could NOT be re-folded: only their
JSONs were pushed, not their `.pt` files** (`*.pt` is untracked here). A
width-256 depth-1 cell was retrained locally on the primary BPE corpus instead
(held CE 4.5583) and is folded and interpreted; it is the `w256` column
throughout.

---

## 2026-08-08 — FINDING 2: at depth 1 the model is a QUADRATIC FORM with an attention-driven input; the residual stream is invisible at the readout

Depth-1 vanilla with `n_slots = 1` folds exactly (verified to 1e-6 relative in
fp32, `decomposition_control` in every `*_interp.json`) into

```
e_i      = Ehn[t_i]                                    (current token only)
p_h[i,j] = s1_h(t_i,t_j,i−j) · s2_h(t_i,t_j,i−j)        (token-pair × distance)
A_i      = Σ_h Σ_{j≤i} p_h[i,j] · OV_h[t_j]
M_i      = T(rms(e_i+A_i), rms(e_i+A_i)) + b
logits_i = 30·tanh( rms(e_i+A_i+M_i) · W_Uᵀ / 30 )
```

Because RMSNorm is a scalar gauge, the pre-tanh logit is **exactly additive** in
the three folded terms, so their shares can be read off with no approximation.
Measured on held text (mean over 3 seeds):

| width | ‖e‖ | ‖A₀‖ | ‖A_past‖ | ‖M‖ | logit share of M |
|---|---|---|---|---|---|
| 32 | 5.66 | 0.85 | 4.0 | 3268 | **0.99988** |
| 64 | 8.00 | 1.7 | 7.3 | 6069 | **1.00017** |
| 128 | 11.31 | 3.1 | 11.2 | 10349 | **1.00006** |
| 256 | 16.00 | 4.9 | 15.9 | 18431 | **1.00018** |

(the four shares sum to 1 to 5e-8 by construction; the small excess over 1 is
the embedding term's *negative* share, −5e-4)

Causal confirmation, not just geometry: discarding the embedding **and** both
attention writes from the residual and keeping only the MLP write reproduces
the model at **KL 1e-5 to 3e-5**. The skip connection into the readout is
functionally dead.

### The attention's whole effect is on the MLP's INPUT

This is the correction of an earlier claim (see the retraction below). The two
routes, as mutually exclusive ablations that bracket the model (KL from the
real model, mean ± sd over 3 seeds):

| stage | w32 | w64 | w128 | w256 (1 seed) |
|---|---|---|---|---|
| no attention at all | 0.285 ± 0.025 | 0.466 ± 0.015 | 0.687 ± 0.010 | 0.911 |
| past attention **direct route only** (MLP frozen at its no-context input) | 0.258 ± 0.020 | 0.431 ± 0.013 | 0.644 ± 0.007 | 0.851 |
| past attention **MLP route only** (A_past removed from the residual) | **0.0000** | **0.0000** | **0.0000** | **0.0000** |
| full model | 0 | 0 | 0 | 0 |

The direct route lands on the no-attention number; the MLP route lands on zero.
100% of what attention buys is delivered by moving the quadratic form's
argument.

---

## 2026-08-08 — FINDING 3 (RUNG 5): the KL ladder, and what each component buys

All stages are weights-free table programs (look up rows of `Ehn`, `A0`, `M0`,
`OV`, index the branch factors by token id, apply the rotary, read out with
`W_U`); no stage calls the network's forward. Scored on **held** text; the only
fitted objects in the ladder (the token-independent distance profile, the
mean-ablation value) are fitted on the **estimation** split. KL from the real
model, nats/token, mean ± sd over 3 seeds.

| stage | w32 | w64 | w128 | w256 (1 seed) |
|---|---|---|---|---|
| embedding only | 8.880 ± 0.201 | 12.294 ± 0.410 | 15.900 ± 0.208 | 18.873 |
| + attention to self (δ=0) | 8.676 ± 0.250 | 11.790 ± 0.470 | 15.171 ± 0.302 | 18.02 |
| **+ MLP ⇒ the model's own bigram** (weights-only V×V table) | **0.258 ± 0.020** | **0.431 ± 0.013** | **0.644 ± 0.007** | **0.852** |
| + past attention, distance ≤ 1 | 0.167 ± 0.013 | 0.266 ± 0.012 | 0.378 ± 0.004 | 0.485 |
| + past attention, distance ≤ 4 | 0.092 ± 0.005 | 0.148 ± 0.006 | 0.211 ± 0.005 | 0.276 |
| + past attention, distance ≤ 16 | 0.029 ± 0.001 | 0.051 ± 0.004 | 0.079 ± 0.001 | 0.106 |
| + past attention, distance ≤ 64 | 0.004 ± 0.001 | 0.007 ± 0.001 | 0.011 ± 0.000 | 0.015 |
| + past attention, all distances (= exact) | 0 | 0 | 0 | 0 |

Ablation variants at the same stage:

| variant | w32 | w64 | w128 | w256 |
|---|---|---|---|---|
| pattern replaced by its token-independent distance profile | 0.240 | 0.260 | 0.271 | 0.292 |
| — i.e. fraction of the attention effect that is PURELY POSITIONAL | 16% | 44% | 61% | 68% |
| pattern with the ROTARY REMOVED (δ=0 table at every distance) | 0.960 | 1.294 | 1.695 | 2.113 |
| top 4 of 8 rotary frequency pairs kept | 0.101 | 0.195 | 0.247 | 0.361 |
| top 2 of 8 rotary frequency pairs kept | 1.252 | 1.560 | 3.151 | — |
| MLP restricted to its 64 most-used hidden units (of 128/256/512/1024) | 0.671 | 1.396 | 2.138 | — |

Readings:

* **Two terms carry the model.** The weights-only bigram table takes KL from 8.9
  to 0.26 at width 32; the folded past attention takes the rest to 0. There is
  no third ingredient.
* **The attention is mostly a learned DISTANCE KERNEL, not a content lookup.**
  Replacing the whole token-pair pattern with its distance-only average keeps
  16% of the attention's value at width 32 but **61% at width 128**; removing
  the distance information and keeping only the token-pair table is
  catastrophic (1.7 nats, worse than having no attention at all). The
  query/key token dependence is the *minority* contribution at the widest cell.
* **Registered prediction REFUTED (`rung3_skipgram`).** We registered that
  distance ≥ 2 would be worth less than distance 1. At width 128 the δ=1 term
  buys 0.649−0.378 = 0.271 and everything beyond it buys 0.378 — the longer-range
  skip-grams are worth **more**, and the same ordering holds at every width.
* **Registered prediction PARTLY REFUTED (`rung3_positional`).** We registered
  that the distance-only pattern would destroy most of the attention gain. It
  destroys most of it at width 32 and a minority of it at width 128.
* **The MLP is not compressible in its own basis.** Half the hidden units (a
  genuine CP-term truncation, since the bilinear MLP *is* a rank-`hidden`
  symmetric CP decomposition) leaves KL 0.67–2.14, i.e. worse than deleting the
  attention entirely.

### Against data baselines (held CE, nats/token; baselines fitted on train/est)

| predictor | CE | parameters |
|---|---|---|
| unigram | 7.260 | 8 192 |
| positional-only (p(next\|position), fitted on est) | 7.718 | 512·8 192 |
| low-rank bigram, rank 32 | 6.649 | 524 288 |
| low-rank bigram, rank 64 | 6.469 | 1 048 576 |
| sparse bigram, top 262 144 counts + unigram backoff | 5.675 | 524 288 |
| sparse bigram, top 1 048 576 counts + unigram backoff | 5.322 | 2 097 152 |
| dense closed-form bigram (α = 1000) | 5.200 | 67 108 864 |
| **model, width 32** | 5.413 | 280 608 |
| **model, width 64** | 5.048 | 598 080 |
| **model, width 128** | 4.723 | 1 343 616 |
| **model, width 256** | 4.461 | 3 400 704 |
| model's own bigram stage, w32 / w64 / w128 | 5.720 / 5.566 / 5.490 | 524k / 1.05M / 2.10M tables |

Honest readings, including the ones that do not flatter the model:

* Widths 64 and 128 beat the dense bigram table with 50–100× fewer parameters.
  Width 32 does **not** (5.413 vs 5.200).
* At **matched parameter count** the weights-only *model-bigram stage* **loses**
  to a data-fitted sparse bigram (5.490 vs 5.322 at 2.1M). The model only wins
  once its attention term is included. So "the model is a better bigram than a
  bigram" is false; "the model is a better *context* model than a bigram" is
  true from width 64.
* The comparison is not made fair by parameter count alone, because the model
  sees the whole prefix and the position. The position profile settles it: at
  **position 0**, where the model and the bigram see exactly the same one token
  of context, the bigram wins at every width (5.489 vs 5.855–6.056). The model
  overtakes it from about position 8 at widths 64–128 and never at width 32.

---

## 2026-08-08 — FINDING 4 (RUNG 2): selection is low rank, content is not — with nulls

`rank ≤ head_dim` and `rank ≤ hidden` are **arithmetic**, not findings. What is
reported is the distance below the bound, measured by spectral-entropy
effective rank `exp(H(σ/Σσ))`, against an iid-Gaussian null of the same shape.

| object | bound | trained (mean ± sd over 3 seeds) | null |
|---|---|---|---|
| branch score table s1, δ=0, w32 | 16 | **2.28 ± 0.61** | 15.991 ± 0.001 |
| branch score table s1, δ=0, w64 | 16 | **2.91 ± 0.32** | 15.991 ± 0.001 |
| branch score table s1, δ=0, w128 | 16 | **3.40 ± 0.32** | 15.991 ± 0.001 |
| branch score table s1, δ=0, w256 (1 seed) | 16 | **5.93** | 15.991 ± 0.001 |
| query factor Q1, w128 | 16 | 5.08 | 15.996 ± 0.001 |
| **value factor Vv, w128** | 16 | **15.56 ± 0.02** | 15.996 ± 0.001 |
| MLP tensor, mode-0 unfolding, w128 | 128 | **121.9 ± 0.1** | 123.2 (random *factored* tensor, same shapes) |
| MLP tensor, mode-0 unfolding, w32 | 32 | 30.0 ± 0.1 | ~31 |
| MLP tensor, mode-0 unfolding, w256 | 256 | 239.7 | 246.8 |
| value factor Vv, w256 | 16 | 15.66 | 15.996 ± 0.001 |

This is the parent program's headline reproduced at the smallest possible
scale: **selection (query/key) is strongly low rank — three effective
directions out of sixteen, against a null of sixteen — while content (the value
factor and the MLP tensor) is spectrally indistinguishable from a random object
of the same shape.** Registered prediction `rung2_low_rank` is **half right**:
the score-table part was predicted and confirmed; the MLP part predicted "well
below its bound" and is refuted.

The low-rank selection claim also has a causal version: keeping only the top 4
of 8 rotary frequency pairs (the only δ-equivariant way to cut a head's
subspace) keeps most of the attention's value.

---

## 2026-08-08 — FINDING 5 (RUNG 3): no induction at depth 1 **or depth 2**, and the metric is calibrated

Three matched synthetic conditions with identical token multisets, scored on
the second copy only: `repeat = [R][R]`, `shuffled = [shuffle(R)][R]`,
`control = [R'][R]`; induction score = CE(shuffled) − CE(repeat) (needs
**order**, hence composition, hence ≥ 2 layers); bag score = CE(control) −
CE(shuffled) (needs only a bag).

| cell | induction score | bag score |
|---|---|---|
| depth 1, w32 (3 seeds) | −0.006 ± 0.002 | +0.015 ± 0.002 |
| depth 1, w64 (3 seeds) | −0.012 ± 0.002 | +0.031 ± 0.006 |
| depth 1, w128 (3 seeds) | −0.026 ± 0.002 | +0.060 ± 0.002 |
| depth 1, w256 (1 seed) | −0.034 ± 0.009 | +0.081 ± 0.009 |
| depth 2, w32 (1 seed) | −0.007 ± 0.008 | +0.021 ± 0.008 |
| **depth 2, w64 (2 seeds)** | **−0.014 ± 0.003** | +0.044 ± 0.014 |

**The registered depth-2 positive control FAILED**: we registered that a depth-2
cell "must show a nonzero induction score, otherwise the metric is broken". It
did not. So the null was re-established a different way — by planting a known
amount of induction and finding the detection floor. Mixing the model with a
perfect induction oracle at weight ε:

| ε | induction score (depth-2 w64) |
|---|---|
| 0 | −0.0154 ± 0.0054 |
| 1e-4 | **+0.940 ± 0.023** |
| 3e-4 | +1.412 ± 0.037 |
| 1e-3 | +2.030 ± 0.053 |
| 1e-2 | +3.399 ± 0.081 |

A mixture weight of **0.01%** already moves the score by 175 standard
deviations. The battery is not blind; these models simply have no induction.
The honest statement is therefore: *at depths 1–2 and widths ≤ 128 on this
corpus and this 15 000-step single-epoch budget, induction is absent to within
~0.02 nats* — which is a statement about this regime, not a proof that depth 2
cannot induct.

The second number is deliberately called a **bag** score, not a copy score:
rung 4 shows the attended token ranks near the *bottom* of what attending to it
boosts, so naming the bag effect "copying" would be inferring a mechanism from
a behavioural delta.

---

## 2026-08-08 — FINDING 6 (RUNG 4): the heads are not copy heads, and the composed pair table barely factorises

Everything here is composed to logits before it is named (the standing sign
rule): the object measured is
`C_h(t,u,δ)[v] = p_h(t,u,δ)·(OV_h[u]·W_Uᵀ)[v]`, never a raw factor.

* **Not copy heads.** For each head's eight strongest keys, the median rank of
  the attended token among the tokens it boosts is **≈ 5 600 of 8 192**.
  Attending to a token pushes its own logit *down* relative to a random pair
  (identity-pair z of −3.4 to +1.8 across heads). This is reported as "attending
  to a token does not push its own logit up", **not** as suppression.
* **The pair table is close to an outer product.** The σ₁ share of the composed
  (query, key) matrix is 0.37–0.85 per head (median ≈ 0.74), with entropy rank
  2.5–14 out of a bound of 256. Most heads therefore have almost no genuine
  *pair* specificity: what they do is approximately (a score for the query) ×
  (a fixed write for the key).
* **What attending does is generic.** Width 128, head 0: the four strongest keys
  are all closing-quote tokens (`,”`, `.”`, `”`, `”.`) and every one of them
  boosts the same continuation set — `.` (+59), ` and` (+47), `,` (+47), ` in`
  (+40), ` to` (+32). That is a punctuation-context head that writes a generic
  "sentence continues" direction, not a content lookup.
* **Token-class claims, with a frequency-matched null** (400 draws, same size,
  drawn with train unigram probability). Only classes at |z| > 3 are named. At
  widths 32 and 64 head 0's strongest value directions are enriched for
  whitespace-initial lowercase word pieces (z = +3.8, +5.6) and depleted of
  capitalised pieces (z = −3.0, −4.9). At width 128 **nothing** clears |z| = 3
  and no class is named.
* **Registered prediction REFUTED (`rung4_tokens`).** We registered that the
  composed copy score would be dominated by a few token pairs and enriched for
  identical tokens. It is diffuse (effective pair fraction 0.08–0.44 of all
  sampled pairs) and identity pairs are *de*-enriched.

---

## RETRACTION (2026-08-08, same day)

`MAILBOX.md` 2026-08-08 05:00 and commit `631ddaa20` reported that at depth 1
"attention to the past buys 0.0005 nats — nothing" and that "every distance
restriction lands on top of the full thing". **That is wrong.** The ladder that
produced it added `A_past` to the residual while holding the MLP frozen at its
no-context input, so it measured the *direct* route only. Attention is worth
0.29 / 0.47 / 0.69 nats of KL at widths 32 / 64 / 128, and every bit of it goes
through the MLP. The distance-restriction table in that entry is superseded by
the one in FINDING 3.

The failure mode is exactly the one the standing sign/gauge rule describes, in
a non-sign form: **a term was scored without composing it through the
downstream nonlinearity.** It is now in the README failure-mode list.

The self-red-team of every claim above, with what was fixed and what could not
be, is `tf_reviewer_round_1.json`.
