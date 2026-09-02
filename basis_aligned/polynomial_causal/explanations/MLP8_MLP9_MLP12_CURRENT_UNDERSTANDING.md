# MLP8, MLP9, and MLP12: current understanding

*Consolidated 2026-09-02 before any new experiment inside these modules. This dossier separates older whole-layer
approximation work from the newly identified equality-correction circuit. It is the duplicate-work check required by
the MLP dossier index.*

## Exact computation and shapes

For any one of these MLPs, the input at one token position is a 1,152-number residual-stream vector `x`. After the
model's normal input normalization, the MLP computes

`z_j = (Left_j x)(Right_j x)` for `j = 1,...,4608`,

then

`y = sum_j Down[:,j] z_j + bias`.

Thus each product term reads two linear combinations of the input, multiplies them, and writes a fixed 1,152-number
direction. `Left` and `Right` are each `4608 x 1152`; `Down` is `1152 x 4608`; the bias has 1,152 numbers. The full
module therefore stores `2(4608)(1152) + (1152)(4608) + 1152 = 15,926,400` numbers. “Product dimension” means the
4,608 scalar products `z_j`; it is neither the 1,152-dimensional input/output nor a measured rank.

A product term is unchanged if its Left row is multiplied by `a`, its Right row by `b`, and its Down column by
`1/(ab)`. Swapping its Left and Right rows also changes nothing. Consequently, raw row norms are not meaningful,
but the complete term `Down[:,j](Left_j x)(Right_j x)` is invariant to these scale and swap choices. Arbitrary
rotations of the 4,608 terms are not generally allowed because elementwise multiplication fixes the paired products;
permutations and genuinely degenerate factorizations remain possible. A claimed group must therefore be defined by
what it does downstream and shown stable across data/source splits, not only by weight similarity.

## What had already been tested

| earlier test | exact object tested | MLP8 | MLP9 | MLP12 | what it established |
|---|---|---:|---:|---:|---|
| mean replacement (§optimal-ablation records) | replace the whole 1,152-vector output by its fitted mean | 0.0474 | 0.0496 | 0.0416 nat damage | These modules matter modestly in that old evaluation. This says nothing about their internal tasks. |
| local linear/quadratic ladder (§§1457, 1481, 1483) | predict the whole output from nearby residual-stream states, then add sampled quadratic features | 45.7% / 47.0% | 48.2% / 52.0% | 43.4% / 43.6% | Generic local predictors explain about half; adding random quadratic features helps little, especially at MLP8/12. |
| native product-term truncation (§§1533–1534) | retain product terms with largest `std(z_j) * norm(Down[:,j])` | K64 40.1%; K256 47.9%; K1024 64.5% | 42.2%; 48.9%; 64.7% | 37.4%; 42.8%; 58.2% | A small native-term core exists, followed by a long tail. These terms were selected by generic output size, not by a task or downstream use. |
| core plus linear residual (§1535) | K256 native terms plus a linear fit for the remaining whole-layer output | 55.5% | 56.4% | 50.9% | This improved both ingredients but was expensive and did not identify circuits. |
| Left/Right intervention (§§1077–1078) | replace one multiplicative branch by the other | self-square cost about 0.14 nat; only 0.04% of terms had activation correlation above 0.7 | not run in that receipt | self-square cost about 0.09 nat | Deep-middle products are not mostly literal squares. The absolute effects are small relative to early MLPs. |
| MLP9 unit clustering (§607) | cluster 300 high-impact native terms by their activation/damage examples | — | split-half ARI 0.167 versus null -0.012 | — | There is weak reproducible structure, but no clean semantic groups. It did not use downstream causal effects or the equality task. |
| task-conditioned product terms (§§1563–1568, 1572) | rank native terms by class-conditional activation times their Down-column projection, then remove exact selected terms | other modules were selected | other modules were selected | other modules were selected | This method found a held-out-replicated 16-term question-mark component in MLP11. Weight-only signs failed; task-conditioned activations were essential. It is the closest precedent for the proposed equality split. |
| cross-class term reuse (§1571) | overlap class-conditioned top-64 sets, then remove the largest shared set | other modules | other modules | other modules | Large overlaps were generic late-layer pools, not causally reused semantic features. Set overlap alone is not evidence for a shared computation. |
| class-projected bilinear form (§§1570, 1573–1574) | contract the complete MLP tensor with one logit direction, then remove eigen-slices of that task-specific quadratic form | other modules | other modules | other modules | A two-direction question-mark component in MLP11 beat its selected term set, but the method was not universal across classes. This is a legitimate alternative if equality-term groups fail, not a reason to call low rank semantic by itself. |
| token/previous-token plus random squared features (`qk_mlp*_program`) | replace the whole output by current-token and previous-token tables plus `((H A^T)^2)U`, with 256 fixed random projections | 24.2% understood after scalar gain fitting | 21.1% | 16.4% | This old program retained roughly natural/shuffled induction ratios, but poorly replaced each full MLP. It did not isolate equality correction. |
| short-context recomputation (§§1183–1185) | recompute each MLP using at most the previous 64 tokens in the old harness | about 0.0032 | about 0.0027 | about 0.0025 nat | Their inputs are largely local attention-pooled states in that setting. It does not imply that their output has no task-specific contextual role. |
| old Down-map factorization (§713) | factor only `Down` on observed product activations | r80 = 512 | r80 = 512 | r80 = 512 | This is a property of the output map on old activations, not the whole MLP or its product width. |

Percentages in the table are recovery of each experiment's own mean-replacement loss, not percentages of the full
model's CE loss, and the old runs used their own historical data/checkpoint conventions. They are useful duplicate
checks, not directly comparable adoption results for the present code corpus.

## New circuit evidence from rungs 465–466

The current question is different from all the tests above. The already-identified equality matcher can be supplied
either by its native source or by a transplanted source. Removing one later module write at a time showed that MLP8,
MLP9, and MLP12 each have the same four-context causal pattern under both sources: they reduce the matcher's effect
for a nearby or multiply occurring predecessor and increase it for a distant or unique predecessor.

Rung 466 removed every subset of the fixed group `T={MLP8, MLP9, MLP12}` and the broad-suppression group
`G={attention14, MLP17}` while later layers recomputed normally. The MLP group had cosine 0.992 with the full
four-context correction under both matcher sources, and its profile agreed across sources at cosine 0.994. It carried
about 47% of the correction's magnitude. The suppressor group was different (all four effects negative), and the
interaction between the two groups was large and reproducible. This identifies a task-shaped cross-module group; it
does not yet identify which computations inside any of the three MLPs implement it.

These results are on already-open code examples. The three modules and their grouping were selected using the same
corpus later used for the five-site factorial. The factorial tested new combinations, but it is not independent-corpus
confirmation. Fresh data and role separation are required before calling an internal group identified or adopting a
replacement.

## Duplicate work that is now closed

The next experiment must not be another sweep over output rank, random quadratic width, top-K terms by activation
size, generic unit clustering, class-conditioned set overlap, or whole-layer CE fitting. Those questions have already
been answered well enough to show that they do not locate the equality circuit. The old class-conditional term method
is reusable only with a stronger discovery signal and held-out causal test. Fewer terms, better reconstruction, and
lower CE are permitted as capacity controls only; they do not count as finding a circuit.

## The new, non-duplicate question

Split the 4,608 exact product terms by their **downstream equality-correction effect**, and allow one group to contain
terms from all three MLPs. For module `l`, source `s`, context cell `c`, and term `j`, first compute the exact change in
that term's write relative to the equality-absent trajectory:

`u[l,j,s] = Down_l[:,j] * (z[l,j,s] - z[l,j,absent])`.

A cheap discovery calculation may score how each `u` points along the downstream loss gradient for the four fixed
context cells. Unlike §§1563–1568, the proposed fingerprint uses four context-dependent causal effects and requires
agreement under two interchangeable equality sources; it is not one class-logit direction. The gradient is only a
screen. It proposes cross-module groups whose fingerprints agree under the native and transplanted matcher sources.
The decisive test must then remove or transplant the **summed exact terms** on held-out documents, let all later
layers recompute, and measure whether the group reproduces the registered `near-/far+/one+/multiple-` causal pattern
while matched-size random, high-activation, and old one-direction-style groups do not.

This directly targets three project goals: split a native MLP by task, group equivalent pieces across modules, and
extract/manipulate the equality correction. It can be killed cleanly: stop this product-term family if discovery-half
groups do not keep their signs and source agreement on held-out documents, or if exact held-out removal is no more
selective than matched-size activation-selected and random groups. A successful held-out result would still need a
fresh-corpus confirmation before adoption.

## Result of the product-term test and its transfer check

Rung 467 selected 450/426/482 exact terms in MLP8/9/12 from the first half of the code role. On the other half, their
joint exact removal matched the complete three-MLP correction direction at cosine 0.885/0.864 under the two matcher
sources, beat matched-count amplitude and random controls, and showed a source-stable cross-MLP interaction. This was
a valid held-out code split.

Rung 468 then froze every index and applied the same intervention to natural text. The direction remained similar
(cosine 0.977/0.926), but the selected terms carried only 17.1%/12.4% of the complete correction, did not beat the
matched controls, reduced to MLP8 as the only individually qualifying module, and lost the code interaction law.
Therefore the selected native term indices are register-specific and are not the general equality component.

This is the predeclared stopping result for term-index selection. Do not tune the 0.70 alignment threshold, select a
new K, or call the code list a model-wide circuit. The next object should remove the coordinate dependence: contract
the full bilinear MLP with its downstream causal response to obtain an input-space quadratic form, or define a
state-level equivalence by indistinguishable downstream interventions. Either route must again pass code/natural
transfer and exact causal removal.
