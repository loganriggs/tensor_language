# Full bilin18 update since `explanation_0326.md` — 14:05 UTC

**Date:** 2026-08-30  
**Coverage:** everything important completed after the 03:26 explanation, through the
frozen causal-response candidates and the latest attention-circuit screens.  
**Primary rule:** a plan, an implemented runner, or a good training fit is not called
an explanation. Numerical claims below come from preserved result artifacts.

## UPDATE STARTS HERE

## 1. The honest short answer

We learned several real pieces of structure, but the strict whole-model totals have
not moved since 03:26:

| Quantity | Value | What it means |
|---|---:|---|
| Certified removable stored values | 29,196,288 / 545,904,054 = **5.348245316%** | A complete executable replacement removed these stored values and passed its registered consequence tests. |
| Deletion CE assigned to named mechanisms | 0.57968 / 5.30682 = **10.923302467%** | This much of a fixed deletion-induced prediction-loss gap has a named causal account. |
| Still causally unnamed | **4.72714 nat = 89.076697533%** | The main quantitative reverse-engineering gap. |
| Circuits passing extraction, selective removal, low collateral, and OOD together | **0 / 68** | No terminal action yet passes the complete standard. |

These denominators differ. The 5.35% number is storage; the 10.92% number is a
cross-entropy effect. They cannot be added.

Since 03:26, the most important positive results are:

1. MLP0 was split into three **exact** tensor branches, and their causal interactions
   were measured.
2. A four-head equality-copy tensor reproduced induction behavior and transported to
   code OOD, although narrow “induction-only” removal was not collateral-free.
3. All 62 curated behavioral slices were causally localized to components, with
   important qualifications about near ties and shared components.
4. A signed causal-response tensor between 49 circuits was collected and a complete
   51-program training grid was fit.
5. A shared rank-32 response code reconstructed 65.17% of **pooled training response
   energy**, but only 5.57% at the median equally weighted owner interface. This is a
   useful low-dimensional regularity, not a claim that 65% of bilin18 is explained.
6. Twenty-seven nondominated response programs were frozen before held-out validation.
   The 114 validation documents have not been scored yet.

The immediate scientific question is now crisp: does any of those fixed programs
predict causal effects on new documents and across small owner interfaces, or did it
mostly compress one high-amplitude MLP16 response family?

## 2. A compact timeline since 03:26

| Thread | Completed outcome | Honest status |
|---|---|---|
| MLP0 token/context algebra | Exact `TT + X + CC` reconstruction; full factorial CE assay | Structural result, not compression |
| MLP1 quadratic router | Already introduced at 03:26; later deprioritized because TopK is hybrid and causal recovery remained incomplete | Useful diagnostic, not final tensor program |
| Previous-token lookup | Fixed offset-minus-one extraction transported to unseen bigrams | Real shared primitive, nonspecific removal |
| Equality/induction copy | Exact executable four-head tensor; strong natural and code extraction | OOD prediction passed; collateral certificate failed |
| Circuit census | 62/62 curated slices localized at concentration at least 2 | Location map, not 62 independent mechanisms |
| Learned subspaces | Rank-4 directions strongly enriched, but recover only 8–23% of owner effect | Sparse signal, not low-dimensional closure |
| Signed response collection | `2 × 49 × 49 × 343` causal interface measured on FIT | Complete FIT measurement |
| Response factor grid | 17 rank pairs × 3 seeds = 51/51 healthy fits | Training-only result |
| Residual rank certificate | Large pooled fit traced mainly to `m16 -> m16` amplitude | Corrected the optimistic pooled reading |
| Candidate freeze | 9 rank pairs × 3 seeds = 27 exact programs frozen | Ready for validation; no winner selected |
| Attention-band vocabulary | Real circuit bands are concentrated into a few heads; named motifs are only modestly enriched | Promising granularity, vocabulary incomplete |

## 3. What the exact MLP0 result says

Let the vector entering MLP0 before RMS normalization be

$$
x=e_t+a,
$$

where $e_t$ is the current-token-derived state and $a$ is context written by
attention. RMSNorm applies a shared scalar $\rho(t,a)$ to this sum. If $L$ and $R$ are
the two MLP0 input matrices, its bilinear gate is

$$
g(x)=(L\rho x)\odot(R\rho x).
$$

Expanding the product gives three branches:

$$
\begin{aligned}
TT &= \rho^2(Le_t)\odot(Re_t),\\
X  &= \rho^2\left[(Le_t)\odot(Ra)+(La)\odot(Re_t)\right],\\
CC &= \rho^2(La)\odot(Ra).
\end{aligned}
$$

- `TT` is the token-token or lexical branch.
- `X` is the symmetric token-context interaction.
- `CC` is the context-context continuous branch.

After the native Down map and bias are handled consistently, these branches reconstruct
the native MLP0 write to relative MSE $3.11\times10^{-13}$ in full precision and
$5.48\times10^{-6}$ in bf16. This is an exact algebraic census, not an approximation.

On 96 held-out SELECT documents, averaged over all factorial backgrounds, the branch
benefits were:

| Branch | Mean CE benefit |
|---|---:|
| `CC` | **1.1778 nat** |
| `TT` | **0.9281 nat** |
| `X` | **0.4008 nat** |

The branches are not independent. The additional pair effects were:

$$
I(TT,X)=+1.7216,
\qquad I(TT,CC)=-1.1537,
\qquad I(X,CC)=-1.0328\ \text{nat},
$$

while the residual three-way interaction was only $+0.0244$ nat.

The strongest current interpretation is therefore:

> MLP0 contains a substantial token/lexical computation and a still larger continuous
> context computation, but its most important non-additive structure is mostly
> pairwise coupling, especially between lexical and token-context terms.

This argues for a **shared or coupled factorization**, not three independent SAEs.
For example, token classes may form a sparse hierarchy, continuous context may use a
low-rank tensor basis, and some factors may be shared across `TT` and `X`. We have not
yet found and validated that joint semantic decomposition. The exact split tells us
what object it must reproduce.

## 4. Two exact attention primitives

### 4.1 Previous-token lookup

Layer-0 head 3 has a strong fixed previous-position service. Its candidate replaces
the head's learned attention pattern by the fixed offset $k=q-1$ while preserving the
head's continuous value/output tensor. After deleting the native head, this one-offset
program recovered **0.9421** of its target CE effect. Recovery was almost identical on
unseen and seen bigrams—0.9417 versus 0.9442—while moving the same tensor to wrong
offsets recovered only 0.1529 or zero.

This is excellent evidence for a reusable previous-token transport primitive. It is
not a selectively removable semantic circuit: removal damage was about +0.0625 nat on
the nominated target and +0.0632 on a matched self-attention control. The head provides
a broad low-level service used by many behaviors.

### 4.2 The strongest executable circuit: equality copying

Four heads—`L5H5`, `L7H3`, `L8H3`, and `L8H4`—implement a broad equality-fetch
service. For query position $q$ and key position $k$, define

$$
M_{qk}=\langle e_{t_q},e_{t_{k-1}}\rangle\mathbf 1[1\le k\le q].
$$

$e_t$ is a one-hot token vector. The inner product is one exactly when the current
query token equals the token immediately before an earlier key. The position $k$ then
contains what followed the earlier occurrence. For head $h$, the extracted write is

$$
z_q^{(h)}=\sum_k M_{qk}A_{qk}^{(h)}v_k^{(h)},
$$

where $A_{qk}^{(h)}$ is the head's continuous native attention weight and $v_k^{(h)}$
is its value vector. Every equality match is summed. There is no argmax, TopK, parser,
or target-label router.

The final source-closed run used 192 fresh natural documents and 192 fresh code
documents. It took **867.38 seconds**. Extraction recovery was computed as

$$
R=\frac{CE_{deleted}-CE_{extracted}}
        {CE_{deleted}-CE_{native}}.
$$

`R=1` means the tensor restores the complete measured effect of deleting the four
heads.

| Quantity | Natural FINAL | Code OOD |
|---|---:|---:|
| Target CE increase when equality service is removed | +0.468556 | +1.501658 |
| Simultaneous 95% lower bound | +0.259002 | +1.292104 |
| Target-minus-matched-control specificity | +0.488797 | +1.281762 |
| Extraction recovery | 0.908508 | 1.010413 |
| Simultaneous lower bound for recovery | 0.698953 | 0.800859 |
| Equal-price deranged-token null recovery | -0.003025 | -0.000902 |

The exact full attention replay had numerically zero KL divergence from native, and
the candidate called none of the native attention projections it claimed to replace.
This is strong evidence that the equality tensor is a real, extractable, OOD-predictive
copy service.

It still received an overall **NO-GO**. Natural off-target damage was only +0.003455
nat at the point estimate, but its simultaneous upper bound was +0.195156, above the
registered 0.01 guarantee. Code off-target damage was +0.138313 nat. The likely reason
is that equality copying supports many useful code and prose operations, not only the
narrow repeated-bigram positions called “induction targets.”

So the correct quality label is:

> Exact executable shared copy service; extraction and OOD prediction pass;
> induction-only selective removal does not pass.

The next decomposition should share the equality matcher once and separate its
payload or use branches. Relabelling all its collateral positions after seeing the
result would not be a valid fix.

## 5. What the broader circuit census taught us

### 5.1 Localization is now broad

For 62 curated behavioral slices, whole-component mean ablation and interchange
ablation were compared across all 36 attention/MLP components. All 62 localized to a
best component with concentration at least 2.0:

- minimum 2.61;
- median 4.08;
- maximum 12.28;
- the two intervention methods chose the same best component for 45/62 circuits.

On disjoint held-out rows, all 62 again cleared 2.0 and the median held-out/in-sample
concentration ratio was 1.0217. However, 17/62 best-component identities moved, and
all 17 were near ties. Thus the census tells us where behavior is concentrated, but a
single winning component should not be treated as exact when its margin is small.

It also does **not** mean there are 62 independent mechanisms. Attention 8 wins 16
slices and attention 16 wins 13. Many behavior labels can read the same shared
service.

### 5.2 Hierarchical structure exists at attention 8, but is not universal

Five attention-8 circuit directions are highly parallel:

- mean pairwise absolute cosine: 0.8942;
- one shared direction explains 91.61% of their directional variance;
- full directions are individually selective for only 1/5 slices.

After projecting out the shared direction, the residuals have mean absolute cosine
0.359 and are selective for 4/5 slices. One residual becomes *more* causally selective
after the shared part is removed. This resembles a small DAG: store one shared
substrate once, then attach behavior-specific children.

But attention 16 falsifies the idea that every crowded component has this form:

| Property | Attention 8 | Attention 16 |
|---|---:|---:|
| Variance in leading shared direction | 0.9161 | 0.4887 |
| Mean absolute cosine | 0.8942 | 0.4271 |
| Selective full directions | 1/5 | 11/13, or 7/13 with a 10% margin |
| Selective residual directions | 4/5 | 7/13, or 6/13 with a 10% margin |

Removing a “shared” direction helps at attention 8 and hurts at attention 16. The
model appears to use at least two organizations: shared-parent-plus-children in some
components and more directly separated directions in others. A universal hierarchy
would be the wrong prior; a mixture of structures is more plausible.

### 5.3 Small learned subspaces are enriched, not complete

Gradient-trained rank-4 subspaces were fit with optimizer-health gates. An earlier run
that had silently stayed near random initialization was discarded; the repaired run
required both movement from initialization and decreased training loss.

Rank 4 recovered only 8–23% of the full component's member-position causal effect.
Since four dimensions are only 0.35% of a 1,152-dimensional stream, this is strong
enrichment—roughly 22–66 times an equal dimensional share—but not a complete circuit.
Even representative rank-64 fits recovered only about 25–35%.

This is why “the behavior is in a low-dimensional direction” and “the component can
be replaced by that direction” must be kept separate.

## 6. The signed causal-response tensor

The largest new experiment does not directly compress activations or weights. It
measures a causal interface among 49 registered circuits.

For source circuit $s$, let $d_s\in\mathbb R^{1152}$ be its fitted unit direction at
the owning component. At every token position, the intervention changes the owner's
native write $y$ to

$$
y'=y-\langle y,d_s\rangle d_s.
$$

This deletes only the rank-one projection along $d_s$, not the whole component.

For target circuit $t$ in document $d$, define

$$
R_{pstd}=
\operatorname{mean}_{i\in M_t}\Delta CE_{psdi}
-\operatorname{mean}_{i\in O_t}\Delta CE_{psdi}.
$$

Definitions:

- $p$ is the **phase**: either the full source direction, or its residual after
  removing the source owner's leading shared direction;
- $s$ is one of 49 source circuits;
- $t$ is one of 49 target circuits;
- $d$ is one of 343 FIT documents;
- $M_t$ is the target's registered member-position mask;
- $O_t$ is its registered off-slice comparison mask;
- $\Delta CE$ is intervened CE minus native CE.

Positive $R$ means deleting source $s$ hurt target positions more than their matched
off-slice positions. Negative $R$ means the deletion helped them relative to the
comparison. Keeping the sign is important: absolute concentration would make helpful
and harmful effects look identical.

The measured tensor has shape

$$
2\times49\times49\times343.
$$

The FIT collection made exactly **12,400 outer model forwards** over 496 rows from
343 documents. The model state hash was identical before and after. Its transaction
published the receipt last and opened neither validation nor EVAL.

Only 70.52% of possible response cells are supported by the registered masks. Missing
cells are marked invalid; they are never filled with zeros or imputed. The 343
documents were prospectively split into 229 factor-training documents and 114
internal-validation documents.

This tensor is not the transformer and it is not 49 completed circuits. It is a
measured input-output table for asking whether many causal effects share a smaller
program.

## 7. The tensor program fitted to that response

The candidate family is

$$
\widehat R_{pstd}=
\sum_{k=1}^{K_0}A_{pk}B_{sk}C_{tk}H_{dk}
+\sum_g\mathbf 1[s\in g]
\sum_{j=1}^{K_g}A^{(g)}_{pj}B^{(g)}_{sj}C^{(g)}_{tj}H^{(g)}_{dj}.
$$

The first sum is a rank-$K_0$ library shared by all source owners. The second gives
each of six source-owner groups a private rank-$K_g$ child library.

For one atom:

- $A$ says how the atom changes between full and residual phases;
- $B$ says which source deletions activate it;
- $C$ says which targets it affects;
- $H$ is its coordinate in a document.

This is a genuine tensor-network program: it contains sums and products with a fixed
owner mask, not data-dependent TopK support. It can be drawn as shared parent atoms
plus owner-private child atoms. Whether those atoms are identifiable or semantic is a
separate test.

Two prices are reported rather than hidden in one chosen scalar:

$$
P=100K_0+355K_g,
\qquad
C=K_0+6K_g.
$$

- $P$ counts persistent factor scalars. The 100 comes from 2 phases + 49 sources + 49
  targets for each shared atom. The 355 is the corresponding total over all six
  owner-private libraries.
- $C$ counts temporary coordinates needed for one new document.

We also report prediction multiply-adds, calibration cells, and code-solve cost. A
small $P$ with an enormous $C$ would merely move complexity from model storage into
per-document inference, so both matter.

## 8. What the complete training grid actually found

The grid tested 17 $(K_0,K_g)$ rank pairs at three optimizer seeds: **51/51 fits were
healthy**, with zero scientific failures. The sum of per-fit optimizer time was
1,052.76 seconds. Validation and EVAL were never read.

The best pooled training point in the grid was shared rank 32:

| Quantity | Value |
|---|---:|
| Shared rank $K_0$ | 32 |
| Private rank $K_g$ | 0 |
| Persistent price $P$ | 3,200 scalars |
| Per-document state $C$ | 32 scalars |
| Median FIT MSE | 0.016047438 |
| Zero-predictor MSE | 0.046072904 |
| Observation-mean MSE | 0.042948380 |
| Pooled energy recovery | **65.17%** |
| Worst owner-pair NRMSE | **1.6219** |

The 65.17% calculation is

$$
1-\frac{0.016047438}{0.046072904}=0.6517.
$$

This means a 32-coordinate shared code explains 65.17% of squared **training response
energy under pooled weighting**. It does not mean 65% of model parameters, 65% of CE,
or 65% of behavior.

Nine rank pairs are nondominated on training price/error coordinates:

$$
(K_0,K_g)\in
\{(1,0),(2,0),(4,0),(4,1),(8,0),(8,2),(16,0),(16,4),(32,0)\}.
$$

All private-only candidates were dominated. The three joint shared/private candidates
remain on a frontier only because they exchange slightly lower persistent price for
more per-document state and worse fit than the next shared-only rank. That is **not**
positive evidence for the proposed owner-private hierarchy.

## 9. The crucial correction: pooled success is dominated by MLP16

The 49 sources and targets belong to six owner components, giving 36 source-owner to
target-owner interfaces. Pooling every valid cell weights a large-amplitude interface
far more than a small one.

For the shared-rank-32 residual:

- `m16 -> m16` residual energy per valid cell is 0.148770;
- that is **9.676 times** the next-largest source-owner residual energy;
- but its normalized rank-16 unfolding tail is 0.14514, only **1.1035 times** the
  next owner.

An **unfolding** reshapes a tensor block into a matrix; the rank-16 tail is the fraction
of squared singular-value energy left after the best rank-16 matrix approximation.
Normalizing by the block's own energy asks about structural rank rather than amplitude.

Therefore `m16 -> m16` is exceptionally large, but not demonstrably exceptionally
higher-rank. Adding a special large MLP16 private library was not justified by this
test.

More importantly:

- shared rank 32 removes **68.41%** of raw `m16 -> m16` energy;
- it removes **-0.32%** of `a16 -> m16` energy;
- it removes **0.38%** of `a3 -> m16` energy;
- the median recovery over all 36 equally weighted owner interfaces is only
  **5.57%**.

Thus the accurate reading is:

> A 32-coordinate shared code captures a strong, high-amplitude MLP16 causal-response
> family. It has not yet compressed the full library of interfaces uniformly.

This correction is why we froze the current programs for validation instead of
immediately increasing rank or tuning a favorable block-balanced loss after seeing
the result.

## 10. What is frozen, and what remains unopened

The union of the pooled and robust training Pareto frontiers contains the nine rank
pairs above. All three seeds were frozen for each pair, giving **27 exact candidate
programs**. No best seed or winner was chosen.

The first freeze artifact was rejected by an independent audit because it included
forbidden training-score fields, left mutation windows between repeated reads, and
lacked post-publication semantic replay. That failed artifact remains preserved and
cannot be promoted.

Freeze v2 contains only identities, hashes, byte counts, and literal $P,C$ prices. It
passes six focused publication/mutation tests plus independent analysis- and
terminal-mutation attacks. Its artifact SHA-256 is
`53f8264228e905ad1a459f32204d1acb07fa044e7753026dbb0bcfb91ac77b98`.

The next validation boundary has now been specified and its pure role reducer tested:

- expose exactly the 114 preregistered validation documents;
- score all 27 candidates, dropping none after seeing outcomes;
- use unconditional prediction and physical calibration budgets of 2, 4, 8, and 16
  source arms;
- report pooled error, every owner-pair error, worst-owner NRMSE, support, conditioning,
  calibration cells, solve cost, and prediction cost;
- do not select a winner inside the scorer.

The validation reducer and related focused suites currently pass **23 tests**. No
validation response value or candidate tensor was opened while writing them. The
remaining work is to wrap this reducer and scorer in the same source-closed,
receipt-last lifecycle as training, then execute it.

## 11. How simplicity is being validated

We no longer treat “low rank,” “sparse,” or “small Frobenius error” as self-justifying.
A proposed simplicity measure is useful only if paying less under it buys a capability.

The current ladder is:

1. **Literal price:** persistent scalars $P$, per-document state $C$, multiply-adds,
   and calibration measurements.
2. **Training reconstruction:** necessary for debugging, insufficient for selection.
3. **Held-out causal prediction:** predict signed effects on untouched documents and
   unmeasured source-target cells.
4. **Balanced coverage:** do not win only by fitting the largest-amplitude interface.
5. **Gauge/identifiability:** different seeds should agree after only known tensor
   permutations and reciprocal rescalings.
6. **Composition:** predictions should survive an RMSNorm/residual/next-component
   interface rather than only reconstructing a local table.
7. **Practical consequence:** use the factors for a fresh intervention, extraction,
   selective removal, or OOD transport.

This is how we can eventually decide whether one definition of simplicity is better
than another: the better definition should predict which cheaper program retains
these downstream abilities.

## 12. Other recent downstream results

### 12.1 Errors can be repaired downstream

For the current compressed early-layer assembly, residual-stream relative MSE is not
monotone with depth. It rises from about 0.52 at block 2 to 1.7415 at block 6, then
falls to 0.5925 by block 17—a 66% recovery from the peak.

An exact layerwise split showed that attention writes inject most early error while
MLPs partly remove it:

- at block 5: attention adds +0.8523 relative-MSE units, MLP adds +0.1059;
- across blocks 2–9: attention contributes +1.8617 while MLPs contribute -1.0857;
- `a1`, `a5`, and `a6` supply most of the positive attention injection.

This matters for compression metrics. Local error at an early boundary is not a lower
bound on final error; charging it as if every error survives would overprice some
directions.

A rank-32 linear “absorber” could read 59.73% of `a1`'s injected residual variance on
held-out positions, but installing it made block-2 error 9.79% worse, improved the
block-6 peak by 11.37%, and changed CE by only +0.0033 nat. The same correction can be
locally harmful and later helpful because downstream modules were fit for the original
context. This closes the naive one-site linear-absorber route; coordinated composition
matters.

### 12.2 Attention is often head-addressable, but our motif vocabulary is incomplete

Of 311 census leaves, 208 are probed by bands inside an attention component rather
than by one named head. For every one of those 208 bands, its top two heads explain a
larger share than random directions at the same component:

- median top-two share above component baseline: +0.1590;
- minimum excess: +0.0203;
- median absolute top-two share: 0.7491;
- 208/208 above their own component baseline.

So a head-grained program can address the circuit bands. But the usual motif labels
(`previous-token`, `self`, `induction`, and so on) are only modestly enriched among
those heads:

- observed named-motif fraction: 0.7260;
- component-matched base rate: 0.6341;
- enrichment: 1.1449 times;
- preregistered bar: 1.20, therefore **failed**;
- `previous-token` supplies 215 of 416 top-head slots, while named induction supplies
  only 13.

The permutation null matches the component-matched base rate within 0.0005. A later
20,000-draw permutation that preserved component clustering gave null mean 0.6338,
standard deviation 0.0207, $z=4.46$, and $p=0.00005$; none of 20,000 draws reached the
observed 0.7260. Thus the enrichment is statistically clear even though its magnitude
misses the registered 1.20 effect-size bar. Statistical certainty and practical effect
size answer different questions.

The simplest proposed composition was then tested on the 31 leaves whose two leading
heads are both previous-token heads: can a per-token-identity lookup predict leaf
membership? Held-out AUC was 0.5086 using the previous token and 0.5130 using the
current token; 0/31 leaves reached 0.60 with either. An AUC of 0.5 is chance. This
refutes the literal rule “the circuit fires when the previous token is token X.” It
does not refute token pairs, token classes, position-conditioned features, or richer
value reads.

The honest conclusion is that circuit bands are concentrated in heads and motif heads
are modestly but certainly overrepresented, while the current motif-plus-unigram
language is insufficient. The strongest reusable attention primitive looks more like
broad previous-token transport whose **contextual use** remains to be decomposed than
a collection of already named induction heads.

## 13. What the mathematics contributed

The mathematical reviews produced four useful changes rather than a magic closed-form
solution:

1. **Exact polynomial expansion** made the MLP0 `TT/X/CC` decomposition possible and
   exposed its pairwise interaction structure.
2. **Signed response factorization** converted a vague “shared dictionary or DAG” idea
   into a falsifiable tensor program with held-out causal predictions and literal
   prices.
3. **Tensor unfolding with amplitude normalization** showed that the apparent MLP16
   rank problem is mostly a weighting problem. This prevented an unjustified private
   rank expansion.
4. **Quotient-Jacobian accounting** was verified on toy matrix and CP factorizations.
   It distinguishes known scaling/basis gauge from additional non-identifiability and
   will be applied only if a response program survives validation.

Two other mathematical tools were verified but are not current selectors:

- sparse Boolean Möbius tomography exactly recovered a planted 8-term degree-at-most-3
  interaction program from 202 subset queries rather than all 4,096; it also confirmed
  that the MLP0 three-way CE interaction is tiny relative to pair terms;
- tensor/Frobenius or HOSVD objectives remain useful controls, but geometry-to-causal
  correlations were weak and sometimes changed sign, so geometry alone cannot select
  editable circuits.

The main lesson is that tensor structure is most useful when it gives an exact local
algebra, a composable candidate family, or a lower-bound/certificate. It is not enough
to call a low-rank fit interpretable.

## 14. Current blockers and confusing results

There is no missing checkpoint, FineWeb cache, dependency, GPU, or user decision
blocking the next safe work.

The blockers are scientific:

1. **Held-out response transport is unknown.** The 65.17% result is training-only.
2. **Pooled and interface-balanced metrics disagree.** A useful compiler must cover
   more than the loudest block.
3. **Document codes may memorize.** A free $H_d$ for each document is not zero-shot
   prediction; new-document codes must be inferred from a small priced calibration
   panel or replaced by observable document features.
4. **Tensor factors have gauge freedom.** Atoms cannot be named until seed-aligned
   stability and conditioning are measured.
5. **Composition is still missing.** A response-table factor has not yet been inserted
   through real RMSNorm/residual/downstream interfaces.
6. **No terminal circuit is surgically editable.** Equality copying is very close on
   extraction and OOD but serves more behaviors than its narrow target mask.

The most confusing result is also the most important: downstream computation can
ignore, compensate, amplify, and later repair local errors. This explains why local
$R^2$, Frobenius reconstruction, and CE can disagree sharply. It also means a good
simplicity measure should be defined on a downstream causal quotient—what later
readers can distinguish—not only on raw activation energy.

## 15. Current plan, in order

### 1. Finish and run the source-closed 114-document validation scorer

Score all 27 frozen programs with no reselection. This is the cheapest experiment
capable of killing or validating the entire response-factor direction.

### 2. Branch only on the prospectively defined failure pattern

- If pooled and block-balanced held-out prediction both work, proceed to factor
  stability and semantic interventions.
- If pooled prediction transports but small interfaces fail, compare one
  prospectively fixed block-balanced fit at matched price.
- If held-out prediction fails broadly, reject causal-response factorization v1 rather
  than repairing it indefinitely.

### 3. Certify any survivor modulo gauge

Align the three seeds under factor permutation and reciprocal rescaling; measure
conditioning and robust Kruskal ranks; then test finite factor interventions. This is
where tensor-similarity and quotient mathematics become useful.

### 4. Test whole-model composition

Use the survivor to predict and execute a fresh intervention across an MLP/RMSNorm/
residual boundary. Measure CE, interface-balanced effects, extraction, collateral,
and OOD. Only this can promote a response regularity into a model mechanism.

### 5. Continue terminal readers with better-scoped shared services

For copying, store the equality matcher once and search for separate payload/use
branches. In parallel, use the head-concentrated census bands as endpoints, but do not
assume the current motif labels are complete. Several good downstream readers are
valuable because common upstream writers can then be selected by their actual reader
interfaces rather than arbitrary activation geometry.

### Alternate entry point if response validation fails

Construct an empirical controllability/observability quotient: choose early directions
by which downstream readers and losses can distinguish them, merge states that have
the same measured future consequences, and factor only the resulting quotient. This
directly incorporates the observed downstream repair and is presently the best-return
alternative to another local MLP fit.

## 16. Glossary

- **CE / cross-entropy:** average negative log-probability of the correct next token;
  lower is better.
- **nat:** one natural-log unit of CE.
- **FIT:** data allowed for fitting candidate structure.
- **validation:** untouched FIT documents used after candidates are frozen to compare
  them; not the final OOD role.
- **EVAL / FINAL / OOD:** protected roles used only after selection and preregistration.
- **owner:** the attention or MLP component where a source direction is deleted.
- **interface or owner pair:** one source-owner to target-owner block of the response
  tensor.
- **rank:** number of multilinear atoms, not automatically the number of semantic
  features.
- **NRMSE:** root-mean-square prediction error divided by a frozen scale for that
  block; values above one mean error exceeds that scale.
- **Pareto frontier:** candidates for which no other candidate is at least as cheap and
  accurate in every registered coordinate and strictly better in one.
- **gauge freedom:** changing factor coordinates—such as permuting atoms or multiplying
  one factor while inversely scaling another—without changing the represented tensor.
- **extraction:** restoring a behavior with only the proposed program after deleting
  its native owners.
- **selective removal:** removing the target computation without materially harming
  matched unrelated behavior.
- **collateral:** prediction damage outside the registered target.
- **tensor-native:** executable using fixed sums, products, and contractions. A TopK
  router is hybrid unless its comparison logic is explicitly represented and priced.

## 17. Primary artifacts behind this explanation

- [MLP0 exact branch findings](../MLP0_TOKEN_CONTEXT_TENSOR_FACTORIAL_FINDINGS.md)
- [induction terminal result](../induction_equality_tensor_final_ood_v2_retry1_result.json)
- [signed FIT response receipt](../causal_response_tensor_v1_fit_receipt.json)
- [factor training analysis](../causal_response_factorization_v1_training_analysis.json)
- [residual unfolding certificate](../causal_response_residual_unfolding_certificate_receipt.json)
- [frozen candidate library](../causal_response_factorization_v1_candidate_freeze_v2.json)
- [current validation amendment](../CAUSAL_RESPONSE_FACTORIZATION_V1_AMENDMENT_16.md)
- [latest full strategic review before this explanation](../HOURLY_STRATEGIC_REVIEW_2026-08-30_1350.md)
