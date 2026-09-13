# Compress the larger even-key parent while preserving its child

13 September 2026. The existing hierarchy defines a parent scalar field $p$,
an extracted child $c$, and remainder $r=p-c$, all writing along the same
declared head9 direction. Parent/remainder removal has a substantial native
spelling effect on the new lexical panel; the small MLP10 cross interaction
does not. See `LEXICAL_MAIN_EFFECT_PRIORITY_V1_AUDIT.json`.

For the parent's two QK factors, the even-key route averages their ordinary
and reflected-key score products. Reflection uses a shared 64-dimensional
source basis $B$. Two existing frozen weight-only fits supply 48-dimensional
rotations $V$: the shared-query product fit and the complete-even fit. Test
$B_{48}=BV$ while retaining full Q/K maps, their normalization denominators,
value reader, positions and the child producer. The compressed remainder is
$\widehat r=\widehat p-c$; the original child's definition is not changed.

Earlier tests evaluated changes induced by an upstream writer and failed
some signed conditions. They did not settle preservation of the direct parent
field. On the existing 72 contexts in two upstream states, both candidates
pass the registered 10% direct-parent scalar criterion across all three groups.
The original scalar replays to at most $1.78\times10^{-7}$ relative error.
The endpoint diagnostic initially read the padded last array position, producing
undefined ratios on some groups. Gathering each actual sequence endpoint
repairs that diagnostic; the all-token criterion is unchanged. The original
diagnostic is preserved separately rather than silently erased.

The stored source basis shrinks from 73,728 to 55,296 scalars. This is 18,432
fewer scalars in one basis, not a 25% reduction of the parent or complete model.
Full readers, value production, child field, native context and downstream
computation remain. Unlike the rejected independent MLP9 factorization, this
candidate preserves the shared upstream producer rather than duplicating it.

Native validation is registered on the frozen 48-row lexical panel: both
candidate bases, parent/remainder removal, strengths -1, 1 and 2. The original
child remains exact. Each candidate scalar is applied as an actual head9 writer
edit; MLP9 and the native suffix are recomputed. Require at most 10% target and
control effect error in every family/branch/strength cell, and no material sign
reversals. This tests downstream fidelity and signed reuse, not broad semantic
selectivity. The initial 21–36% capability and control-ratio observations remain
limited to their single control contrast.

CPU code/receipts: `direct_parent_key_compression_v1.py`,
`DIRECT_PARENT_KEY_COMPRESSION_V1_RESULT.json`, and the preserved
`DIRECT_PARENT_KEY_COMPRESSION_V1_PADDED_DIAGNOSTIC.json`.
Managed validation: `ops/run_parent_key_main_effect_v1.py`, frozen
`PARENT_KEY_MAIN_EFFECT_V1_BINDING.json`; follow
`PARENT_KEY_MAIN_EFFECT_V1_RESULT.json` for the terminal verdict.

## Native signed result

The managed run completed in 6.13 seconds. Parent-field replay is exact and
paired native capability remains positive in all four lexical styles. Both
frozen candidates pass every registered aggregate target/control error cell
across parent/remainder removals and strengths -1, 1 and 2. Worst target errors
are 4.197% for the query-product fit and 4.879% for the complete-even fit;
worst control errors are 3.859% and 3.889%, respectively. This is a new
preservation result for the larger main effects, not a retraction of earlier
failures on upstream-induced interaction differences.

The strict no-material-sign-reversal predicate fails for both candidates.
Each has five reversals, all on the work/jobs control contrast, none on the
spelling target. They occur in three prompts, primarily at doubled strength.
The largest absolute effect error is about 0.00145 nats, so they should not
be dismissed as numerical replay noise. Some references are much smaller,
but the originally declared $10^{-5}$ material threshold remains unchanged.
`PARENT_KEY_MAIN_EFFECT_V1_SIGN_AUDIT.json` lists every event. These localized
control failures coexist with the aggregate preservation passes; unrestricted
adoption or universal selective manipulation is not established.

Literal parent-interface pricing counts its four full Q/K maps, value reader,
writer and key basis. It falls from 665,856 to 647,424 scalars: **2.768%**, not
25%. The latter percentage describes only the basis. This preserves shared
native producer dependencies, but native prefix, child and suffix remain
outside this interface cost. See `DIRECT_PARENT_KEY_COMPRESSION_V1_PRICE.json`.
The next relevant check is broader downstream preservation and selectivity,
rather than claiming these two readouts cover all consumers or refitting on
the few observed exceptions.

## Full-vocabulary preservation, with individual-prompt limits

The next managed test evaluates unit parent removal on the same frozen 48
lexical rows, scoring all 50,304 modeled output logits. This expands the output
scope, not the input panel or intervention strengths. Let $d$ be the native
removal's logit change and $e$ the candidate's error in that change. Compare
their norms after subtracting their vocabulary means, so uniform score shifts
do not count as behavioral information. A second metric uses baseline token
probabilities as weights and subtracts the corresponding weighted means.
Square errors and reference norms are summed over each family before division.

The distribution comparison uses

$$
\frac{\sum_x D_{\mathrm{KL}}(p_{\mathrm{native\ removed}}\Vert
 p_{\mathrm{candidate\ removed}})}
{\sum_x D_{\mathrm{KL}}(p_{\mathrm{native\ removed}}\Vert p_{\mathrm{baseline}})}.
$$

All four registered predicates pass in 2.80 seconds, including prior selected
readout replay within $4.77\times10^{-7}$. The query-product candidate has
2.405–4.028% centered vocabulary error and 2.368–4.185% probability-weighted
error. Its KL error is 0.054–0.168% of the native-removal KL; mean absolute
candidate KL ranges from $9.34\times10^{-7}$ to $1.42\times10^{-5}$ nats.
The complete-even candidate also passes, with worst weighted error 4.798%.
Thus the preservation result is not confined to the two selected readouts.

An individual-prompt audit narrows this result. Each candidate exceeds 10%
weighted error on five of 48 prompts; maxima are 14.752% for query-product
and 14.219% for complete-even. These exceptions are American-cued prompts
with relatively small native-removal KL. The query-product maximum centered
error is 15.465%. These are not erased by family aggregation, and the earlier
specific control sign failures remain. The supported claim is aggregate
full-vocabulary preservation for this input panel and unit parent removal,
not uniform per-token/per-prompt fidelity, corpus-wide OOD or universal
composition with other edits.

Executor/binding: `ops/run_parent_full_vocab_v1.py` and
`PARENT_FULL_VOCAB_V1_BINDING.json`. Primary receipt and countercheck:
`PARENT_FULL_VOCAB_V1_RESULT.json`, `PARENT_FULL_VOCAB_V1_PROMPT_AUDIT.json`.

## Corpus transfer fails uniformly across domains

Forty fixed 128-token prefixes extend the test beyond spelling instructions:
eight FineWeb documents and eight each from Pile discussion, reference,
biomedical and legal/patent panels. FineWeb is the training-corpus comparison;
Pile supplies corpus shifts. These are existing cached panels, with no new
score filtering or fitting. The Pile panel was originally coverage-filtered
for another study; pretraining disjointness and historical nonuse are not claimed.
Only the parent is needed, so this runner executes the native prefix directly
without constructing the child. Exact 64-basis sign re-encoding replays its
scalar field exactly.

| Domain | Query-product weighted error | Complete-even weighted error |
|---|---:|---:|
| FineWeb | 13.94% | 14.59% |
| Discussion | 8.32% | 9.08% |
| Reference | 8.55% | 9.49% |
| Biomedical | 1.51% | 2.22% |
| Legal/patent | 16.16% | 18.05% |

Both candidates fail the all-domain centered, probability-weighted and relative
KL criteria. For the query-product candidate, centered errors are 12.41% on
FineWeb and 13.58% on legal/patent text; KL ratios are 1.924% and 2.616%,
above the 1% bar. The run takes 2.48 seconds. The spelling-panel aggregate
success therefore does not justify general corpus preservation.

Deleting any one document leaves both failed domains above 10% weighted error.
For query-product, leave-one-out errors remain 12.52–14.90% on FineWeb and
14.95–17.97% on legal/patent text. Thus a single outlier does not explain the
failures. Some passing discussion/reference aggregates are less stable under
document deletion, which also limits broad positive claims from eight documents.
Context length and content both differ from the lexical panel; this experiment
does not attribute the gap to corpus identity alone. Keep the exact 64-dimensional
parent as the general reference. Further work should diagnose the transfer gap
before refitting on these documents or claiming the smaller interface adopted.

Rows, binding and results: `PARENT_CORPUS_TRANSFER_V1_ROWS.json`,
`PARENT_CORPUS_TRANSFER_V1_BINDING.json`, `PARENT_CORPUS_TRANSFER_V1_RESULT.json`,
`PARENT_CORPUS_TRANSFER_V1_DOCUMENT_AUDIT.json`. Managed implementation:
`ops/run_parent_corpus_transfer_v1.py`.

## Long-range error hypothesis fails (13 September 13:28 UTC)

Split the routing error into query–key distances below64 and at least64,
add each part separately to the exact parent scalar, then propagate the actual
writer removal through the native suffix. No candidate was refitted.
The long-distance component accounts for only9.66% of probability-weighted
full-error alignment on FineWeb and16.01% on legal/patent, missing the70% bar.
Short-distance-only error remains12.74% and13.88%, above the10% fidelity bar.
The output errors nearly add: closure error is0.24–0.97% of full error across
all five domains. Field partition error is3.83e-15 and prior metric replay exact.

Deleting one document at a time leaves long-distance alignment at2.30–16.12%
on FineWeb and12.59–20.25% on legal/patent. A single document does not rescue
the long-distance explanation. Alignment is a signed projection, not a
nonnegative variance allocation. These hybrids retain the exact64-dimensional
reference and are diagnostic, not compressed implementations. This rejects
one explanation of the transfer gap; it neither identifies content as its cause
nor rules out effects of context length on nearby-token representations.
Receipts: `PARENT_LAG_ERROR_V1_RESULT.json`,
`PARENT_LAG_ERROR_V1_DOCUMENT_AUDIT.json`; frozen execution/binding:
`ops/run_parent_lag_error_v1.py`, `PARENT_LAG_ERROR_V1_BINDING.json`.


## Product-space compression: exact spectrum and a pricing rejection

Instead of pruning the shared source readers, consider compressing their products.
Let $u=B^T x\in\mathbb R^{64}$, $A=K_1B\in\mathbb R^{128\times64}$,
and $C=K_2B$ with the same shape. The inside/inside contribution has source feature

$$
F(u)=(Au)(Cu)^T\in\mathbb R^{128\times128}.
$$

Its contraction with the two query vectors is the product of both QK scores.
This is one retained numerator term. The outside/outside term, source-position
rotations, native normalizers and scalar value are still required. In particular,
this is not the spectrum of the whole behavioral parent or its actual query distribution.

On an orthonormal basis of symmetric64-by64 matrices, the linear map
$X\mapsto AXC^T$ has2,080 input coordinates and16,384 output coordinates.
We compute its coefficient Gram exactly from $A^TA$ and $C^TC$, without
materializing the full feature map. A tiny explicit-matrix control agrees to
$1.99\times10^{-16}$ relative error. The best rank128 approximation has84.19%
relative coefficient error; reaching10% requires1,885 directions.

The stronger practical rejection is cost: a dense rank1,885 two-adapter map
stores34,804,640 scalars, versus16,384 in the original two factored maps $A,C$.
Even rank1 would require18,464 scalars. Shared $B$, native producers and other
terms remain on both sides. These figures price the explicit materialized
$A,C$ representation; native-map reuse could make its incremental storage even
smaller. Consequently this dense adapter family cannot provide the requested
local storage saving. Low matrix rank is not the same as low arithmetic complexity:
the original product expression is already a compact nonlinear program.

A repeated-input countercheck changes the norm. For isotropic Gaussian $u$,
the second moment of its symmetric quadratic features is

$$
M=2I+tt^T,\qquad t=\operatorname{coordinates}(I_{64}).
$$

The Gaussian output spectrum is obtained from $M^{1/2}GM^{1/2}$, where $G$
is the coefficient Gram. It needs1,883 directions for10% error; rank128 gives
83.53% error. Its trace agrees to$1.12\times10^{-15}$ with the independent identity

$$
\mathbb E\|F(u)\|_F^2
=\operatorname{tr}(A^TA)\operatorname{tr}(C^TC)
+2\operatorname{tr}(A^TA C^TC).
$$

Thus the broad spectrum is not repaired by this particular function-space norm.
It does not rule out sparse/block arithmetic graphs, actual-query-constrained
compression, or reuse across terms. Do not launch a learned dense product-space
fit here: its storage loses before optimization. A useful successor must retain
factored adapters and exploit the complete query/source contraction, including
position semantics, rather than expand the source map into dense output features.

Code and receipts: `source_product_spectrum_v1.py`,
`SOURCE_PRODUCT_SPECTRUM_V1_RESULT.json`,
`SOURCE_PRODUCT_SPECTRUM_V1_GAUSSIAN_AUDIT.json`.


## Keep the nodes, prune reader edges

A different candidate keeps64 source directions but sets the smallest25% or40%
of entries of the original basis $B$ to zero. Call the resulting matrix $S$.
Its columns are no longer orthonormal, so using $SS^T$ as a projector would
change the parity construction. Instead use

$$
P_S=S(S^TS)^{-1}S^T=QQ^T,
\qquad Q=S\operatorname{chol}(S^TS)^{-T}.
$$

Thus reflection remains a reflection in the new subspace. Both candidates are
chosen entirely from weights in the original basis gauge; no text fitting.
The payload stores a bitmask and FP32 nonzero values. The CPU/GPU reference
reconstructs the orthobasis in FP64 after unpacking. Projection orthogonality
and an independent factored-inside contraction replay to below3e-15. Native
runtime normalization and all other readers remain unchanged.

The25% candidate passes direct-parent validation on72 cached contexts in two
states: all-token relative error3.06–5.30%, actual-endpoint error1.68–2.69%.
The40% candidate fails, with all-token errors up to17.95%. It is not promoted.
The25% basis is not the original subspace: its projector error is12.48%.
This is an approximation, not an exact re-encoding of the original parent.

Pricing at common FP32 precision: original basis294,912bytes; sparse25 payload,
bitmask, shape metadata and a conservatively charged dense64-by64 correction
require246,800bytes, a16.31% saving. The declared entire parent interface saves
only1.806%. This is an edge-sparse representation with an additional small dense
correction, not a node reduction. The current reference expands a dense FP64
basis (589,824bytes), so it demonstrates neither resident-memory nor runtime
savings. Archive-container overhead is excluded from the logical payload count.
The40% candidate's larger storage saving does not rescue its failed fidelity.

The frozen25% candidate then runs unit parent removal through native MLP9 and
the entire suffix on the same40 fixed128-token corpus prefixes used earlier.
All50,304 output logits are scored. Exact64-basis re-encoding replays exactly.

| Domain | Probability-weighted effect error | Relative KL |
|---|---:|---:|
| FineWeb |5.42%|0.317%|
| Discussion |9.81%|0.963%|
| Reference |7.95%|0.651%|
| Biomedical |2.77%|0.076%|
| Legal/patent |16.48%|2.714%|

Uniformly centered output error passes10% in all domains (maximum9.65%), but
weighted error and the1% KL-ratio criterion fail on legal/patent. These are
separate metrics; the centered pass cannot replace the failed weighted claim.
FineWeb improves over the earlier rank48 candidate's13.94%, at a different
storage budget. This is not a matched-cost superiority result.

Leaving out one document at a time keeps FineWeb weighted errors at2.43–6.96%
and legal/patent at10.07–18.18%. The latter failure is not erased by deleting
one document. Discussion/reference passes are fragile to document deletion;
individual prompts also fail in every domain. These small reused corpus panels
support a candidate comparison, not broad OOD certification or selective reuse.

This fixed-gauge magnitude pruning is only a baseline for sparse readers. A
weights-only orthogonal re-encoding can change which edges are small while
leaving the original unpruned subspace fixed. A useful next structural test
would alternate that re-encoding with sparse approximation, with the projection
correction still charged. Do not fit the failed legal documents to erase this
validation failure. Full native extraction/removal/composition checks remain
required even if a later weight-only sparse fit improves preservation.

Code/payload/results: `sparse_parent_reader_v1.py`,
`SPARSE_PARENT_READER_V1_PROGRAM.pt`, `SPARSE_PARENT_READER_V1_RESULT.json`;
managed `ops/run_sparse_parent_corpus_v1.py`,
`SPARSE_PARENT_CORPUS_V1_BINDING.json`, `SPARSE_PARENT_CORPUS_V1_RESULT.json`,
`SPARSE_PARENT_CORPUS_V1_DOCUMENT_AUDIT.json`.


## Learned sparse coordinates: converged fit, mixed behavioral transfer

Keep the same55,296 nonzero reader entries and optimize their coordinate frame:

$$
\min_{R^TR=I,\;\|S\|_0\le55296}\|BR-S\|_F^2.
$$

For fixed $R$, retaining the largest-magnitude entries of $BR$ solves the $S$
step exactly. For fixed $S$, write $B^TS=U\Sigma V^T$; $R=UV^T$ solves the
orthogonal Procrustes step exactly. Each cycle therefore cannot increase this
objective. It is a nonconvex alternating solver, with no global recovery guarantee.
The objective approximates a basis after a freely chosen orthogonal re-encoding;
it does not directly minimize the full normalized QK parent or behavioral error.

Ten starts (identity plus nine seeded random orthogonal frames) all reach
Riemannian tangent-gradient RMS below1e-7 in568–1,383 cycles. Total CPU time is
15.28seconds. No recorded cycle increases the objective. The best weight-only
arm reduces squared basis loss from0.52723 to0.21062, a60.05% reduction, at the
same representation cost. Its projector error falls from12.48% to8.00%.
The best arm is selected by weight loss, before native validation.

The saved sparse values are FP32. Recovering a Procrustes frame from this payload
and redoing hard thresholding changes zero support entries. The threshold margin
is6.04e-6 and tangent-gradient RMS remains8.38e-8. This supports a stable local
fixed point after storage rounding. Cross-start node correspondence was not
measured, so no stable identified-reader claim is made.

| Domain | Original-frame sparse25 weighted error | Learned-frame sparse25 weighted error |
|---|---:|---:|
| FineWeb |5.42%|8.05%|
| Discussion |9.81%|3.48%|
| Reference |7.95%|5.02%|
| Biomedical |2.77%|1.96%|
| Legal/patent |16.48%|16.52%|

The unchanged native corpus test again passes centered-vocabulary error in all
domains but fails the probability-weighted and KL-ratio criteria on legal/patent
(relative KL2.723%). Better weight fit is not uniformly better circuit preservation.
The comparison now has matched support/storage and locally converged optimization.
It does not establish that the global best sparse basis would fail, nor that a
composed-operator objective would behave like this unweighted basis objective.

A paired-document audit finds lower squared output error on30/40 documents,
including6/8 FineWeb and6/8 legal/patent. Yet those two aggregate metrics worsen:
counting wins hides error magnitude. Legal/patent remains10.99–19.10% weighted
error after deleting any one document; the other four domains stay below10%
under that check. Do not select another start using these validation outcomes.

The useful next change is the *weight-only composed fitting objective*, retaining
the sparse reader graph and charging its projection correction. The shared two-QK
product and full even-key terms supply a more relevant metric than uniform basis
error. Existing query-product and complete-even objectives are prior art to reuse,
not newly discovered ideas. Native normalization, positions and outside terms must
remain explicit. These results support investigating that objective mismatch;
they do not prove it is the sole cause of the legal/patent failure.

Code/payload/receipts: `sparse_reader_rotation_v1.py`,
`SPARSE_READER_ROTATION_V1_PROGRAM.pt`, `SPARSE_READER_ROTATION_V1_RESULT.json`,
`SPARSE_READER_ROTATION_V1_PAYLOAD_AUDIT.json`; native runner
`ops/run_rotated_sparse_parent_corpus_v1.py`,
`ROTATED_SPARSE_PARENT_CORPUS_V1_BINDING.json`,
`ROTATED_SPARSE_PARENT_CORPUS_V1_RESULT.json`,
`ROTATED_SPARSE_PARENT_CORPUS_V1_DOCUMENT_AUDIT.json`.


## Composed sparse fitting passes the five-domain aggregate test

The important change is to optimize the actual even-key *numerator composition*,
not basis reconstruction. This still uses only model weights. The old
`complete_even_key_objective_v1.py` assumes a candidate subspace contained in
original $B$. Sparse readers move outside that subspace, so applying that formula
unchanged would be invalid. The new kernel allows an arbitrary full-rank $S$.

Write $H_S=I-2S(S^TS)^{-1}S^T$. At a fixed relative position let
$M_i=Q_i^T R^T K_i$, with the same rounded position convention as before.
The complete even numerator is the polynomial

$$
F_S(x,y)=\frac12\left[(x^TM_1y)(x^TM_2y)
+(x^TM_1H_Sy)(x^TM_2H_Sy)\right].
$$

The two query slots share $x$, and the two source slots share $y$, so the
coefficient tensor must be symmetric in each pair. Comparing with original
$H_B$, the unchanged first term cancels. Both reflected product tensors have
the same coefficient norm because reflection is orthogonal. Their cross-inner
product therefore suffices to calculate the squared difference exactly.

For $M_i=L_iK_i$ with $L_i\in\mathbb R^{1152\times128}$, the kernel uses
128-by128 query Grams and cross-Grams of the reflected key readers. It avoids
materializing a1152-to-the-fourth tensor. Small explicit tensors verify the
loss to3.4e-16 and directional gradients to2.5e-10 normalized error. The original
projector gives zero loss to numerical precision. Native value/gradient cost
is13–15ms on two CPU threads.

Fitting retains the same55,296 nonzero entries and the same two candidate masks
(original-frame and learned-frame). Five starts per mask perturb the retained
weights by different amounts. Each forward pass normalizes column scales and
uses the corrected projector. FP64 L-BFGS with strong-Wolfe line search fits four
fixed relative positions:1,4,16,63. All ten fits reach unit-coordinate gradient
RMS below8.4e-10 in19.08seconds total; Gram conditions are1.43–1.72. Each mask's
five starts converge to essentially the same objective, but this is not a
certificate over all masks or all local optima.

The best composed objective falls from0.00486426 for the learned-frame sparse
baseline to0.00284073, a41.60% reduction. These are squared coefficient errors
normalized by the full unreflected product's squared norm, summed over the four
positions. They are not native behavioral error percentages. FP32 payload
rounding changes the objective by less than2.3e-15. The frozen best arm also
improves the coefficient objective at all seven tested unfit positions:
0,2,8,32,64,96,127. This checks numerator-position transfer, not normalized
attention or corpus transfer by itself.

Native validation then keeps the original normalization, values, positions,
context and suffix, and performs unit parent removal on the same40 corpus
prefixes. All four registered predicates pass in2.46seconds:

| Domain | Weighted effect error | Relative KL |
|---|---:|---:|
| FineWeb |5.06%|0.239%|
| Discussion |3.60%|0.130%|
| Reference |4.35%|0.183%|
| Biomedical |3.85%|0.148%|
| Legal/patent |4.02%|0.162%|

Centered-vocabulary error is2.66–5.07%, also below10%. This is the first
all-domain aggregate preservation pass among the directly tested parent
compression candidates. Legal/patent improves from16.52% for the basis-fitted
candidate to4.02% at identical support and storage price. It supports the user's
proposal to fit the composition rather than independent module reconstruction.
It does not prove that numerator fitting is sufficient for general circuits.

Counterchecks: every leave-one-document-out domain aggregate stays below5.62%
weighted error. However,6 of40 individual documents exceed10%, with a maximum
of25.00%; an aggregate pass is not a uniform per-document guarantee. These are
small, reused validation panels. Although no model text fitted the weights,
this panel has informed method development, so it is not a fresh final test.
Signed strengths, child/remainder reuse, selective controls and genuinely new
corpus inputs remain required for promotion.

The logical storage price is unchanged:16.31% saving on the basis representation
and1.806% on the declared parent interface, including the bitmask and correction.
The existing dense reference execution still establishes no runtime or working-
memory gain. Neither full-model compression nor all four circuit properties is
claimed.

Kernel/controls: `sparse_complete_even_v1.py`,
`check_sparse_complete_even_v1.py`, `SPARSE_COMPLETE_EVEN_V1_CONTROL.json`.
Fit: `fit_sparse_complete_even_v1.py`, `SPARSE_COMPLETE_EVEN_FIT_V1_RESULT.json`,
`SPARSE_COMPLETE_EVEN_FIT_V1_PROGRAM.pt`, `SPARSE_COMPLETE_EVEN_V1_POSITION_AUDIT.json`.
Native: `ops/run_composed_sparse_parent_corpus_v1.py`,
`COMPOSED_SPARSE_PARENT_CORPUS_V1_BINDING.json`,
`COMPOSED_SPARSE_PARENT_CORPUS_V1_RESULT.json`,
`COMPOSED_SPARSE_PARENT_CORPUS_V1_DOCUMENT_AUDIT.json`.


## Signed hierarchy preservation and additional-document transfer

The same frozen composed sparse candidate now replaces the parent while keeping
the existing child exact. Its remainder is still defined as compressed parent
minus original child. On48 lexical prompts, evaluate actual parent and remainder
writer removal at strengths$-1,1,2$, then recompute MLP9 and the native suffix.
This covers signed use of both hierarchy branches; it does not yet test every
independent combination of child and remainder strengths or every output token.

All family/branch/strength aggregate errors pass10%: maximum1.994% on spelling
targets and1.761% on the work/jobs control contrast. Native parent replay is
exact and paired spelling capability remains positive. The strict no-material-
sign-flip criterion fails twice, both on controls:

| Row | Branch/strength | Native effect | Candidate effect | Absolute error |
|---|---|---:|---:|---:|
|34|parent,2|0.00058770|-0.00006342|0.00065112|
|38|remainder,1|0.00015831|-0.00013638|0.00029469|

Effects are logit-contrast changes in nats. These are small effects, but exceed
the registered1e-5 reference threshold; they are not relabelled numerical noise.
No spelling-target sign reversals occur. This strengthens signed preservation
while explicitly failing universal control-sign preservation.

For input transfer, a second corpus panel contains96 distinct128-token prefixes:
32 FineWeb documents and16 each from discussion, reference, biomedical and
legal/patent. Fixed source order selects documents excluded from the40-prefix
compression-development panel. Exact prefix hashes are distinct within the new
panel and disjoint from the old one. The Pile metadata contains repeated entries
for different target positions, so selection deduplicates source-row indices;
the old40-prefix panel was also checked and has no such duplication.

No candidate changes or text fitting precede this test. These are existing
project caches, not historically untouched documents. The Pile cache was
coverage-filtered for another study, and pretraining disjointness/near-duplicate
exclusion are not claimed. This is input transfer outside this compression
comparison's development panel, with those provenance limits.

All four native unit-parent criteria pass in3.91seconds:

| Domain | Documents | Weighted effect error | Relative KL |
|---|---:|---:|---:|
| FineWeb |32|5.07%|0.258%|
| Discussion |16|2.69%|0.073%|
| Reference |16|2.36%|0.053%|
| Biomedical |16|3.10%|0.096%|
| Legal/patent |16|3.13%|0.098%|

Centered-vocabulary error is2.37–5.10%. Each domain still passes after deleting
any one document (maximum weighted error5.65%). However,12/96 individual
prefixes exceed10% relative error. The two largest ratios,74.65% and66.52%,
have tiny reference effects: probability-weighted reference RMS1.46e-5 and
8.86e-5nats, with candidate-distribution KL5.89e-11 and1.74e-9nats. Other failures
have larger absolute effects; the legal example at20.46% error has weighted
error RMS0.000839nats. Relative exceptions remain recorded rather than erased
by an after-the-fact threshold.

Together these results support approximate preservation of this declared parent
interface across additional corpus inputs and signed hierarchy use. They do not
establish universal per-input fidelity, new semantic selectivity, arbitrary
cross-circuit composition or faster execution. Logical storage savings remain
16.31%of the reader representation and1.806%of the parent interface.

Signed receipts: `COMPOSED_SPARSE_SIGNED_V1_BINDING.json`,
`COMPOSED_SPARSE_SIGNED_V1_RESULT.json`, `COMPOSED_SPARSE_SIGNED_V1_SIGN_AUDIT.json`;
runner `ops/run_composed_sparse_signed_v1.py`.
Additional-document receipts: `COMPOSED_SPARSE_NEW_DOCS_V1_ROWS.json`,
`COMPOSED_SPARSE_NEW_DOCS_V1_BINDING.json`, `COMPOSED_SPARSE_NEW_DOCS_V1_RESULT.json`,
`COMPOSED_SPARSE_NEW_DOCS_V1_DOCUMENT_AUDIT.json`,
`COMPOSED_SPARSE_NEW_DOCS_V1_SMALL_EFFECT_AUDIT.json`;
runner `ops/run_composed_sparse_new_docs_v1.py`.


## Actual sparse execution: correct, but slower and larger in memory

The packed payload is now executed through an actual sparse source projection,
without expanding an orthobasis. Let $G=S^TS$ and precompute both adapters

$$
A_i=K_iSG^{-1}\in\mathbb R^{128\times64}.
$$

For current source rows $X$, calculate $Z=XS$ **once**, then produce both
inside-key projections as $ZA_1^T$ and $ZA_2^T$. The inverse correction is
absorbed into the adapters and need not remain in runtime state. This makes
shared computation explicit: both QK factors consume the same64 source reads.

The fair dense comparator uses the same approximate projector's orthobasis $Q$,
precomputes both $K_iQ$, and also computes $XQ$ once. It does not repeat the
source read just because the earlier reference routing loop did so. Runtime
comparison is consequently between two implementations of the same candidate,
not a comparison that credits sparse execution for fixing a duplicate baseline.

The sparse implementation stores $S^T$ in CSR with32-bit indices. On cached
native ports and independent probes, inside-key replay error is at most2.05e-15
in FP64 and5.40e-7 in FP32 against the dense FP64 reference. Normalization,
position rotation, score products, value contraction and suffix are outside
this kernel benchmark; earlier native validation establishes candidate behavior.

Two-thread CPU benchmarks use7 warmups and31 alternating timing samples at
1,19,128,512 token rows. Sparse/dense speed ratios are0.277–0.534 in FP64 and
0.226–0.458 in FP32. Thus this CSR kernel is about1.9–4.4 times slower, failing
the1.1-times speed criterion at every tested size. Prepared resident storage,
including both folded-key adapters, also loses:

| Precision | Shared dense | CSR sparse | Change |
|---|---:|---:|---:|
| FP64 |720,896bytes|794,884bytes|+10.26%|
| FP32 |360,448bytes|508,164bytes|+40.98%|

The bitmask/value payload remains smaller for storage. CSR replaces compact mask
bits with per-entry indices, which erase that gain at75% density. Do not present
the packed-storage saving as a memory or inference-speed improvement.

A layout countercheck explicitly makes the sparse right-hand input contiguous,
and separately times an optimistic prepacked input that excludes copying cost.
Neither wins: even prepacked speed ratios are only0.237–0.612. Input transposition
alone does not explain the negative runtime result. No GPU implementation was
benchmarked, so this is a CPU/library result rather than a hardware-independent
speed bound.

The candidate remains a useful behavioral and packed-representation result,
not an adopted runtime compression. If execution becomes the priority, the next
structural assumption should constrain the sparsity pattern during weight-only
composed fitting (for example small fixed groups), instead of continuing to
optimize this irregular mask and expecting an execution gain from entry count.
Higher sparsity may harm fidelity and must be measured, not assumed.

Code/results: `sparse_parent_executor_v1.py`,
`SPARSE_PARENT_EXECUTOR_V1_RESULT.json`,
`SPARSE_PARENT_EXECUTOR_V1_LAYOUT_AUDIT.json`.
