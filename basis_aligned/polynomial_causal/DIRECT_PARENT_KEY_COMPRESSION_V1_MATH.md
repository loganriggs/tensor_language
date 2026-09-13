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
