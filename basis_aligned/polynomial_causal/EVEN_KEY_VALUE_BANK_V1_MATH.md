# Full-value even-key interaction, 14 September

The existing exact key-coordinate rewrite now supplies all 128 value channels of
head9.8. For its fixed projection, let G be the product of both native normalized
QK score arrays, and G_ref the product with reflected key numerators and unchanged
native denominators. The extracted component uses the causal gate

\[
G_{even}=(G+G_{ref})/2,\qquad
Y=G_{even}[(1-\lambda)XV_9^T+\lambda X_0V_0^T]O_9^T.
\]

X and X0 are supplied normalized current and initial inputs. Both QK factors,
rounded rotary constants, native signed mixture, and full residual output map
are retained. This is a conditional interaction decomposition; reflection is not
a new native forward pass under a key edit. The chosen key projection is fixed
from the earlier scalar circuit. Extending its value channels does not identify
128 new circuits or establish new semantic roles.

[Four-context replay](EVEN_KEY_VALUE_BANK_V1_RESULT.json) agrees with independent
old scalar executions within 5.2e-16. The program stores 1,048,577 scalars in
4,263,970 serialized bytes, including QK, key coordinates, both value maps, output
map and mixture. Native input generation and downstream consumers remain external.

**Timing correction:** the first receipt's `pred_b` used 256 separate current and
inherited scalar evaluations, so it does not satisfy the registered 128-call
comparison. Its 50.1x figure is not the accepted benchmark. The
[matched correction](EVEN_KEY_VALUE_BANK_MATCHED_V1_RESULT.json) mixes values
first and evaluates exactly 128 scalar channels: 73.38 ms versus 2.81 ms, or
26.11x, on the first fixed 18-token context. Both arms include value projections,
mixture and output map. This measures reuse versus repeated scalar evaluation,
not speedup over already-vectorized native attention or the full model.

The separate [standalone response extraction](COMPLETE_PORTS_EXTRACTION_V2_RESULT.json)
passes 432 cases with no repository imports. It retains its existing rank-64
approximation and supplied pristine states. The
[whole-tensor](INTERACTION_PACKAGE_SHARED_BANK_V1_RESULT.json) and
[row-sharing](INTERACTION_PACKAGE_ROW_SHARING_V1_RESULT.json) screens across five
distinct conditional programs both fail the 1% saving gate. Stop literal
deduplication for that portfolio; those nulls do not rule out algebraic sharing.

[Native-module recomposition](EVEN_KEY_NATIVE_RECOMPOSITION_V1_RESULT.json) now
passes on four cached contexts: even plus independently computed dense-basis odd
matches the actual FP32 attention module within 3.16e-7 relative error, and direct
FP64 algebra within 8.96e-16. Omitting odd causes 10.37–15.27% full-head output
error; the even component is not a faithful whole-head replacement. Both branches
retain the native signed current/first value mixture and output map.

## Native selective-removal result

[The registered full-value test](EVEN_KEY_FULL_VALUE_NATIVE_V1_RESULT.json)
completed 768 body forwards in 9.82 seconds on the reused 72 regional and 32
natural prefixes. Instrument and historical replay A/D pass: live even+odd versus
native head write <=3.99e-7, self-donor exactly zero, native capability present.
Regional selectivity B and unrelated preservation C both fail.

| Regional family | Full-even removal coverage | Earlier scalar coverage | Even donor transfer |
|---|---:|---:|---:|
| 0 | 43.02% | 42.28% | 47.46% |
| 1 | 41.84% | 41.96% | 46.75% |
| 2 | 44.51% | 43.42% | 47.48% |

All removal pairs and donor rows have the directed sign, but all families miss
the frozen 50% removal/transfer bars. Family0 also fails the unrelated/target
removal ratio: 0.527 versus <=0.5. Mean absolute newline CE changes are
0.03448/0.02522 versus <=0.02; the first half's maximum 0.11365 also misses <=0.1.
The earlier scalar's unrelated-readout changes are much smaller while its cue
coverage is similar. This does not promote the expanded component as a selective
circuit or refute the earlier scalar result.

The strongest explanation is that added value directions carry other computations.
The [physical remainder test](EVEN_KEY_REMAINDER_NATIVE_V1_RESULT.json) removes
full-even minus the known scalar, rather than merely subtracting measured effects.
Its 208 forwards finish in 3.33 seconds with bit-identical native anchors.
Paired cue-change norm is only 4.89/3.91/6.36% of the scalar's; mean absolute
unrelated-readout changes are 6.26/4.23/7.09 times the scalar's. Both directional
predictions pass. This localizes the cue-dependent part to the old scalar while
its complement carries substantial other effects. The complement still changes
individual cue-token logits; small paired cue changes do not mean zero effect.

Separate scalar and remainder effects predict full-even removal within 1.54–4.48%
on both regional readouts in all three families. However, overall composition C
fails: the first natural half's newline CE error is 5.458%, above 5%, while the
second half is 1.733%. Remainder mean absolute newline CE is 2.12/5.76 times
the scalar's. These are conditional development-panel results, not a newly
named semantic circuit, universal additivity or OOD identification.

The [CE-boundary control](EVEN_REMAINDER_CE_BOUNDARY_V1_RESULT.json) completed 128
forwards in 2.85 seconds, with all four native CE anchors bit-identical. The
analytic decomposition is exact, but the hypothesis that metric curvature mostly
explains the failure is rejected. Raw-score/final-RMS interaction has L2 norm
0.726/1.045 times the total CE interaction; softcap curvature 0.026/0.054 and
logsumexp curvature 0.355/0.359. These signed terms can cancel; they are not
nonnegative shares. Native downstream nonlinearity remains a material limit.

## One shared graph for the three components

The [shared graph](EVEN_VALUE_SHARED_GRAPH_V1_RESULT.json) expresses the old
scalar reader in the native current-value row space and its writer in the native
output column space. Both weight-derived spans replay below 9.72e-16. The scalar
branch needs two 128-coordinate vectors instead of two 1152-coordinate vectors.
S, R=even−S and O share routing, value reads and the output map. Their requested
writes combine in 128-dimensional head space before the unchanged nonlinear
suffix; no additive-logit approximation is assumed.

All 36 tested binary/signed gain cases pass <=1e-10 write replay on four old
contexts. The [storage audit](EVEN_VALUE_SHARED_GRAPH_PACKED_V1_RESULT.json)
caught padded least-squares buffers in the first serialization. Explicit owned
clones preserve every tensor value and produce a 4,265,602-byte packed program
versus 4,281,986 bytes for a matched private-reader/writer reference: 0.383% saving
for this conditional program. The scalar branch itself drops 18,432→2,048 bytes.
The whole native model, external contexts, source code and suffix are not included
in that program-file saving. This is exact reuse, not quantization.

The [factorial identifiability control](EVEN_VALUE_FACTORIAL_V1_IDENTIFIABILITY.json)
shows that the six measured S/R/O removal combinations leave two independent
ambiguities among pair and triple interactions. The missing S+O and R+O removals
were [registered](EVEN_VALUE_FACTORIAL_NATIVE_V1_PREREGISTRATION.md) and then
measured. The [completed factorial](EVEN_VALUE_FACTORIAL_NATIVE_V1_RESULT.json)
adds 312 forwards in 4.91 seconds, with identical native anchors and shared-write
error <=1.32e-15. Pair terms predict full-removal effects within 0.145–0.275%
across the registered regional readouts and natural halves. This supports a
conditional low-order response description on this development panel.


## Crossing MLP9 with an exact conditional interaction program

Write the post-attention residual as z(a)=z0−aS*S−aR*R−aO*O. The bilinear
MLP9 numerator is quadratic in these three strengths: one constant, three linear
and six quadratic vector coefficients. Native RMS contributes a quadratic scalar
denominator, including the original FP32 epsilon. The residual and Down bias
remain explicit. [The compiler](sro_mlp9_rational_v1.py) derives these coefficients
from the original weights and pristine states; it does not fit intervention outputs.

[44 CPU cases](SRO_MLP9_RATIONAL_V1_RESULT.json) replay the FP64 state within
3.00e-16 relative error and its own change within 3.29e-14. The MLP9 triple term
is 0.090–0.191% of full removal change; holding RMS fixed makes it <=1.03e-14.
Thus normalization alone can produce local higher-order coupling even with a
quadratic numerator. This does not attribute all downstream nonlinearity to MLP9.

[Native suffix validation](SRO_MLP9_SUFFIX_NATIVE_V1_RESULT.json) compares five
old prefixes at eleven binary and signed strength combinations: 110 forwards,
2.20 seconds. Compiled post-MLP9 states have relative error <=2.13e-7 and full
vocabulary scores <=6.30e-7. All registered baseline-subtracted target/control/CE
checks pass. The compiled arm replaces block9 with its per-context program while
keeping the native prefix, inherited values and downstream blocks explicit.

[CPU amortization](SRO_MLP9_AMORTIZATION_V1_RESULT.json) compares a baseline
that already caches shared input projections. Including preparation, 32 queries
cost 83.07→26.62 ms at batch1 and 497.51→207.20 ms at batch8 (3.12×/2.40×).
One query is slower (0.524×/0.492×); eight queries give 1.121×/0.944×.
These are seven interleaved trials on one 16-token prefix, repeated at batch8,
using FP64 and two CPU threads. They do not establish GPU or whole-model speed.
Prepared program sizes across the five suffix prefixes are 2.07–20.67 MB.
All 15,926,400 original MLP scalars and context generators remain required.
This is exact conditional computation reuse, with no quantization or demonstrated
static whole-model parameter saving.

## Which branches carry paired cue changes?

The [selected development-panel reduction](SRO_CUE_REDUCTION_V1_RESULT.json)
retains S, O and their pair term SO, then predicts every measured removal corner.
Maximum paired-cue error per family is 4.36/3.28/4.38% of the full-head removal
cue effect, passing the registered 10% criterion. S alone gives 14.84/16.69/31.47%.
Adding O is material; SO has only a small incremental benefit on these cases.
This is a response reduction selected after development results, not an independent
validation or a learned static circuit. O is a computational complement whose
semantic role remains unresolved.

Dropping R is inappropriate for general outputs: the same reduced description
misses up to 72.6–79.3% of unrelated-control effects and 63.9/87.2% of natural CE
effects. Retain the original full-value selectivity failure. The useful hypothesis
is a cue-specific S/O interaction with a separately priced remainder, not a
universal head replacement. Next test: freeze this reduction on unseen prompt
constructions and reserve R-containing intervention combinations for prediction.


## New-construction reduction failure and its cause

The [frozen 72-row test](SRO_FRESH_CUE_V1_RESULT.json) executed all eight removal
corners: 576 forwards in 8.05 seconds. Shared-write replay passes <=2.78e-15.
Using only corners 0/1/4/5 to predict R-containing corners gives maximum cue errors
8.20/31.96/10.43% for archival notes, opposed reviewer/diary locations, and
community newsletters. The 10% reduction gate fails. Native positive cue pairs
are 12/8/12 of 12, so the opposed-role capability gate also fails; newsletters
pass capability and still miss the reduction bar. No failed family is discarded.

The [post-failure decomposition](SRO_FRESH_REMAINDER_V1_RESULT.json) finds direct
R paired effects of 8.11/31.42/10.43% of the full-head effect. Restoring that main
term reduces maximum all-corner cue prediction errors to 1.69/4.80/4.44%.
This is explanatory analysis on the failed test, not independent validation of a
repaired reduction. Full pair-only output predictions still meet the transferred
1% bar: individual target/control errors range 0.057–0.455%. The failure concerns
which branch carries the cue, rather than a breakdown of low-order response
composition. Withdraw any context-general treatment of R as unrelated background.

Next discriminating direction: separate the full even remainder's current-value
and inherited-value sources with the same routing/output maps, then measure which
source carries the newly material cue effect. Earlier fixed-writer scalar sector
results do not answer this full 128-channel remainder question.


The [native remainder source test](REMAINDER_SOURCES_NATIVE_V1_RESULT.json)
adds 288 forwards in 4.31 seconds with identical native/R anchors. Removing only
Rc approximates R cue effects with 22.18/18.88/9.82% error, failing the 20% bar
in archival notes. Rc has the larger cue-effect norm, but inherited values cannot
be discarded uniformly. Separate Rc/Rf effects compose within 1.12–4.49% across
paired cue and control readouts. This remains localization on the reused panel,
not selective identification of either source. The source instrument costs two
routing evaluations plus a third reference call during this native test.


## Confirmation and article-confound correction

The [four-term confirmation](SRO_FOUR_TERM_CONFIRMATION_V1_RESULT.json) executes
576 forwards in 8.02 seconds. Five calibration corners predict the remaining
three with 5.60/4.75/2.96% maximum cue error. The response-error criterion passes,
but overall confirmation does not: native positive pairs are 12/1/12, so the
opposed publisher/author role family fails capability. No clean writer-role
interpretation follows from accurately predicting the model's response.

**Instrument correction:** text inspection found “A Oxford museum” here and
“For a Austin community newsletter” in SRO_FRESH_CUE_V1. The old validator only
checked British/American articles. Mark family2 in both original panels as
article-confounded; native capability does not remove that confound. Prior raw
results and thresholds remain, but the newsletter result alone cannot establish
clean template generalization failure. Archival and opposed-role families do not
contain this particular confound, and their results remain separately available.
The new shared city guard rejects any indefinite article immediately before a
controlled city as an experimental restriction, not a general grammar checker.
[SRO_ARTICLE_CORRECTION_V1](SRO_ARTICLE_CORRECTION_V1_PREREGISTRATION.md) freezes
uniform article correction across all 48 affected rows and tests both response
rules at the original10%bar. No favorable-row selection or silent replacement.


[Corrected article results](SRO_ARTICLE_CORRECTION_V1_RESULT.json): 384 forwards,
5.55 seconds; A/B/C pass, D fails. Both corrected families have 12/12 positive
native cue pairs. Four-term maximum cue errors are4.20/2.54%; three-term errors
10.049/8.231%, so the original10%bar still rejects the three-term rule narrowly.
Do not present the marginal miss as a large effect or silently round it to a pass.
Four-term unrelated-control errors reach7.72/13.50%; no general preservation
criterion was registered or passed. The unaffected manuscript family passes
four-term prediction at5.60%, but the opposed publisher/author family remains
capability-failed. This supports limited conditional response prediction with
five calibration corners, not a clean writer-role circuit or static compression.

The next source-localization question concerns inherited values at the controlled
city positions versus all other positions. Even routing remains computed from
the complete native context; value-source localization alone cannot identify all
cue information entering through Q/K or the upstream prefix.


## Inherited city sources are insufficient

[192 native forwards](INHERITED_SOURCE_POSITIONS_NATIVE_V1_RESULT.json),3.72seconds,
replay the corrected native anchor exactly. City-source-only cue errors are
112.68/90.46% of the full inherited effect, failing20%. Whole inherited cue effects
are1.591/0.667% of full-head effects, so museum also fails the1%materiality floor.
City/other source effects compose within0.11–0.57% on cue/control readouts. Thus
values outside the changed city position dominate this conditional inherited
contribution. Routing and upstream context remain intact; this does not locate
all cue information or show that city information is absent from Q/K.

Demote further work on this small inherited contribution. The larger current
remainder admits an exact cross-boundary port factorization Rc(r,v)=G_even(r)Z_R(v),
where Z_R(v)=(1−lambda)V(v)−(V(v)·c)u. The original SRO coordinates specify c,u.
No new model coefficients are fitted. [CPU control](CURRENT_REMAINDER_CROSSED_V1_CPU_CONTROL.json)
replays the original current remainder within1.69e-16 and independently contracts
(dG)(dZ), giving the complete three-term change within2.77e-16. Synthetic mixed
norm is1.437×full change; cancellation makes scalar norm fractions non-additive.
Native behavioral sufficiency and composition are tested separately by the
[registered crossed-port test](CURRENT_REMAINDER_CROSSED_NATIVE_V1_PREREGISTRATION.md).


[Crossed native results](CURRENT_REMAINDER_CROSSED_NATIVE_V1_RESULT.json) add288
forwards in3.51seconds. Native/joint-recomposition anchors are identical, and
component/field identities are <=4.02e-16. Joint current-remainder interchange
has11.43/11.19% of full-head removal's paired-cue norm. Value-only error is
33.74/14.09%, rejecting uniform20%sufficiency. Separately measured routing,
value and mixed effects compose within0.49–3.47% across cue/control readouts.
Omitting mixed effects gives14.73/9.83% cue error and32.26/48.65%control error,
failing5%. Retain the mixed computation. These are conditional recipient-suffix
interventions, not three independent semantic circuits. Cached routing/value
ports for48prefixes cost1,051,456bytes, plus unchanged model/context generators.

## Compressing coupled routing/value interactions without dropping mixed terms

If routing and values interpolate with strengths a,b, the current-remainder
write change is a*Dr+b*Dv+a*b*Dm. The mixed strength is determined by those two
inputs. Feeding this state through bilinear MLP9 with RMS gives a rational
function with numerator and denominator both of bidegree(2,2). Its nine monomials
are1,a,b,ab,a²,a²b,b²,ab²,a²b². The earlier generic three-direction compiler has
ten coefficients; its linear mixed coefficient and routing/value cross coefficient
now multiply the same ab monomial and can be added exactly.

[100 CPU cases](COUPLED_ROUTING_VALUE_MLP9_V1_RESULT.json) use four old cached
paired contexts and all25 strengths from{-1,0,.5,1,2}². All registered checks pass:
FP64 state error<=2.64e-16, own-change error<=8.39e-14, agreement with the generic
compiler<=5.01e-17, native FP32 MLP state error<=1.20e-7. Context payload shrinks
by7.113–7.116%, including residual basis, bias and denominator. First serialized
examples are2,336,025bytes generic versus2,169,945bytes coupled. No quantization
or response fitting is used. All15,926,400MLP scalars and native context/source
weights remain required to construct programs. This does not reduce static
whole-model weights or support arbitrary independent mixed-strength edits.
The next validation boundary is the native suffix under these coupled strengths.


The [coupled native suffix test](COUPLED_NATIVE_SUFFIX_V1_RESULT.json) adds208
forwards in2.63seconds. All100signed cases pass: post-MLP9 state error<=1.08e-7,
full-vocabulary scores<=7.07e-7, and all baseline-subtracted readout floors hold.
Prepared savings remain7.11%at18–19tokens. This validates the constrained two-input
interface through the native suffix on four fixed contexts.

## Projecting the rational state into complete attention10

After block10 re-entry, write raw10=P(a,b)/rho(a,b). P has bidegree(3,3), hence
16vector coefficients, including re-entry lambdas and x0. Set
r=mean(P²)+epsilon*rho². The normalized attention input is P/sqrt(r); a normalized
query/key is QP/sqrt(mean((QP)²)+epsilon*r). These epsilon scale factors are
required by the native model; normalization is not treated as exactly scale-free.

Project all16coefficients through each of Q1,Q2,K1,K2,V once during preparation.
A16×16Gram matrix per token computes mean(P²) without reconstructing ambient P.
Runtime attention consumes these projected ports, original output weights,
mixture and fixed first values; both QK products, rounded RoPE and causal masks
remain explicit. This extends the existing shared-denominator principle to a
complete normalized attention consumer, rather than dropping attention or bias
as in a selected polynomial self-product.

[25 CPU cases](COUPLED_ATTENTION10_PORTS_V1_RESULT.json) on one native18-token
context pass all gates: normalized ports<=1.30e-15, full write<=7.20e-16, native
FP32 attention write<=3.22e-7. Conditional consumer payload34,183,446→18,700,562
bytes (-45.29%), serialized18,705,109bytes. Original model/context generators
remain globally required. FP64 output-map cast cache costs10,616,832bytes;
casting all independent attention weights costs63,701,000bytes, separately from
stored payload. The measured layout saves payload only through43tokens; longer
prefixes may lose. No latency or global parameter reduction follows from this.

The first checker completed calculations and serialized its program, then failed
on an st_size metadata typo. Recovery preserved the artifact and verified every
recomputed tensor bit-identical before producing the result. No scientific
threshold or result was replaced. Complete block9/attention10 replacement also
needs the source program for the residual path; its native test charges that
additional dependency instead of claiming the consumer-only45.29%for the pair.


## Combined native replacement and layout recovery

The first combined run stopped at its first compiled attention10 call: the native
first-value tensor is B,T,9,128, whereas the CPU fixture was B,T,1152. The consumer
incorrectly used that headed shape for the final output-map multiplication.
[V1 failure receipt](COUPLED_ATTENTION10_NATIVE_V1_FAILURE.json) preserves the run,
code, log and partial example. V2 canonicalizes this interface; all50 flat/headed
CPU comparisons are bit-identical and malformed shapes are rejected. No weights,
mathematical formula, rows, amplitudes or scientific criteria changed.

[Combined V2 native validation](COUPLED_ATTENTION10_NATIVE_V2_RESULT.json) adds208
forwards in2.86seconds. All100cases pass A/B/C. Post9 state error<=1.08e-7,
post10<=1.53e-7, full-vocabulary score<=6.73e-7; all baseline-subtracted readout
floors and exact zero-effect controls pass. Compiled arm replaces MLP9 output
and full attention10 while retaining native re-entry, MLP10 and the later suffix.

Charging both the source program needed by the residual path and required x0,
combined payload is20,950,564bytes at18tokens versus34,183,452independent, and
21,819,060bytes at19tokens versus34,312,548:36.41–38.71%saving. The example file
is20,873,585bytes; required x0 is supplied externally and charged separately.
These are prepared-context executors, with all original model/context generators
still required globally. The earlier45.29%figure remains consumer-only. Long
prefixes, general parameter reduction and execution speed are separate questions.


[Matched CPU query cost](COUPLED_ATTENTION10_AMORTIZATION_V1_RESULT.json) retains
B FAIL. Including attention-port preparation,32queries speed up1.358×atbatch1
but0.991×atbatch8, below1.1×in both cells. Single-query ratios are0.115/0.081;
eight-query0.658/0.480. Warm batch8 also misses acceleration at8/32queries
(0.852/0.779); cold/warm ordering varies, so do not infer broad latency benefits.
Three interleaved trials use one18-token prefix repeated atbatch8, FP64/two CPU
threads. Dense weights and common source preparation are already cached;
both arms produce source state and attention write. Replay error<=5.77e-17.
Storage, fidelity and timing verdicts remain separate.

[Isolated extraction](COUPLED_INTERACTION_EXTRACTION_V1_RESULT.json) exports
[the fixed-context package](extracted_circuits/coupled_interaction_v1/README.md).
Python-I executes all25strengths with exactly zero discrepancy from the compiled
reference and no project imports outside the package. Total package size,
including program, executable, manifest and README, is20,965,029bytes versus
34,183,452bytes of independent payload. It returns postMLP9, attention10 and the
post-attention10 residual; explicit x0 and re-entry lambdas are included. x0 was
recomputed on CPU from checkpoint embeddings, while the postMLP9/attention10
programs were copied from the native-validated example; this provenance is in
the manifest. The package handles one fixed recipient/donor context. It requires
no checkpoint at runtime, but original weights/native prefix generated it and
MLP10/later blocks remain external. No arbitrary-text model extraction is claimed.


The [longer-prefix forecast](COUPLED_NATURAL_LENGTH_V1_FORECAST.json) fixes two
natural-text pairs at84 and98tokens before model scores. With the residual source
program and x0 included, the combined layout breaks even only through35tokens
(the consumer-only boundary was43). Forecasts are78.27MB versus42.70MB at84tokens,
and90.43MB versus44.51MB at98tokens, so the general size claim is already rejected
for this layout. The registered204-forward native test will check accuracy on
these contexts without replacing them with shorter favorable cases. The forecast
is a payload calculation, not executed long-context accuracy or timing evidence.


[Longer-context native validation](COUPLED_NATURAL_LENGTH_V1_RESULT.json) completes
204forwards in3.71seconds. All100signed cases pass accuracy and own-effect gates:
post10 error<=2.59e-7, full scores<=6.55e-7. Price C fails exactly as forecast:
combined payload is1.833×independent at84tokens and2.032×at98tokens. Donors are
different natural texts, not controlled cue swaps; this tests a different input
and length distribution without identifying a new semantic circuit.

## Keep the denominator factored: twelve rather than sixteen vectors

Write P=rho*(B0+aB1+bB2+abB3)+sum_j N_j*m_j, with nine bidegree(2,2)
monomials m_j. Since rho itself is in that nine-monomial span, absorb rho*B0
into the nine N_j. The exact representation then needs three vectors weighted
by rho*a,rho*b,rho*ab, plus nine vectors weighted by m_j:12vectors total.
Use a12×12norm Gram instead of16×16. Compile directly from original vectors;
do not recover coefficients by division through small mixed-denominator terms.

The scalar functions are generically independent when the highest mixed rho
coefficient is nonzero: rho*a,rho*b,rho*ab add distinct leading monomials outside
the nine-monomial rectangle. The integer design control has rank12 for both
old13columns and selected12columns. Degenerate physical contexts can have lower
rank; no per-context minimality is claimed.

[50 CPU cases](DENOMINATOR_FACTORED_ATTENTION10_V1_RESULT.json) on18- and84-token
native-context fixtures pass: normalized ports<=5.21e-16, attention writes
<=4.52e-16 against expanded16, and native FP32 attention<=3.45e-7. Identical CPU
x0 frames and actual captured first values are used for both constructors.
Consumer payload ratios are0.822/0.771; combined payload ratios to independent
are0.515/1.469. The long-context size failure survives this exact improvement.
The full short/long native-suffix test is separately registered.


### Native twelve-vector validation and shared-weight limit

[Native validation](DENOMINATOR_FACTORED_NATIVE_V1_RESULT.json) passes all200
signed comparisons across eight short/long contexts (412 total forwards,5.52s):
post10 error<=2.59e-7, full scores<=6.73e-7, and registered own-effect floors hold.
Consumer payload is17.8–23.2% smaller than expanded16. Combined short programs
save46.67–48.46%;84/98-token programs remain1.469/1.624 times the independent
implementation. Prediction D fails. This is conditional execution fidelity,
not a new semantic identification or evidence of faster execution.

[Shared-weight portfolio audit](COUPLED_SHARED_WEIGHT_PORTFOLIO_V1_RESULT.json)
counts each native map once across all eight contexts (438 tokens total).
Projected programs require304,594,284bytes versus88,403,772bytes for the shared
native bank, a3.44549 ratio. All256 assignments of projected versus dense contexts
were enumerated; retaining the bank and projecting no contexts is smallest.
The fixed-context export benefit therefore does not establish shared-service
weight compression. Stop further short-prefix byte-only sweeps on this route.

Accounting correction: prior combined payloads omitted8bytes for two block10
re-entry scalars. The new audit adds them explicitly; original receipts remain
unchanged and all verdicts survive. Original prefix/model generators are still
required globally. Runtime and working memory were not tested by this audit.

### Shared-bank conditional executor

The [standalone shared executor](COUPLED_SHARED_EXECUTOR_V1_RESULT.json) retains
one original-dtype attention10 weight bank and accepts source coefficients, x0
and first values per context. It exactly executes post-MLP9, attention10 and the
post-attention10 residual state for both18- and84-token examples. Across25
registered and seven off-grid strength pairs per context, maximum relative
error is7.55e-16. A Python isolated process imports no project module and reads
no checkpoint. Headed and flattened first-value layouts agree within6.80e-16.

The package plus two example contexts is45,039,673bytes versus75,012,204bytes
for two projected12-vector programs. This demonstrates reuse of one conditional
executor across lengths. It does not remove the original prefix/model and MLP9
weights needed to generate each source program, and MLP10 plus the later suffix
remain external. Native suffix fidelity for this shared-bank interface is the
next registered boundary; the CPU result alone is not behavioral or OOD proof.

[That native boundary](COUPLED_SHARED_EXECUTOR_NATIVE_V1_RESULT.json) passes all
registered checks in140forwards/2.57seconds. Across eight existing recipients,
zero plus seven off-grid coupled interventions give post9 error<=1.65e-7,
post10<=2.32e-7 and full-score error<=7.88e-7. Every nonzero intervention is
live and all baseline-subtracted target/control effect bars pass. The complete
bank, eight caller contexts and runtime source occupy88,406,675bytes, only
2,903bytes above the tensor-only shared-native formula and29.02%of the projected
portfolio. This supports native conditional extraction and reuse across existing
contexts, not unseen-text prediction. All545,902,902native parameters remain
required for context generation and the suffix, so whole-model compression is
still absent.

### Odd-branch source localization

The [exact source helper](ODD_SOURCE_POSITIONS_V1_CPU_CONTROL.json) partitions
the reflection-odd branch by attention source and reconstructs O within1.47e-16.
The [native screen](ODD_SOURCE_POSITIONS_NATIVE_V1_RESULT.json) then removes
changed-city sources, all other sources, or all O on48 corrected rows. Native
and frozen all-O anchors replay exactly; live source recomposition is<=1.61e-16.

City-only localization fails decisively: its cue effect differs from all-O by
92.02% and96.46% in the two families. Yet all-O removal is material at13.10%
and15.86% of full-head cue norm, while city-source control/target RMS ratios are
.107/.141 and pass the registered selectivity criterion. O is therefore a
selective, material cue-dependent branch on this panel whose effect is carried
mainly through contextual source states, not a direct read at the changed city
token. This does not identify which contextual positions carry it or establish
a semantic label. The next split separates before-city, post-city and self
sources without fitting.

[The contextual-position screen](ODD_CONTEXTUAL_POSITIONS_NATIVE_V1_RESULT.json)
passes the exact instrument, after-city coverage, relay and selectivity gates;
the opposing final-self hypothesis fails. Post-city nonfinal sources reproduce
the all-O cue effect with8.73%/3.18% error. Adding self changes this only to
8.46%/3.78%. Self-only errors are99.92%/100.58%, and pre-city errors exceed100%.
After-city control/target RMS ratios are.124/.075. The native computation thus
relays the material selective O cue through later context-token states, rather
than consolidating it in the final token's self-source. These are source reads
inside head9.8; upstream descendants remain native. The next frozen semantic
split separates instruction/framing tokens from the copied quoted clause.

[The semantic-position screen](ODD_SEMANTIC_POSITIONS_NATIVE_V1_RESULT.json)
passes the framing-relay and selectivity predictions; the opposing clause-relay
prediction fails. Framing-only removal differs from all-O by30.68%/8.09%, while
the three copied-clause sources miss by78.65%/95.85%. Framing control/target RMS
ratios are.117/.072. Native, post, self and all-O arms replay the preceding
experiment exactly, and source recomposition is<=1.63e-16. The supported screen
description is: O reads regional information carried in the post-city prompt
framing before the quoted clause. The exact token or transformation within that
framing is not yet identified, and both templates were already known. A new
template/city panel is needed for held-out prediction.

The first semantic-mask CPU receipt is retained as a control-code failure: it
required bitwise-zero post addition despite the board's1e-10 bar. Its measured
error was<=5.47e-17. V2 applies the frozen numerical tolerance and passes; no
native prediction or boundary changed.

[The fresh panel](ODD_FRAMING_FRESH_V1_RESULT.json) uses two newly authored
templates, Cambridge/Phoenix and Leeds/Chicago, and six endpoints without model
selection. Native capability is12/12 in both templates; all-O remains material
at11.73%/14.59% of full-head cue norm. Framing transfer passes at29.99%/24.18%
error while copied-clause errors are83.89%/83.52%. Selectivity is not robust:
the radio template passes with a.298 work/jobs control ratio, but the local-history
template fails at.863. This is held-out evidence for the framing computation,
not identification. A frozen multi-control test will distinguish broad collateral
from semantic collision with the work/jobs control.

[That diagnostic](ODD_FRAMING_CONTROL_FAMILIES_V1_RESULT.json) passes all frozen
gates. Across cat/dog, red/blue, Monday/Tuesday and apple/orange, every ratio is
below.5: the local-history median is.226 (maximum.374) and the radio median is
.101 (maximum.191). Target and work/jobs arms replay bitwise. This makes broad
readout collateral less likely and identifies the old local-history work/jobs
failure as control-family sensitivity, while preserving that failure. It still
does not identify a unique semantic unit. The next split separates the
post-city descriptive continuation from the explicit reproduction instruction.

[The role split](ODD_FRAMING_ROLE_SPLIT_NATIVE_V2_RESULT.json) rejects both
uniform semantic-role hypotheses. Local-history is description dominated
(description error31.89%, instruction70.04%), while radio is instruction
dominated (71.89% versus28.72%). Separate effects nearly compose the framing
effect (1.86%/.39% error), and the supported arm in each template is selective
against the four new controls. V1 is preserved as an instrument failure: its
arm labeled all-O removed full S+R+O, causing only the all-O anchor to fail;
V2 changes that call alone and replays all anchors exactly. Because the role
with more/equally many source tokens wins in each template, semantic identity
is not stable. An equal-count early/late framing split is the next discriminator.

[The equal-count split](ODD_FRAMING_EQUAL_HALVES_NATIVE_V1_RESULT.json) also
rejects a stable positional role. Local-history remains early dominated
(31.89% versus70.04% error), while radio is distributed with late closer
(62.46%/38.21%) but outside the35% sufficiency bar. Both separate halves compose
within1.86%/.40%, all frozen anchors pass, and controls are small. Thus neither
semantic labels, source count, nor a universal early/late half identifies the
relay.

[Paired interchange](ODD_FRAMING_SOURCE_SWAP_V1_RESULT.json) is live, highly
aligned and selective: swap-versus-twice-removal cosine is.989/.999 and all
four control ratios are below.1. The radio magnitude passes at12.00% error;
local-history fails at56.08%. Thus the distributed framing state transports
the cue direction, but a uniform additive exchanged-write magnitude law fails.
Because swapping current source states changes both odd routing and current
values, the next exact expansion separates routing, value and their mixed term.

[That expansion](ODD_SOURCE_SWAP_INTERACTION_NATIVE_V2_RESULT.json) identifies
value transport. With every recipient query fixed, value-only swap differs from
full key+value swap by2.03%/2.04%; routing-only misses by103.11%/102.76%.
Omitting the explicit routing-times-value swap interaction changes the target
effect by only2.17%/2.29%, despite a live algebraic mixed term. All controls
remain small. V1's failed full-swap anchor compared against a legacy instrument
that also mutated nonfinal query rows; correction-only V2 preserves B-E and
reports the.01098 score gap. The next split separates current versus inherited
first values within the identified value transport.

[The value-source split](ODD_VALUE_SOURCE_SPLIT_NATIVE_V1_RESULT.json) has a
failed live-arm gate and an exact structural answer. Full-value and current-only
swaps are bitwise identical in scores; inherited-first swap is exactly zero.
Current/full cue error and separate-effect composition error are0; inherited
error is100%. `pred_a` fails because the frozen inherited-delta tripwire is zero.
This is expected from the native interface: block0 sets `v1` to its tokenwise
value projection before any contextual attention output, and the paired framing
tokens are identical. The existing blockzero audit independently reports token
identity variance fraction and embedding prediction R2 of1.0. Thus no city cue
can enter O through paired inherited framing values; it is carried by the
contextual block9 current-value input. The next intervention traces when that
current-value difference appears across upstream block boundaries.

[The equal-count split](ODD_FRAMING_EQUAL_HALVES_NATIVE_V1_RESULT.json) also
rejects a stable positional role. Local-history remains early dominated
(31.89% versus70.04% error), while radio is distributed with late closer
(62.46%/38.21%) but outside the35% sufficiency bar. Both separate halves compose
within1.86%/.40%, all frozen anchors pass, and controls are small. Thus neither
semantic labels, source count, nor a universal early/late half identifies the
relay. The next stronger test swaps complete framing-source states between each
British/American pair inside O and predicts signed counterfactual transport.
