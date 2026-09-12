# Shared cubic source features with private query computations

**Correction, 12 September 12:34 UTC:** the newest `REGIONAL_BEHAVIOR_CONTROLS_V1`
panel contains “A American journalist.” Its regional/control interpretation and
the subsequent token-routing comparisons are confounded by grammaticality.
Earlier corrected source-block V2/OOD, stream-split and full-vocabulary panels
are unaffected. The conditional compiler identities still hold on their actual
inputs, but do not establish clean semantic generalization. The article-only V2 repair and shared validator are complete; see the corrected
results at the end of this note. Original receipts remain as confounded history.


This factorization allows heads to reuse part of a computation even when their
complete functions are different. It searches for source features shared across
heads and solves each head's query-dependent use of them exactly. The dense
coefficient, execution and derivative controls pass; four independent planted
fits recover the shared feature. Native pilot results must be read from the
terminal receipt, not inferred from those controls.

The target is the complete attention17 QK1–QK2–value–OV numerator from the
[joint operator](JOINT_ROUTING_VALUE_POLYNOMIAL_V1_MATH.md). The previous
whole-function CCA failure does not test this representation: two heads can
read the same source feature with different query gates and have orthogonal
complete functions.

## Representation

The source $s$ has2304 coordinates, including current and block0-value inputs.
Each shared feature is one product of three linear readers:

$$
h_r(s)=(a_r^Ts)(b_r^Ts)(c_r^Ts),\qquad r=1,\ldots,R.
$$

Each head uses these features through a private quadratic query-to-output map:

$$
\widehat a_h(q,s)=\sum_r h_r(s)\,\beta_{hr}(q).
$$

Thus a source feature is a concrete nonlinear intermediate with potentially
multiple consumers. It is not a small linear projection through which every
query and source input must pass. All three source vectors are optimized on
unit spheres; permutations and compensating signs of the factors are equivalent
representations. A stable source function matters more than a factor label.
At rank16 this dictionary uses at most48 linear source directions, so it is
still a small-capacity source approximation. The query input is not restricted
to that same span. This limitation matters when interpreting low coverage.

## Exact elimination of the query-dependent writers

Let $H_r=\operatorname{Sym}(a_r\otimes b_r\otimes c_r)$ be the symmetric cubic
coefficient tensor. Its small Gram matrix is

$$
G_{rt}=\langle H_r,H_t\rangle
=\frac16\sum_{\pi\in S_3}\prod_{i=1}^3
\langle u_{ri},u_{t,\pi(i)}\rangle,
\qquad (u_{r1},u_{r2},u_{r3})=(a_r,b_r,c_r).
$$

Contract the exact joint attention tensor with $H_r$ on its source slots.
The result $C_{hr}(q)$ is a vector-valued quadratic in the query. Its factorized
expression has only six source permutations:

$$
C_{hr}(q)=\frac16\sum_{\pi\in S_3}
(q^TQ_{1h}^TK_{1h}u_{r,\pi(1)})
(q^TQ_{2h}^TK_{2h}u_{r,\pi(2)})
O_hV_hu_{r,\pi(3)}.
$$

Here key maps already include the fixed relative rotary transformation and
zero padding for the base-value source coordinates. Values include the native
signed mixture. Fixed metric head scales can be absorbed into $O_h$.

The least-squares-optimal private writers are

$$
\beta_{hr}(q)=\sum_t(G^{-1})_{rt}C_{ht}(q).
$$

There is no iterative fit of their enormous dense query/output tensors. Their
small factored expression and the16-by16 source Gram suffice in the pilot.
For symmetric query quadratics, the needed inner product is

$$
\langle\operatorname{Sym}(u\otimes v),
\operatorname{Sym}(x\otimes y)\rangle
=\frac{(u^Tx)(v^Ty)+(u^Ty)(v^Tx)}2.
$$

This gives $K_{h,rt}=\langle C_{hr},C_{ht}\rangle$ exactly, including the
physical output inner product. Captured coefficient energy is

$$
\mathcal C=\sum_h\operatorname{tr}(G^{-1}K_h).
$$

The code uses diagonally scaled Cholesky solves and differentiates this exact
objective through all source readers. It does not use sampled gradients. The
only nonlinear optimization is over the shared cubic features. Exact elimination
of linear coefficients is often called *variable projection*; it does not make
the remaining problem convex or guarantee global recovery.

## Metric and native scope

Heads retain separate output labels in the objective, because their actual
normalization gates differ. We do not permit a numerator cancellation to erase
a head whose differently normalized contribution would remain live. Each output
is weighted through its own native $O_h$ and a fixed weight-derived reference
gate

$$
g_h^{(0)}=\frac1{\|Q_{1h}\|_F\|K_{1h}\|_F
\|Q_{2h}\|_F\|K_{2h}\|_F}.
$$

This substitutes expected isotropic squared RMS values into the native divisor.
It fixes a weight-only geometry; it is not the actual input-dependent gate.
Native gates and residual/background dependencies remain required. This pilot
fits the joint attention numerator, not the complete nonlinear downstream
MLP17/unembedding objective. The exact downstream fold remains available for
later frozen-candidate validation.
Because the fitted physical writers already include $g_h^{(0)}$, native
evaluation must multiply each fitted head contribution by
$g_h(q,s)/g_h^{(0)}$, not by $g_h(q,s)$ a second time. Those private gates retain
their original projected query/key norm dependencies.

## Controls and pilot protocol

[Dense controls](SHARED_CUBIC_SOURCE_PROJECTION_V1_CONTROL.json) agree on source
Gram, query/output cross-Gram, captured energy, Pythagorean projection error,
all source-reader gradients and executable diagonal outputs, within1.13e-15.
The [optimizer control](SHARED_CUBIC_SOURCE_FIT_V1_CONTROL.json) uses one common
source cubic with different private query maps. Four independent starts reach
stationary points, squared errors below6.80e-10, and gradients below6.66e-7.
This is a planted control, not a trained-model convergence claim.

The [native preregistration](SHARED_CUBIC_SOURCE_NATIVE_V1_PREREGISTRATION.md)
sets16 cubic features and two starts: native key/value-aligned and fully random.
Each gets up to4000updates/300fit seconds with the existing safeguarded manifold
L-BFGS. Native finite differences and measured gradient price gate the fits.
Invalid Cholesky trials are rejected without adding an unregistered ridge.
The objective scale is each start's fixed initial captured energy, so a tiny
fraction of total tensor energy cannot automatically satisfy the stopping bar.

The pilot reports absolute coverage, held-position capture, cross-start source
function matches, literal multi-head use and cancellation between source terms.
Those are structural screens, not the four behavioral properties. A failed
rank16 pilot does not rule out a larger or differently structured dictionary.
Source readers alone cost110592coefficients; Gram256. Private query/value maps,
all native normalization dependencies and any remaining opaque computation must
also be priced. No whole-model saving is claimed by this representation yet.


## Native pilot and the optimizer scaling correction, 12 September

The [native pilot](SHARED_CUBIC_SOURCE_NATIVE_V1_RESULT.json) ran601.56seconds.
Derivative and price checks passed; both fits reached their time limits, so
convergence and sharing predictions failed. The aligned/random arms captured
0.2431%/0.3415% of estimated reference coefficient energy at the fitted position,
and0.1053%/0.1150% at the held position. These percentages use1024 independent
coefficient probes per position; reference energy52.9773±0.8140 standard error
at source7 and50.5641±0.8451 at source0. No text validation is implied.
Only one cubic source feature matched across starts at coefficient cosine>=.9.
Neither arm had an atom meeting the literal multi-head-use bar. Cancellation
ratios1.8003/1.5924 and Gram condition numbers80.60/14.45 were moderate.

The [history audit](SHARED_CUBIC_SOURCE_NATIVE_V1_SCALE_AUDIT.json) found a
specific optimization problem: the old safeguard reset curvature history on
75.49%/99.73% of updates. With gradient $g$ and proposed direction $d$, it used

$$
g^T d < -10^{-3}\|g\|^2.
$$

For an inverse-Hessian-scaled direction, multiplying the objective changes the
right-hand side quadratically while the left typically changes only linearly.
Thus this check can reject a useful descent direction solely because of units.
The V2 safeguard instead requires

$$
g^T d < -10^{-3}\|g\|\,\|d\|.
$$

This bounds the direction's angle with the negative gradient. The
[controlled quadratic comparison](LBFGS_DESCENT_SCALE_V1_CONTROL.json) changes
only this check: at objective scale1000, V1 resets255times and misses its256-update
limit; V2 reaches stationarity in22 history rows with no resets. At scale1 the
solvers agree. This establishes a bug in the safeguard, not global convergence
of the complete optimizer, whose other numerical thresholds remain finite.
The [V2 cubic planted control](SHARED_CUBIC_SOURCE_FIT_V2_CONTROL.json) also
recovers the shared source in all four starts, errors<=6.8e-10.

The [registered continuation](SHARED_CUBIC_SOURCE_CONTINUE_V1_PREREGISTRATION.md)
warms both saved native dictionaries, changes that safeguard, and uses the common
fixed divisor $c=0.1809357634075061$. The loss remains

$$
\mathcal L=-\operatorname{capture}/c.
$$

Its minima are unchanged, but its numerical stopping units are different.
The old final gradients become4.33e-5/9.67e-5 in these common units, both still
above the new1e-6 bar. Original convergence misses are preserved. This continuation
combines a safeguard repair and a common-unit choice; their separate native
contributions are not isolated. It tests whether properly scaled local fitting
resolves the current convergence obstacle. Multi-head use and cross-start
agreement remain independent requirements. No circuit has been promoted.

A separate [QR control](SHARED_CUBIC_SOURCE_QR_V1_CONTROL.json) implements direct
coefficient QR in the span of the source readers, avoiding source normal equations.
Ordinary value/gradient agreement is within1.67e-15. At a planted Gram condition
near2e10, relative capture error drops from7.28e-7 to3.20e-11 against direct SVD.
This is useful for nearly coincident atoms, but current native conditions do not
justify the more expensive evaluator. The continuation retains the original exact
Gram objective. Detached local spans are justified for first derivatives of this
squared norm; no higher-derivative claim is made.


## Continuation outcome and source-space audit

The [continuation](SHARED_CUBIC_SOURCE_CONTINUE_V1_RESULT.json) completed in93.09seconds:
A passed, B/C failed. Arm0 reached the new stationary bar in1015 history rows;
capture increased by0.000971%. Arm1 failed its line search after671 rows with
capture gain0.001048%, Gram condition320512 and cancellation ratio1831.
There were two cross-start atom matches and no literal multi-head atoms.
This repairs one convergence obstacle without producing materially better structure.

The [native CPU QR audit](SHARED_CUBIC_SOURCE_NATIVE_QR_V1_AUDIT.json) at arm1's
terminal point compares the exact same objective using the independent coefficient
QR evaluator: capture differs by2.97e-13 absolute and tangent gradients by4.79e-6
relative. QR costs2.44seconds versus0.52seconds for Gram plus gradient on CPU.
Directional finite differences have large truncation error at step1e-3, falling
to0.15% for QR at1e-5; they support the same sharply curved local behavior.
These findings do not support blaming the line-search failure primarily on
normal-equation roundoff. GPU and CPU capture differ by roughly1e-12, so numerical
noise can still matter for extremely small improvements.
The [coalescence audit](SHARED_CUBIC_SOURCE_CONTINUE_V1_COALESCENCE.json) identifies
arm1 atoms5/6 at absolute coefficient cosine0.999970986. Near-coincident products
with opposing writers can represent a finite difference or tangent direction.
This motivates testing a block/difference representation; it does not prove
border-rank degeneracy or nonexistence of a finite optimum.

The atom-use count also depends on coordinates. For source Gram $G$ and per-head
cross coefficient Gram $K_h$, define the source energy operator in orthonormal
source coordinates and its overlap as

$$
M_h=G^{-1/2}K_hG^{-1/2},\qquad
\omega_{hk}=\frac{\operatorname{tr}(M_hM_k)}
{\|M_h\|_F\|M_k\|_F}.
$$

For independent source functions these operators transform by a common orthogonal
change under any invertible dictionary re-encoding, so the overlap is invariant.
Their trace is the captured coefficient energy per head. Linear combinations of
cubic products need not themselves be single products: this is a function-span
diagnostic, not an equivalence preserving the original product-atom price.

The [planted control](SOURCE_ENERGY_OVERLAP_V1_CONTROL.json) rotates two disjoint
source functions: overlap remains zero, while literal shared atoms change from
zero to two. [Original native dictionaries](SOURCE_ENERGY_OVERLAP_SHARED_CUBIC_SOURCE_NATIVE_V1_RESULT.json)
and [continued dictionaries](SOURCE_ENERGY_OVERLAP_SHARED_CUBIC_SOURCE_CONTINUE_V1_RESULT.json)
are audited without fitting. Large overlap alone is misleading here: original
arm0 heads4/5 have overlap0.901, but captured energies0.07619 and0.00001149.
Thus a tiny projection onto a dominant head's source direction can appear highly
aligned. The original no-sharing prediction remains failed; no causal or global
source-reuse claim follows from this diagnostic. Any later grouping test must
report each participating head's absolute captured fraction, not just normalized
similarity in a selected dictionary.


## An explicit difference block replaces the canceling pair

The literature supplies a reason to take coalescence seriously: ordinary tensor
rank-constrained approximation can fail to attain a best solution
([de Silva and Lim, 2008](https://arxiv.org/abs/math/0607647)). That theorem concerns
general tensor rank, not a proof about this separately symmetric cubic dictionary
with variable-projected query writers. We use it as motivation for an explicit
algebraic test, and do not claim this trained example is an unattained optimum.

Align the three readers of the two products by permutation and signs. Write them
as $a_i+t d_i$ and $a_i-t d_i$, with $t$ the joint norm of their half difference.
Let $a_i$ and $d_i$ below abbreviate their scalar readings on the same source.
The pair spans the following two functions for nonzero $t$:

$$
E=a_1a_2a_3+t^2(d_1d_2a_3+d_1a_2d_3+a_1d_2d_3),
$$

$$
O=d_1a_2a_3+a_1d_2a_3+a_1a_2d_3+t^2d_1d_2d_3.
$$

They are exactly the average of the two products and their difference divided by
$2t$. Executing these expressions avoids subtracting nearly equal products.
At $t=0$ the same formulas define a tangent block; this limit is a distinct
source space, not an exact rewriting at finite separation.

The [CPU control](CUBIC_SECANT_BLOCK_V1_CONTROL.json) checks the polynomial
identity down to separations4.42e-5, relative diagonal discrepancy<=3.25e-12.
The [native audit](CUBIC_SECANT_BLOCK_NATIVE_V1_RESULT.json) finds $t=0.00390310$:
exact-block capture agrees with the original within1.65e-12 relative, while
source Gram condition falls from320512 to27.821 and individual-energy ratio
from1831 to1.863. These are improvements in the coordinates used to execute
one unchanged projected computation, not increased coverage or new behavior.
The tangent limit changes capture by less than5e-11 relative in the fit position.
Equality of capture alone does not establish equality of the two projected
functions; a native replacement still requires direct function-error validation.

A stronger simplification is suggested by aligned reader cosines
0.999999989,0.999999999,0.999969544: two readers are almost shared. The
[common-quadratic test](COMMON_QUADRATIC_SOURCE_BLOCK_V1_RESULT.json) replaces
the pair by

$$
z(s)=(a_1^Ts)(a_2^Ts),\qquad
h_1(s)=z(s)(u^Ts),\quad h_2(s)=z(s)(v^Ts).
$$

This is a concrete two-consumer arithmetic graph. Its shared intermediate $z$
is quadratic, and each child multiplies it by another linear reading. It needs
four independently stored source readers instead of six. Across the complete
16-feature candidate, source reader count falls48 to46, before counting private
query writers and all native dependencies. The prototype artifact materializes
duplicate shared readers for comparison; a compiled representation must store
and execute $z$ once to realize that price.

Without optimizing these readers, the collapsed block retains99.9861% of the
pair's marginal captured energy at source7 and99.9908% at source0; marginal
means the increase over refitting the other14 features alone. Total capture
falls0.001633%/0.001538%, and Gram condition remains27.821. The block's individual
energy is overwhelmingly assigned to head17.2 (0.02128 at source7), so this is
within-head shared computation, not evidence of reuse across heads or tasks.
The nearly shared readers warrant a direct projection-function comparison and
local block fit. They do not establish any of the four behavioral properties.

The [direct coefficient-function comparison](CUBIC_SOURCE_BLOCK_FUNCTION_COMPARISON_V1.json) closes the scalar-capture gap: tangent-limit squared relative error is around1e-14 (roundoff-sensitive), while the common-quadratic function has0.4708%/0.5827%relative norm error at source7/0. This is relative to the entire fitted projection, not a relative-error guarantee for its much smaller block alone. Native gates and text behavior remain untested.


## Frozen native FineWeb extraction/removal screen

The [registered native screen](COMMON_QUADRATIC_NATIVE_V1_PREREGISTRATION.md)
[passed A/B/C](COMMON_QUADRATIC_NATIVE_V1_RESULT.json), using64cached FineWeb
9-token prefixes and8partial-body forwards in1.36seconds. Actual full attention
replay error was1.78e-7. This includes both normalized QK factors, rounded rotary,
the signed first-layer value mixture and native source/query inputs.

The simplified versus stable-secant full projected writes differ by0.0731% at
source0 and0.0442% at source7. Their two-child block writes differ by0.4197% and
0.3026%. After deleting the parent (both children at both source positions), the
full50304-logit effect differs by0.3683%, with original effect norm5.2068 across
all rows/logits. The complete native MLP17, its input RMS, the final RMS and tanh
cap were recomputed. The two individual child effects are nearly additive here,
with0.1465%relative nonadditivity, but joint prediction does not assume additivity.

The [per-prefix audit](COMMON_QUADRATIC_NATIVE_V1_PREFIX_AUDIT.json) finds median
summed-block write error0.2891%,90th percentile0.6648%, maximum0.7759%; none of
64prefixes exceeds1%. Child norm cancellation ratios range1.019–1.468. This
checks aggregate masking for writes; per-prefix suffix-effect errors have not
been computed. The preserved block is around3.96%of the norm of the selected
native source writes, which is a descriptive norm ratio, not additive energy
coverage or a claim that the entire attention operation has been extracted.

This establishes a native preservation screen for one frozen fitted component.
It does not establish semantic selectivity, cross-task reuse, general OOD
prediction or independent whole-circuit sufficiency. Upstream inputs, private
query writers, normalizers and native background remain explicit dependencies.
The actual implementation still uses dense native private query factors.

[Post-hoc token inspection](COMMON_QUADRATIC_NATIVE_V1_DESCRIPTIVE_TOKENS.json)
finds many largest shifted logits among regional spellings (realise, realised,
colour) and geographic terms (Australia/NZ, Philippine/Filipino, Norwegian).
Largest displayed changes reach0.0856nats. These are not necessarily likely
continuations and can reflect downstream output geometry. They are hypotheses
for a future independently registered behavior screen, not circuit labels.
Prior dossier checks show head17.2 already participates in exclamation and
capitalization computations. The new block must not inherit either semantic
label just because it lies mostly in the same head.


## Prospective regional-spelling screen: small signed contribution, bar missed

The [new regional-spelling protocol](REGIONAL_SOURCE_BLOCK_V1_PREREGISTRATION.md)
uses British/American cue pairs for eight spelling contrasts in two templates.
It is a prospective behavioral test of frozen weights, motivated by the preceding
post-hoc vocabulary inspection. No word, reader or threshold was fitted to scores.
[The result](REGIONAL_SOURCE_BLOCK_V1_RESULT.json) passes instrument/capability A,
but fails B and C. Native cue shifts average1.836/2.578nats and are positive for
all8contrasts per family. Removing the parent at source0 and the immediately
previous position reduces those shifts by0.0496/0.0564nats, positive in all16cases.
That is2.70%/2.19%of the native cue effect, below the registered10%bar.

The norm-matched paired-context direction control has negative mean reductions
(-0.0091/-0.0220nats). Actual-minus-control improvements are about3%of native,
below the registered5%bar. Unrelated token-contrast differential effects average
0.0017/0.0018nats, much smaller than the regional changes. This is evidence of a
small consistently signed contribution on these rows, not a passed selective
circuit screen. The whole model's spelling capability is not attributed to this
small projected component.

After execution, a grammar defect was found in family1: the American cue produced
“In a American newspaper.” Family0has no such article defect. The original result
and misses remain recorded. [V2](REGIONAL_SOURCE_BLOCK_V2_PREREGISTRATION.md) changes
that article to “the” and separately measures the original two-source program
and a broader all-causal-source program. All-source evaluation charges additional
position-specific maps/gates and must not be described as the same tested program.
All thresholds remain unchanged. Extending across source positions changes only
where the fixed weight-derived source computation is evaluated, not its readers.


### All-source block passes the corrected basic semantic screen

[V2 results](REGIONAL_SOURCE_BLOCK_V2_RESULT.json) preserve the two-source miss:
A passes, B/C fail after correcting the article. The separately declared all-source
program passes A/B/C on both templates. Native cue effects are1.836/2.732nats;
parent removal reduces them by0.2406/0.4965nats, or13.11%/18.17%. The paired,
norm-matched direction control gives0.0105/-0.4296nats reduction. Unrelated
contrast differential changes average0.00893/0.01690nats, about3–4%of the
regional changes. All-source collapsed/old-block write error is0.1221%.
The run uses7bodyforwards and completes in3.62seconds, with no reader fitting.

The [per-contrast audit](REGIONAL_SOURCE_BLOCK_V2_PAIR_AUDIT.json) retains every
case: all16native cue effects and all16removal reductions are positive; individual
reductions span9.40–27.05%of the native cue effect, with15/16above10%. Thus the
family-mean success is not produced by a few positive cases hiding opposite signs.

This is a passed prospective basic screen for a weight-discovered shared
quadratic component contributing to regional spelling. It is not a four-property
circuit promotion. The component explains a minority of the cue effect; native
inputs, private query writers, gates and downstream background remain required.
The two-source miss was a real coverage limitation for that smaller program.
All-source replication reuses source readers but adds position-specific edges.
Matched-context direction controls and eight unrelated contrasts are useful but
not exhaustive semantic or downstream-output-geometry controls.

The next frozen test changes both the output spelling pairs and the cues:
[48new rows](REGIONAL_SOURCE_BLOCK_OOD_V1_ROWS.json) use London/Boston and
Edinburgh/Chicago contexts, six previously unused spelling pairs and two new
sentence templates. These are out-of-template/lexicon tests, not a new natural
corpus or an OOD guarantee. No scores were inspected during their construction.


### Unseen spellings and city cues also pass the basic screen

The [frozen OOD protocol](REGIONAL_SOURCE_BLOCK_OOD_V1_PREREGISTRATION.md)
[passes A/B/C in all four families](REGIONAL_SOURCE_BLOCK_OOD_V1_RESULT.json).
Six previously unused spelling pairs and London/Boston or Edinburgh/Chicago cues
replace the original words and explicit regional adjectives. Native cue gaps
average2.335,3.226,1.652,2.982nats; all-source block removal reduces them by
0.310,0.447,0.168,0.364nats, or13.28%,13.86%,10.18%,12.21%. All24paired contrasts
have positive native gaps and positive removal reductions. The norm-matched
paired-direction controls give negative mean reductions in every family;
unrelated contrast changes are about11%of the regional changes. Collapsed/old
block write error is0.206%. Execution takes5.53seconds and10bodyforwards.

This supports transfer across these unseen lexical contrasts, cue formulations
and templates, with no factor fitting. It does not prove natural-corpus OOD or
exhaustive semantic selectivity. Individual removal fractions span roughly5.6–20.3%;
the registered bars apply to each family mean, not every individual pair.
See the [full per-pair audit](REGIONAL_SOURCE_BLOCK_OOD_V1_PAIR_AUDIT.json).

The [11:00 mathematical review](THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-12_1100.md)
and [next preregistration](REGIONAL_FACTOR_INTERCHANGE_V1_PREREGISTRATION.md)
separate the block into actual gate, quadratic parent, child-linear bundle and
private query writers. Exact controlled extraction and16vertex intervention
algebra now permit a test of which port carries the cue. A useful shared operation
need not itself be the semantic label; preserve that distinction during naming.


## Factor interchange identifies the child readings as the cue-carrying port

The [native four-port experiment](REGIONAL_FACTOR_INTERCHANGE_V1_RESULT.json)
passes A/B and fails C, the registered quadratic-parent dominance prediction.
Recipient and complete-donor write replay agree with the saved implementation
within1.20e-15; all16finite intervention vertices reconstruct exactly at reported
precision. Full-block paired transplant transfers0.2476/0.5060nats of regional
cue effect. Parent-only swaps transfer0.0126/0.0113nats, just5.09%/2.24%of that.
Swapping the child-linear tuple transfers0.2583/0.5318nats, or104.35%/105.10%.
Gate-only and query-writer-only transfers are small and negative on these rows.
Values above100%are possible because other ports and interactions partly oppose
this transfer. Do not sum single-port effects as an exact decomposition.

The [registered held-out role test](REGIONAL_FACTOR_INTERCHANGE_OOD_V1_PREREGISTRATION.md)
then predicts child-tuple dominance on the geographic/new-spelling rows, without
fitting. [All four families pass A/B/C](REGIONAL_FACTOR_INTERCHANGE_OOD_V1_RESULT.json).
Child-only transfers are0.3213,0.4656,0.1812,0.4000nats versus complete-block
0.3201,0.4592,0.1766,0.3785nats. Thus the two-dimensional linear reading carries
most of this paired cue difference. The shared quadratic remains a reusable
modulating computation, not itself the dominant regional cue representation.
This distinction is a result of the intervention test, not a semantic name
assigned merely from factor shape or top output tokens.

[Weight stream accounting](REGIONAL_FACTOR_INTERCHANGE_V1_ROLE_AUDIT.json) finds
first-layer-stream squared reader-norm fractions around0.097%/0.109%for the two
children, even smaller for the parent readers. This suggests current residual
inputs dominate weight geometry, but does not justify dropping the first stream
without functional validation. The upstream producer of the regional reading
remains unresolved; these are conditional port interventions, not input erasure.

## The shared graph is stable across the two independent weight fits

Earlier literal cubic-atom matching failed. A
[weight-only graph audit](REGIONAL_SOURCE_CROSS_START_GRAPH_V1_AUDIT.json) instead
matches the common quadratic parent to all atoms in the other independently
initialized source fit, without semantic data. Exactly two leading matches,
atoms11and2, have both parent-reader absolute cosines above0.99966. Their two
cubic source functions span a subspace with principal cosines0.999894/0.999670
against the current two-child block. This does not amend the original atom-level
prediction; it identifies a more stable grouped unit.

The [complete coefficient-function comparison](REGIONAL_SOURCE_CROSS_START_FUNCTION_V1_AUDIT.json)
keeps each fit's own full-dictionary private query writers. Block function cosines
are0.999640/0.999623at source7/0; relative norm errors2.685%/2.747%. Thus the
agreement is not only in input readers. This is coefficient-space stability with
fixed reference gates, not yet a native behavioral correspondence test between
fits. The next comparison should replay both independently derived blocks on the
same held-out prefixes and compare their signed intervention effects.


### Independent fits agree on native held-out interventions

The [native cross-start test](REGIONAL_CROSS_START_NATIVE_V1_RESULT.json) passes
A/B/C in all four geographic/spelling families. Independent blocks differ by
1.067%in native write norm. Signed individual-prefix regional removal effects
differ by0.658–0.970%; paired cue reductions differ by0.395–0.463%. The independent
fit retains the registered>=10%native cue contribution in every family and all
24pair reductions are positive. Its own private query writers were used; no
semantic refit or shared decoder was introduced. Runtime4.97seconds,10forwards.

This establishes stable native behavior for this grouped component across the
two independent starts, despite earlier atom-by-atom matching misses. It does
not close broad unrelated-behavior preservation, whole-model OOD, upstream
independent extraction or total program pricing. The regional function remains
a minority contribution implemented within retained native context.

## Folding the cue-carrying readings through MLP16

The previous parent1 MLP16 study already supplies the algebra, but for different
readers. [The new regional fold](REGIONAL_PAYLOAD_MLP16_FOLD_V1_RESULT.json) compares
the two frozen current-stream child readers with that older three-reader span:
principal cosines0.0450/0.0321, so this is a distinct component target. No new
theorem or generic folding technique is claimed.

Let $C\in\mathbb R^{2\times1152}$ contain the current-stream child readers and
$C_0$ their first-attention-stream parts. Let $p_{16}$ be the residual before
MLP16's addition and $x_{16}$ its normalized input. Then

$$
r_{17}=\lambda_0\{p_{16}+D_{16}[(L_{16}x_{16})\odot(R_{16}x_{16})]+b_{16}\}
+\lambda_1x_0.
$$

The payload is $u=C r_{17}/\rho_{17}+C_0x_{\mathrm{first}}$. Folding the two
readers into the previous Down matrix gives

$$
A=\lambda_0CD_{16}\in\mathbb R^{2\times4608},\qquad d=\lambda_0Cb_{16},
$$

$$
u=\frac{\lambda_0Cp_{16}+A[(L_{16}x_{16})\odot(R_{16}x_{16})]+d+
\lambda_1Cx_0}{\rho_{17}}+C_0x_{\mathrm{first}}.
$$

The executed CPU native-weight identity agrees within2.71e-15, preserving bias,
actual RMS epsilon, both source streams and residual background. The artifact
stores only the two folded Down rows, bias, readers and residual coefficients;
native Left/Right and other dependencies remain required. No native producer
mediation has yet been measured. In particular, a small first-stream reader norm
or a large MLP output norm cannot substitute for a cue-transfer test.

The next test should separate MLP16 numerator, preceding residual numerator,
residual re-entry numerator, shared RMS divisor and first-stream contribution.
Keeping the divisor separate prevents attributing normalization changes to a
particular producer. This traces the identified source reading upstream while
retaining the exact composed regional block and final native readout.


## Upstream producer accounting: cue transfer is distributed

The child tuple is linear in the normalized attention17 input. Write its two
current-stream readers as the rows of $C$. The native residual recurrence admits
an exact numerator expansion:

$$
r_{17}=e x_0+\sum_{j=0}^{16}s_j(A_j+M_j),
\qquad s_j=\prod_{k=j+1}^{17}\lambda_{k,0}.
$$

Here $A_j,M_j$ are actual attention and MLP writes, and $e$ collects all learned
embedding re-entries. Thus the two child values are

$$
u=\frac{eCx_0+\sum_j s_j CA_j+\sum_j s_j CM_j}{\rho_{17}}
+C_{\mathrm{first}}x_{\mathrm{first}}.
$$

This exposes35 numerator terms plus a shared RMS divisor and the first-stream
term. Swapping a producer numerator while holding the recipient divisor and
remaining downstream inputs fixed measures that edge's conditional mediation.
It does not measure the result of removing that entire module throughout the
network. The [residual-unroll control](RESIDUAL_PAYLOAD_UNROLL_V1_CONTROL.json)
agrees to2.09e-16 on synthetic writes using the actual learned coefficients.

The [MLP16 producer test](REGIONAL_PAYLOAD_PRODUCER_V1_RESULT.json) passes replay
but fails the registered dominance prediction. Its signed transfers are+0.00397
and−0.00992nats, versus0.25834/0.53183 for the whole child tuple. The pre-MLP16
residual instead transfers0.26277/0.54792. This executed partition test rules out
spending another decomposition fit on MLP16 solely because its fold is simple.
It does not imply that MLP16 is globally unimportant.

The [all-producer atlas](REGIONAL_PAYLOAD_ATLAS_V1_RESULT.json) also passes replay
and fails embedding dominance:0.01932/0.01385nats, despite its coefficient
$e=145.12$. A large recurrence coefficient alone does not locate semantic signal.
Attention8,9,13 are the strongest group across the discovery templates. Their
[CPU joint audit](REGIONAL_PRODUCER_GROUP_V1_AUDIT.json), recomputing the native
nonlinear suffix after summing write deltas, transfers0.11118/0.29729nats:43.04%
and55.90%of whole-child transfer. All16 pairs are positive; unrelated contrast
mean absolute transfers are0.00415/0.01054nats. Baseline replay error is1.90e-6.

This group was selected using these outcomes. Its prospective producer-role test
on the four previously defined geographic/spelling families is registered in
[the held-out protocol](REGIONAL_PAYLOAD_ATLAS_OOD_V1_PREREGISTRATION.md).
The earlier embedding and MLP16 failures remain failures. The group may support
cross-module extraction, but still requires native intermediate states and does
not yet provide an independent circuit or broad selective-removal guarantee.


### Frozen producer group generalizes; exact upstream value fold executed

The [held-out producer result](REGIONAL_PAYLOAD_ATLAS_OOD_V1_RESULT.json) passes
all three registered criteria in every family. The fixed attention8/9/13 group
transfers52.06%,52.69%,70.69%,65.91%of the whole-child effect. All24 pairs move
in the predicted direction; unrelated contrast mean absolute effects are
10.9–11.4%of regional effects. Native execution took4.02seconds. This supports
the selected group across these lexical/cue/template shifts; it does not prove
that these modules are independently sufficient or that their individual heads
are regional specialists.

The [next CPU computation](REGIONAL_PRODUCER_OV_FOLD_V1_RESULT.json) folds the two
current-stream readers into each selected attention layer. For head $h$ in layer
$j$, define the two-output maps

$$
F_{jh}=s_j C O_{jh},\qquad
E_{jh}=(1-\mu_j)F_{jh}V_{jh},\qquad
H_{jh}=\mu_jF_{jh}V_{0h}.
$$

The contribution to the child numerator at recipient position $t$ is exactly

$$
s_j C A_{j,t}=\sum_{h=1}^{9}\sum_{s\leq t}
\gamma_{j,h,ts}\left(E_{jh}\bar r_{j,s}
+H_{jh}\bar r_{0,s}\right).
$$

$\gamma$ retains the product of BOTH native QK scores, their normalization and
positions. The first-layer value stream is explicit; signed mixing is retained.
This computes the two downstream-relevant numbers directly from each producer's
input streams, without reconstructing its1152-dimensional output. Synthetic
arbitrary-input/routing identities agree to2.91–3.82e-15. Native routing replay
of this newly folded executor remains untested.

The artifact stores131,382 tensor entries including diagnostic singular values.
The two fused current/base reader maps alone use124,416 values across three
layers. This is not the total extracted-program cost: unchanged QK weights alone
cost15,925,248 entries, and producer inputs still require the upstream model.
Thus the fold is progress toward an explicit interface, not a standalone small
circuit claim.

Descriptive cross-layer principal cosines of the fused value-reader row spaces
have maxima0.530,0.594,0.281 for8/9,8/13,9/13. These are different layer input
boundaries, so the comparisons measure coefficient geometry in residual
coordinates, not equality of the realized functions. They do not show an exactly
shared value-reader subspace or rule out shared upstream computation. The next
useful test is native replay of these folded producers and paired routing-versus-
value interchange with both QK factors kept together. It should distinguish
shared cue values from task-dependent routing before attempting a new factor fit.


### Upstream values carry the cue; routing remains an explicit dependency

The [original-template routing/value test](REGIONAL_PRODUCER_ROUTE_VALUE_V1_RESULT.json)
passes native producer replay (maximum1.97e-7relative) and the saved three-layer
joint swap replay (2.34e-7). This validates the folded OV executor on native
inputs, beyond the earlier synthetic identity. Each producer is evaluated as

$$
N(\Gamma,Z)_t=\sum_{h,s}\Gamma_{h,ts}Z_{h,s},
\qquad Z_{h,s}=E_h\bar r_s+H_h\bar r_{0,s}.
$$

The experiment evaluates recipient/recipient, donor-routing/recipient-values,
recipient-routing/donor-values, and donor/donor. Routing includes both normalized,
position-corrected QK factors together. The projected current and first-stream
values travel together. These are upstream producer ports; downstream attention17
routing, parent, writers and recipient normalization stay fixed. The native last
MLP, final normalization and capped unembedding are recomputed after each edit.

On the original two templates, value-only transfers are0.11346/0.28650nats,
versus0.11118/0.29729for the full group:102.05%/96.37%. Routing-only transfers
are−0.00137/+0.00690. The value-dominance criterion passes. The
[prospective held-out role test](REGIONAL_PRODUCER_ROUTE_VALUE_OOD_V1_RESULT.json)
then passes all four families: value/full ratios102.01%,97.68%,103.68%,99.56%.
This is role generalization on previously defined held-out spellings and location
cues, not discovery on untouched text or full-vocabulary preservation.

The executed [original-prefix audit](REGIONAL_ROUTE_VALUE_ROLE_AUDIT_ORIGINAL.json)
and [held-out-prefix audit](REGIONAL_ROUTE_VALUE_ROLE_AUDIT_OOD.json) check that
small average effects are not hiding large opposed effects. Across the24held-out
pairs, every value-only effect is positive and its ratio to the full group lies
between96.20%and105.40%. At individual-prefix level, value-only effect errors
are2.79–7.25%in relative Euclidean norm; routing effects are2.74–6.94%of full
norm. The finite interaction (full minus routing-only minus value-only) reaches
13.44%of full norm. Interaction includes the final nonlinear suffix response;
it is not solely the bilinear numerator cross term.

The resulting specification is sharper: regional cue changes are transported
mainly by value readings in this fixed producer group, while its recipient routing
provides the context in which those values are used. This does not license
replacing routing with a constant or dropping either QK factor. The next unresolved
input distinction is current-layer value state versus the shared first-layer
value stream; those were bundled here. Closing that distinction can determine
which upstream computation needs to be folded next. Broad selective removal,
independent extraction and reuse across different behaviors remain unfinished.


### A token-only branch and a contextual branch both contribute

The [current/first stream test](REGIONAL_PRODUCER_VALUE_STREAM_V1_RESULT.json)
passes execution and the both-value reference, but fails the registered
current-stream dominance prediction. Current values explain58.7%/45.0%of the
value-mediated effect; first-layer values explain41.3%/55.0%. Both streams move
all16pairs correctly. There is no optimization involved in this negative result;
the exact stream split and successful replay show a real second contribution.

The first-value stream's token-only nature is already established in the
[channels dossier](../bilinear_quotient/modules/channels.md). We reused that fact
for these specific54producer readings. For token $v$, its embedding $e_v$ gives

$$
x_0(v)=\operatorname{RMS}(e_v),\qquad
x_{\mathrm{first}}(v)=\operatorname{RMS}
\left(\lambda_{0,0}x_0(v)+\lambda_{0,1}x_0(v)\right),
\qquad z^{\mathrm{first}}_{jha}(v)=(H_{jh})_{a:}x_{\mathrm{first}}(v).
$$

Keep native normalization epsilon and signed coefficients. In particular the
selected producer mixing coefficients are4.0,−0.65625,4.1875; they are not
probability weights. The [token compiler](REGIONAL_FIRST_TOKEN_V1_RESULT.json)
produces54readings pertoken, with1.74–2.47e-7relative error against the unfolded
native first-value calculation. Its48-token panel table stores2592scalars.
All40original/held-out pairs change exactly one cue-token position, so token-only
readings at every unchanged position have exactly zero donor difference.

The [held-out stream test](REGIONAL_PRODUCER_VALUE_STREAM_OOD_V1_RESULT.json)
actually uses this compiled token table for the first branch. All4families pass:
current shares51.7–56.3%, first shares43.8–48.3%; bothstreams have24/24positive
pairs and pass unrelated-contrast bars. This new two-source prediction does not
replace the original failed current-dominance prediction. The table covers these
48IDs; arbitrary vocabulary support needs the original embedding table plus the
fixed normalization and reader maps. No text fitting produced these readings.

The [composition audit](REGIONAL_VALUE_COMPOSITION_V1_RESULT.json) checks

$$
\Delta w_{\mathrm{both}}=
\Delta w_{\mathrm{current}}+\Delta w_{\mathrm{first}}.
$$

Relative write errors are2.17e-16original and2.48e-16held-out. Final nonlinear
contrast interactions have relative norms0.10–0.20%on held-out individual
prefixes, and0.021–0.026%on paired cue effects. These measured small interactions
justify near-additive effect prediction on this interface; they are not a claim
of globally linear logits or independent behavior under arbitrary edits.

This closes a concrete source branch: its values can be computed directly from
cue-token identity and reused at three producer layers. Native routing, current
values, other residual contributions and downstream normalization still provide
context. Independent end-to-end extraction, broad removal selectivity and reuse
across different behaviors remain open. Next folding should target the current
branch or the routing dependency, not re-establish the known token-only channel.

Operational note: disk exhaustion interrupted audit-file creation before that
script ran. Removing only the regenerated npm download cache recovered~100MB;
model, installed packages and research artifacts were preserved. The audit then
ran successfully. [Storage receipt](NPM_CACHE_STORAGE_2026-09-12_1154.json).


### Full-vocabulary selectivity of the token branch on these endpoints

The [full-vocabulary audit](REGIONAL_FIRST_FULL_VOCAB_V1_RESULT.json) recomputes
all50304capped logits from the saved baseline and first-value-swap endpoints.
This adds a stronger output measurement without fitting or new model-body runs.
Replay of saved target/control contrasts is within1.64e-6relative. It does not
introduce a new held-out text panel: these are the already tested80prefixes.

For each row, remove its two regional target tokens and renormalize the remaining
probabilities to obtain $p_{\neg T},q_{\neg T}$. The primary spillover metric is

$$
E_{\neg T}=\sqrt{\mathbb E_{\text{rows}}
\sum_{v\notin T}p_{\neg T}(v)
\left[\log q_{\neg T}(v)-\log p_{\neg T}(v)\right]^2}.
$$

Compare it with the RMS change in target-pair log odds. All6families pass the
registered25%ratio bar: observed ratios5.55–8.28%. Individual-prefix ratios are
at most17.68%. Off-pair conditional total-variation means are0.00072–0.00288,
below the0.005bar, and conditionalKL means3.69e-6–5.31e-5nats. These are
probability-weighted measures; they do not guarantee tiny effects on every
rare token. Target-pair total logmass changes have RMS0.0090–0.0375nats and are
reported separately rather than hidden by conditional renormalization.

On the held-out panel, full-distribution TV means (including the intended pair)
are0.00259–0.00560. This is a different metric from the registered off-pair TV;
do not describe the entire distribution as unchanged. The measured branch also
accounts for only part of the overall regional computation, so small absolute
spillover alone would be weak evidence without the relative target-effect bar.

The [descriptive token profile](REGIONAL_FIRST_OFFPAIR_PROFILE_V1.json) completes
the registered follow-up. Large raw-logit changes include regional spellings
such as “realise”, “colour”, “recognise”, “centre”, and geographic/currency tokens.
Probability-weighted changes also include quotation marks, “reporter”, “handling”,
and other words. Rankings are descriptive, not semantic annotations or proof of
cross-task reuse. Other regional tokens outside a row's target pair are counted
as off-pair; off-pair is not synonymous with semantically unrelated.

Thus the compiled token branch now has a passing full-vocabulary endpoint
spillover screen in addition to narrow control contrasts. This is still a paired
interchange with recipient routing/background fixed. Broad behavior-preservation
panels, standalone extraction and independent multi-task reuse remain open.


### Fresh behavior controls and matched write directions

A [new32-prefix panel](REGIONAL_BEHAVIOR_CONTROLS_V1_ROWS.json) tests regional
spelling alongside tense, number agreement and simple factual completions. Each
pair changes only “British” to “American”; none of the prefix texts appeared in
the earlier panels. Regional answer words overlap earlier tests, so this is new
context/control evidence rather than entirely new output vocabulary. The first
branch now executes from arbitrary input-token embeddings and the frozen RMS/H
maps, without restricting inputs to the earlier48-token lookup table.

The [native test](REGIONAL_BEHAVIOR_CONTROLS_V1_RESULT.json) passes all registered
criteria. Native regional cue gap is2.6993nats; first-branch transfer is0.05613nats,
or2.079%of that gap. This passes the2%bar narrowly and is a small fraction of the
full regional computation. All24control prefixes are natively correct and remain
correct after the swap. Their mean absolute correct-minus-foil changes are
0.000710nats for tense,0.000259for number,0.002972for factual completions. Native
mean margins are4.171,7.769,3.983nats respectively. These are relatively easy
controls; absence of flips alone would be weak evidence. The registered effect
size bars also pass. Six body forwards took2.26seconds; no fitting occurred.

To test whether a small generic perturbation could explain the target effect,
the [matched write control](REGIONAL_FIRST_MATCHED_WRITE_V1_RESULT.json) applies
16fixed-seed common signed-coordinate permutations to the saved first-branch
write deltas. For a row matrix $\Delta W$ and orthogonal signed permutation $Q$,

$$
(\Delta WQ)(\Delta WQ)^\top=\Delta W\Delta W^\top.
$$

Thus every write norm and every cross-row inner product is preserved; only its
orientation relative to the model changes. Gram error is1.05e-15relative and
CPU suffix replay error7.10e-7. Actual regional transfer0.05613nats exceeds the
controls'95thpercentile0.002315nats (largestcontrol0.004723). This supports
orientation-specificity of the extracted write. These controls are signed
coordinate permutations, not uniformly random rotations or a universal null
model for all circuit mechanisms.

The fresh control screen strengthens selective-interchange evidence for this
small token-source branch. It does not establish full source removal, guarantee
all unrelated behaviors, or provide the remaining routing/current-state inputs.
The next extraction work should close those dependencies rather than accumulate
more easy positive screens of the already-tested token branch.


## Exact conditional token-to-logit path

The next extraction step composes the already validated first-value branch across
attention8/9/13, the two child readings, attention17, and the native final suffix.
It introduces no learned approximation. All recipient context-dependent ports
remain fixed and explicitly required.

Let $z_{jha}$ denote the two first-value readings ($a=1,2$) for each of nine heads
in each of three producer layers. At the one changed cue position $s_*$ there
are54such values. Let $R_{jh,t s_*}$ be the producer's joint QK routing. Define

$$
B_{tao}=\frac{p_t}{\rho_{17,t}}
\sum_H \frac{g_{Ht}}{g_H^{(0)}}\beta_{Htao},
\qquad
K_{jhao}=\sum_t R_{jh,t s_*}B_{tao}.
$$

Here $p_t$ is the frozen shared quadratic parent; $g/g^{(0)}$ is the native
attention17 normalization factor relative to the coefficient reference; and
$\beta$ is the private query-dependent writer. Both QK factors and all positional
operators remain represented. The full path change is

$$
\Delta w_o=\sum_{j,h,a}K_{jhao}\Delta z_{jha}.
$$

The [native compilation test](REGIONAL_TOKEN_PATH_COMPILE_V1_RESULT.json) passes
on all32fresh-panel prefixes: actual donor-write error6.57e-16, independent
synthetic54-port write error2.42e-16, and saved suffix-margin replay error0.
The synthetic values are independent port edits, not necessarily token-realizable
values. This is a conditional multi-layer operator, not a claim that changing
an actual input token leaves its other model descendants fixed.

A dense $K$ stores62,208numbers per context. Keeping the contraction graph is
better:27routing coefficients per position and two1152-dimensional downstream
writers per position. At the tested7–11positions that uses16,317–25,641numbers.
[Literal interface pricing](REGIONAL_TOKEN_PATH_V1_PRICE.json). The two child
values are shared intermediate computations with27incoming producer-head paths.
This is exact structural reuse already exposed by the folded computation; it
is not new low-rank fitting, nor does it make native context generation free.

### Folding the final bilinear layer and full unembedding

Flatten the54source changes into $\delta$ and let $y=y_0+K^\top\delta$ be the
input to the final MLP. Precompute its reader maps $LK^\top,RK^\top$ and the
normalization polynomial

$$
\rho(y)^2=
\frac{\|y_0\|^2+2\delta^\top Ky_0+\delta^\top KK^\top\delta}{1152}
+\epsilon.
$$

The residual after the MLP is exactly

$$
h(\delta)=y_0+K^\top\delta+
D\frac{(Ly_0+LK^\top\delta)\odot(Ry_0+RK^\top\delta)}{\rho(y)^2}+b,
\qquad
\ell(\delta)=30\tanh\!\left(\frac{U\operatorname{RMS}(h(\delta))}{30}\right).
$$

This retains the entire50304-row unembedding and both live suffix normalizations.
It is a rational computation followed by the cap, not a global polynomial in
raw tokens. The two folded input maps cost497,664numbers per context; Down, U,
K/context generation and all native background dependencies still count.

The original strict [full-U audit failed](REGIONAL_TOKEN_SUFFIX_V1_FAILURE.json)
before saving cell values. The [logged audit](REGIONAL_TOKEN_SUFFIX_V2_RESULT.json)
preserves that miss: all state/full-logit relative errors are~1e-15, and donor,
negative, half and1.5-times-donor effect errors are at most1.43e-12, but the small
synthetic edit has relative effect error5.67e-9 against the1e-10bar. Its effect
norm is only0.0002033; absolute output discrepancies remain~1e-12.

The executed [same-direction scale check](REGIONAL_TOKEN_SUFFIX_SCALE_V1_RESULT.json)
raises synthetic amplitude by10,100,1000. Absolute logit errors remain
7.6–8.9e-13 while relative effect error falls to5.36e-10,5.77e-11,6.59e-12.
This supports cancellation at floating-point precision, not a missing algebraic
term. It does not retroactively pass the original tiny-edit criterion. These
are FP64 algebra checks at two preselected contexts using nativefloat32epsilon;
they are not new OOD behavior or a bitwisefloat32 model equivalence claim.

The result is a complete conditional token-value-to-all-logits specification for
this path. The unresolved extraction problem is now explicit: generate the
routing and downstream parent/query/normalization quantities with a smaller
transparent program. Another fitted dense context-specific K would hide that
problem. Retain the shared two-child-per-position graph when pursuing it.


## Trying to generate routing without contextual producer states

The [raw token-routing baseline](REGIONAL_TOKEN_ROUTING_V1_RESULT.json) applies
the actual producer QK weights to token-derived normalized inputs, retaining
both QK factors, their normalization and native RoPE. Nothing is fitted to text.
The first-value readings and downstream conditional path stay unchanged.
Execution passes, but write fidelity fails:102.8%relative error for token-derived
queries and keys,113.8%with native queries/token keys,69.0%with token queries/
native keys. The regional-effect errors are97.3%,115.4%,69.6%respectively.
These failures reject the simple input approximation, not weight-based discovery.

The [gain/direction audit](REGIONAL_TOKEN_ROUTING_V1_GAIN_AUDIT.json) shows the
raw token candidate has cosine−0.317with the native write. Even a hypothetical
best scalar leaves94.9%error. No such scalar is adopted. Keeping native keys
improves direction cosine to0.939, but its scalar-only error floor is34.4%,
still above the10%bar. The issue is not just output amplitude.

A known confound is the large residual background documented in the channels
and middle-pooling dossiers. The [background-restored test](REGIONAL_ANCHORED_ROUTING_V1_RESULT.json)
constructs a zero-input trajectory through the actual weights, with $x_0=0$,
and caches the resulting raw producer inputs $b_j$. It then uses

$$
\widehat r_j=b_j+e_jx_0,\qquad
 e_{-1}=1,\quad e_j=\lambda_{j,0}e_{j-1}+\lambda_{j,1},
$$

followed by native RMS/QK/RoPE. The coefficients at layers8,9,13are29.4523,
34.5761,70.7335. No lexical anchor, corpus statistic or learned correction is
used. One extra zero-input pass supplies the position-dependent background.
This approximation retains direct embedding re-entry but omits the
input-dependent changes in attention and MLP updates.

Background restoration improves all-input write error to53.0%, but the original
10%fidelity bar still fails; regional-effect error remains88.1%. Native queries
with background-restored keys give84.7%write error; background-restored queries
with native keys give26.9%. The latter still has45.4%regional-effect error.
The [scoped gain audit](REGIONAL_ANCHORED_ROUTING_V1_GAIN_AUDIT.json) finds
single-scalar error floors33.8%for both approximated sides and25.9%when native
keys are kept. On regional rows alone these floors are30.7%and21.2%. Thus neither
pooling across control rows nor a single gain correction explains away the miss.

The missing object can be written exactly. For actual attention/MLP updates
$A_k(x),M_k(x)$ and their zero-input counterparts,

$$
r_j(x)-b_j-e_jx_0=
\sum_{k<j}\left(\prod_{m=k+1}^{j}\lambda_{m,0}\right)
\left[A_k(x)-A_k(0)+M_k(x)-M_k(0)\right].
$$

The next weight-folding target is this contextual update remainder as read by
the joint QK computation. The tests demonstrate that a token-only value source
can coexist with genuinely context-dependent routing. They do not identify
which individual omitted update supplies the missing information, and they do
not justify separate tasks for QK1andQK2. Preserve the shared two-child path
while tracing these remaining reader dependencies.


## Corrected article panel and routing diagnosis — 12 September 12:48 UTC

The fresh behavior panel's repeated “A American” error is repaired with “The”
on both sides. Cases, labels and thresholds are unchanged. The shared validator
rejects both known malformed historical panels and accepts all three corrected
panels ([regression receipt](REGIONAL_CUE_ROW_CHECK_V1_PANEL_AUDIT.json)). This
is a specific regression check, not a universal grammar checker. Earlier V1
fresh-panel semantic conclusions and routing numbers above are superseded by
this section; their original receipts are preserved.

The [corrected behavior test](REGIONAL_BEHAVIOR_CONTROLS_V2_RESULT.json) passes:
the native cue gap is 2.6303 nats and first-value transfer is 0.06187 nats, about
2.35% of that gap. All 24 tense/number/meaning controls retain their correct
answers. Mean absolute contrast changes are 0.000728, 0.000218 and 0.002894 nats.
The [matched-write control](REGIONAL_FIRST_MATCHED_WRITE_V2_RESULT.json) also
passes: true transfer 0.06187 versus scrambled-write 95th percentile 0.002482
nats (maximum 0.004960). Small selective transfer remains supported; independent
extraction is still unproved.

| Corrected routing approximation | Write error | Regional signed-effect error |
|---|---:|---:|
| Token queries and token keys | 102.55% | 97.44% |
| Native queries and token keys | 106.57% | 109.31% |
| Token queries and native keys | 70.71% | 60.19% |
| Background-restored queries and keys | 38.98% | 58.59% |
| Native queries and background-restored keys | 89.25% | 95.51% |
| Background-restored queries and native keys | 25.13% | 24.04% |

[Raw routing receipt](REGIONAL_TOKEN_ROUTING_V2_RESULT.json) and
[background-restored receipt](REGIONAL_ANCHORED_ROUTING_V2_RESULT.json) both
pass native replay but fail their unchanged 10% fidelity bars. Both QK factors
remain jointly evaluated in every arm. The zero-input background helps; the
omitted input-dependent updates still matter for this branch.

The executed [geometry audit](REGIONAL_ROUTING_V2_GEOMETRY_RESULT.json) tests
whether an arbitrary scalar could eliminate the remaining write error. For
flattened predicted and reference changes $v,r$, the diagnostic optimum is

$$
\alpha_* = \frac{v^\top r}{v^\top v},
\qquad
\frac{\|\alpha_*v-r\|}{\|r\|}
=\sqrt{1-\frac{(v^\top r)^2}{\|v\|^2\|r\|^2}}.
$$

These identities agree numerically within $10^{-12}$. Even allowing a signed
scalar, the raw two-sided approximation has 94.47% minimum error; restoring
the background reduces that floor to 28.75%. With native keys and restored
queries it is still 23.64%. No scalar is installed or counted as a successful
replacement. This rejects a single gain mismatch as the complete explanation,
not other weight-based factorizations.

### A smaller causal boundary for the next key computation

The same CPU audit counts only 4 unique cue prefixes in the original corrected
32-row panel, 8 in the 48-row geographic panel, and 2 in the fresh 32-row panel.
For a causal transformer, a key at cue position $c$ depends on tokens through
$c$, not later words. Thus the fresh panel's keys depend on “The British” or
“The American”; varying the following task does not provide 32 independent key
contexts. This limits the breadth of the routing validation and suggests a
cheaper exact reference for tracing the missing key updates.

The immediate scientific target is to replay those short prefixes and fold the
joint QK readers through their contextual updates. Prefix truncation/native
key equality has not yet been measured here. A finite prefix table would only
cache native computation: it would not establish general extraction, and it
would not close the full-prompt queries or downstream writers. Those dependencies
remain explicitly charged.


## Causal key boundary and folded contextual-update ports — 12 September

The [native prefix experiment](REGIONAL_KEY_PREFIX_V3_RESULT.json) executes all
112 corrected prompts and the **12 unique prefixes in their union**. The earlier
per-panel counts 4/8/2 overlap: the two fresh prefixes were already in the original
panel. V1 caught this count mistake before loading the model; V2 was rejected by
the queue's literal-key schema. V3 preserves all rows and original numerical bars.

Across attention8/9/13, full-prompt and truncated-prefix joint keys agree within
$9.96\times10^{-7}$ relative error. The exact raw residual update expansion agrees
within $1.31\times10^{-7}$; reconstructed joint keys within $5.27\times10^{-7}$.
All are below the registered $2\times10^{-5}$ tolerance. The 33 native batches
took 1.35 seconds of experiment time, excluding managed startup/loading.

Restoring only one family of contextual update differences is insufficient:

| Key layer | Background only | Restore attention differences | Restore MLP differences |
|---|---:|---:|---:|
| 8 | 106.12% | 96.70% | 47.25% |
| 9 | 96.69% | 81.59% | 40.60% |
| 13 | 90.48% | 76.57% | 40.73% |

Entries are worst-prefix relative errors of the joint key outer product over
heads. Both family-sufficiency predictions fail the unchanged 10% bar. MLP
restoration helps more here, but these are interacting update families, not
additive explained-variance shares or independently identified circuits.
The exact all-update arm passes, so the family misses are not explained by a
broken residual unroll. Grouping all updates of one native type may hide useful
cross-type combinations; this test does not reject those combinations.

These edits replace terms in the key's incoming-residual expression. They do
not recursively remove earlier modules, and no downstream behavior was measured
by this experiment. Prefix replay reduces the necessary key-input context; it
still computes that context with native weights.

### Folding both key readers and retaining their normalization

The saved port matrix $Z$ has one row for the background plus embedding term,
and one row per earlier attention/MLP update difference at the cue. Let $a$
contain their editable scalar amplitudes, so $r=Z^\top a$. Compile

$$
P_{i,h}=ZK_{i,h}^{\top},\qquad
G=ZZ^\top/d,\qquad
\rho^2=a^\top G a+\epsilon,
\quad i\in\{1,2\}.
$$

$K_{i,h}$ is the native key reader for QK factor $i$ and head $h$. With
$p_{i,h}=P_{i,h}^{\top}a$, the two nested RMS operations reduce exactly to

$$
\operatorname{RMS}_{128}\!\left(K_{i,h}\frac{r}{\rho}\right)
=
\frac{p_{i,h}}
{\sqrt{\|p_{i,h}\|^2/128+\epsilon\rho^2}}.
$$

Use the native float32 epsilon in this expression, including when calculating
in float64. Fixed-position rotary maps apply afterward. The input RMS nearly
cancels, but dropping its remaining epsilon term would change the exact model.
Both normalized key vectors still participate together in the routing product.

The [executed CPU port audit](REGIONAL_KEY_PORTS_V1_RESULT.json),
[code](audit_regional_key_ports_v1.py), checks native amplitudes, each update
family alone, and an independently seeded arbitrary amplitude edit at all36
prefix/layer combinations. All144 cases pass the $10^{-10}$ FP64 error bar;
maximum key-vector error is $2.50\times10^{-15}$. This is an exact conditional
execution check, not a new semantic result or native FP64 equivalence claim.

At layer13 the representation has27ports and62,937 scalar coefficients for
both reader projections plus the norm Gram matrix. This prices a single
context's compiled interface only. The native weights and work generating $Z$,
the queries, and the downstream branch are still required and charged. The next
extraction task is to simplify those port-producing computations jointly across
prefixes, rather than treating a native-generated table as the circuit.


### Shared native-update support: selection and actual-query audit

[Preregistered support localization](REGIONAL_KEY_SUPPORT_V1_PREREGISTRATION.md)
selects13of26 native update terms on the four original cue prefixes, shared
across keys at8/9/13. This is dependency localization of the existing frozen
weight-discovered branch, not new unsupervised discovery or a weight fit.
The eight geographic prefixes do not enter support selection. Both forward
and backward supports are serialized before geographic scoring.

[CPU results](REGIONAL_KEY_SUPPORT_V1_RESULT.json): forward greedy gives25.39%
worst original joint-key error and27.53% geographic error. Backward elimination,
the registered optimization red-team, gives20.73% and28.46%. Both miss10%.
All-port replay passes3.54e-7 and dense/contracted metric agreement is5.55e-17.
Backward search does not rescue the forward prediction; neither establishes a
global optimum or rules out smaller cross-module features.

The [actual-query native audit](REGIONAL_KEY_SUPPORT_ROUTES_V1_RESULT.json)
addresses a different confound: key error may lie in directions that queries
do not use. It evaluates both QK factors together with native queries and RoPE
on all112prompts. Forward routing error improves to19.41% on the original panel
and23.40% on geographic prefixes, but both10%predictions still fail. Positive
all-port routing replay passes4.28e-7; experiment time1.25seconds,18batches.
The criterion aggregates heads and causal query positions separately per layer
and panel; it is not a per-head guarantee.

The remaining readout question is whether these score errors matter to the
specific first-value branch after its value and downstream write weighting.
[That next test is registered](REGIONAL_KEY_SUPPORT_EFFECT_V1_PREREGISTRATION.md)
and its existing five-arm executor adaptation is written and syntax-checked.
Binding, managed preflight and native execution remain pending. It keeps both
supports frozen, preserves every earlier miss and uses correctedfresh32 queries
for a basic fidelity/control screen. These duplicate original key prefixes;
new geographic downstream evidence is not claimed.


## Composed-path fidelity improves; geographic amplitude remains unresolved

The [fresh downstream test](REGIONAL_KEY_SUPPORT_EFFECT_V1_RESULT.json) now
passes all three registered predictions. The unchanged forward13 key support
has6.32% first-branch write error and3.81% regional signed-effect error; controls
retain their answers. Backward13 gives6.07%/6.52%. This does not repair earlier
key and routing errors: it measures their consequences after value and downstream
write composition. The distinction is evidence for examining composed paths.

The [geographic test](REGIONAL_KEY_SUPPORT_EFFECT_OOD_V1_RESULT.json) uses all48
previously registered geographic rows and eight key prefixes excluded from
support selection. No support or scalar is refit. Its overall verdict is
**A pass, B/C fail**:

| Geographic family | Forward write error | Forward signed-effect error | Transfer signs preserved |
|---|---:|---:|---:|
| 0 | 3.72% | 3.57% | 6/6 |
| 1 | 5.53% | 5.44% | 6/6 |
| 2 | 16.93% | 17.30% | 6/6 |
| 3 | 5.63% | 6.40% | 6/6 |

Family2 exceeds the unchanged10% bars. Backward13 also misses there at15.73%
write and15.80% effect error. Exact all-update write replay is at most2.31e-7.
The native experiment takes3.44seconds excluding managed startup/model loading.
This is geographic/template transfer, not a fresh natural-text corpus test.

The executed [write geometry diagnosis](REGIONAL_KEY_SUPPORT_EFFECT_GEOMETRY_V1_RESULT.json)
shows family2's forward write has cosine0.999078 with the true first-value write
but norm ratio0.835264. The best diagnostic scalar1.1961 leaves4.29% write error;
backward gives cosine0.999248, ratio0.846837 and3.88% scalar-error floor. No gain
is installed, fit on selection data, or validated as a replacement. The original
17.30% effect miss remains. Across the four forward families, diagnostic scales
range0.9495–1.1961, so these receipts do not establish one universal correction.

This localizes the remaining discrepancy mainly to the magnitude of an already
closely matched composed write, rather than a different output direction.
It does not identify normalization as the cause: the joint QK numerators can
also change amplitude. A useful next algebraic target is the omitted-update
contribution to the product of both QK scores, including its mixed terms and
changing key normalizers. Whole native update deletion is only one possible
structural assumption, and its failure is not absent weight structure.

All prefix update terms are still generated natively. Thirteen selected terms
in a conditional expression are not thirteen recursively executable modules;
removing upstream computation can change later retained terms. Independent
extraction and joint composition of the resulting replacements remain open.


## Mixed key interactions plus shared normalization predict geographic effects

The [registered correction experiment](REGIONAL_KEY_CORRECTION_V1_RESULT.json)
now separates the missing numerator interaction from key normalization. The
frozen13 support,48geographic rows and original native generators are unchanged.
No scalar or support is fit. The result is **A pass, B fail, C pass**: exact
replay holds, normalization alone fails, and retained-plus-mixed numerator with
the true denominator predicts every family's write and signed effect within10%.

Let $r_0$ be the retained raw key input, $\delta r=r-r_0$ the omitted contribution,
and $p_i=K_i r_0$, $\delta p_i=K_i\delta r$. For a fixed native query and cue
position, absorb the actual rounded rotary map into the query-key pairing:

$$
a=q_1^\top R_c p_1,\quad b=q_2^\top R_c p_2,
\qquad
\delta a=q_1^\top R_c\delta p_1,\quad
\delta b=q_2^\top R_c\delta p_2.
$$

Both QK factors belong to the same routing operation. The numerator is exactly

$$
N=(a+\delta a)(b+\delta b)
=\underbrace{ab}_{N_0}
+\underbrace{a\delta b+b\delta a}_{N_{\mathrm{mixed}}}
+\underbrace{\delta a\delta b}_{N_{\mathrm{omitted}}}.
$$

For raw input $r$, retain the shared denominator

$$
D(r)=128^2\sqrt{
\left(\frac{\|K_1r\|^2}{128}+\epsilon\rho(r)^2\right)
\left(\frac{\|K_2r\|^2}{128}+\epsilon\rho(r)^2\right)},
\qquad \rho(r)^2=\frac{\|r\|^2}{1152}+\epsilon.
$$

The key norms are computed **before** rotation; rounded RoPE is not silently
assumed exactly orthogonal. The compared routes are $N_0/D(r_0)$,
$N_0/D(r)$, $N/D(r_0)$, $(N_0+N_{\mathrm{mixed}})/D(r)$ and $N/D(r)$.
Native queries, first-token values and downstream readers remain fixed.

| Geographic family | Denominator-only write/effect error | Numerator-only write/effect error | Mixed numerator + true denominator write/effect error |
|---|---:|---:|---:|
| 0 | 23.99% / 23.94% | 34.62% / 34.57% | 1.74% / 1.73% |
| 1 | 19.91% / 20.00% | 31.46% / 31.78% | 1.23% / 1.21% |
| 2 | 28.96% / 29.23% | 17.34% / 17.25% | 3.01% / 3.07% |
| 3 | 20.24% / 21.33% | 19.28% / 19.62% | 1.73% / 1.89% |

The CPU product identity error is2.80e-16. Native execution takes3.35seconds,
10bodybatches and7suffixarms. The passing approximation omits only
$\delta a\delta b$ while keeping both mixed terms and the full denominator.
It is linear in the omitted contribution **in its numerator**; the full
normalized computation is not linear in that contribution.

The [executed factorial write audit](JOINT_KEY_CORRECTION_ACCOUNTING_V1_RESULT.json)
checks numerator/denominator compensation, with7.01e-16 identity error. Let
$w_{00}$ use retained numerator/denominator, $w_{10}$ restore the numerator,
$w_{01}$ restore the denominator, and $w_{11}$ restore both. Then

$$
w_{11}-w_{00}
=(w_{10}-w_{00})+(w_{01}-w_{00})
+(w_{11}-w_{10}-w_{01}+w_{00}).
$$

On family2, signed projections onto the required write correction are
+2.006 for numerator restoration,−0.702 for denominator restoration, and−0.303
for their interaction; these sum to1. Thus restoring normalization alone moves
opposite the needed correction. Norm changes by themselves did not identify
the mechanism. In other families the terms also substantially cancel; their
norm ratios are not additive explained-variance shares.

This supports a specific composed interaction approximation across the frozen
geographic panel. It preserves the earlier uncorrected support failure and does
not establish discovery of an independently executable small circuit. Computing
$\delta a$, $\delta b$ and $D(r)$ still uses omitted native information. Expanding
mixed terms may even use more scalar arithmetic than directly multiplying two
full scores; degree reduction alone is not a storage/compute saving. The next
extraction requirement is to fold and simplify those correction readers and the
shared normalization together, with all generator and adapter costs included.


## Recursive extraction fails: conditional support is not a closed program

The [recursive key-generator experiment](REGIONAL_RECURSIVE_KEY_V1_RESULT.json)
executes retained updates on its own candidate state. Omitted attention/MLP
writes are replaced with fixed outputs from a zero-input trajectory, rather
than evaluated on native actual-prefix states. Actual residual reentry, token
embeddings, shared first-layer values, RMS and RoPE are retained. No saved
actual-prefix update is consumed by the candidate generator. The all26-update
control checks the same execution machinery against native first-branch writes.

All26 replay passes at<=1.86e-7. Both13-update candidates fail decisively:

| Geographic family | Recursive forward write error | Signed-effect error | Transfer signs preserved |
|---|---:|---:|---:|
| 0 | 80.71% | 81.88% | 6/6 |
| 1 | 73.32% | 74.17% | 6/6 |
| 2 | 102.82% | 103.36% | 1/6 |
| 3 | 91.47% | 92.18% | 6/6 |

Backward13 has75–101%write error and retains only2/6signs in family2. Recursive
key inputs differ from conditional selected-update inputs by up to32–33%; the
all-update comparison is1.71e-7. The experiment takes3.45seconds. This rejects
this fixed-zero-background recursive pruning scheme, not equivalent shared
features or all weights-first methods. It does not undo the earlier conditional
mixed-term prediction result: that result retained native omitted information.

The helper verifies actual update-call counts. Per prefix, forward executes
5attention and8MLP updates plus a direct first-value projection; backward uses
4attention and9MLP updates plus that projection. The all-update control uses
13of each. Two zero-input trajectories require52one-time update calls and
239,616cached scalars (958,464bytes inFP32). Native full-prompt queries and
all downstream computation remain external even in this stronger test.

The [executed dependency and parameter audit](REGIONAL_KEY_DEPENDENCY_PRICE_V1_RESULT.json)
explains why a selected native support was a weak simplicity claim. Treating
native modules as opaque operations, the forward13's producer dependencies
expand to24updates and the backward13's to26. This is conservative graph closure,
not a proof that every edge is semantically necessary in an equivalent program.

Counting unique checkpoint tensors, full-vocabulary embeddings, first values,
residual reentry and needed key maps, the forward selected-module package already
has231.81million scalars; its native producer closure has347.27million. Backward
has239.77million and371.16million respectively. These counts still exclude
full-prompt queries, attention17 shared features/writers, finalMLP/unembedding,
and temporary runtime state. Regenerating the fixed zero cache also requires
omitted weights unless the cache is retained as a charged constant.

The next simplification target must therefore cross native module boundaries
and preserve how retained features are produced. Whole-update selection and a
fixed zero background are demoted as a standalone extraction method. The useful
objects retained from these tests are the explicit correction readers, their
mixed interactions and shared normalizers—not a claim that the13selected native
modules form a small independently executable circuit.


A new [norm-aware feature sufficiency test](NORM_AWARE_FEATURE_SUFFICIENCY_V1_MATH.md) replaces native-update counting with an explicit feature-interface criterion. Positive/negative controls pass; weight-generated equal-feature/equal-norm witnesses show the fourcurrent source readers alone cannot determine both MLP16child contributions on allreal inputs. This is a fixed-interface counterexample, not a natural-text or absent-structure claim. The next norm also needs its own closed update; passing a scalar reader test alone cannot certify multilayer extraction.
