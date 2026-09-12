# Shared cubic source features with private query computations

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
