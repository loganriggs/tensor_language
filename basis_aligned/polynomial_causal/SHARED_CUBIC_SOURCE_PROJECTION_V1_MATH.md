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
