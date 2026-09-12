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
