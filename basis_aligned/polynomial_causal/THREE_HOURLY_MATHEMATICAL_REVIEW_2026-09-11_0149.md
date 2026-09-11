# Three-hour mathematical review — 11 September 01:49 UTC

Review work began across the due boundary; executable controls and native
protocol completed afterward. User authority remains the bilinear handoff and
weights-first structural search, not the stale better_math_ideas goal wording.

## Decision

Test a shared **input** factor with unrestricted output-linear partners. This
does not require the whole tensor to have a small output rank. The candidate
has an exact local removal specification and an explicit composition correction.
Its native writer follows from algebra, avoiding arbitrary sparse token writers.
The native solver repair remains an independent live branch.

The key limitation is already quantitative: any one such factor captures at
most2.0673%of full coefficient energy or1.1309%of centered energy. A shared
reader can nominate a partial computation, not explain the whole layer. Those
are not limits on behavioral importance or on many shared readers together.

## Actual object, domain and price

For normalized MLP17 input x inR^1152, the homogeneous module contribution is

$$
f(x)=D[(Lx)\odot(Rx)],\qquad
T_v=\sum_{j=1}^{4608}(UD)_{vj}\operatorname{sym}(l_jr_j^\top).
$$

L,R have4608rows; D is1152by4608; U is50304by1152. The contraction has two
tied input slots, a native product index, residual writer, and vocabulary reader.
It is degree2in normalizedx, signed and even underx→−x. Downbias, residual
background, finalRMS and30tanh(score/30) stay outside this tensor. Earlier model
layers have additional normalized QK products and residual re-entry, so the
entire raw-input model is not this degree-two polynomial.

Allowed formal inputs are allrealx; the exact intervention is performed after
the MLP input normalization without renormalizing the edited vector. Discovery
uses Frobenius coefficient norm across all tokens/input pairs. This native
Euclidean input metric changes under nonorthogonal coordinate changes unless
the metric is carried along. No activation distribution or language labels enter.

A candidate scalar a^Tx gates a residual linear mapM_a x. With||a||=1, signa
is arbitrary and reversing it reversesM_a; their product is invariant. There
is no imposed token sparsity or output-rank ceiling. Full price is1152+1152²
coefficients, nativeU and the remaining model retained. Rankrpartner costs
1152+2*1152*r numbers, but its approximation error must be measured separately.
At r16 this is38,016numbers, not an exact native replacement by assumption.

## Literature mapping and rejected shortcuts

The subspace of symmetric matrices sym(a b^T) has the familiar low-rank matrix
projection structure. Equation3.5of
[Candès–Recht](https://arxiv.org/pdf/0805.4471) projects matrices onto a space
with prescribed row/column subspaces. Setting both tospan(a), then restricting
to symmetric matrices, gives the exact projection below. No random-sampling,
incoherence or matrix-completion recovery theorem is imported: we know all
weights and are selecting the subspace, not filling missing entries.

[De Lathauwer–De Moor–Vandewalle](https://epubs.siam.org/doi/abs/10.1137/S0895479898346995)
study best multilinear-rank approximation and higher-order iteration. A plain
rank-(r,r,output) Tucker core would keepPQP and discard cross-boundary terms.
Our subspace instead retains every interaction touching the selected reader.
The relation motivates subspace optimization but does not make the two models
equivalent or supply a global solution to our quartic sphere problem.

[Bhargava–Saraf–Volkovich](https://arxiv.org/abs/2105.01751) give reconstruction
algorithms for exact constant-rank tensors and constant-top-fan-in arithmetic
circuits. The native4608-product sum and high-error current approximations do
not establish that regime. Their constant-rank guarantees cannot certify our
native fit, and factoring one scalar polynomial would miss multi-output reuse.

[Li–Precup–Rabusseau](https://arxiv.org/abs/2010.10029) connect weighted automata,
tensor trains and linear second-order recurrent networks through finite-rank
Hankel structure. This supplies a genuine exact realization theorem for that
linear recurrent class. Our normalized attention/residual stack is not given as
that transition model, nor have we established a finite Hankel rank or obtained
its required blocks. A tensor-train notation alone does not transfer the theorem.
Building those data-driven blocks now is also lower priority under the user's
weights-first direction.

Thus the immediate theorem-level result is a fixed-reader projection and exact
intervention identity, plus a bound valid for every reader. Searching the reader
still needs numerical optimization and stability checks.

## Fixed-reader optimum and executable extraction

SetP=aa^T with||a||=1. The orthogonal coefficient projection is

$$
T_v^{(a)}=PT_v+T_vP-PT_vP
=\operatorname{sym}(a b_v^\top),
\qquad b_v=2T_va-(a^\top T_va)a.
$$

For fixeda this is the unique least-squares projection; the mapb→sym(a b^T)
is injective. Its complement is(I−P)T_v(I−P), so

$$
x^\top T_v^{(a)}x
=x^\top T_vx-[(I-P)x]^\top T_v[(I-P)x].
$$

The full residual implementation is

$$
f_a(x)=(a^\top x)M_ax,
$$

$$
M_a=D[\operatorname{diag}(Ra)L+\operatorname{diag}(La)R]
-D[(La)\odot(Ra)]a^\top.
$$

Hence U M_a is automatically a legal vocabulary writer. This is a shared scalar
input with many distinct linear partners; it has no small output-rank assumption.
Extraction is exact for the chosen local input-edit effect, with explicit native
background. It does not yet establish a semantic circuit or final-logit additivity.

For orthogonal readersa,b, adding their separate removals double-counts the
mixed term. The correction in tokenv is

$$
2(a^\top T_vb)(a^\top x)(b^\top x).
$$

Subtract it once to obtain removal of both coordinates. Nonorthogonal readers
need their joint-span projector; the orthogonal formula is not assumed there.
This supplies a concrete composition interface instead of silently dropping
the cross terms that defeated previous closed-block interpretations.

## Reader search, global bound and identifiability limits

LetS=sum_v T_v². The coefficient energy captured by a reader is

$$
E(a)=2a^\top Sa-\sum_v(a^\top T_va)^2,
\qquad E(a)\leq2\lambda_{\max}(S).
$$

BothS andtraceS were already computed in the congruence study. Its stored
operator bound is4lambda_max(S)/traceS; dividing by2 gives the one-reader
capture ceiling quoted above. This is an analytic inequality evaluated from
verified FP64 spectra, not a floating-point interval certificate.
[Reused bound receipt](SHARED_INPUT_FACTOR_V1_PRIOR_BOUND.json).

The exact gradient is4Sa−4sum_v(a^TT_va)T_va. Withh=(La)⊙(Ra) and
K=(UD)^T(UD), the quartic term ish^TKh. Thus evaluations useO(4608²+4608*1152)
work andO(4608²+1152²+4608*1152)storage after preprocessing, not a
50304by1152by1152tensor. Sphere projected ascent uses the analytic gradient,
Armijo backtracking and independent starts. A small tangent gradient proves
stationarity only, not a global maximum.

If an exact family shares a linear polynomial factor and its remaining linear
partners are not all collinear, their common factor is unique up to scale.
With a single product, either of its two factors can instead play the shared
role. Approximate native factors require quantitative stability evidence;
agreement across starts is useful but not a uniqueness theorem. No sparse-code
or statistical-independence assumption is needed for the fixed-reader identity.

## Partner simplicity and actual consequence

The partner map remains dense until tested. WithJ=(I+aa^T)^(1/2),

$$
\sum_v\|\operatorname{sym}(a(UM_a)_v)\|_F^2
=\tfrac12\|UM_aJ\|_F^2.
$$

WhitenU withCholesky(U^TU), then take the SVD ofroot^T M_a J. Its singular
spectrum gives the exact best rank-r partner error for fixeda in this norm.
The inverse transforms retain the native output interface. This is a conditional
linear simplification, not an assumed low-rank input/output bottleneck.

Executed CPU controls: projection and analytic-gradient identities<=2.2e-16,
native local-removal/composition replays<=5.1e-16, planted shared-factor recovery
from a random start converges in22steps with100%capture.
[Controls](SHARED_INPUT_FACTOR_V1_CONTROL.json).

The next native experiment is implemented: fourstarts each forfull andcentered
objectives, exact bound recheck, native64formal-input intervention replays,
reader stability and weighted partner spectrum. Its opposing outcomes are a
stable shared reader with a simple partner versus a dense partner or unstable
reader, with the global one-reader limit kept visible.
[Protocol](SHARED_INPUT_FACTOR_NATIVE_V1_PREREGISTRATION.md).

This is higher-information than another identical sparse-product chunk: it tests
input reuse without the low global output-rank restriction and exposes its exact
removal/composition interface. The existingMLP17calibration/common-channel dossier
must still be checked before naming any returned direction. OOD prediction,
semantic extraction, selective removal and model-level reuse remain unproved.

## Native consequence completed and next CPU step

Nativeexperiment completed02:02:46 in4.16s: A/B/Cheld,Dfailed. All8starts
converged. Centeredcapture1.04876%,readeragreement>=.999999999995; fullcapture
1.65837%. Centeredpartner rank16 retains55.2567%,rank90=279. Fullpartner
rank16 retains79.5631%,rank90=105. Localnative removal/compositionreplay<3.8e-15.
[Receipt](SHARED_INPUT_FACTOR_NATIVE_V1_RESULT.json).

Red-team ofD: the reader was selected for unrestrictedpartnercapture. CPU
cross-evaluation shows the full-selectedreader givescenteredrank16capture.0062663,
better than.0057951forcentered-selectedreader. Thus jointreaderselection can
change under the smallpartnerobjective. The fixedreaderSVDmiss remainsvalid,
but doesnotreject the jointconstraint. [Audit](SHARED_INPUT_PARTNER_OBJECTIVE_V1_AUDIT.json).

The next joint variable-projection score is now implemented and CPUcontrolled:
half the sum of the top-r squared singularvalues ofroot^T M_a J, differentiated
with respect to the normalizedreader. Dense nativecoefficient capture agrees
5.6e-17; tangentfinite-difference gradient2.4e-11. A cutoff singularvaluegap is
required for ordinary smooth derivatives. No nativejointoptimizationyet.
[Control](SHARED_INPUT_RANKED_PARTNER_V1_CONTROL.json).

Next mathematical review04:49UTC; next hourly02:22UTC.
