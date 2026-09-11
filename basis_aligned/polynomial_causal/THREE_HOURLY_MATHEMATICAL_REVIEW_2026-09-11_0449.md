# Three-hour mathematical review —11September04:49UTC

The previous turn made progress in b46827995. Current shared-span search has
completed with36converged starts but no passing candidate. The full goal remains
OOD prediction, extraction, selective removal, composition/reuse and a simpler
explicit program. The user's bilinear handoff and weights-first correction
override the stale better_math_ideas wording. No data-guided discovery is needed
for the next selected consequence.

## Decision and actual objects

Return to the unconverged overlapping multi-output block family on the whole
unembedding/MLP17 tensor. A solver change is warranted before any structural
negative. Stop extending the single-product search inside the same nine source
spans: its analytic bounds are now informative and its scalar simplifications
are too weak for the registered screen.

The tensor is

$$
T_v=\sum_{j=1}^{4608}(UD)_{vj}\operatorname{sym}(l_jr_j^\top),
\qquad f_v(x)=x^\top T_vx,
$$

with$v=1,\ldots,50304$, normalized$x\in\mathbb R^{1152}$, and native two-reader
product index$j$. This contraction is degree2 and even in$x$. Attention routing
has two QK factors multiplied together, source/value multiplication and sums;
RMS denominators, residual re-entry, bias, and final30tanh remain explicit. The
raw-input model is not a degree-two polynomial. No theorem for unrestricted
polynomials automatically accounts for these normalization dependencies.

Discovery uses the full Frobenius coefficient metric, retaining the native
Euclidean input geometry. This is not an activation distribution. A nonorthogonal
coordinate change requires carrying the metric or mapping errors back.

The next approximation family has16possibly overlapping blocks:

$$
\widehat T_v=\sum_{g=1}^{16}\sum_{m=1}^{4}(Uw_{gm})_v
E_g C_{gm}E_g^\top,\qquad E_g^\top E_g=I_{16}.
$$

Each symmetric core has unit Frobenius norm; its scale is absorbed by its writer.
The loss remains coefficient error plus the existing$.01$component-energy
penalty. Conditional writer solves are exact. Parameter price377344includes
294912reader entries,8704core entries and73728residual writer entries. There
are2176unique within-block quadratic monomials, plus reader/core/output mixing;
U and native remainder are retained. This prices a candidate subprogram, not
the whole model or a certified extracted circuit.

## Mathematical matches and limitations

**Low-rank elements in a matrix subspace.** The just-completed source experiment
maps to a17-dimensional subspace of1152-square symmetric matrices. The published
low-rank-basis method uses shrinkage and projection with local convergence
results. Our one-real-product constraint additionally limits inertia to one
positive and one negative eigenvalue. We used a spectral objective on the unit
coefficient sphere, not that paper's full algorithm or a global guarantee.
[Nakatsukasa–Soma–Uschmajew](https://arxiv.org/abs/1503.08601).

For any orthonormal matrix basis$H_i$, unit$c$and$S=\sum_iH_i^2$, Cauchy–Schwarz
implies$Q(c)^2\preceq S$. A$k$-product approximation has rank at most$2k$, yielding
the eigenvalue-sum upper bound recorded in the latest report. This derivation
needs no random-weight, independence or identifiability assumption. Native
one-product ceilings are10.99–20.56%; the16-product ceiling is vacuous. Therefore
one product cannot capture half of any function in these fixed spaces, while
larger or differently selected programs remain open. Numerical eigensystem and
construction checks passed6.7e-15; four heads retain different local solutions.

**Overlapping block terms and constrained optimization.** The family above is
a sum of small multilinear blocks with tied symmetric input factors. Structured
block-term methods support this kind of model and coupling; the exact selected
penalty and parameter sharing still need to be implemented correctly.
[Tensorlab](https://www.tensorlab.com/doc/btd.html). The new adapter reuses the
existing Torch objective and standard Pymanopt constrained conjugate gradient.
Its local stationarity test is not global recovery or factor uniqueness.

For a full-rank block frame$B_g=E_gR_g$, the transformation

$$
B_g A_{gm}B_g^\top=E_g(R_gA_{gm}R_g^\top)E_g^\top
$$

preserves every function. Renormalize the transformed core and transfer its
scale to the writer. This removes nonorthogonal within-block redundancy without
forcing$E_g^\top E_h=0$for different blocks. An orthogonal gauge remains:
$E_g\mapsto E_gO_g$, $C_{gm}\mapsto O_g^\top C_{gm}O_g$. The Product manifold
does not quotient out that remaining gauge. Consequently compare functions and
block spans, not raw parameters, and retain restart checks. Basis collapse or
ill-conditioned writer Gram matrices must remain explicit failures.

**Generic tensor-network/minimal-realization shortcuts.** Tensor trains or
hierarchical Tucker could reorganize contractions, but arbitrary splitting of
the1152coordinates introduces unverified structure and does not solve the
current symmetric overlapping-block objective. The constant-rank arithmetic
reconstruction and weighted-automaton results mapped in the01:49review still
lack their required exact rank/transition assumptions here. Neither route
currently dominates correcting the known unconverged block fit. No new global
identifiability claim follows from referring to this model as a tensor network.

## Immediate executable consequence

The Product(Stiefel per block, unit-core Oblique) adapter already passes
independent dense contractions7.2e-15 and conditional finite gradients4.7e-12;
its toy fit converges in1.48seconds. After recording this review, check the
saved native QR-gauge initialization against the new adapter's block functions,
writer scales and inter-block overlap. This is the remaining cheap bridge before
the native wrapper and managed continuation. Keep lambda=.01 and the original
full-U metric. Native convergence is still untested; do not substitute this toy
result for it.

This direction changes the ability to identify reusable within-block computations
and overlapping output groups. A better coefficient fit alone is not promotion:
the resulting blocks still need stable identification and frozen FineWeb
interventions addressing the four requested properties. Next mathematical
review07:49UTC; hourly review05:22UTC.
