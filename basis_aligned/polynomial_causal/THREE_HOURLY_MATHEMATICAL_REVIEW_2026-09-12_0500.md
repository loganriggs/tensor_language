# Mathematical review: independent blocks versus reusable arithmetic

12 September2026,05:00 UTC. Previous review02:00. The goal remains a simpler
explicit computation with OOD prediction, extraction, selective removal and
composition/reuse. The later bilinear handoff and weight-first directions
supersede stale better_math_ideas wording. No new circuit satisfies all criteria.

## Exact object and the current evidence

The native model has18blocks, residual dimension$d=1152$, bilinear width$m=4608$,
9attention heads and$V=50304$unembedding rows. The current isolated path is

$$
x\xrightarrow{L_{16},R_{16}}h=(L_{16}x)\odot(R_{16}x)
\xrightarrow{\lambda D_{16}}p
\xrightarrow{L_{17},R_{17},D_{17}}q
\xrightarrow{U}\text{logit numerator}.
$$

$x$ is the normalized MLP16 input. This homogeneous numerator has degree4 in$x$;
the actual model includes biases, residual and attention source-pair cross terms,
input-dependent RMS denominators and final$30\tanh$logits. These remain explicit.
Neither this polynomial nor an additive pre-head decomposition alone predicts
the effect of renormalizing or jointly changing native inputs.

The exact producer coefficient Gram$H$ gives$z=H^{-1/2}p$, and the folded forms
are$S_v=\sum_j(U_cD_{17})_{vj}\operatorname{sym}(l_jr_j^T)$ with
$L=L_{17}H^{1/2},R=R_{17}H^{1/2}$. Here$U_c$centers vocabulary rows in the
**discovery metric**. Actual residual writers retain their full native output;
common pre-tanh shifts are not declared behaviorally irrelevant. The paired
coefficient norm is not the fully symmetrized quartic norm or a text norm.
Orthogonal basis changes preserve this metric; arbitrary coordinate transforms
must transport it. Producer realizability restricts which$z$occur in the model.

Recent results distinguish representation from computation. Rank32full-output
capture improves with producer weighting, but native sufficiency fails. A
selected two-output component has useful developmental fidelity yet fails fresh
and contextual swap checks. Common/difference rewrites preserve its graph
exactly; deleting small difference parents loses important mixed products.
Direct half-dimensional block rotations finish at cuts0.839/0.840 after1500steps
each, unconverged, with overlap0.718. These are no absence-of-structure results.

## Mathematical matches, assumptions and limits

**Common reducing subspaces.** An orthogonal projector$P$defines exact independent
blocks if$[P,S_v]=0$for every output. This is a direct matrix-algebra/commutant
mapping, as in [Maehara–Murota](https://epubs.siam.org/doi/10.1137/090779966).
Our native matrices satisfy real symmetry, but exact reducibility and a separated
block spectrum are unknown. We do not implement or inherit that paper's complete
error-controlled finest-block algorithm. Approximate partitions need separate
stability, energy-balance and behavioral evidence.

Our derived normalized relaxation uses

$$
\Phi(X)=\sum_vS_vXS_v,\quad K=\Phi(I),\quad
\mathcal L(X)=KX+XK-2\Phi(X),\quad\mathcal M(X)=KX+XK.
$$

For$X=P-\operatorname{tr}(PK)I/\operatorname{tr}K$, the generalized Rayleigh
quotient equals the exact normalized cut. The true smallest eigenvalue after
removing identity bounds every orthogonal projector's cut. Positive-definite$K$
is required for the selected whitening; unsupported directions cannot be ridged
silently. A numerical Krylov Ritz minimum is not a certified lower bound, even
with a small residual. Full derivation and caveats are in
[the primary note](NORMALIZED_COMMUTANT_RELAXATION_V1_MATH.md).

Each exact action uses the hidden writer Gram and costs$O(m^2d+md^2)$work,
$O(m^2+md+d^2)$working storage. Symmetric packed dimension is664,128;32Lanczos
vectors add about170MB float64host storage, avoiding a multi-terabyte dense
operator. The prepared native implementation uses the positive-semidefinite
shift$I-\overline{\mathcal L}/2$, two starts and explicit residual/action limits.

**Low-rank matrix elements and bases.** A family of output mixtures spans a
matrix subspace, matching [Nakatsukasa–Soma–Uschmajew](https://arxiv.org/html/1503.08601v2).
Their method uses singular-value shrinkage then rank-constrained alternating
projections, with local convergence analysis; the general nonzero low-rank
search is NP-hard. A dense square SVD costs$O(d^3)$per step, and projection into
a stored$k$-dimensional matrix span costs$O(kd^2)$. Our real-product constraint
also restricts inertia, unlike generic rank2. The prior17Dsource-span experiment
already studied this mapping and found informative one-product bounds. Changing
the paper citation is not a new method; a full-output span or new composition
would need its own tractable projection and assumptions. This is lower priority
than completing the current exact-operator comparison.

**Shared arithmetic instead of independent blocks.** For unit$a$, the projection
$P_aS_v+S_vP_a-P_aS_vP_a$collects every interaction involving$a$. The existing
shared-parent solver optimizes its exact total energy and retains unrestricted
linear partners. This maps directly to one shared node with multiple consumers,
without an independence assumption. Its sphere gradient is exact, but the
objective is nonconvex; the one-parent upper bound and multi-start convergence
do not imply global uniqueness. Per evaluation the dominant cost is$O(m^2+md)$
after constructing$K$, followed by a$d$-square partner spectrum when pricing.

## Executed consequences and decision

The dense generalized relaxation recovers the planted blocks that trapped
local optimization; matrix-free eigenvalues match the dense controls within
3.1e-15. The native spectral test completed at04:59:41 in30.95seconds: both starts
converged, with relaxed estimates0.417819/0.528010 and agreement4e-15. The
rounded cuts0.900116 fail separation despite perfect partition agreement.
This resolves the registered spectral comparison; no text fitting or
certified global lower bound is claimed.

The exact family$f_j=x_0x_j$has one reused parent and five products, yet scalar
commutant and relaxed minimum$2/3$. This executed and analytically checked
counterexample proves that even a certified block negative cannot reject
simple reusable DAGs. Given the spectral separation failure, the already prepared producer-folded
shared-parent test is now submitted. Prioritize it over further half-block
threshold/rank sweeps.
It reuses the old solver; the earlier single-layer result was stable but had
broad partners, so folding changes the object rather than concealing repetition.

A second consequence is executable now: consumers can be folded before pricing
an extracted producer. Folding the existing64readers into$D_{16}$replays the
saved component at3.85e-15relative error and reduces its conditional matrix
price from16,001,344to10,914,112values. Native$L_{16},R_{16}$and background remain;
this is not a whole-model reduction or behavioral repair. See the
[executed receipt](PRODUCER_READER_COMPILE_V1_RESULT.json). This corrects automatic
full-producer charging when only a few output combinations are needed.

Any promising new shared-parent candidate must expose its read, partner
operations and consumers, retain cross-term accounting for joint edits, undergo
dossier/alias checks and frozen native validation. Coefficient concentration,
nice eigenvectors and exact replay do not replace the four behavioral properties.
Next mathematical review08:00; hourly review05:27.
