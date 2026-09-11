# Mathematical review: shared output groups versus stable computations

11 September 2026, 22:56 UTC. Previous review: 19:51. User-directed weights-first interaction-path discovery remains the current route. The four requirements remain OOD prediction, extraction, selective removal, and composition/reuse; none is established by the new decomposition alone.

## The decision

The converged sparse path programs have some reproducible coefficient edges, but the frozen eighteen-edge banks failed native behavioral replication. Midpoint accounting attributed a larger discrepancy term to output writing than to reading. This motivates grouping edges by shared output structure. It does not justify fitting the failed validation examples.

The exact restriction solvable now is **output versus quadratic-input-function matrix factorization**, with each input basis and its edge support fixed. This is a classical SVD, already used elsewhere in this project. The new question is whether it identifies stable intervention groups in the newly converged joint path programs.

## Explicit object and domain

Let $d=1152$, $V=50304$, and let $U\in\mathbb R^{V\times d}$ be the entire unembedding. The formal input has two ports:

$$
\xi=(r,z')\in\mathbb R^{2304},\qquad
z'=s z,\qquad s=\|O_{17}\|_F/\sqrt d.
$$

The native pre-MLP residual is $x=r+O_{17}z'/s$. Each frozen joint bank supplies 16 orthonormal readers per port, hence 32 orthonormal readers $b_i$ in the combined input space. Its 96 selected edges have symmetric, coefficient-orthonormal matrices

$$
H_{ii}=b_i b_i^\top,\qquad
H_{ij}=\frac{b_i b_j^\top+b_j b_i^\top}{\sqrt2},\quad i<j.
$$

Write $\phi_e(\xi)=\xi^\top H_e\xi$. The frozen approximation is

$$
\widehat q(\xi)=\sum_{e=1}^{96}w_e\phi_e(\xi),
\qquad W=[w_1,\ldots,w_{96}]\in\mathbb R^{1152\times96}.
$$

This is a degree-two numerator in formal ports. It is not degree two in original tokens: attention contains its own nonlinear normalized readers and products. The executable write is $\widehat q(\xi)/(\operatorname{mean}(x^2)+\epsilon)$. Native bias, direct residual, upstream producers, final RMS and $30\tanh(\cdot/30)$ remain outside this approximation. Port independence defines the coefficient norm, not a claim about reachable native states.

The full output coefficient metric is $G=U^\top U$. Its centered companion subtracts $V\bar u\bar u^\top$. Both are positive definite in this computation, so use $J^\top J=G$. Centering is a structural control; it does not license deleting common pre-tanh writing from the nonlinear model.

## The exact matrix restriction

Our derivation uses edge orthonormality to identify the output/function matricization with

$$
C=JW=P\Sigma V^\top\in\mathbb R^{1152\times96}.
$$

Consequently

$$
\widehat q(\xi)
=\sum_j \underbrace{J^{-1}p_j\sigma_j}_{\text{physical writer}}
\underbrace{\sum_e V_{ej}\phi_e(\xi)}_{\text{scalar quadratic input function}}.
$$

One component can read many products and write a single direction. This is an output-sharing block with an unrestricted symmetric core inside the existing 32-reader space. It need not be one CP product or a low-rank input matrix. For fixed $C$, truncating to $k$ output/function terms minimizes coefficient Frobenius error and leaves squared error $\sum_{j>k}\sigma_j^2$. This standard matrix result does not solve optimization of the input readers or edge support. See Stewart's [SVD review, section 3](https://users.math.msu.edu/users/iwenmark/Teaching/MTH995/Papers/SVD_Stewart.pdf).

This is also a one-mode Tucker restriction: hold the input function dictionary fixed, optimize output rank. Joint Tucker mode optimization and sparse core selection are different problems; a sparse core is basis dependent. The [Kolda–Bader review](https://www.kolda.net/publication/TensorReview.pdf) distinguishes mode unfoldings, CP, and Tucker and describes HOSVD initialization. We do not inherit a globally optimal sparse Tucker solver from the matrix restriction.

The input bank has 36,864 floats. Retaining $k$ groups costs $1152k+96k$ additional floats, 96 scalar products, $96k$ group accumulation multiply-adds and $1152k$ output accumulation multiply-adds, plus input reads and the native dependencies. At $k=8$ this is 46,848 fitted floats. At $k=96$ the redundant factorized representation is larger than the original writers; it is an identity control, not a compression improvement. SVD cost is $O(1152\cdot96^2)$ after the output metric is available. No cubic input tensor is materialized.

## Gauges, uniqueness and stability

Reader signs compensated by writer signs, edge permutations, and simultaneous SVD left/right sign changes preserve the function. Repeated singular values permit rotations of paired singular vectors. A selected whole group is invariant to these internal rotations. A gap separates a potentially stable group from its complement; it does not establish semantic meaning.

Wedin's bound controls left/right singular-subspace rotation through residuals divided by separation from complementary and zero singular values; see [Stewart, section 7](https://users.math.msu.edu/users/iwenmark/Teaching/MTH995/Papers/SVD_Stewart.pdf). Our two starts occupy different edge dictionaries. Therefore comparing their two $1152\times96$ arrays directly would violate a common-coordinate assumption. Instead embed the scalar functions in their common quadratic coefficient space, using the exact cross-feature Gram $K_{ef}=\langle H_e^{(0)},H_f^{(1)}\rangle_F$. We measure principal angles and group differences there. The reported within-fit gaps are diagnostics, not a numerical Wedin certificate; cross-fit separation and residual bounds would also be required.

For two normalized Schmidt terms the complete coefficient cosine is

$$
\rho_{ij}=(p_i^{(0)\top}p_j^{(1)})
\left(v_i^{(0)\top}K v_j^{(1)}\right).
$$

Writer similarity alone omits the second factor. Group inner products sum $\sigma_i^{(0)}\sigma_j^{(1)}\rho_{ij}$ over retained pairs. This contraction is invariant to any coordinated rotation inside a group; rematching names cannot repair a failed group-function comparison.

## Alternatives and limits

Unconstrained CP fitting is not an exact substitute: best low-rank approximants can fail to exist for some higher-order tensors. [De Silva–Lim](https://arxiv.org/abs/math/0607647) establishes this general obstruction. It does not prove degeneracy in our constrained path fits, whose reported local convergence remains valid. Output-sharing blocks avoid that particular unrestricted tensor-rank claim but still impose a hypothesis about circuit organization.

Hierarchical Tucker or tensor trains can factor the newly available quartic coefficient operator across chosen input partitions. A tree contraction does not discover an arbitrary multi-parent arithmetic DAG. Shared polynomial intermediates and distributive rewrites require additional structure search and execution pricing; low rank of an unfolding alone supplies neither a preferred rewrite nor causal reuse. The existing composed-quartic oracle enables that separate route. No quartet of normalized attention inputs can be silently replaced by a fixed polynomial map when extending it through QK.

## Executed consequence and next test

[Code](path_output_schmidt_v1.py) and [receipt](PATH_OUTPUT_SCHMIDT_V1.json) compute exact full and centered decompositions of both frozen 96-edge joint fits. Reconstruction, orthonormality and spectral-tail checks hold within $1.80\times10^{-14}$. Registered atomic and group stability predictions both fail: only four of the leading sixteen centered components match at cosine 0.95, and the leading-eight group cosine is 0.91037, below 0.95.

There is a useful distinction within the miss. The centered leading-eight **output subspaces** align very closely: their smallest principal cosine is 0.99362. The corresponding input-function subspaces have minimum cosine 0.67536. Leading-eight groups retain 89.89% and 88.15% of their own centered fitted coefficient energies, yet differ by 43.04% symmetric relative coefficient RMS. These percentages concern the small fitted programs, not coverage of the entire native MLP tensor. Full-metric leading rank one is more reproducible, but centering destroys that simple picture.

Thus shared output destinations are much more reproducible than the complete computations sent to them. This is compatible with several input computations using the same write space; it is not eight recovered circuits. It also does not contradict the earlier eighteen-edge midpoint accounting: the decomposition and comparison object have changed.

The discriminating next step is native execution of the already specified centered leading-eight group, alongside the entire 96-edge program as a control. Freeze its weights and rank before reading native outcomes. Reuse the existing developmental cache and replication thresholds; do not claim fresh/OOD validation. If group behavior remains unstable, mere SVD relabeling has not repaired identification. If it improves, retain the miss above and test the frozen group on fresh text before promotion. Either outcome informs whether a shared output-mode constraint deserves a native-tensor refit; it does not authorize fitting validation data.

## Native execution and executed red-team

The [native group test](PATH_OUTPUT_GROUP_NATIVE_V1.json) now completed. Instrument, physical-write replication and removal-CE replication pass; swap replication fails. Leading-eight physical disagreement is 2.861%, and per-family removal-CE disagreements range from 0.01475 to 0.01899 nats. Swap-margin relative disagreements are 6.80%, 1.90%, 27.03%, and 62.62% for agreement verbs, count nouns, past and progressive. Progressive signs agree only 75%. The last two families miss the unchanged 25% bar.

The full 96-edge control already has lower physical disagreement, 1.551%, and also passes removal replication while missing past/progressive swaps. Therefore this is **not evidence that SVD grouping repaired replication**. Selecting the earlier eighteen individually matched edges was a materially different intervention boundary; their failure does not imply that the full fitted program is equally unstable. Passing removal replication means the two replicas agree on damage, not that either removal is selective or desirable. Agreement-family removal effects are around two nats in mean absolute CE change, so these are substantial interventions.

[Level-versus-change accounting](PATH_GROUP_CHANGE_ACCOUNTING_V1.json) executes the strongest immediate alternative explanation. About 95.5% of leading-eight write energy lies in its mean over this cache. Removing that mean raises replica disagreement to 6.79%. Paired physical-write changes disagree by 5.83%, 5.04%, 29.18%, and 17.17% across the four families. The registered all-family 10% change-agreement prediction fails. Thus close endpoint-level agreement does not assure reproducible context changes; however, agreement is not solely a shared-mean artifact either.

For a base residual $h$, perturbation $\delta$, RMS $\rho=\sqrt{\operatorname{mean}(h^2)+\epsilon}$, and token reader $u_t$, let $\ell_t=u_t^\top h/\rho$. The diagnostic first-order native-tail change is

$$
\mathrm d f_t[h](\delta)
=\left(1-\tanh^2(\ell_t/30)\right)
\left[\frac{u_t^\top\delta}{\rho}
-\ell_t\frac{h^\top\delta}{d\rho^2}\right].
$$

This follows by differentiating the explicitly retained RMS and tanh, not fitting a local surrogate. The analytical diagnostic uses float64 with native float32 epsilon; original behavioral verdicts retain float32 execution. Its swap-replica disagreements are 6.67%, 2.00%, 27.18%, and 61.63%, close to the exact effects above. The all-family 10% first-order accuracy prediction still misses: agreement errors are about 10.5%, and one progressive replica is 10.17%. Nonlinearity is therefore measurable, but it is not a persuasive sole explanation for the large past/progressive replica gaps: most of those gaps are already present in first-order readout responses.

The result supports shared output destinations and some reproducible native effects, with incomplete context-sensitive identification. It does not support eight semantic circuits or general removal/composition success. The next substantive method comparison should change the discovery object—jointly constrain shared outputs while fitting the native coefficient tensor, or fit the actual deeper quartic composition—rather than keep relabeling this frozen fit until a developmental panel passes.
