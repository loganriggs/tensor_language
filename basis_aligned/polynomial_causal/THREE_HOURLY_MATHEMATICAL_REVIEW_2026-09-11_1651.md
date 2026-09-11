# Mathematical review — 11 September16:51 UTC

The goal remains a simpler executable model with OOD prediction, extraction, selective removal and composition/reuse. Follow the bilinear handoff and Logan's weights-first/DAG clarification. Today's precise question is whether unstable fitted groups hide stable computations that can be recovered by reorganizing their linear combinations. This review supplies exact solutions of that restricted question, not a theorem about the existence of circuits in the native model.

## Object, scope and price

The folded final branch is

$$
T_{vij}=\sum_{k=1}^{4608}(UD)_{vk}\operatorname{sym}(L_kR_k^\top)_{ij},
\qquad T\in\mathbb R^{50304\times1152\times1152}.
$$

Both input modes act on the same normalized vector. Coefficients are real and signed; antisymmetric input coefficients are invisible. The unembedding Gram admits an exact1152-dimensional output isometry. The local object is homogeneous quadratic; bias, residual addition, actual RMS and30tanh tail remain separate. Earlier attention includes normalized QK products and rounded rotary operations. Folding upstream must retain those operations, not replace the whole model with an unconstrained polynomial.

Current fitted groups have the form

$$
T_g=c_g\otimes Q_g,\qquad Q_g=B_gH_gB_g^\top,
\qquad H_g=H_g^\top,\quad \operatorname{rank}Q_g\le16.
$$

There are64groups. In the shared graph, some columns of group bases arise from shared linear readers; groups can consume several parents. The contraction is linear reads → products/small quadratic cores → output writes. Input-frame transformations, output/core rescaling and group permutations create gauges. Changing a group mixture generally destroys its rank16/single-output structure; an arbitrary group-basis transformation is a symmetry of the *linear span*, not of the LL1 model family.

The objective uses the full folded coefficient norm plus0.01times summed group tensor energy. Original storage is1,254,400floats with1024linear readers and1024variable products; shared graphs separately price saved floats, indices and extra products. Unembedding, whitening, bias and native background remain required. Computing a dense mixture of64vector-valued group outputs requires additional coefficients and vector additions unless algebraically absorbed; its small64×64 analysis matrix is not its execution price.

## Exact restrictions and primary-literature mapping

**Symmetric generalized eigenproblems.** Let $\Phi_0v=\sum_gv_gT_g^0$ and $\Phi_1v=\sum_gv_gT_g^1$. With $A=\Phi_0^*\Phi_0$ and $K=(\Phi_1-\Phi_0)^*(\Phi_1-\Phi_0)$, the best same-coefficient mixture minimizes

$$
\rho(v)^2=\frac{v^\top Kv}{v^\top Av}.
$$

Whitening $A$ reduces $Kv=\lambda Av$ to a symmetric eigensystem. This is exactly the symmetric-definite problem documented by [LAPACK](https://www.netlib.org/lapack/lug/node34.html). Here $A$ has full rank64and condition11.38; supported-range threshold1e-12drops no dimensions. Eigenvectors can be normalized in the $A$ metric; repeated eigenvalues identify a subspace rather than unique axes. After the exact group Gram contractions, the solve costs $O(G^3)$ time and $O(G^2)$ storage. Group Gram construction costs roughly $O((Gr)^2d+G^2d_o)$ with existing factor contractions, not the size of the vocabulary tensor. This gives a global solution over mixture coefficients for these *fixed* banks; it says nothing about optimizing new readers.

**Independent mixtures and principal angles.** The same-coefficient condition could miss a mere change of group coordinates. Remove it: write $B=\Phi_1^*\Phi_1$, $C=\Phi_0^*\Phi_1$, whiten both Grams and take

$$
W_0^\top A W_0=I,\quad W_1^\top B W_1=I,
\qquad W_0^\top C W_1=P\Sigma R^\top.
$$

The canonical old/new functions $f_i=\Phi_0W_0p_i$ and $g_i=\Phi_1W_1r_i$ have unit norm and inner product $\sigma_i$. Orthogonal projection of $f_i$ onto the new span is $\sigma_i g_i$, hence, directly,

$$
\min_w\|f_i-\Phi_1w\|^2=1-\sigma_i^2.
$$

This derivation is basis invariant under invertible changes within either group bank. It is the principal-angle/SVD restriction associated with Björck–Golub's1973work, DOI10.1090/S0025-5718-1973-0348991-3; publisher/mirror PDF fetches returned403today, so no uninspected theorem detail is being imported. The equations above derive the implemented projection identity directly. Complexity after Grams remains $O(G^3)$; uniqueness of individual canonical vectors requires separated singular values. Numerical rank and inner-product checks matter; all selected CP projection replays are measured independently.

**Other routes reconsidered.** [Generic CP identifiability](https://arxiv.org/abs/1403.4157) concerns exact rank-one decompositions and specified generic/nonsingularity conditions. These noisy, partially symmetric, unconverged LL1 approximations have no verified mapping to those assumptions. [Hierarchical tensor recovery](https://arxiv.org/abs/1404.3905) organizes mode subspaces and analyzes restrictions such as tensor restricted isometry; our known third-order coefficients and arbitrary shared arithmetic graph do not satisfy an established recovery mapping. Its hierarchy is not automatically a hierarchy of reusable circuits. [Weighted automata/TT spectral learning](https://arxiv.org/abs/2010.10029) has a provable linear2-RNN/finite-Hankel object; no such exact realization of this normalized residual network has been established. [egg](https://arxiv.org/abs/2004.03082) can organize supplied exact arithmetic rewrites; it does not itself fit approximate shared real-valued readers. These remain distinct alternatives, not solved native problems.

**Optimization methods remain relevant.** [Kasai–Mishra](https://proceedings.mlr.press/v48/kasai16.html) exploit Tucker symmetry and least-squares geometry through a quotient-manifold metric. Our sum of tied symmetric LL1 groups needs its own metric and treatment of shared readers. For one unit-writer group with orthonormal $B$ and a horizontal perturbation $B^\top\delta B=0$,

$$
\|\delta(BHB^\top)\|_F^2
=2\operatorname{tr}(\delta B^\top\delta B H^2).
$$

Thus the fixed-core self Gauss–Newton Hessian for squared residual scales as $4H^2$; this suggests a small-core preconditioner, not an exact coupled reduced Hessian. Cross-group interactions and the eliminated cores still contribute. [Large-residual variable projection](https://arxiv.org/abs/2402.13865) corrects Gauss–Newton curvature; our roughly88%unexplained coefficient energy makes small-residual approximations questionable. Current L-BFGS already approximates full curvature, so this paper does not establish a defect or a speedup without a matched implementation. Core elimination is converged; the outer optimization is not.

## Executed consequence and red-team chain

The first new original-family arm finishes with11.883872%capture, slightly below the old20-minute11.889098%. Both miss convergence. Whole functions have cosine0.97049but24.29%relative difference; group changes cancel strongly. We tested whether organizing existing factors repairs their identity before commissioning more fitting machinery:

1. [Exact small coarsening](LL1_OPTIMIZER_COARSENING_V2.json): enumerate41,728subsets of size≤4containing each of10poorly matched anchors. Only1/10can halve its singleton drift energy; none also achieves≤10%relative subset error. Numeric replay≤7.1e-15. Both scientific bars miss. V1's NumPy-boolean serialization failure is preserved separately; V2 only repairs reporting casts.
2. [Same-coefficient signed mixtures](LL1_STABLE_MIXTURES_V1.json): full generalized eigensolve finds2directions with≤10%relative change, below the16-direction bar. None of the10unstable anchors has≥50%energy in that stable old subspace. Both bars miss; numerical checks≤3.8e-15.
3. [Independent-mixture span alignment](LL1_SPAN_ALIGNMENT_V1.json): the more permissive, basis-invariant solve still finds only2directions, with projection errors2.496%and6.309%; the third is21.941%. No unstable anchor is half covered. Numeric replay≤4.5e-15; both scientific bars miss. This removes the same-coefficient confound within the fixed-bank comparison.

The two canonical old directions almost coincide with existing groups8and18: squared cosines0.9999981and0.9999275. This is not evidence for a newly discovered hierarchy. Prior dossier records already associate group18's main square with earlier square150; the immediate next CPU action checks aliases against the saved256-square bank before any new component identity.

## Decision and continuation

The exact linear relation search is cheap and informative, but it has now answered its restriction. Do not keep searching smaller group subsets or different linear mixing penalties on these same two fixed banks. Their disagreement is not broadly repaired by a coordinate change. Because both underlying fits are unconverged and rank/budget restricted, absent native structure does not follow.

Finish the already registered matched shared-graph fits, score their executable parent removals with the existing interaction correction, and retain the stronger old optimizer reference. A graph family should earn its place through reproducible shared-node functions and interventions, not a slightly better loss. If optimization remains the bottleneck, measure native small-core conditioning before choosing a manifold/preconditioned continuation; do not relabel this diagnostic as that implementation. No FineWeb-guided fitting is opened.

Next action: alias-check the two frozen canonical directions using existing exact CP contractions; then score the completed spectral graph when it arrives. Spectral-graph is live and the two native-start arms remain managed. No identified four-property circuit is added. Next mathematical review19:51UTC; hourly review remains due17:22.

Executed continuation: [canonical-mode alias audit](LL1_CANONICAL_MODE_ALIASES_V1.json) confirms the modes coincide with existing LL1groups8/18 (squaredcosines0.999998/0.999928), but neither meets the0.95single-prior-square alias bar. Best prior atoms are190at0.73742and150at0.90148. NumericAandexisting-groupCheld; single-squareBmisses. Do not equate a whole LL1group with its previously identified shared-square component. No new circuit is counted.
