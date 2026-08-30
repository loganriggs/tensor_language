# Sparse routed tensor decomposition for the MLPs

**Date:** 2026-08-30 00:35 UTC  
**Question:** can an MLP be globally high-rank but use only a small, possibly
hierarchical set of tensor circuits at each token position?

## Short answer

Yes. This is a coherent and promising model class, and the repository already contains
positive discovery evidence for its flat, per-position version. The strongest existing
result is a hard-top-k **weight-action dictionary**: MLP1's globally complicated Down
action is approximated by a bank of 2,048 atoms while each token position uses only
8, 32, or 64 atoms. At 32 active atoms it recovers `0.9384` of MLP1's CE contribution,
versus `0.3101` for the matched rank-32 activation-weighted SVD. At 8 atoms it recovers
`0.8702`, while the SVD program is worse than deleting the component.

This is already a routed tensor decomposition after an algebraic fold described below.
It is not yet a complete cheap tensor program: its current implementation computes the
full 4,608-dimensional native product vector before routing, and its dense encoder and
decoder are not fully priced. The next mathematical step is to factor and route the
quadratic atom bank itself.

We have also tried hierarchy and cross-layer sparse wiring. The evidence says:

- per-position sparse dictionaries are real and useful;
- context-conditioned unions of low-rank MLP1 output circuits are real;
- a jointly trained reconstruction-anchored MLP0→MLP1 sparse wiring can preserve CE
  while reducing graph degree;
- independently trained fine/coarse SAE atoms co-activate hierarchically, but they do
  **not** form a clean geometrically nested tree;
- a single global shared trunk plus site-private residuals loses to all-private maps at
  the tested rank-512-scale prices.

So the right next object is not a fixed dense Tucker decomposition or a post-hoc tree
over SAE atoms. It is a jointly learned **sparse routed block-term tensor program**, with
the router, atoms, downstream couplings, and execution cost all explicit.

## 1. The exact tensor form

For one bilin18 MLP, with residual input $x\in\mathbb R^{1152}$,

$$
g(x)=(Lx)\odot(Rx)\in\mathbb R^{4608},
$$

$$
y(x)=b+Dg(x).
$$

Equivalently, the quadratic coefficient tensor is

$$
T_{oij}=\sum_{n=1}^{4608}D_{on}
\frac{L_{ni}R_{nj}+R_{ni}L_{nj}}{2},
$$

and $y_o(x)=b_o+\sum_{ij}T_{oij}x_ix_j$.

The existing weight-action top-k program has encoder rows $e_a$, decoder atoms $d_a$,
and computes

$$
\widehat y(x)=b'+\sum_{a\in\operatorname{TopK}(Eg(x))}
d_a\,[e_a^Tg(x)]_+.
$$

But every pre-top-k score is itself a quadratic form:

$$
e_a^Tg(x)
=\sum_n e_{an}(L_nx)(R_nx)
=x^TQ_ax,
$$

where

$$
Q_a=\frac12\left[
L^T\operatorname{diag}(e_a)R+
R^T\operatorname{diag}(e_a)L
\right].
$$

Therefore the program is a bank of quadratic tensor blocks

$$
\widehat T(x)=\sum_{a\in S(x)} d_a\otimes Q_a,
\qquad |S(x)|=k,
$$

selected separately at each token position. Globally the bank can have high rank and
many atoms; locally only $k$ blocks contribute. This is exactly the desired “many
circuits, sparsely active per datapoint” structure.

The hard `TopK` means the complete function is **piecewise quadratic**, not one global
quadratic polynomial. It is still a tensor network plus a discrete selection node. If
one insists on a single polynomial tensor network, the scores $x^TQ_ax$ can be merely
approximately sparse and all blocks remain present; exact conditional execution then
cannot be obtained without a threshold, discrete index, or other routing operation.

## 2. What has actually been run

### Flat per-position sparse action dictionaries: positive discovery

For MLP1 Down, dictionary size $P=2048$:

| active atoms $k$ | output $R^2$ | CE recovery | matched A-SVD CE recovery |
|---:|---:|---:|---:|
| 8 | 0.7643 | 0.8702 | -2.4805 |
| 32 | 0.8244 | 0.9384 | 0.3101 |
| 64 | 0.8572 | 0.9507 | 0.6926 |

Random overcomplete controls fail catastrophically, so the advantage comes from the
learned dictionary rather than merely having many atoms. Source result:
`basis_aligned/bilinear_quotient/weight_action_topk_results.json`.

At $k=8$, the same family was tested at MLP0 and MLP2--5. CE recovery was respectively
`0.9744`, `0.6290`, `0.5261`, `0.8493`, and `0.3525`; every site beat its matched A-SVD,
but the family is not uniformly near-faithful. Source:
`basis_aligned/bilinear_quotient/weight_action_multilayer_results.json`.

These are discovery experiments, not terminal certificates. They use relatively small
fit/evaluation populations, and their program prices omit the cost of replacing the
current full-gate router with a genuinely cheaper one.

### Union of context-conditioned low-rank circuits: positive discovery

MLP1 is high-rank globally but much easier to approximate with a token-cluster-specific
output subspace. At output rank 8:

- one global subspace has CE recovery `-3.1166`;
- eight real cluster-conditioned subspaces have recovery `+0.0135`;
- equal-size shuffled clusters have recovery `-1.0721`.

At rank 32 the corresponding values are `0.1384`, `0.7047`, and `-0.1185`. Increasing
the number of real clusters from 1 to 32 monotonically raises rank-8 recovery from
`-3.1166` to `0.3544`, and real assignments beat shuffled assignments at every $K>1$.
This is direct evidence for a union of low-rank functional pieces, although the current
cluster router is not yet a general context router. Sources:
`rspd_cluster_ce_results.json` and `rspd_cluster_ce_ksweep_results.json`.

### Sparse cross-layer wiring: promising but incomplete

For independently fitted MLP0-write and MLP1-read dictionaries, the weight-only coupling

$$
C=E_{\mathrm{read},1}D_{\mathrm{write},0}
$$

is exactly data-split stable, and per-token activity touches only about `0.001104` of the
possible atom-to-atom edges. The fixed coupling itself is not sparse enough: `15.954%`
of entries are strong, and its mean causal-pattern correlation is only `0.2173`; coupling
degree does not predict CE importance.

A later joint, reconstruction-anchored CE fit reduced effective incoming degree from
about `291` to `70` while keeping Down0 output $R^2\approx0.77$ and CE recovery
`0.9446`, slightly above the independently fitted `0.9375`. Removing the reconstruction
anchor collapses local fidelity. This is a real positive for jointly learning a sparse
interface, but it covers one early link and has not been composed into a whole-model
program. Sources: `weight_action_compose_results.json`, `edge_causality_results.json`,
and `real_joint_ce_v2_results.json`.

### Hierarchy/DAG evidence: mixed

Independent MLP0 dictionaries at $P=64,256,1024$ show strong activation containment:
fine-child activation raises its assigned coarse-parent activation probability by
`0.3691` above base rate, versus `-0.0070` for random parents. But the fine/coarse decoder
cosine is only `0.2698` and the coarse-in-child-span residual is `0.6751`. The registered
geometric hierarchy claim therefore fails. A subsequent co-activation group scorecard
also fails to produce stable or more causal circuit units.

This does **not** rule out a hierarchy learned jointly. Independently trained dictionaries
have permutation, rotation, splitting, and merging ambiguities, so post-hoc atom matching
is a harsh way to find a tree. It does rule out claiming that the present atoms already
form a canonical nested tensor tree. Source: `hierarchy_nesting_results.json`.

A separate 36-site shared/private low-rank hierarchy also loses to an exact-price
all-private allocation at rank-512-scale budgets. That tests a context-independent shared
output trunk, not per-token routing. It says one universal trunk is too crude, not that
conditional hierarchy is impossible. Detailed result:
`basis_aligned/polynomial_causal/HIERARCHICAL_SHARED_PRIVATE_RRR_REAL_V2_RESULT.md`.

## 3. The tensor-native hierarchical/DAG version

Give each node $v$ a low-rank quadratic block and an output dictionary factor:

$$
Q_v=U_vS_vU_v^T,
\qquad
c_v(x)=x^TQ_vx,
\qquad
w_v(x)=d_vc_v(x).
$$

A tree program selects an ancestor-closed path $S(x)$ and writes

$$
\widehat y(x)=b+\sum_{v\in S(x)}w_v(x).
$$

Parents implement coarse computations and children add residual refinements. A DAG or
overlapping-group support allows multiple parents, so a token/context can use both a
“capitalized” and a “city/entity” circuit rather than being forced into one class.
Tree-structured and overlapping-group sparse penalties provide standard optimization
machinery for constraining allowed support patterns.

For cheap execution, the router must also be hierarchical:

1. evaluate a small bank of coarse low-rank $Q_v$ scores;
2. select a few parents;
3. evaluate only their child blocks;
4. continue until the desired error/cost threshold.

This gives $O(k)$ active tensor blocks plus tree-search cost rather than evaluating all
$P$ dense atom scores. Shared $U_v$ factors within a branch turn the construction into a
hierarchical Tucker/block-term network. A learned tree is justified only if its full
stored values and executed products beat a flat dictionary at equal held-out causal
fidelity.

## 4. Why ordinary HOSVD did not already answer this

Dense HOSVD/Tucker asks for one small subspace that works for all inputs. MLP1/2's exact
folded coefficient tensors are full-rank in every measured mode, and registered dense
Tucker price points lose on storage and product count. That is compatible with sparse
routing: a union of many local low-rank pieces can have full global rank.

Hierarchical Tucker by itself also activates every core for every input. It compresses
the coefficient tensor globally; it does not create per-datapoint conditional support.
The new ingredient is structured sparsity over tensor blocks or a sparse expert router.

Relevant primary mathematics and methods:

- Jenatton et al., [Proximal Methods for Hierarchical Sparse Coding](https://www.jmlr.org/papers/v12/jenatton11a.html): rooted-tree support with an exact, near-linear-cost proximal operator.
- Mairal et al., [Convex and Network Flow Optimization for Structured Sparsity](https://www.jmlr.org/papers/v12/mairal11a.html): general overlapping groups/DAG-like allowed support patterns.
- Grasedyck, [Hierarchical Singular Value Decomposition of Tensors](https://epubs.siam.org/doi/10.1137/090764189): hierarchical Tucker representation, truncation, and approximation control.
- Rontogiannis et al., [Block-Term Tensor Decomposition: Model Selection and Computation](https://arxiv.org/abs/2002.09759): hierarchical sparsity for selecting block terms and their ranks.
- Stevens et al., [Tensor-Dictionary Learning with Deep Kruskal-Factor Analysis](https://proceedings.mlr.press/v54/stevens17a.html): each tensor-valued datapoint as a sparse sum of low-rank Kruskal tensor atoms.
- Shazeer et al., [Sparsely-Gated Mixture of Experts](https://arxiv.org/abs/1701.06538): the conditional-computation version—many global experts, few active per example.

These papers provide pieces of the construction, not a theorem that bilin18 has this
structure. In particular, standard sparse-recovery guarantees assume generative
sparsity/incoherence that has not been established for bilinear gate activations.

## 5. Highest-return next experiment

The cheapest decisive successor should start from the already positive MLP1 weight-action
dictionary rather than fit a new model from scratch.

1. **Fold the learned encoder into quadratic forms.** Construct every $Q_a$ exactly from
   $E,L,R$, and verify that $x^TQ_ax=e_a^T[(Lx)\odot(Rx)]$ numerically.
2. **Measure the atom-bank structure.** For high-usage atoms, compute data-weighted ranks,
   shared input subspaces, pairwise reuse, and whether coarse-to-fine routing scores can
   be predicted without constructing all 4,608 native gates.
3. **Oracle routing bound.** Given the frozen tensor atoms, find the best $k$ active blocks
   per held-out position. If even oracle routing cannot meet the desired fidelity/price,
   stop. The oracle-versus-current-router gap says whether routing, not representation,
   is the bottleneck.
4. **Executable flat versus tree/DAG router.** Compare a cheap flat router, a rooted-tree
   support, and overlapping-DAG support at equal total bytes and executed products. Count
   router factors, indices, biases, decoder atoms, and every evaluated quadratic product.
5. **Joint consequence objective.** Retain an MSE/interface anchor, but use the downstream
   consequence metric only if the frozen Rayleigh HELDOUT experiment validates it. Then
   test native CE, MLP0×MLP1×MLP2 composition, selective removal collateral, and OOD.

The decisive curve is not reconstruction versus $k$ alone. It is

$$
(\text{stored values},\ \text{executed products},\ \mathbb E|S(x)|)
\quad\text{versus}\quad
(\Delta CE,\ \text{interface error},\ \text{composition},\ \text{OOD/edit collateral}).
$$

If a hierarchy achieves the same right-hand side with fewer evaluated blocks or a shorter
description than the flat dictionary, then it has earned the claim of being simpler.

## 6. Addendum: tensor-similarity optimization and interaction tensors

This section incorporates Logan's proposed use of
[When Are Two Networks the Same? Tensor Similarity for Mechanistic Interpretability](https://arxiv.org/abs/2605.15183).
The paper's metric is not merely a flattened Frobenius dot product. For tensors $A,B$
and a positive-semidefinite metric operator $M$, it is the normalized inner product

$$
\operatorname{sim}_M(A,B)
=\frac{\langle A,MB\rangle}
{\lVert A\rVert_M\lVert B\rVert_M},
\qquad
\lVert A\rVert_M^2=\langle A,MA\rangle.
$$

The input legs must first be symmetrized, because permuting copies of the same input
does not change the represented polynomial. Symmetrization removes antisymmetric
coefficient pieces that cancel functionally. Hidden-unit permutations and reciprocal
rescalings disappear for a different reason: they are alternative factorizations of
the same contracted coefficient tensor. Keeping these two invariances conceptually
separate matters when regularizing factors rather than the contracted function.
The paper's Gaussian metric uses the $2n$-th input moment

$$
\Lambda=\mathbb E_{x\sim\mathcal N(0,I)}[x^{\otimes 2n}],
$$

so the metric inner product equals expected output inner product under Gaussian input.
The global tensor need not be materialized: Gram contractions compute the comparison
recursively. For one bilinear MLP, however, it is much easier and safer to work directly
with its order-three symmetric tensor $T_{oij}$.

### Similarity alone does not preserve scale

Maximizing tensor cosine alone identifies a positive scalar multiple of the target.
That is appropriate when asking whether two mechanisms point in the same functional
direction, but a layer replacement also needs the correct amplitude. A suitable
one-layer faithfulness loss is the normalized squared metric distance

$$
\mathcal L_{\mathrm{tensor}}(T,\widehat T)
=\frac{\lVert T-\widehat T\rVert_M^2}{\lVert T\rVert_M^2}
=1+\rho^2-2\rho\operatorname{sim}_M(T,\widehat T),
$$

where $\rho=\lVert\widehat T\rVert_M/\lVert T\rVert_M$. This prices both angular
misalignment and norm error. An equivalent two-term version is

$$
1-\operatorname{sim}_M(T,\widehat T)
+\eta\left(\log\frac{\lVert\widehat T\rVert_M}{\lVert T\rVert_M}\right)^2.
$$

If a global scalar multiplier is considered essentially free, optimize it analytically:

$$
\alpha^*=\frac{\langle T,M\widehat T\rangle}
{\lVert\widehat T\rVert_M^2},
$$

store that scalar, and report both the raw and scale-corrected errors. The affine bias
must also be stored exactly or included by lifting the input to $\widetilde x=(1,x)$;
otherwise high homogeneous-tensor similarity can conceal a cheap but important constant
error.

### Recommended one-layer objective

Let $\widehat T_\theta$ be a flat, tree-routed, or DAG-routed block-term tensor program.
A practical objective is

$$
\min_\theta
\lambda_T\mathcal L_{\mathrm{tensor}}(T,\widehat T_\theta)
+\lambda_{CE}\,\mathbb E_{d\in\mathrm{fit}}
   [CE_d(\widehat T_\theta)-CE_d(T)]
+\lambda_S\,\mathbb E_x|S_\theta(x)|
+\lambda_G\Omega_{\mathrm{tree/DAG}}(S_\theta)
+\lambda_P\operatorname{Price}(\theta).
$$

Adam can optimize the factors directly. The experiment should sweep or constrain these
terms to produce a Pareto curve rather than declare one post-hoc lambda canonical.
Tensor distance supplies global weight-level faithfulness; CE tells us which remaining
differences matter for prediction; support and graph penalties ask for conditional
simplicity; and `Price` prevents a sparse code with a huge dense router from receiving
false compression credit.

The structural regularizer itself can reintroduce gauge dependence. Before applying
factor-norm penalties, balance each block or use invariant quantities such as
$\lVert d_a\rVert\lVert Q_a\rVert_M$ and sparsity of the actual scores $x^TQ_ax$.
Always symmetrize $Q_a$ and compare the contracted functional tensor, not raw factor
columns. Atom labels remain permutation-invariant; tied subspaces remain rotatable.

For a hard-top-k routed program, tensor similarity applies exactly to each fixed-support
piece, while the whole map is piecewise polynomial. A high similarity between the
ungated atom banks does not certify a correct router. The CE term, held-out route tests,
and finite interventions are therefore essential rather than optional.

### Decompose the interaction, not only the components

The interaction between two replacements may be substantially simpler than either full
component. Put the four models into the same lifted tensor space and define the second
Möbius difference

$$
T^{\mathrm{int}}_{A,B}
=T_{A,B}-T_{A,0}-T_{0,B}+T_{0,0}.
$$

This cancels the baseline and both main effects, retaining only computation that requires
the joint presence of $A$ and $B$. We can then fit

$$
\widehat T^{\mathrm{int}}
=\sum_{a\in S(x)}d_a\otimes Q_a
$$

with the same normalized tensor-distance, norm, structured-sparsity, price, and CE terms.
If the interaction tensor has fewer blocks, lower hierarchical ranks, or a smaller active
support than the component tensors, that gives a direct sparse description of their
composition failure. Tensor-diff similarity from the paper is especially natural here:
candidate atoms can be compared or attributed against the interaction difference rather
than against the large common computation.

There are two distinct interaction targets worth testing:

1. **Local weight-path interaction.** Contract the MLP0 write tensor into the MLP1
   read tensor (or a specified downstream reader) and factor that contracted object.
   This is fully weight-based, cheap, and symmetry-controlled.
2. **Finite whole-suffix interaction.** Use the actual four-arm intervention difference
   in logits or CE. This captures RMSNorm, attention mixing, and compensation, but is no
   longer a single data-free bilinear tensor unless those intervening operations are
   explicitly lifted.

Residual addition is linear and has an exact copy/add tensor representation. RMSNorm can
be represented as a tensor contraction conditional on its scalar inverse norm, but the
map $x\mapsto1/\sqrt{\operatorname{mean}(x^2)+\epsilon}$ is not a finite multilinear
tensor by itself. The clean options are to keep that scalar as an explicit nonlinear
edge, condition the local tensor comparison on it, or use a certified polynomial/rational
approximation over a bounded norm range. We should not silently apply the global
multilinear tensor-similarity theorem through RMSNorm or the final tanh softcap without
making this treatment explicit.

### Concrete experiment order

1. Start with one native bilinear MLP, preferably MLP1 because its flat routed dictionary
   already has a positive baseline. Optimize the structured tensor directly under
   $\mathcal L_{\mathrm{tensor}}+\lambda_S\Omega$, with exact bias and scale correction.
2. Add a small CE term and measure whether it moves the same-price tensor-similarity/CE
   frontier outward. Retain an untouched document split because CE is data-dependent even
   though the tensor term is not.
3. Compare flat support, rooted tree support, and overlapping-DAG support at identical
   full price. Initialize from the existing folded $Q_a$ bank so the comparison tests
   structure rather than optimizer luck.
4. Separately construct the local MLP0→MLP1 interaction tensor and ask whether its best
   sparse frontier dominates the full-component frontier.
5. Only after these one-layer/local-path tests work, include explicit RMSNorm scalar edges
   and finite suffix CE. Do not begin by optimizing the full 18-layer tensor.

This order uses tensor similarity for what it certifies best—global, symmetry-invariant
functional agreement of a tractable multilinear object—while using CE and finite causal
tests for the distributional and non-multilinear parts it deliberately does not certify.

## 7. Mandatory toy-model gate for this and future mathematics

No new mathematical objective should first be debugged on bilin18. Before a method can
produce real-model evidence it must pass a small known-answer model covering:

1. a planted positive case whose true factors or functional tensor are known;
2. every claimed symmetry or gauge transformation;
3. a null case and at least one deliberately false control that must fail;
4. scale, affine bias, and any lifted constant coordinate;
5. agreement between the closed form and an independent brute-force or Monte Carlo
   computation;
6. gradient flow through the exact objective used in the real fit; and
7. any boundary where the theorem stops applying, such as hard routing or RMSNorm.

This repository already has unusually strong toys for the tensor-similarity metric in
`tensor_sim_regularized_bilinear_transcoders/`. In particular:

- `sanity_checks.py` compares the closed form with an explicit tensor contraction and
  Monte Carlo, checks permutation/rescaling/input-leg-swap invariance, includes a random
  invertible hidden mixing that **must fail**, and detects the centered-moment bug for a
  lifted constant coordinate;
- `e1_synthetic_recovery.py` tests Adam recovery of planted sparse CP factors, including
  data-subspace versus full-support OOD behavior;
- `e5_hierarchy_via_depth.py` and `e5b_hierarchy_spectrum_depth.py` plant known shallow
  and hierarchical functions and test when depth/bottleneck width can recover them; and
- `e6_pareto.py` explicitly distinguishes similarity of the dense underlying tensor from
  fidelity of the actually deployed hard-TopK routed function.

Those tests validate the inherited metric implementation; they do not establish the new
sparse-interaction hypothesis. The missing targeted toy is now
`toy_sparse_routed_interaction_tensor.py`, with pytest coverage in
`test_toy_sparse_routed_interaction_tensor.py`. It checks:

$$
E[(Lx)\odot(Rx)] = \left(x^TQ_1x,\ldots,x^TQ_Px\right)
$$

after the encoder is folded into the $Q_a$ (where the $E$ on the left denotes the
encoder matrix, not expectation); exact CP gauges; cosine's blindness to amplitude;
analytic scalar correction; exact four-arm Möbius cancellation; zero interaction as a
null; and a wrong-router control. In that control the candidate owns **exactly the same
atom bank**, so bank similarity is 1, but swaps which atom is active. Its finite output
error must remain large. This is the smallest counterexample to treating atom similarity
as routed-program faithfulness.

The runner also fits a planted rank-2 interaction with Adam using

$$
\mathcal L
=\mathcal L_{\mathrm{tensor}}
+0.1\,H\!\left(p_{\mathrm{teacher}},p_{\mathrm{candidate}}\right),
$$

where $H$ is cross-entropy against the original toy function's full softmax distribution.
Both terms have the same optimum, so failure cannot be excused as an objective conflict.
The emitted JSON receipt is a code-validation artifact only, not evidence that bilin18's
real interaction is sparse.
