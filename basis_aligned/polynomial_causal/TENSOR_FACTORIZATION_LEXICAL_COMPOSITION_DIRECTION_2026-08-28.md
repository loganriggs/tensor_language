# Tensor factorization, overlapping lexical codes, and CE composition

Date: 2026-08-28

This note responds to the proposed next directions after the MLP1 physical-gate assay.
It separates what each mathematical decomposition can actually buy.

## 1. The two factorizations are importantly different

Ignoring the separate bias, one bilinear MLP computes

$$
y_o(z)=\sum_{n=1}^{4608}D_{on}(\ell_n^\top z)(r_n^\top z).
$$

Because $z_i z_j=z_j z_i$, its function is represented by the partially symmetric
third-order tensor

$$
T_{oij}=\sum_n D_{on}\,
\frac{\ell_{ni}r_{nj}+r_{ni}\ell_{nj}}{2},
\qquad y_o(z)=\sum_{ij}T_{oij}z_i z_j.
$$

Factoring `Down` alone and factoring $T$ answer different questions.

### Factor `Down`

For $D\approx UV$, the program becomes

$$
y\approx U\bigl(Vh(z)\bigr)+b,
\qquad h_n(z)=(\ell_n^\top z)(r_n^\top z).
$$

This can reduce the storage and multiply cost of the final mixing map, but it still
computes all 4,608 native products. The existing C512 result is evidence for this
family. It is the cheap, conservative curve.

### Factor the folded third-order tensor

A CP-like refactorization seeks

$$
T\approx\sum_{a=1}^{q}c_a\otimes u_a\otimes v_a,
$$

which gives a new MLP with $q$ products. This can reduce the multiplication gates as
well as `Down`, and is therefore the more fundamental simplification. A Tucker/HOSVD
factorization instead finds mode subspaces and a core,

$$
T\approx G\times_1 U_{out}\times_2 U_{in,1}\times_3 U_{in,2}.
$$

Tucker rank is not automatically gate count: a dense core can still require many
cross-products. It is valuable as a diagnostic and preconditioner; CP, block-term, or
a sparse core is needed for a small executable gate program.

The tensor need not be materialized. Its unfolding Gram matrices and contractions can
be computed from $D,L,R$, preserving the exact folded function and avoiding a dense
$1152^3$ array.

## 2. What “minimize norm, then HOSVD” can guarantee

There are two possible meanings.

First, each checkpoint gate has a scalar gauge:

$$
(\ell_n,r_n,d_n)\mapsto(a\ell_n,b r_n,d_n/(ab)).
$$

Minimizing the sum of squared factor norms over this gauge balances the three factors.
That removes arbitrary scale and improves numerical conditioning, but leaves
permutations, sign conventions, degeneracies, and all genuinely different tensor
decompositions.

Second, geometric invariant theory defines a minimum-Frobenius-norm representative
over a larger gauge orbit. The Kempf--Ness picture gives a balanced representative;
for tensor networks, minimum canonical forms characterize gauge equivalence up to
orbit closure and are unique only up to a compact/unitary residual gauge. This is a
real canonicalization theorem, not a semantic-factor theorem. It applies only when the
chosen group action exactly represents function-preserving gauges; orbit-closure/null-
cone behavior must also be handled. See the rigorous [minimal canonical form paper](https://arxiv.org/abs/2209.14358).

Running HOSVD after balancing is sensible because HOSVD then sees a well-conditioned,
gauge-controlled tensor. HOSVD always supplies orthogonal mode subspaces and an
all-orthogonal core, derived from the SVDs of tensor unfoldings
([De Lathauwer, De Moor, and Vandewalle](https://www.math.ucdavis.edu/~saito/data/tensor/lathauwer-etal_mulilinear-SVD.pdf)).
Truncated HOSVD has approximation guarantees, while higher-order orthogonal iteration
can improve the least-squares Tucker fit
([best multilinear-rank approximation](https://epubs.siam.org/doi/abs/10.1137/S0895479898346995)).

It does **not** by itself guarantee:

- minimum CP/gate rank;
- a sparse or nearly diagonal core;
- unique axes when mode singular values are repeated or nearly repeated;
- lexical semantics;
- preservation of CE or downstream causal effects.

The right test is therefore: balance first, compute the implicit HOSVD/Tucker spectrum,
then test whether its core is compressible in an executable grammar and whether the
candidate transfers downstream. Zach's suggestion is a strong canonical first stage,
not the whole reverse engineering.

## 3. An overlapping lexical code is more plausible than disjoint clusters

The current “shared lexical + token-specific + continuous context” result was not an
SAE result. It used token means, class/group means, token residuals, and a continuous
context regression. Consequently, the class labels were disjoint by construction and
cannot express that `Paris` is simultaneously city-like, capitalized, geographic, and
name-like.

A sparse dictionary model can express that overlap:

$$
\mu_t\approx A c_t,
\qquad \lVert c_t\rVert_0\ll K,
$$

where $\mu_t$ is the cross-context mean MLP0 write for token $t$, columns of $A$ are
shared lexical atoms, and $c_t$ may activate several atoms. This is ordinary sparse
coding/SAE structure. Dictionary recovery theorems require assumptions such as sparse
latent codes, adequate excitation, incoherence/identifiability, and controlled noise;
without these, an SAE representation is one convenient basis rather than a recovered
ground truth. Tensor/dictionary methods explicitly connect sparse coding and tensor
decomposition; one theoretical example is the [sum-of-squares dictionary-learning
analysis](https://arxiv.org/abs/1407.1543).

The useful hybrid is not “SAE everything.” It is

$$
m_0(z_{t,c})
=A c_t
+B\,q(z_{t,c}-\bar z_t)
+\text{sparse token--context interactions}
+\epsilon_{t,c}.
$$

- $A c_t$ is an overlapping, sparse lexical main effect.
- $q$ is a low-rank linear/quadratic function of the context-dependent deviation from
  the token's mean input state.
- sparse interactions allow a lexical atom to change with context without giving every
  token an unconstrained table.

Identifiability requires centering constraints—for example the context term must have
zero fit-set mean within token—otherwise information can move arbitrarily between the
token and context terms. Fit documents and validation documents must remain separate.

This can be partly folded into weights. The MLP's polynomial tensor can be factorized
directly, and token-main-effect codes can be derived from the embedding/token path. The
attention-conditioned state and RMS normalization should remain explicit program nodes;
folding them into a static token table would erase the very context term being modeled.

The stronger objective is joint: choose the MLP0 dictionary so that MLP1/2 have sparse
readers in the same coordinates. That is a simultaneous factorization/shared-dictionary
problem, not an SAE trained only to reconstruct MLP0 activations.

## 4. Clarifying the MLP1 “10--17 dimensions” result

It was **not** that 10--17 documents explain the energy. For each document, randomized
categorical-Fisher probes produce a matrix of downstream derivatives with respect to
MLP1's 4,608 gates. Within one document, about 10--17 singular directions contained 95%
of that measured derivative energy.

Two independent probe halves on the same document did not recover the same top-16
subspace. Thus the evidence supports “a smooth, moderately concentrated response
spectrum,” but not “one stable global set of 16 directions.”

The proposed alternatives are entirely plausible:

- different low-dimensional subspaces for different contexts;
- a union of shared parent features plus context-specific children;
- a hierarchical Tucker/tree tensor network;
- a sparse DAG of reusable response atoms;
- simply insufficient probes around a spectrum with no sharp gap.

The current physical-gate assay tests the shared-dictionary version. A negative result
does not rule out hierarchical or context-routed low dimension; it rules out the
registered global sparse native-gate supports.

## 5. Are 32 documents enough?

They are enough only for a relatively cheap, prospective discriminator.

- 16 documents fit supports and coefficients.
- 16 untouched documents evaluate them.
- each document contributes 128 scored token positions and two independent sets of 32
  Fisher probes.
- documents, not token positions or probes, are the statistical independence unit.

Therefore the validation sample size for natural-text variation is still only 16.
The simultaneous 48-comparison bootstrap is conservative and fail-closed, but it cannot
create information absent from those documents. A pass would indicate a large stable
effect worth replication; a failure could be low power.

The production run is expected to take minutes, not hours. If it passes or is close, the
next stage should be frozen prospectively at 128--512 documents, include nested doubling
curves, and require support/metric stability when the data are doubled. Final CE claims
need much larger token counts—at least hundreds of thousands of scored tokens across
many documents and more than one domain/task distribution.

## 6. CE-only success: the strongest argument both ways

### The case for mostly caring about CE

The model was trained on CE. If a smaller **whole program** preserves held-out CE across
large, diverse samples, remains fast, and calls none of the replaced native modules,
then it is a valid behavioral compression even if its internal variables differ. MLP2
continuing to compensate for a simplified MLP0 is positive under this objective: the
compensation is part of the program's working mechanism.

This suggests end-to-end joint compilation of MLP0--2, with CE as the primary objective,
rather than forcing each replacement to match native internal states.

### The case against CE as the only criterion

Average in-distribution CE can hide rare-token damage, changed argmax decisions, OOD
failures, and fragile cancellation between modules. It also cannot support claims about
extracting or selectively removing a circuit. The rank-640 result already demonstrates
this: very small CE damage coexists with about 4% top-token disagreement.

If the objective is reverse engineering rather than compression alone, we need two
products:

1. a CE-optimized behavioral compiler, allowed to use compensating internal changes;
2. a causal abstraction whose variables support prediction under interventions and
   selective edits.

Neither should be forced to satisfy the other's unnecessary constraints, but they must
be labeled separately.

## 7. The right MLP0/1/2 composition test

Yes: independently reduce MLP0, MLP1, and MLP2, then test all eight combinations in one
shared run:

$$
NNN,\ PNN,\ NPN,\ NNP,\ PPN,\ PNP,\ NPP,\ PPP.
$$

Here $N$ is the frozen baseline component and $P$ its independently fitted reduction.
From their CE/KL and causal-response changes we can compute main effects and interaction
terms, for example

$$
I_{01}=\Delta_{01}-\Delta_0-\Delta_1.
$$

Large interactions mean independent simplicity does not compose. If `PPP` is good even
though the single replacements are imperfect, the three reductions have found a new
compatible behavioral program. If `PPP` fails while some mixed cells succeed, the cube
localizes which interface needs joint refitting.

The next improvement is then to jointly refit only the incompatible edge or pair, not
to discard all local structure. The cube should be repeated on large held-out CE,
different domains, the causal bank, and selected edit/removal tests. This is the most
direct way to decide whether MLP2's compensation is robust useful computation or a
fragile in-distribution cancellation.

## Recommended ordering

1. Complete the already frozen MLP1 physical-gate screen.
2. In parallel on CPU, implement implicit folded-tensor Gram/HOSVD spectra for MLP0 and
   MLP1 after scale-balanced canonicalization.
3. Compare `Down`-only SVD, Tucker/sparse-core, and CP/block-term curves at matched
   executable cost—not Frobenius error alone.
4. Fit an overlapping sparse lexical-main-effect dictionary plus centered continuous
   context residual; compare it to PCA, disjoint classes, and token tables at matched
   price and downstream consequence.
5. Put admitted MLP0/1/2 candidates into the eight-cell composition cube, then either
   accept a CE compiler or jointly refit the localized incompatible interface.
