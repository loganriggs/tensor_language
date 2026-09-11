# Mathematical review — 11 September 07:51 UTC

Weights remain the discovery object. FineWeb is the subsequent validation
corpus; Pile is a separate corpus-shift test. The immediate decision is to
finish the shared-reader comparisons and distinguish reader error from the
restriction that they retain the original output matrix. More activation
collection does not answer either question. The full circuit goal remains
prediction, extraction, selective removal, and composition/reuse.

The structured-transform continuation has finished: coefficient capture rose
to **3.5097% and 3.3098%**, but both starts remain unconverged. This is not a
negative about the existence of structure. Full-rank orthogonal reader MSP is
now running; the covariance/Tyler oblique comparison is queued. No third
identical structured continuation was submitted.

## Object, metric, and assumptions

For normalized MLP17 input x, native readers l_j,r_j, output matrix D, and
unembedding U, the quadratic branch is

$$
q(x)=D\phi(x),\quad \phi_j(x)=(l_j^\top x)(r_j^\top x),\qquad
T_v=\sum_{j=1}^{4608}(UD)_{vj}\operatorname{sym}(l_jr_j^\top).
$$

Here x has1152 coordinates, D is1152 by4608, and U is50304 by1152.
The tensor is symmetric in its input indices, degree two, and even under
x becoming -x. Scale transfers between readers and writers, paired sign
changes, and product permutations are gauge freedoms. Evenness is an
architectural fact, not a guarantee of semantic antipodal clusters.

Discovery minimizes coefficient Frobenius error. It does not fit an input
distribution. The explicit50304 by1152 by1152 tensor would require about
267GB in FP32; Gram contractions avoid allocating it. U is folded into the
metric M=U^T U. Native residual, bias, RMS, and final30tanh are retained;
this polynomial is not the entire model. Source/position-dependent attention
and its normalization remain outside this two-module object.

## Exact conditional consequence: reuse the existing writer solver

Let candidate readers define products psi_k. In the coefficient inner product,
write

$$
G_{kh}=\langle\psi_k,\psi_h\rangle,\quad
C_{jk}=\langle\phi_j,\psi_k\rangle,\quad B=DC.
$$

The existing `joint_quadratic_fit_v1.optimal_writers` computes

$$
W_*=BG^\dagger.
$$

This is already our conditional least-squares method, not a newly invented
factorizer. A useful explicit consequence is that the same unrestricted W_*
minimizes the coefficient error for **every positive semidefinite output
metric M**, including U^T U. The residual is orthogonal to every candidate
product: W_*G=B. This holds even when G is singular because a linear dependence
among candidate functions also annihilates the corresponding columns of B.
Therefore, for any writer perturbation E,

$$
\mathcal L_M(W_*+E)-\mathcal L_M(W_*)
=\operatorname{tr}(MEGE^\top)\ge0.
$$

This does not remove U from reader optimization: M still weights the residual
and changes which reader family is preferred. Writer sparsity, fixed writers,
rank constraints, penalties, and numerical rank truncation require separate
treatment; the unrestricted conclusion cannot be silently applied to them.

This maps directly to **variable projection**: eliminate linear output
parameters exactly while optimizing nonlinear readers. Its conditional solve
is global; the remaining nonlinear search has no generic global guarantee.
[O'Leary and Rust](https://www.cs.umd.edu/users/oleary/software/varpro.pdf).
At K candidate products and J native products, dense Gram construction costs
O((K^2+JK)d), storage O(K^2+JK), and a generic pseudoinverse O(K^3).
At K=4608, conditioning and cubic solve cost merit measurement; this is not
automatically cheap enough for every reader iteration.

The executed [CPU control](CONDITIONAL_WRITER_METRIC_V1_CONTROL.json) reused
that solver and independently formed dense symmetric coefficient tensors.
It covered full/singular output metrics and duplicate candidate products.
All registered predictions held: maximum normal-equation residual2.33e-15,
explicit/implicit error4.30e-16, and the nonnegative perturbation identity
held for80 random changes. This is an algebra/numerics control, not native
improvement. Applying it to the new dictionary artifacts remains pending.

## Alternative mappings and what their guarantees require

**Complete sparse dictionary learning.** Native normalized L/R weight vectors
are the observations; the unknown dictionary is a shared input feature map.
MSP maximizes their fourth-power coordinates under an orthogonal transform.
Each update uses dense matrix products and a polar SVD, approximately
O(nd^2+d^3). Published recovery statements assume a random sparse generative
model; local algorithmic convergence does not establish that trained readers
satisfy it. Our paired3072/1536 product split tests generalization to unused
weight vectors, not unused text. Planted sparse and dense-null controls were
executed before the native job.
[Zhai et al.](https://www.jmlr.org/papers/volume21/19-755/19-755.pdf).

Orthogonality is a real restriction. The queued ordinary-covariance and Tyler
shape preprocessing allow oblique dictionaries, with fixed-support least
squares after selecting128 coordinates per reader. Whitening still assumes
useful second/angular-moment geometry; top-coordinate support selection is a
heuristic. Native reader weights need not be independent, exchangeable sparse
samples. A complete dictionary also excludes overcomplete features. Retaining
native product pairings and Down limits the family further.

**Block terms and tensor networks.** Our overlapping symmetric cores map to
coupled block-term decompositions, but a theorem about rank-(1,L,L) terms is
not automatically a uniqueness theorem for these tied, penalized blocks.
Generic identifiability conditions have not been verified on this model.
[Domanov and De Lathauwer](https://arxiv.org/abs/1808.02423).
Tensor trains/hierarchical formats instead assume useful separations after
choosing index groupings. Fixed mixed-radix stages are one executable
arithmetic-program hypothesis, with full rank permitted; their wiring and
nonconvex optimization remain restrictions. A matrix unfolding rank or the
rank of the native product Gram does not lower-bound all reusable arithmetic
programs; our saved constructive counterexample already disproves that move.
Weighted-automaton/Hankel realization would need a finite linear state closure
for the composed routing/RMS dynamics, which is not established here.

## Consequence for circuits and next decision

Dictionary atoms provide explicit shared inputs and an executable feature
removal/interchange interface. Weight sparsity alone does not identify their
meaning or demonstrate selectivity. Compare frozen original-Down and optimal-
Down versions before rejecting a reader representation on folded error. They
have the same dense output parameter count, but different output weights.
The current sparse program stores7,815,168 matrix coefficients plus1,179,648
support indices and bias; U/background remain charged. This is much larger
than the276,480-coefficient structured bank, so raw capture is not a fair
matched-budget ranking. Sparse execution is implemented; runtime speedup is
unmeasured.

The next discriminating evidence is held-out **weight** sparsity and exact
folded error from the managed dictionary jobs. The conditional writer control
is complete and available for their interpretation. Freeze candidates before
FineWeb behavioral/intervention validation. No new data-guided discovery,
duplicate GPU run, or change to frozen queued sources is warranted.
