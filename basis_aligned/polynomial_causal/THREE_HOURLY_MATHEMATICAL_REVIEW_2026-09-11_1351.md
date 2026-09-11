# Mathematical review — 11 September13:51, recorded13:54 UTC

The target remains a simpler executable circuit decomposition with OOD prediction, extraction, selective manipulation and composition/reuse. The user now explicitly asks whether output-sharing LL1 is a better interpretation of the desired factors. The preceding shared-input group work does not answer that question. Its native pilot remains draft/unbound/notqueued. The [requested appendix](explanations/for_logan/research_update_2026-09-11_1327.md#factorization-explained) explains the distinct objects; this review records the executable consequence.

## Object, modes and assumptions

The local folded branch is the real symmetric quadratic tensor T_vij=sum_k(UD)_vk sym(L_k R_k^T)_ij, with dimensions50304x1152x1152 and4608native products. Output metric U^T U permits exact isometric computation in1152coordinates. Bias is separate; residual addition, native RMS and final tanh mean the whole network is not this quadratic. Input indices act on the same normalized vector; antisymmetric coefficients are functionally invisible and must not be optimized as new structure. Factors are signed; nonnegative models would impose an unsupported assumption.

Output-sharing symmetric LL1 is T=sum_g c_g outer Q_g, symmetric rank(Q_g)<=L_g. Axis ranks are bounded by(1,L_g,L_g), or standard(L_g,L_g,1) with output last. Write Q_g=A_g H_g A_g^T, H_g symmetric indefinite. Groups may overlap. Literal storage includes dL_g input basis, a symmetric L_g core, and doutput coefficients in isometric coordinates pergroup; alternative signed eigencoordinates replace the core with L_g eigenvalues. A small number of output groups can hide a large sum of input ranks. General symmetric two-reader products can have input rank2, so product count is not LL1 matrix rank.

Gauge freedoms include rescaling between c and Q and changing input coordinates within a group. Global decomposition is nonconvex and can be unstable even when exact conditional solves exist. The new shared-input family instead reuses one reader across multiple output directions; after symmetrization its rank bounds are(r,r+1,r+1), with extra tied structure. It must not be called output-sharing LL1.

## Literature mapping and limits

[Tensorlab LL1](https://tensorlab.net/doc/ll1.html) defines a vector times a low-rank matrix, documents generalized-eigenvalue initialization, nonlinear least squares and refinement. This exactly matches the *unsymmetric* LL1 shape after permuting the output axis. Our same-input symmetry needs a symmetric parameterization or explicit symmetrization with correctly charged ranks. No generic uniqueness guarantee or GEVD recovery assumptions have been verified for these overcomplete learned weights. A library's optimizer is not a global certificate.

[Rontogiannis, Kofidis and Giampouras](https://arxiv.org/abs/2002.09759) address block-count and rank selection through hierarchical sparsity and iteratively reweighted least squares. This is a serious candidate for heterogeneous block sizes; its precise native symmetry/implicit-contraction adaptation remains unimplemented and its reported recovery is not evidence for this model. Prior CP/Tucker, graph-width, arithmetic-circuit and Hankel comparisons remain in the [10:51review](THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-11_1051.md). In particular, efficient tensor contraction does not discover a semantic graph, and the normalized whole network has no established finite weighted-automaton realization to which the earlier Hankel guarantees apply. This cycle changes the grouping object rather than relabelling those routes.

## Executable consequence: exact symmetric LL1 conditional updates

Given a residual E and fixed nonzero output vector c, minimizing ||E-c outer Q||² over symmetric rank(Q)<=L reduces to

$$
Q_* = \operatorname{bestSymRank}_L\left(\frac{\sum_o c_oE_o}{\|c\|^2}\right).
$$

Diagonalize the symmetric matrix and retain the L eigenvalues of largest absolute magnitude, retaining their signs. For fixed nonzero Q, the best output vector is

$$
c_o=\frac{\langle E_o,Q\rangle_F}{\|Q\|_F^2}.
$$

These are global optima of the individual subproblems, not of the full block decomposition. Zero-vector/matrix cases require explicit handling before a native controller; current controls use nonzero states. Forming the contracted matrix from an implicit product residual costs O(n*d²) in a direct implementation and O(d²) working storage; full eigendecomposition costs O(d³). Multiple groups introduce residual updates and nonconvex basin choices. No vocabulary-sized dense tensor is needed; the current demonstrator uses small dense E solely for independent checking.

The new [conditional tool](symmetric_ll1_conditional_v1.py) has [executed controls](SYMMETRIC_LL1_CONDITIONAL_V1_CONTROL.json): coefficient-gain identity2.32e-17, executor2.22e-16, and two overlapping signed-rank2outputblocks recover from nearby initializations to4.63e-13relativeerror in10alternations, monotonically. This is the first concrete consequence of the user's output-sharing clarification during this cycle, not a new native circuit. Previous shared-input random-start and cancellation failures warn against relying on this easy planted case for global recovery.

## Decision

A native output-sharing LL1 comparison is distinct and potentially high-information: it asks whether multiple input operations implement one output variable. Preserve the prepared shared-input cost pilot as draft; compare its intended information against a symmetric LL1 initialization/native conditional pilot before queueing sustained work. Reuse CP contractions and the new exact scalar projection rather than allocate the full tensor or use text for discovery. Next implementation must add robust zero-case handling and implicit native residual contractions, then assess multiple initializations and heterogeneous ranks. Candidate groups still need frozen functional and intervention tests to progress from factorization to circuits.

Concrete continuation already executed: the symmetric LL1 conditional CPU controls above. Current research state and requested browser brief explicitly distinguish completed native dictionary work, planted shared-input tests, and unrun LL1/native pilots. Next math review16:51; hourly14:22. No program-level success or global recoverability claim.
