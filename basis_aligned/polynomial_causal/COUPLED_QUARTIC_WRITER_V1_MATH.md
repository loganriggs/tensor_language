# Exact coupled output solve for the composed quartic

The nested spectral baseline independently approximated each inner quadratic and then retained its old outer weight. It failed native write fidelity, despite an adequate outer approximation. This experiment asks whether jointly sharing those fixed quadratic features across both output directions can repair extraction. It does not yet move their input readers.

Let each inner feature be a symmetric quadratic with its own input directions:

$$
Q_j=B_j\operatorname{diag}(\nu_j)B_j^\top,
\qquad q_j(x)=x^\top Q_jx,
\qquad f_j(x)=q_j(x)^2.
$$

Its fully symmetric quartic coefficient tensor is

$$
(F_j)_{abcd}=\frac{(Q_j)_{ab}(Q_j)_{cd}+(Q_j)_{ac}(Q_j)_{bd}+(Q_j)_{ad}(Q_j)_{bc}}{3}.
$$

The tensor Frobenius inner product counts all ordered input slots. This is the standard coefficient inner product discussed in [Kolda and Bader's tensor review](https://www.kolda.net/publication/TensorReview.pdf); the following contraction formulas are our application to these quadratic squares, not a recovery theorem from that reference.

The exact Gram matrix is

$$
K_{ij}=\langle F_i,F_j\rangle
=\frac{\operatorname{tr}(Q_iQ_j)^2+2\operatorname{tr}(Q_iQ_jQ_iQ_j)}{3}.
$$

Evaluate the traces using the small cross-reader matrices $B_i^\top B_j$. There is no need to materialize a fourth-order tensor. Negative quadratic eigenvalues are allowed.

For target scalar quartic $T_m$ (the full two-MLP numerator folded into frozen centered-output reader $m$), the exact right-hand side is

$$
C_{jm}=\langle F_j,T_m\rangle
=\sum_{a,b}\nu_{ja}\nu_{jb}\,T_m(b_{ja},b_{ja},b_{jb},b_{jb}).
$$

This uses the existing fully symmetric native contraction oracle. Each rank16 quadratic needs256 basis-pair evaluations. Both native seeds require16,384 contractions in total. These are deterministic weight contractions, not synthetic training examples or activation fitting.

For mixing matrix $A$, the approximating scalar outputs are $\widehat T_m=\sum_j A_{jm}F_j$. The fixed physical output writers are orthonormal under the centered-unembedding metric, so the objective is

$$
\|T-\widehat T\|_F^2
=\|T\|_F^2-2\operatorname{tr}(A^\top C)+\operatorname{tr}(A^\top K A).
$$

Thus $KA=C$ gives the global linear optimum in this fixed feature span. We scale each coefficient feature to unit norm before solving and report the retained Gram rank and normal-equation residual. An eigenvalue cutoff of1e-12 relative to the largest eigenvalue handles numerical rank; if it excludes a meaningful direction, the residual gate can fail. No claim of a global optimum over the input readers follows.

The old assignment uses16 features exclusively for each of two outputs. The new solve lets all32 features serve both outputs, adding32 mixing floats:592,704 fitted floats per seed. Native input normalization, the last-layer denominator, other computational paths and final readout remain outside the candidate boundary.

[Dense controls](COUPLED_QUARTIC_WRITER_V1_CONTROL.json) compare the Gram and target contractions against explicit small tensors and recover a planted two-output mixture, all within1.8e-15. The native experiment requires: (A) normal residual at most1e-8 and nonincreasing exact coefficient objective; (B) at least50% reduction in native write error for both seeds; (C) native write error at most10% for both. All native checks use the previously reused developmental cache, not fresh/OOD data. A passed A with failed B/C means that output optimization in this fixed feature span is insufficient for native extraction. It does not prove that no other mixing would fit this cache, because cache fitting is deliberately not the objective.

Native results belong in `COUPLED_QUARTIC_WRITER_V1_RESULT.json` when execution completes. No circuit is identified by a successful linear solve alone.

## Native result and next nonlinear step

The [native exact solve](COUPLED_QUARTIC_WRITER_V1_RESULT.json) passes its numerical criterion: both Grams retain all32 directions, minimum unit-scaled eigenvalues are0.216 and0.226, and normal residuals are below1.2e-15. The coefficient objective decreases by about7.1e15 for each seed. The reported2.4% is relative to the objective with its target-norm constant omitted; it is **not** a percentage reduction in total coefficient error.

Native write errors increase from55.44/54.89% to56.91/56.31%, failing both improvement and10% fidelity bars. Poor linear-system conditioning or an unfinished output solve does not explain this failure. The fixed input features must change for this weight-objective approach to have another chance at extraction. We have not proven that arbitrary activation-fitted mixing in the same span would fail.

[Gradient controls](COUPLED_QUARTIC_GRADIENT_V1_CONTROL.json) now validate the next objective. For moving reader/eigenweight parameters $\theta$, solve the mixing $A_*(\theta)$ at every step. The envelope differential is

$$
d\mathcal J=\operatorname{tr}(A_*^\top(dK)A_*)-2\operatorname{tr}(A_*^\top dC).
$$

Derivatives of the optimal mixing cancel by the normal equations. On a nontrivial random symmetric target, envelope and fully differentiated solves agree within4.1e-16; finite-difference errors for readers and inner eigenweights are below1.4e-10. This validates the local derivative, not recovery or convergence.

The next one-seed pilot moves32 orthonormal input banks and32 unit-norm signed eigenweight vectors. These constraints remove divergent scale directions while allowing each rank16 quadratic its own subspace. A QR retraction updates input frames and normalization retracts eigenweights to their spheres. Every trial point recomputes exact coefficient contractions and the optimal two-output mixture. Armijo backtracking requires descent; the objective is divided by its fixed initial captured energy. The12-step budget is a pilot, with a separate projected-gradient threshold1e-6. A missed stationarity threshold requires further optimization or diagnosis, not an absent-structure claim. Native directional derivative checks precede fitting, and cached behavioral states are used only for fixed before/after reporting.
