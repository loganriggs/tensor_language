# Many small full-quadratic input frames: native protocol

Motivation: exact conditional four-output block updates have settled but the
fit remains weak. Conditional spectra show modest additional output structure
in the same input spans. Test a different allocation of input versus output
capacity at slightly lower coefficient cost, without text-guided discovery.

23 overlapping orthonormal input frames E_g, each1152x4. Every frame supplies
all10 symmetric quadratic monomials of its four coordinates, with independent
1152-dimensional output writers. The core basis is fixed, generated analytically;
there are no trained core parameters. Cross-block overlap remains unrestricted.

$$
q(x)=\sum_{g=1}^{23} W_g\,\operatorname{svec}
\left[(E_g^\top x)(E_g^\top x)^\top\right].
$$

svec uses diagonal coordinates and sqrt(2) times upper off-diagonal entries,
making its Euclidean norm equal the symmetric matrix Frobenius norm. Applying
the complete unembedding U gives the output coefficient tensor to fit.
Fixed-frame writer solves are exact. Learn frames on the Grassmann manifold:
rotating one orthonormal frame only changes coordinates within its full core.

Price105984reader+264960writer=370944learned coefficients, compared with377344
in the old16x16x4block model. There are230quadratic features instead of64.
Unembedding, biases, normalization and residual/background computations remain
native and charged. These prices do not count native background as removed.

Objective is full-U coefficient residual plus .01 times the sum of squared
Frobenius norms of the BLOCK functions. With the complete orthonormal core,
this equals .01 sum_g||UW_g||_F², giving conditional regularized Gram G+.01I.
It is invariant under within-frame rotations. It is DIFFERENT from the old
four-output factor-energy penalty after writer/core gauge minimization.
Report raw reconstruction and each penalty separately; do not compare their
penalized loss values as the same objective or claim a controlled solver-only gain.

Two independent native initializations, seeds0and937. For each, sample46distinct
native product indices with probability proportional to their full-U individual
product coefficient energy. Pair them in draw order; QR their four Left/Right
reader vectors to initialize each frame. Record actual sampled indices and
reject any rank-deficient initialization before fitting. This uses only weights.
Both starts get up to300fit seconds initially; a limit is not convergence.
Use the exact-Hessian Grassmann trust-region solver that passed the corrected
independent-start control, not the CG version that failed seed1902.

Pred_a: conditional writer solve, frame orthogonality, independent loss replay
and native Hessian finite-difference checks <=1e-6 relative (loss replay <=1e-8).
Pred_b: each start has tangent gradient norm<=1e-7 and relative stationarity
||grad||*sqrt(92)/capture<=1e-4. Save the state if either fails.
Pred_c: both starts capture >=1.05 times .08634383041327387 in the full-U
coefficient metric. This is an exploratory representation comparison, not a
matched-regularizer superiority claim. Also report fitted-function similarity
between starts; a capture gain without stable functions is not identification.

Null: smaller input spans discard useful interactions, or fitting remains
unfinished. Do not reject the family from two unconverged starts. Red-team
initialization, solver behavior, penalty and capacity before a structural claim.
Circuit target is shared-read/multiple-write grouping and stable executable
units; this screen alone does not establish OOD, extraction, removal or reuse.

Implementation status: core and exact-curvature solver implemented; dense,
gauge and finite-gradient controls pass. Original toy seed544 accidentally
repeated the planted initial span. Independent starts exposed a real CG failure
(seed1902); exact-curvature TR recovered all three independent spans, projector
errors<=2.22e-8 and gradient norms<=1e-7. Both failures/corrections are preserved.
Native wrapper, native Hessian preflight and managed submission remain pending.
