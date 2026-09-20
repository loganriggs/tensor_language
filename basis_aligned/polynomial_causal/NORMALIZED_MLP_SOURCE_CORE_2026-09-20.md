# Exact normalized MLP source core

For fixed local readers Q, background h and affine source directions K, compile the native MLP write Q D[(L(h+Ka)) elementwise (R(h+Ka))]/s(h+Ka)+Q b. Here s(x)=mean(x squared)+epsilon, so normalization is retained exactly. Q is held fixed for this local operation; its downstream variation belongs to a separate chain-rule term.

Let l=Lh, r=Rh, A=LK, B=RK and C=QD. The reader-indexed numerator is N_o(a)=n0_o+n1_o dot a+a^T N2_o a, where n0=sum(C*l*r), n1=A^T(C*r)+B^T(C*l), and N2=sym(A^T diag(C) B). The shared denominator is s0+s1 dot a+a^T S2 a, with s0=mean(h squared)+epsilon, s1=2K^T h/d, S2=K^T K/d. Bias is Qb. This is a contracted third-order numerator tensor with a shared explicit normalization operation, not a fixed polynomial representation of the full model.

At zero amplitudes, gradient is n1/s0-n0*s1/s0^2. Hessian is 2N2/s0-(n1*s1^T+s1*n1^T)/s0^2-2n0*S2/s0^2+2n0*s1*s1^T/s0^3. This supplies the readout-contracted local curvature term. In a composition, downstream curvature and nonlinear source-to-local-state terms must still be included; the prior boundary experiment rejects dropping either class.

[Implementation](normalized_mlp_source_core.py) computes these contractions without forming a d-by-d quadratic matrix. Input dimensions may be full residual width while source width remains small. The numerator core is not forced sparse; no rank or sparsity identification claim is made. Future pruning must preserve named source self/cross effects, full-span intervention coverage, and collateral outputs.

[CPU test](test_normalized_mlp_source_core.py) uses independent native bilinear/RMS computation, nonzero bias, four readers, five source directions and two contexts. Exact amplitude replay error4.55e-13; gradient2.27e-13; Hessian2.05e-12. L-row/R-row inverse scaling from.01 to100 preserves compiled coefficients1.42e-13. Intentionally omitted bias causes4.63 error, a live tripwire. [Receipt](NORMALIZED_MLP_SOURCE_CORE_V1_CPU_RESULT.json).

Literal tested runtime318 scalars uses full symmetric matrices, not packed upper triangles: each context has four numerators (constant, five linear,25 quadratic values), four biases, and one shared denominator (constant,five linear,25 quadratic values). Reader/background/source generators and native weights are compiler inputs and remain chargeable. No model-wide memory or speed claim follows from the small runtime.

Next native test must use captured local backgrounds, true source sensitivities and downstream readers, verify exact local replay and analytic Hessians, then measure this local contribution against the full source Hessian. That verification is not yet done. The exact CPU compiler is a concrete folded-path implementation, not an identified or adopted circuit.
