# Exact normalized MLP source core

For fixed local readers Q, background h and affine source directions K, compile the native MLP write Q D[(L(h+Ka)) elementwise (R(h+Ka))]/s(h+Ka)+Q b. Here s(x)=mean(x squared)+epsilon, so normalization is retained exactly. Q is held fixed for this local operation; its downstream variation belongs to a separate chain-rule term.

Let l=Lh, r=Rh, A=LK, B=RK and C=QD. The reader-indexed numerator is N_o(a)=n0_o+n1_o dot a+a^T N2_o a, where n0=sum(C*l*r), n1=A^T(C*r)+B^T(C*l), and N2=sym(A^T diag(C) B). The shared denominator is s0+s1 dot a+a^T S2 a, with s0=mean(h squared)+epsilon, s1=2K^T h/d, S2=K^T K/d. Bias is Qb. This is a contracted third-order numerator tensor with a shared explicit normalization operation, not a fixed polynomial representation of the full model.

At zero amplitudes, gradient is n1/s0-n0*s1/s0^2. Hessian is 2N2/s0-(n1*s1^T+s1*n1^T)/s0^2-2n0*S2/s0^2+2n0*s1*s1^T/s0^3. This supplies the readout-contracted local curvature term. In a composition, downstream curvature and nonlinear source-to-local-state terms must still be included; the prior boundary experiment rejects dropping either class.

[Implementation](normalized_mlp_source_core.py) computes these contractions without forming a d-by-d quadratic matrix. Input dimensions may be full residual width while source width remains small. The numerator core is not forced sparse; no rank or sparsity identification claim is made. Future pruning must preserve named source self/cross effects, full-span intervention coverage, and collateral outputs.

[CPU test](test_normalized_mlp_source_core.py) uses independent native bilinear/RMS computation, nonzero bias, four readers, five source directions and two contexts. Exact amplitude replay error4.55e-13; gradient2.27e-13; Hessian2.05e-12. L-row/R-row inverse scaling from.01 to100 preserves compiled coefficients1.42e-13. Intentionally omitted bias causes4.63 error, a live tripwire. [Receipt](NORMALIZED_MLP_SOURCE_CORE_V1_CPU_RESULT.json).

Literal tested runtime318 scalars uses full symmetric matrices, not packed upper triangles: each context has four numerators (constant, five linear,25 quadratic values), four biases, and one shared denominator (constant,five linear,25 quadratic values). Reader/background/source generators and native weights are compiler inputs and remain chargeable. No model-wide memory or speed claim follows from the small runtime.

Next native test must use captured local backgrounds, true source sensitivities and downstream readers, verify exact local replay and analytic Hessians, then measure this local contribution against the full source Hessian. That verification is not yet done. The exact CPU compiler is a concrete folded-path implementation, not an identified or adopted circuit.

## Captured native-context validation

native_mlp_source_core_v1 now captures actual MLP11 input states, five-source sensitivities through attention11, and four downstream readouts through suffix12–17. Folded output replay4.83e-15, analytic local gradient5.83e-16 and Hessian1.35e-16 against independent dense autograd. Full-source gradient replay against the prior five-source receipt passes1e-8. The native_source_observables helper was factored into explicit attention/MLP primitives; this replay validates unchanged gradient behavior.

[CPU replay checker](check_native_mlp_source_core.py) loads only the exported prepared core, not a checkpoint or generator. One fixed text/site context has1590 full-matrix coefficients across ten token positions, artifact17,111bytes. CPU replay1.85e-15 across zero, unit-B, modal-null and all-five-unit amplitudes. It computes the local fixed-reader MLP observable on the affine input h+Ka; it is not the entire finite pre11 edit, whose attention and downstream readouts change nonlinearly. [Audit](NATIVE_MLP_SOURCE_CORE_CPU_AUDIT.json).

The scientific sufficiency gate fails: full linear source reader plus MLP11 curvature alone has worst number error53.11% and modal error8.16% of number effect on the prior unitB/nullfull/nullhalf arms. Thus an exact local compiler cannot replace missing curvature elsewhere. All fullnative source, Jacobian and adjoint generation remains charged. The exact local representation is a computational-path building block, not circuit adoption.

## Full-path continuation

For a chain of native residual/attention/MLP operations, source Hessian equals the sum of K_l^T Hessian(q_l dot F_l) K_l over nonlinear nodes, plus the final readout curvature. Here K_l is the source Jacobian before the node and q_l the downstream output adjoint after it, both evaluated at baseline. Linear residual mixing has no local curvature but changes these transports. This retains cross-layer interactions through the transported Jacobians and adjoints; local readouts must be held fixed within each term.

[source_differential_ports.py](source_differential_ports.py) implements shared capture of all seven MLP inputs/outputs, source Jacobians and downstream readers. It is newly implemented and awaits native full-path validation. [Planted chain-rule check](SOURCE_CURVATURE_CHAIN_CPU_RESULT.json) reconstructs a nonlinear chain Hessian within7.77e-16; dropping readout curvature gives0.1017 error. Next native run must reconstruct the full stored source Hessian before any attention/MLP curvature pruning claim. Existing five-source finite-amplitude tests do not cover all15 symmetric coefficient directions; exact derivative closure and finite intervention prediction remain separate requirements.
