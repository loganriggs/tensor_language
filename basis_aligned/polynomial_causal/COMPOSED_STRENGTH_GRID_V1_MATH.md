# Reusing the composed interaction across independently varied edit strengths

13 September 2026. The shared context now supports a native intervention-strength screen, rather than only a timing loop. All registered regional/numerical criteria pass. FineWeb CE-effect fidelity remains weak for tiny effects; the worst discrepancy is reproduced exactly by the older generator.

## What was tested

Take the fixed child scalar field $a_t$ and remainder field $b_t$ on two historical prefixes: regional row 0 and FineWeb row 96. Independently choose $\alpha,\beta\in\{0,0.5,1,1.5\}$ and remove $\alpha a_t w$ or $\beta b_t w$ at head9. Thus each prefix has 16 pair configurations, including nine pairs where neither edit is zero. Only the original unit-strength pair had been used in the preceding integration check.

One pristine context is prepared per prefix. The shared graph generates the resulting MLP9 response, both attention10 changes, pre-MLP10 changes $c(\alpha)$ and $r(\beta)$, and joint normalization. It predicts the full direct cross contribution

$$
F(\alpha,\beta)=
\frac{D_{10}[(L_{10}c(\alpha))\odot(R_{10}r(\beta))
+(L_{10}r(\beta))\odot(R_{10}c(\alpha))]}
{\operatorname{mean}[(z_0+c(\alpha)+r(\beta))^2]+\epsilon}.
$$

Native MLP9/attention10 branches at those same strengths supply an independent reference product. The output test compares the effect of adding the reference or generated product to the **same native additive post-MLP10 background**, followed by the native suffix. This is prediction of that conditional product intervention, not the entire child/remainder interaction in the original model. Normalization response terms outside this direct cross contribution remain outside the target.

## Results

All 32 configurations are retained. Every nonzero pair passes the 0.1% native product-error bar. Whenever either edit is zero, the generated product is exactly zero.

Across the nine nonzero regional pairs, target-effect relative L2 error is **0.802%**, versus the registered 5% bar; control-effect error is 1.12%. No target or control signs reverse. Maximum target/control absolute discrepancy is 5.72e-6. This extends conditional composition evidence to partial and stronger removals without fitting new coefficients.

Across the nine nonzero FineWeb pairs, newline-CE-effect relative error is **50.3%** and newline/comma margin error is 8.06%. Signs agree, but sign agreement is not quantitative fidelity. FineWeb was reported descriptively without a uniform small-effect pass criterion. Do not generalize the regional passing result to accurate FineWeb causal magnitudes. The initial two-prefix grid took 13.50 CPU seconds.

## Executed red-team: did the new compression cause the FineWeb discrepancy?

The largest absolute CE-effect error occurs at $(\alpha,\beta)=(1.5,1.5)$. Native CE effect is -6.56e-7 nats; the shared generator predicts -1.67e-6. Absolute error is 1.01e-6, or 154.5% relative on that individual cell. The state/product relative error there is only 5.68e-5, illustrating how small product discrepancies can accompany a large relative discrepancy in a tiny downstream effect.

We reran that selected case with the older uncached composed generator and physically evaluated its suffix. Its product agrees with the shared version within 1.69e-14; its measured target/control effects match exactly. This rejects the explanation that the newly shared response/projection implementation caused the discrepancy. It does **not** prove that all remaining error is harmless floating-point noise, remove the quantitative failure, or establish general precision robustness. The reference remains native FP32 execution and the conditional generator is an algebraic approximation to that numerical process.

The audit is explicitly post-result: the selection criterion and original cells remain saved, and the regional criteria were not retroactively changed. Native branch/background dependencies and historical row use are unchanged. This is an amplitude-domain screen, not language OOD, an autonomous extracted circuit or stable semantic factor identification.

[Registered test](COMPOSED_STRENGTH_GRID_V1_PREREGISTRATION.md) · [Complete 32-cell receipt](COMPOSED_STRENGTH_GRID_V1_RESULT.json) · [Worst-case selection](COMPOSED_STRENGTH_GRID_WORST_V1_SELECTION.json) · [Legacy-generator discriminator](COMPOSED_STRENGTH_GRID_LEGACY_AUDIT_V1_RESULT.json) · [Executor](composed_strength_grid_v1.py).
