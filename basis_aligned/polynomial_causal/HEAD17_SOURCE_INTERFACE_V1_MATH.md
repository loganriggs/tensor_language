# Reusing head17.2 projections for the existing regional component

13 September 2026, 03:04 UTC. Weight-only interface analysis, followed by cached native-state execution. This concerns the previously extracted shared quadratic source component; equality with the newer child/remainder mixed-write computation has not been established.

The existing full-head source-support result already gives a maximum 384-dimensional source space. The new question is whether the four readers of the old extracted component belong to that particular head's space. Head dominance in the output does not imply exact membership: the original source factors were fitted jointly across consumers.

For normalized current and first-layer source states, let

$$
s=\begin{bmatrix}x\\x_0^{\rm attn}\end{bmatrix}\in\mathbb R^{2304},
\qquad S=\begin{bmatrix}K_1&0\\K_2&0\\(1-\mu)V_{17}&\mu V_0\end{bmatrix}
\in\mathbb R^{384\times2304}.
$$

Here all matrices are the actual head2 slices, and the signed mixture coefficient is -0.0888671875. These are **raw** key and mixed-value projections, before key normalization and rotary application. The normalized score arrays used by the recent three-contraction predictor are not interchangeable with this interface.

Let R contain the four original FP64 source readers. An untruncated SVD solves

$$
C=RS^+,
\qquad E=R-CS,
\qquad Rs=C(Ss)+Es.
$$

The last expression gives an approximate interface by omitting four residual readings Es, or an exact interface by supplying them. S has condition number9.88; SVD and independent QR projection agree within4.39e-15 relative error. Reader residuals are0.334%,0.183%,0.302%,0.211%. Thus exact membership fails narrowly, without evidence of numerical ill-conditioning. The projected two cubic source features differ by0.442%/0.430% on2048 independent Gaussian probes. No text fitting was used.

## Cached native check and executable consequence

The existing COMMON_QUADRATIC_NATIVE_V1_PORTS cache contains64 FineWeb prefixes of length9. We evaluated all nine causal sources for target position8, retaining the old query writers, dual coefficients and exact four Q/K normalization scalars. Changing only the four source readers to CS produces1.251% aggregate summed-write error. Per-source write errors range0.875–1.448%. This is larger than reader error, as multiplication and cancellation can amplify it. It is not a final behavioral-effect measurement.

`shared_head_native_ports_v1.execute` now evaluates the old shared-component branch from raw query projections256, source projections384, the relative rotary matrix and the existing folded atom writers. Its projected execution agrees with the old source-input executor within2.93e-15. Adding the four Es readings recovers the original branch within2.68e-15. These checks use native saved states in FP64 arithmetic; they do not validate a new FP32 deployment.

The local executor stores69,152 scalars:1,536 reader coefficients,32 dual coefficients,12,288 key-projected atom coefficients and55,296 folded physical writers. Native projection producers remain required and must be charged. The exact variant also requires the four residual reader maps or their supplied readings. This local accounting is not a whole-model compression result.

This creates a usable interface for the next reuse test. It does not identify the old regional source block with the newer finite mixed-write circuit. A native behavioral check must compare the projected and exact-corrected old component and then test its contribution to the mixed response under the same declared background. The small weight residual is not enough to assume that comparison passes.

[Reproducible analysis](head17_source_interface_v1.py) · [Result](HEAD17_SOURCE_INTERFACE_V1_RESULT.json) · [Port executor](shared_head_native_ports_v1.py) · [Existing branch definition](COMPILED_SHARED_HEAD2_V1_MATH.md).
