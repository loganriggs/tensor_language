# The current graph has a native-coordinate capacity limit

21 September 2026, 10:10 UTC. **The 399-product graph cannot achieve the recent 7–9% coefficient errors under the native isotropic metric: a necessary input-span bound puts its error at least at 27.81%.** Those smaller errors use calibration-shaped coordinates. This explains a concrete representational restriction; it does not establish that the trained 59.8% native error is optimal.

**Where the bound comes from**

The graph uses 367 shared products of two learned linear projections and 32 private squares. Its quadratic outputs are sums of matrices of the form

$$
\frac12(\ell_k r_k^\top+r_k\ell_k^\top)
\quad\text{and}\quad v_kv_k^\top.
$$

Every output's column space therefore lies in a common reader span of dimension at most

$$
r=2(367)+32=766.
$$

Let $T_o$ be the six native quadratic matrices, scaled so each pair has unit squared Frobenius norm, and set

$$
S=\sum_o T_oT_o^\top.
$$

For any rank-$r$ orthogonal projector $P$ containing the student's reader span, $(I-P)\widehat T_o=0$. Hence

$$
\sum_o\|T_o-\widehat T_o\|_F^2
\geq\sum_o\|(I-P)T_o\|_F^2
\geq\sum_{i>r}\lambda_i(S),
$$

where eigenvalues are ordered from largest to smallest. Dividing by total teacher energy and taking a square root gives the reported necessary bound. It is independent of the optimizer and chosen feature directions.

| Graph topology | Maximum common reader span | Necessary native error from this bound |
|---|---:|---:|
| 383 shared mixed products | 766 | 27.81% |
| 367 shared mixed products + 32 private squares | 766 | 27.81% |
| 512 shared mixed products | 1,024 | 12.67% |
| 560 shared mixed products + 32 private squares | 1,152 | 0% from this bound |

A zero bound is not evidence of an exact solution: the product-sharing constraints still apply. Likewise, the 512-shared-product row is a hypothetical global topology, not the earlier partial-sharing 512-product program. The bound is specific to these quadratic dictionaries and does not constrain arbitrary deeper arithmetic DAGs.

**The resulting comparison**

We retain the original 367+32 graph and add the 560+32 graph, comparing two explicitly different coefficient metrics. In native coordinates $Q$, the isotropic metric treats all matrix coefficients equally. The covariance-shaped metric compares $S_c Q S_c$, where $S_c$ is the square root of the calibration second moment about the fixed mean. Each metric is normalized separately within each of the three output pairs.

$$
E_\alpha=\alpha E_{\mathrm{native}}+(1-\alpha)E_{\mathrm{calibration}}.
$$

The registered comparison uses $\alpha=0,0.5,1$ at both widths. Each has an inherited initialization (with random extra columns at the larger width) and a fully random native-coordinate initialization. Adam updates input directions for 1,500 cosine-scheduled steps at initial rate 0.01; exact constrained readout solves run at every step. Winners are chosen by their fitting objective. There are 12 fits and no native model forwards in this fitting job.

| Program | Products | Floating coefficients |
|---|---:|---:|
| Narrow | 399 | 896,198 |
| Wider | 592 | 1,342,028 |

The primary is the wider graph at $\alpha=0.5$. Its registered requirements retain all three opened component limits, require at least 20% lower native coefficient error than the equally continued same-width covariance-only control, and allow at most 10% deterioration in calibration-shaped coefficient error. Both norms are reported for every arm.

The wider graph costs more than the existing separate and partial-sharing baselines. Passing these diagnostic requirements would therefore not establish a fair-cost improvement or circuit adoption; it would justify a separately frozen behavioral and cost-frontier comparison.

**Checks and limits**

Five seeded small cases compare implicit and dense loss, direction gradients and derivatives through the exact output solve; discrepancies are below $3\times10^{-16}$. Native-width checks cover both graph sizes, exporter slicing, private-branch storage, finite gradients and dense-loss replay. The inputs and later state remain supplied by the original model; no semantic, OOD or extraction claim follows from these checks.

This comparison follows the measured failure of the current dictionary under native coefficient geometry. It does not repeat output-only fitting, which was already shown unable to repair that error. The broader aim remains simpler reusable circuits; rank and coefficient accuracy are diagnostics toward that aim, not substitutes for behavioral evidence.

Evidence: [spectral bounds](../../direct_tensor_match/SOURCE_CAPACITY_BOUNDS_V1.json), [registered comparison](../../direct_tensor_match/DUAL_GEOMETRY_SOURCE_PLAN_V1.json), [kernel checks](../../direct_tensor_match/DUAL_GEOMETRY_SOURCE_PREFLIGHT_V1.json), and [native-width checks](../../direct_tensor_match/DUAL_GEOMETRY_NATIVE_PREFLIGHT_V1.json). This report records the design, not fitting outcomes.
