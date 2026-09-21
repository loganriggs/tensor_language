**The preceding residual supplies the dominant carry mean**

The named-source run completed in2.70seconds; term replay2.34e-13 and energy replay1.40e-14 pass. The preceding residual contributes graph-minus-baseline mean11.9062 in natural states and12.1010 in donor hybrids. Attention17 contributes0.77285 and0.59163. Embedding-skip and MLP-bias contributions are small; numerical-closure contributions are below1e-7 in these means. These are signed descriptive terms, not evidence of semantic circuit identity or independently additive error energies.

This suggests folding the residual/MLP interaction explicitly. In the model Block.forward, the MLP receives RMS-normalized residual input. Write the pre-MLP16 state as $r_{16}=s_{16}z$ in real arithmetic. Then the carried read entering MLP17 is $\lambda_{17,0}s_{16}(A^\top z)$, and its interaction with $q_b=z^\top Q_bz$ has cubic numerator

$$
(A^\top z)(z^\top Q_bz).
$$

The RMS factors remain explicit. This identity requires a native numerical replay before using it as an exported replacement; model FP32 addition and normalization introduce small discrepancies.

**Actual successor CPU analysis**

For a symmetric quadratic matrix $Q$, the symmetric coefficient tensor of $(a^\top z)(z^\top Qz)$ has exact squared Frobenius norm

$$
\|\operatorname{Sym}(a\otimes Q)\|_F^2
=\frac{\|a\|_2^2\|Q\|_F^2+2\|Qa\|_2^2}{3}.
$$

The implementation checks this against an explicitly materialized7-dimensional tensor, then evaluates all three components and all three programs in native and covariance-shaped geometries without constructing a1152-cubed tensor. Component-three graph cubic error is53.7615% native and4.9444% covariance-shaped, versus56.7287% and5.4552% for the covariance baseline. The graph also has smaller reader-contracted error in both geometries. Therefore this additional exact homogeneous coefficient metric **does not explain the graph's worse code effects**. Switching to it alone has no evidence of being a repair.

Affine read corrections produce lower-degree terms; the complete error also depends on normalization, other residual contributions and input-distribution moments. The next useful native test is to replay the cubic branch with explicit RMS scales and measure its complete affine-corrected error, rather than assume the homogeneous coefficient score predicts transfer. No candidate or frozen baseline changed.

[Named-source results](NAMED_CARRY_ERROR_V1.json) · [Implicit cubic coefficient analysis](RESIDUAL_CUBIC_METRIC_V1.json) · [Executable analysis](audit_residual_cubic_metric.py).
