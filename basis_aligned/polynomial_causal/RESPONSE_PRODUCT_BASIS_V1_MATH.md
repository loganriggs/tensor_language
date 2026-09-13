**13 September update:** The ten-product upper bound is sharpened exactly to six by combining normalization-dependent coefficients. [Three-vector/six-product derivation and controls](RESPONSE_PRODUCT_BASIS_V2_MATH.md). The original result below remains valid.

# A reusable conditional basis for the two-layer residual interaction

13 September 2026, CPU consequence while the native backward-fold test is queued.

The fixed-writer response through MLP9 has four vector-valued ingredients. For a fixed background, changing the removal strength changes only their scalar coefficients. Passing two such responses into MLP10 therefore needs at most ten symmetric products. This is an exact intervention-family identity, not a newly fitted decomposition or evidence of ten semantic circuits.

Let the residual width be \(d=1152\), product width \(m=4608\), and fixed writer \(w\in\mathbb R^d\). The pristine pre-MLP9 state is \(z\), and its bias-free normalized MLP output is \(m_0\). Define

$$
B_9(z)=D_9[(L_9z)\odot(R_9z)],\qquad
\rho_a=\|z-aw\|^2/d+\epsilon,
$$

$$
J_w=D_9[\operatorname{diag}(R_9w)L_9+
\operatorname{diag}(L_9w)R_9].
$$

Here \(L_9,R_9\in\mathbb R^{m\times d}\), \(D_9\in\mathbb R^{d\times m}\), and \(J_w\in\mathbb R^{d\times d}\). Removing \(aw\) before MLP9 changes its residual-plus-MLP output by

$$
\Delta(a)=-aw+(\rho_0/\rho_a-1)m_0
-\frac{a}{\rho_a}J_wz+\frac{a^2}{2\rho_a}J_ww.
$$

The four basis vectors, including the actual block10 re-entry coefficient \(\lambda\), are

$$
(v_0,v_1,v_2,v_3)=\lambda(w,m_0,J_wz,J_ww).
$$

Their coefficient vector is

$$
u(a)=(-a,\rho_0/\rho_a-1,-a/\rho_a,a^2/(2\rho_a)),
\qquad \lambda\Delta(a)=\sum_{i=0}^3u_i(a)v_i.
$$

Define the symmetric MLP10 cross-product numerator

$$
K(x,y)=D_{10}[(L_{10}x)\odot(R_{10}y)
 +(L_{10}y)\odot(R_{10}x)].
$$

Bilinearity and symmetry give

$$
K(\lambda\Delta(a),\lambda\Delta(b))
=\sum_i u_i(a)u_i(b)K(v_i,v_i)
+\sum_{i<j}[u_i(a)u_j(b)+u_j(a)u_i(b)]K(v_i,v_j).
$$

There are four diagonal and six off-diagonal vector products. The factor of two belongs inside the diagonal definition of \(K\); it must not be added again to its coefficient. Divide the resulting vector by the supplied joint MLP10 RMS denominator to obtain the residual/residual contribution used in the native experiment.

## Executed control and limits

`check_response_product_basis_v1.py` checks actual checkpoint weights on four synthetic backgrounds and all 121 pairs from eleven signed strengths, including zero. Direct response reconstruction has relative error \(1.23\times10^{-16}\); the ten-product reconstruction versus direct MLP10 multiplication has relative error \(3.96\times10^{-15}\). Either zero edit produces exactly zero interaction. The result is in `RESPONSE_PRODUCT_BASIS_V1_CONTROL.json`.

This can amortize repeated strength sweeps: prepare ten product vectors once per context, then combine them with scalar coefficients. Storing those products costs 11,520 scalars **per token/context**, before background generators and normalization. For just one pair, preparation may cost more than direct evaluation; no timing win has been measured. Ten is an upper bound, not a minimal-rank certificate.

The native input-source test already showed that residual/attention mixed products matter. This identity does not remove them. Attention responses change with edit strength and need their own generator; treating them as fixed would silently change the experiment. The basis also changes with the pristine background, so it is not a global ten-factor decomposition of the model. Its immediate value is a precise, reusable response family for future removal/composition tests, conditional on the pending native-fold result.
