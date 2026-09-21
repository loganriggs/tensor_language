# Why a symmetric target can favor an asymmetric approximation

21 September 2026, 01:53 UTC. Follow-up to the [overall review](research_update_2026-09-21_0104_full_coverage_and_shared_baselines.md).

The exact folded source-dependent bilinear function is symmetric in its two already-normalized input slots. Our compact approximation is not. This is a structural discrepancy worth testing, but it does not by itself establish a code bug: the approximation was fitted with input roles that have different distributions.

## The distinction

Write the reduced-output target as

$$
F(n,m)=\sum_k c_k\big[(\ell_k^\top n)(r_k^\top m)
+(r_k^\top n)(\ell_k^\top m)\big].
$$

Then $F(n,m)=F(m,n)$. A learned approximation can instead use an ordered product dictionary:

$$
\widehat F(n,m)=\sum_j w_j(a_j^\top n)(b_j^\top m).
$$

Its symmetrization is

$$
\widehat F_{\mathrm{sym}}(n,m)
=\frac{\widehat F(n,m)+\widehat F(m,n)}{2}.
$$

For an isotropic coefficient Frobenius metric, symmetrizing cannot increase distance to a symmetric target: the discarded antisymmetric part is orthogonal to every symmetric tensor. This argument also holds with a fixed output metric, provided the input-slot metric remains exchange invariant.

It need not hold for error on ordered native input pairs. Here $n$ is a midpoint residual vector and $m$ is a specific upstream source contribution. Their distributions differ. This symmetry statement does not license physically exchanging residual sources, recomputing normalization, and claiming the whole model is unchanged.

## Native approximation audit

For the current 512-product graph, 25.99% of coefficient energy is antisymmetric under slot exchange, using isotropic input coordinates and the fixed vocabulary-centered output metric. This percentage is an energy fraction, not a relative reconstruction error.

Post-hoc averaging increases full calibration variation error from 20.00% to 28.85%. The symmetrized implementation uses 1,024 products, although its factors can share storage. Similar behavior occurs for the earlier 1,024-product graphs. A dense toy verifies the coefficient-space orthogonal-projection identity to numerical precision.

## A planted explanation, with a repair

Consider a symmetric target matrix and an asymmetric approximation:

$$
K=\begin{bmatrix}0&1\\1&0\end{bmatrix},\qquad
A=\begin{bmatrix}0&1\\0&0\end{bmatrix}.
$$

On role-specific inputs $n=(u,0)$ and $m=(0,v)$, both compute $uv$. Averaging $A$ with its transpose gives only $uv/2$: data error increases from zero to 50%, while coefficient error decreases.

But refitting a scalar output coefficient to 2 repairs the symmetric approximation exactly. This demonstrates both that the metric conflict is possible and that a failed post-hoc average does not establish failure of symmetric fitting. It does not establish that native inputs have the toy's disjoint support.

## Refit the native output coefficients

We held the learned 512 factor pairs fixed and compared ordered products against their symmetric pairs. We fitted output coefficients against the original native target, with ridge penalties anchored to the existing output map. Means and column scales were fitted on the first 24 calibration documents; the final eight supplied conditional validation.

The factors had already been learned using all 32 documents. This is a conditional refitting screen, not independent generalization evidence. No fresh native confirmation panel was used to fit or select the models.

| Dictionary and fitting | Products | Conditional validation error |
|---|---:|---:|
| Ordered, original output map | 512 | 19.24% |
| Ordered, best tested anchored refit | 512 | 19.21% |
| Symmetric, original output map | 1,024 | 28.82% |
| Symmetric, best tested anchored refit | 1,024 | 23.45% |

We tested ridge strengths 0.001, 0.01, 0.1, 1, 10 and 100 after normalizing product columns. The ordered and symmetric minima occurred at 10 and 1 respectively. Weak regularization substantially improved training fit while worsening conditional validation, another example of fitting error failing to predict transfer.

Refitting therefore repairs part of the symmetry damage, but this dictionary does not beat the cheaper ordered program. We do not promote it to native-model confirmation. This result tests fixed factor pairs plus output fitting; it does not test learning symmetric factors jointly from scratch, and establishes no expressivity lower bound.

The implication for circuit discovery is that cheap reconstruction on native input roles does not guarantee a reusable, role-independent bilinear operation. Future reuse claims need to specify their allowed input roles and test the corresponding interchange, rather than infer reuse from the target's algebraic symmetry.

Evidence under `direct_tensor_match`: `MIDPOINT_INPUT_SYMMETRY_V1.json`, `MIDPOINT_ROLE_SYMMETRY_TOY_V1.json`, and `MIDPOINT_SYMMETRY_REFIT_V1.json`. Executable analyses: `midpoint_input_symmetry_audit.py` and `midpoint_symmetry_refit.py`.
