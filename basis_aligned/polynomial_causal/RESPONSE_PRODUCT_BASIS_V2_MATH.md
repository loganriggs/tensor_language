# Exact compression of the generated two-layer interaction: ten products become six

13 September 2026. The previous four-vector response and ten-product construction were valid upper bounds. Expanding their normalization coefficients exposes a linear dependence. Removing it gives **three response vectors, two mixed-response vectors, and six downstream symmetric products**, with no approximation or optimization.

This is an improvement to setting 1's actual producer-dependent response family. It does not contradict the unfavorable compression bounds for arbitrary independent input vectors.

## Derivation from the existing response

For a fixed pristine pre-MLP9 state $z\in\mathbb R^{1152}$ and writer $w$, remove amplitude $a$ along $w$. Let $m_0$ be the pristine bias-free normalized MLP9 output, and let $J=J_w$ be the already compiled fixed-writer mixed map. Define

$$
\rho_0=\frac{\|z\|^2}{d}+\epsilon,\quad
\beta=\frac{\langle z,w\rangle}{d},\quad
\gamma=\frac{\|w\|^2}{d},\quad
\rho_a=\rho_0-2\beta a+\gamma a^2,
\qquad d=1152.
$$

The exact residual-plus-MLP response is

$$
\Delta(a)=-aw+
\frac{2\beta a-\gamma a^2}{\rho_a}m_0
-\frac{a}{\rho_a}Jz+
\frac{a^2}{2\rho_a}Jw.
$$

The coefficient of $m_0$ is a linear combination of the coefficients of $Jz$ and $Jw$. Therefore define two combined vectors

$$
p=Jz-2\beta m_0,\qquad q=Jw-2\gamma m_0.
$$

Then

$$
\boxed{\Delta(a)=-aw-\frac{a}{\rho_a}p+
\frac{a^2}{2\rho_a}q.}
$$

All normalization dependence remains exact. Implementation retains the previously validated nonnegative expression for the denominator using the component of $z$ perpendicular to $w$; it does not evaluate a cancellation-prone quadratic merely because the derivation uses one.

For two edits on the same pristine context, the linear writer term cancels from their mixed response:

$$
\Delta(a+b)-\Delta(a)-\Delta(b)
=
\left[-\frac{a+b}{\rho_{a+b}}+\frac{a}{\rho_a}+\frac{b}{\rho_b}\right]p
+
\left[\frac{(a+b)^2}{2\rho_{a+b}}-\frac{a^2}{2\rho_a}-\frac{b^2}{2\rho_b}\right]q.
$$

Thus only two context-dependent directions are needed for this local mixed response. This is not a claim that two global model circuits explain the downstream behavior.

## Folding through the next bilinear multiplication

Let $\lambda$ be the learned block10 residual re-entry coefficient, and set

$$
(v_0,v_1,v_2)=\lambda(w,p,q),\qquad
u(a)=\left(-a,-a/\rho_a,a^2/(2\rho_a)\right).
$$

The generated residual change is $\lambda\Delta(a)=\sum_i u_i(a)v_i$. For the symmetric MLP10 bilinear cross operator

$$
K(x,y)=D_{10}[(L_{10}x)\odot(R_{10}y)+(L_{10}y)\odot(R_{10}x)],
$$

prepare $P_{ij}=K(v_i,v_j)$ for $0\le i\le j<3$. There are six such vectors. The response-product numerator is

$$
K(\lambda\Delta(a),\lambda\Delta(b))
=\sum_i u_i(a)u_i(b)P_{ii}
+\sum_{i<j}[u_i(a)u_j(b)+u_j(a)u_i(b)]P_{ij}.
$$

Both branches reuse the same three vector functions and the same six product vectors. No extra factor of two is applied to diagonal coefficients: $K(v_i,v_i)$ already includes the two symmetric contributions.

## Executed evidence and price

Actual checkpoint weights, four synthetic pristine contexts, and all 121 pairs of eleven signed strengths including zero:

- Response versus original exact implementation: relative error 1.37e-16.
- Six-product numerator versus direct MLP10 multiplication: 4.02e-15.
- Two-direction mixed response versus original implementation: 1.97e-16.
- Either edit zero: exactly zero product.
- FP32 combination of rounded prepared product vectors and coefficients versus FP64 reference: 6.31e-8. This last check does not include preparing the entire basis in FP32 or running a native suffix.

The product bank shrinks from 11,520 to 6912 scalars per context, a 40% reduction. The response basis shrinks from 4608 to 3456 scalars. These are working-state savings; they do not eliminate stored native L/R/D weights or the compiled J map.

A fair vectorized CPU comparison projects all basis vectors through L/R once and applies D to all required pair products. On four contexts and two CPU threads, median product preparation falls from 10.38 ms to 7.79 ms (about 25% faster). With coefficients prepared for both representations, the 121-pair combination grid takes 3.68 ms for ten products and 3.60 ms for six. This small execution difference is not a whole-model speedup. New basis assembly/background preparation is outside these product-preparation timings; a one-pair query can still be cheaper without preparing a bank at all.

## Scope and next use

This is genuine algebraic reuse across edit strengths in one fixed context, not merely equal weights evaluated on incompatible inputs. Each new pristine context requires a new bank. The six functions are not proven minimal and are not six identified semantic mechanisms.

The residual/attention mixed terms remain important in the native circuit; this identity covers only the residual/residual numerator. Attention responses change with amplitude, so treating them as constants would invalidate an amplitude sweep. Native joint MLP10 normalization and the background/suffix remain supplied. Existing native-fold evidence validates the original formula, but native validation of this rewritten preparation/execution remains pending.

Compared with the small gains from coordinate sparsity, the useful lesson is concrete: respecting the producer's rational coefficient relations removes computations exactly. The next extension should carry this response interface into the mixed attention terms while keeping their changing inputs explicit.

[Executable basis](response_product_basis_v2.py) · [Actual-weight control](RESPONSE_PRODUCT_BASIS_V2_CONTROL.json) · [Reproducible test](check_response_product_basis_v2.py) · [Previous upper-bound derivation](RESPONSE_PRODUCT_BASIS_V1_MATH.md).
