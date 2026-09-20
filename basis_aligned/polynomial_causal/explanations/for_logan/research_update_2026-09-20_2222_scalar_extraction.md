# The fixed features now have an executable shared scalar program

2026-09-20 22:22 UTC

We folded the four canonical readouts into the frozen primary program. Each
feature now has the explicit form

$$
p_i=(a_i^Tx)(b_i^Tx),\quad i=1,\ldots,6,
\qquad h_j=(\ell_j^Tp)(r_j^Tp),\quad j=1,\ldots,4,
$$
$$
s_g=c_g+\sum_{j=1}^4 q_{gj}h_j+\sum_{i=1}^6 k_{gi}p_i,
\quad g=0,\ldots,3.
$$

The six quadratic products are reused by the quartic stage and direct output
terms. All four scalars share the same ten product nodes and13,916 learned
coefficients. A separately evaluated scalar costs13,883coefficients/10products;
four independent copies would duplicate almost all the input computation.

Native residual writing uses four fixed vectors, totaling4,608 additional
coefficients. The common-background edit is

$$
x_{17}^{\rm out}\leftarrow x_{17}^{\rm out}
-\frac{V\,\operatorname{diag}(g)\,s(x_{16}^{\rm norm})}
{\operatorname{mean}(h_{17}^2)+\epsilon}.
$$

Here g specifies removal strengths; h17 is the actual MLP17 input BEFORE its
RMSNorm. Final normalization and softcap still run afterwards. The API accepts
these states explicitly; it does not silently recompute or replace attention,
residual cross terms, or the surrounding model.

The managed replay on actual code inputs passes: extracted scalar error4.5e-8,
archived intervention-energy error1.8e-7. A subsequent CPU API check compares
against the full-vector projection and verifies zero edits and composition of
strength vectors on a fixed residual background. This proves residual-level
composition, not additive loss or behavior after nonlinear downstream operations.

These are four operational features, not semantic labels. Extracting them does
not fix previous mode1/KL failures. Nor do they reconstruct the entire primary
output: the added skip can write outside their four-dimensional output span.
This is a concrete extraction/reuse artifact with a specified interface, not
a claim that all circuit requirements are met.

[Scalar program](../../direct_tensor_match/extract_scalar_modes.py),
[intervention API](../../direct_tensor_match/scalar_interventions.py),
[native replay](../../direct_tensor_match/SCALAR_EXTRACTION_NATIVE_V1.json),
[interface check](../../direct_tensor_match/SCALAR_INTERFACE_AUDIT_V1.json).
