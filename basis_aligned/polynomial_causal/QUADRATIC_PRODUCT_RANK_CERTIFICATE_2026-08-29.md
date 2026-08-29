# Family-F quadratic product-rank certificate

**Date:** 2026-08-29  
**Program artifact:** `block3_consequence_family_f_v1_programs.pt`  
**Artifact SHA256:** `d4af5bfbae03f8df9be8127e2e06c6f1a66b189be180ce72e5c74b6c7ac7a038`

## Result in plain language

The saved Family-F programs with 256 or 512 selected multiplication gates cannot be
rewritten **exactly** with fewer than 256 or 512 scalar products, respectively, if the
replacement remains a single vector-valued quadratic layer of the same form. This is
not an empirical reconstruction score: it is a lower bound that meets the product
count already displayed by the program.

This closes one tempting but narrow simplification route. An exact CP-like
refactorization of these particular quadratic maps cannot silently merge their chosen
gates. It does **not** say that approximate, deeper, finite-domain, or downstream-
equivalent programs need this many products.

## Object being certified

Each saved program has the form

$$
q(x)=b+\sum_{g=1}^{K}d_g\,(\ell_g^\top x)(r_g^\top x),
$$

where $x\in\mathbb R^{1152}$ is the normalized residual-stream input,
$\ell_g,r_g\in\mathbb R^{1152}$ are the two linear factors for gate $g$, and
$d_g\in\mathbb R^{1152}$ is the decoder column that writes that product back to the
residual stream. The bias $b$ is affine and free of multiplication gates, so it does
not change quadratic product rank.

Because the same vector $x$ enters both slots, the observable quadratic coefficient is
not $\ell_g\otimes r_g$ but its symmetric part

$$
s_g=\operatorname{sym}(\ell_g\otimes r_g)
=\frac{\ell_g\otimes r_g+r_g\otimes\ell_g}{2}.
$$

This distinction matters: certifying the rank of an arbitrary bilinear extension
would not certify the polynomial actually computed on the diagonal $(x,x)$.

## The certificate

Put the decoder columns into $D=[d_1\;\cdots\;d_K]$ and the vectorized symmetric
products into $S=[\operatorname{vec}(s_1)\;\cdots\;\operatorname{vec}(s_K)]$.
An output-mode unfolding of the order-three partially symmetric coefficient tensor is

$$
M=D S^\top.
$$

If both $D$ and $S$ have column rank $K$, then $M$ has matrix rank $K$. Every
$J$-product representation of the same quadratic map produces an unfolding of rank at
most $J$, so $J\ge K$. The saved representation already supplies $K$ products, hence
the minimum is exactly $K$.

The large $S$ matrix never has to be materialized. Its Gram matrix is

$$
(S^\top S)_{gh}
=\frac12\left[
(\ell_g^\top\ell_h)(r_g^\top r_h)
+(\ell_g^\top r_h)(r_g^\top\ell_h)
\right].
$$

Positive definiteness of $D^\top D$ and $S^\top S$ therefore certifies both required
column ranks. The implementation converts the serialized float32 coefficients exactly
to float64 and combines standard matrix-product forward-error bounds, Weyl's
inequality, and a conservative eigensolver allowance. The numerical margins are over
$10^8$, so the conclusion is not near a floating-point threshold. This is a robust
standard-model numerical certificate, not an interval-arithmetic or formally verified
BLAS proof.

## Measured results

| Saved program | $K$ | $\lambda_{\min}(D^\top D)$ | decoder margin | $\lambda_{\min}(S^\top S)$ | product margin | conclusion |
|---|---:|---:|---:|---:|---:|---|
| real F, native Down | 256 | 3.2410 | $3.15\times10^9$ | 1005.1216 | $9.56\times10^9$ | exact rank 256 |
| real F, refit Down | 256 | 5.3982 | $7.74\times10^8$ | 1005.1216 | $9.56\times10^9$ | exact rank 256 |
| real F, native Down | 512 | 1.4850 | $5.03\times10^8$ | 1004.0389 | $3.36\times10^9$ | exact rank 512 |
| real F, refit Down | 512 | 2.3296 | $2.03\times10^8$ | 1004.0389 | $3.36\times10^9$ | exact rank 512 |
| Family A, uncalibrated | 512 | 2.0397 | $1.96\times10^8$ | 866.7833 | $2.58\times10^9$ | exact rank 512 |

Here “margin” is the computed smallest eigenvalue divided by the full construction and
eigensolver error allowance. Six focused known-answer tests cover full rank, duplicate
products, dependent decoder columns, reciprocal gate scaling, left/right exchange, and
input validation.

## What this does and does not buy us

It is useful as a **certified pruning rule**: within the exact depth-two quadratic
grammar, searching for a smaller decomposition of these saved candidates is wasted
effort. It also gives a gauge-safe complexity measure: reciprocal rescaling
$\ell_g\mapsto a\ell_g$, $r_g\mapsto r_g/a$ leaves the polynomial and certificate
unchanged.

The certificate does not cover the native $K=4608$ MLP. Its decoder lives in an
1152-dimensional output space, so this unfolding cannot lower-bound its product rank
above 1152. It also does not exclude:

- an approximate lower-product program with small CE or KL;
- a deeper arithmetic circuit that reuses intermediate products;
- equality only on the token/context manifold rather than all $x\in\mathbb R^{1152}$;
- a different internal computation with the same downstream behavior;
- selective extraction or removal of a semantic circuit.

Those are exactly why the next priority is downstream reachable/observable port
reduction rather than further exact refactorization of Family F.

