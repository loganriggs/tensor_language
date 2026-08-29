# Grouped block coefficient screen v1: preregistration

Frozen before loading or decomposing the block-3 tensors on 2026-08-29.

## Question

The exact RMS polarization identity shows that attention block 3 and MLP3 form a
single typed supernode.  Before collecting activations, is the coefficient interface
from the attention write into the two MLP3 product factors already low rank?

This is a cheap screen, not a causal or behavioral result.  It can justify a larger
typed-port experiment, but it cannot earn whole-model ledger credit.

## Exact object

Let (C\in\mathbb R^{d\times d}) be attention 3's output projection, let
(L,R\in\mathbb R^{m\times d}) be MLP3's two product factors, and let
(D\in\mathbb R^{d\times m}) be its down projection.  For gate (j), the native
gauge

\[
L_j\mapsto s_j L_j,
\qquad
R_j\mapsto s_j^{-1}R_j
\]

leaves every product gate unchanged.  Choose the positive minimum-norm gauge

\[
s_j=\sqrt{\frac{\lVert R_j\rVert_2}{\lVert L_j\rVert_2}},
\qquad
\widetilde L_j=s_jL_j,
\qquad
\widetilde R_j=s_j^{-1}R_j.
\]

This minimizes

\[
\lVert s_jL_j\rVert_2^2+
\lVert s_j^{-1}R_j\rVert_2^2
\]

over the exact scale gauge.  It is not a full canonicalization: gate permutation,
sign, and any nongeneric degeneracies remain.

Weight each gate by (w_j=\lVert D_{:,j}\rVert_2), and form

\[
A=
\begin{bmatrix}
\operatorname{diag}(w)\widetilde L C\\
\operatorname{diag}(w)\widetilde R C
\end{bmatrix}.
\]

The singular spectrum of (A) is the attention-input-mode HOSVD screen after exact
per-gate norm balancing and a downstream-output importance weight.  It is not the
full cross-term tensor: it deliberately ignores cancellations between distinct gates
and the activation distribution.

## Frozen measurements and decision

Report:

1. the maximum relative mismatch between paired balanced row norms;
2. the ratio of balanced to native weighted factor norm;
3. the stable rank \(\lVert A\rVert_F^2/\lVert A\rVert_2^2\);
4. the smallest ranks containing 90%, 95%, 99%, and 99.9% of squared singular-value
   energy;
5. relative Frobenius approximation error at ranks 64, 128, 256, and 512.

The coefficient screen is **promising** iff the 95%-energy rank is at most 256.  If
it passes, extend the exact same frozen screen to blocks 4--8 and prioritize a
vector-valued typed-port response collection at block 3.  If it fails, prune raw
coefficient HOSVD as the next experiment and move directly to activation- and
downstream-consequence-weighted grouped factorization.  A failure does **not** reject
that latter possibility.

## Integrity

The result records hashes of (C,L,R,D), the resolved checkpoint blob, software
versions, elapsed CPU time, and the complete spectrum.  Synthetic tests must establish
the norm-minimizing gauge, exact product invariance, scale-gauge invariance of the
balanced screen, and exact spectrum accounting before the authoritative run.

