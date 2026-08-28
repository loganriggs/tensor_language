# Norm minimization before HOSVD: exact toy result

Date: 2026-08-28  
Status: CPU toy executed; 4/4 tests pass  
Code: `norm_minimization_hosvd_toy.py`  
Tests: `test_norm_minimization_hosvd_toy.py`

## Short answer

Norm minimization is useful, but for a narrower reason than “it finds a lower-rank
tensor.” It removes arbitrary gauge conditioning and can delete truly dormant bond
directions. It **cannot change the exact folded tensor, its multilinear rank, or its
HOSVD spectrum**.

Therefore:

- use it before interpreting or comparing the *displayed factors*;
- use it to detect and remove unused internal bond dimensions;
- do not expect it to rescue the already-dense HOSVD spectrum of the folded MLP1 or
  MLP2 coefficient tensor;
- after balancing, test sparse dictionaries, block structure, or causal compression
  as separate hypotheses. Minimum norm does not imply any of them.

## Case 1: the bilinear MLP's scalar product-gate gauge

One product gate contributes

\[
T_g=d_g\otimes \ell_g\otimes r_g.
\]

For nonzero scalars \(\alpha,\beta\),

\[
(d_g,\ell_g,r_g)
\mapsto
\left(\frac{d_g}{\alpha\beta},\alpha\ell_g,\beta r_g\right)
\]

leaves \(T_g\) exactly unchanged. Let the three original factor norms be
\(n_d,n_\ell,n_r\), and define

\[
m=(n_dn_\ell n_r)^{1/3}.
\]

The arithmetic--geometric mean inequality gives

\[
\left\|\frac{d_g}{\alpha\beta}\right\|^2
+\|\alpha\ell_g\|^2
+\|\beta r_g\|^2
\ge 3m^2.
\]

Equality holds exactly when all three transformed norms equal \(m\). Thus the
per-gate balancing already used in the MLP1/MLP2 implicit-folded-tensor code is the
exact minimum of the factor norm over this scalar gauge.

But the folded physical tensor

\[
T=\sum_g d_g\otimes\operatorname{sym}(\ell_g\otimes r_g)
\]

does not move anywhere on this orbit. Every mode unfolding is therefore identical,
so its singular values, HOSVD energy curve, and exact multilinear ranks are
identical too. Balancing can reduce roundoff when materializing \(T\), but cannot
create mathematical compressibility.

### Executed toy

An intentionally ill-conditioned scalar gauge produced:

| quantity | before | after |
|---|---:|---:|
| displayed factor squared norm | 503,681,572.52 | 120.43 |
| balanced log-norm defect | — | \(3.63\times10^{-16}\) |
| relative folded-tensor drift | — | \(4.46\times10^{-16}\) |
| maximum relative drift of any HOSVD mode spectrum | — | \(5.19\times10^{-16}\) |

This is the strongest possible counterexample to “balancing improves folded HOSVD”:
the parameter presentation improved by over six orders of magnitude while the HOSVD
did not change beyond floating-point noise.

## Case 2: a genuine full \(GL(r)\) internal edge

Suppose two tensor-network nodes, matricized along their shared edge, are

\[
A\in\mathbb R^{m\times r},\qquad
B\in\mathbb R^{r\times n},\qquad P=AB.
\]

The gauge action is

\[
A\mapsto AG,\qquad B\mapsto G^{-1}B,qquad G\in GL(r).
\]

If

\[
P=U\Sigma V^\top,
\]

then the square-root factors

\[
A_*=U\Sigma^{1/2},\qquad B_*=\Sigma^{1/2}V^\top
\]

obey

\[
A_*^\top A_*=B_*B_*^\top=\Sigma.
\]

They attain the exact factorization minimum

\[
\min_{AB=P}\bigl(\|A\|_F^2+\|B\|_F^2\bigr)
=2\|P\|_*,
\]

because \(\|AB\|_*\le
(\|A\|_F^2+\|B\|_F^2)/2\), with equality for the square-root SVD factors.
The remaining minimum-norm ambiguity is orthogonal:

\[
(A_*Q,Q^\top B_*),\qquad Q^\top Q=I.
\]

This is the setting where “minimum norm, then HOSVD” really is a useful
canonicalization recipe. It removes nonorthogonal distortion first; HOSVD then fixes
the residual orthogonal basis when singular values are distinct. Repeated singular
values still identify only a subspace, not unique axes.

### Executed ill-conditioned-edge toy

| quantity | before | after |
|---|---:|---:|
| displayed factor squared norm | 8,071,367.95 | 25.7000000001 |
| proven minimum \(2\|P\|_*\) | — | 25.7000000001 |
| relative physical-contraction drift | — | \(3.75\times10^{-12}\) |
| relative physical singular-spectrum drift | — | \(3.75\times10^{-12}\) |
| balanced Gram defect | — | \(9.78\times10^{-16}\) |

The \(10^{-12}\) contraction drift is roundoff from first constructing the product
through a condition-number-\(10^6\) gauge. A declared relative singular-value cutoff
of \(10^{-10}\) rejects the resulting numerical ghost rank.

### Executed dormant-bond toy

Two nonzero columns of \(A\) were connected to exactly zero rows of \(B\). They cost
space but contributed nothing to \(P\).

| quantity | before | after |
|---|---:|---:|
| displayed bond width | 5 | 3 |
| displayed factor squared norm | 43.14 | 25.40 |
| relative physical-contraction drift | — | \(4.64\times10^{-16}\) |

This is a real exact simplification. Technically, reducing the width reaches the
closure of the original \(GL(5)\) orbit rather than using an invertible \(5\times5\)
gauge. That distinction matters: it is licensed because the removed directions are
certifiably annihilated by the neighboring node.

## What this means for bilin18

### The third-order MLP tensor

The native elementwise-product core permits independent scalar rescaling and
permutation of product gates. It does **not** permit arbitrary \(GL(h)\) mixing of
those gates while retaining the same diagonal product core. A general hidden change
of basis would turn the core into a dense Tucker core and must pay for the extra
cross-products.

We already applied the exact scalar minimum-norm balancing before the implicit MLP1
and MLP2 folded-tensor HOSVD. Those HOSVD spectra were dense. The toy and proof say
that no better scalar balancing can change that conclusion.

### The MLP0 Down matrix

For a factorization \(D=UV\), the same two-factor theorem applies. Balancing can make
\(U,V\) canonical and can remove redundant latent width, but it cannot change the
singular values or rank of \(D\). Ordinary SVD already supplies its best
coefficient-Frobenius rank-\(k\) approximation. A better MLP0 simplification must use
the activation/causal metric, sparse input-dependent codes, shared downstream wiring,
or a different executable grammar—not a different scaling of the same matrix.

### Where to use the idea next

The promising use is a **joint producer--consumer edge**, such as an MLP0 sparse
write dictionary together with every MLP1/attention reader of that dictionary.
Before measuring atom sparsity or fitting a hierarchy/DAG:

1. contract or jointly declare the producer and all consumers;
2. minimize norm over the actual shared \(GL(r)\) gauge;
3. remove certified dormant directions;
4. diagonalize the balanced Gram/HOSVD, treating degenerate eigenspaces as blocks;
5. only then compare sparse SAE, hierarchical/DAG, and dense controls at matched
   executable cost and causal fidelity.

This prevents an arbitrary latent basis from being mistaken for either sparsity or
semantic structure. It is a prerequisite for a fair comparison, not itself the final
decomposition.

## Decision rule

Norm minimization has earned a role if at least one of these occurs:

- factor statistics become stable under arbitrary gauge replays;
- numerical condition number and materialization error fall;
- a dormant bond direction is deleted exactly;
- repeated fits agree on invariant subspaces after quotienting the residual orthogonal
  symmetry.

It has **not** earned a simplicity claim merely because factor norms fell. Promotion
still requires smaller serialized/runtime cost plus held-out CE/KL, OOD, composition,
and edit/removal tests appropriate to the intended use.
