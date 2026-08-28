# MLP1 implicit folded-tensor diagnostic v1

Status: prospective, CPU mathematics only. No bilin18 checkpoint tensor may be
loaded and no checkpoint-derived number may be computed until this document,
`mlp1_implicit_folded_tensor_v1.py`, and its test are committed, pushed, and
bound by a separate create-only source/weight authority. This document does not
itself authorize a model load or a scientific result.

## Question and registered object

For bilin18 MLP1, write the bias separately and define

```text
y(z) = b + D ((L z) odot (R z))
T[o,i,j] = sum_n D[o,n] (L[n,i] R[n,j] + R[n,i] L[n,j]) / 2.
```

Only the partially symmetric tensor `T` is analyzed. `b` is copied and priced,
but is never folded into an augmented input and never contributes to a tensor
spectrum. The native hidden index may be permuted or independently rescaled as
`L[n]*=alpha`, `R[n]*=beta`, `D[:,n]/=(alpha*beta)` without changing any
registered folded statistic.

Before matrix-only comparisons, each nonzero term is put in the positive
minimum-squared-norm scalar gauge

```text
norm(L[n]) = norm(R[n]) = norm(D[:,n])
           = (norm(L[n]) norm(R[n]) norm(D[:,n]))**(1/3).
```

Exactly zero terms are zeroed in all three factors. There is no relative
threshold and hence no outcome-dependent term deletion.

## Frozen diagnostics

The diagnostic has three distinct arms.

1. **Balanced Down-only SVD.** Report all squared singular values of balanced
   `D`, their cumulative Frobenius fractions, and the first ranks reaching
   `0.90, 0.95, 0.99, 0.999`. This is a final-mixing compression and continues
   to execute all native products.
2. **Folded multilinear/HOSVD spectrum.** Compute the exact output-mode and
   input-mode Gram matrices in float64 by hidden-index blocks. The implementation
   may materialize `o*o`, `d*d`, and one `block*h` workspace, but must never
   materialize `T` or any `o*d*d`/`d**3` object. Report squared mode singular
   values, the same four energy ranks, the equality of the two input modes, and
   agreement of output/input Gram traces. Negative eigenvalues no larger than
   the registered numerical tolerance may be clipped to zero; a larger negative
   eigenvalue is a hard failure.
3. **Executable Tucker/CP requirements.** For registered HOSVD ranks, project
   only the requested core `G[ro,ri,ri]`, symmetrize its input modes, and report
   dense-core and top-COO coefficient curves. Top-COO selection is by folded
   tensor energy: a diagonal coefficient has mass `G[a,b,b]^2`; an off-diagonal
   symmetric pair has mass `2*G[a,b,c]^2`. Ties are deterministic by
   `(output,b,c)`. This is a diagnostic, not a CP fit. A future CP result must
   serialize `C[o,q], A[q,d], B[q,d], b[o]`, execute exactly `q` products, and
   be evaluated causally under a separately frozen protocol.

The exact Gram identities are

```text
S_n = (l_n r_n^T + r_n l_n^T)/2
K_nm = <S_n,S_m>
     = ((l_n.l_m)(r_n.r_m) + (l_n.r_m)(r_n.l_m))/2
G_out = D K D^T
G_in  = sum_nm (d_n.d_m) S_n S_m.
```

The second input Gram equals the first because `T[o,i,j]=T[o,j,i]`.

## Exact price ledger

Prices are per token and include the separate bias. A multiply-add is counted
once per stored dense coefficient; elementwise products and bias additions are
separate counters.

```text
native:
  float storage = 2*h*d + o*h + o
  multiply-adds = 2*h*d + o*h; products = h; bias additions = o

balanced Down rank r:
  float storage = 2*h*d + r*(h+o) + o
  multiply-adds = 2*h*d + r*(h+o); products = h

symmetric dense Tucker (ro,ri), p=ri*(ri+1)/2:
  float storage = d*ri + o*ro + ro*p + o
  multiply-adds = d*ri + ro*p + o*ro; products = p

symmetric COO Tucker with s scalar coefficients and p active input pairs:
  float storage = d*ri + o*ro + s + o
  integer storage = 3*s COO indices
  multiply-adds = d*ri + s + o*ro; products = p

CP rank q:
  float storage = q*(2*d+o) + o
  multiply-adds = q*(2*d+o); products = q
```

No index entropy coding, parameter sharing, hardware fusion, or inherited-weight
discount is claimed. Both full-component and replacement-only counts must remain
available so Down-only is not mislabeled as a gate reduction.

## Registered output schema and decision boundary

A later authority-bound runner may publish one create-only JSON containing:

- exact source, commit, model/config/checkpoint, and tensor byte hashes;
- shapes/dtypes, CPU runtime, block size, and peak-workspace estimate;
- bias shape/hash/norm, separately labeled;
- balancing dead-unit list and before/after log-norm defects;
- Down-only squared spectrum, cumulative fractions, threshold ranks, and prices;
- output/input folded squared spectra, cumulative fractions, threshold ranks,
  Gram traces, symmetry residual, trace residual, and prices;
- requested projected-core ranks, exact core hash, dense and sparse-core curves;
- the CP executable contract and registered price curves, with no fitted-CP claim.

This v1 is descriptive. It may say that an output mode or input mode is
Frobenius-compressible at a registered threshold. It may not claim preserved CE,
stable lexical semantics, minimum CP rank, fewer causal gates, or a working
replacement until a separately preregistered executable intervention passes.

## Synthetic acceptance tests before source freeze

- exact small-tensor Gram equality against explicit materialization;
- known diagonal tensor spectra and exact full-rank HOSVD reconstruction;
- function and bias preservation under balancing;
- invariance under hidden permutation, branch swap, signed/log-scale gauge;
- deterministic sparse-core ordering and folded-energy accounting;
- exact native, Down-rank, dense/COO Tucker, and CP price formulas;
- rejection of nonfinite, malformed, non-CPU, or inconsistent factors.
