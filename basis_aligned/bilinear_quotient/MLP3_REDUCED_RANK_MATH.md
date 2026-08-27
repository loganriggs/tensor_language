# MLP3 reduced-rank ridge: theorem to implementation

## Why coefficient SVD is the wrong frontier

The MLP3 affine candidate fits a residual target `Y` from centered local-DAG inputs
`X = [attn3.c_proj, mlp2.output]` with ridge penalty `lambda`:

```text
J(W) = ||Y - XW||_F^2 + lambda ||W||_F^2.
```

Ordinary SVD truncation of the full coefficient `W*` minimizes
`||W-W*||_F`, not `J(W)`. Those objectives agree only when the penalized input
metric is proportional to identity. That is implausible here: residual-stream
coordinates and the two writer blocks are strongly anisotropic. An ordinary
coefficient SVD would therefore make rank a coordinate-norm property rather than a
minimal predictive program at that budget.

Reduced-rank regression dates to Izenman's multivariate formulation, and reduced-rank
ridge explicitly combines a coefficient-rank constraint with ridge regularization
([Izenman 1975](https://doi.org/10.1016/0047-259X(75)90042-1);
[Mukherjee and Zhu 2011](https://doi.org/10.1002/sam.10138)). The final truncation
uses the classical best-Frobenius-rank result of
[Eckart and Young 1936](https://doi.org/10.1007/BF02288367).

## Closed form used here

Let

```text
A = X'X + lambda I       C = X'Y       W* = A^-1 C.
```

Completing the square gives

```text
J(W) = constant + ||A^(1/2) (W-W*)||_F^2.
```

Because `A^(1/2)` is invertible, `rank(A^(1/2)W) = rank(W)`. Define

```text
Z* = A^(1/2) W* = A^(-1/2) C.
```

If `Z* = U diag(s) V'`, Eckart--Young gives the rank-`r` optimum

```text
W_r = A^(-1/2) U[:, :r] diag(s[:r]) V'[:r, :].
```

Every candidate is thus a prefix of one frozen decomposition, full rank reconstructs
ordinary ridge, and the penalized fit objective cannot worsen with rank. This is a
fit-metric optimality statement, not a promise that held-out cross-entropy will be
monotone; failure of operational monotonicity remains a preregistered falsifier.

## Implementation consequences

- `reduced_rank_ridge.py` consumes only sufficient statistics `X'X` and `X'Y`.
- It symmetrizes `X'X`, requires a positive ridge penalty and a positive-definite
  metric, then diagonalizes `A` and SVDs `A^(-1/2)C`.
- `mlp3_fit_artifact.py` stores `A^(-1/2)U diag(s)` and `V'`. Prefix contraction
  directly constructs each semantic `W_r`; the canonical byte codec subsequently
  removes its internal factor gauge. Canonicalization is performed separately on
  each already-constructed `W_r`. Re-SVDing the full `W` and taking those prefixes
  would silently replace the prediction-optimal family with a coefficient-Frobenius
  family, so a discriminating encode/decode regression test forbids that shortcut.
- CPU tests verify full-rank equality to ridge, rank bounds, monotone penalized
  objective, and superiority to ordinary coefficient-SVD prefixes on an anisotropic
  fixture.

## Falsifying and boundary tests

1. If full-rank factors do not reconstruct `solve(A,C)`, reject the implementation.
2. If any prefix has higher penalized fit loss than the ordinary coefficient-SVD
   prefix of the same rank, reject the implementation.
3. If held-out or composed CE is materially nonmonotone, preserve that result: fit
   optimality does not imply operational optimality under distribution shift or
   downstream recurrence.
4. If the full affine family misses causal/OOD gates, add an explicitly priced
   quadratic residual; do not hide the failure by changing the metric after seeing
   evaluation rows.
