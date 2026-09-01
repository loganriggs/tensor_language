# MLP0 p448 shared-output quadratic producer — preregistration

Date: 2026-09-01 17:05 UTC  
Rung: 410  
Claim level: executable producer screen; adoption requires fresh/OOD, composition, and intervention gates

## Question

Can the held-out rank-64 output-error interface selected by rung 409 be produced causally and cheaply from the
current MLP0 input, rather than read from the native output as an oracle?

## Weight-derived computation

Let `z in R^1152` be the normalized MLP0 input. Native and covariance-p448 MLP0 are bilinear quadratic programs:

`y_n(z) = D_n[(L_n z) * (R_n z)] + b_n`,

`y_p(z) = D_p[(L_p E z) * (R_p E z)] + b_p`.

Let `U in R^(1152 x 64)` be the deterministic total-error output basis reconstructed exactly as in rung 409 from
training source documents 0:192. The desired correction coefficients are

`q(z) = (y_n(z)-y_p(z)) U in R^64`.

For every coefficient `j`, contracting output direction `U[:,j]` into `D_n` and `D_p` gives a symmetric matrix
`A_j` and constant `beta_j` such that

`q_j(z) = z^T A_j z + beta_j`.

This is derived directly from the weights. It does not use a target token, future loss, document identity, or native
forward at runtime.

## Frozen data metric and factorization

On source documents 0:192 only, compute the uncentered second moment `C=E[z z^T]`. Eigendecompose `C`, floor its
eigenvalues at `1e-6 * lambda_max`, and define mutually inverse square-root factors. For each exact quadratic form,
form `C^(1/2) A_j C^(1/2)`, take signed eigencomponents ordered by absolute eigenvalue, and retain ranks
`r in {8,16,24}`. Runtime coefficients are

`q_hat_j(z)=beta_j + sum_(k=1)^r lambda_jk (z^T C^(-1/2) v_jk)^2`.

Also run:

- `FULL1152`, retaining all components, as the algebra/implementation control;
- `AFFINE`, one ridge affine map from `z` to the 64 target coefficients, fit only on documents0:192;
- `SHUFFLED24`, seed410 permutation of the real rank-24 eigencomponents across coefficient outputs, retaining the
  same values, directions, price, and operation count.

Score only source documents192:384 in waves192:288 and288:384. `FINAL` remains closed. Baselines are native,
covariance-p448, covariance-p640, covariance-p768, and the reconstructed rung409 `TOTAL_ERROR_64` native-error
oracle.

## Literal price

The shared output basis costs `1152*64 = 73,728` values. A rank-`r` quadratic producer stores 64 groups of `r`
1,152-dimensional directions plus one signed scalar per component. Constants are charged separately even if their
measured difference is zero:

`price(r) = 73,728 + 64*r*(1,152+1) + 64`.

Therefore:

- rank8 producer/interface: 664,128 values; p448 total 10,618,560;
- rank16: 1,254,464 values; p448 total 11,208,896;
- rank24: **1,844,800 values; p448 total 11,799,232**;
- affine producer/interface: `1152*64+64+1152*64 = 147,520` values, counting basis, matrix, and bias;
- p640: 11,945,088 values; p768: 13,272,192 values.

Rank24 leaves 145,856 values of storage headroom below p640. Report multiplication/addition/square counts and peak
workspace separately; fit-time dense forms and eigendecompositions are not runtime storage.

## Registered predictions

### Prediction A — exact instrument and derivation

All rung409 row/program/basis reconstruction hashes, split, p448/p640/p768 saved losses, shapes, calls, and finite
checks hold. `FULL1152` reconstructs the direct analytical 64 coefficients with relative squared error at most
`1e-6`, and its physical CE differs from the reconstructed rung409 `TOTAL_ERROR_64` oracle by at most `0.0002` nat.
The rank24 storage count is exactly 1,844,800 and total p448 price is below p640.

### Prediction B — priced rank24 recovers a useful fraction of the oracle

Let `G_oracle = damage(p448)-damage(TOTAL_ERROR_64_oracle)`. On evaluation documents, rank24 gains at least
`0.5*G_oracle`, gains at least `0.0015` nat in each wave, and has damage below `0.0064` nat. This is the pre-hoc
Pareto-contention bar: at a lower storage price than p640, it must recover at least half the available shared-output
gain.

### Prediction C — the quadratic spectrum is live and ordered

Held-out coefficient relative squared error and physical CE damage are non-increasing from rank8 to rank16 to
rank24. Rank24 retains at least 70% of the data-metric quadratic-form energy and improves rank8 CE by at least
`0.0005` nat.

### Prediction D — the weight-derived quadratic producer is specific

Rank24 beats `SHUFFLED24` by at least `0.001` nat and beats `AFFINE` by at least `0.0005` nat on pooled held-out CE,
with positive advantage in both waves. The affine control may still be selected by the frozen decision if it beats
rank24; prediction D would then honestly fail.

## Strong null

The strong null fires if A fails, rank24 improves p448 by less than `0.0002`, `SHUFFLED24` is within `0.0002` of
rank24, the rank24 literal price is not below p640, or neither rank24 nor AFFINE recovers 25% of the oracle CE gain.

## Frozen decision

- If A/B and either C or D hold without the null, promote the better of rank24 and AFFINE to one larger fresh/OOD,
  signed-intervention, and composition gate at its exact price. No rank tuning.
- If AFFINE beats rank24, select AFFINE despite prediction D's failure; the correction is effectively linear on the
  observed state manifold even though the exact weight object is quadratic.
- If rank24 has specific gain but misses B, keep it only as a mathematical screen and prefer p640 operationally.
- If the strong null fires, close this rank64 output-producer route and move to direct nonlinear whole-program CE or
  the later-layer quadratic surrogate family.

This rung cannot itself establish fresh/OOD transport, composability, intervention fidelity, or adoption.
