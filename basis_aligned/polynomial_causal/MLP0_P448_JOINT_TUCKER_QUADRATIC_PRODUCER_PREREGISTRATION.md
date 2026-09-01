# Preregistration — joint Tucker producer for the MLP0 p448 output correction

**Date:** 2026-09-01 17:18 UTC  
**Rung:** 411  
**Claim level:** held-out executable tensor-factorization and price screen; not adoption

## Decision and exact object

Rung410 proved that the native-minus-p448 correction projected into one train-only output basis
`U in R^(1152 x 64)` is exactly

`q_j(z) = z^T A_j z + beta_j`, for `j=1,...,64`.

Its independent rank-24 approximation was specific but retained only 35.7% of the data-weighted form spectrum and
recovered 32.0% of the output oracle. This rung changes the factorization object rather than increasing that rank.
It tests whether all 64 symmetric matrices share one input subspace:

`q(z) = core(u,u) + beta`, `u = V^T C^(-1/2) z`.

Here `C` is the uncentered training second moment, `V in R^(1152 x p)` is shared by every output coordinate, and
`core in R^(64 x p x p)` is symmetric in its last two indices. Computationally, form
`B_j=C^(1/2) A_j C^(1/2)`, take the top `p` eigenvectors of the mode-input HOSVD Gram
`G=sum_j B_j B_j^T`, and set `core_j=V^T B_j V`. This is a tied-input Tucker/HOSVD approximation of the partially
symmetric three-index tensor `B[output,input,input]`. It is not rung381: that older screen compressed the native
tensor's output mode; this one has already fixed a 64-dimensional residual output interface and compresses the two
input modes of its exact error tensor jointly.

## Frozen data and roles

- Reuse rung409/410's exact 384-source-document authority and hashes.
- Documents `[0,192)` determine `U`, `C`, and all Tucker factors.
- Documents `[192,384)` are evaluation only, reported as two fixed 96-document waves.
- Positions `[64,256)` are scored. `FINAL_opened=0`.
- Runtime producers may read only the current normalized MLP0 input and stored factors. They may not read native
  MLP0 output, targets, future loss, document identity, or an evaluation lookup.

## Fixed ranks, arms, and literal prices

Tucker ranks are fixed before execution at `p in {96,160,226}`. A symmetric core stores only `p(p+1)/2` values
per output. Including `U`, shared input directions, and 64 offsets, producer price is

`1152*64 + 1152*p + 64*p*(p+1)/2 + 64`.

| Tucker rank | producer values | p448 + producer | equal-or-cheaper covariance control |
|---:|---:|---:|---:|
| 96 | 482,368 | 10,436,800 | p494 = 10,431,360 |
| 160 | 1,082,432 | 11,036,864 | p552 = 11,032,704 |
| 226 | 1,975,808 | 11,930,240 | p638 = 11,924,352 |

The exact covariance price increment is 10,368 values per added input rank. Rank227 is forbidden because it would
cost 11,945,920 total, 832 values more than p640. Physical arms are native, p448, p494, p552, p638, p640, the
non-executable U64 oracle, full analytic U64, Tucker96/160/226, a seed411 Haar shared-rank226 control, and rung410's
independent-r24 producer reconstructed without changing it.

## Frozen measurements

- exact row/loss/program/output-basis identities, live call counts, state replay, and full-form algebra;
- pooled and two-wave CE damage above native for every physical arm;
- coefficient relative MSE against the deployed U64 oracle;
- one-mode and tied-two-mode data-weighted tensor energy retained by each Tucker rank;
- literal storage and per-token multiplication counts;
- real-versus-random shared-subspace specificity.

## Frozen predictions

**A — authority and construction.** All external identities and calls hold; full analytic replay remains within
`0.0002` nat of rung409's oracle; form identity relative MSE is at most `1e-6`; ranks, shapes, and every matched
price are exact.

**B — a priced joint producer enters contention.** Tucker226 recovers at least 50% of the U64 oracle gain, gains at
least `0.0015` nat over p448 in each wave, and has damage at least `0.0002` lower than the equal-or-cheaper p638.

**C — shared input structure is materially more efficient than separate form ranks.** Tucker energy and coefficient
MSE are monotone with rank; Tucker160 damage is no worse than rung410 independent-r16 damage `0.00715957` despite
using 172,032 fewer values; Tucker226 retains at least 60% tied-two-mode tensor energy.

**D — alignment is specific.** Tucker226 beats the Haar226 control by at least `0.001` nat and rung410's
independent-r24 producer by at least `0.0005` nat, with both differences positive in both waves.

## Strong null and frozen decision

The strong null fires if A fails; if no Tucker arm beats its own equal-or-cheaper covariance-rank control by at least
`0.0002` nat; if Tucker226 is within `0.0002` of Haar226; or if every Tucker arm recovers less than 25% of the U64
oracle gain.

If A holds and a Tucker arm clears its matched covariance control without the null, select the smallest-price passer
for fresh/OOD, signed-intervention, and composition gates. Otherwise close the fixed low-rank quadratic-producer
family at MLP0: the residual remains exactly characterized but ordinary added covariance rank is a better use of
storage. Do not tune Tucker rank, metric floor, output rank, or bars after viewing the result.
