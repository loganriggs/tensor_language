# MLP2 folded-tensor findings

Date: 2026-08-28

## Bottom line

MLP2 is slightly more concentrated than MLP1 in ordinary coefficient space, but it
is not qualitatively simpler. All three measured mode ranks are numerically full
rank, and every registered dense Tucker program is much larger and more expensive
than the native MLP. This closes coefficient-Frobenius HOSVD as the next MLP2
compiler. It does **not** close compression under the natural activation distribution
and says nothing about MLP2's observed compensating causal role.

The closed result transaction is:

- source/weight authority SHA-256: `eed76e66ad2dd13d75f814bb5669c3bf90f3da9a36aed32ccef095ce327976e6`;
- result SHA-256: `0994555ebd9eb991bcb1bb721fffa4c08a54523e5835ae5ba78030e2baa8da97`;
- outcome-authority SHA-256: `13003b3576ba15bdc8dbb6904220ac84c1fe1b54946c0bc6803ae3477cf028c3`;
- runtime: 832.33 seconds on one CPU thread; and
- zero data rows, zero model forwards, and no materialized 1,152-cubed tensor.

An independent post-outcome audit replayed the authority, result, and final validators
and verified all hashes, prices, ranks, and the absence of a failure or lock.

## What was measured

Ignoring the separately preserved bias, a bilinear MLP is

\[
F(x)-b=D((Lx)\odot(Rx)).
\]

This defines a third-order coefficient tensor (T) with one output mode and two
input modes. We never build (T). Instead, exact contractions of (D,L,R) produce
the three small Gram matrices whose eigenvalues are the squared singular values of
the tensor unfoldings.

An energy rank is the number of singular directions needed to retain a chosen
fraction of 

\[
\|T\|_F^2=\sum_{o,i,j}T_{oij}^2.
\]

This weights every coordinate direction equally. It is a property of the polynomial
coefficients, not a measure of the inputs the model actually visits.

## Like-for-like MLP1 versus MLP2 result

| retained coefficient energy | MLP1 output | MLP2 output | MLP1 input | MLP2 input | MLP1 balanced Down | MLP2 balanced Down |
|---:|---:|---:|---:|---:|---:|---:|
| 90% | 835 | 826 | 937 | 922 | 846 | 840 |
| 95% | 962 | 956 | 1,033 | 1,023 | 970 | 966 |
| 99% | 1,103 | 1,101 | 1,123 | 1,119 | 1,105 | 1,104 |
| 99.9% | 1,147 | 1,147 | 1,147 | 1,146 | 1,147 | 1,147 |

Every numerical rank is 1,152. MLP2 needs between one and fifteen fewer directions
than MLP1 at the listed thresholds. That is a reproducible quantitative difference,
but not a different low-rank regime.

## Executable price consequence

The native standalone MLP2 stores 15,926,400 floating values and evaluates 4,608
bilinear products per token.

At 90% balanced-Down energy, rank 840 stores 15,456,384 values: a reduction of
470,016 values, or 2.951%. It still evaluates all 4,608 products. At 95% energy and
above, the two-factor reduced-rank Down representation is already larger than the
native matrix.

The registered dense symmetric Tucker points are worse:

| energy point | output rank | input rank | HOSVD relative-error upper bound | stored values | products |
|---:|---:|---:|---:|---:|---:|
| 90% | 826 | 922 | 54.667% | 353.48M | 425,503 |
| 95% | 956 | 1,023 | 38.626% | 503.01M | 523,776 |
| 99% | 1,101 | 1,119 | 17.218% | 692.49M | 626,640 |
| 99.9% | 1,147 | 1,146 | 5.097% | 756.49M | 657,231 |

All lose to the native program on both storage and product count. The preregistered
useful dense-Tucker window required input rank at most 95 and a certified error bound
at most 5%; MLP2 misses both by a wide margin.

## What is now pruned, and what remains open

Pruned:

- ordinary coefficient-Frobenius dense Tucker/HOSVD as the next MLP2 compiler;
- the hypothesis that MLP2's compensating behavior should appear as a dramatically
  smaller Euclidean folded-tensor rank; and
- more coefficient-HOSVD refinement at this layer without changing the metric or
  grammar.

Still open:

- learned CP or block-term rank; HOSVD does not prove CP rank;
- activation-weighted or empirical-fourth-moment compression;
- consequence/Fisher-weighted compression;
- conditional refitting on the state produced by compressed MLP0/MLP1;
- CE/KL, suffix transport, OOD, extraction, or selective removal; and
- the MLP0/MLP1/MLP2 composition cube.

Accordingly, this result changes the branch choice but moves none of the global
explanation or causal-recovery ledgers.

