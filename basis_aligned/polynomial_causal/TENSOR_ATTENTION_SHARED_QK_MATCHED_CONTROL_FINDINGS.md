# Matched weighted independent versus shared QK finding

Date: 2026-08-28

Status: discovery-only executable attention-subsystem result.

## What the control resolved

The original routing-384 arm recovered about 97.1% while shared-QK-384 recovered about
99.4%, but the two used different fitting objectives. The matched control gives both
classes their exact activation-weighted Eckart--Young solution and fits each bottom-up
on its own deployed trajectory.

| class | skip-7000 recovery | skip-11000 recovery | stored values | multiply-adds |
|---|---:|---:|---:|---:|
| independent weighted Q/K rank 384 | **99.8412%** | **99.8428%** | 111.478M | 130.460B |
| shared weighted Q/K rank 384 | 99.4635% | 99.4342% | **87.590M** | **105.998B** |

The shared replay reproduces the parent result within $1.1\times10^{-8}$ recovery.
The shared constraint costs 0.3777 percentage points on skip-7000 and 0.4086 points on
skip-11000, both within the preregistered 0.5-point materiality threshold. All frozen
predictions pass.

## Correct interpretation

Most of the old 97.1% to 99.4% improvement was caused by replacing ordinary SVD of the
ridge coefficient with the correct activation-weighted low-rank solution. Sharing does
not improve fidelity: the independent weighted class is better by about 0.4 recovery
points and only +0.0134 to +0.0153 nat CE.

The common interface nevertheless remains a real Pareto simplification. It forces the
four routing maps to read one 384-dimensional state subspace instead of four potentially
different subspaces. Relative to the matched independent program it removes 23.888M
stored values (21.43% of complete attention-program storage) and 24.461B multiply-adds
per production forward (18.75%), for a replicated sub-threshold fidelity cost. Within
the QK factor block itself the price falls from $8Dr$ to $5Dr$, a 37.5% reduction.

This is operational evidence for a shared routing interface under the two tested
FineWeb roles. It is not yet a semantic interpretation of the 384 coordinates, an OOD
certificate, or a proof that rank 384 is minimal. The encoder/decoder factorization has
the gauge freedom

$$
E\mapsto EG,\qquad D_j\mapsto G^{-1}D_j,
$$

so coordinate-level claims require a canonical gauge and stability tests.

## Execution controls

- Parent shared result is source-hash bound and replayed essentially exactly.
- Both programs have total support, zero token tables, and zero literal native attention
  calls.
- All 18 sites dispatch in order with exact block and first-value-bus transaction
  closure.
- Held-out roles contain 27,974 and 27,497 scored positions.
- Runtime is 102.4 seconds; the two trajectories share only batch execution, not
  covariances, prefixes, or programs.

## Consequence for the project

The attention component is now understood at three levels:

1. exact tensor operator identity;
2. near-lossless activation-weighted routing compression;
3. a cheaper shared 384-dimensional routing interface with a measured 0.4-point cost.

The largest whole-model blocker is no longer attention. It is the absence of a true
pre-execution, zero-native-call MLP/local program and the unmeasured factorial interaction
between that program and shared attention. Further attention rank sweeps are useful for
the simplicity frontier, but lower priority than closing this missing whole-model
interface.
