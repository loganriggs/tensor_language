# Exact-minimum causal-subspace toy ladder

## High-level purpose

The central comparison should not be “DAS versus difference in means.” It should be whether an
estimator recovers the smallest subspace whose intervention preserves the desired computation over
the full declared population. Synthetic models let us know that answer exactly, distinguish
estimation failure from non-identifiability, and create memorization failures deliberately.

The first CPU-only ladder has four rungs: basis-aligned rank one, rotated rank three, a shared/private
two-task tensor, and a subset-gated factor. It is an algebraic benchmark, not evidence about the
language model.

## Exact linear certificate

Let columns of `V` be an orthonormal basis for every allowed activation change and let `R` stack all
downstream linear readers that must be preserved. An orthogonal intervention projector `P` is exact
when

```text
R P V = R V.
```

Because `rank(R P V) <= rank(P)`, every exact projector has

```text
rank(P) >= rank(R V).
```

Let `Q` span the row space of `R V`. Then

```text
U* = V Q
P* = U* U*^T
```

attains equality, since `(R V) Q Q^T = R V`. Thus `rank(R V)` is a lower-bound certificate and
`span(U*)` is a canonical exact minimum subspace. This is basis/gauge invariant.

The certificate uses the full allowed-change span and the full reader family. Restricting either to
observed training rows changes the question and can make a memorized lower-rank answer look exact.

## Progressive scenarios

1. `axis_aligned_rank1`: one coordinate is causal and two orthogonal directions vary but are never
   read. Exact minimum and DIM are both rank one.
2. `rotated_rank3`: three causal directions are rotated away from the model basis and two nuisance
   directions vary. The mean change points mostly along only one causal direction, so DIM is rank one
   and necessarily misses the other two; the exact method recovers rank three.
3. `shared_plus_private`: task A reads shared plus private-A; task B reads shared plus private-B.
   Each task's minimum is rank two, their intersection is exactly rank one, and their union is rank
   three. This is the smallest toy of the proposed shared-circuit tensor decomposition.
4. `subset_gated`: common rows read one factor, while a rare subset reads that factor plus a second.
   Training only on common rows certifies rank one on its restricted population yet fails the full
   population. This is a controlled analogue of construction memorization.

## Metrics and next rungs

Each scenario reports the certified lower bound, attained rank, maximum universal response error,
DIM rank/error, and scenario-specific shared/subset diagnostics. The first non-omniscient estimator
is response regression: it sees paired activation changes and downstream response changes, fits the
minimum-norm linear map by least squares, and returns that map's input row space. It never reads `R`
directly. On the noise-free full-span rungs it should recover the certified projector exactly; on a
common-only subset it can only identify rank one and must fail the hidden rare-subset factor.

Next comparisons should include delta-SVD, optimized orthogonal projectors, sparse or group
projectors, and tensor-factor methods. The ladder should then add finite samples, observation noise,
correlated nuisance, nearly degenerate singular values, nonlinear MLP readers, and bilinear
attention-like context gates.

Success requires more than behavioral fit: correct minimum rank, small projector distance to the
identifiable ground truth, held-population response preservation, and rejection of spurious nuisance
directions. When the ground truth is not identifiable from the training distribution, the benchmark
must say so rather than credit any optimizer for guessing the hidden factor.
