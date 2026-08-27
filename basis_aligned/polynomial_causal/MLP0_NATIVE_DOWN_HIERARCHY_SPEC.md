# MLP0 native-Down hierarchy: registered executable discriminator

## Claim boundary

This experiment asks whether a coarse lexical state lowers the description price of
MLP0's **write map** at fixed causal fidelity.  It does not claim to simplify all of
MLP0: the exact RMS-normalized input and exact `Left` and `Right` maps remain in the
program, and their parameters and operations are reported as common cost.

For normalized MLP0 input `z`, retain the exact native product state

```text
h(z) = (Left z) * (Right z),                 h in R^4608.
```

The original `Down` call is poisoned during candidate evaluation.  Each arm produces
the residual write through a serialized bf16 program

```text
y_hat[b,r](token,z) = b(token) + mu[b] + A[b,r] B[b,r] (h(z) - mu_h),
```

where every constant is learned on fit rows only.  `A B` is fit directly to the
residual `Down(h)-b(token)`; it may not read, project, or cache the live MLP0 output at
evaluation time.  A cloned full-rank native map is an integrity arm, not a candidate.

## Registered arms and price

Two price rungs are fixed by continuous native residual maps `C256` and `C512`:

- `C_R`: `b(token)` is one shared mean and the residual map has rank `R`;
- `Q_r`: `b(token)` is the fit-frozen reader-metric K=64 lexical centroid used by
  Stage 0, plus the largest residual rank `r` whose complete serialized price does
  not exceed `C_R`;
- `A_r`: the analogous activation-k-means K=64 centroid plus the largest affordable
  residual rank;
- `Qnull_r` and `Anull_r`: identical token assignments, occupancy, rank, precision,
  and byte price, but occupied centroid vectors are deterministically deranged before
  the residual map is refit.

Ranks for Q/A are determined once from fit-only occupancy and the price formula, not
from evaluation outcomes.  For every arm, price includes bf16 `A`, `B`, intercept,
centroids, full-vocabulary assignments at `ceil(log2(K_occupied))` bits (or an
explicitly larger actual serialization), padding/metadata needed by the decoder, and
the decoder program hash.  The charged price is the larger of the analytic bit count
and actual serialized bytes.  Exact common `Left`/`Right` cost is reported separately
and never described as saved.  FLOPs and peak interface width are separate Pareto
coordinates.

Factor gauges are canonicalized by SVD into balanced factors with descending singular
values and a deterministic sign rule, then serialized to bf16 before evaluation.
Float32 fidelity priced as bf16 is forbidden.  A candidate whose reconstructed
serialized bytes exceed its matched continuous rung fails the price gate.

For a null, occupied centroids are matched by a deterministic minimum-cost
derangement using standardized `(log fit mass, centroid norm)` with identity edges
forbidden.  Assignments stay fixed.  The residual map is refit after derangement.
Mass/norm mismatch is reported; the null is not called exact matching unless both
multisets match exactly.

## Row authority and effective sample size

The fit role reuses the authoritative Stage-0 fit rows.  Evaluation uses a new
network-independent window beginning at FineWeb dataset-document index 25,000 from
the pinned local parquet.  It selects exactly 384 eligible source documents in
dataset order and one to three non-overlapping 513-token chunks per document.  No
evaluation source document, row, or 32-token prefix may occur in fit or any prior
registered role.

The first 192 source documents are replication wave A and the next 192 are wave B;
all chunks from one document stay in one wave.  The paired source document—not a
chunk, position, or forward-pass token—is the resampling unit.  The receipt must
report unique documents, chunks, predicted positions, chunks-per-document, and every
cell's source-document support.  Each wave must retain at least 192 documents,
98,304 raw next-token positions before masks, at least 60 source documents per cell,
and at least 90% covered evaluated positions.  Failure of a mechanical row gate is
inconclusive before model evaluation; it cannot be repaired after outcomes are read.

This design is an explicit doubling audit: wave A is a complete 192-document test,
then wave B doubles the independent-document count without changing candidates,
ranks, cells, or thresholds.

## Consumers, cells, and raw ledger

Every arm is installed globally and scored on the same rows against exact MLP0 `O`.
Retain paired sums and counts for every source document, wave, arm, consumer, and the
Stage-0 2x2x2x2 cells:

- `KL(p_O || p_arm)` and signed `CE_arm - CE_O`;
- block-1 attention-output nRMSE;
- block-1 MLP-output nRMSE.

The cell axes and all scales/medians are fit-frozen exactly as in Stage 0.  Margins
remain KL `.01`, CE harm `.0075`, and each direct nRMSE `.05`.  Report means for
diagnosis, but acceptance always uses the maximum standardized effect over consumers
and cells.  Candidate execution must instrument and assert zero original-`Down`
calls; the integrity arm must reproduce `O` before any candidate is scored.

## Simultaneous inference and doubling-stability gates

Use at least 20,000 paired source-document bootstrap replicates with one common set
of resample indices across all arms, rungs, consumers, and cells.  Recompute the
entire maximum statistic in every replicate; no arm or consumer receives a separate
uncorrected interval.  Seeds and document ledgers are serialized so the result can be
replayed without the model.

An arm earns **absolute causal-fidelity credit** only if:

1. wave A's simultaneous one-sided 95% UCB on maximum standardized distortion is
   below `1.0`;
2. wave B independently has UCB below `1.0`;
3. the pooled 384-document UCB is below `0.8`, reserving 20% margin against sampling
   and deployment drift;
4. coverage and per-cell document-support gates pass in each wave;
5. the conclusion and identity of every over-margin consumer family are unchanged
   when wave B is added.

Thus a pooled pass cannot be manufactured by cancellation between halves.  If one
wave passes and the other fails, or pooled UCB lies in `[0.8,1.0)`, the arm is
inconclusive rather than accepted.  A decisive equivalence rejection requires the
pooled simultaneous 95% lower bound to exceed `1.0`; point estimates alone are
reported as failures of the practical margin but not as powered rejection.

A hierarchy earns **priced lexical-simplicity credit** at a rung only if it has
absolute credit and, under the same paired bootstrap:

1. it is no worse than its matched `C_R` in every consumer/cell point estimate;
2. the one-sided family-wise 95% lower bound for `max(C_R)-max(H_r)` is positive in
   wave A, wave B, and pooled data;
3. it beats its structured null under the same rule;
4. its serialized price is no greater than `C_R`.

Two rungs are a registered family, not two independent chances to claim success.
The common bootstrap maximum corrects over both.  No rank, K, cell, or threshold is
changed after wave A.

## Registered interpretations

- Absolute + continuous + null gates pass: coarse lexical structure reduces the
  write-map price at fixed causal fidelity.
- Hierarchy beats continuous but misses absolute fidelity: suggestive price ordering,
  no replacement or whole-model credit.
- Hierarchy ties its null or loses continuous: no evidence for lexical simplicity at
  that price.
- Every arm fails at the 512 rung: this native low-rank residual grammar fails; this
  is not a proof that MLP0 or the full tensor network is incompressible.

Only a passing, serialized program can enter a later exact `{MLP0,MLP1,MLP2}`
composition test.  Oracle projections of live outputs, average-only CE recovery, and
new hard-cluster/donor-swap sweeps are explicitly pruned by this specification.

## Execution authority

This file authorizes only outcome-blind row freezing and CPU tests.  A separate,
committed collector authority must bind the checkpoint, model source, exact
construction constants, serialized candidate hashes, poison instrumentation, response
scales, and result path before any candidate model forward.
