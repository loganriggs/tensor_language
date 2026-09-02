# Rung 467: split the equality correction inside MLP8, MLP9, and MLP12

Status: prospective design, frozen after the MLP8/9/12 dossier audit and before computing any per-product-term
equality fingerprint. This uses the already-open 192-document code corpus, but product-term selection uses documents
`0:96` only and exact causal evaluation uses documents `96:192` only. The validation documents are held out from
term selection, but they are not a new corpus and the parent module-level effects on them are already known.

This experiment targets within-module splitting, cross-module grouping, held-out prediction, exact removal, and
composition. It does not test whether a low-rank approximation reconstructs an MLP.

## Fixed parent circuit

The matcher sources, scale, code rows, context cells, and absent/native/hybrid trajectories are inherited without
search from rungs 464--466. The two equality sources are:

- `N`: the native layer-8-head-4 equality source;
- `H`: the transplanted layer-5-head-5 score with the layer-8-head-4 output.

The fixed target modules are `M = {MLP8, MLP9, MLP12}`. Rung 466 showed that removing these three complete writes
has the same four-context direction under `N` and `H`, with coordinate order
`(near, far, one predecessor, multiple predecessors)` and signs `(-,+,+,-)`. No other module, source, task cell,
rank, unit count, or direction may be selected after this registration.

## Exact product terms

At module `l`, its normalized input is `x_l`, and its 4,608 product activations are

`z[l,j] = (Left[l,j] x_l)(Right[l,j] x_l)`.

Term `j` writes `Down[l,:,j] z[l,j]` into the 1,152-dimensional residual stream. For source `s`, its source-induced
write relative to the equality-absent trajectory is

`u[l,j,s] = Down[l,:,j] * (z[l,j,s] - z[l,j,0])`.

This complete term is unchanged by reciprocal rescaling of its Left/Right/Down factors or by swapping Left and
Right. Product-term indices may still be permuted, so all claims are about downstream effect, not index identity.

## Discovery fingerprint on documents 0:96

For each source `s` and context cell `c`, differentiate that cell's mean CE through the full suffix with respect to
the MLP write `w_l`. Define

`q[l,j,s,c] = - sum_positions <d CE[s,c] / d w_l, u[l,j,s]>`.

The minus sign makes positive `q` mean that removing the source-induced term is predicted to increase CE. This is a
first-order screen, not the causal result. It is computed efficiently as `(gradient @ Down[:,j]) * (z_s-z_0)_j`;
no 1,152-by-4,608 term tensor is materialized.

Let `t=(-1,+1,+1,-1)/2`. Compute fingerprints on the pooled discovery documents and independently on `0:48` and
`48:96`. A product term enters the fixed proposed group for its MLP only if all of the following deterministic
conditions hold:

1. pooled cosine `cos(q[l,j,s],t) >= .70` for both `N` and `H`;
2. pooled source cosine `cos(q[l,j,N],q[l,j,H]) >= .70`;
3. `dot(q,t) > 0` for both sources in each 48-document discovery half; and
4. the two discovery-half fingerprints have positive cosine for each source.

There is no top-K fallback. Terms that fail remain outside the group even if the resulting group is empty. Report
the selected count for every MLP, all fingerprint distributions, and overlap of the selected sets under selection
performed separately on the two discovery halves. The separate-half selections are a stability diagnostic only;
the pooled-discovery selection above is the sole intervention group.

## Exact held-out intervention on documents 96:192

For a selected set `S_l`, replace those product activations at each live MLP call by the same-document absent-
trajectory activations:

`z_live[l,j] <- z[l,j,0] for j in S_l`.

All unselected product terms use the live input, and every later module recomputes normally. This is an exact removal
of the selected source-induced product terms; the gradient is not used during evaluation.

Run all eight subsets of the three proposed MLP groups under both sources. Also run two fixed, matched-count controls
per MLP and their three-module union:

- `amplitude`: the same number of terms with largest discovery RMS of `u`, excluding proposed terms when possible;
- `random`: the same number of non-proposed terms from a SHA256-seeded permutation.

If a proposed group has zero terms, both matched controls for that MLP are empty. The controls use the identical
activation-replacement intervention. No control can change the proposed selection.

For any removed set, define its causal vector as full-source CE effect minus the removed-arm CE effect in the four
fixed cells. Positive coordinates mean the removed terms were useful; negative coordinates mean they suppressed an
over-strong matcher. Parent full-module removal vectors on `96:192` are recomputed in this run.

## Registered predictions

### A. Instrument and parent reproduction

All frozen hashes and model identities hold. Native/replay relative squared error is at most `1e-12`; bilinear
attention reconstruction error is at most `1e-10`; each absent product activation has shape `[batch,256,4608]` and
is consumed exactly once at its named MLP; empty selected sets are exact no-ops; full-module removal effects reproduce
rung 466's validation-half values within `1e-10 nat`; every registered group/subset/control executes exactly once;
SEALED remains closed; and no product fingerprint or selected index is saved outside the result receipt.

### B. A stable proposed split exists

At least two of the three MLPs select at least four terms, the union contains at least 12 and at most 3,456 of the
13,824 available terms, and for at least two MLPs the two independent discovery-half selections have Jaccard overlap
at least `.20`. This is only the stability gate for testing the group, not circuit identification.

### C. Exact held-out removal carries the correction

For both sources, removing the union of all proposed groups on documents `96:192` has the `(-,+,+,-)` sign pattern,
cosine at least `.80` with the parent three-MLP removal vector, and projection magnitude between `.20` and `1.25` of
that parent vector. Native and hybrid proposed-group causal vectors have cosine at least `.80`. Each cosine must be
positive in both 48-document validation halves. The context-vector norm must be at least `.01 nat` for each source,
and its norm must be at least twice the absolute off-target effect.

### D. Task-conditioned selection beats matched-count controls

For both sources, the proposed union's cosine with the parent vector is at least `.15` higher than the larger of the
amplitude and random control cosines, **or** its projection on the parent is at least twice the larger positive control
projection while retaining cosine at least `.75`. The same winning comparison must hold in direction (positive
difference) in both validation halves. This prevents a generic high-amplitude slice of an already task-shaped MLP
from being called the equality component.

### E. The group is genuinely cross-module and its composition is measured

At least two individual MLP groups have held-out causal-vector cosine at least `.60` with their own complete-module
parent vector under both sources, source-to-source cosine at least `.70`, and positive half cosines. The three-module
subset factorial reports every singleton, pair interaction, and the order-3 interaction. It classifies composition as:

- approximately additive if the union-minus-sum-of-singletons norm is at most `.25` of the union norm under both
  sources; or
- stably interactive if that norm is at least `.01 nat` under both sources and the two source interaction vectors
  have cosine at least `.80`.

Prediction E holds if the two-module membership test holds and either registered composition branch is resolved.
Interactions between `.25` relative error and `.01 nat`, or source-unstable interactions, are reported as unresolved.

The strong null is an invalid instrument; no selected term in every MLP; proposed-union context norm below `.005 nat`
for both sources; native/hybrid proposed-union cosine at most zero; or the proposed group losing to both controls on
both alignment and positive projection under both sources.

## Decision rule and price

- A--E pass: identify a held-out, cross-module product-term component of the equality correction on this code corpus;
  next require a fresh-corpus/OOD confirmation before adoption.
- B passes but C/D fails: product-term fingerprints are reproducible but do not define the causal component; stop
  threshold/unit tuning and test the class-projected full bilinear form or a state-level causal quotient.
- B fails: the correction is diffuse or the first-order fingerprints are unstable; do not introduce a top-K rescue.
- C/D pass but E fails: retain a within-MLP split but do not claim reuse across MLPs.

The deployed price is unchanged: zero parameters saved and zero added. Discovery stores temporary fingerprints and
selected indices only for the duration of the run. This rung earns circuit evidence only from exact held-out removals,
control separation, source interchange, and cross-module composition—not from sparsity, rank, or reconstruction.
