# Equality L5H5 residual-source graph V1

## Question

The exact L5H5 score node currently consumes four post-projection Q/K
activations. Move its boundary to the single pre-attention residual and ask
whether that residual has a sparse upstream provenance graph for equality.

At the L5 attention boundary, decompose the native residual exactly into six
semantic additive groups:

`E, L0=(A0+M0), L1=(A1+M1), ..., L4=(A4+M4)`.

Every write is captured on the native trajectory and propagated through later
block-lambda coefficients in FP32. A separately typed correction port is the
difference between their sum and the deployed BF16 residual. It is always
retained and is implementation bookkeeping, not a selectable semantic source.
RMS normalization remains inside the extracted executor as an open context
operation; it is not distributed over sources.

## Frozen selection and validation

On all 192 `final_natural` documents, evaluate all 63 nonempty semantic source
subsets at equality-fetch edges. Select among subsets with score relative L2 at
most `.15` and cosine at least `.98` by: fewest sources, then lowest error, then
integer subset mask. If none qualifies, select the lowest-error subset, then
fewest sources and mask. No behavioral outcome is used for selection.

Freeze that support before evaluating all 192 `ood_code` documents. On code,
compare its score with the native L5H5 score and insert it through the already
frozen one-scalar adapter and exact reversible L8H4 executor. Also compute the
complete Möbius expansion over the selected source set with the RMS context
recomputed for each subset; this makes all source interactions explicit rather
than incorrectly distributing normalization.

## Predictions

- **A — exact extracted boundary:** the full six-source residual reconstruction,
  residual-to-score executor, and resulting donor logits each have relative L2
  error at most `2e-6`; the correction port is below `.01` of residual norm.
- **B — sparse natural support:** the frozen selection rule chooses at most three
  of six semantic sources and meets `.15` score error / `.98` cosine on natural
  equality edges.
- **C — OOD score prediction:** without reselection, code equality-edge score
  error is at most `.20`, cosine at least `.95`, and both document halves have
  error at most `.25`.
- **D — OOD causal installation/removal:** selected-support donor recovery is in
  `[.85,1.05]`, differs from the exact factor donor by at most `.10`, every
  registered copy cell and half exceeds `.65`, and both native removal stakes
  are positive.
- **E — selectivity:** selected-support noncopy mean damage is at most `.01` nat,
  and inherited L7H3 wrong-donor recovery remains negative.
- **F — composition/reuse:** the complete selected-source Möbius graph reconstructs
  its joint code score within `2e-6`; each source has a named reusable port and
  the package has zero learned parameters.

If A fails, first audit operation order, dtype, lambdas, residual addition, Q/K
row slicing, rotary layout, and causal masking; it is an implementation failure,
not evidence against sparse sources. If A passes but B/C/D fails, retain the
exact residual executor and record a scientific sparsity/transfer null rather
than changing the support or thresholds.

## Price

One checkpoint load. Ninety-six prefix executions for source analysis (48 per
role) plus five complete code forwards per four-document batch: 336 executions
total. Sixty-three fixed natural subset evaluations and at most 64 code Möbius
evaluations per batch are tensor-only node calls. No fits, gradients, parameter
updates, or new text.
