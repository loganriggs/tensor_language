# Equality L5H5 L2–L4 module-write graph V1

## Question

The correction-free sparse parent graph consumes three native layer-write
ports: `L2`, `L3`, and `L4`. Split each layer port into its actual attention and
MLP residual writes:

`A2, M2, A3, M3, A4, M4`.

All writes are captured on the native trajectory and propagated through later
block-lambda coefficients exactly as in the bound parent. RMS remains a shared
operation inside the extracted residual-to-score node. No roundoff correction
is supplied.

## Frozen selection

On 192 `final_natural` documents, evaluate all 63 nonempty module-write subsets
against the parent `L2+L3+L4` score on equality-fetch edges. Qualifying subsets
have relative L2 at most `.10` and cosine at least `.995`. Select by fewest
writes, then lowest error, then integer mask. If none qualifies, select lowest
error, then size and mask. No behavioral outcome participates.

Freeze the support before opening 192 `ood_code` documents. Compare it both to
the parent three-layer score and to the full native L5H5 score, then install it
through the frozen adapter and reversible L8H4 node. Compute the complete
selected-write Möbius expansion with RMS recomputed for every subset.

## Predictions

- **A — lawful refinement:** the all-six module-write score matches the parent
  three-layer score within `2e-6` relative L2, all terms are live, and the bound
  parent remains valid.
- **B — sparse natural refinement:** at most four of six writes qualify under
  `.10` error / `.995` cosine.
- **C — OOD prediction:** frozen parent-score error is at most `.15`, cosine at
  least `.98`, and both half errors are at most `.20`. Error versus the complete
  native score is reported but cannot be better than the parent by assumption.
- **D — causal use/removal:** code recovery is in `[.80,1.05]`, differs from the
  correction-free parent recovery by at most `.08`, every registered copy cell
  and half exceeds `.55`, and removal stakes in both halves are positive.
- **E — selectivity:** noncopy mean damage is at most `.01` nat and the inherited
  wrong-donor control remains negative.
- **F — composition/reuse:** complete selected-write Möbius closure is at most
  `2e-6`, with named module-write ports and zero learned parameters.

Failure of A is an implementation/order/dtype error until disproven. A/F pass
with B/C/D failure is a valid module-write sparsity or transfer null; do not
retune thresholds or regroup attention with MLP after observing it. Passing
still leaves each selected native module write as an external port rather than
recursively extracting its heads, values, or MLP products.

## Price

One checkpoint load; 96 prefix executions plus four complete code forwards per
batch: 288 executions. No fits, gradients, parameter updates, or new text.
