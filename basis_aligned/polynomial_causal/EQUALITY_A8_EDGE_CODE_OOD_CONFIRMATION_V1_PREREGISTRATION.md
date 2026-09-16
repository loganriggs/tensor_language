# Equality A8 edge code-OOD confirmation V1

## Frozen hypothesis

The opened-panel pre-MLP9 factorial selected singleton support `100`: restore
the native complete attention8 write while retaining the recipient-absent MLP8
and attention9 writes.  It recovered `.92683` of the natural-text copy effect;
MLP8-only recovered `.14238` and attention9-only `.02339`.

Without changing that support or any threshold, apply the same three singleton
oracle-edge interventions to all 192 documents of the separately frozen
`ood_code` role.  These documents are repository-disjoint code contexts frozen
before the model campaign.  The action remains the established L5H5-score to
L8H4 recipient removal.  No OOD outcome is used for selection or fitting.

## Predictions

- **A — instrument:** custom native and absent logits/MLP9 writes reproduce the
  authoritative forwards, and all installed writes have relative L2 error at
  most `2e-6`.
- **B — frozen A8 transfer:** `100` recovers at least `.85` of the aggregate
  code-OOD copy-positive NLL effect and differs from its frozen natural-text
  recovery by at most `.15`.
- **C — preservation:** `100` changes all-noncopy mean NLL by at most `.01` nat
  from native.
- **D — frozen singleton ordering:** MLP8-only and attention9-only each recover
  at most `.35`, and A8-only exceeds each by at least `.40`.
- **E — stability:** A8-only recovery exceeds `.70` in both 96-document halves
  and in near, far, one-predecessor, and multiple-predecessor cells.
- **F — removal remains live:** the absent-minus-native copy effect is positive
  in aggregate and in both halves.

Passing establishes OOD prediction and removal/installation for the frozen
one-edge oracle support.  Because the intervention still copies a row-specific
native attention8 write, it is not yet an extracted executor.  The next step is
to replace that oracle write difference by the already exact L8H4 equality-term
formula and test the same OOD behavior without native-state donation.

## Price

One checkpoint load; all 192 frozen code-OOD documents; authoritative native
and absent forwards, custom native and absent capture forwards, and three frozen
singleton forwards per four-document batch: exactly 336 forwards.  No fitting,
new text generation, gradients, or parameter updates.
