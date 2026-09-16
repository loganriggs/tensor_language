# MLP9 contextual DCT causal fresh test V1

## Frozen program

Before opening this panel, freeze the rank-16 contextual DCT package, input
direction 2, diagonal pair `(2,2)`, finite scale `32`, and final-token reader
`logit[21215] - logit[6165]`.  These choices come from the immutable response,
reader-discovery, and scale-discovery receipts.  Freeze three collateral token
contrasts from the other diagonal reader candidates:

- `23482 - 23095` (direction pair 0,0),
- `44178 - 36538` (direction pair 1,1),
- `30724 - 30175` (direction pair 3,3).

Evaluate 64 score-blind prefixes created only after all choices above were
fixed.  For `h(z) = z + MLP9(RMS(z))`, form `h0=h(z)`, `h1=h(z+32v)`,
`h2=h(z+64v)`, the additive state `ha=2h1-h0`, and packaged interaction
`p=32² node(z)[2,2]`.  Run the exact native suffix from `ha+p` (installation)
and `h2-p` (removal).  Compare with the exact local interaction `h2-ha`.

Eight deterministic isotropic residual vectors matched to `||p||` are frozen
as removal controls independently for each prefix.  They may not be selected
or rescaled after outcomes are visible.

## Predictions

- **A — instrument:** MLP9 and batched suffix replay are each within `2e-6`.
- **B — OOD local prediction:** packaged versus exact finite MLP9 interaction
  is at most `.02` relative L2 overall and in every family, with overall cosine
  at least `.999`.
- **C — OOD causal installation:** the packaged installed all-logit local effect
  is within `.03` relative L2 of the exact installed local effect overall and
  `.05` in every family, with overall cosine at least `.999`.
- **D — removal:** after packaged removal, the remaining all-logit finite mixed
  effect is at most `.10` of the native mixed effect overall and `.15` in every
  family.  The frozen reader obeys the same `.10/.15` bounds with perfect
  aggregate family sign.
- **E — specificity/collateral:** packaged removal changes the frozen reader at
  least four times the median and twice the maximum of the eight matched random
  removals.  Each of the three frozen collateral contrasts changes by at most
  `.50` of the target-reader change.
- **F — extraction and reuse remain bound:** the package still has one native
  activation port, isolated replay at most `2e-6`, and the immutable rank-16
  receipt has all six predictions true, including its eight mixture reuse gate.

Passing establishes the four project traits only for this explicit synthetic
local-interaction primitive: fresh-context effect prediction, standalone
execution, causal installation/removal of the induced interaction, and frozen
bilinear reuse.  It does not make the token contrast a natural-language
behavior, remove a naturally occurring feature, eliminate the native suffix,
or constitute a complete sparse model circuit.

## Price

One checkpoint load; 64 new prefixes; 14 suffix states per prefix (six primary
and eight random removals); no gradients, fitting, parameter updates, donor
prompts, or adaptive controls.
