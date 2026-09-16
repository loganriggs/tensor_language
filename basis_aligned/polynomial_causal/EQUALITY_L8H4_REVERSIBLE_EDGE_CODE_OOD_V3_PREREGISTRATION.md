# Equality L8H4 reversible edge code-OOD V3

## Question

V2 made the semantic equality term and its removal bit-exact, but exact
reinsertion failed because BF16 subtraction and addition are not inverse
operations.  Preserve that valid null.  Test an explicit graph arithmetic
boundary that keeps semantic and implementation computation separate:

1. compute the exact semantic L8H4 equality term;
2. `split(full, term)` into the removed write and an FP32 roundoff correction;
3. `merge(removed, term, correction)` back to the native BF16 write.

The correction is an implementation port, not a learned or semantic edge.  Run
the same removal and A8-only intervention on all 192 frozen code-OOD documents.

## Predictions

- **A — exact reversible replay:** semantic term, split removal, merged write,
  removal logits/MLP9, and installed logits/MLP9 each replay their authoritative
  counterpart within `2e-6` relative L2.
- **B — behavioral identity:** merged A8-only and oracle A8-only recovery agree
  within `.001` aggregate and `.005` in every cell and half.
- **C — removal:** split removal retains the positive copy effect in aggregate
  and both halves.
- **D — OOD installation:** merged A8-only recovery is at least `.85`, every
  cell and half exceeds `.70`, and noncopy mean damage is at most `.01` nat.
- **E — extraction:** the package and receipt report zero learned parameters
  and distinguish semantic from implementation ports.
- **F — compositionality:** the FP32 live-port semantic four-term expansion has
  relative L2 error at most `2e-6`.
- **G — small arithmetic port:** correction norm is at most `.005` of the full
  attention8-write norm and `.05` of the semantic-term norm.  Nonzero density is
  reported but not thresholded.

Passing establishes an exact reversible graph boundary for this extracted
edge.  It does not solve the earlier natural-to-code magnitude-calibration null
or extract the native score/payload producers.

## Price

One checkpoint load; 192 code-OOD documents; five forwards per four-document
batch, exactly 240 forwards; no fitting, gradients, new text, or parameter
updates.
