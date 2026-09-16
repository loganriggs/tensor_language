# Equality L5H5 causal bilinear score-node code-OOD V2

## Correction question

V1 reconstructed the two bilinear Q/K dot products but omitted the native
lower-triangular causal mask. It consequently reported `1.0932` relative score
error while producing exactly identical donor logits and `0.97287` recovery:
the extra upper-triangle entries were excluded later by the induction support
mask and were causally inert.

Add the missing deterministic causal mask to both direct execution and the
`2x2` branch-composition expansion. Repeat the frozen 192-document code-OOD
comparison without changing Q/K ports, the frozen score adapter, or the L8H4
executor.

## Predictions

- **A — corrected exact execution:** the causally masked reconstructed score
  and resulting donor logits each match factor capture within `2e-6` relative
  L2, with live terms and positive removal effects in both halves.
- **B — behavioral identity:** extracted and factor-capture recoveries differ
  by at most `.001` aggregate and `.005` in every registered cell and half.
- **C — OOD causal use:** aggregate recovery is in `[.85,1.05]`, every cell and
  half exceeds `.70`, and noncopy mean damage is at most `.01` nat.
- **D — compositionality:** the causally masked expansion of the two additive
  half-dot branches into a `2x2` product grid has relative error at most
  `2e-6`.
- **E — extraction:** the package still has zero learned parameters and four
  Q/K inputs plus one score output.
- **F — inherited specificity:** the bound parent retains its passed wrong-
  donor control and frozen one-scalar adapter.

Failure after this correction counts scientifically only after checking tensor
shape, dtype, rotary ordering, dot-product divisor, and mask orientation against
the native factor implementation.

## Price

One checkpoint load; 192 held-out code documents; exactly 192 forwards. No
fits, gradients, new text, or parameter updates.
