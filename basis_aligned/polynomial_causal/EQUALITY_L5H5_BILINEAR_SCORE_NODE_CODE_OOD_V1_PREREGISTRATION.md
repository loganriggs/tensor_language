# Equality L5H5 bilinear score-node code-OOD V1

## Question

The frozen L5H5 score adapter now drives the exact L8H4 equality node at
`.97287` code-OOD recovery.  Replace the opaque L5H5 score tensor by an
independent zero-parameter reconstruction from its four actual post-projection,
post-RMS, post-rotary ports:

`dot(q1,k1)/128 * dot(q2,k2)/128`.

Insert that reconstructed score through the already frozen scalar adapter and
exact L8H4 raw-payload/output node on all 192 code-OOD documents.  Compare with
the prior factor-capture implementation.  No Q/K direction, scale, or source is
fit or selected.

## Predictions

- **A — exact score execution:** reconstructed L5H5 score and resulting donor
  logits each match the factor-capture implementation within `2e-6` relative
  L2; all terms and removal effects remain live.
- **B — behavioral identity:** extracted-score and factor-capture recoveries
  differ by at most `.001` aggregate and `.005` in every cell and half.
- **C — OOD causal use:** extracted-score recovery lies in `[.85,1.05]`, every
  registered cell and half exceeds `.70`, and noncopy mean damage is at most
  `.01` nat.
- **D — compositionality:** splitting each dot-product branch into its first and
  second 64 coordinates and expanding the resulting `2x2` product grid has
  relative L2 error at most `2e-6`.
- **E — extraction:** package and receipt report zero learned parameters and
  four named Q/K input ports.
- **F — inherited specificity:** the bound parent retains its passed L7H3
  wrong-donor control and one-scalar frozen adapter without alteration.

Passing extracts the score-computation node and moves the remaining upstream
boundary to residual-to-Q/K projection plus normalization/rotation.  It does
not yet identify a sparse residual-source support for those Q/K ports.

## Price

One checkpoint load; 192 code-OOD documents; four forwards per four-document
batch (native, absent, factor-capture donor, extracted-score donor): exactly
192 forwards.  No fits, gradients, new text, or parameter updates.
