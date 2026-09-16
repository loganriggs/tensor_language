# MLP9 contextual DCT node V1

This package evaluates the exact RMS-conditioned MLP9 second-derivative
coefficient for four frozen, behavior-blind DCT input directions, then writes
it through a frozen low-dimensional output basis.  It returns the complete
symmetric 4×4 table of mixed responses, not only diagonal terms, so arbitrary
linear combinations of the four inputs compose by ordinary tensor contraction.

Activation ports:

- `state`: native pre-MLP9 residual `z9`, shape `[..., 1152]`.

Frozen parameters:

- full MLP9 `Left` and `Right` matrices;
- the MLP9 `Down` matrix projected to the selected output coordinates;
- four input directions and their precomputed left/right projections; and
- the selected output directions and RMS epsilon.

The executor calls no language model, tokenizer, suffix, behavioral reader, or
donor activation.  It outputs sixteen contextual mixed-Hessian response
vectors.  This is a generic nonlinear response node, not a CrossFirst
behavioral circuit.

Run `python check.py <this-directory>` after `weights.pt` and `fixture.pt` have
been exported.
