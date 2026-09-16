# L5H5 bilinear score node V1

This zero-parameter node extracts the multiplicative score computation used by
L5H5:

`score = causal_mask(dot(q1,k1)/128 * dot(q2,k2)/128)`.

Its inputs are already RMS-normalized and rotary-transformed Q/K ports. Each
dot-product branch can be additively split, and their product expands into a
sparse grid of pair interactions before the same causal mask is applied. The
package does not yet extract the native linear projections or the residual
sources feeding those projections.
