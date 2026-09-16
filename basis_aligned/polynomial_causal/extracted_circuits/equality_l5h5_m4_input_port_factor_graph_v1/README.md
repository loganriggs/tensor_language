# Equality L5H5 M4 normalized-input factor graph V1

This executor moves the product-port graph through the frozen bias-free MLP4
`Left` and `Right` maps. Its state boundary is the normalized MLP4 input,
retained `M2,A3,M3,A4` writes, and rotary context. It constructs all 4,608
native bilinear products, the four rank-256 residual corners, all native L5H5
Q/K ports, and the four precision-corrected factor nodes internally.

This is an extraction-boundary improvement, not product compression. The
4,608 products and all 16 Q/K projections remain charged, while the product
tensor, residual corners, raw Q/K activations, and scores are no longer input
oracles.

On 192 frozen natural and 192 frozen code documents, 4,076,863,488 internally
constructed product values and 4,076,863,488 residual-corner values are both
bitwise identical to their native authorities. Score closure, behavioral
replay, node removals, noncopy selectivity, and the rolled-arithmetic control
all match the product-port parent exactly.

Under the shared four-trait rubric this is a frozen OOD-predictive,
standalone-extracted, selectively removable, and composition-reusable graph.
Its boundary is weaker than token-input extraction: it consumes five native
activation ports (`mlp4_normalized_state,M2,A3,M3,A4`) plus deterministic
rotary context. The export verifier binds those claims to the source result and
package hashes. No product-width or checkpoint-compute compression is claimed.
