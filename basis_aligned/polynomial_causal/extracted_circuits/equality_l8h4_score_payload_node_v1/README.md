# Equality L8H4 score/payload node V1

This package exposes the isolated equality-supported attention term as a
zero-parameter node with three explicit ports:

`write = bmm(score * equality_support, projected_payload)`.

The node can be removed from or installed into the L8H4 attention write.  Its
score and payload ports remain native upstream computations, so this package
extracts the edge executor rather than the entire producer subgraph.  The
support mask is an exact predicate of input-token equality, not a fitted or
behavior-selected activation direction.

The bound code-OOD confirmation tests removal, A8-only installation, and the
four-term bilinear expansion under sums of two live score/payload ports.
