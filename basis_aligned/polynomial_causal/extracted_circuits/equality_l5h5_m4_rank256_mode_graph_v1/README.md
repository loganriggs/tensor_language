# Equality L5H5 M4 rank-256 contracted-mode graph V1

This zero-learned-parameter package replaces the opaque M4 write port with 256
weight-only scalar bilinear modes and writer directions derived from the SVD of
the M4 Down matrix contracted with all four frozen L5H5 Q/K readers.

The basis is float64-certified (`2.06e-13` operator reconstruction error), then
deployed in float32/BF16.  Natural five-write score error/cosine is
`.04462/.99906`; frozen code is `.03818/.99927`; causal recovery is `.89159`;
noncopy damage is `.00155` nat; and an equal-norm one-token roll drops recovery
to `.71098`.

This package is deliberately retained from valid nulls.  Rank 256 exceeds the
preregistered compact ceiling of 64.  It also evaluates all 4,608 native L/R
products, so it compresses the product-to-writer interface rather than product
cost.  Alternating-mode composition failed at `.490`; a prospective
most-additive nested split improved code score composition to `.133` but still
failed, with `.421` behavioral composition error.  Use this as the executable
parent for an explicit child/remainder/interaction graph, not as a claim that
two additive pieces explain M4.
