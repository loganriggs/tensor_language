# Equality L5H5 M4 two-factor correction graph V1

The exact four-port Möbius lattice has 15 terms and cannot be truncated by
order.  The attention score nevertheless factors exactly into two QK products.
Let `a0,b0` be the two factors from derived child ports and `a1,b1` the factors
from native child ports.  Then

`a1*b1 = a0*b0 + (a1-a0)*b0 + a0*(b1-b0) + (a1-a0)*(b1-b0)`.

This prospectively tests the three nonconstant macro-nodes: first-QK
correction, second-QK correction, and their cross-factor product.  The graph is
frozen algebra, not fitted or behavior-selected.

On 192 natural and 192 code documents:

1. The three-node score graph must reconstruct the native child score within
   `2e-6` relative L2, and its full behavioral additive arm must reproduce the
   exact interaction graph's additive removal vector within `2e-5` relative L2
   and cosine at least `.99999` in every subtype and half.
2. Removing each macro-node must produce a nonzero copy-positive behavioral
   effect at least `.02` of the exact interaction-removal norm overall, with a
   nonzero effect in both halves, on both panels.
3. Each node's incremental noncopy mean NLL change must be at most `.01` nat.
4. Rolling the cross-factor score by one query must preserve its score norm
   within 1% but have behavioral-effect cosine at most `.80` to true
   cross-factor removal.
5. The graph has three nonconstant nodes, 16 native Q/K projections, and zero
   learned parameters.

Failure of gate 1 is invalid.  Failure of later gates is a valid graph-node or
specificity null; it does not invalidate the exact algebra.  Report every
node's score norm and behavioral removal effect regardless of verdict.

Price: one checkpoint load; 192 frozen natural and 192 frozen code documents;
no fits, gradients, parameter updates, or new text.
