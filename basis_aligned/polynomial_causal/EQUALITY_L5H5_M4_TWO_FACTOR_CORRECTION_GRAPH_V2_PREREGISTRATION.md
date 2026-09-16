# Equality L5H5 M4 two-factor correction graph V2 precision correction

V1 is invalid: its real-arithmetic three-term identity was evaluated through
native BF16 dot/product operations, and the omitted arithmetic residual caused
`.339/.353` natural/code score-closure error.  The positive behavioral control
also failed.  No V1 node-removal result is interpretable.

V2 keeps the same frozen panels and factor definitions, evaluates the
real-arithmetic factor terms in float32, and adds a fourth explicit node:

`native_child_score - (base + first_effect + second_effect + algebraic_cross)`.

This is the native arithmetic residual, not a learned correction.  The four
nonconstant nodes are first-QK correction, second-QK correction, their
algebraic cross-product, and the native arithmetic residual.

Gates:

1. Full score closure is at most `2e-6` relative L2, and the full behavioral
   additive arm matches authority within `2e-5` relative L2 and cosine at least
   `.99999` in every natural/code subtype and half.
2. Removing each of the four nodes has copy-positive behavioral norm at least
   `.02` of exact interaction removal on both panels and is nonzero in both
   halves.
3. Every node-removal incremental noncopy mean is at most `.01` nat.
4. A one-query roll of the arithmetic-residual node preserves its full score
   norm within 1% but has code behavioral-effect cosine at most `.80` to true
   arithmetic-residual removal.
5. The exact graph has four nonconstant nodes, 16 native Q/K projections, and
   zero learned parameters.

Failure of gate 1 remains invalid.  Later failures are valid node/specificity
nulls.  V1 remains recorded as an implementation failure rather than being
overwritten.

Price: one checkpoint load; 192 frozen natural and 192 frozen code documents;
no fits, gradients, parameter updates, or new text.
