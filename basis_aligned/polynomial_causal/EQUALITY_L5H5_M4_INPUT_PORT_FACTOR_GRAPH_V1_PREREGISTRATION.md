# Equality L5H5 M4 normalized-input factor graph V1

The product-port graph is exact but still accepts the native MLP4 product as a
state oracle. This test moves the boundary through the frozen MLP4 `Left` and
`Right` maps to the normalized MLP4 input.

On 192 frozen natural and 192 frozen code documents:

1. Every internally constructed product value and every residual-corner value
   must be bitwise identical to the native/product-port authorities; score
   closure remains at most `2e-6` relative L2.
2. Full behavioral replay matches authority within `2e-5` relative L2 and
   cosine at least `.99999` in every subtype and half on both panels.
3. All node-removal magnitudes and score norms match the product-port parent
   within `.002` and `2e-6`.
4. Noncopy means match within `2e-5` nat and rolled-arithmetic effect cosine
   within `.002`.
5. The package declares zero product, residual-corner, raw-Q/K, and oracle-score
   inputs, zero learned parameters, selected rank 256, and 4,608 native products
   executed internally.

Failure of gates 1--2 is invalid. Later failure is an extraction-equivalence
null. This changes the graph boundary and does not claim product compression.

Price: one checkpoint load; 192 natural and 192 code documents; no fits,
gradients, parameter updates, or new text.
