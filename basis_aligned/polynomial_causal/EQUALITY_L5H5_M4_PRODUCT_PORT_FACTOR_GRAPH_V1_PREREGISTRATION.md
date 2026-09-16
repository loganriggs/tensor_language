# Equality L5H5 M4 product-port factor graph V1

The residual-port graph still accepts four constructed M4 corners.  This test
moves their rank-128/rank-256 mode construction inside a standalone executor.
Its dynamic state inputs are the native M4 product tensor, retained
`M2,A3,M3,A4` writes, and rotary context.

On 192 frozen natural and 192 frozen code documents:

1. Every constructed baseline/child/remainder/joint residual value must be
   bitwise identical to the validated corner constructor; graph closure remains
   at most `2e-6` relative L2.
2. Full behavioral replay matches authority within `2e-5` relative L2 and
   cosine at least `.99999` in every subtype and half on both panels.
3. All four node-removal magnitudes and score norms match the residual-port
   parent within `.002` and `2e-6`.
4. Noncopy means match within `2e-5` nat and rolled-arithmetic effect cosine
   within `.002`.
5. The package declares zero residual-corner, raw-Q/K, and oracle-score inputs,
   zero learned parameters, rank 256, and all 4,608 native products external.

Failure of gates 1–2 is invalid.  Later failure is an extraction-equivalence
null.  This changes the graph boundary, not native product cost.

Price: one checkpoint load; 192 natural and 192 code documents; no fits,
gradients, parameter updates, or new text.
