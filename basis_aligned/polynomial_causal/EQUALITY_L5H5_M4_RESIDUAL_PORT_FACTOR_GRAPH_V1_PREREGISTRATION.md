# Equality L5H5 M4 residual-port factor graph V1

The no-oracle raw-port graph is exact but exposes eight Q/K activation ports.
This prospective extraction test moves all frozen Q/K projections inside a
standalone executor.  It accepts four residual corners, the four native full
Q/K weight matrices, head index/width, and rotary context.

On the same frozen 192 natural and 192 code documents:

1. Residual-to-raw projection must be bitwise-equivalent to the native module
   path, and graph score closure must be at most `2e-6` relative L2.
2. Full behavioral replay must match authority within `2e-5` relative L2 and
   cosine at least `.99999` in every subtype and half on both panels.
3. All four node-removal magnitudes and score norms must match the raw-port
   parent within `.002` and `2e-6`, respectively.
4. Noncopy means must match within `2e-5` nat and the rolled arithmetic effect
   cosine within `.002`.
5. The package declares four residual state ports, zero raw-Q/K activation
   inputs, zero oracle scores, zero learned parameters, and 16 native
   projections.

Failure of gates 1–2 is invalid.  Later failure is an extraction-equivalence
null.  The test changes the port boundary, not the projection FLOP count.

Price: one checkpoint load; 192 frozen natural and 192 frozen code documents;
no fits, gradients, parameter updates, or new text.
