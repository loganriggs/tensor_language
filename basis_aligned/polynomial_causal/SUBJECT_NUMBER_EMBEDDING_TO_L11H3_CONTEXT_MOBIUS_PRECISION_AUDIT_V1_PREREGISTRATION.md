# Subject-number context Möbius precision audit V1

Registered after the crossed V1 assay failed its `1e-4` replay gate. V1 used a
128-row batch whereas the prior scalar assay used 64 rows; its replay maximum
error was `0.0010071`, while the synthetic decomposition fixtures passed near
machine precision. This audit tests the batch-size explanation without changing
V1, its bars, or its invalid terminal.

Split the frozen 128-row crossed authority into two 64-row batches: the exact
`near`/`behind` sequence and ordering used by the prior assay, and the analogous
`under`/`above` sequence. Recompute base and removed L11H3 scalars for each batch
and reconstruct the same 32×4 matrix.

The precision audit passes only if:

1. removal geometry is within `1e-5`;
2. the exact-batch `near`/`behind` replay maximum error is at most `1e-4`;
3. V1-minus-audit alpha RMS is at least `1e-6` (the batching change is real),
   while the audit's additive relative-L2 and rank-one energy differ from V1 by
   at most `1e-4` (the scientific decomposition is stable);
4. the unchanged synthetic additive and rank-one fixtures remain within
   `1e-12`;
5. the rebatched interaction is at least `5.0` RMS and at least `1e5` times the
   exact-batch replay RMS.

Passing qualifies the numerical instrumentation and the stability of V1's
descriptive decomposition. It does not retroactively change V1's registered
predictions, license labels as circuit inputs, or establish OOD behavior.

Price: four partial forwards over 256 sequences; no logits, fitting, gradients,
permutations, or parameter updates.
