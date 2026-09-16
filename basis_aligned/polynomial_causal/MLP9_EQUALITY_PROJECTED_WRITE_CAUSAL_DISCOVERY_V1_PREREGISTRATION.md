# MLP9 equality projected-write causal discovery V1

## Question

Raw equality-response energy is not stable in a fixed low-dimensional output
basis.  Determine whether its behaviorally relevant part is nevertheless
low-dimensional.  On opened documents 564--627, remove the established L5H5
recipient score, measure the exact resulting MLP9 write difference
`r = write_absent - write_native`, and rerun the absent arm while replacing its
MLP9 write with:

- `write_absent - r` (exact MLP9-write restoration ceiling),
- `write_absent - P_DCT16 r`,
- `write_absent - P_EQ16 r`,
- `write_absent - P_EQ64 r`,
- four fixed random rank-16 projections of `r`.

`P_EQ` comes from the immutable consumer-response discovery artifact, whose
basis was fitted only on documents 500--563.  No projection may be refitted or
rescaled using causal outcomes.

## Predictions

- **A — instrument:** native replay is exact and every installed MLP9 write
  matches its prescribed tensor within `2e-6` relative L2.
- **B — MLP9 write sufficiency ceiling:** exact write restoration recovers at
  least `.30` of the native copy-token NLL effect and changes all-noncopy NLL by
  at most `.01` nat relative to native.
- **C — rank-16 consumer sufficiency:** EQ16 recovers at least `.80` of the exact
  restoration effect, with all-noncopy damage at most `.01` nat.
- **D — rank-64 consumer sufficiency:** EQ64 recovers at least `.90` of the exact
  restoration effect and its absolute copy recovery is at least `.30`.
- **E — behavior-conditioned advantage:** EQ16 copy recovery exceeds DCT16 by
  `.15` and the maximum random rank-16 recovery by `.10`.
- **F — sign/cell stability:** exact, EQ16, and EQ64 have positive recovery in
  both 32-document halves and on near, far, single-predecessor, and
  multiple-predecessor copy cells.

Passing licenses a fresh/OOD projected-write test.  Exact-ceiling failure means
the observed MLP9 response is a reader/diagnostic rather than a sufficient
causal site.  Projection failure with an exact pass means a fixed low-rank raw
write coordinate is inadequate.

## Price

One checkpoint load; 64 opened documents; native, absent, exact, DCT16, EQ16,
EQ64, and four random-rank16 arms; no new text, fitting, gradients, or parameter
updates.
