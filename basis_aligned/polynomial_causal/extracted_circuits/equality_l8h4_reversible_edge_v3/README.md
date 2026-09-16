# Equality L8H4 reversible edge V3

The semantic node is V2's exact deployed-order equality contraction.  V3 adds
one explicitly typed implementation port for BF16 roundoff:

- `split(full, term) -> removed, correction`
- `merge(removed, term, correction) -> full`

The correction is computed in FP32 from the BF16 split and contains no learned
parameters.  It prevents implementation arithmetic from being mislabeled as a
semantic circuit edge while allowing exact removal and reinsertion.
