# Joint routing/value and final-bilinear native-weight replay

CPU two threads, zero model-body forwards,12 synthetic normalized query/source
ports, query position8 and source positions0/7. Use native attention17 QK1/QK2,
rotary semantics, signed current/block0 value mixing, output map, MLP17 and full
unembedding. Never materialize the vocab-by-product folded weight matrix.

The complete quintic numerator is multilinearized with two query slots and
three source slots. On the diagonal, supply actual four query/key RMS factors
and128-squared score scale as per-head external gates. Compare against direct
native-weight normalized attention contributions, then sum both sources.

- A: both source contributions and their sum replay within1e-10 relative error.
- B: implicit full-U residual/residual, mixed and attention/attention expansion
  of the final bilinear layer replays direct normalized computation within1e-10.
- C: all source-slot permutations agree within1e-10 for random distinct slots;
  runtime<=120seconds. Dense/gradient controls must already pass1e-10.

These are algebraic instrument criteria, not learned-structure predictions.
Null is an implementation/normalization discrepancy. No sparse factors, OOD
behavior, removal or reusable circuit are claimed by passing. RMS denominators,
the residual background and each source's head gates are retained explicitly.
Attention is bilinear-product attention without softmax. Source0/7 are causal
for query8; there is no self-source coefficient-geometry claim.

Price: no GPU queue or body execution; checkpoint read-only mmap; small batches
of full-vocabulary outputs, no large high-order tensor or folded U-D matrix saved.
