# Rung 579 preregistration: independent CPU audit of R576

R579 will audit the saved R576 result without loading the model or opening any new data. Its purpose is to distinguish
a real scientific null or held result from bookkeeping, aggregation, split, or threshold errors.

The audit will independently reconstruct, from the saved row-level measurements:

1. the exact FIT and conditional SELECT row sets for every family and endpoint;
2. the five numbered-list necessity decisions, including the fixed 2,000-resample bootstrap lower bounds;
3. the three active copy-control decisions using the FIT list scales unchanged on SELECT;
4. the three sequence-successor reuse decisions;
5. the conditional split-opening rule, total forward price, exact-compilation predicates, and terminal decision;
6. all frozen input hashes and the fact that FINAL_TEST/OOD remained closed.

The audit will also report whether copy controls received a nonzero intervention. It may explain a failure but may not
change a threshold, omit a family, reinterpret a failed cell, or promote the claim. It uses zero model forwards and
zero backwards. Its implementation is frozen before the R576 outcome is read.
