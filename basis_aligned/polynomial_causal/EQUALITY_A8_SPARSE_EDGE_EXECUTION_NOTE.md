# Equality A8 sparse-edge execution and red-team note

The pre-MLP9 three-edge factorial passed all registered predictions.  On the
opened natural panel, the attention8 write alone (`100`) recovered `.9268297`
of the copy-token removal effect.  MLP8-only recovered `.1423781`,
attention9-only `.0233874`, and every pair/third-order recovery interaction was
below `.029` absolute.  Exact native/absent corner and write replay errors were
zero.  This selects a one-edge oracle support; it does not by itself extract
the edge.

Frozen transfer to the separately sealed 192-document code-OOD role was a
valid calibration null, not an instrument failure.  A8 remained dominant,
positive in every cell, and stable across halves (`1.17405/1.18067`), with
`.00660` nat noncopy mean damage.  Aggregate recovery was `1.17756`, however,
versus `.92683` on natural text, missing the registered absolute-drift ceiling
of `.15`.  MLP8-only and attention9-only were negative on code OOD.  Thus edge
identity and sign transfer, but a natural-text scalar does not calibrate its
code-OOD behavioral magnitude.

Two zero-new-parameter extraction implementations then separated algebra from
deployed arithmetic:

1. V1 computed `bmm(score * support, projected_payload)`.  Its bilinear
   composition identity passed (`1.68e-7`), and installed recovery nearly
   matched the oracle (`1.17664`), but projecting before summing changed BF16
   operation order.  Term error was `.002368` and logit replay error about
   `.016--.017`; V1 is therefore a valid projected-payload executor null.
2. V2 computes the deployed order exactly: contract the raw head payload, then
   apply the native output-projection slice.  Term error, removal-logit error,
   and removal-MLP9 error are all exactly zero; its FP32 bilinear composition
   error is `2.87e-7`.  Reinstallation is not bit-exact because BF16 subtraction
   is not invertible: `(full - term) + term` need not equal `full`.  Installed
   recovery is still `1.17494` versus oracle `1.17756`, but logit relative error
   `.01582` fails the frozen `2e-6` exact-install gate.  V2 is consequently also
   a valid null at the stronger exact bidirectional-executor claim.

The retained result is precise: L8H4 now has a sparse, zero-new-parameter,
compositionally reusable edge executor with exact removal and close behavioral
reinstallation on the tested code-OOD contexts.  Exact reinsertion needs either a
canonical graph arithmetic boundary or an explicit rounding/remainder port.
OOD scalar calibration also remains unsolved.  Neither issue licenses changing
the failed thresholds after seeing outcomes.
