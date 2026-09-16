# L5H5 residual correction-port red-team V1

## Risk

The residual-source graph selected `L2+L3+L4` and passed OOD behavior, while an
explicit FP32-to-BF16 correction port had only `.00365` of residual norm. That
norm is not sufficient evidence of semantic inertness: per-head Q/K RMS can
remove amplitude and preserve the direction of a tiny vector.

## Test

Repeat all 63 natural subset scores with the correction set to zero and freeze
the same registered selection rule. On code, evaluate the frozen correction-
free support against the true full score. Then behavior-test two variants
through the frozen L5H5 adapter and exact L8H4 executor:

1. correction zeroed;
2. correction rolled by one token position, preserving its exact norm but
   breaking document-position alignment.

The native, absent, and exact factor-donor arms remain unchanged. Recompute the
complete correction-free Möbius graph for the newly frozen support.

## Predictions

- **A:** parent receipt and exact instruments remain valid; all arms and terms
  are live.
- **B:** correction-free natural selection chooses the same `L2+L3+L4` support
  and still meets `.15` relative score error / `.98` cosine.
- **C:** correction-free code score error is at most `.20`, cosine at least
  `.95`, and half errors at most `.25`.
- **D:** correction-free code recovery is in `[.85,1.05]`, all registered cells
  and halves exceed `.65`, and noncopy mean damage is at most `.01` nat.
- **E:** the correction is operationally dispensable: zero-correction recovery
  differs from the parent selected recovery by at most `.03`; zeroing worsens
  code score error by at most `.03`; and position rolling changes recovery by
  at most `.05` relative to zeroing.
- **F:** the correction-free Möbius graph closes within `2e-6`.

If B–E fail with A/F passing, the sparse support remains a valid
correction-conditioned boundary graph but must not be described as three
semantic sources alone. Inspecting this control cannot change the V1 support or
thresholds.

## Price

One checkpoint load; 96 prefix executions plus five complete code forwards per
batch: 336 executions. No fits, gradients, parameter updates, or new text.
