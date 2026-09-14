# Coupled writer tail: local architectural constraint

Declared before numerical execution, September 14, 2026, 10:04 UTC.

The preceding full-support gradient screen failed: independent 2048-context
gradients had cosine 0.001 and the fitted step worsened held-out energy 0.456%.
This does not establish stationarity. Instead of expanding that arbitrary
isotropic-edit fit, inspect the architectural producer law omitted from it.

At the attention9 output, child and remainder removals are source-position
scalar fields multiplying the same fixed 1152-dimensional direction stored in
MLP9_CROSSFIRST_RESPONSE_V1_PROGRAM.pt. For five positions, local derivatives
through MLP9, blocks10–16 and the affine input to attention17 span at most five
directions at a fixed background. Finite changes need not share that span.

Implement a CPU functional tail from actual checkpoint weights with float32
RMS epsilon explicitly retained and native bfloat16-rounded rotary tables.
Before using derivative evidence, compare this tail against the existing native
module implementation in FP32 on two fixed synthetic contexts (seeds170223000
and170223001, sequence length5), including inherited first-layer values from
the normalized initial residual. Require relative Frobenius discrepancy <=1e-4.
Do not patch global native normalization or substitute FP64's default epsilon.

Then use FP64 functional arithmetic on those same declared contexts. Normalize
the upstream writer to unit RMS (preserves its span). Five central differences
at step1e-4 define the local derivative basis. Fixed standard Gaussian child
and remainder scalar fields (same generator after background draws) are propagated
at steps1e-3 and5e-4. Measure relative derivative replay and joint linearization
errors. Prediction: derivative replay <=1e-5 against direct directional central
differences; halving step reduces relative linearization error by a factor >=1.5
unless both errors are below1e-8. Report both contexts separately, singular
values and absolute errors. No fitting, native text, compression or global
low-rank conclusion. Cap execution at120 CPU seconds/two threads; a timeout
is inconclusive, not a pass. Preserve all failures before any amended experiment.
