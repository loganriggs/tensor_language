# Rung 529 first smoke: invalid instrument, no scientific outcome

**Managed execution:** 2026-09-03 11:35:22--11:35:30 UTC, runner exit `0`

**Result:** `equality_shared_private_transition_consensus_rung529_gpu_smoke_results.json`

**Result SHA256:** `41256c9281a93da2ab98ffc4495afa180de29e2757f04277eca7c8da0054e232`

The smoke retained no task, CE, or circuit effect. All 37 expected model forwards ran, all 26 constructed edit states
were nonzero, and downstream continuation patches were live. Prediction A nevertheless failed: computing
`private = target - consensus` and then `consensus + private` in FP32 changed a reconstructed boundary by one FP32
unit (`9.5367431640625e-7`). The preregistered requirement is exact reconstruction at the model boundary, so this
smoke is instrument-invalid and cannot license the full run.

The v2 repair performs only the consensus/private construction in FP64, then uses the same single FP32/model-dtype
boundary conversion as before. It changes no action, scale, document, circuit, continuation, control, threshold,
prediction, or execution-price count. A separate namespace and managed smoke are required before any outcome run.
