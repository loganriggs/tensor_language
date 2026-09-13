# Cache the existing composed regional predictor for frozen reader validation

13 September 2026. Reuse the completed `COMPOSED_LAST_BLOCK_PREDICTOR_V1` executor, controls and fixed 120 rows: 24 old anchors and four groups of 24 previously tested constructions. No new factors, interventions, semantic hypotheses or OOD rows are introduced.

The original artifact saved outcomes only. Collect final states and matching linear contributions for the additive background, full-head predictor, compact three-contraction predictor and MLP-only control. For the additive background, the linear part is the unrounded `zC+zR-zN`; for generated states it is the actual FP32 input passed to the MLP. Subtracting this linear part and native MLP bias defines the corresponding bias-free contribution, including the additive combination in the background.

Preserve the prior registered checks unchanged:

- `pred_a`: native N/C/R outcomes and additive-background outcomes replay the earlier artifact within 1e-4 relative error.
- `pred_b`: full-head-plus-MLP local-effect error <=0.10 in each of the five groups.
- `pred_c`: compact-head-plus-MLP local-effect error <=0.10 in each group.

This run repeats known positive controls to establish the cache, not a new discovery. Before interpreting compressed readers, independently replay all cached state readouts against the original saved outcomes, including signed effect differences.

Price: 360 full forwards, 360 extra MLP evaluations, 480 final readouts, roughly 9 MB of added FP64 state tensors, 180-second limit. Native model weights, normalization and input/background generators remain. Run only through the managed GPU lane. Preserve the original source and artifact; the new runner changes storage only.

The resulting cache supports the same quadratic-only and whole-unembedding interventions already registered for FineWeb, now on regional target/control margins and full/compact local effects. Freeze each completed grouped program and compare with its weight-only matched global baseline. No refit or selection from the regional panel. This evaluates conditional circuit-effect preservation, not selective removal of a newly identified factor or complete extraction.
