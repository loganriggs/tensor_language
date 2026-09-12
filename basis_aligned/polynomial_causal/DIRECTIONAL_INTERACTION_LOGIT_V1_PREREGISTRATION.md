# Frozen directional map: native signed logit interaction

Registered after rank64 was selected on the earlier 72 short cue prompts, before scoring these 72 longer prefixes. Same three cue families and six spelling endpoints, two added newsletter/transcription prefix contexts. No score filtering. This is context/position transfer, not corpus OOD.

Six arms, in each context: native; selective head8 removal; remove8 plus actual updated head9 scalar removal; remove8 plus rank64 predicted head9 scalar removal; remove8 plus frozen pristine head9 scalar removal; remove8 plus full directional-map predicted head9 scalar removal. All head9 scalar edits use the frozen physical writer and the original rank64 even-key runtime. 432 body forwards, 120-second managed GPU cap.

Predict the head9 input using only pristine pre-MLP8 residual, native bias-free MLP8 output, native raw attention9 input, and head8 removal amplitude. The exact fixed-writer quadratic response retains changed MLP8 normalization; recompute attention9 normalization and both joint QK routing and value. Rank64 is a weight-only SVD of the existing mixed map. Full mixed map is the positive instrument control. No changed-state oracle enters the predictor.

Signed interaction is the final target margin in the dynamic head9 removal arm minus the corresponding frozen-head9 removal arm. This isolates propagation of the head8 edit into the head9 component, relative to a frozen mediator. It is conditional on the native background and suffix.

A: every family's full-map signed-interaction relative error <=1e-4; actual interaction norm >1e-6; native mean paired cue contrast >=0.2 and >=10/12 positive pairs.
B: A and rank64 signed-interaction relative error <=5% in each family.
C: B and rank64 joint-removal effect error <=1% in each family, where effect is edited minus native target margin.
Report native joint cue coverage and unrelated-margin error descriptively. No 50% coverage threshold is imposed on predictor accuracy. Do not reselect rank from these scores. If the full anchor fails, distinguish native numerical/instrument error before interpreting low-rank failure.
