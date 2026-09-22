# Interpret the queued250-step learner against optimizer-duration controls

22 September2026,03:42UTC. Do not modify the frozen queued runner or retrospectively change its criteria. The matched planted value-loss cohort gives250-step median errors1.41%Adam/14.07%Muon; earlier500-step value-loss cohort gives0.69%/0.83%, all10cases/optimizer below5%. Rates.1/.01 and starts match; mathematically equivalent teacher-minus-parent versus explicitresidual contractions differ computationally, so this is not an exactprefix replay. Native dimensions, output weighting and loss normalizer also differ. This flags duration sensitivity, not a prediction that500steps necessarily fix native fidelity.

When the native result arrives, inspect per-arm25-step objective histories and selected checkpoint positions. If loss is still falling, record underconvergence as unresolved rather than infer an architecture limit. Any continuation must preserve the original result and explicitly distinguish a repeated500-step fit, a resumed optimizer-state trajectory, and a warm start from normalized exported factors. These are different experiments. No additional native sweep is queued from this note alone.

[Original toy results](PAIRED_RESIDUAL_LEARNING_TOYS_V1.json) · [Earlier500-step cohort](LOCAL_QUARTIC_RESIDUAL_TOY_SWEEP_V2.json).

## Additional configuration caveat: normalized-direction movement

A simple known-target control at the actual 96-by-1152 parameter shape finds only about 3.7 degrees of median direction rotation in 250 default-Muon steps at the selected rate. Defaults include shape adjustment and weight decay that differ from Adam. This is not a native result or a revised protocol; it requires caution when interpreting any native optimizer difference. [Details and primary implementation sources](NORMALIZED_OPTIMIZER_GEOMETRY_INTERPRETATION_V1.md).

[Wide quartic controls](WIDE_QUARTIC_OPTIMIZER_INTERPRETATION_V1.md): Adam2/10 and default/RMSMuon0/10 recoveries below5% at1152 dimensions; unit raw initialization improves Muon median99.99%→29.69% but still0/10. Known capacity does not establish optimization adequacy at native width.
