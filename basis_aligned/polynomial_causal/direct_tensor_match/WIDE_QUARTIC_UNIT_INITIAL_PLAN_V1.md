# Unit raw-parameter initialization control

22 September 2026, 04:36 UTC. Follow-up after all 60 original wide-quartic fits completed. Preserve original predictions and failures.

Run only the ten wide default-Muon cases again, rescaling each raw initial parameter row to unit norm before optimization. This preserves the initial normalized factors and polynomial/readout objective exactly; assert normalized-factor replay below1e-14. Keep seeds, target families, learning rate0.01, default decay0.1,250steps, ridge and all other conventions fixed. No further renormalization of raw parameters after updates. No new capacity or changed target.

Prediction: at least8/10 cases reach5% exact Gaussian value error. Report all value/response/coefficient errors and direction movement regardless of success. This is a parameterization/optimizer diagnostic, not native performance or a chosen replacement configuration. The native queue is unchanged, and the control still uses four factor rows rather than96.
