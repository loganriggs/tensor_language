# Matched-strength control for five-source modal-null edits

The frozen five-source gradient-null edit suppresses three modal-gradient contrasts but usually retains only a small fraction of the original unit-B number effect. This control tests whether reduced collateral is explained merely by weaker target effects.

For each example, use the frozen analytic quadratic number response to choose a scale t in[0,1] on B=(0,0,1,1,1), matching the predicted full-null number effect. Select the smallest valid root; if none exists, choose the closest endpoint/stationary point and mark it unmatched analytically. No new native outcomes determine scales.26/192 inputs have no valid bounded root. Run the full native suffix for unitB, null_full and matched_B, with a common baseline, and preserve all cells.

Instrumentation passes: prior unit/null effects replay exactly; native-versus-float64 reference error4.51e-6. Actual counts24prefix/64native+64reference suffix, no new derivatives or outcome fitting. Existing native-capability failures remain.

**The all-cell test fails.** Only22/32 cells match number-effect vectors within10% of the null number norm; worst mismatch48.71%. It would be invalid to report an all-cell matched-strength selectivity success.

The22 matched cells are nevertheless informative: all have material modal collateral in the control (at least1% of the null number norm), and all show more than twofold collateral reduction with the null direction. Null/control maximum-modal-norm ratios range0.0253–0.3377. The comparison uses the maximum of the three modal contrast norms consistently. This is a conditional diagnostic of selective direction geometry at reduced strength, not a post-selection rescue of the preregistered gate. Original80%target-retention failure persists.

A separate root diagnosis links strength mismatches to bounded-root availability, distinguishing control range limitations from finite nonlinear error. It does not change amplitudes or exclude rows. Modal preservation here covers only three contrasts; semantic/task controls and prospective inputs are still needed. Full native source and derivative generators remain required.

Receipts: ../bilinear_quotient/circuits/followups/matched_strength_null_v1_result.json; MATCHED_STRENGTH_AMPLITUDES_V1.json; MATCHED_STRENGTH_NULL_CPU_AUDIT.json; MATCHED_STRENGTH_ROOT_DIAGNOSIS.json. The managed runner and amplitude selector preserve the registered thresholds and original failed full-strength result.
