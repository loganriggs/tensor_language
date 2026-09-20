# Fixed primitive decoder comparison — 2026-09-20 22:41 UTC

Question: do root structure or fitted readouts limit four native scalar predictions with the same six primitive quadratic inputs? Keep primitive readers and canonical output directions fixed. Fit on original 2048 calibration rows only, using cached weight-evaluated native quartic targets and checked token/scale provenance. This is empirical data-informed polynomial regression, not weights-only fitting.

Compare frozen primary; refit constant, six primitive and four original root readouts (11 columns); and full degree-two polynomial in the six primitives (28 columns: constant, six primitives, 21 unique products). The dense model increases products from10 to27 and is an upper-capacity diagnostic, not automatically a preferable circuit. Export actual graphs and price coefficients/additions/products.

Use standardized features, ridge0/1e-4/1e-2; primary ridge1e-4 fixed before outcomes. Score all arms on calibration and reused disjoint context64/256 diagnostic panels; no fresh/OOD claim. Baseline constant uses calibration target mean only. Report four individual relative errors, errors relative to centered target variation, and combined error. No diagnostic model selection.

Registered predictions: instrument: token hashes/scales agree, FP32 export replay<1e-5, ridge-zero calibration fit no worse than original and dense no worse than fixed. Readout hypothesis: fixed primary-ridge refit improves mode1 MSE at least10% over frozen baseline on both diagnostic panels. Root hypothesis: dense primary-ridge fit improves mode1 MSE at least10% over fixed refit on both diagnostic panels. Failures distinguish overfitting from capacity when calibration gains do not transfer. Neither failure proves that arbitrary functions of the primitives cannot help. Native behavioral interventions remain a separate test.

## Native follow-up registered after CPU fit, before native outcomes

Run fixed_0.0001 and dense_0.0001 on the reused32-document FineWeb and16-document code context256 confirmation panels, all four native removals plus joint. Same original directions, means, native background and normalization. No fitting or ridge selection. A vector-output adapter is evaluation scaffolding; literal candidate prices come from exported scalar DAGs.

pred_a_replay: native target energies/token hashes match the archived confirmation runs within1e-5, native evaluator replay<1e-5, adapter scalar projection replay<1e-5. pred_b_fixed: code mode1 cosine>.9, relativeeffecterror<.4, and its squared error improves at least10% versus frozenprimary. pred_c_dense: code mode1 squared effecterror improves at least10% versus fixed refit. Also report every mode and any worsened effects without replacing the primary.96nativeforwards total; costs13916coeff/10products fixed,13936/27 dense, plus4608fixed residualwritercoefficients each.
