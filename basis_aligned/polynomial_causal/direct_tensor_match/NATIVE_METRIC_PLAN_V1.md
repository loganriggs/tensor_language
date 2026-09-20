# Native Gaussian metric comparison — 2026-09-20

Endpoint: full MLP17 unnormalized numerator projected through the unembedding, with preceding output projections folded into the input frame. Same teacher as full quadratic V2, but use the saved covariance capture's float64 QR factor R exactly. This is a polynomial component, not the complete normalized network.

Compare isotropic, centered covariance with zero mean, uncentered second moment with zero mean, and covariance with the actual nonzero mean. Add0.001 times average variance as an isotropic ridge. The last metric uses exact noncentral Gaussian fourth moments. None equals the empirical fourth moment in general.

Sixteen fits: four metrics, widths128/512, two random restarts, Muon learning rate0.005 with cosine decay,600steps. This is a first metric comparison, not an optimizer ranking. Students begin as identical physical random functions across metrics; optimization uses each metric's whitened coordinates, so objective and subsequent optimizer geometry change together. Earlier paired toy whitening controls quantify why this distinction matters.

Report training relative error, isotropic Gaussian and coefficient Frobenius errors, and actual empirical output errors on both captured panels. Fit moments from calibration only. Select restarts by training error before comparing evaluation error. Panels are separate cached document offsets, not a claim of broad OOD generalization.

Predictions: exact metric/coordinate checks pass1e-10/2e-4; width512 noncentral training relative error below0.5; selected noncentral fit improves evaluation error over selected isotropic fit. Preserve failure of any prediction. Null: covariance weighting does not improve actual heldout matching.

The noncentral moment formula is independently checked using27-point Gauss–Hermite quadrature and all parameter gradients in float64. Fixed frame costs remain explicit. No checkpoint edits, native forward calls, activation-regression optimization, CE or causal claims. Registered in runner docstring before enqueue; companion plan written immediately after correcting its initial relative-path error.
