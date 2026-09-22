# Residual predictability from token identity and position

22 September 2026, 06:21 UTC. Registered before executing this diagnostic. CPU only; no new replacement or GPU work.

Question: are the completed Gaussian Adam correction errors largely predictable from simple observed input conditions? Prior Gaussian derivative analysis found diffuse input-coordinate sensitivity, but that does not answer predictability on actual text states.

Freeze parent CP1001 and both local Gaussian Adam seeds25001/25002. Compute native-minus-candidate residuals on6144calibration states and16384opened evaluation states. Standardize each of outputs4–15 by calibration native target RMS. Fit four descriptive predictors of the residual vector on calibration only: constant mean; position mean; current-token mean shrunk toward global mean with20pseudo-observations; additive position plus shrunk token mean of position-residuals. Unseen tokens get global mean, or position mean in additive model. No regularization sweep or evaluation selection.

Separately use the same calibration grouping to predict squared standardized residual energy (risk). Report actual error share in the10%evaluation states ranked highest by predicted risk, plus seen-token coverage and per-document changes. Tie ordering deterministic; constant risk is not scored as a ranking. Report remaining residual relative to original and constant-only baseline, per output and total. A positive result is an association, not causal identification of responsible variables.

Predictions: both starts' token model removes>=20% evaluation squared error beyond the constant model; position removes>=10% beyond constant; additive calibration risk ranking captures>=30% of evaluation error in its top10%states. Null: simple conditions transfer weakly, or improvements are mostly mean bias. Do not refit using evaluation labels. These are opened documents, not new OOD confirmation.

Controls: planted token/position signals recovered without token leakage; unseen-token fallback; finite values; token hashes agree with each state cache; candidate hashes agree with completed native receipt. Costs are diagnostic table storage only; no tables become part of the proposed circuit. Keep failure evidence and all arms.
