**The newline-associated feature predicts contextual variation in one native quartic readout, but fails 10% fidelity.**

Frozen feature1 of the empirical384 graph, no refit. Match current newline token IDs exactly across different prefixes, choosing nearest position then lowest flattened donor index. Native target is the pure quartic polynomial projected along the feature's vocabulary output writer using the vocabulary Euclidean metric. This tests one computational specification; it is not full-logit fidelity or selective feature-removal causality.

|Panel|Directed pairs / recipient prefixes|Response error|Signed correlation|Predicted/true response norm|Ordinary scalar value error|
|---|---:|---:|---:|---:|---:|
|Original calibration|36 / 21|1.16%|0.99994|1.0046|0.305%|
|Opened second panel|30 / 21|15.32%|0.98962|0.9362|2.66%|

Instrument PASS: native projected scalar independently evaluated via unfurled and folded reader agrees below6.1e-15; compiled root scalar matches whole graph projected along this reader within1.52e-7. Outputwriter vocabulary Gram off-diagonal max1.96e-7. Shape criterion PASS, primary response-fidelity criterion FAIL. Reference response energy is nonzero; a current-token-only function predicts zero change and100%relative error on these exact-token matches. Thus there is transferable contextual variation, while high AUC or ordinary-value agreement would overstate its precision.

Follow-up CPU audit removes every pair touching each prefix, whether recipient or donor. Every second-panel exclusion still fails10%:13.54–17.38%. An in-sample oracle scalar changes15.32%to14.36%, retaining87.85%of squared error. Failure is not explained by a single prefix or just global response amplitude. Thirty directed pairs contain only20distinct unordered pairs; do not treat directed/reversed pairs as independent samples. Calibration has26unordered pairs. This is sensitivity analysis, not bootstrap confidence or independent validation.

Interpretation: the output-shared feature has an explicit native polynomial reader and meaningful within-token state dependence, but is not yet a sufficiently faithful extracted circuit. The basis remains inherited from a data-informed output SVD. A projection-specific prediction does not prove that a feature removal has selective behavioral meaning. Preserve broader native19–29%response failures, calibration compression failure and UTF8support failure. Do not promote to a semantic unit or omit other quartic output directions.

Next decision: a causal behavior claim needs a native observable and controlled manipulation, not just further association. Before another native fit or larger model, separate feature1's inherited approximation error from its contribution to an actual output contrast with same-token controls; full objective remains OOD, extraction, removal, reusable composition and simplicity.

[Plan](ROOT_MATCHED_READER_PLAN_V1.md) · [Scalar predictions](ROOT_MATCHED_READER_V1.json) · [Prefix and scale audit](ROOT_MATCHED_ERROR_AUDIT_V1.json) · [Reader code](audit_root_matched_reader.py) · [Error audit code](audit_root_matched_errors.py).
