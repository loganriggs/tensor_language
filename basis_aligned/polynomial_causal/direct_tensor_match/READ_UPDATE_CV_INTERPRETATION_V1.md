**Calibration-only regularization nearly switches off the coefficient update**

A conditional leave-one-chunk-out comparison used only the seven calibration chunks to select a ridge penalty. The grid included no update, unregularized least squares, and powers of ten from 1e-6 through 1e4. Each fold scaled design columns using its training rows only. Selection minimized pooled held-out error energy; other historical chunks did not participate in selection.

The selected penalty is 1000. Cross-validation relative error improves by only 0.0194%, failing the registered 5% improvement target. Refitting all seven chunks yields generated component-three error 8.8643%, versus 8.8715% before the update. Error on the other 25 historical chunks changes from 7.70249% to 7.70151%; the no-deterioration prediction passes, with a negligible gain. This is not a meaningful repair of the native transfer failures.

An independent execution check changes the second quadratic read and recomputes the generated component directly. It agrees with the linear design to 5.70e-15 relative error. The use of linear least squares is exact only because the first read and supplied normalization/context remain fixed.

The successor spectral audit records the standardized design eigenvalues, effective degrees of freedom at the selected penalty, and each fold's individually preferred penalty. These explain the strength and stability of regularization without fitting to the other chunks. No new program is exported or adopted.

**Scope and decision**

The feature directions and other graph parameters were previously constructed from historical data. This is conditional cross-validation of the latest coefficient update, not independent cross-validation of the full discovery pipeline. Historical chunks are not verified independent documents. Original fidelity constraints were deliberately not used during folds, avoiding their dependence on the held-out chunk; consequently no constrained-fidelity claim is made.

Stop treating a larger coefficient-only fit in this frozen 32-direction bank as the leading repair. The earlier train-only gains largely disappear under chunk separation. This does not prove that more expressive graph edits, independently calibrated features, or a full joint tensor decomposition would fail. The next substantive structural study should separate feature discovery from coefficient selection and evaluation, and return to the larger folded computation rather than count this tiny conditional gain as a circuit result.

[Preregistered comparison](READ_UPDATE_CV_PLAN_V1.json) · [Results and spectral audit](READ_UPDATE_CV_V1.json) · [Implementation](crossvalidate_read_update.py).
