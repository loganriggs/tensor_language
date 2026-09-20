# Extracted scalar features predict native removal effects unevenly

2026-09-20 21:38 UTC

On16 new FineWeb documents, we removed each fixed output-shared component of
the native quartic branch, then compared that intervention with removing the
component predicted by the ten-product program. Both edits use the original
native background, fixed calibration mean, and actual normalization/softcap.

| Feature | Logit-effect cosine | Relative logit-effect error | Token-loss effect error |
|---|---:|---:|---:|
| 0 | 0.991 | 15.8% | 19.5% |
| 1 | 0.965 | 31.3% | 37.0% |
| 2 | 0.907 | 48.3% | 54.3% |
| 3 | 0.816 | 59.0% | 71.8% |
| Joint removal | 0.986 | 19.1% | 16.1% |

All registered logit-effect bars pass. Replay error is2e-15. These are live
interventions: native logit-effect RMS ranges0.283–1.007, not numerical noise.
Feature3 remains weak: its document-bootstrap cosine interval is[.704,.900],
crossing the registered .8 point-estimate bar. Passing the pooled test does not
establish robust identification of every feature.

The native comparison is defined using the learned output projection. It tests
whether our scalar computation predicts that operational component; it does not
prove an intrinsic unique unit or semantic selectivity. Centered-mode removal
also differs from deleting the entire uncentered branch: the retained mean and
other components can cancel, so removal damages are not interchangeable.

The next registered test keeps every candidate and threshold frozen and changes
the input domain to local Python code. Its deterministic16-file panel is built
and hashed. It is a narrow domain-shift test, not a representative code benchmark.

[Results](../../direct_tensor_match/NATIVE_MODE_INTERVENTION_V1.json),
[uncertainty](../../direct_tensor_match/NATIVE_MODE_INTERVENTION_UNCERTAINTY_V1.json),
[domain-shift plan](../../direct_tensor_match/CODE_SHIFT_INTERVENTION_PLAN_V1.md).
