# Joint input-mode baseline and SVD numerical correction

The recent extracted path still has no qualifying compressed input basis. The red-team check found a numerical failure in the CUDA SVD basis; an independent CPU float64 SVD repairs full-rank replay but does not rescue rank512. These are opened-panel baseline results, not a bound against sparse bilinear DAGs.

## Comparison

All arms execute the same MLP16 -> attention17 -> normalized MLP17 scalar parent, retain the same three upstream ports, and use the v629 scale interventions. Relative error means L2 prediction error divided by target L2; the maximum includes scalar and intervention-change errors. Absolute squared numerators and denominators are retained in the JSON receipts.

| Basis | Rank | Full-program storage fraction | Maximum calibration error | Maximum validation error |
| --- | ---: | ---: | ---: | ---: |
| dense | 1152 | 1.000 | 0.0610 | 0.0569 |
| raw_input | 512 | 0.781 | 2.8136 | 3.0055 |
| raw_input | 1152 | 1.055 | 0.0605 | 0.0562 |
| raw_cpu64 | 512 | 0.781 | 2.8136 | 3.0056 |
| raw_cpu64 | 1152 | 1.055 | 0.0610 | 0.0569 |
| balanced_input | 512 | 0.781 | 2.8108 | 3.0030 |
| balanced_input | 1152 | 1.055 | 0.0605 | 0.0563 |
| joint_tensor_input | 512 | 0.781 | 2.6653 | 2.7898 |
| joint_tensor_input | 1152 | 1.055 | 0.0610 | 0.0569 |

## Numerical red-team result

The CUDA float32 SVD full basis has relative orthogonality error5.65e-4 and full-rank conditional scalar replay error1.59e-3, failing the3e-5 gate. CPU float64 SVD, cast to float32 for the same executor, reduces orthogonality error to3.59e-7 and conditional replay error to9.99e-7. The dense native reference has8.59e-7 error. This isolates the failure to the computed basis, not the native target or factorized executor. The underlying library cause has not been diagnosed.

At rank512, corrected CPU SVD still has maximum calibration error2.81358, versus2.81363 for CUDA SVD. Thus this numerical issue does not reverse the rank512 negative result. Earlier raw-SVD runs remain archived with this qualification; do not treat all their numerical controls as passed.

The joint input-mode Gram is formed from the complete symmetric MLP16 output tensor T_aij, not from separately compressed factors. Its top eigenspace defines P, and the executable evaluates T(PP^T x,PP^T x) through the native factors. This is an input-mode HOSVD baseline; the retained4608 products are unchanged. It is not optimized Tucker, HT, or a learned sparse core. Full-rank joint-basis replay is1.28e-6. Its Gram changes only3.26e-7 relative under deliberately broad left/right channel rescaling.

Native left/right balancing factors range0.959–1.061, so the toy gauge pathology is not a large left/right imbalance in this checkpoint. It was still necessary to check. v630 joint ranks256/512/768 all fail; v631 independently audits rank512 and full rank.

## Claim audit

| Claim | Tag | Evaluation set | Status |
| --- | --- | --- | --- |
| CPU-corrected factorization recovers full-rank parent | fold | opened rows, four interventions | passes |
| Joint-tensor basis is invariant to tested factor gauge | fold | native weights | passes |
| Rank512 saves >=20% and preserves errors <=10% | fold prediction of native edits | opened calibration and validation panels | fails |
| Low-rank input failure rules out sparse interaction circuits | inference | no supporting evidence | unsupported |
| Dense fallback proves compressed validation | inference | same panels | false; gate now requires saving too |

v630 used16native forwards/208local replays; v631 used16/144. Eight existing joint-tensor tests pass. No fitting, no model updates. The aggregate v631 numerical gate deliberately remains false because the known-bad CUDA arms are retained; the per-arm receipts identify the corrected reference.

## Consequence

Use CPU float64 SVD (or another independently verified orthonormal solver) for future spectral references. Require full-capacity recovery before interpreting truncation. Do not keep spending on raw input-rank sweeps: this baseline does not select sparse interactions or merge reusable quadratic computations. A sparse-core/DAG comparison must retain enough input span and earn simplicity by reducing interactions and repeated feature construction. The current parent still lacks selective semantic evidence and useful reuse, so lower error alone would not complete discovery.

[v630 receipt](../bilinear_quotient/circuits/followups/recent_joint_basis_v630_result.json) · [v631 correction](../bilinear_quotient/circuits/followups/recent_joint_basis_v631_result.json) · [baseline ledger](DECOMPOSITION_BASELINES_2026-09-20.md)
