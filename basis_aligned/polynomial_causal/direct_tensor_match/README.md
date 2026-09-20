# Direct weight-tensor decomposition study

Primary research focus: **20 September 15:10 UTC to 22 September 15:10 UTC**, per user direction. [Scope and questions](FOCUS_2026-09-20_TO_2026-09-22.md). This replaces the prior head/branch omission agenda for this period.

We optimize randomly initialized student computations directly against the folded teacher's weights. Exact coefficient losses include symmetric Frobenius and Gaussian function norms; correlated-Gaussian metrics can incorporate covariance estimated from calibration data. No sampled activation targets are used to train the isotropic decompositions.

[Initial scientific findings](INITIAL_FINDINGS_2026-09-20_1522.md) · [Plots](INITIAL_RESULTS_V1.png) · [Original preregistration](PLAN.md) · [Native pilot plan](NATIVE_PREREGISTRATION_V1.md).

| Artifact | Status and scope |
|---|---|
| CORE_CHECK_V1.json | Exact symmetrized/Frobenius, Gaussian quadrature, gradients, repeated-input cancellation passed |
| IMPLICIT_QUADRATIC_CHECK_V1.json | Implicit factor inner products/gradients agree with dense reference |
| COVARIANCE_CHECK_V1.json | Correlated Gaussian quadrature and singular-covariance blindspot passed |
| TOY_SWEEP_V1.json | 120 fits, five structures, two optimizers, two rates, three restarts, matched/wide-sparse variants |
| PLANTED_STRUCTURE_AUDIT_V1.json | Separates functional recovery, feature-space recovery and core support |
| TOY_COVARIANCE_SWEEP_V1.json | 80 fits, isotropic versus synthetic-calibration covariance, matched/wide unpenalized variants |
| NATIVE_QUARTIC_PILOT_V1.json | 96 fits on one native quartic target, two objectives; initial plan prose mistakenly said192 |
| NATIVE_QUARTIC_AUDIT_V1.json | Independent dense polynomial replay and same-target spectral baseline |
| NATIVE_QUARTIC_CONTEXTS_V1.json | Complete:384fits across16contexts; independent per-context fits |
| TREE_RESTARTS_V1.json | Complete:48fits;10,000step matched-rank tree optimization controls |
| NATIVE_FULL_QUADRATIC_V1.json | Complete:12fits; full last-layer joint numerator, CP widths32/128/512 |

JSON snapshots may be partial while a run is live. Read record counts and process/queue state before reporting a completed sweep. Thresholded weights do not establish deployable sparse storage. Native quartic generators and normalized model boundaries remain external dependencies.

Next questions: matched-rank optimization traps, explicit support pruning/refit, sparse-basis selection within a recovered subspace, covariance conditioning versus objective choice, and native activation-covariance metrics. Do not replace these with further data-based circuit screens during the focus period.

Next managed job: NATIVE_FULL_QUADRATIC_V2,24fits comparing objective/rate/width with an explicit radial baseline.

Latest: [Sparse basis recovery and paired covariance conditioning](BASIS_AND_CONDITIONING_2026-09-20_1536.md). Hard four-entry Tucker program improves11.985%to0.00629%error after function-preserving basis search. Paired-coordinate whitening helps somequadratic cases but hurtsquartics at thetestedrate. Nativecovariance capture is queued.
