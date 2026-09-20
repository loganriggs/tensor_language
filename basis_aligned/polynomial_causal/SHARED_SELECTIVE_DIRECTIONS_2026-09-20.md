# Shared selective directions and the fixed all-five control

The per-input modal-null LP chooses a different five-source amplitude vector for every context. We tested whether a single vector could replace that dependence. On calibration constructions near_greeted/outside_called, let n_c=-G[c,number,:], M_c=-G[c,modal,:], b=(0,0,1,1,1), and s_c=sign(n_c.b). Maximize t over a in[-1,1]^5 subject to s_c*n_c.a >= t*abs(n_c.b) and abs(M_c.a)<=0.1*s_c*n_c.a in every calibration example. This is a six-variable LP, using no finite-outcome labels.

Common and separate subject/attractor LPs each give zero worst-case linear retention. Primal-dual gaps are zero within floating-point reporting, stationarity below8e-16. This is a restricted numerical infeasibility result for positive uniform retention under these per-input first-order constraints, not a statement about all shared nonlinear circuits or finite group metrics. Stacked modal gradients have rank5 even within role; therefore an exactly shared nonzero modal-null vector is also absent in this source interface.

A planted common selective direction recovers retention1, unchanged by output-sign flips. Greedy deletion gives small conflicting native sets: four subject-plural examples, and two attractor examples differing only in grammatical number:

- The cook that the judge outside the office called earlier
- The cooks that the judges outside the office called earlier

Removing either attractor example restores positive feasible retention. These witnesses are deletion-minimal under the chosen order/tolerance, not proven globally smallest. The result diagnoses context dependence of these coarse source coordinates; it does not deny the existence of semantic features elsewhere in the network.

## Number-stratified exception and native test

Allowing separate vectors by role and subject number gives one nonzero solution: singular-subject calibration reaches t=1.4338 with a=(1,1,1,1,1). The other three strata retain zero. This simple fixed edit needs no per-input derivative computation to choose amplitudes, though source-port generation remains native.

The native all-five test evaluates all32cells and preregisters the eight singular-subject cells as its restricted primary target. Instrumentation passes: native/reference4.73e-6, prior unitB replay exactly zero;24prefix and48native+48reference suffix calls. The all-five edit has1.55–1.68x aligned unitB target strength and full-quadratic number prediction error<=1.82% in the primary stratum.

**Selectivity still fails.** Seven of eight primary cells satisfy the10%modal bar; congruent earlier_noticed|singular reaches10.422%. This is not rounded down or removed. The B baseline itself also passes seven of eight. The median all-five/baseline modal-to-number ratio is0.912, a modest specificity change rather than a demonstrated broadly selective new circuit. Fixed amplitudes alone do not repair the behavioral gate.

All outcomes are opened; construction exclusion applies to direction discovery, not prospective OOD evaluation. Other roles/numbers are retained in the native receipt. Full native-capability failures, source generation, and stronger semantic controls remain outstanding. Original per-input modal-null strength failures remain unchanged.

Evidence: SHARED_SELECTIVE_DIRECTION_V1_RESULT.json; SHARED_SELECTIVE_DIRECTION_CONFLICT_RESULT.json; SHARED_DIRECTION_NUMBER_STRATIFICATION.json; ../bilinear_quotient/circuits/followups/native_all_five_source_v1_result.json; ALL_FIVE_SOURCE_SPECIFICITY_AUDIT.json. The corresponding LP/conflict executors and managed native runner preserve numerical controls and the restricted scope.
