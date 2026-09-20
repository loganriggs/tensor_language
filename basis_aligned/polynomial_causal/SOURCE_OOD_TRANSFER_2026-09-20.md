# Prospective source-response transfer: new vocabulary and constructions

The full five-source quadratic response procedure, modal-null LP rule, model and noun-number decoder were frozen before generating new outcomes. Rows were generated and hashed before GPU execution. The new test has48texts and96 subject/attractor sites: driver, nurse, dancer, sailor, mechanic and musician (with plurals), in two constructions:

- The {subject}, according to the {attractor},
- After speaking with the {attractor}, the {subject}

The latter reverses the earlier subject/distractor order and places the readout on the subject token. Vocabulary is new to this five-source series, not claimed unseen in the overall project or model training. Exact row overlap was checked against the preceding source/subject-control panels. Opposite-number and congruent-number distractors remain separate.

All15 single/pair source settings are evaluated, plus unitB, all5, full-null and half-null edits:19 arms. Given analytic linear coefficients, the positive single/pair outer products span all15 symmetric five-source quadratic coordinates. This does not identify linear and diagonal quadratic terms jointly from Boolean samples; the linear terms are independently differentiated.

## Results

Native agreement capability is100% in all16 distinct family/role/panel cells. Instrumentation passes: native/reference effects7.13e-6 maximum, finite-difference gradient/Hessian and symmetry gates pass, monomial design rank15. Calls:12prefix,192float64 reference suffix,160native suffix,32gradient and160Hessian-row reverse passes,96 bounded LPs.

The preregistered full-quadratic prediction gate passes every arm/cell. Maximum number error6.409%; maximum modal error2.129% of the corresponding number-effect norm. Restricting to the15 independent single/pair settings, maximum number error4.467%. This is prospective transfer of the response procedure; it still computes fresh native derivatives and source contexts for every input. It is not a frozen coefficient table or autonomous token-only circuit.

The selective-retention gate fails again. Full-null modal collateral is at most4.060%, but only1/16cells retains at least80% of the unitB target effect. Minimum aligned retention12.039%. The failure generalizes alongside the predictor's success. No threshold was relaxed or cell excluded; these rows are now opened.

## Fixed-rule baselines

After opening the primary outcomes, an independent CPU comparison applied the already defined signed spectral rank-two rule and linear tangent rule. No rank selection or finite-outcome fitting occurred. This baseline comparison is posthoc; the full-quadratic test above was preregistered.

| Predictor | Values/context, four outputs | Worst number error | Worst modal error / number budget | Gate |
|---|---:|---:|---:|---|
|Full symmetric quadratic|80|6.409%|2.129%|Pass|
|Per-output signed spectral rank2|68|7.527%|3.568%|Pass|
|Linear tangent|20|61.473%|13.406%|Fail|

Full-quadratic CPU replay passes1e-12. Native context/derivative generation remains fully charged for every method. Rank-two eigenspaces are per-context and per-output; passing does not establish a shared semantic feature dictionary, selective removal, or compositional reuse. This strengthens the conditional predictive baseline while keeping the complete-circuit goal unresolved.

Evidence: SOURCE_OOD_V1_ROW_AUDIT.json and both frozen SOURCE_OOD_V1_*_ROWS.json files; ../bilinear_quotient/circuits/followups/source_ood_v1_result.json; SOURCE_OOD_V1_CPU_AUDIT.json; SOURCE_OOD_BASELINES_V1_RESULT.json. Executors: prepare_source_ood_v1.py, managed run_source_ood_v1.py, audit_source_ood_v1.py and audit_source_ood_baselines.py.
