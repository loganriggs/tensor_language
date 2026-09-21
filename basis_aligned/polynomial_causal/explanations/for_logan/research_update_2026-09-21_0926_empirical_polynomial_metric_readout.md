# Empirical polynomial fitting helps training more than transfer

21 September 2026, 09:26 UTC. Follow-up to [the higher-moment diagnostic](research_update_2026-09-21_0917_covariance_and_lifted_polynomial_metrics.md).

**We implemented the richer metric and tested it, while retaining a coefficient-space penalty.** It improves the fitted examples, but the registered setting still fails component fidelity and the transfer-improvement requirement. Increasing its weight eventually overfits. This is a controlled failure of the fixed-feature readout experiment, not proof that higher-moment objectives or jointly learned feature directions cannot help.

## What changed

We kept the399-product graph's input directions and connections fixed. Its367mixed products serve all six source reads;32private squares serve only the sixth. Only output coefficients were fitted, under their existing connection constraints. Every candidate still stores896,198floating coefficients.

The coefficient term reconstructs all six original quadratic matrices. Two empirical additions were compared:

- **Source:** squared errors of the centered quadratic source reads on actual training states.
- **Downstream:** squared linearized errors of the three component values, using each training state's original sensitivity to its two source reads.

For a component phi=AB, the latter uses

$$
\delta\phi_{\rm lin}
=-\frac{B}{2s}\,\delta q_a
+\frac{A}{s}\,\delta q_b.
$$

The fitting objective is a normalized coefficient error plus lambda times the normalized empirical term. The empirical term retains actual fourth moments and, in the downstream case, their correlation with sensitivity. It is evaluated using the graph's product features, without constructing a664,128-by-664,128matrix.

With directions fixed, both objectives are quadratic in the output coefficients. We solve their constrained normal equations exactly. Component pairs decouple for this coefficient metric, while the private square branch remains restricted to the sixth read. Five synthetic problems per objective—ten checks—verify the dense objective, its gradient and the centered-feature design.

We tested lambda0,0.01,0.1,1and10for each family. The registered primary is **downstream0.1**. The coefficient-only control is recomputed under this experiment's isotropic output metric; it is not silently treated as the previous half-output-weighted parent's objective. Both lambda-zero controls agree with the independently implemented coefficient solve.

## Results at fixed feature directions

The fitting set is the original1536states. The opened diagnostic set has448states and is not fresh. Percentages below are actual third-component value errors, not the linearized training loss.

| Objective / weight | Fitting states | Opened states | Original coefficient error |
|---|---:|---:|---:|
| Coefficient only |15.61%|16.09%|8.63%|
| Source,0.1 |15.01%|15.83%|8.64%|
| Source,1 |13.11%|15.41%|9.01%|
| Source,10 |11.11%|16.42%|10.79%|
| Downstream,0.1—primary |12.71%|15.57%|8.69%|
| Downstream,1 |8.77%|16.09%|9.29%|
| Downstream,10 |6.29%|18.96%|11.40%|

The primary's opened errors for all three components are **2.53%,2.52%,15.57%**. It retains the requirement that original coefficient error be at most1.10times the coefficient-only control. It fails the third-component requirement: both the15%absolute cap and the1.10-times-separate-baseline cap remain binding.

Stronger downstream weighting reduces fitting error dramatically while transfer stops improving. At lambda10it also violates the coefficient-fidelity guard. That is an overfitting warning, not evidence that the target component can be represented faithfully at this graph's cost.

All ten fits completed on CPU in22.27seconds. Normal equations, compact execution, literal storage and the coefficient-only control passed their checks. Input directions were not optimized in this experiment.

## A larger out-of-fit calibration check

Before fitting these readouts,232additional64-token prefixes had already been captured to estimate input covariance. They exclude all32original calibration/evaluation prefixes. We evaluated the fitted readouts there without changing them.

These are **out of this readout fit**, but not fresh or fully held out: their input statistics were already used elsewhere in constructing the metric. The cache retains the later read needed for component3only, so this diagnostic cannot validate the other two components or the complete circuit.

| Objective / weight | Third-component error on232additional prefixes |
|---|---:|
| Coefficient only |12.49%|
| Source,0.1 |12.22%|
| Source,1 |11.65%|
| Source,10 |12.29%|
| Downstream,0.1—primary |12.00%|
| Downstream,1 |11.99%|
| Downstream,10 |14.09%|

The primary improves3.93%relative to the coefficient-only control, failing the registered10%improvement requirement. The deterioration under heavy empirical weighting transfers to this larger pool as well. No alternative weight is retrospectively promoted based on this table.

## What this establishes

The richer empirical metric can move the fitted function in a useful direction without immediately sacrificing coefficient fidelity. But its gains are much larger on the fitted examples, and the current primary still fails the complete scalar comparison. This distinguishes a real metric effect from an acceptable circuit repair.

The next comparison should address calibration coverage or jointly adapt the feature directions under a retained global coefficient constraint. Repeating an unconstrained empirical fit would ignore both the observed transfer gap and the earlier lifted-metric nullspace controls. All six original source reads, individual components and combined native interventions must remain visible.

No new native model forward, fresh/OOD confirmation, semantic identification or upstream closure occurred here. The existing fresh-intervention failure remains authoritative for the earlier frozen candidate; these new coefficient fits are not adopted.

## Evidence

- [Ten dense/design/gradient checks](../../direct_tensor_match/EMPIRICAL_PAIR_METRIC_PREFLIGHT_V1.json).
- [All ten native readout fits and registered gates](../../direct_tensor_match/EMPIRICAL_PAIR_READOUT_V1.json).
- [Transfer to the existing expanded calibration pool](../../direct_tensor_match/EMPIRICAL_PAIR_TRANSFER_V1.json).
