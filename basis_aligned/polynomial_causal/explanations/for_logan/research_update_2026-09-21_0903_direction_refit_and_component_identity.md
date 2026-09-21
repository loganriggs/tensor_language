# Refitting directions helps, but useful private computations are not yet stable units

21 September 2026, 09:03 UTC. Follow-up to [global sharing and graph edits](research_update_2026-09-21_0853_global_sharing_and_targeted_graph_edits.md).

**Jointly fitting the input directions improves the edited graph, but it still fails accuracy.** Its private correction is useful, yet differs substantially between fitting starts. We also checked an exact ambiguity in the original three-component decomposition: because the components share one output direction, their factors can be rotated while preserving their summed computation. This makes semantic or operational identification essential; a factorization alone does not establish those units.

## Complete the continuous step after the graph edit

The previous experiment replaced16shared mixed products with32private squares, all writing to the sixth quadratic source read. This preserves the projection-coefficient budget. Fixed-direction readout fitting did not make it accurate enough, so we now optimized the shared and private input directions jointly, solving the constrained output coefficients analytically at every step.

The unedited global graph received the same additional budget: Adam rate0.01,2000cosine steps, two1%perturbation starts (seeds816and1816). Both structures use the same output metric. Selection uses coefficient loss, not component-value errors. The four fits took154.54seconds. These are perturbed warm starts, not new independent random initializations.

| Program after equal additional fitting | Products | Coefficients | Component errors on opened states |
|---|---:|---:|---|
| Unedited global graph |383|896,262|2.48%,2.44%,18.79%|
| Edited shared/private graph |399|896,198|2.58%,2.52%,16.09%|

The edited graph's third error improves14.35%relative to the continued global graph, below the registered20%requirement. It also exceeds both the15%absolute cap and the1.10-times-separate-baseline limit. The earlier partial-sharing program remains more accurate on that component at11.94%on these states, albeit with512products.

Both edited restarts fail. The selected winner has16.09%third error; the other has17.68%. An intermediate commentary initially quoted the latter as the outcome; the results file's registered weight-loss selection establishes16.09%as the primary result. No behavioral selection was substituted.

Five small explicit-tensor checks validate the implicit loss, input gradients and analytic-readout envelope derivative, including the empty-private control. Native-size preflights also pass. Independent exported-factor reconstruction agrees with recorded coefficients and scalar errors below1e-8. Original-feature fidelity remains below0.99for several quadratic groups. The failure is not explained by the tested export or derivative bugs.

## The private branch is useful, but not stable between starts

A private square branch computes a quadratic correction. We removed its centered contribution while retaining the original mean and linear terms, consistent with the programs' existing convention.

For the selected graph, removing it raises third-component error from **16.09%to24.88%**. Including it therefore reduces that error by35.33%. The other two components are exactly unchanged, as expected from the graph's connections. The other restart also benefits:17.68%with the branch versus23.78%without it.

However, the two private quadratic forms have cosine **0.707**, failing the registered0.99identity test. Their relative difference is78.04%in the fitted coefficient geometry. Thus the branch carries useful computation, but the algorithm has not recovered one reproducible private feature. Its output selectivity is algebraic; we have not shown selective effects on semantic behaviors.

## An exact ambiguity in the outer components

Let the original component factors be

$$
A_j=\frac{h^\top a_j-\tfrac12 q_{2j}(z)}{s(h)}-\alpha_j,
\qquad
B_j=\frac{q_{2j+1}(z)}{s(h)}-\beta_j.
$$

The three component values are A_j B_j. Here z and h are supplied earlier and later native inputs, q are quadratic source reads, and s is the explicit RMS denominator. All three write along the same residual direction w, so their combined contribution is

$$
y=w\sum_{j=1}^{3}A_jB_j=wA^\top B.
$$

For an orthogonal matrix O,

$$
\widetilde A=O^\top A,\qquad
\widetilde B=O^\top B,\qquad
\widetilde A^\top\widetilde B=A^\top B.
$$

Each rotated factor is still a permitted linear combination of the same quadratic reads, h-readers and constants. The total function is exactly unchanged, while its individual terms generally change. This uses the common writer; it would not automatically hold for three independent output writers.

We checked identity, a45-degree rotation of the last two factors, and two seeded orthogonal rotations on the original and fitted programs. Total replay passes1e-10. The fixed45-degree rotation changes the original component array by20.55%in relative norm while preserving its sum. For the edited fit, the displayed per-component errors become2.58%,9.15%,6.99%, while the combined error remains2.55%.

**Those rotated scores do not repair the original failed component gates.** The components now mean different functions. Their baselines and semantic tests would also need to follow the changed definition. We did not select a favorable rotation or promote it as a circuit.

The opened-state combined scalar errors are2.55%for the edited graph,2.57%for the continued global graph, and2.71%for the earlier partial graph. These are conditional combined-output diagnostics, not fresh logit-intervention results or a replacement for individual fidelity requirements.

## Consequence for the two-stage research plan

Decompositions can supply useful candidate computations, but both their primitive products and their outer component boundaries can be non-unique. Graph simplification should therefore retain the original function and identify meaningful units through explicit operational or semantic tests. It should not assume that every stage-one factor is already a fixed concept, nor redefine a failed target merely by rotating its coordinates.

All current graphs still depend on native inputs. Prediction on fresh/OOD inputs, standalone extraction, selective semantic manipulation and reusable identified units remain open. The next comparison should distinguish preserving the common-writer computation from preserving its proposed constituent interventions, with both sets of results visible.

## Evidence

- [Registered direction fit](../../direct_tensor_match/SHARED_PRIVATE_DIRECTIONS_PLAN_V1.json), [all four outcomes](../../direct_tensor_match/SHARED_PRIVATE_DIRECTIONS_V1.json).
- [Independent execution, private removal and identity audit](../../direct_tensor_match/SHARED_PRIVATE_DIRECTIONS_AUDIT_V1.json).
- [Exact outer-factor mixing check](../../direct_tensor_match/OUTER_COMPONENT_GAUGE_V1.json).

Diagnostics use448previously examined states and native intermediate inputs. No new model forward, fresh text panel or semantic validation was performed. The follow-up algebra and intervention diagnostics ran on CPU inFP64.
