# Which attention factors carry the two behaviors?

**Correction following the user's clarification:** this experiment tested whole stored score factors. It did **not** test whether the correlative and disjoint behaviors read different input subspaces through the joint QK1×QK2 product. Both circuits may use both factors, with distinct paired input directions. The failed whole-half hypothesis does not count against that proposal. The OV audit likewise did not identify those joint routing subspaces.


The new test does **not** support a simple division where one stored query–key factor serves the correlative behavior and the other serves the disjoint behavior. At the tested sites, changing the value payload matters much more for both behaviors. This rejects only that whole-factor assignment; the user’s proposed joint input-subspace split remains untested.

## What was tested

The earlier result split a shared set of 26 head outputs into a saved projector P and remainder R. P transferred both/neither→and/nor; R transferred either/not→or/but. Selective mean removal failed, so those parts were not independently removable circuits. The subsequent [OV weight pullback](attention_ov_input_reader_overlap.md) found overlapping per-head input readers.

A bilinear attention head computes

    head_output = sum_source QK1(source) * QK2(source) * value(source).

Each score factor includes the real query/key normalization, positional transform and width scaling. Value includes the actual local/first-layer mixture. We tested all seven nonempty subsets of donor factors: QK1, QK2, their pair, value, either score with value, and all three. Each resulting head change went through P or R. Unchanged factors were recomputed from the recipient’s CURRENT state, including earlier edits. These are factor interchanges, not changes to trained weights.

The full-factor arms reproduce the independent original head-space swaps. Prior P routing-only and value-only recoveries also reproduce. All 48 native sentence pairs have correct answers at both endpoints. The run used 54 body forwards/864 sequence instances at 15:17:04–15:17:09 UTC; executor time was 2.51seconds. All native weights remain.

## Conditional factor dependence

For each factor j, we compare the full donor-factor interchange with the same intervention except that j stays live in the recipient. The reported loss of recovery is

    I_j = mean((margin_without_j - margin_full)
               / (base_native_margin + donor_native_margin)).

Margins use each endpoint’s own intended answer. Thus these values are fractions of the native cue-induced answer-margin change, not CE damage, output variance, or additive shares of a circuit.

| Own behavior and branch | Keep QK1 at recipient | Keep QK2 at recipient | Keep value at recipient |
|---|---:|---:|---:|
| both/neither, bare frame, P | .0658 | .0594 | .8105 |
| both/neither, report frame, P | .0654 | .0642 | .7802 |
| either/not, R | .0320 | .0130 | .8923 |

For example, in the first row, full recovery is .9693; keeping recipient values lowers it to .1589. Their difference is .8105. Both current routing scores matter less than this contextual value change. Those values already contain information computed by earlier layers, including earlier query–key operations. A small current-score interchange effect does not locate that upstream computation.

The registered hypothesis required the QK1-minus-QK2 dependency difference to be at least .20 in magnitude and reverse sign between the two behaviors, with agreement across the two correlative frames. Instead, the differences are .0064/.0012/.0189, all positive. The hypothesis fails. Paired95% intervals are [.00045,.01242], [−.00467,.00774], and [.01132,.02683]. Value-dependence intervals all remain above .72, supporting the narrower shared dependence on contextual values.

QK1/QK2 are labels of stored weight blocks. They can be exchanged independently inside each head without changing the head’s product. Any future semantic attribution must respect this symmetry rather than attach universal meanings to “first” and “second.” This screen tests the given global label arrangement only.

## A narrower asymmetry, and why it is not yet a circuit

The complete lattice reveals a smaller distinction in full-vocabulary prediction. For the disjoint R task, swapping QK1 plus value leaves 9.51% relative error versus the full branch effect; QK2 plus value leaves 19.47%. In the P task, either score-plus-value arm still leaves 11.7–13.5% error. These were measured descriptive cells of the lattice, not a newly selected and independently validated circuit. The R cell near 10% would need a separate test on new data before promotion; it does not rescue the failed opposite-score-half hypothesis.

Value-only swaps retain strong answer-margin recovery (.853/.780/.936), but leave20.9%/22.1%/25.3% full-vocabulary error. This repeats the P limitation and extends it to R: “most answer transfer” is weaker than predicting the entire effect.

The effects are also coupled. The discrepancy between the full effect and the sum of three singleton-factor effects is 39.6%/33.4%/36.4% of the full effect norm. We computed the complete inclusion–exclusion decomposition, sometimes called a Möbius expansion: singleton effects, three pair interactions and one triple interaction sum exactly to the full change. Numerical closure is below 1.63e-16 relative error. These are interactions between whole recomputed trajectories; they are not just the local product of three factor differences. Interaction norms cannot be summed as causal percentages.

## What remains open

The evidence now says: the two output branches often access overlapping value-input spaces; both depend strongly on contextual value changes; and a clean global assignment of their behavior to opposite stored QK halves is unsupported. A head-specific routing mode or a more explicit contextual-value producer can still distinguish them. Neither the broad remainder nor a matrix-overlap score provides that explanation by itself.

The original P/R donor dissociation and failed selective removal remain unchanged. The token/hierarchy backward-unembedding findings are also preserved: an explicit conditional token-reader fold predicts a separate small opposing MLP16 effect, while coarse group means are insufficient. No four-property circuit or structural model reduction is claimed.

Evidence: [registered test](../../CORRELATIVE_THREE_FACTOR_V1_PREREGISTRATION.md), [native result](../../CORRELATIVE_THREE_FACTOR_V1_RESULT.json), [paired audit](../../CORRELATIVE_THREE_FACTOR_AUDIT_V1_RESULT.json), [factor executor](../../correlative_three_factor_executor_v1.py).

## Follow-up through 19:56

The corrected joint QK experiments are separate from folding unembedding through MLP17 into final attention. V2 removes positional coordinates before fitting task spaces. The weight pullback compares sym(Q1ᵀWQ2) readers on a common input; average overlaps .164 query/.285 key, with native denominators retained as distinct state-dependent factors. No fully disjoint or mostly shared claim passed. [Full explanation and receipts](unsupervised_products_and_position_corrected_qk.md).
