# Task 14 MLP15/17 product-subspace causal interchange — design preregistration

**Frozen:** 2026-09-04 17:53 UTC

**Finalized after Phase-0 independent audit:** 2026-09-04 18:16 UTC

**Status:** BLOCKED AND SUPERSEDED BY PHASE 0; no projector fit, model, GPU, queue, or execution authorization

## Blocking Phase 0 amendment

This document is a historical subspace design, not an execution-ready preregistration. The authoritative prerequisite
is `TASK14_MLP15_17_FULL_RANK_CONDITIONAL_RESPONSE_PANEL_PREREGISTRATION.md` and its machine-readable Phase-0
contract. No rank, optimizer, or projector run may start until that full-rank panel is implemented, executed, and
reviewed.

The Phase-0 amendment corrects five issues in the original design:

1. The fitted response must be the conditional product change from the actual head-swap trajectory,
   $$\delta z=z_H-z_B,$$
   not a natural donor prompt's raw product vector. Its signs are
   $$E_{\rm full}=f(z_B)-f(z_H),$$
   $$E_{\rm remove}=f(z_H-P\delta z)-f(z_H),$$
   $$E_{\rm suff}=f(z_B+P\delta z)-f(z_B),$$
   so full-rank removal targets $E_{\rm full}$ and full-rank sufficiency targets $-E_{\rm full}$.
2. Target/control role comes only from `expected_relation`, never from the A/P/C family letter. Predictions are
   evaluated by target class and recipient subject state; neither number direction may be pooled away.
3. A joint MLP15+MLP17 intervention must recompute the live MLP17 product after changing MLP15. The cached factorial
   is prior context, not evidence of endogenous recomputation.
4. Product features are identifiable only modulo $\ker(W_D)$, of dimension at least 3,456. A later result may claim
   a causal output function or dense compiled quadratic tensor, never a unique hidden product basis.
5. The numbered-list control is deferred because causal liveness at MLP15/17 has not been shown. Any future unrelated
   control first needs a preregistered full-module liveness floor.

The Phase-0 contract also freezes the exact three-term expansion of the bilinear response and separately prices its
eight-corner causal factorial. Every section below is subordinate to these corrections wherever there is a conflict.

## Question and interpretation target

The landed causal screens support a downstream Task-14 path through MLP15 and MLP17. Their effects are strongly
direction-dependent and nearly additive; this does **not** yet say which internal computation either MLP performs.
The next question is whether each module's response to the head-11.3 subject-number counterfactual is carried by a
small, stable set of bilinear product features or only becomes simple after the module writes to the residual stream.

This targets within-module splitting, full-output causal sufficiency/removal, cross-module composition, and exact
translation to weights. Dimension is only the cost of the smallest **task-conditioned causal intervention** that
passes. Reconstruction, activation variance, tensor rank, and parameter count never enter fitting or selection.

## What is reused and what is new

Rung 536 already proved on planted examples that a projector in a bilinear product space compiles exactly into
quadratic weights. `PRODUCT_PROJECTOR_TO_QUADRATIC_WEIGHTS.md` and its tested compiler are reused unchanged. The old
MLP16 DAS and MLP17 output-rank/four-quadratic studies are not rerun: they used census masks, variance, or generic
whole-layer approximation rather than Task-14 donor pairs.

The new object is a projector tied to the exact causal response that has now been localized to MLP15 and MLP17. The
prior-art receipt is `circuits/followups/task14_mlp15_17_product_subspace_prior_art.json`.

## Exact MLP and counterfactual

For layer \(\ell\in\{15,17\}\), the normalized MLP input is \(x\in\mathbb R^{1152}\),

$$
z_\ell(x)=(W_{L,\ell}x)\odot(W_{R,\ell}x)\in\mathbb R^{4608},
\qquad
y_\ell(x)=W_{D,\ell}z_\ell(x)+b_{D,\ell}\in\mathbb R^{1152}.
$$

For each Task-14 natural donor pair, \(B\) is the native base trajectory and \(H\) is the same base prompt with only
attention head 11.3's 128-dimensional pre-output value replaced by its natural donor value. Let \(z_b\) be the
product activation on \(B\). At the selected layer, the full-output reference replaces the current \(z\) by \(z_b\).
Because `Left`, `Right`, and `Down` are otherwise unchanged, this exactly restores the native-base MLP output.

For a product-space projector \(P=UU^{\mathsf T}\), the two complementary interventions are

$$
z^- = z_b + (I-P)(z-z_b),
\qquad
z^+ = z_b + P(z-z_b).
$$

The first removes the proposed feature response from \(H\); the second keeps only that response above the base MLP
output. At rank zero they reduce to the unchanged/full-restored endpoints in opposite order; at rank 4608 they swap
those endpoints. Those endpoint equalities and the already measured full-output contribution are live tripwires.

We also fit the matched output-space object \(V V^{\mathsf T}\) in \(\mathbb R^{1152}\). It answers a different question:
does the response become simple only after `Down` combines the 4,608 products? Product and output projectors are
reported separately; one is not silently substituted for the other.

## FIT, inner SELECT, and sealed outer validation

Use the frozen Task-14 v2 partition and donor authority. Within the existing `DISCOVERY` partition:

- FIT groups: \(\{0,9,10,11,16,25,26,27\}\), containing 153 retained relations;
- SELECT groups: \(\{1,4,6,15,17,20,22,31\}\), containing 145 retained relations.

The mirror-paired lexical groups are indivisible, and the endpoint sets are disjoint. FIT can see activations,
gradients, full-output references, and optimizer state only for FIT relations. SELECT chooses the smallest passing
rank but cannot update a projector. The existing outer `VALIDATION` half remains physically unopened until a frozen
candidate passes; opening it requires a separate program and hash-bound receipt.

Targets are precisely the relations with `expected_relation=opposite_subject_toward_donor`; controls are precisely
those with `expected_relation=same_subject_zero_projected_effect`. P and C each contain both target and control arms.
Every metric must be separated by arm, family, matching rule, and recipient subject state. The signed reference is
each module's complete product-restoration effect divided by the frozen FIT median magnitude of the full head-11.3
effect. This avoids dividing by a module effect that is correctly near zero in some direction cells.

## Learning and selection

For each module and parameterization, fit ranks \(k\in\{1,2,4,8\}\), three deterministic starts per rank, 100 finite-
interchange updates per start, batch size 32. The loss is the equal-cell robust error between the selected and full
module effects for **both** removal and sufficiency, plus P/C answer-margin and full-vocabulary movement. There is no
reconstruction or variance term. Sixteen dimension-matched Haar projectors and two permuted-label fits are frozen
controls at each rank.

On inner SELECT, both removal and sufficiency four-cell effect vectors must have relative \(L_2\) error at most 0.20
from the full-module vector and cosine at least 0.95. Every reference cell of magnitude at least 0.03 must preserve
sign with normalized absolute error at most 0.03. P/C answer-margin movement and full-vocabulary RMS must each be at
most 0.05 and no more than the full-module control plus 0.02. The candidate's worst-cell error must beat the 95th
percentile random projector by at least 0.10.

The smallest rank with at least two of three passing starts is provisionally selected. Two confirmation starts are
then added; at least four of five must pass. Across passing seeds, median and 90th-percentile row-effect differences
must be at most 0.10 and 0.20. If responses are stable but projector overlap is not, the claim is an operationally
equivalent response class, not a unique internal basis. No rank above 8 is opened.

For a within-MLP split, the complement may carry at most 25% of the full module effect norm. Otherwise the selected
subspace is only sufficient and redundant.

## Full-output, unrelated-behavior, and composition controls

The full-output reference is measured live for every Task-14 pair; matching only the final `is`/`are` answer without
matching removal and sufficiency cannot pass. Full-vocabulary movement is also retained so an answer-steering
direction cannot hide large collateral effects.

The previously proposed numbered-list control is not valid here: its known circuit is live at L8H7/L8H3 with candidate
writes in MLP8--14, not at MLP15/17. It is deferred until an unrelated behavior first passes a preregistered MLP15/17
full-module liveness floor. The old 62 overlapping census masks may eventually be reported as a response fingerprint,
but they are not treated as 62 independent semantic controls.

Finally install the selected MLP15 and MLP17 interventions together, in depth order, with the live MLP17 product
recomputed after the MLP15 intervention. The joint four-cell vector must
match the exact full MLP15+17 reference within 0.25 relative \(L_2\), and its interaction beyond the sum of the two
selected single-module effects must be at most 0.03 in every normalized cell. This is the test that the independently
learned features preserve the additive path discovered by the preceding factorial.

## Exact translation into quadratic weights

For a product basis column \(u_j\), reuse the existing compiler:

$$
Q_j=\frac12\left(W_L^{\mathsf T}\operatorname{diag}(u_j)W_R+
W_R^{\mathsf T}\operatorname{diag}(u_j)W_L\right),
\qquad d_j=W_Du_j.
$$

For an output basis column \(v_j\), let \(a_j=W_D^{\mathsf T}v_j\), use the same formula with \(a_j\) in place of
\(u_j\), and set \(d_j=v_j\). In both cases the selected function is exactly

$$
F_P(x)=\sum_{j=1}^{k} d_j\,x^{\mathsf T}Q_jx.
$$

RMS normalization remains an explicit input operation. `Down_bias` stays in the background MLP rather than being
misassigned to the Task-14 response. Direct projected output, compiled output, and donor-minus-base compiled
interchange must agree to the existing compiler tolerance before any scientific result counts.

Because $W_D$ maps 4,608 products to 1,152 output coordinates, at least 3,456 product directions lie in its null
space. The compiled causal output function is the identifiable object; any product-space basis is an equivalence class
modulo that null space.

## Registered outcomes and execution boundary

1. **Product-space causal feature:** a product projector passes all individual, control, stability, composition, and
   compilation bars.
2. **Output-only causal write:** output space passes but product space does not; the downstream write is simple while
   its product-space realization remains distributed.
3. **Operational response class:** causal effects are stable across seeds but geometries are not.
4. **Small-linear-subspace null:** neither parameterization passes through rank 8; do not widen the rank sweep.
5. **Instrument invalid:** any authority, split, endpoint, rank-0/full-rank, optimizer, hook, or compilation tripwire
   fails.

This document authorizes no model run. Before execution, a create-only implementation amendment must freeze the exact
runner hash and maximum forwards, backwards, examples, retained bytes, and runtime. A pass is still a Task-14 FIT/
SELECT circuit identification step, not OOD evidence or adoption.
