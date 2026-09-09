# Structured data augmentation for temporal is/was DAS

**Status:** experiment design, not a result or preregistered GPU claim
**Date:** 2026-09-09 UTC

## High-level answer

We should add data augmentation. The completed constrained-DAS searches used isotropic Gaussian
noise on cached head-response vectors, which is materially weaker: the optimizer still saw the same
A1 construction and could learn an A1-specific solution. The decisive failure was transfer from A1
to sealed A2, so the regularizer must expose construction and lexical variation during fitting and
selection.

The fair test is structured augmentation plus environment-robust optimization, compared with DIM at
the same rank and on the same environments. The already-generated v24 construction must remain
sealed; using it to choose the augmentation recipe would turn it into validation data.

## What was and was not tried

The strongest four-head cDAS tournament perturbed each training response by Gaussian noise with
`sigma in {0,.05,.10}` times that head's response RMS. It also tried control KL, a local Jacobian
penalty, target-feasibility barriers, early stopping, and cross-fold projector stability. It did not
generate new text or causal pairs. In particular, there was no:

- reporter-noun substitution;
- tense-preserving construction paraphrase;
- syntactic-frame augmentation;
- donor rematching within the same causal class;
- leave-one-construction-out model selection.

Calling that prior intervention “data augmentation” would therefore be inaccurate.

## Augmentation axes

Each augmented view must preserve the intended present/past causal label and the aligned intervention
position. Proposed training environments vary four independently checkable factors:

1. **Lexicon:** history-disjoint reporter nouns, split by reporter family rather than row parity.
2. **Construction:** multiple present/past paraphrase pairs, with each construction treated as its own
   environment.
3. **Direction:** present-to-past and past-to-present swaps must both satisfy the target constraint.
4. **Donor matching:** rotate donors within the same declared causal class without using output
   answers, margins, or downstream effects to choose the match.

Activation perturbations can remain as a secondary robustness term. Besides isotropic Gaussian noise,
a later arm may use bounded perturbations aligned with measured lexical/construction nuisance
directions or downstream-reader Jacobians. Those arms must be declared before their sealed test.

Any augmented row that fails the frozen native capability criteria remains counted as a failure or
invalidates its environment according to a predeclared rule; it cannot be silently filtered after
looking at model outputs.

## Environment-robust objective

Let `e` index construction-by-lexicon environments, `T_e(U)` be signed target recovery for projector
`U`, and `C_e(U)` be the control cost, including full-vocabulary KL and top-1 margin damage. Optimize
the worst environment, not the pooled average:

```text
minimize_U       max_e C_e(U) + lambda_consistency * R_consistency(U)
subject to       min_e T_e(U) >= 0.75
                 min_e direction_fraction_e(U) >= 0.875
                 U^T U = I.
```

`R_consistency` penalizes variation in the signed behavioral effect across label-preserving augmented
views after normalizing by each view's complete-response effect. Separate dual variables per
environment are preferable to a single weight-100 scalar barrier because a pooled scalar can trade
away a weak environment.

Model selection should use leave-one-construction-out cross-validation: fit on all but one training
construction, score the held construction, rotate, and select by the worst held-construction result.
Reporter groups remain disjoint across fitting and held-environment scoring. Multiple frozen restarts
and a longer optimization budget address the separate possibility that eight Adam steps were too
short.

## Comparison and sealed test

- Fit structured-augmentation DAS and rank-matched DIM on exactly the same construction, lexicon,
  direction, and donor groups.
- Charge all ranks, restarts, checkpoints, augmentation recipes, and regularization choices to the
  search budget.
- Select without reading v24 causal outcomes.
- Run v24 once only if its queued native-capability gate passes every declared family requirement.
- Require the same target bars, zero top-1 control flips, and a predeclared broad-control KL bound.
- If v24 is opened and the method is subsequently changed, build a new history-disjoint v25 bank for
  the next sealed claim rather than reusing v24 as if it were still held out.

The null must be specific: “this frozen augmented-DAS recipe did not beat matched DIM on the sealed
construction,” not “optimization cannot improve on DIM.”

## Why this is higher information than more isotropic noise

Gaussian activation noise asks for local smoothness around the same examples. Structured augmentation
asks for invariance under the exact lexical and construction shifts on which the selected projector
failed. If augmented DAS still loses to DIM under a genuinely sealed construction, that is evidence
against this optimized estimator. If it wins, the earlier result was primarily a validation-distribution
and regularization-target failure.

No GPU work is launched by this document. The current managed queue already contains the v24 native
capability gate and the higher-priority physical reader localization needed to interpret either DAS or
DIM as part of a circuit.
