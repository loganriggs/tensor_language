# Constrained DAS regularization red-team — 2026-09-07

The user's diagnosis is the right causal framing: complement-constrained DAS is vulnerable to task-family memorization. A direction can make the complement look inert on the examples/readout used for optimization without identifying the reusable variable.

The existing receipts do **not** show that regularization is useless. They show three narrower facts:

1. On the original single-family objective, tangent noise barely changed the unregularized axis on fresh A1/A2 (`0.454/0.556` versus `0.459/0.551` full-vocabulary loss). KL moved toward DIM and improved the optimized geometry relative to the memorizing axis, but sacrificed much of its scalar advantage.
2. In the later zero-fit multi-reader tournament, noise was useful and pooled alignment was best; KL alone remained weak. This is evidence that smoothing plus environment diversity matters more than a KL coefficient by itself.
3. The hard-feasible run selected the pooled **step-zero** checkpoint. Therefore it did not provide a positive test of whether regularized optimization can beat a good initialization; it showed only that family-level validation safely refuses harmful updates.

The corrected objective is a nested robust optimization, not a single weighted complement loss:

\[
\min_U\;\max_{e\in E_{train}} L_{comp/full-vocab}(U;e)
 + \lambda_{noise}\,\mathbb E_\epsilon L(U+\epsilon;e)
 + \lambda_{KL}\,L_{distribution}(U;e),
\]

subject to hard per-environment target-retention constraints. Hyperparameters and stopping time must be selected on entire held-out construction families, and the final axis must be opened once on sealed families. A row split within one template is not a generalization test. The complement test remains an evaluation criterion; it is not, by itself, an identifying objective.

Operationally, the next DAS optimizer should train across multiple construction families and use leave-one-family-out selection, comparing four fixed arms: no regularization, noise, KL, and noise+KL. It passes only if a moved checkpoint beats the pooled/DIM step-zero baselines on every held-out family while preserving downstream-reader transfer and full-vocabulary selectivity. This directly tests the user's proposal without allowing the optimizer to answer “is this the subspace you were looking for?” on the same task family.

This stays secondary to the current circuit promotion: the newly bidirectional five-MLP rank-16 source program is being translated into exact weight factors first, because that immediately connects a validated causal subspace to upstream/downstream tensor structure.

## Quantitative adjudication of the regularization hypothesis

The regularization hypothesis has positive evidence and should not be described as a failed idea. On the original held-out split, KL reduced the normalized full-vocabulary objective from `.5292` (unregularized) to `.0632`, while tangent noise reduced it slightly to `.5187`. On the separate multi-reader tournament, noise also beat the DIM baseline substantially. The price was that KL worsened the narrow scalar objective (`.00111` to `.00209`), which is exactly what we should expect if the scalar readout is the overfit quantity.

Accordingly, the decisive tournament will not ask whether KL wins the scalar complement loss. It will compare no-reg/noise/KL/noise+KL under a common six-term evaluation score, with margin and L15 target retention imposed as hard constraints. Training and checkpoint selection will exchange complete construction families, not rows within a family. This makes the opposing outcomes crisp:

- regularization succeeds if a moved noise/KL checkpoint beats pooled step zero on every family fold and on a once-opened sealed family;
- the objective is still wrong if learned checkpoints improve their training family but pooled step zero continues to win complete-family selection;
- rank one is inadequate if no moved checkpoint can improve selectivity while respecting target-retention constraints.

## Fully instrumented whole-family result

The deterministic v2 replay validates the measurement and sharpens the diagnosis. It counted 24 native model calls and 1,208 differentiable manual-reader evaluations separately (1,232 total), persisted zero manual-reader closure error on v8/v10/v12, and passed every hash, disjointness, finiteness, update, and forward-price conjunct. Thus the following differences are scientific evidence rather than an accounting artifact.

KL was selected by the registered worst-fold rule and its two fold axes were geometrically stable (`|cos|=.8827`). It improved v8-to-v10 worst six-term loss from `.7261` to `.6968`, but worsened v10-to-v8 from `.3486` to `.4007`. The v8 failure is not caused by a hard target-limit violation: every selected checkpoint is feasible. Instead, KL increases margin and L15 match/inert terms on both v8 panels while barely improving—or slightly worsening—the vocabulary terms that it was meant to regularize. On v10's difficult second panel, by contrast, KL reduces the dominant vocabulary terms enough to offset its larger margin terms.

After the prospectively selected KL arm was refit on v8+v10, it generalized strongly to the then-sealed v12 family: mean/worst loss fell from `.9803/1.0095` to `.3979/.4803`, and vocabulary match plus inertness improved on both panels while all target limits remained satisfied. The terminal is therefore `regularization_fold_heterogeneity`, not “optimization failed” and not an unqualified DAS success.

This rules out two simplistic next moves:

- merely increasing fixed KL weight, because the weak side is already harmed primarily through margin/L15 terms rather than insufficient vocabulary pressure;
- treating small tangent noise as a distinct solution, because at the tested scale its scores track the corresponding no-noise arms to within roughly `.001`.

The next optimizer should select KL strength and stopping time without seeing its outer construction family, using inner panel/family validation, and should include an explicit cross-construction effect-variance term. A new v14 text family is being reserved before that optimizer is authored, so the next result cannot be tuned to the already-open v12 success.

## Frozen design direction after the v14 capability null

V14 failed native capability before any DAS fitting (A1 had 29 jointly correct rows, A2 only 19/32), so it is excluded rather than rescued by a lower threshold.  V15 has now been generated with zero target-text overlap against v1–v14 and is undergoing capability-only screening.  The adaptive optimizer must not be claimed or allowed to inspect v15 outcomes until that manifest passes.

If v15 qualifies, the next DAS comparison will be nested rather than another outer-fold hyperparameter sweep:

1. The outer folds remain `train v8 -> test v10` and `train v10 -> test v8`.
2. Within each outer training family, split both A1/A2 panels by cue direction.  For each frozen configuration, fit one direction across both panels and select on the opposite direction, then reverse.  Thus KL strength, noise strength, and stopping time never see the outer construction family.
3. Compare the fixed configurations `no_reg`, KL weights `.25/1/4`, tangent noise `.10`, and noise `.10` combined with KL `.25/1`.  The current `.03` noise arm is not repeated as a headline method because it tracked its no-noise partner within about `.001`.
4. Use checkpoints `0/5/10`, retain pooled/DIM-like step zero, and impose the existing per-environment margin/L15 feasibility limits.  Add one fixed variance penalty on the normalized six-term loss vector across the two training panels; this targets construction inconsistency rather than simply increasing vocabulary pressure.
5. Refit the inner-selected configuration on the complete outer training family and evaluate the untouched outer family once.  Separately refit the inner-selected no-reg comparator under the same budget.
6. Choose the final configuration only from aggregated v8/v10 inner scores, refit on all v8+v10 panel-direction environments, and open capability-qualified v15 once.  V12 is reported only as historical evidence and is never part of selection or validation.

The registered success bar should require a nonzero regularized checkpoint to beat both its nested no-reg comparator and pooled step zero on each outer fold, then improve mean and worst six-term score on v15 without violating target limits.  If hyperparameters differ across the two outer folds but both win, the result supports adaptive regularization but not one reusable global objective.  If inner selection fails to predict either outer fold, the complement/KL loss family remains non-identifying even with nested tuning, and the route should move to a multi-reader causal operator rather than a denser coefficient grid.  A conservative ceiling is 5,000 counted reader/model evaluations and 350 updates; exact live accounting remains mandatory.

## Nested result: the current scalar-axis objective is rejected

The nested test is valid and returns `inner_selection_mispredicts_construction`.  Every authority,
row-disjointness, capability, closure, finiteness, and price check passes; the run counted 48 native
and 2,448 differentiable reader calls (2,496/5,000 total) and 305/350 updates.  Thus this is an
objective result, not another instrumentation ambiguity.

Inside v8, the aggregate selector chooses noise `.10` plus KL `1.0`, although the two held-out cue
directions disagree between KL-only and noise+KL.  The moved refit beats pooled step zero on v10
(`1.2052` versus `1.9098` worst six-term loss), but loses to identically budgeted no-reg (`1.1555`).
Inside v10, every configuration selects step zero on both directions, so the registered selector
chooses no-reg with zero updates.  It therefore makes no improvement over pooled when transferred
to v8.  The direction-stability, both-outer-fold, and generalization-gap predictions all fail.

The global inner score chooses noise `.10` plus KL `1.0` for five updates.  On the once-opened v15
family it lowers mean/worst six-term loss from `1.3122/1.8406` to `.6927/1.2115`, but is infeasible:
one v15 environment violates the hard L15-match limit by `.15642`.  This is the exact failure the
hard constraint was meant to catch—the optimizer improves complement/full-vocabulary terms partly
by discarding a required downstream target effect.  Predictions D and E therefore fail despite the
headline aggregate improvement.

This closes denser KL/noise tuning for the present rank-one scalar-axis loss.  Regularization remains
a useful ingredient, and optimization itself is not rejected; what is rejected is treating the
current six scalar contractions as an identifying objective.  The next DAS-like object should fit a
finite causal-response operator across held-out downstream-reader and construction blocks, with
target effects represented in the object rather than enforced only by a soft/barrier scalar loss.
