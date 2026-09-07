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
