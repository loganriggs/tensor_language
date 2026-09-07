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
