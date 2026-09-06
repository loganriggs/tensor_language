# DAS regularization and task-memorization evidence synthesis — 2026-09-06

## Corrected conclusion

Optimization is not intrinsically worse than difference-in-means (DIM). The existing evidence
supports the user's narrower hypothesis: the scalar complement objective admits task-specific
solutions, and task augmentation plus a vector-aligned loss improves generalization. Noise and KL
alone were insufficient at the tested strength and rank.

## What each experiment establishes

| Test | Positive evidence | Remaining failure |
|---|---|---|
| Rank-1 cDAS target red-team | cDAS beats DIM on its held-out scalar match-plus-complement objective. | It loses to DIM on centered full-vocabulary match-plus-complement, so the scalar objective is target-specific. |
| Noise/KL regularization | Noise is the best genuinely perturbed regularizer and changes the fitted basin; KL moves close to DIM. | Neither beats unregularized cDAS on both panels of either distant evaluation bank. |
| Direct full-effect objective | Validation selects an interior weight `lambda=0.3`; the selected axis beats DIM on sealed A2 full-vocabulary error. | It loses the preregistered scalar-usefulness gate, so one downstream vector target still trades off behavior. |
| Multicue aligned objective | Pooled aligned rank 1 beats both the single-task axis and pooled DIM on A1 and A2 full-vocabulary objective. A1: `.7588 < .8308 < .8953`; A2: `.7664 < .8304 < .8576`. | Scalar complement fractions remain `.345/.354`, and the rank-2 union does not identify the shared subspace. Terminal: `task_conditioned`. |

Thus regularization/augmentation works as an estimator improvement, exactly as hypothesized, but
the present objective remains under-observed. The strongest evidence is not “add more KL”; it is
that exposing more task environments changes the optimum in the right direction.

## Why the complement loss can memorize

Let `P_U` be the learned subspace projector and let `J_e` denote the finite causal response map in
environment `e`. A scalar objective observes only `w_e^T J_e P_U delta x` and its complement. Any
change to `P_U` inside the joint nullspace of the observed `w_e^T J_e` rows is unpenalized, even if
it changes other vocabulary coordinates or downstream readers. Noise smooths the local input
neighborhood; KL anchors a preferred estimator. Neither adds the missing downstream observations.

The correct next optimization object is therefore a block causal operator with multiple
environments and readers:

`loss(U) = sum_{e,r} [match(J_{e,r} P_U) + complement(J_{e,r}(I-P_U))]`

with entire lexical/construction/reader blocks held out. Projector stability across fit splits and
zero-refit cross-task causal prediction determine identification. DIM, scalar cDAS, noise cDAS,
and anchored/vector cDAS are matched-rank baselines, not separate notions of correctness.

## Next discriminating test

Do not repeat the completed single-reader regularization sweep. Fit one rank-fixed subspace across
at least two independently varying downstream reader families, including the already identified
temporal/is-was Q8 causal response block. Freeze it before opening a third reader/construction
block. Require:

1. lower sealed full-vector causal error than pooled DIM and scalar cDAS;
2. materially lower complement effect than the current `.345/.354`;
3. stable projector across lexical splits/seeds;
4. correct cross-task intervention composition;
5. nontrivial overlap with, and causal prediction through, the weight-derived Q8 interface.

A pass promotes optimized DAS from task-conditioned estimator to stable identification. A failure
means the multireader response has higher state dimension or no single invariant subspace at this
site; increasing regularization strength without changing observations would then be low value.
