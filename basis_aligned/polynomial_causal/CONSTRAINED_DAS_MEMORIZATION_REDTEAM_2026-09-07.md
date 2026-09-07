# Constrained DAS memorization red-team — 2026-09-07

## Question

Does constrained distributed alignment search (DAS) lose to the difference-in-means
(DIM) direction because optimization is intrinsically unhelpful, or because the
scalar target/complement objective admits task-specific, non-transferable solutions?

## Evidence already capable of separating the explanations

1. **The objective can actively move away from the stronger extraction axis.** In
   `unit_dim_init_constrained_v82_result.json`, starting the constrained optimizer at
   DIM did not preserve DIM's quantifier extraction: it moved from 0.824 to 0.752,
   approximately the random-start constrained result (0.757). This rejects bad
   initialization as the primary explanation. The optimized margin is not an adequate
   proxy for the causal variable we intend to identify.

2. **Full-vocabulary KL attacks the narrow-readout shortcut; tangent noise alone did
   not.** On held-out A2 in
   `temporal_auxiliary_will_had_block11h3_regularized_cdas_v1_result.json`, the
   unregularized constrained axis had full-vocabulary joint error 0.4485 versus DIM's
   0.2385. KL reduced the error to 0.2445 while noise alone remained 0.4486. On the
   validation panel, unregularized error was 0.5292, KL was 0.0632, and DIM was
   0.05923. Thus the useful regularizer in this test was output-geometry KL, not
   generic tangent noise. Noise remains plausible only when coupled to broader
   environments or used as a flatness selector inside a feasible set.

3. **A better-aligned weighted objective can edge past DIM, but only by a tiny amount
   and without preserving every target bar.** The selected weight-0.3 axis in
   `temporal_auxiliary_will_had_block11h3_aligned_objective_cdas_v1_result.json`
   improved validation full-vocabulary error from 0.0592296 to 0.0590696 and sealed
   A2 error from 0.238513 to 0.233426. Its sealed target extraction fraction was
   0.7502 versus DIM's 0.7529, and its scalar joint error was 0.02377 versus DIM's
   0.02326. This is evidence that optimization has a slightly better solution in the
   vicinity of DIM, but a weighted sum still permits a target-sufficiency trade.

4. **Row holdout is insufficient to prevent memorization.** The noisy
   worst-environment multi-reader fit improved its odd-row selection objective yet
   failed to improve the sealed construction families, producing the registered
   `task_memorization` terminal. The follow-up complete-family selector correctly
   chose step zero. This localizes the overfitting boundary at the construction/task
   family, not merely at individual examples.

## Red-team verdict

The statement “optimized DAS is worse than DIM” is too broad. Current evidence instead
supports: **the scalar complement objective is under-specified and is exploitable by a
task-specific direction**. Complement inertness is necessary evidence for a candidate,
but it is not sufficient evidence that the target subspace is correct. KL provides a
real corrective signal; the tested tangent noise by itself does not.

## Next discriminating experiment: feasibility before regularization

Use DIM as an eligible baseline and fit rank-one axes with the following lexicographic
rule:

1. A checkpoint is feasible only if target extraction and downstream L15 transport
   meet a frozen fraction of the DIM value in **every** training and complete-family
   selection environment. No reduction in complement loss can compensate for
   infeasibility.
2. Among feasible checkpoints, minimize worst-environment complement effect plus
   centered full-vocabulary KL. This directly tests the justification previously
   assigned to the complement test.
3. Compare a deterministic fit with an antithetic tangent-noise fit. Noise is a
   flatness/stability selector inside the feasible set, not a substitute for target
   sufficiency.
4. Select on a complete held-out construction family and evaluate once on untouched
   v10/v11 families. DIM, unregularized constrained DAS, weighted-KL DAS, and the
   frozen pooled axis remain controls.

Opposing outcomes are diagnostic:

- If a non-DIM feasible axis lowers sealed complement/KL while retaining target
  extraction, optimization was useful and the old target was wrong.
- If all learned moves are rejected and DIM remains best, the present rank-one site
  may already be at the transferable frontier.
- If a move passes row holdout but fails complete-family selection, memorization is
  reconfirmed.
- If noise helps only in-sample or changes the selected axis without sealed gains,
  noise regularization is rejected for this use; KL/geometry remains the supported
  mechanism.

This experiment changes circuit targets of stable identification, held-out/OOD
prediction, extraction, and selective manipulation. It is not a rank or compression
sweep.
