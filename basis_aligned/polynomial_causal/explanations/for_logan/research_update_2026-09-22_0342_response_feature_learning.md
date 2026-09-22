# Learning missing features: response loss did not beat value loss in the toy study

22 September 2026, 03:42 UTC.

The new loss can learn missing quartic features in controlled examples, but **it does not show the registered advantage over ordinary value matching**. At the fixed 250-step settings, Adam succeeds more often than Muon. Earlier longer runs show that this is strongly affected by training duration, so it is not a universal optimizer ranking.

This follows the native experiment where changing only the output readout barely improved response fidelity. Here the input directions themselves move. The test is synthetic and has known representational capacity; it is not evidence that we have found native circuits.

## What the toys represent

We start with a known two-layer bilinear polynomial and a fixed known parent computation. The missing part has two scalar output features, each with exactly two quartic terms:

$$
R_g(x)=\sum_{j=1}^{2}c_{gj}
\prod_{s=1}^{4}\bigl(a_{gjs}^{\top}x\bigr),
\qquad x\in\mathbb R^4.
$$

The learner has exactly that capacity. Its four input-factor matrices start randomly; all their row directions are optimized. Each output's two coefficients are solved by a small ridge-regularized least-squares problem during fitting. The parent stays fixed. We compare Gaussian value loss with the exact paired-Gaussian response loss at $\rho=0.5$, defined in the [preceding report](research_update_2026-09-22_0325_gaussian_response_fit.md).

The original cohort has five target configurations: generic products, two constructions of products of squares, within-output shared quadratic factors, and fourth powers. **A red-team check found that the two square constructions are the same polynomial structural class after permuting input slots.** The original 40 fits and registered criteria are retained, but they should be described as five configurations covering four structural classes.

We added eight descriptive fits with a quadratic factor shared across both output branches:

$$
q(x)=(a^\top x)(b^\top x),\qquad
R_g(x)=q(x)\sum_j c_{gj}(u_j^\top x)(v_j^\top x).
$$

This supplies a fifth distinct structural class. Its shared-factor evaluation is independently replayed against the expanded product representation. It is a supplement, not a retroactive replacement of the original cohort.

## Matched fitting settings and results

There are two restarts, 250 updates, and two objectives for each optimizer and target. Factor rows are normalized. Rates are Adam0.1 and Muon0.01, fixed from the earlier **value-loss** toy sweep; response loss did not receive its own rate search. Best checkpoints are selected only by their fitting objective. This comparison is therefore bounded to these settings.

| Optimizer / training objective | Median Gaussian value error | Median Gaussian response error | Median coefficient error |
| --- | ---: | ---: | ---: |
| adam, value | 1.41% | 1.54% | 2.45% |
| adam, response | 1.33% | 1.44% | 2.25% |
| muon, value | 14.07% | 15.20% | 23.66% |
| muon, response | 13.25% | 14.27% | 20.71% |

The first registered criterion required at least eight of ten response-trained fits per optimizer to reach below5% response error. Adam passes with **8/10**; Muon fails with **1/10**. The second required at least10% lower mean response error than value-trained fits for the same optimizer. Both fail: the ratios are **1.003 for Adam** and **0.980 for Muon**. Median improvements do not override this mean-based criterion or the failed cases.

The added cross-output sharing case also gives mixed results:

| Optimizer / restart | Value-trained response error | Response-trained response error |
| --- | ---: | ---: |
| adam, 0 | 0.49% | 0.56% |
| adam, 1 | 4.00% | 3.46% |
| muon, 0 | 14.59% | 25.39% |
| muon, 1 | 7.17% | 6.54% |

The learner can represent the shared polynomial even when its CP form does not explicitly cache the common quadratic. Recovering that computational reuse is a separate graph-simplification question. These tests measure function recovery, not recovery of unique factors or the simplest arithmetic program.

## Training duration changes the interpretation

The earlier 500-step **value-objective** runs on the original toy targets, with the same starts and selected rates, achieved median value errors of **0.69% for Adam** and **0.83% for Muon**; all ten cases for each optimizer were below5%. The 250-step value-objective medians here are **1.41% and 14.07%**. The value objectives are mathematically equivalent, but the prior run used native teacher-minus-parent cross contractions and this run uses its known residual CP expression; this is not a bitwise replay of the optimization trajectory.

That large duration effect makes an under-converged short Muon run a plausible explanation for some toy failures. It does not prove that doubling steps fixes a native fit. The queued native learner has12outputs,96new terms,1,152inputs and a different loss normalizer; these toys have2outputs,4terms and4inputs. Its frozen250-step protocol remains unchanged. Inspect its histories before attributing any failure to representation rather than optimization.

## Instrument checks and limits

The new rectangular correlated-moment recurrence passed15 independent paired quadrature and gradient checks across five configurations and three correlations. The earlier degree-resolved implementation passed20 additional paired/derivative checks. Original40fits took about68seconds and the eight-fit supplement14seconds on two CPU threads. Population value, response and symmetric-coefficient errors are computed exactly; no finite probe sample selects the checkpoint.

There is no demonstrated reason yet to replace the pending native value-based learner with this response-loss variant. We retain the valid objective for future use but do not enqueue another native sweep on the strength of these toys alone. Both the learned native residual and the native-removal geometry diagnostic remain separate pending experiments. The overall circuit goal is active.

[Preregistered cohort and structural-coverage amendment](../../direct_tensor_match/PAIRED_RESIDUAL_LEARNING_TOY_PLAN_V1.md) · [All original fits](../../direct_tensor_match/PAIRED_RESIDUAL_LEARNING_TOYS_V1.json) · [Sharing supplement](../../direct_tensor_match/PAIRED_RESIDUAL_SHARING_SUPPLEMENT_V1.json) · [Independent recurrence checks](../../direct_tensor_match/CORRELATED_GAUSSIAN_CP_CONTROLS_V1.json) · [Learning code](../../direct_tensor_match/toy_paired_residual_learning.py) · [Earlier500-step value sweep](../../direct_tensor_match/LOCAL_QUARTIC_RESIDUAL_TOY_SWEEP_V2.json).

## Follow-up: small-to-large optimizer transfer needs its own control

A direct direction-recovery test now exposes another confound. With the native 96-by-1152 raw parameter shape, default Muon at the selected rate rotates normalized feature directions by only about 3.7 degrees in 250 steps, whereas Adam almost exactly recovers the known targets. The alternative Muon RMS scaling rotates about 25 degrees but still does not solve this control. These are simple synthetic direction targets, not quartic fits or a native result. The queued comparison is unchanged. Before generalizing from its optimizer scores, we need wide-dimensional quartic controls and explicit accounting for defaults and parameter normalization. [Full explanation, trajectories and sources](../../direct_tensor_match/NORMALIZED_OPTIMIZER_GEOMETRY_INTERPRETATION_V1.md).

## Wide quartic follow-up: optimizer adequacy remains unresolved

The stronger test embeds five known quartic structures into 1,152 coordinates, adding irrelevant inputs while preserving the exact teacher function. Adam reaches below 5% error in only 2/10 runs, compared with 9/10 at four dimensions. Neither tested Muon configuration succeeds in the wide setting. Changing only the raw initialization scale—leaving the initial normalized features unchanged—reduces default Muon median error from about 100% to 30%, but still gives 0/10 accurate recoveries. These controls establish a real optimization confound; they do not establish a native fix. The queued native experiment is unchanged. [All configurations, predictions and limitations](../../direct_tensor_match/WIDE_QUARTIC_OPTIMIZER_INTERPRETATION_V1.md).
