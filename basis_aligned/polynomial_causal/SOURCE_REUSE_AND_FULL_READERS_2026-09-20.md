# Source reuse, failed rescues, and full finite readers

The source test uses A=embedding recurrence+early0–3 writes and
B=middle4–7+MLP8+MLP10. Each is propagated exactly through block11. The shared
suffix is then tested on either post11 response and their sum. This sum differs
from the pre11 joint intervention by5.03% of the original joint-effect norm;
block11 nonlinear cross terms cannot be silently omitted.

| Frame / core | Worst A target | Worst B target | Worst A modal | Worst B modal |
| --- | --- | --- | --- | --- |
| Four readers, sparse18 (v671) | 9.96% | 6.77% | 6.11% | 7.71% |
| Four readers, dense36 (v672) | 10.14% | 4.87% | 6.08% | 6.81% |
| Extra source calibration, sparse18 (v674) | 11.26% | 5.78% | 6.21% | 8.08% |
| Extra source calibration, dense36 | 11.39% | 5.05% | 6.18% | 7.22% |
| Full-native/all-position calibration, sparse18 (v678) | 18.41% | 10.11% | 10.35% | 9.10% |
| Full-native/all-position calibration, dense36 | 18.57% | 10.56% | 10.33% | 8.99% |

Every candidate fails the preregistered individual-source preservation gate5%
per cell. The earlier joint-reader gate was pooled, so this is a stricter test.
Sparse v671 predicts nonlinear superposition better than adding individual effects,
but cross-error/sum is not evidence that interaction is small relative to the
smaller component. No independence or complete reuse claim follows.

Independent CPU scoring confirms identical native targets in sparse/dense runs.
Dense-core rescue does not fix the problem, and broader calibration at width8
also fails. Full-native calibration changes both dynamic readers and token-position
coverage; its failure does not isolate which change matters. These are failed
empirical spaces, not a lower bound against simpler circuits.

## Exact finite readers

For bilinear M, delta M=M(delta a,mid b)+M(mid a,delta b). Composing this identity
with exact RMS divided differences yields endpoint-dependent readers through
bothQK products, their product with values, rotary maps, causal masks, residuals,
MLPs and final softcap. CPU finite closure and equal-endpoint/autograd tests pass.
v675 validates attention12–17 natively; v676 validates four readers at every one
of seven full-suffix boundaries: maxabsolute closure2.22e-6, norm-scaled6.99e-7,
with exact causal support. These readers require both executions, so they are
calibration/attribution instruments rather than predictors.

The scheduled review adds a concrete warning: the same cubic function grouped as
(ab)c or a(bc) has different midpoint readers while both close exactly. See
[executed counterexample](REVIEW_SECANT_CONTROL_2026-09-20_0727.json). Exact closure
and matching derivative limits do not establish unique features or OOD reuse.

Related primary work: [discrete gradients](https://arxiv.org/abs/math-ph/9805021)
and [DeepLIFT](https://proceedings.mlr.press/v70/shrikumar17a.html). These motivate
finite-change propagation; no theorem from them certifies our compressed model.
[Full math review](THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-20_0726.md) maps HT/QB
and arithmetic-circuit literature to the actual object and states mismatches.

Two reviews overlapped at07:26/07:31. Preserve both substantive receipts; the
scheduled prompt now rechecks freshness immediately before writing so a future
concurrent review becomes an addendum instead of resetting the three-hour clock.

Next WEIGHT_FOLDING action: retain the best supported v665 frame and compare an
exact factorized attention contraction against the expanded T^2r^2 score cores.
This is a same-function storage/extraction baseline, not a claimed repair of reuse.
Source-specific hierarchical feature spaces remain a separate future hypothesis;
no additional rank or calibration sweep is registered here.
