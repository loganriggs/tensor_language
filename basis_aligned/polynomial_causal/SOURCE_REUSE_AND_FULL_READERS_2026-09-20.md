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

## v680: final projection is not the main source-reuse failure

The frozen v665 final encoder/decoder was tested on true native final response states for A, B and post11-superposition. Modal error maxima fall to0.210%,0.326%,0.297% of each cell's source-target effect. At the five failing A/B modal cells, the oracle retains only0.7–5.0% of the candidate error, decisively failing the registered hypothesis that it retains at least75%. Full-state float64 selected-logit replay differs by at most3.24e-6, within the1e-4 instrument bar.

Thus the final eight-feature output interface can represent these tested control responses when provided the native final state. Loss during reduced propagation (including its initial projection) is the next localization target. This reverses the explanation found for the older v659 frame in v664; it does not invalidate that older result or establish a unique basis. Oracle states are unavailable to a deployed predictor. v681 tests exact projected-state replacements at interior boundaries, with no change to calibration, width or sparse cores.

## v681: upstream propagation, with nonmonotonic resets

At boundary k, replace the reduced response by the projected native response after block11+k and run the remaining reduced suffix. k0 preserves the exact original coordinate port; k6 is the final-state oracle. Both controls reproduce their respective earlier outputs exactly. Interior boundaries k4 and k5 pass A/B target10% and modal5% in every opened cell. k4 (post15) has worst target errors2.69%/5.83% and modal2.46%/3.21% for A/B; sum target6.57%,modal3.06%.

Earlier substitutions are not monotonic improvements: k1 makes A modal error13.01% versus6.11% at k0; k2 gives A target14.40%. Thus native-state substitution plus projection can disrupt compensating errors or expose state directions omitted by the next interface. These experiments localize a diagnostic distinction to earlier versus later suffix computation, not a unique faulty layer. They do not justify adopting the whole circuit, or replacing a legal initial interface by an uncharged native oracle.

CPU audit independently summarizes all boundaries and telescopes the modal-prediction differences. Adjacent differences are differences between reset experiments, not isolated causal layer effects. Primary receipt: `subject_attention_freeze_v681_result.json`; audit: `SOURCE_BOUNDARY_CPU_AUDIT_2026-09-20.json`. Next informative test separates attention-input projection from post-attention/MLP propagation in blocks12–15, keeping the same basis and control rows.
