# Wide quartic recovery: small toy success does not transfer automatically

2026-09-22 04:38 UTC. Completed 60 matched fits and a separately specified 10-fit raw-initialization follow-up. Native jobs remain queued and unchanged.

**Adding irrelevant input dimensions substantially worsens optimization even when the exact same quartic polynomial remains representable.** Adam performs best among the tested configurations, but its wide recovery prediction fails. Muon is strongly affected by raw parameter scale; fixing that scale improves it without producing accurate recovery at this step budget.

Five distinct structures were tested: generic products, products of squares, a shared quadratic factor within an output, fourth powers, and a shared quadratic factor across outputs. Each target has two outputs and exactly two quartic atoms per output. The same factors and shifted-Gaussian mean are embedded from four coordinates into 1,152 by zero padding; unused coordinates have independent Gaussian noise. Exact Gaussian Gram replay verifies that the teacher function and its population norm are unchanged by this embedding.

The student learns four normalized linear forms per atom, plus profiled coefficients with ridge $10^{-6}$. It has exactly the teacher atom count. Methods share starts; the wide start shares its first four raw coordinates with the small start but adds random nuisance coordinates. Therefore the normalized initial functions differ between widths. These are exact-population weight/moment fits, not sampled text regression. Checkpoints are selected by their fitting objective.

| Width | Configuration | Fits below 5% value error | Median value error | Worst value error |
| ---: | --- | ---: | ---: | ---: |
| 4 | adam | 9/10 | 1.41% | 8.41% |
| 4 | muon_original | 0/10 | 13.81% | 21.66% |
| 4 | muon_match_rms | 0/10 | 51.73% | 82.63% |
| 1152 | adam | 2/10 | 13.85% | 21.70% |
| 1152 | muon_original | 0/10 | 99.99% | 100.00% |
| 1152 | muon_match_rms | 0/10 | 98.41% | 99.62% |

The prediction of at least eight wide Adam recoveries fails (two succeed). The prediction of at most two wide default-Muon recoveries holds (none succeeds). RMS-adjusted Muon fails the predicted 20% median-error improvement: 98.41% versus 99.99% is only a small reduction. The alternative scaling improves direction movement but does not solve the quartic task. These predictions are not circuit-adoption criteria.

## A function-preserving initialization change

The follow-up repeats the ten wide default-Muon cases after scaling each raw parameter row to unit norm. Forward factors are already row-normalized, so this leaves the initial normalized factors, initial polynomial features and fitting objective unchanged. Direct factor replay is checked below $10^{-14}$. All targets, seeds, learning rate, weight decay, step count and ridge remain fixed; raw parameters are not subsequently renormalized after every update.

Median value error falls from 99.99% to **29.69%**, with worst error 38.97%. Median direction rotation rises from about 4.1 degrees to 79.1 degrees. This demonstrates a consequential parameterization/optimizer interaction. But **0/10** reach 5% error, so the follow-up recovery prediction also fails. A large relative improvement from almost-zero prediction is not accurate reconstruction.

## What changes for the native experiment?

Do not equate an exact representational witness or a successful low-dimensional toy with successful optimization at the native width. The pending 250-step native run is still informative about its stated configuration, but an unsuccessful Muon arm is not evidence against the quartic architecture. The initial-direction scale is a concrete confound alongside duration, initialization, Gaussian metric mismatch and nonconvexity. Even Adam fails most wide controls at this budget, so the issue is broader than Muon alone.

These controls use four atoms rather than the native 96 and a teacher with intrinsic dimension four. They do not select a native replacement rate or prove that unit raw initialization will solve native fitting. All response and coefficient errors are retained in the receipts; response error here is an exact correlated-Gaussian metric, not an intervention on text. No artifacts are exported for native use.

The 60-fit suite took 78.7 seconds and the 10-fit follow-up 15.2 seconds on two CPU threads. Existing queued helpers were not modified. Preserve the native outcome, then use its optimization history and these known-capacity failures to distinguish optimization trouble from absent structure before launching another native sweep.

[Original protocol](WIDE_QUARTIC_OPTIMIZER_PLAN_V1.md) · [All 60 fits](WIDE_QUARTIC_OPTIMIZER_V1.json) · [Unit-initialization protocol](WIDE_QUARTIC_UNIT_INITIAL_PLAN_V1.md) · [All 10 follow-up fits](WIDE_QUARTIC_UNIT_INITIAL_V1.json) · [Executable code](toy_wide_quartic_optimizer.py).
