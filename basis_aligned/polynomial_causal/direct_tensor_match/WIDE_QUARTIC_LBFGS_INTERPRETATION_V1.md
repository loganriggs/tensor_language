# Line search helps the wide quartic controls, after fixing an absolute-scale stop

2026-09-22 04:58 UTC. Two ten-fit controls on the same five axis-aligned wide targets and starts as the earlier Adam/Muon comparison. No native result or queued protocol change.

**L-BFGS with initial-objective scaling recovers 8/10 cases below 5% error.** This is a stronger optimizer candidate than the tested fixed-rate configurations for this particular control, while two remaining failures prevent a general recovery claim.

Every target still uses two outputs and four quartic atoms in 1,152 input dimensions. Only four input coordinates matter, but the optimizer is not given their identity. Factors are normalized in the forward model and output coefficients are solved analytically with ridge 1e-6. This separable linear/nonlinear optimization is an instance of variable projection; solving the linear part does not guarantee a global optimum of the nonlinear factors. [O’Leary and Rust’s treatment](https://www.cs.umd.edu/~oleary/software/varpro/varpro.pdf).

## Why three first attempts stopped immediately

The first L-BFGS configuration uses strong-Wolfe line search, rate 1, history 50, at most 250 quasi-Newton iterations and 500 objective/gradient evaluations. It reaches five successes. Three cases stop after one evaluation while their relative value error is nearly 100%.

Independent reconstruction of those first gradients confirms the mechanism in the installed PyTorch implementation. Maximum gradient entries are 3.7–5.4e-9, above the 1e-12 gradient tolerance, but the first steepest-descent directional derivatives are only -2.1e-15 to -6.0e-15. They trigger the absolute 1e-14 directional-change tolerance before a useful line search. This is a configuration/numerical-scale issue, not evidence of accurate reconstruction. [Recorded checks](WIDE_QUARTIC_LBFGS_STOP_CHECK_V1.json).

The follow-up divides the target-energy-normalized objective by its absolute initial value, floored at 1e-10. That is a fixed positive scalar and leaves the mathematical minimizer unchanged. It changes the optimizer’s numerical stopping and initial update scales. The pending native learner already uses an analogous initial-objective convention; we do not modify its code.

## Results and cost

| Configuration | Successes below 5%, first 251 evaluations | Successes below 5%, full run | Full median value error | Full worst error |
| --- | ---: | ---: | ---: | ---: |
| L-BFGS, target-energy scaling | 4/10 | 5/10 | 4.61% | ~100% |
| L-BFGS, additional initial-objective scaling | 7/10 | 8/10 | 2.57% | 17.98% |

For the scaled version the median error at the first 251 evaluations is 2.71%. Every scaled case proceeds beyond one evaluation, and the registered eight-success prediction passes. Full runs use 281–311 objective/gradient evaluations and all 250 quasi-Newton iterations. The original comparison used 251 objective evaluations and 250 gradient updates for Adam/Muon; our first 251-evaluation snapshot includes 251 gradients. Thus this is nearly matched evaluation accounting, not identical compute, wall time, memory, or update count.

The original L-BFGS suite took 14.85 seconds; the scaled follow-up took 18.88 seconds on two CPU threads. Strong-Wolfe trial points count as evaluations, and the best valid trial checkpoint is retained by objective value. L-BFGS also stores curvature history, which must be priced for a native run. Native loss contractions are much larger, so these toy timings do not predict native runtime.

The corresponding earlier wide Adam run obtained 2/10 successes at 250 updates, while both Muon variants obtained 0/10. These comparisons apply to the exact tested initialization, regularization, budget and objective conventions. They are not evidence that L-BFGS universally dominates Adam/Muon, nor that it will solve the much larger native residual. The two remaining scaled-L-BFGS failures are preserved.

## Next decision

A separately registered native L-BFGS comparison is now scientifically motivated: it changes the optimization mechanism rather than the target capacity. It must retain the native class, starts, exact objective, output protections and evaluation interfaces, report actual function/gradient counts, and be compared against the unchanged Adam/Muon runs. A native success still needs held-out response, removal, composition and simplicity evidence. These controls do not identify semantic circuits.

[Original protocol](WIDE_QUARTIC_LBFGS_PLAN_V1.md) · [Initial-scaling follow-up](WIDE_QUARTIC_LBFGS_SCALE_PLAN_V1.md) · [Original results](WIDE_QUARTIC_LBFGS_V1.json) · [Scaled results](WIDE_QUARTIC_LBFGS_SCALE_V1.json) · [Executable fits](toy_wide_quartic_lbfgs.py).
