# Native shared/private timing and first-sweep registration

13 September 2026. Follow the user's interaction-compression campaign and the two capacities in `FULLU_SHARED_LOCAL_FEASIBILITY_V1_MATH.md`.

**Object:** all 50,304 centered token quadratic functions from U folded through the last bilinear layer. Exact coefficient Hilbert coordinates from its Cholesky metric; retain the mean exactly. No token text, corpus activations, CE, labels, or task selection enter fitting.

**Arms:** `(global width, groups, private width)` = `(64,32,8)` and `(128,64,16)`. Seed 9518, one initialization and one complete conditional sweep each. This is native scheduling evidence and an initial fit, not a multistart/convergence claim. Do not select representation capacity from language-model validation here.

**Implementation:** smaller-Gram eigensolve for leading banks, direct SVD fallback for undersized or numerically deficient groups, joint code/assignment solve through orthonormal spans. CPU controls compare actual fitted projections to independent direct SVD in tall, wide, undersized and rank-deficient cases. Repeat those controls on GPU before native computation. FP64, TF32 disabled.

Registered predictions:

- `pred_a`: GPU projection/orthogonality controls and independently recorded native full-tensor energy replay within 1e-8.
- `pred_b`: each complete conditional sweep does not increase normalized squared coefficient error by more than 1e-8.
- `pred_c`: initialization plus one sweep takes at most 180 seconds for each capacity.

The null is an expensive or weak initial fit. No observed first-sweep error is evidence against a well-optimized representation. Report full coefficient error, initial and updated centered errors, group counts, timing, and peak GPU allocation. A time limit is not convergence. Hard process limit 900 seconds; zero model body forwards.

**Literal price:** retain native L/R/Down and all bilinear products. Price only the candidate replacement of U: mean, global and private banks, per-token global/private coefficients, and int32 group IDs. Compute all group banks for full-vocabulary evaluation. Neither lower output-map storage nor group assignment establishes semantic circuits or reuse across behaviors.

**Outcome-dependent next step:** use the measured sweep cost to budget ten independent starts at each capacity and refine promising starts. Require explicit intrinsic-gradient and assignment checks, then compare at matched literal storage with the exact global spectral baseline. If costly, improve the demonstrated bottleneck without changing the objective. If the instrument fails, repair it before interpreting fit quality. Native behavioral validation is separate and follows a frozen useful fit.
