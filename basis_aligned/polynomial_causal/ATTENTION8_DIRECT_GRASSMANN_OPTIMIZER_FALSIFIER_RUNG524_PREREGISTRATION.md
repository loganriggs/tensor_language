# Rung 524 preregistration — planted direct-subspace optimizer falsifier

**Frozen:** 2026-09-03 09:27 UTC  
**Owner:** Codex  
**Compute:** CPU only; no model or GPU queue access  
**Claim level:** optimizer-instrument test, never circuit evidence

## Why this is the last optimizer test on the attention8 route

Rungs 522--523 show that Adam applied to an unconstrained matrix and differentiated through a QR decomposition is not
a valid instrument: fixed loss normalization removes the catastrophic spikes, but the best arm still passes only
7/15 leave-one-circuit-out fits. Rung 523's frozen decision closes learning-rate, normalization, and rank tuning in
that parameterization.

Rung 524 changes the mathematical object. It treats a rank-4 subspace as a point on the Grassmann manifold—the set of
all four-dimensional subspaces of `R^64`. At each step it:

1. evaluates the loss using the projector `P = Q Q^T`, where the columns of `Q` are orthonormal;
2. computes the ordinary gradient `G = dL/dQ`;
3. keeps only the tangent component `(I - Q Q^T)G`, which changes the subspace rather than rotating its basis; and
4. takes a backtracked step and orthonormalizes only after the step.

Thus QR is not inside the differentiated computation. This is a direct subspace optimizer, not another Adam setting.
Rank 4 is fixed only because the planted truth has dimension four; recovering that known truth is an instrument check,
not evidence that real attention8 has rank four.

## Frozen planted computation

Dimensions are `D=64`, planted rank `r=4`, three targets, four maps, and 12 scalar output measurements per
target/map. A fixed seed creates an orthonormal planted frame `Q_*` and projector `P_* = Q_* Q_*^T`.

For each target, map, split, and example, draw an input `x`. The exact target response is

`y = A_(target,map) P_* x`,

where `A_(target,map)` is a fixed random `12 x 64` readout. A candidate frame `Q` predicts

`y_hat = A_(target,map) Q Q^T x`.

Control inputs `c` are projected to the orthogonal complement of the planted subspace, so the correct response is
zero. The normalized target/map loss is

`mean((y_hat - y)^2) / mean(y^2) + 24 * mean((A Q Q^T c)^2) / mean(y^2)`.

The fixed denominator is computed from all eligible FIT responses for that target and map. This mirrors rung 523's
repaired scale while giving us a known answer. Each of 15 fits omits one target from training and uses one of five
deterministic random initializations. Training takes exactly 200 direct-subspace updates on the other two targets.
Common validation evaluates all three targets on independently generated examples, including the omitted target.

The OOD split changes the coefficient distribution from Gaussian to a centered heavy-tailed Student-t distribution
and changes the covariance of the orthogonal nuisance/control directions. It remains inaccessible until all FIT and
VALIDATION gates pass.

## Frozen optimizer

- initial step size: `0.5`;
- Armijo coefficient: `1e-4`;
- backtracking factor: `0.5`;
- at most 16 backtracks per update;
- minimum accepted step: `0.5 * 0.5^16`;
- exactly 200 accepted updates per fit;
- objective: maximum normalized loss over the two non-omitted targets and all four maps;
- QR retraction after, and outside differentiation through, each proposed tangent step;
- no momentum, Adam state, weight decay, rank choice, or hyperparameter sweep.

If a step cannot satisfy the Armijo decrease rule, the fit fails immediately and is retained.

## Registered predictions and gates

### Prediction A — numerical and held-out health

All 15 fits must:

- complete 200 accepted updates with finite losses and gradients;
- keep `max(abs(Q^T Q - I)) <= 1e-5`;
- have final FIT loss at most 5% of initial FIT loss;
- have final common VALIDATION loss at most 5% of initial common VALIDATION loss; and
- have zero evaluated losses above 100.

### Prediction B — recover the known subspace

For all 15 fits:

- projector error `||Q Q^T - P_*||_F / ||P_*||_F <= 0.10`; and
- the smallest singular value of `Q_*^T Q` is at least `0.995`.

These are basis-invariant: rotating the four columns of either frame does not change the score.

### Prediction C — OOD response transfer

Only if A and B both pass, open the sealed OOD split. All 15 frozen frames must have maximum normalized OOD loss
across all three targets and four maps at most `0.05`. No refitting or selection occurs after opening OOD.

## Decision and null

The direct-subspace instrument is licensed for one unchanged model calibration only if A, B, and C all pass. It is
falsified if any fit fails its numerical, validation, recovery, or OOD gate. A failure closes this attention8
optimizer route: do not vary step size, rank, dimensions, or loss weights. Pivot to the exact MLP0 token-only,
token-by-context, and context-only decomposition.

Even a complete pass is not circuit evidence. It only permits one model-side FIT/VALIDATION calibration using the
same circuit-selectivity question and frozen downstream measurements as rung 522.

## Literal price and safeguards

The runner records exact objective evaluations, gradient evaluations, accepted updates, backtracks, and wall time.
It asserts CPU tensors, fixed dimensions/seeds/census, a create-only result path, a sealed OOD access state, and no
model imports. Unit tests cover the tangent projection, basis invariance, gate logic, and fail-closed OOD boundary.
