# Calibration scalar: fitting stability and useful context dependence

Frozen before native execution, 2026-09-10. Original bilinear handoff and pilot
are the scientific authority. This follows the positive V2 removal result;
it cannot retroactively turn reused evaluation texts into pristine discovery.

The alternative is a useful mostly constant offset: removing it can be harmful
even if its context-dependent component is unnecessary. Test this before naming
the scalar a context-sensitive calibration algorithm.

Reuse the frozen V1 rows and V2 producer. Fit two additional covariance axes on
the first and second 24 FIT rows separately. Each fit constructs its own token
histogram and log(1+count) labels. These are disjoint row samples, not certified
independent documents. Both axes remain frozen before the 42 FW_HOLDOUT and
16 PILE_SHIFT rows. Evaluation common/rare classes remain the original FIT top20.
No additional choice of dimension, threshold, gain, or sign based on outcomes.

Write final residual h=b+q*w, with b=h-q*w and original folded q=u^T Q u+beta.
For arbitrary replacement s, A=U*b, B=U*w, c=mean(b^2)+eps32,
d=mean(b*w), e=mean(w^2). Exact conditional logits are

    z(s)=30*tanh((A+s*B)/(30*sqrt(c+2*d*s+e*s^2))).

The derivative before softcap is
    (B*c-A*d+s*(B*d-A*e))/(c+2*d*s+e*s^2)^(3/2).
Multiply by 1-tanh(t/30)^2 for the final logit derivative. Its sign is not
globally fixed. This is our algebraic derivation, not a claim of monotone
frequency control. All native upstream states and coefficients remain charged.

Six arms: native; projection removal along original/A/B axes; replace original
q by its mean across all 48 FIT rows; replace q by another row's folded q at
the same token position. Donor of row i is row (i+1) modulo cohort size, fixed
without inspecting tokens, q, labels or effects. This preserves the empirical
position-conditioned scalar marginal and breaks its recipient pairing. Within
FW this is row-disjoint, not certified document-disjoint. Pile donors are
different documents. Producer/replacement never uses evaluation target labels.

Predictions, scored exactly:

* A instrument: 32 observed body forwards / 126 sequence instances, finite
  outputs; full-FIT direction replay relative error <=1e-10; original folded
  scalar relative error <=1e-5 on both cohorts; native formula and online
  donor bridges max absolute <=1e-3 and relative <=1e-5. One facade replay
  and two online donor executions (native versus folded scalars) per cohort.
* B operational stability: |cos(w_A,w_B)| >=.90; each split axis's centered
  full-vocabulary removal effect relative error <=.20 against the original
  axis, separately on both cohorts; each retains rare CE damage >=.10 nat and
  frequent CE improvement >=.02 nat on both cohorts. No cosine-only promotion.
* C useful pairing: mean replacement AND cyclic donor replacement each add
  >=.005 nat mean next-token CE on BOTH cohorts. Failure closes this registered
  broad context-pairing claim, not the previously established removal effect.

Report all-row losses, common/rare losses, q mean/SD, centered energy fraction,
and donor versus mean loss difference. No unregistered rescue. Later bootstrap
intervals are uncertainty descriptions, not changes to these gates.

A passing bridge means numerical interchange correspondence for this scalar,
not semantic donor-task transfer or standalone token-to-logit extraction.
This distinction follows the intervention correspondence requirement in
[Geiger et al., Causal Abstraction](https://jmlr.org/papers/v26/23-0058.html).
No entire-model parameter saving; original 545,902,902 parameters remain.
32 body forwards, 126 length256 sequences, six final readout arms over58rows;
bounded 900seconds runtime, no training and no full-logit artifact dump.
CPU exact-path/derivative/nonmonotonicity controls precede managed execution.
