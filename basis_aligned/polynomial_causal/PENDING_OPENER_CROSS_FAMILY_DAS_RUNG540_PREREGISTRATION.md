# Rung 540: cross-family pending-opener DAS as an operational equivalence class

**Frozen:** 2026-09-03 15:05 UTC, before any R540 optimization or model forward

## Question

At the verified residual-stream location entering block 8, does one causal
subspace carry the pending-opener variable across two genuinely different
counterfactual constructions, while ignoring two answer-preserving changes?
Several rotated or physically different subspaces may be valid. Therefore the
primary identity criterion is equality of held-out causal responses, not overlap
of their coordinate bases.

## Intervention and fitted objective

For target state $x$, donor state $x'$, and an orthonormal basis
$Q\in\mathbb{R}^{1152\times k}$, the intervention is

$$
x_{\mathrm{final}} \leftarrow x_{\mathrm{final}}
 +(x'_{\mathrm{final}}-x_{\mathrm{final}})QQ^\top.
$$

For each direction, let $\Delta_Q$ be its donorward closer-logit movement and
$\Delta_{\mathrm{full}}>0$ the saved, row-matched complete-state movement from
R538. Define normalized recovery

$$
r_Q=\frac{\Delta_Q}{\Delta_{\mathrm{full}}}.
$$

Fit $Q$ on FIT only by minimizing the mean squared error $(r_Q-1)^2$ over both
directions. Full-state denominators are frozen inputs, not refit targets. Use
ranks $k\in\{1,2,4,8,16\}$, seeds $\{0,1,2\}$, and three training sources:
direct opener substitution only, structural close-and-reopen only, and their
joint union. Each fit uses 240 deterministic minibatch updates of size 16 with
Adam at learning rate 0.005. QR orthonormalization is applied at every update.

## Held-out selection bars

No SELECT row enters optimization. For each candidate, report each target family
and direction separately. A target cell passes when:

- median normalized recovery is at least 0.50;
- a 2,000-resample group-bootstrap lower bound on mean normalized recovery is
  above zero; and
- at least 75% of individual normalized recoveries are positive.

The surface/distance and non-opener punctuation families are negative controls.
For each SELECT control cell, compute the projector's absolute closer-margin
change and compare it with the row-matched complete-state effect from R539. It
passes only when both the mean absolute change is at most 0.10 logit units and
the ratio to the complete-state mean absolute change is at most 0.25.

Five dimension-matched random subspaces per rank must have mean target recovery
below 0.10. A rank is eligible only if at least two of three seeds for **each**
training source pass every target and control cell. Select the smallest eligible
rank. If none is eligible, report the complete matrix and do not open FINAL_TEST
or OOD.

## Multiple valid subspaces

For the selected rank, choose the lowest-numbered passing seed from each training
source. Construct each projector's ordered SELECT response vector from:

1. row-level normalized recovery on both answer-changing families and directions;
2. row-level normalized signed endpoint changes on both answer-preserving
   families and directions.

For a control row with full-state signed effect $E_{\mathrm{full}}$, the stored
response coordinate is

$$
\frac{E_Q}{\operatorname{sign}(E_{\mathrm{full}})
\max(|E_{\mathrm{full}}|,0.05)}.
$$

The 0.05-logit floor is frozen before fitting and prevents nearly zero
individual control denominators from dominating the equivalence comparison.

Two projectors are operationally equivalent when their response-vector cosine is
at least 0.90 and their response-vector root-mean-square difference is at most
0.15. Also report principal-angle overlap of the fitted subspaces, but do not use
it as an identity gate. High response agreement with low coordinate overlap is
evidence for several valid physical realizations of the same causal variable,
not a failure of identifiability at the functional level.

Interpret outcomes as follows:

- two-way cross-family transfer plus operational equivalence supports one shared
  pending-opener variable;
- strong own-family recovery but failed cross-transfer supports separate or
  construction-specific variables;
- only joint training passing suggests that diverse counterfactuals are needed
  to identify the shared response;
- target recovery with control leakage is a nuisance/position subspace, not the
  desired circuit;
- unstable seeds indicate optimization non-identification and block promotion.

At selected projectors, report doses $0,0.5,1,1.5$; overshoot is a causal
nonlinearity diagnostic and never evidence of stored dimension.

## Scope and price

The actual local checkpoint bytes must be verified. R537 rows, R538 row-level
target ceilings, and R539 row-level control ceilings are hash-frozen inputs.
FINAL_TEST and OOD remain sealed. The maximum planned work is 45 fits × 240 =
10,800 gradient-bearing suffix evaluations, plus no more than 700 no-gradient
suffix evaluations for candidate, random, and dose scoring; zero model weights
are updated. This is an activation-identification experiment, not a rank or
compression result.
