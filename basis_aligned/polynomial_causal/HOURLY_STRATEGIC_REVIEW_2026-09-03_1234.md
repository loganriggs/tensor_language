# Hourly strategic review — 2026-09-03 12:34 UTC

## Circuit interpretation targets

A useful decomposition must eventually provide all seven properties:

1. specify what information is read, what operation combines it, what is written, and who reads it;
2. merge pieces across heads/MLPs when downstream computation treats them as one variable, and split native modules
   when their pieces serve different tasks or compositions;
3. predict activations and behavioral effects on held-out and OOD inputs;
4. extract a sufficient executable circuit or a precise interface plus background;
5. support selective removals, swaps, and edits while preserving unrelated behavior and accounting for redundancy;
6. predict composition and reuse when shared and task-specific pieces operate together; and
7. identify units stably across documents, gauges, and fitting restarts, or by a clear downstream equivalence.

The full program goal is a smaller transparent tensor program that is jointly predictive, composable, manipulable,
and cheaper under literal storage, compute, edge, state, and program prices. Lower rank, quantization, reconstruction,
or CE alone is not completion.

## What changed since 11:33

- R529 found a real leave-one-action-out common state in discovery, but it missed the held-out wrong-sign margin by
  about two percentage points. It closed without moving the bar.
- R530 showed that native/transplanted attention0 matchers select nearly the same parameter direction, while their
  named-circuit effects fail to reproduce. Stable parameters are not stable circuits.
- R531 directly compared the two multiplicative score factors across four causally implicated equality heads. No
  whole factor was scalar-equivalent on held-out equality edges. Moderate 78--86% similarities and stable factor
  swaps are structure, not identification.
- Before R531 outcomes, algebra caught and corrected an impossible “beat the optimal product scalar” gate.
- R532 changes the definition of sameness to downstream computation: insert one source factor with the target's
  native companion and compare signed effects over all 32 discovery and 30 held-out circuit families.
- R532's first smoke exposed a BF16/FP32 operation-order mismatch and a non-fail-closed status. V2 fixed both. The
  first full launch then exposed a circuit-mask axis error after one forward and before outcomes. V3 now exercises
  the support accumulator as well as every physical arm before the full run can reopen.

## Is R532 still the highest-information route?

Yes. It can change targets 1, 2, 3, 5, 6, and 7 directly:

- a passing single-factor arm groups part of L8H3 with part of L8H4 below head boundaries;
- two donor-present/absent backgrounds measure whether that equivalence depends on coexistence with the donor write;
- 32+30 frozen circuit families and two new document halves test transfer rather than reconstruction;
- permutation/direct-assignment controls test whether the claimed factor, rather than generic position or scale,
  causes the effect; and
- the 2-by-2 finite difference measures non-additive composition of the two factor replacements.

No rank is optimized. The measured object is a physical attention term and its signed downstream effect.

## Confound audit

- **Baseline subtraction:** every arm is compared with target-absent under the same donor background, so the target
  effect does not absorb the donor-removal main effect.
- **Multiple mediators/nonlinear CE:** donor-present/absent backgrounds expose coexistence dependence; the registered
  second difference reports interaction rather than assuming additivity. CE is still nonlinear, so this interaction
  is explicitly an outcome-scale interaction, not an internal tensor coefficient.
- **Frame mixing:** both heads are in layer 8 and read the same incoming normalized residual. Removing the donor's
  output cannot change either head's score in that layer; it changes only the downstream background.
- **Shared token difficulty:** all comparisons are paired on identical documents, tokens, and masks. Member and
  slice-control effects use the same absent baseline.
- **Overlapping circuit masks:** the 62 tags are not an orthogonal or statistically independent basis. Their effect
  vector is an operational set of downstream questions. A cosine match means the same signed effects on those
  questions, not a proof of 62 independent latent variables.
- **Leakage/post-selection:** R531 selected the pair and scales using rows 0:500 and raw factor statistics only.
  R532 uses rows 500:1000 and never switches pairs, scales, arms, or thresholds after outcomes. The 32/30 tag split
  predates both rungs.
- **Dead controls:** v2 proved every donor and target edit is live. V3 additionally executes the real circuit-support
  accumulator. A failed smoke cannot open results.
- **Precision:** native identity now multiplies BF16 factors before FP32 conversion, exactly matching deployed code;
  hybrid arithmetic remains the preregistered FP32 construction before the model-boundary cast.
- **Small supports:** a CPU census found no empty tag/mask/half cell; the smallest held-out member supports are 32 and
  46. Per-tag estimates may still be noisy, so gates use whole 30/32-dimensional paired fingerprints and both halves.

## Genuinely different next routes

1. **Current finite factor-by-companion intervention.** Kill it if the product control fails, either single-factor
   arm fails the fixed downstream bars/controls, or identity/support checks fail. A positive opens OOD confirmation,
   not immediate adoption.
2. **Shared factor-feature vocabulary constrained by downstream effects.** If literal factors fail but show structured
   similarity, learn smaller reusable factor features jointly across heads, with held-out circuit-effect prediction
   and physical feature swaps as the objective. Kill it if features are unstable across document halves or cannot
   selectively reproduce circuit effects. Sparsity/rank would be matched-capacity controls, not the success metric.
3. **Exact local interaction tensor.** Differentiate or finite-difference the calibrated downstream readers with
   respect to score1 and score2 perturbations, then factor the resulting source-factor-by-companion response tensor.
   Kill it if tangent predictions do not forecast finite swaps or if the response factors rotate across corpora.
4. **Predictive-state causal quotient.** Group interventions by equivalence of their complete future 62-circuit
   response distributions rather than by activations. Kill it if equivalence classes fail held-out interchange or
   require storing essentially every native action.
5. **Exact algebra/minimal-realization route.** At the next three-hour review, map the finite action/background/readout
   table to a Hankel or weighted-automaton realization and determine whether a canonical minimal state exists under
   the actual nonlinear suffix. Kill it if closure/compositional assumptions fail on registered action products.

R532 remains first because it supplies the finite causal table needed by routes 2--5 and can falsify literal factor
reuse at low conceptual ambiguity. The next safe action is the source-frozen v3 smoke, followed by the full run only
if every structural and numerical clause passes.
