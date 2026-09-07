# Hourly strategic and circuit-throughput review — 2026-09-07 21:15 UTC

## Circuit interpretation targets

A useful circuit must state what is read, computed, and written; group semantic variables across
native modules while splitting modules where required; predict held-out/OOD effects; provide an
executable sufficiency interface; support selective removal, swapping, or editing; compose and
reuse shared subcomputations; and remain stable across data splits, gauges, and fitting restarts.
The full program goal remains predictive, composable, selectively manipulable, and simpler under
literal storage, compute, edge, state, and program price. Compression is not identification.

## What changed since 20:14

The attention operation retry was repaired from additive to absolute downstream clamps and became
valid. It showed that homogeneous value transport carries `.83077/.85314` A1/A2 but flips five P
and one C rows, whereas pattern and interaction are mostly selective but carry only `.09-.15`.
The clamp distinction is now enforced by a reusable contract and regression tests.

Two subsequent screens finished at roughly nine-minute claim-to-result spacing. The exact 64-arm
head-specific value lattice found no selective allocation: all strict zero-flip arms omit value and
reach at most `.12692` A1. The 12-component head-by-factor dual greedy then evaluated 143 unique
arms in 152 forwards. It is mechanically valid: self error is zero, factor closure is `1.14e-5`,
and full-parent numeric replay is `4.62e-7` with exact categorical agreement.

The dual-greedy terminal is `no_selective_dual_greedy_program`. Its best zero-flip/low-KL visited
arm reaches `.13763` A1. The target-first path reaches `.84510` A1 but has five P and two C flips;
the full program reaches `.80535/.86829` A1/A2 with five P and three C flips. This rejects the two
registered greedy paths, not every continuous or nonlinear combination.

The broad circuit lane increased from at least 63 to at least 74 distinct circuit records during
the hour, while the temporal branch added two honest null receipts. The prior ceremony failure was
repaired: both new screens reused the shared absolute-clamp machinery and landed in about nine
minutes each without instrumentation retries.

## Confounds and path decision

- Baseline subtraction and frame mixing are controlled by absolute base-plus-projected-response
  installation, aligned equal-length rows, and exact parent replay.
- Nonlinear loss composition remains central: discrete factor combinations do not imply that no
  continuous projector works.
- Shared token difficulty is separated through P and C panels, but A2 helped earlier candidate
  localization and is therefore a no-reselection construction check rather than pristine discovery.
- Post-selection is the main threat for DAS. A1/P parity must select all hyperparameters while A2/C
  remain sealed.
- A complement-only objective has a dead-knob solution. Target transfer must be a hard feasibility
  constraint, not one term that can be sacrificed for low complement loss.
- Noise is a local sensitivity penalty, not new identifying information; full-vocabulary KL and
  fold stability provide the additional observations.

The current route remains highest-information only after changing the optimization object. Another
exact subset or coefficient sweep is unlikely to resolve the target/control overlap exposed at
every value-carrying head. The successor is therefore a head-local orthonormal DAS projector with
hard A1 feasibility, worst-group P KL, Gaussian-noise sensitivity, and cross-fold stability.

Alternatives are ranked as follows:

1. **Target-feasible regularized head DAS.** Changes within-module splitting, selective
   manipulation, held-construction prediction, and stable identification. Kill it if no moved
   checkpoint is A1-feasible on both parities, if it does not beat DIM, or if sealed A2/C fail.
2. **Weight-tensor QK/OV writer-reader translation.** Changes computational specification and
   extraction. It becomes primary only after a selective response subspace exists; before then it
   risks translating a nonselective probe basis.
3. **Input-conditional/nonlinear routing.** Changes composition and selective manipulation. Open it
   only if the fixed linear projector family is infeasible or construction-specific.
4. **Tensor-train or rank compression.** Demoted: it changes literal price but none of the present
   identification decisions.

The target-feasible DAS preregistration is now written at
`circuits/prior_art/temporal_iswas_v15_head_response_target_feasible_regularized_das_v1.json`.

## Throughput audit

The serial temporal path was approximately `prior-art/spec -> implementation/test -> managed GPU
run -> scored null` in nine minutes for the head-specific lattice and nine minutes for dual greedy.
Scientific design plus execution again dominated; reusable validation remained smaller, and GPU
runtime was under thirty seconds for the largest screen. There was no idle wait and no direct GPU
launch. The managed runners remain healthy.

CIRCUIT_FOCUS: PASS

CEREMONY_BUDGET: PASS

NOVELTY_LESSON_GATE: PASS
