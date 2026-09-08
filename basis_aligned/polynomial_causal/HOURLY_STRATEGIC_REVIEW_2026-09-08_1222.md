# Hourly strategic circuit review — 2026-09-08 12:22 UTC

## Circuit interpretation targets

1. **Computational specification:** identify what is read, computed, written, and consumed.
2. **Cross-boundary grouping and within-module splitting:** group operationally equivalent pieces
   across native modules and split heads/MLPs when only parts implement the variable.
3. **Held-out/OOD prediction:** predict causal effects on unseen constructions without reselection.
4. **Extraction/sufficiency:** execute an isolated circuit or explicit interface plus background.
5. **Selective manipulation:** support additions and removals while preserving unrelated behavior.
6. **Composition/reuse:** predict joint behavior and shared subcomputations across branches/tasks.
7. **Stable identification:** survive corpus shifts and legal gauges, or use an operationally
   invariant definition.

The full objective remains a smaller transparent tensor program that is jointly predictive on
fresh/OOD text, composable, selectively manipulable, and simpler under literal storage, compute,
edge, state, and program prices. No completed rung yet proves adoption of the full program.

## What changed since 11:22

Five circuit receipts landed in 25 minutes of wall time:

- The cumulative downstream screen selected P7=`A11,M11,M12,M15,M13,M16,M10`, recovering
  `.5276/.5376` on original FIT/HOLDOUT. The additive forecast incorrectly predicted P4; physical
  overwrites delayed the threshold to P7.
- The independent state x response 2x2 found residual identity `.5209/.5088`, module response
  `.4896/.5029`, exact joint `1.0`, and only `.0128/.0148` interaction residual.
- Direct residual v1 preserved exact state identities but was invalid because it compared logits
  outside the registered intervention footprint. The matched-footprint v2 repaired only that
  projection and proved direct/dynamic equality at `3.81e-6--5.72e-6` selected-logit error.
- The scientific direct-route OOD claim still failed: identity recovery fell to `.4582/.4803`.
  Exact algebra is therefore not stable circuit identification by itself.
- Fixed original-selected P7 plus identity passed as an OOD two-stream interface: P7
  `.5685/.5479`, identity `.4582/.4803`, joint exactly `1.0/1.0`, both joint contributions
  material, nonadditivity `.0296/.0326`, and zero temporal collateral.

The next P7 leave-one-module-out necessity test is implemented and hash-bound in managed lane 1.
It is delayed behind verified-live unrelated v258, not by idle circuit work. During the wait, the
A11 all-head/singleton/LOO executor was preregistered and completed conditionally. Its causal-order
mock exposed and repaired a frame-mixing mistake: the no-A11 head tensor must be arm-local after
writer x10 and earlier P7 clamps, not globally native. A prospective exact M11 product-factorial
has also been frozen and its model-free factor algebra implemented.

## Confound audit

- **Baseline subtraction/frame mixing:** repaired before A11 execution; head and MLP factors use
  arm-local live baselines. Parent/full and no-component route replays remain mandatory.
- **Nonlinear loss/effect composition:** conclusions use finite task-margin vectors, not sums of
  scalar losses. The measured P7/identity interaction remains explicit.
- **Shared token difficulty/leakage/post-selection:** the P7 order was selected only on original
  FIT; all OOD tests freeze rows/modules/heads. H3 and M11 are prospectively nominated from older
  original-distribution evidence. No OOD union selection is allowed.
- **Dead knobs/hooks:** every capture/intervention has one-call accounting; all-nine/all-three
  exact route replays are live tripwires.
- **Precision floor:** comparisons use float32-aware state metrics and registered-logit tolerance
  `1e-5`; v1's off-footprint discrepancy is preserved rather than retro-passed.
- **DAS memorization:** no learned subspace is used in the live branch. The physical two-stream
  result directly answers composition and avoids complement-loss shortcutting.

## Alternatives and decision

1. **Native-boundary splitting (current):** LOO necessity followed by A11 head endpoints and exact
   MLP product factors. This directly changes grouping/splitting, specification, and extraction.
   Kill it if full route replay fails or no member/factor is stable on both OOD phases.
2. **Cross-module causal-response basis:** group distributed writes by equivalent downstream
   response rather than native labels. Use only if singleton/LOO evidence remains diffuse; kill it
   if held-out interchange and removal fail. A low-rank reconstruction alone is insufficient.
3. **Signed finite composition algebra:** retain physical streams and use Möbius/ANOVA terms to
   specify interactions. This is exact for registered finite arms but grows exponentially; use it
   only for promoted small sets such as the three MLP factors.
4. **Regularized/constrained DAS:** currently demoted. It could search nonlinear or conditional
   state variables, but prior complement objectives admit task memorization and do not yet beat
   the zero-fit physical interface on manipulation/composition. Revive only with sealed transfer,
   noise/Jacobian stability, and native weight-reader/writer evidence.

The current path remains highest information: it converts an already OOD-composable physical
interface into smaller operational pieces with exact replay controls. Next moves are ranked:

1. Consume the immutable P7 module-necessity receipt; failure of stable module necessity forces
   frozen backward pruning or a cross-module response grouping.
2. If A11 passes, bind and enqueue the completed 20-arm head atlas; all-nine route failure kills
   the instrument, while H3 failure retains a distributed A11 boundary.
3. If M11 passes, bind and complete the exact product-factor executor; all-three route failure
   kills the instrument, while interaction/linear nulls determine the retained algebra.

## Throughput and systems audit

Authoritative serial GPU times for the five receipts were `4.59, 3.11, 4.55, 6.57, 4.58` seconds.
Five receipts landed between 11:22 and 11:48, comfortably faster than one per ten serial minutes.
From 11:48 to this review, time went to scientific interpretation, conditional implementation,
focused validation, and a verified managed-queue wait; no circuit GPU was idle by our control.
The only elevated validation was the direct-route footprint repair and A11 arm-local mock, both
justified by concrete instrument defects they caught. Reusable parent hooks and exact algebra kept
repeated authoring bounded.

`CIRCUIT_FOCUS: PASS` — five causal receipts plus physical within-module split machinery.

`CEREMONY_BUDGET: PASS` — GPU screens were seconds; added validation caught two real scope/frame
defects and did not expand into a bespoke compiler or audit framework.

`NOVELTY_LESSON_GATE: PASS` — prior L11H3 and MLP8 instruments were searched and reused; static
weight norms, global-native frames, OOD selection, reconstruction, and complement-DAS shortcuts
were explicitly excluded.

Concrete continuation: the exact module-necessity runner remains hash-bound behind live v258;
the M11 fail-closed executor implementation proceeds on CPU while that managed predecessor runs.
