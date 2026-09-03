# R585 third repair: independent exact-byte pre-execution review

<!-- BQLANE: cpu -->

**Review date:** 2026-09-03 UTC
**Reviewed commit:** `1143aab7c444f32ff1f3fc59942c61ff652cb7d2`
**Verdict:** **BLOCKED — do not execute or enqueue these exact bytes**

## Exact candidate and outcome boundary

This review used immutable Git blobs from commit `1143aab7c`, never the moving
working-tree producer:

- producer:
  `a3987dc053ba9b18a92a950c526acb1127f2cec9ee97d1142158ca4ef6483ddd`;
- owner test:
  `5365d3d473f3385d3b052f7ff09af78f8e2209a0d3e6a75eca264beaf082c11f`;
- dry run:
  `de33550e530c35c1236095e2354d3724c7ff70de16242424f17d5ed7a81433a6`;
- managed adapter:
  `064b6bf1abdde4d196c43fbae0d589949778dea7ef0f167bebb00e26f5274e21`;
- adapter test:
  `95d60bbca55f3dfd01fe9be92a26ad60e1b174a6006a68ab278cacb4f2a6e542`.

At review start and again at `2026-09-03T22:32:13Z`, the live R585 result,
receipt, and evidence namespaces were absent. I used only existence checks and
never opened or created an R585 outcome. No model, CUDA, GPU, queue, board, or
registry was opened.

## What is now closed

The third repair closes the preceding computation-binding blocker.

1. Endpoint evidence is checked field-by-field against the frozen tokens,
   length, final position, source/payload positions, condition, and answer IDs.
2. Every directed row is joined to its frozen recipient, donor, semantic
   metadata, answer IDs, and one of exactly three arms.
3. Factor-exactness rows have an exact schema and are recomputed from the saved
   endpoint $e/u$, canonical term, independent remainder, and head output.
4. For every direction, arm, and site, the validator reconstructs the inserted
   vector from the saved factors and checks

   $$
   \mathrm{live\_removed}+\mathrm{hook\_delta}=\mathrm{inserted}.
   $$

   It also recomputes all four delta norms and their median activity.
5. Endpoint replay/native margins and CE values are recomputed from their saved
   logits. Directed margin, CE, vocabulary RMS, and the causal sufficient
   statistics $n,d,q$ are independently recomputed from the joined endpoint and
   intervention measurements.
6. FIT scales are recomputed from the directed rows. Each scored split's entire
   report, 124-cell bootstrap realization, and scientific failure lists are
   recomputed with the frozen 2,000 replicates.

The fresh tests reject isolated mutations to every item above. A report swap
with fixed JSONL rows is rejected. A correlated JSONL/report change reaches the
earlier $n,d,q$ join and is rejected when the endpoint measurements remain
fixed; recomputing only the summary cannot hide a changed primitive causal row.

Phase handling is also exact across all 13 terminals:

| Terminal phase | endpoint rows | arm rows | factor rows | operations | scored reports | forwards |
|---|---:|---:|---:|---:|---|---:|
| FIT stop | 1,728 | 11,232 | 6,912 | 13,824 | none only for FIT invalid instrument; otherwise FIT | 459 |
| FIT + SELECT | 2,592 | 16,848 | 10,368 | 20,736 | FIT only for SELECT invalid instrument; otherwise both | 690 |

FINAL_TEST and OOD remain closed. The maximum price remains
$459+231=690$ forwards, with zero backwards and zero updates. The managed
adapter still performs byte verification and conservative producer recovery
before the unused-namespace check. Recognized partial state is quarantined;
complete or arbitrary namespaces are preserved and refused.

## Remaining blocker: invalid-instrument clauses are not evidence-derived

The new score reconstruction verifies scientific null and held decision lists,
but `invalid_instrument` and `select_invalid_instrument` take different paths:

- an FIT invalid-instrument result returns from saved-score validation after
  checking only that `fit_scales == {}`;
- a SELECT invalid-instrument result reconstructs the completed FIT report but
  never reconstructs the SELECT integrity-failure list.

Therefore arbitrary text can replace the actual integrity failure, with
`failed_clauses` changed to match, while preserving the same invalid terminal.
This is not merely a low-level hypothetical. The independent test constructs a
fully completed FIT evidence package with exact hashes, schemas, ID orders,
endpoint semantics, directed semantics, factor rows, arm-specific vectors,
delta norms, endpoint measurements, primitive identities, $n/d/q$, zero
instrument maxima, and an empty structural-check list. It then claims
`invented-unrelated-integrity-failure`. Full `validate_result` accepts it.

Several producer outputs needed to derive the real list are currently trusted
but not validated:

- `raw_evidence.structural_identity_checks` is not required, checked for exact
  manifest membership, or mapped back to structural failure clauses;
- native-attention reconstruction and replay/native comparison maxima are not
  joined to the clauses they supposedly generated;
- padding-tripwire failures, per-endpoint factor/reconstruction failures,
  frozen-insertion failures, and primitive failures are not assembled into the
  exact phase-specific invalid-instrument list during result validation.

This matters even though an invalid terminal makes no positive scientific
claim. The receipt licenses one exact reason why the experiment was invalid;
currently that reason can be unrelated to the evidence, so a later repair or
rerun cannot tell which integrity condition actually failed.

## Required repair

1. Define one deterministic function that derives the exact FIT or SELECT
   invalid-instrument clauses from validated evidence. Use it both in execution
   and in result validation, or implement an independent validator equivalent.
2. Require an exact schema for all raw-evidence fields. Require
   `structural_identity_checks`, validate its exact manifest membership and
   finite values, and derive structural clauses from it.
3. Materialize enough per-batch/per-endpoint raw instrument evidence to recover
   the exact native-attention, replay/native, and padding clauses rather than
   relying only on aggregate maxima. Join factor, frozen-insertion, primitive,
   and hook errors already present in the saved rows/arrays.
4. For FIT invalid, require the reconstructed FIT invalid list and no scales or
   scores. For SELECT invalid, require the reconstructed SELECT invalid list
   plus the already reconstructed FIT scales/report/failures. Prefix SELECT
   clauses only after reconstruction.
5. Add the full completed-result attack and both phase-specific low-level
   attacks from the independent test to the owner suite.

Nonfinite values already fail closed before publication and must continue to
abort as integrity errors rather than becoming scientific nulls.

## Validation performed

- All five immutable candidate hashes matched.
- Exact row, manifest, replacement-adversarial, and producer-owner suites:
  **85 passed**.
- Exact adapter suite, after restoring the producer test's path-dependent dry
  run in the disposable worktree: **11 passed**.
- Producer gate/preflight: **PASS/PASS**.
- Adapter gate/preflight: **PASS/PASS**.
- Managed `BQLIB_DRYRUN=1` adapter: **PASS**, with zero model/GPU work and no
  opened outcome.
- Fresh independent test:
  `basis_aligned/bilinear_quotient/ops/test_induction_selector_payload_frozen_factor_rung585_final_review_adversarial.py`.
  Default: **8 passed, 3 strict xfailed**. With `--runxfail`, all three attacks
  fail as `DID NOT RAISE`: the full FIT completed-result attack, the direct FIT
  invalid-list attack, and the SELECT invalid-list attack.
- Fresh test gate/preflight and `git diff --check`: **PASS**.

## Disposition

Do not execute or enqueue commit `1143aab7c` with the five hashes above. The
semantic computation and scientific-score chain is now strong, but invalid
terminal reasons must be deterministically reconstructed from the saved raw
instrument evidence before these bytes are execution-ready.
