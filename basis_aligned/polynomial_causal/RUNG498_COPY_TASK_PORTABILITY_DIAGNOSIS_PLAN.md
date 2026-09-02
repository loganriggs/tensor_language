# Rung498 copy-task portability diagnosis — registered after the null, before CPU analysis

**Registered:** 2026-09-02 18:08 UTC

**Claim level:** post-result diagnosis using only rung498 discovery sufficient statistics. It cannot reverse rung498,
pass its validation gate, or establish a new circuit.

## Question

Rung498 called every position with any earlier equal-token edge “equality positive.” Rung459's known-positive causal
effect used a narrower copy-task definition: the token after the nearest earlier occurrence of the current token must
equal the current next-token target. These are different observations. Determine whether rung498 missed the old
positive because of this task-mask mismatch, because the relation changed on the FineWeb circuit census, or because
it depends on retaining/removing the earlier L5H5 contribution.

## Frozen analyses

On documents0:250 and250:500, recompute without a model:

1. the broad any-equality mask used by rung498;
2. the exact nearest-predecessor copy-positive mask used by rung459;
3. broad equality positions that are not copy-positive;
4. near/far and one/multiple-predecessor partitions inside the copy-positive mask; and
5. exact row overlap between rung459's192 rows and the1000-row circuit census.

For `L5H5 score -> L8H4`, report the pooled recovery, per-document effect cosine, positive-scale residual, native
recipient effect in nat, and hybrid effect in nat under both early-present and early-absent backgrounds. Report the
same recovery/cosine/residual for the `L7H3` score and `L5H5` payload controls. No threshold or scale changes.

## Diagnostic classifications

- `task_mask_mismatch`: in all four half/background cells, the copy-positive mask satisfies rung498 B's
  recovery `[.75,1.30]`, cosine `>=.75`, and residual `<=.70`, while the non-copy equality mask fails at least one
  recovery cell.
- `corpus_or_action_shift`: the copy-positive mask still fails B in at least one cell.
- `earlier_service_interaction`: the copy-positive recovery changes by at least`.20` between early-present and
  early-absent in either half.

The classifications are descriptive. If `task_mask_mismatch` holds, the lawful successor is a newly preregistered
prospective calibration on the still-unopened documents500:1000 using the exact copy-task mask. Rung498 remains an
A-true/B-false result under its registered broad mask. If it does not hold, follow rung498's pre-outcome corpus/action
route instead. GPU use, validation NLLs, hidden states, logits, and model weights are forbidden.
