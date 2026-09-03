# Rung 528 CPU scoring implementation receipt

**Implemented:** 2026-09-03 10:52 UTC

The first CPU-only implementation unit is complete. It fixes the scalar orientation, response comparison, four-arm
continuation interaction, materiality rules, and pre-control discovery gates before any model outcome is available.

One design issue was caught before freezing the code. Because `Z7` and `Z8` already include the validated sign
correction, permitting a second negative fitted scale would make the uncorrected wrong-sign controls pass. The final
preregistration and scorer therefore require `0.25 <= beta <= 4`. The planted suite recovers a positive relation with
`beta=1.5`, rejects the same source with the wrong sign, rejects an unrelated source, and verifies the exact
two-continuation factorial interaction.

Seven CPU tests pass, covering exact positive and negative least-squares scales, zero-source failure, the registered
factorial arm order, input shape/finiteness checks, no refit on the second document half, and the full planted suite.

Frozen hashes after the sign-control correction:

- preregistration: `96b62e3265698467be05848bf239dc49fb4daecbdea7145e4955c550ade5ea2d`
- math implementation: `1363371d5df5a8e8e14682907a172ccb65b39df7c5c5e9ffb3ce0dfe72ae5728`
- tests: `bb16476d2c62f010746296095affb61fab0bac6e2f28413f2c60203bfd63b6ea`

GPU remains ineligible. The next implementation unit must expose the raw post-MLP12 residual, prove self-insertion
reproduces every action's suffix, and account exactly for the carried embedding and first-value attention state.

## Raw-boundary implementation — 2026-09-03 10:58 UTC

That next unit is now implemented. The new source-closed runner copies the observed model's exact residual recurrence,
captures the unnormalized residual immediately after MLP12, and permits exactly one replacement there. It separately
records the embedding skip state and the attention first-value state. Only the registered attention14 and MLP17
continuation writes can be captured or replaced.

The managed smoke will make 22 forwards on four documents: one direct native run, one score-absent run, four action
runs, and 16 boundary-insertion runs covering four actions by four continuations. It requires exact native replay,
exact self-insertion logits and boundaries for every action, identical carried states, live action transitions, live
continuation patches, and exact call counts. It retains no task or circuit effect.

Five new runner tests join the seven scoring tests, for 12/12 CPU tests passing. They cover frozen dependencies and
the 62-circuit population, single-rounding BF16 boundary construction, fail-closed shapes/scales, a toy 18-block
recurrence proving that replacement occurs after MLP12, continuation patch accounting, and the frozen `1,984`/
`11,330` forward prices. Both experiment scripts pass the static queue gate, the full fast suite passes, and the
GPU-free plan preflight reports no findings.

- runner: `10883ffcee417d88540339eeaa7d88a8bf7dcf818bfdf40728a43a7619b8b9df`
- smoke wrapper: `9a12418cbb4b0c6272b3fe47fbc1f499690975654754fcf08bc3bb21f8e709fa`
- runner tests: `0d261d64eb2ced7d88b38901466518725a59e717b541a67a8e46f671ad74b763`

The managed smoke becomes eligible only after this implementation unit is committed and pushed.

## Dry-run wrapper correction — 2026-09-03 11:01 UTC

The first wrapper hashes above are historical and must not be re-enqueued: their preflight path ignored
`BQLIB_DRYRUN`. The corrected runner accepts an explicit smoke output path, and both wrappers execute only the CPU
dry run during queue preflight. The new v2 namespace remains absent after that preflight, proving it did not touch
CUDA or write an outcome.

- corrected runner: `08f19c7dfd22c52786188cbdd7351c00bbddc219595fc232ec70e6049e12e8a8`
- corrected original wrapper: `a89be4a915b0013c53e056eb4ec4231d456946149609bd4e756e4d01348a9ca1`
- dry-run-safe v2 wrapper: `ea5e4aec0b0d3b4a8b32c1bfb4f31b59b62faede576ddaa6e4145edc5f33d927`

The v2 wrapper passes the static gate, its preflight reports `model_loaded=false` and `outcomes_opened=false`, the
combined 12-test CPU suite still passes, and the v2 result namespace is sealed. It is eligible for managed enqueue
after this correction is committed and pushed.
