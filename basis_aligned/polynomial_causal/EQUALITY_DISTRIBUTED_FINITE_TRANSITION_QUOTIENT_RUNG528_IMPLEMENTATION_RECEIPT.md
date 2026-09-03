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
