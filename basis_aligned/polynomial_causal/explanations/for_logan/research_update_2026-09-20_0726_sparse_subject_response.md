# Subject-number response: a sparse conditional program, with a reuse limit

We can now execute a folded six-block attention/MLP response program without the
checkpoint, given explicitly prepared contexts and starting response coordinates.
It predicts subject-number intervention effects and three modal-reader controls
on fresh regular-noun syntactic structures. Keeping half the MLP numerator pair
interactions also passes tested effect gates. The exported consumer replays on CPU
within1.8e-15; its contexts and native generators remain substantial dependencies.
This is not yet a small token-input circuit.

The stronger reuse test splits earlier contributions into two groups, runs each
through block11, and asks the same suffix to predict either write and their sum.
The nonlinear interaction prediction improves on adding the two effects. But
per-cell preservation predictions fail. Restoring dense cores and adding source
variants to calibration do not rescue this. We retain these failures; readout
sharing alone did not prove source-component reuse.

A likely mismatch is now testable: the old calibration readers held late attention
fixed, although the candidate executes attention dynamically. Exact finite-change
readers now cover both QK products, values, normalization and all six blocks.
Native closure passes at every boundary, but these readers require both endpoints;
they are calibration instruments, not predictors. The next candidate uses full-
native responses and readers instead of the conditional calibration object.

The irregular-noun panel also includes a native capability failure on “The mice
that the tooth saw”. It is grammatically number-labeled but semantically implausible.
No row was dropped. It limits the OOD claim even though the surrogate predicts the
native effect accurately.

[Evidence and prices](../../JOINT_READER_SPARSE_RESPONSE_2026-09-20.md),
[source reuse CPU audit](../../SOURCE_REUSE_CPU_AUDIT_2026-09-20.json),
[three-hour math/literature review](../../THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-20_0726.md).
