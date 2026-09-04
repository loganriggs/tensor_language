# Task-14 localization-v2 physical compiler v3: finite-screen-terminal clarification

**Frozen prospectively:** 2026-09-04 13:23 UTC, while compiler v3 remains a mutable CPU-only draft and before any
v3 compiler artifact, producer, model, checkpoint, GPU, task outcome, queue entry, or execution authorization exists.

**Controls over:**

- `TASK14_SUBJECT_VERB_AGREEMENT_FIT_LOCALIZATION_V2_PHYSICAL_COMPILER_PREREGISTRATION_V3_2026-09-04.md`,
  commit `c08a69c5201e7d9b222a8f41ba52a69be6cf9d83`; and
- the frozen task-14 localization-v2 scientific design and producer-acceptance addenda already named in that
  preregistration.

The v3 preregistration says that the only publishable `instrument_invalid` cases are fully completed, finite
joint-rank-one or selected-family/rank optimizer/seed-health failures. That sentence is too narrow. It accidentally
omits two first-precedence scientific invalidity cases already fixed by section 7 and section 12 of the approved v2
localization design:

1. every scheduled discovery gradient value is finite, but a required gradient denominator is at or below
   $10^{-12}$; or
2. every scheduled discovery ceiling and natural-margin value is finite, but a required natural-margin denominator
   is at or below $10^{-6}$ nat.

These completed, finite threshold failures are publishable `instrument_invalid` terminals. They occur before
eligible-site selection; all later site, fit, validation, necessity, redundancy, and reader fields must therefore be
absent. In the natural-margin case, normalized screen scores and eligibility decisions depending on the invalid
denominator are undefined and must be null or represented by exact empty/zero derived fields. The compiler may not
accept arbitrary unused values there.

This does not change the controlling operational-failure rule. A nonfinite, incomplete, runtime, deadline, memory,
hash, source, schema, canary, call-order, evidence, or publication failure is an operational abort with no scientific
package. A fully completed finite optimizer/seed-health failure at either registered fit stage remains a publishable
`instrument_invalid`. A finite valid screen with no eligible H or Q site remains `no_intervention_ceiling`.

The exact precedence relevant here is therefore:

1. operational faults: abort, no package;
2. completed finite invalid gradient or natural-margin denominator: `instrument_invalid`;
3. completed finite invalid registered fit health: `instrument_invalid`;
4. valid denominators and fit instrument but no eligible H or Q site: `no_intervention_ceiling`;
5. the remaining registered scientific terminals in their unchanged frozen order.

This addendum changes no authority row, partition, donor, model call, fit initialization, seed, rank, objective,
threshold, selection rule, intervention, retained array, price, or retry rule. It is not compiler approval and grants
no producer/model/checkpoint/GPU/outcome/queue access or execution authority.
