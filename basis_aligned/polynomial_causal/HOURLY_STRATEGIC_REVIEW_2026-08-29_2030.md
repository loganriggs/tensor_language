# Strategic review — 2026-08-29 20:30 UTC

## Outcome first

The audited free-factor MLP2 experiment is complete. At the same rank-512 price as
the failed native-coordinate programs, arbitrary learned factors reduce the CE cost
from 0.16222 nat for deleting MLP2 to 0.05147 nat. This is a real 68.27% recovery,
but it is not a faithful port: the preregistered limit was 0.02 nat, and all absolute
faithfulness gates failed. The run's formal status is `optimization_failure`, not a
rank-512 impossibility result, because local write NRMSE was 0.6866 and all 1,200
steps were used while the development curve was still declining.

Detailed definitions and numbers are in
`MLP2_RANK512_REFIT_V2_RECOVERY_FINDINGS.md`.

## What ran and how long it took

1. Registry-fresh data freezing created 192 training and 192 never-before-used
   evaluation documents in about 30 seconds.
2. The first scientific attempt trained for about 87 seconds, but a concurrent Git
   branch move tripped an overbroad lifecycle check before the candidate bundle was
   saved. No evaluation rows were opened. The failure is permanently recorded.
3. Recovery hardening and adversarial testing added stable byte-pinned lineage and
   final publication-boundary checks. The independent audit ran 16 focused tests in
   2.15 seconds and issued GO without seeing outcomes.
4. The recovered end-to-end run took 118.29 seconds: one capture of native MLP2
   training states, 600 `DOWN512` optimizer steps, 1,200 each for `FULL512` and
   `RANDOM512`, then six physical whole-model arms on 192 evaluation documents.

The mathematical training and final evaluation are now only a roughly two-minute
job. The avoidable delay was lifecycle recovery, not data loading or GPU throughput.
The recovery was necessary because the evaluation role was genuinely sealed; the
experiment could not be silently resumed after changing source rules.

## Honest progress balance sheet

These measures answer different questions and must not be conflated:

- **36/36 intervention surfaces** are structurally addressable. This means we can
  replace every attention/MLP site through the dispatcher; it does not mean their
  algorithms are understood.
- **5.348245316% of model storage** has a certified removable description.
- **10.923302467% of the measured causal CE gap** has a named and recovered causal
  account. The remaining named-gap complement is **4.72714 nat (89.077%)**.
- **0/68 terminal actions** currently satisfy the complete extraction/removal/OOD
  standard.

The new MLP2 result does not move those strict numbers because it has not been
replicated, composed with upstream programs, or tested for terminal extraction,
selective removal, and OOD transport. It does materially improve our search: it
rejects native channel identity as the only rank-512 grammar and shows that joint
factor learning has high return at a fixed simplicity price.

## Largest remaining gaps

1. **Interface composition.** We have several locally useful compressed programs,
   but do not yet know whether their errors compose or whether later live layers
   repair each one separately.
2. **Objective alignment.** Local MLP2 write NRMSE is 0.6866 while final CE damage
   is only 0.0515. Unweighted local MSE is evidently not the same metric the rest of
   the model cares about.
3. **Semantic coordinates.** The random-start full fit is only 0.00592 nat behind
   the informed full fit. We have useful mixed factors but not stable semantic names
   or a uniqueness theorem for them.
4. **Late consumers.** The four-head copy bundle is causally important, but its
   current position-mean replacement causes 0.024409 nat off-target damage, above
   the 0.01 limit. Capitalization, numerical formatting, syntax, and entity consumers
   are not yet a verified bank that earlier writes can be interpreted relative to.
5. **Terminal utility and OOD.** No circuit yet has all of faithful prediction,
   independent extraction, selective removal, and distribution transport.

## Confusing results that constrain the story

- Keeping 512 original MLP2 products is worse than deleting all of MLP2. Relearning
  just their output matrix and bias fixes much of that damage. This is coordinated
  cancellation, not a bag of independently useful neurons.
- A freely learned rank-512 program is much better, but a random native starting
  support nearly catches it. Native channels are therefore poor candidates for a
  canonical semantic alphabet by themselves.
- The free program nearly preserves task accuracy (only 0.54 percentage points
  lower) while disagreeing with the original top token 12.48% of the time. A model
  can be a good extractor/predictor without being a faithful emulator.
- A shared document-level response direction explained 97.62% of variation across
  earlier MLP2 interventions, yet document mean and second moments predicted only
  4.64% of its squared error. The regularity exists downstream but is not a simple
  global context gate.
- Earlier MLP0-C512 distortion was attenuated downstream, but aligned MLP2 suffix
  writes did not beat a shuffled control. “MLP2 compensates” remains plausible only
  in a broad systems sense, not as the specific aligned repair already tested.

## How simplicity and the mathematical reviews helped

This assay used an operational definition of simplicity rather than a visual one:
fixed numbers of products, stored coefficients, dense multiplications, and native
component calls. `LOCAL512`, `DOWN512`, `FULL512`, and `RANDOM512` all have 512
products and 1,770,624 coefficients. The result therefore identifies **better
coordinates at equal executable complexity**, which is something a raw
reconstruction score alone could not establish.

The tensor/CP viewpoint supplied the free-factor grammar. Gauge-quotient mathematics
supplied a function-preserving minimum-norm canonicalization and exposed that our
absolute float32 canary tolerance was poorly scaled. Causal abstraction and MDL
motivated testing downstream consequences and equal price rather than declaring a
low local MSE “simple.” These were concrete gains. HOSVD, local polynomial
extrapolation, simple document moments, and native K sweeps have not produced a
composable program and are currently pruned. Hankel/minimal-realization ideas have
not yet yielded an executable result.

## Ranked next actions

1. **Fresh-row MLP0-C512 × frozen MLP2-FULL512 composition telescope.** Highest
   information per GPU second and directly tests whole-model composability. Measure
   each arm and the interaction
   \(I=\Delta CE_{0+2}-\Delta CE_0-\Delta CE_2\). Negative interaction indicates
   shared/redundant error or repair; positive interaction indicates an incompatible
   compressed interface.
2. **Downstream-sensitive rank-512 fitting on a new evaluation role.** Replace
   unweighted write MSE with a teacher-logit, Fisher/Gauss--Newton, or consequence-
   weighted objective. The cheap falsifier is whether it beats frozen `FULL512` at
   the same price on final CE/KL without increasing OOD or off-target damage.
3. **Build a verified late-consumer bank.** Start with capitalization and numeric
   formatting, then syntax/entity and the already found copy bundle. Each candidate
   must pass sufficiency, necessity, off-target, and shuffled controls. These
   consumers give earlier MLP factors an operational semantic basis.
4. **Conditional block-term/shared-dictionary MLP2.** Permit a small gate to select
   a few jointly balanced quadratic blocks rather than single products. Its value is
   falsified if it cannot beat the ungated fixed-price curve after accounting for
   gate price.
5. **Copy-head conditional interaction analysis.** Test subsets or a structured
   Shapley screen of the four-head bundle to find a selective terminal action. This
   is cheaper than broad searching but narrower than resolving the MLP interfaces.

The first action is the next safe unblocked experiment. No user decision is needed;
it must use fresh rows because the present evaluation role was opened by this result.
