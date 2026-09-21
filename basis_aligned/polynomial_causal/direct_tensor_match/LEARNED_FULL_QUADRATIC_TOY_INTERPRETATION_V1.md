**Learning wider first and then deleting products is more reliable than a narrow random fit in these controls**

Five planted families were tested: independent mixed products, a shared input factor, a shared output direction, squares, and redundant cancelling teacher terms. Each function has a four-product representation. All fits use the same exact symmetric coefficient objective plus an artificial directional-response objective, two random starts, and explicit function-error checks. The recovered factors are not presumed unique semantic units.

| Fitting method | Student products | Steps | Recoveries among10family/start cases |
|---|---:|---:|---:|
|Joint Adam, learning outputs and both inputs|4|600|4|
|Joint Muon|4|600|0|
|Exact conditional outputs + Adam input fitting|4|600|3|
|Exact conditional outputs + Muon input fitting|4|600|1|
|Exact conditional outputs + Adam input fitting|4|3000|8|
|Exact conditional outputs + Adam input fitting|6|3000|9|

These rows use learning rate.05; the separately recorded.01runs perform worse under the tested schedule. Recovery means both coefficient and response error<1%. No tested single configuration recovers all10starts. Width6does recover each family in at least one of its two starts. This is evidence about optimizer schedule, width and initialization, not universal Adam superiority or native structure.

Independent controls compare implicit versus dense tensor values/gradients and profiled output solutions versus an explicitly lifted least-squares problem. One initial control incorrectly omitted the solver's tiny ridge from its reference and failed at2.70e-8. Including the identical ridge makes the reference agree below9e-16 without loosening the threshold. Envelope-gradient finite differences pass the existing1e-6bar. Redundant cancelling teacher terms produce the same objective and gradient as the uncancelled base function (within6.1e-17tensor error); their presence alone is not the cause of that example's harder optimization.

**Restricted graph simplification**

Starting from each family's better width6fit, enumerate single-product deletions, refit the remaining input directions and conditional output coefficients for400steps, and accept only if both errors stay below1%. Repeat toward width4. Products reused by several outputs are counted once; all dense input/output coefficients are charged.

Four of five families reach4products and84coefficients from6products/126coefficients. The cancelling-term case remains at6products because the best proposed deletion fails the error limit. Fidelity and the preregistered4/5simplicity prediction both pass. This realizes a restricted two-stage search—wider decomposition proposals followed by deletion/refitting—not arbitrary arithmetic-DAG optimization, exact recovery for all cases, or proof of factor identity.

**Native continuation**

The managed full-layer pilot now allows both input factors to move, with exact conditional output solves, at the existing3686product/14,067,072coefficient budget. It compares inherited and random starts for100steps each, in the historical covariance and source-response metrics. This is a bounded pilot, not a convergence guarantee; fitting loss alone selects the export. Native behavior and the tracked continuation computation must be tested separately before any adoption.

[Joint sweep](LEARNED_FULL_QUADRATIC_TOYS_V1.json) · [Profiled sweep](PROFILED_FULL_QUADRATIC_TOYS_V1.json) · [Longer fit](PROFILED_FULL_QUADRATIC_LONG_V1.json) · [Width6](PROFILED_FULL_QUADRATIC_WIDTH6_V1.json) · [Deletion/refit](QUADRATIC_DELETE_REFIT_V1.json) · [Corrected independent controls](PROFILED_QUADRATIC_CONTROLS_V1.json) · [Cancellation control](QUADRATIC_CANCELLATION_CONTROL_V1.json).
