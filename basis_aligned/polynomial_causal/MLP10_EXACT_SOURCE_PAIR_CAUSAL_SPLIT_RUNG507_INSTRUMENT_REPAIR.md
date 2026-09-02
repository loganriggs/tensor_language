# Rung 507 instrument repair: separate BF16 output rounding from the named bilinear terms

Status: frozen after the first managed CUDA smoke opened only numerical diagnostics and before any task attribution,
candidate, finite-removal, confirmation, or validation outcome was computed.

The original rung507 smoke passed every algebra and liveness check except one. The 253 named input-pair terms plus
the terms containing the explicit normalized-input remainder reconstructed the independent float32 MLP10 score
change with relative squared error `1.71e-16`. Each independent float32 MLP10 write matched its deployed BF16 write
with relative squared error at most `1.12e-5`, inside the registered `2.44e-4` bound. However, the relative squared
error between the two *changes* was `.01629`: subtracting two separately rounded, similar writes makes the rounding
residual large relative to their much smaller difference. Thus the original change-level BF16 bound was not a valid
consequence of the per-write BF16 bound.

The repair does not relax that failed check or alter any scientific selection bar. For each score condition `a`, add
the explicit deployed-output rounding remainder

`rho_a = deployed_BF16_write_a_as_float32 - independent_float32_write_a`.

The exact deployed score change is then checked as

`delta_deployed = delta_named + delta_input_numerical + (rho_a - rho_absent)`.

The repaired instrument requires:

- the original float32 named-plus-input-numerical change closure at relative squared error at most `1e-10`;
- each independent float32 write versus its deployed BF16 write at relative squared error at most `16*(2^-8)^2`;
- the repaired deployed-change closure, including the explicit output-rounding remainder, at relative squared error
  at most `1e-12`; and
- the original pre-repair change discrepancy remains reported as a diagnostic and is not silently reclassified as a
  passing result.

`rho` is bookkeeping for the actual deployed numerical computation. It is excluded from the 253 named terms, the
gradient selector, all finite removals, same-output tests, shared-input labels, and every composition fit. A term
removal still subtracts only the named score-dependent bilinear term from the actual deployed MLP10 write. Therefore
this repair changes only whether the implementation can verify the BF16 write it is intervening on; it cannot change
which term qualifies or any measured causal effect.

The next managed smoke must print both the failed pre-repair discrepancy and the exact repaired closure. Only after
that smoke passes may the original frozen scientific experiment run.
