# Temporal/is-was L11H3 source-writer weight-ordered greedy result — 2026-09-08

## Verdict

The prospectively ordered greedy screen passes all five frozen predictions and terminates
`compact_source_writer_prefixes`. Original FIT alone selected the smallest qualified prefix for
each task, and those identities validated without reselection on original HOLDOUT and both OOD
phases:

- temporal: P3 = `L7H7 + L9H4 + L9H1`;
- is-was: P2 = `L7H7 + L9H4`.

This exactly matches the timestamped pre-outcome forecast. It is prospective compactness evidence
for the physical weight order, not proof of global subset optimality.

Result artifact:
`basis_aligned/bilinear_quotient/circuits/followups/temporal_iswas_l11h3_source_writer_weight_ordered_greedy_v1_result.json`

SHA-256: `6a2ae6598ae95269f7d9e24e13196155342f690d331c78efed5115a8cf84bda2`.

## Frozen selection and validation

| role | selected arm | original FIT recovery / cosine / residual | original HOLDOUT | OOD FIT | OOD HOLDOUT |
|---|---|---:|---:|---:|---:|
| temporal | P3 | `.5294 / .9937 / .4744` | `.5344 / .9949 / .4687` | `.5499 / .9948 / .4536` | `.5606 / .9946 / .4432` |
| is-was | P2 | `1.0994 / .9218 / .4729` | `1.2280 / .9324 / .5279` | `1.6705 / .9690 / .7942` | `1.6884 / .9679 / .8162` |

Direction agreement is `1.0` in every pooled cell. Across individual templates, minimum signed
recovery is `.5150` for temporal and `.9721` for is-was on original text, and `.5314` for temporal
and `1.5142` for is-was OOD; every template has direction agreement `1.0`. Command-gold collateral
is at most `.00472` for temporal and exactly zero for is-was. Reference replay and later-to-earlier
causal-zero errors are exactly zero; all capture/coverage and literal 26-forward price checks pass.

## What changed

The top-five causal unions were sufficient but the is-was union was badly overcomplete at
`2.30-2.91x` recovery. The selected P2 reduces the shared is-was writer set from five heads to two
while retaining held-out/OOD causal direction and staying below the frozen `1.75` OOD ceiling.
Temporal stops at three writers, not five, because favorable finite-prefix interaction lifts the
pre-outcome additive P3 estimate `.4947` above the `.50` FIT floor to `.5294`.

The result supports a shared physical writer core `L7H7+L9H4`, with temporal-specific addition
`L9H1`, upstream of the exact task-typed L11H3 source interface. It does not yet prove that the
selected writer effect travels through that exact formula; the already frozen 18-forward
controlled-mediation test is the next discriminating experiment.
