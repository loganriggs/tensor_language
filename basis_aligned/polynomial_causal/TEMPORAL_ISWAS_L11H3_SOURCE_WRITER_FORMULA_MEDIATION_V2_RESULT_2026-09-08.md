# Temporal/is-was selected-writer → L11H3 formula mediation result — 2026-09-08

## Verdict

The controlled-mediator test is valid but asymmetric. It terminates
`writer_formula_partial_mediation`: the exact source-local L11H3 value formula mediates the
selected temporal writers strongly, but only a small fraction of the selected is-was writers.

Result artifact:
`basis_aligned/bilinear_quotient/circuits/followups/temporal_iswas_l11h3_source_writer_formula_mediation_v2_result.json`

SHA-256: `383c162f2dcc505248122cd0d98d5b2bb7afedc019299f9b4be537cc791a18c4`.

## Frozen scores

| role | original FIT | original HOLDOUT | OOD FIT | OOD HOLDOUT |
|---|---:|---:|---:|---:|
| temporal | `.7956 / .9989 / .2079` | `.7993 / .9985 / .2056` | `.7841 / .9978 / .2222` | `.7829 / .9982 / .2223` |
| is-was | `.1775 / .9764 / .8234` | `.1650 / .9873 / .8354` | `.2031 / .9819 / .7979` | `.2025 / .9785 / .7987` |

Cells report signed mediation recovery / cosine / relative residual against the selected writer
effect. Direction agreement is `1.0` throughout. Temporal templates range `.7495-.8053`; one OOD
template falls just below the frozen `.75` template floor. Is-was templates range only
`.1634-.2385`, far below both pooled and template mediation floors.

Prediction A passes: the exact binding, pairing, replay, capture, coverage, finiteness, and
18-forward price hold. Prediction B passes: the arbitrary induced-value formula matches the
observed L11H3 head delta within `9.54e-6`, with selected-logit error at most `9.54e-6`.
Predictions C and D fail. Prediction E passes: later-to-earlier causal zero is exact and formula
collateral is at most `.00203` of command gold.

## Interpretation and next falsifier

This is a directly intervened controlled mediator, not a natural indirect effect and not evidence
for a unique causal path. The failed is-was recovery is not explained by algebra error, reference
replay, missing capture, task collateral, or OOD instability. It says that the selected
`L7H7+L9H4` writer intervention has a large command effect that is not carried by its induced
L11H3 value change restricted to the previously identified postcue source positions.

The next route-closure ladder should reuse the same captures and compare three nested mediators:

1. source-local induced L11H3 value under recipient-native routing (the present arm);
2. induced L11H3 value over the full causal prefix under recipient-native routing;
3. the full observed L11H3 head-3 preprojection delta, including writer-induced routing changes.

If arm 2 rescues is-was, upstream writers spread the task value beyond the command-donor postcue
region before L11. If only arm 3 rescues it, writer-induced routing changes are essential. If arm
3 remains low, most selected-writer behavior bypasses L11H3 and the shared-writer claim must be
split into an L11 source branch plus another downstream route. This ladder directly targets
computational specification, within-module splitting, extraction, composition/reuse, and stable
identification; it is not a rank or compression experiment.
