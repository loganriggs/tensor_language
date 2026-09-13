# Grouped readout factors fail the regional circuit preservation checks

13 September 2026. [Protocol](SHARED_LOCAL_REGIONAL_V1_PREREGISTRATION.md), [result](SHARED_LOCAL_REGIONAL_V1_RESULT.json), [same-target comparison and sign audit](SHARED_LOCAL_REGIONAL_COMPARISON_V1_RESULT.json).

**Neither grouped factorization preserves the tested circuit accurately enough. The larger model is worse than its matched global baseline in every context group**, despite better aggregate coefficient fit and FineWeb KL. This is an executed circuit-level limitation, not a rejection of all unsupervised compression or of the uncompleted optimization.

## Native cache and target

The validated three-trajectory executor was reused for 120 previously tested prefixes. Its 360 body forwards and 360 additional MLP evaluations took 6.92 seconds; original native/background outcomes replayed exactly. The [new cache](COMPOSED_LAST_BLOCK_STATES_V1_ARTIFACT.pt) stores the additive background and full-head, compact-head and MLP-only final states, together with their matching linear parts. It introduces no new rows or discoveries.

Frozen scoring changes only output readers. Compare each generated state's target/control margin minus the same program's additive-background margin with that predictor under native readers. This preserves the counterfactual definition and separately records background drift. The quadratic-only route replaces U's reading of the bias-free bilinear contribution; the whole-U route changes every final residual reading. Normalization and separate token softcaps remain exact for each state.

The CPU native-reader replay differs from cached FP32 margins by at most 6.06e-6. Full/compact target-effect vector errors are 0.077–0.384%, below the registered 5% instrument ceiling and much smaller than the compressed effects' errors. Frozen scoring took 4.93 seconds, with no new forwards or fitting.

## Compact predictor, quadratic-only replacement

Relative errors below compare the same compact predictor before and after readout compression. They are additional errors, not the original compact predictor's approximately2–4% error against the local native interaction.

| Context group | Grouped G64 | Global rank78 | Grouped G128 | Global rank167 |
|---|---:|---:|---:|---:|
| Old anchors | 19.13% | 21.37% | 17.28% | 10.35% |
| Reader reply | 14.36% | 15.42% | 18.22% | 10.21% |
| Grew up | 17.60% | 18.29% | 16.73% | 9.76% |
| Return home | 10.98% | 11.52% | 12.02% | 6.90% |
| Exact spelling | 7.63% | 7.95% | 10.65% | 5.03% |

The registered grouped target bar requires at most10% in all groups for both full and compact variants. Both capacities fail. Full-head variants show the same ordering: G64 beats its matched baseline in all five groups, whereas G128 loses in all five. Increasing capacity improves the aggregate weight objective but is not uniformly better for this circuit.

The compact quadratic-only variants reverse14/G64 and9/G128 target signs. The largest native effect among those reversals is2.37e-4/G64 and7.64e-5/G128, so the misses cannot all be dismissed as tiny rounding effects. The unrelated control-effect deviations reach6.13e-4/G64 and5.82e-4/G128, above the1e-4 ceiling. Thus control preservation also fails.

Whole-U replacement is worse: compact target-effect errors range54.84–70.62% for G64 and51.54–65.74% for G128. All individual endpoints, zero/sign cases and background drift are retained in the receipt; no successful subset replaces the registered all-group criterion.

## Interpretation and next direction

The result answers a question the FineWeb average could not: these frozen output clusters are not currently faithful replacements for the regional interaction. The matched-global comparison and native precision replay are executed checks against capacity or arithmetic explanations for the ordering.

Neither fit met its local-convergence criterion, so missing optimization remains a limitation. However, simply preferring the lower global coefficient error or the larger model would select the worse representation for this circuit. The next compression family should directly address the composed interaction graph and its shared inputs, while retaining weight-only discovery and freezing candidates before validation. Avoid interpreting another global output fit as automatically discovering the relevant circuit.

This is still a conditional readout test: native background, MLP, normalization and upstream generators are supplied. Keeping native U on other routes removes any inference of whole-model unembedding storage savings. No compressed program is adopted, and the four-property goal remains unfinished.
