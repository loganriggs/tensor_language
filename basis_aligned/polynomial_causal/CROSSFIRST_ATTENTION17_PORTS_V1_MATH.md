# Native joint QK/value interaction in head17.2

13 September2026. This tests the exact finite mixed-product decomposition derived in the02:29mathreview on the existing160nativeprefixes. Both normalized/rotated QK scores and the current/first mixture value are captured from native attention. No fitting or newheldout data is used.

## Native replay and causal groups

All native five-arm, additive-background and head2-output anchors replay exactly. The jointport expansion reproduces the independently recorded mixedwrite with aggregate relativeerror5.08e-5regional and1.73e-4FineWeb. Its finalregionaleffect error is0.020–0.043%. A passes.

The12 cross-edit products alone fail the20%effecterrorcriterion inallfourregionalgroups. The seven groups containing inheritedmixedport changes are also needed. Their separately measured effects compose within0.154–0.342%relativeerror, passing C.

| Regional group | Cross-only error | Cross aligned effect | Inherited aligned effect | Effect-addition error |
|---|---:|---:|---:|---:|
| Old near | 34.8% | 65.6% | 34.5% | 0.342% |
| Near-message | 31.8% | 68.5% | 31.5% | 0.264% |
| Near-person | 32.1% | 68.1% | 31.9% | 0.179% |
| Distant | 36.8% | 63.3% | 36.7% | 0.154% |

These are projections of effectvectors, not semantic fractions. The failure is a limitation of dropping inheritedportmixedchanges, not a failure of the full productdecomposition. The executed outcome counter-review finds cross/inherited effectcosines0.975–0.996: they largely reinforce each other rather than forming an unstable cancellation. Cross-only has the reference effectsign94/96, inherited90/96, andfullformula96/96. No individualtaskdirection is asserted from matching a branch's sign.

FineWeb formulaeffect errors are0.83–5.64%, with maximumabsoluteCEerrors0.24–1.55e-6nats. Cross/inherited effectaddition errors5.4–28.8%are descriptive; the registered composition criterion concernsregionalgroups. Do not silently promote a uniform control-effect predictor. The controlmean effects remain tiny.

800forwards+800finalreadouts took9.18seconds. Nativeportcorners were cached toenable subsequent CPUanalysis without repeating the fullmodel.

[Native result](CROSSFIRST_ATTENTION17_PORTS_V1_RESULT.json), [registered criteria](CROSSFIRST_ATTENTION17_PORTS_V1_PREREGISTRATION.md), [mathreview/kernel](THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-13_0229.md).

## Cached product audit: a guide to the next split

The CPUaudit sums exactsource terms before the actualheadoutputprojection. Among the12cross monomials, terms involving a changed value plus a score from the opposite edit carry muchmore residualwrite than the two score-score terms retaining the nativevalue. Their norm is55.1%of the fullregionalmixedwrite versus2.44%for score-score-only. This is writegeometry, not finaleffect attribution.

For inheritedmixedinputs, a term containing only the secondscore's mixeddefect has52.4%of thefullregionalwrite norm; the firstscoredefect14.5%, andvalue-onlydefect2.81%. Terms involvingtwo ormore simultaneousdefects are individually below0.024%. These terms still multiply theother additivereference scores/values. This doesnot meanQK2doesone task andQK1another, or permit deleting eitherfactor.

A useful nextquestion is whether the score/valuecrossgroup plus single-score inheritedterms suffices physically, then what generates those mixedscore changes: query/key products, normalization, sourcepositions, or a knownwithinhead subspace. The source/valueproducts and inheritedscore contributions mustbe evaluated through their actualfinalreadout before making semanticclaims fromthese norms.

The currenthead isalreadyknown. No equalitywiththeolder sharedquadraticsourceblockhasbeenestablished, andthe fixednativeport/background interface isnot autonomous extraction. The full four-property goal remainsopen.

[Executed cached audit](CROSSFIRST_ATTENTION17_PORTS_V1_AUDIT.json), [reproducible scorer](crossfirst_attention17_ports_audit_v1.py).

## 02:40 — Three-contraction conditional predictor and frozen confirmation rows

The fixed approximation retains score/value cross terms and the single-score inherited defects. Exact algebra regroups these into three source contractions; equivalence to the selected groups holds within5.61e-16 on cachednativeports. The other terms remain omitted, so this is an approximation to the fullinteraction, not a new exact19-term identity.

Cached-native readout checks A/B/Cpass: exactbase/fullanchors, regional effecterrors4.67%,3.56%,2.87%,4.00%, signs96/96.480finalreadouts andzero transformerforwards took0.63seconds. This is local arithmetic/readout efficiency, not wholemodel speedup or autonomous extraction: four context-derived portcorners, outputprojection andnativebackground/readout remain required.

FineWebrelativeerrors9.3–21.8%andmaximumabsoluteCEpredictionerrors1.91–6.68e-6nats remain diagnostic. Keep the full19-term reference. The approximation was selected fromtheknownpanel geometric audit, so passingthisscreen isnotfreshconfirmation.

96newfullprefixes have now been frozen without model scoring: fourconstructions, citypairsLondon/Boston andManchester/Chicago, andthe existing sixspellingendpoints. They pass the pairedone-token-cue check and full-prefix duplicate scan againstrepositoryrowfiles. These citypairs are new tothislocalpathscreen, not globally unstudiedtokens. Nativelexical/construction confirmation hasnotrun; nocorpusOODclaim.

[Three-contraction executor](joint_attention_three_group_v1.py), [control](CROSSFIRST_THREE_GROUP_V1_CONTROL.json), [native screen](CROSSFIRST_THREE_GROUP_V1_RESULT.json), [frozen confirmation rows](CROSSFIRST_THREE_GROUP_FRESH_V1_ROWS.json).

## 02:46 — Frozen predictor generalizes to the new city and construction panel

The unchanged three-contraction predictor passed all registered confirmation checks on 96 previously unscored prefixes, with 24 old anchors replaying exactly. The validated upstream state executor generated the child and parent fields on these new inputs; no cached old fields or fitted replacement coefficients were substituted.

| New construction | Effect prediction error | Nonzero effect signs matched | Native regional pairs positive |
|---|---:|---:|---:|
| Reader reply | 5.20% | 24/24 | 12/12 |
| Grew up | 3.97% | 24/24 | 12/12 |
| Return home | 3.63% | 24/24 | 12/12 |
| Exact spelling | 3.53% | 23/23 | 12/12 |

The remaining exact-spelling reference effect was exactly zero at the measured precision; the candidate also predicted exactly zero. The registered count was at least23/24 per group and remains scored as written. Across the panel, all95 nonzero signs match. These signs concern the conditional head17.2 mixed write, not a universally positive regional-support direction.

An executed counter-review compares the candidate directly with the independent native head write rather than only the full port formula: errors remain3.53–5.20%, with the same sign results. Native capability is positive on48/48 city pairs. Thus this is useful lexical/construction generalization of a fixed conditional response, not an artifact of an inaccurate formula reference or weak native behavior.

The scope remains limited. Candidate response norms are21.9–29.0% of the original child/remainder behavioral nonadditivity and61.8–78.8% of the final block's local mixed-generation effect under the additive-state background. These ratios are not global causal coverage fractions. Its mean absolute regional-margin changes are0.000194–0.000717. It predicts a small contribution accurately; it does not explain the full regional mechanism. The original child removal changes regional contrast by+0.79%,+4.16%,+0.12% and−0.47% across the four new constructions, preserving context dependence.

600 native forwards and480 final readouts took7.54seconds. Full-formula local write errors are below0.023% in every old/new group. Four native input corners, background states and model weights are still required. No corpus-OOD or autonomous extraction claim is made, and the new panel is now diagnostic for subsequent work.

[Confirmation result](CROSSFIRST_THREE_GROUP_FRESH_V1_RESULT.json), [preregistration](CROSSFIRST_THREE_GROUP_FRESH_V1_PREREGISTRATION.md), [independent native-reference and scale audit](CROSSFIRST_THREE_GROUP_FRESH_V1_AUDIT.json).
