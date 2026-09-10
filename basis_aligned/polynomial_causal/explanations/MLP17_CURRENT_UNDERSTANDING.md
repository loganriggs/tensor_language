# What we currently understand about MLP17

*Created 2026-09-01; updated 2026-09-10 after the component-dossier audit.*

**Calibration and quadratic output readers were already known.** The September 10 work extends their intervention tests; it does not discover the calibration direction or quadratic folding for the first time. Read the additions below before another MLP17 experiment.

## Short version

MLP17's output is unusually concentrated and contains a causally supported frequency-calibration direction, but
“MLP17 is two or four quadratic functions” is only an approximation. Four high-variance output directions recover
about 75% of its loss benefit; low-variance directions carry disproportionate remaining loss. An activation-
conditioned rank-4 factorization of `Down` recovers about 83%. These results make a generic new L17 Tucker screen
duplicative unless it targets the upstream product computation and beats these existing executable baselines at
equal price.

## Native object

`x in R^1152 -> Left x, Right x in R^4608 -> product in R^4608 -> Down -> y in R^1152`.

Native price: 15,926,400 stored numbers.

## Rank-2 quadratic-form replacement (§§7–10)

The original late-layer study projected the output onto four principal directions and represented each scalar
coefficient by a rank-2 symmetric quadratic form. That is four scalar quadratic functions and eight signed squared
input projections. The old report counted 13,832 numbers; including its required 1,152-number constant output
vector gives a corrected literal price of 14,984.

For MLP17, measured from the untouched model, this replacement cost 9.5% of the damage of deleting the layer,
about +0.102 nats in that evaluation. Of those 9.5 percentage points, 8.8 came from restricting the output to four
directions and only 0.7 from truncating each quadratic form to rank two. Thus the quadratic coefficients are very
low rank once the four output directions are chosen, but the discarded output-direction tail matters.

Primary artifacts: `bilin18_layer17_results.json`, `bilin18_layer17_readout.json`,
`bilin18_layer17_verify.json`, `bilin18_depth_followup_results.json`, ledger §§7–10.

## Functional output-direction rank (§§660–661)

Replacing MLP17's output by its top output-variance directions recovered these fractions of the module's loss
benefit: rank 1 = 33%, rank 2 = 58%, rank 3 = 69%, rank 4 = 75%, and rank 8 = 78%. The top eight directions carry
95% of output variance but only 78% of loss benefit. The low-variance tail is therefore functionally important,
especially for rare-target prediction.

Primary artifact: `mlp17_functional_rank_results.json`; ledger §§660–661.

## Activation-conditioned Down-map rank and causal interpretation (§§694, 696, 731)

An activation-conditioned SVD of `Down`, fitted on real 4,608-dimensional product activations and tested by live
CE, found rank 1 recovered 55%, rank 4 recovered 83%, and rank 8 recovered 88% of the module's loss benefit. A
random subspace recovered approximately zero. The leading rank-4 direction aligned strongly with an independently
identified frequency-calibration direction and survived a causal ablation test. The finer names assigned to the
other three directions did not survive equally well; they should be described only as weaker open-vocabulary
directions.

This simplifies `Down` on observed data. It does not show that the 4,608 upstream product values can be generated
by four products.

Primary artifacts: `rspd_mlp17_functional_rank_results.json`; ledger §§694, 696, and 731.

## Current mode spectrum (§2482)

The newer gauge-invariant screen found product-mode retained energy 0.973 at width 2,304 and output-mode retained
energy 0.997 at width 512. This says a joint core is plausible, not that it will beat the prior quadratic-form or
Down-map replacements. No current-harness L17 Tucker physical result exists.

## Next legal question

Do not run “L17 is low rank” again. A new L17 experiment must state exactly how it differs from:

- the four-output-direction/rank-2 whole-layer surrogate;
- the output-variance functional-rank sweep;
- the activation-conditioned rank-4 `Down` replacement; and
- the current invariant product/output spectra.

The highest-information late-layer action is first to port the much stronger old L16 corrected-14,984-number surrogate to
the current harness. Its outcome will tell us whether the old quadratic-form representation is a viable modern
baseline or whether its apparent advantage was evaluation-specific.

## Current-harness resolution — rung 415

That prerequisite is now complete. L16 generalized, but the same clean-split physical object at L17 did not.
The historical overlapping-row L17 CE reproduces exactly at3.55755478, and the executable factorization is exact:
four output directions, eight signed squared projections, one constant,14,984 retained float32 values,no dense forms,
and dense-to-factor error about2.5e-8.

Fit only on FineWeb rows24:48, however, the clean L17 program has heldout/fresh local R2 of -29.685/-32.076,
census damage+.303140 with0/62 certificates, WikiText mean damage+.3920, and FineWeb-fresh mean+.3145. A same-price
random-output control is still worse in census (+.6013) but has local R2 near-.05, showing that the selected rank2
forms are specifically distribution-fragile. The strong null fired.

Therefore the overlap result is not evidence that MLP17 is a generalizing four-quadratic-function compiler. Close
this whole-layer R4k2 family without rank or output tuning. The older activation-conditioned Down-map rank and its
frequency-calibration causal direction remain valid narrower facts; they do not license a whole-layer replacement.

## Existing normalized response and context-gate analysis

The v181–v186 series studied MLP17 as a response to answer-axis perturbations, including both/neither correlative tasks. In [v184](../../bilinear_quotient/circuits/followups/unit_tier5_mlp17_gain_terms_v184_result.json), its normalized bilinear response was already expanded into a rescaled base product, a base-by-perturbation cross product, and a perturbation-square term. The cross-damping and small-rescaling predictions passed; the registered quadratic-magnitude and symmetric-half predictions failed. In the two both/neither cells, the cross term was −1354.1/−1720.5 along the pushed axis, versus total counter-writes −1066.7/−2285.1. These are that experiment's residual-vector projection units, not CE or recovered fractions.

[v185](../../bilinear_quotient/circuits/followups/unit_tier5_mlp17_gate_anatomy_v185_result.json) traced the gate to context and earlier writers; all five registered predictions passed. MLP9 ranked first in both both/neither cells, but removing it changed the measured counter-write by only 7.2%/−2.6%, illustrating the distinction between direct algebraic contribution and live causal effect. [v186](../../bilinear_quotient/circuits/followups/unit_tier5_mlp17_self_saturation_v186_result.json) retained its instrument, slot-gate and self-saturation predictions while failing the causal-single-token and self-term-overlap predictions. Do not turn the positive diagnostics into a complete token circuit.

Source caveat: v184's analytical helper uses epsilon 1e−6 and a 5% on-axis bridge; the current native float32 RMSNorm default is approximately 1.19e−7. Reuse the decomposition, not an unqualified claim of exact numerical interchange with the current stricter harness.

Other prior components are distinct: [subword isolation](../../bilinear_quotient/mlp17_subword_isolation_results.json) found no measurable loss of subword probability under its rank-one removal, while rank-eight removal exceeded whole-layer mean-removal damage. [The MLP16-to-MLP17 channel quadratic](../../bilinear_quotient/mlp17_channel_results.json) added only .0027 loss-recovery fraction over the linear baseline and failed all three predictions. Thus “fold farther back” must specify a new reader or causal question, rather than repeat a generic channel-quadratic fit.

## September 10 extension: an existing calibration component with two explicit uses

[The two-consumer result](../CALIBRATION_TWO_READERS_V2_RESULT.json) folds one FIT-selected calibration scalar through MLP17 weights and tests its unembedding-numerator and final-RMS-denominator uses separately. Held-out rare-token CE damage under full removal was +.500/+ .573 nats on FineWeb/Pile; frequent-token changes were −.287/−.213. Instrument and held-out calibration passed; neither isolated consumer met the 10% full-vocabulary-effect error limit. This is a more explicit conditional producer/readout test of a previously known component.

[Independent-fit stability](../CALIBRATION_STABILITY_CONTEXT_V1_RESULT.json) failed its operational-stability predicate; [a common reference](../CALIBRATION_COMMON_REFERENCE_V1_RESULT.json) did not repair it. [Token/context simplifications](../CALIBRATION_TOKEN_CONTEXT_V1_RESULT.json) and [reader-energy replacement](../CALIBRATION_READER_ENERGY_V1_RESULT.json) also failed their sufficiency tests. These nulls prevent claiming an independently extracted scalar algorithm. All native parameters remain required. See the [six supporting calibration notes](2026-09-10/README.md).

The fitted September 10 axis must not be declared identical to an older axis without comparing the actual saved tensors. “Frequency-correlated” does not mean a literal log-frequency bias: the old [layerwise-axis receipt](../../bilinear_quotient/layerwise_calib_axis_results.json) records .5288 for that constructed log-frequency direction diagnostic, not a general identity.

## User-directed next question: token and structured unembedding readers

Logan requested two views on September 10: individual token rows of the unembedding, and clusters/hierarchical structure of those rows, each folded backward beyond MLP17. The elementary contraction U_t Down17 and the resulting quadratic form are known algebra. New evidence must concern shared reader structure, its retained token-specific remainder, transfer into earlier computations, and live causal prediction. Preserve RMS normalization, the final softcap, intervening attention, and every remaining opaque weight. A conditional fold with attention held fixed cannot establish a live two-layer circuit without the corresponding live-attention test.

### Result: two backward unembedding views, 10 September 14:59 UTC

The [registered test](../UNEMBEDDING_BACKWARD_VIEWS_V1_PREREGISTRATION.md) has completed: instrument, capability, live prediction and signed CE prediction held; the fixed hierarchy failed. Full-vocabulary effect error of the two-MLP fold versus live attention17 is .04556/.05609/.04608 on A1/A2/C; CE-change MAE .00335/.00216/.00256 nats. The 16-leaf unembedding shared-mean predictor leaves .94366/.96177/.97643 relative error. Its token-specific remainder remains required. Explicit contractions cover 518 token readers; the compiled state is scored on all 50,304 tokens.

The native MLP16 donor swap has negative task recovery −.02162/−.02928/−.03475. Thus this predicts a small opposing contribution, not target-circuit sufficiency. All native weights plus 47,545,895 temporary folded coefficients remain charged. The largest weight cluster holds 35,758 rows; the hierarchy null is limited to this fixed coarse mean predictor. No conclusion that all structured unembedding decompositions fail. [Full explanation and receipts](2026-09-10/unembedding_token_and_hierarchy_backward_folds.md).

### Whole-token quadratic sharing screen, 10 September 15:33 UTC

[Exact input-function comparison](2026-09-10/token_readers_as_bilinear_functions.md) of the same 518 readers found zero pairs meeting full and trace-free quadratic error <=.10 with raw-reader error >.50. Median closest-partner errors were .9710/.9722. Fixed old centroid median errors were .9583/.9592 after folding, versus .9553 in U. The sole close pair before the raw-reader control was literally duplicate U rows196/50274; 50274 is outside the tokenizer vocabulary. This is a weight-only nomination null, zero native forwards, not a behavioral failure or proof against shared subterms. Exact product Gram controls held, trained FP64 bridges <=4.14e-8. No rank, cluster-count, threshold or sample rescue. Future work must distinguish shared summands with meaningful inputs/consumers from proportional whole-reader functions.

### Shared gerund contrast, 10 September 15:49 UTC

[Eight weight-pair contrasts](2026-09-10/shared_gerund_component_from_unembedding.md) define a bare/-ing direction with partial native causal transfer on sixteen distinct verbs. All64 endpoint pairs capable. Final-state scalar recovery .5864/.6215 in two frames, cross-verb delta .5842/.6200; the registered .80 sufficiency fails. MLP-only scalar .07134/.09990, full MLP .15043/.21065, versus incoming-state scalar .51528/.52152. Most scalar change is carried into MLP17, not produced there. Zero removal target mean CE+.5646/+.6860, unrelated either/not mean absolute CE .02318: registered selective-removal predicate held. Can/may swap control absCE .04429, but its removal absCE .22092 limits broader preservation. Native fold/online bridges held. No independent producer, broad OOD or model saving; all native weights retained plus1,332,865 coefficients. Preserve sufficiency failures when recording the partial live state.

The [16:06 global write-interface test](2026-09-10/gerund_scalar_writes_and_live_feedback.md) extends the fixed direction to all attention/MLP output projections. Joint swaps recover .9358/1.0167 and selective distributed zero removal holds against C, but frozen-background full-vocabulary prediction errors .6537/.6696 reject a closed scalar-only edit model. Weighted native accounting assigns84–87% of scalar cue difference to MLP outputs, not a causal contribution percentage. The final donor scalar identity holds while complementary state changes: explaining upstream production and later consumers still requires more than the output scalar. This is a global interface test, not promotion of an individual early module or an independent one-dimensional circuit.


### Fresh grammatical transfer and agreement failure, 10 September 16:31 UTC

The fixed all36-port interface transfers to new16 verbs and might/were or would/was cues: recovery .95175/.86341, paired intervals [.92557,.97813]/[.83145,.89820]. All96 native pairs capable; old R replay and instrument pass. Broader preservation/removal fail on closer he/they runs/run agreement G: swap recovery .13342 and meanabsCE .10793, zero meanabsCE .54788 (CI .51396–.58346), versus .10 limits. G is one agreement contrast over16 contexts, not16 agreement readouts; it remains a failed control. Original C-only selectivity and failed scalar prediction are preserved. No independent producer, single-module promotion, pretraining-OOD or saving. The updated gerund_scalar_writes_and_live_feedback.md explains both individual-token and shared-structure-plus-remainder folds. Canonical distributed-port revision records positive fresh transfer and failed broader selectivity separately.


### Grammatical scalar consumers, 10 September 16:42 UTC

Fixed all18 post-RMS MLP e reads are held at natural base values before the original all36 output e edits. Exact weight-folded cross/square correction, no refit. Target swap recovery falls .95175->.66036 and .86341->.57286: losses .29139/.29055 with paired intervals [.25625,.32386]/[.26203,.31871]. Base-only G zero CE falls .38682->.11766 (69.6% reduction), not directly comparable with the previous two-endpoint .54788. Native scalar prediction errors .64876/.64467 become .15287/.08449 on the altered blocked computation; joint <=.10 closure still fails. A/B/C/E held D failed. All64 native pairs capable, exact replay, folded relative bridge <=1.60e-6,28forwards448seq in1.7175seconds. Folded context readers k_v=2Q(v)e for e and runs/run have median cosine .2053 across layers; descriptive immediate-output maps, no site selection or semantic separation proof. Original v184/v185 algebra reused; new causal consumer evidence under subroutine.readout.gerund_mlp_scalar_consumers. No independent context producer, OOD or selective repair. See gerund_scalar_writes_and_live_feedback.md and GERUND_MLP_CONSUMER_V1_RESULT.json.


### Selected-reader response state, mathematical review16:49

The exact MLP17 finite-difference contraction compiles two context rows K and squared coefficients a for e and runs/run. With q=Ku, response=delta*q+delta^2*a and q_next=q+2delta*a. Trained FP64 selected-response error1.01e-13 and sequential-composition error1.59e-13; saved2306-coefficient program reload also passes without native checkpoint. This is local conditional response extraction, not absolute outputs or a text-to-answer producer. All16 synthetic equal-e/equal-K/equal-norm reflection pairs still have different full-output responses; any common prediction has minimum relative error .1087–.2416. No natural-text reachability/OOD claim. Canonical gerund_selected_response is specified for selected local operation, rejected for full-output closure; zero native forwards. Review THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-10_1649.md proves the kernel criterion and state update, compares exact reduction methods, and keeps initial contextual producers/downstream readers as the missing work. LATEST16:54 and existing gerund explanation carry this scoped result.


### Native context transfer rejects the two-reader replacement, 10 September17:04

The frozen local response program is valid on native MLP17 scalar-input edits: selected-reader errors4.57e-7/3.74e-7 and all64 endpoint pairs capable. However reference-context errors .26068/.24985 and cyclic next-verb errors .19779/.08618 fail their joint.10 criteria. Even the exact-context minimum-norm two-reader output writer leaves .99009/1.00935 centered full-vocabulary effect error and CE prediction MAE .77457/.46752. Native local CE damage+.71521/+.41998, cue recovery .09755/.07401. Managed13forwards193seq,1.33336sec; A instrument held B/C/D failed. Paired exact-writer errorCIs .98670–.99386/1.00793–1.01076. Canonical gerund_native_response_gate rejected; original local selected equation/composition remains valid but insufficient for behavioral replacement. No new gate/rank/donor choice. Next distinguish omitted token-specific outputs and RMS effects with existing calibration/two-reader dossiers as precedent. See NATIVE_RESPONSE_GATE_V1_RESULT.json and updated gerund_scalar_writes_and_live_feedback.md.


### Grammatical token numerators and RMS,10 September17:17

Existing calibration two-consumer algebra reused on the current grammatical MLP17 scalar-input response. Neither norm-only (.93533/.97952 error), numerator-only (.47626/.58483), nor old two-reader numerator with true norm (.87871/.91634) meets the joint.10 full-effect bound. Missing normalization alone therefore does not repair omitted token readers. Endpoint interaction .04486/.06194 is descriptive, not independent-circuit composition. A conditional program folds actual answer/foil numerators and shared quartic RMS into11 context coefficients; max native tokenerror8.03e-6, margin8.73e-6, sequential composition5.33e-15. Initial h/u-derived coefficients and full-vocabulary probabilities remain native-dependent. A/E held B/C/D failed. V1 publication st_size() bug preserved; V2 same science12forwards192seq in1.37302sec, V1/V2 saved states exactly equal. Paired/saved-state CPU audit actually performed with no model load; cache3,010,885bytes retained for future CPU reuse. Canonical gerund_token_norm_program specified conditional token scores, rejected isolated simplifications. See updated gerund_scalar_writes_and_live_feedback.md. Next identify upstream producers of required token/context readers and norm moments; no fitted output-rank rescue.


### Token-context source census, 10 September 17:37 UTC

Actual answer-minus-foil reader v is folded to k_perp=2Q17(v)e-2(e^T Q17(v)e)e.
Tau=k_perp·u17 supplies the context part of the local scalar-input response.
Same-frame cyclic-verb donors with the recipient reader fixed tested all36
single attention/MLP output sites. A/C held, B failed: no shared-source nominee,
maximum target transfers .3591/.3701, minimum errors .6566/.7107. All48 base
endpoints correct; all-chain/no-op/MLP17-output causal-zero controls exactly0.
117forwards1872seq,3.51036sec. No single module promoted. CPU paired intervals
for all sites and saved mean-plus-token-remainder accounting actually executed;
mean-only target gate errors .64924/.90508 on already opened rows. No independent
producer, new OOD or structural saving. Both user-directed unembedding paths
remain explicit; cluster/hierarchy remainder cannot be silently dropped.
Primary receipts: TOKEN_CONTEXT_SOURCE_V1_RESULT.json and
TOKEN_CONTEXT_SOURCE_AUDIT_V1_RESULT.json in polynomial_causal; explanation
2026-09-10/gerund_scalar_writes_and_live_feedback.md. Do not repeat singleton
whole-module localization as a new circuit discovery or select a head subset
from this null without a different registered hypothesis and dossier lookup.


### Shared first-value lexical/context test, 10 September 17:48 UTC

Known token-only first-value producer (channels/v173/v174) was tested as a
cross-layer source for current actual-token MLP17 context gates. Cyclic primed
verb at position3; ordinary token stream versus attention0-returned value cache
factorial, preserving attention0 own output. All later native consumers live.
A instrument held, B context source/C joint lexical selectivity failed.
Value-only lexical recovery .84174/.80855 with all32pairs both-endpoint capable,
but gate transfer .48551/.60670/error .55717/.56737; remaining-stream gate
magnitude .53379/.59949. G absCE .70470 (CI .40194–1.07834) fails preservation.
Seven harmed/nine improved; signedCE -.08536 does not rescue the absCE failure.
Edited foil logits were not saved; do not attribute this CE change specifically
to agreement-margin damage. CPU synthetic equal-margin/different-CE witness
and paired audit executed. Full-vocabulary interaction .36498/.32789.
Token-to-V0/noop/saved-u checks exact0;15forwards240seq,1.42524seconds. No smaller
consumer program, OOD, independent extraction or selective circuit promotion.
Primary receipts TOKEN_CONTEXT_BROADCAST_V1_RESULT.json,
TOKEN_CONTEXT_BROADCAST_AUDIT_V1_RESULT.json and
TOKEN_CONTEXT_BROADCAST_CONTROL_CE_V1.json in polynomial_causal; updated dated
gerund_scalar_writes_and_live_feedback.md. Next lexical/form readout separation,
not a head-index slice or rank rescue of the failed whole-source test.


### Lexical/form command reuse and coupling, 10 September18:03 UTC

Fixed lexical command: first-value position3 cyclic donor. Fixed form command:
all36 output e scalars from original-lemma ing cue, reused without adapters for
the cyclic lemma. Native2lemma x2cue grid, both singles in both contexts andjoint.
A instrument held; B/C/D/E failed. A1native16/16everycorner; A2Y/Z15/16, shop
and laugh contexts prefer traveling in their respective four-token comparisons.
L recovery .84174/.84124 and .80855/.78531; form drift .0462–.0976. F recovery
.95175/.95041 and .86341/.86437; lexical drift .1149–.1688 fails. Joint correct
14/16,13/16; four-token error .17660/.19757; full additive error .12298/.08285.
G agreement-margin meanabs .23328(CI .16004–.30963)fails .10 and old CE .70470
replays. This resolves the previous missing-foil limitation; no threshold rescue.
Exact four-reader Hadamard decomposition and MLP17 product fold held1.16e-14.
Actual CPU score-coordinate accounting: interaction contributes58–81% of F's
squared lexical drift, descriptive not causal mediation. Selected-score addition
error .04706/.04069 versus ACTUAL joint command, not native joint target; saved
state addition+exact norm/softcap .04604/.06495 does not fix joint choice failure.
23forwards368seq,1.55786sec. Saved7,476,061bytes reader/product coefficients and
finalstates for CPU reuse; no independent producers or structural saving.
Primary LEXICAL_FORM_INTERCHANGE_V1_RESULT.json, its AUDIT_V1 and
LEXICAL_FORM_DRIFT_COMPONENTS_V1_RESULT.json in polynomial_causal; current dated
gerund_scalar_writes_and_live_feedback.md. Next investigate explicit coupled
lexical/form operations and their producers, not independent-axis relabeling.


### 18:13 coupled-readout null

`LEXICAL_FORM_READOUT_FACTORS_V1_RESULT.json` rejects physical scalar-only, scalar+actual-norm and complement+actual-norm explanations of F lexical drift (all .20 bars fail). Full saved-state readout bridges hold. Required complementary changes must be explained upstream; this does not reopen generic MLP17 rank approximations. Details and exact cancellation limits are in the current dated `gerund_scalar_writes_and_live_feedback.md`.


### MLP17 terminal complementary sources, 18:50 UTC
V2 instrument held; MLP-only complement (.710–.833 error) and carried-only (.342–.373) both fail .20 lexical-drift sufficiency. Actual last-MLP complement clamp matches the physical readout counterfactual. V1 precision-check failure preserved, shared V1/V2 states bitwise equal. See polynomial_causal/explanations/2026-09-10/terminal_complement_sources.md and TERMINAL_COMPLEMENT_SOURCE_V2_RESULT.json. No independent producer or circuit promotion.

## Unsupervised products and existing gender readout, 10 September 19:45

The old pronoun-class eigenaxis, signed reflection response and 64-unit committee are already documented in bilinear_quotient/BILIN18_CONNECTION.md §§1583 and 1589–1591 and reflect_gender.py. The all-token joint32 fit recovers two stable pronoun-related products without labels. One reader pair spans the old axis at .961 projection length, the other .257; together they leave .9035 full-vocabulary coefficient error in zero-centered reflection. This is partial unsupervised recovery, not a new complete gender circuit. The historical mean-centered reflection is not exactly class-form invariant merely because its axis is an eigenvector: the change includes $4\lambda\mu(\mu-s)$. Measured historical results remain intact. [Current explanation and receipts](2026-09-10/unsupervised_products_and_position_corrected_qk.md).

The follow-up STABLE_JOINT32_REFLECTION_V1_RESULT.json used 127 pronoun targets on cached natural contexts: instrument held; two-product reflection sufficiency and selective removal failed. Full-vocabulary response errors .853/.856, target-CE errors .306/.326. Other-target absolute removal CE .01161/.01204 exceeds .01. Preserve partial causal relevance and the failed joint claim; no rank rescue. Math1949 supplies fixed-group unembedding support bounds, which do not excuse behavioral failures.


## Unsupervised structural campaign, 10 September 20:46

The new campaign confirms that coefficient geometry and natural-state output
geometry can rank the same partial fits differently. Two unconverged weight
fits have validation squared errors .09341 (free products) and .08208 (shared
readers) in the full-unembedding metric on 6,400 stored states. The affine
calibration baseline reaches .03156 with more parameters. These are different
metrics/panels from historical causal-response ceilings and do not overturn them.
The free-product parameterization contains severe cancellation; an exact
pair-to-block rewrite preserves its function while reducing that cancellation.
The data-weighted fit and explicit cancellation-penalty route are active research,
not new identified modules or repairs to the earlier removal failures.
[Current derivation, prices, convergence limits and primary receipts](2026-09-10/unsupervised_structure_campaign.md).


### 10 September21:20: converged penalty and larger input panel
Explicit energy-penalized128-product fit convergedlocally, capture8.699%coefficientenergy withcancellation1.219. This is a changedbias, notconvergence oftheoriginalfit. Naturalshared-reader validation.01944 beatscurrentfreeproducts/blocksbutremainsunconverged. NewPilepanel1048576tokens captured; newdatafitpending. See [appended optimization/data explanation](2026-09-10/unsupervised_structure_campaign.md#your-questions-one-million-tokens-optimization-and-cost--2114-utc) anditsprimaryreceipts. No causalpromotion orrevisionofearlierremoval/OODfailures.

### 10 September22:26: exact common-output channel
The full-U last-bilinear quadratic has an exact vocabulary-mean output summand
accounting for7.195%of coefficientenergy. Current16x16x4blockfit captures72.03%
of thischannel butonly3.707%ofcenteredremainder; stillunconverged. Leadingblock
outputaxes are80–90%uniform, notindependentsemanticwrites. CommonQ needs440
signedsquaredirections for90%energy; best128realproductcapture73.964%, attained
byexactpositive/negativeeigenpairing. PreservecommonchannelbeforeRMS/tanh, not
a removablefinalsoftmaxgauge. [Derivation and receipts](2026-09-10/unsupervised_structure_campaign.md#what-the-blocks-found-common-output-versus-token-contrasts--2226-utc).

### 10 September22:41: converged squares and sparse-core baseline
Signed256squarefit convergedlocally after17.26spolish, capture9.64522%; both
originalparameter-gauge andcanonicalstationaritypass1e-4. Moreparameters than
128products; no global/circuitclaim. Centered128input/256edge spectralbaseline
captures1.401%, fullprojectedcore6.406%; inputframeunoptimized. Exactconditional
edgechoice andStiefelframeupdate controls held. [Explanation and receipts](2026-09-10/unsupervised_structure_campaign.md#signed-squares-converged-sparse-core-basis-update-prepared--2241-utc).

### 10 September23:00: fitted identifiability and native interfaces
Sparsecenteredframefit convergedlocally at3.6695%capture. Signedsquarefactors
have numericalfullcolumnranks; two-slice recoverymatchesall256readers tocosine
>1-3e-15 for threefixedmixtures. This concerns theexactfittedtensor, notnative
identification oruniqueapproximation. Learned128inputframe has24.14%mixed
interactionenergy inthenativecenteredtensor; active92readerunion has16.56%.
Graphcomponents needexplicitnativeinterfaces beforeextraction. [Review and
receipts](../THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-10_2249.md).

### 10 September23:06: restart dependence and block gauge
Second sparse-frame fit converged at3.2859%centeredcapture versus3.6695%;
functioncosine.7955 failsreproducibility. Different-qualitylocalfits, notproof
ofmultipleequallygoodglobaloptima. Broaderblockfunction/penalty arepreserved
bywithin-blockQR/coretransport, errors2.53e-15/2.22e-16; differentblocks remain
allowedtooverlap. [Latest calculation](2026-09-10/unsupervised_structure_campaign.md#restart-disagrees-broader-block-conditioning-repaired--2306-utc).

### 10 September 23:28: block optimization and a different output-sharing hypothesis
Overlapping-block manifold fit A/C held, B failed after240seconds: full capture8.6285%, centered3.7167%, common71.98%. Still improving; no structural negative. Resumable checkpoint retained. New varimax test rotates exact centered output-rank128 factors and their token loadings together, preserving the projected tensor while testing sparse overlapping usage. Distinct from failed16-leaf whole-reader mean clustering. Controls passed; native job queued, no result yet. [Details](2026-09-10/unsupervised_structure_campaign.md#block-result-and-overlapping-token-factors--2328-utc).

### 10 September 23:32: output varimax result and red-team audit
Raw varimax converged62.24s; A/B held, C failed. Median factor participation38.18→28.62 satisfies25%reduction, top4loading retention32.39%fails50%. Median53factors needed for90%per-token projection energy; fixed outputrank128captures28.999%centered native energy. Leading4functions need461–476signed-square axes for90%; not a joint-computation lower bound. Raw solution has equal-token objective gradient0.613, so equal-row-normalized comparison queued with same tensor projection. No data, no promoted circuit. Receipts OUTPUT_VARIMAX_V1_RESULT.json and OUTPUT_VARIMAX_V1_AUDIT.json.

### 10 September 23:48: terminal attention output pullback and source constraints
Equal-token varimax converged147.82s; A/B held C failed, median27.61factors, equal-token top4=.32056. Exact centered U→MLP17→O17 pullback A held B/C failed: top128capture30.54%vs29.00%original and29.18%left-scramble; native same-head11.86%vs11.25%right-scramble. This is not the old attention-fixed MLP16 fold or separate QK experiment. Coordinate gain explains raw attention-port energy inflation. New shared-source symmetry/antisymmetry algebra has dense and low-rank controls; cross-source determinant term must remain live. Native channel mass not yet measured. [Derivation](../SHARED_SOURCE_ATTENTION_QUADRATIC_V1_MATH.md).
