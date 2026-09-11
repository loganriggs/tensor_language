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

### 11 September 00:07: full source/value tensor and routing contrast modes
Full centered antisymmetric fraction43.39%vs43.45%permuted; including common output43.01%vs43.08%. Exact routing-wedge covariance: best4capture17.22%, rank90=31. Common-output bound limits all-U4mode capture to24.59%, requires at least30for90%. Leading mode96.28%native17.2/17.3pair; weak crosspaircorrelationmax.0525, not new circuit discovery. Joint QK1×QK2 numerator and complete denominator coefficient-Gram controls passed; native QK comparison pending. [Update](2026-09-11/explanation_2026-09-11_0005.md).


## 11 September00:35: native-product coefficient selection

Exact multi-output OLS over native4608products gives full-U128capture6.86% and
1024capture31.58%, with well-conditioned exact writer solves. This is weight-only
coefficient geometry, unlike the older natural-data own-neuron ceiling. No broad
structural negative. See [result and next nonorthogonal block method](2026-09-11/explanation_2026-09-11_0035.md).


## 11 September00:54: full-U congruence block spectrum

Independent search converged with full-FP64verification; two-near-null and gap bars
failed. Leading approximate dyad responds along a relatively weak input direction,
with substantial known common-output contribution. This does not rule out approximate
or overlapping blocks. [Spectrum, response and common-channel audits](2026-09-11/explanation_2026-09-11_0054.md).


## 11 September01:10: common channel plus centered blocks

Centered congruence search converged but missed the near-null/gap screen; keeping
the exact common function separate did not rescue that criterion. The new sparse
nonorthogonal token-function dictionary is an overlapping-use hypothesis, not
another independent-block test. [Result and method](2026-09-11/explanation_2026-09-11_0110.md).

## 11 September: overlapping token functions do not yet simplify input computation

The full centered quadratic family admits an exact 1,152-dimensional function
embedding. A learned 512-function nonorthogonal sparse dictionary locally
converged in95.65s, capturing16.56%coefficientenergy withmedian6functions/token.
Frozen-support coefficient debias gives20.72% withsame315718links. The eight
strongest functions require406–470signed-square eigenmodes for90%of their own
coefficientenergy. This is distinct from the earlier activation-weighted four
output directions/rank-two approximation: different target, metric and family.
Sparse token usage is not itself a simple input program or a causal circuit.
See [explanation](2026-09-11/explanation_2026-09-11_0110.md),
[fit](../TOKEN_FUNCTION_DICTIONARY_V1_RESULT.json),
[debias](../TOKEN_DICTIONARY_DEBIAS_V1_AUDIT.json),
[input audit](../TOKEN_DICTIONARY_FUNCTIONS_V1_AUDIT.json).

## 11 September: sparse token writers need an output-interface check

Previous debiased512-function dictionary has40.22%of its centered approximation
energy outside centeredU's column space. Restoring the exactcommonfunction and
checking originalU still leaves29.63%of fullapproximationenergy outside its
column space. These are approximation-energy fractions, not behavioral or target
fractions. Projecting centeredwriters improvescenteredcapture20.72%→29.05% but
usually destroys token sparsity. Fullprojection can change the common function.
Do not call arbitrary sparse output codes a native residual/MLP replacement.
This quantifies the earlier sparse-support warning for the actual fitted tensor.
[Receipt](../SPARSE_WRITER_INTERFACE_V1_AUDIT.json),
[explanation](2026-09-11/explanation_2026-09-11_0130.md).

## 11 September: combined product/token sparsity is not yet converged

512single-product input functions with sparse token usage reached1.22%centered
capture in12sweeps without jointconvergence. Fixedsame-supportdebias1.89%; exact
unrestrictedwriter refit5.56%centered. This limits the returned functions only.
SeparatefullUlegalwriter baseline6.99%; weakerL1contrastpenalty provisionally6.07%
withmoreconcentratedloadings, butconstraintconvergenceunmet. Same-objective repair
queued; no circuit identification or native physical replacement claim.
[Source](../SPARSE_PRODUCT_DICTIONARY_V1_RESULT.json),
[writer audit](../PRODUCT_WRITER_REDTEAM_V1_RESULT.json).

## 11 September: stable shared-input component with an exact local edit interface

Weight-only one-reader factor f_a(x)=(a^T x)M_a x equalsnativef(x)-f((I-aa^T)x)
afterinputnormalizationwithout renormalization. Full/centered searches eachconverge
from4starts; centeredreaderstable,capture1.0488%ofcenteredcoefficientenergy.
Anyone-readerfactor ceiling1.1309%centered. Partner16retains55.26%,rank90=279;
inputreuseidentified numerically butpartnernotyet small. Fullcomponentselected
usingcenteredobjective has37.9%commonoutputenergy, so do notrenameoldcalibration.
Olderactivation-whitened readoutJSONs lackreader vectors; directidentityunestablished.
Crossmetricfixedreader audit showsrank16objective choosesdifferently: full-selected
readercenteredcapture.0062663 vs.0057951. Jointsmallpartnerobjective nowCPUcontrolled,
nativenotyetfit. Foursemanticcircuitpropertiesremainuntested.
[Receipt](../SHARED_INPUT_FACTOR_NATIVE_V1_RESULT.json),
[audit](../SHARED_INPUT_PARTNER_OBJECTIVE_V1_AUDIT.json),
[explanation](2026-09-11/explanation_2026-09-11_0206.md).


## 11 September 02:26 — joint compact shared-input component

Joint reader/rank-16 partner optimization converged from all four starts with
whole-function cosines above 0.999999999999. Centered coefficient capture 0.6506%
beats the prior same-price 0.6266% by 3.83%, missing its 5% target. Numerical,
convergence and stability clauses held. This is a stable local weight component,
not semantic identification. Full-U component energy is 52.59% common output;
known calibration evidence remains relevant. The compact partner retains 66.72%
of its exact reader-removal component, so an omitted remainder is required for
exact removal. Price: 38,016 numbers, 17 input projections, 16 products plus
native background. CPU native OV/source pullback replay passed at 6.5e-15 with
signed multisource routing explicit. Descriptive source angles do not identify
behavioral QK sharing. [Explanation](2026-09-11/explanation_2026-09-11_0226.md),
[result](../JOINT_SHARED_READER_RANK16_V1_RESULT.json),
[upstream audit](../SHARED_READER_UPSTREAM_PORTS_V1_AUDIT.json).


## 11 September 02:43 — joint routing/value source ports

The frozen compact MLP17 component's 17 OV source readers touch 7.06% of joint
QK1×QK2 numerator coefficient energy across nine heads and two distinct source
distances, versus 5.55% for matched random downstream frames. The registered
alignment and 17.2/17.3 bars failed. Separating current/base streams barely
changes this (7.10% versus 5.58%). This is a fixed-component numerator-space
result, not a negative about task-specific sharing.

An executed source-influence spectral calculation instead finds rank-17 source
spaces with 59.63% mean touch (45.7–70.2%), demonstrating that the choice of source
space matters. Mean inside energy is 14.49%, mixed energy 45.14%; outside readers
and full normalizers remain required. Within-head overlap across distances is
0.746–0.896, while the maximum across heads is 0.075 (mean squared principal
cosines). No new behavior or identified circuit; head17.4 enrichment was post hoc.

Primary receipts in polynomial_causal: JOINT_QK_VALUE_PORTS_V1_RESULT.json,
JOINT_QK_VALUE_STREAM_CLOSURE_V1_AUDIT.json, JOINT_QK_SOURCE_BOUND_V1_RESULT.json,
JOINT_QK_SOURCE_SPACE_COMPARISON_V1_AUDIT.json. Explanation0243 defines the metrics.


## 11 September 02:55 — exact conditional normalized source edits

For the frozen QK spectral source spaces, numerator-only source-removal accounting
has10.8–33.3% routing-effect error on formal probes. Recomputing key normalizers
is essential to the exact local formula. A compiled predictor now uses only
baseline score/norm/value ports plus source/query/norm projections; no edited
endpoint enters prediction. Six native-weight edit cases across all9heads and
both distances, including order and gauge checks, pass at6.42e-15. Price per
head/position:92 dynamic numbers,595 small-core numbers,97,920 projection-map
numbers plus native background. This is a conditional intervention tool, not a
semantic circuit or full-model simplification. The next position-shared source
influence method is implemented and CPU-controlled; native selection pending.
Receipts: NORMALIZED_QK_SOURCE_EDIT_V1_RESULT.json,
COMPILED_QK_SOURCE_EDIT_V1_RESULT.json,
JOINT_QK_POSITION_INFLUENCE_V1_CONTROL.json in polynomial_causal.


## 11 September 03:10 — source variables shared across positions

A weight-only common rank17 source space per attention17 head is highly stable
across odd/even distance splits (mean squared principal cosine>0.9998). Mean
validation numerator touch is41.64%:5.95%inside and35.69%mixed. Per-head coverage
34.9–49.7%; common/separate spectral retention77.6–92.0%. Registered all-head
retention/coverage targets failed; numerical and split-stability clauses passed.
Discovery-space bounds below45% for heads1/4/8 are scoped to discovery positions.
No claim that QK source variables form closed or semantically identified circuits.

A fixed adapter using the native common frames and raw query/key vectors passes
local normalized source-edit replay across five position pairs at4.69e-15. It
uses24,531additional stored numbers per head, plus native background, and92dynamic
prepared ports. This is not a whole-model saving. Next: validate frozen features
on FineWeb without refitting; no new data-guided discovery.
Receipts: POSITION_SHARED_QK_SOURCE_V1_RESULT.json,
POSITION_SHARED_QK_SOURCE_V1_REDTEAM.json, SHARED_POSITION_QK_EDIT_V1_RESULT.json.
Explanation0310 defines the metrics and retained interfaces.


## 11 September03:52 — mixed-interaction shared-reader family

Weight-only full-U/centered native calculation completed2.66s,A/B/Cheld. Allowing
all interactions touching rank11inputspan captures6.09%centeredcoefficientenergy,
with6.80%universalspan ceiling. Rank128capture38.56%, insideonly6.41%; at least147
readers necessaryfor50%capture. Not semantic irrelevance or arbitraryprogrambound.
Numerical/nonorthogonalbasisredteam passed. NextproducerfunctionGram tool controlled;
nativeMLP16fold comparison notrun. See explanation0352,
SHARED_INPUT_SUBSPACE_NATIVE_V1_RESULT/REDTEAM and PRODUCER_FUNCTION_OVERLAP_V1_CONTROL.


## 11 September04:06 — MLP16 producer functions behind QK/OV reads

Weight-only frozenfeaturefold AheldB/Cmissed: meanfunctionoverlap5.81%vsraw4.75%
androtatedcontrols5.33%. Noheadcos>=.95. Exactlocalbias/residual/x0/RMSreplay3.2e-15.
Fulljointkeyspace envelope raisesoptimal17featureoverlap33.63%, butmaximumcos
perhead.672–.843; noidentityat.95. The8positionroutingauditfinds QKtouch42.03%old
vs16.63%produceraligned. This is a selectiontradeoff, not semantic identification.
MLP16quadraticoutputcoefficientrank90=848; doesnotcontradictnaturalstate/CEdossier.
Newjointobjectivecontrolled,nativenotfit. Seeexplanation0406 and
MLP16_PRODUCER_OVERLAP_V1_RESULT, MLP16_PRODUCER_KEY_ENVELOPE_V1_AUDIT,
PRODUCER_COUPLED_QK_TOUCH_V1_AUDIT, COUPLED_PRODUCER_ROUTING_OBJECTIVE_V1_CONTROL.


## 11 September04:33 — coupled producer/routing selection completed

Joint weight-only fit improves producer-function overlap4.77x but retains73.58%
of prior QK touch, missing80%. All36 starts have converged continuations after
same-objective repair; original miss preserved. Fixed midpoint gives2.94x sharing
with90.95% retention on already inspected positions: post-result audit only.
Leading common MLP16 scalar functions remain dense under exact product spectra:
mean16-product capture18.20%, median256products for90%, versus259private. These
fixed-function results do not rule out simpler feature combinations or shared
multioutput programs. No behavioral identification. [Combined explanation](2026-09-11/explanation_2026-09-11_0433.md).


11September04:48: selecting combinations within the frozen17D common producer spans improves meanone-product capture3.01%to5.38%; all36starts converge, noheadpasses joint gain/sharing screen. Whole-span one-product upperbounds<=20.57%, not a limit on larger/different programs. Returning to unconverged overlapping full-U blocks; standard manifold adapter and saved-state bridge pass, native continuation pending. [Details](2026-09-11/explanation_2026-09-11_0433.md).


11September05:03 correction: custom manifold block fit had already run240s (MULTIOUTPUT_MANIFOLD_V1_RESULT), capture8.62847%, unconverged. LibraryCG continuation from that newer state gives8.62854%, still fails stationarity after a line-search stop. Same16x16x4full-U model. Correct reduced-Hessian trust-region solver prepared; native status in current runner. [Latest account](2026-09-11/explanation_2026-09-11_0503.md).

11 September 05:31: longer block TR completed 600.49 fit seconds, capture8.63438%, still unconverged. Exact within-block writer/core mixing preserves each function and reaches its penalty minimum, but saves only7.36e-10 of penalty; balancing is not the remaining repair. [Current explanation](2026-09-11/explanation_2026-09-11_0531.md), [native result](../BLOCK_TRUST_REGION_THIN_V1_RESULT.json), [gauge audit](../BLOCK_WRITER_CORE_GAUGE_V1_AUDIT.json).

11 September05:49: exact fixed-frame writer/core updates settle in4.35s with no meaningful gain; remaininggradient is mainlyinputframes. Conditional rank8 expansions show modest isolated gains in10/16blocks. [Combined explanation](2026-09-11/explanation_2026-09-11_0531.md) linksreceipts; nojointgain orcircuitidentityclaimed. Next23smallfull-quadraticframes implemented/controlled, nativepending.

11 September06:10:23x4full-quadraticframe nativefits bothlocallyconverged, capture5.9467/5.9385%, no gain overoldblocks. Commoncapture43.66/44.46%; centered3.02/2.95%. Input-supportceiling18.76%for92readers. [Latest explanation](2026-09-11/explanation_2026-09-11_0608.md) linksreceipts. Compactfull-rankmixed-radixfamily onlyhasplantedcontrols, no nativeclaim.

11 September06:34: nativefullrankstructured-map fit isrunning. Exactfull-Uobjective/parameterchain controls pass; V1stoppedonFDtruncation beforejointfit, V2reuses savedinitialization withverifiedRichardsoncheck. No structuralverdict yet. [Current update](2026-09-11/explanation_2026-09-11_0608.md#native-fit-update0634).

- 11 September06:48: full-rank structured maps completed initial chunks at1.4968/0.9388% full-U coefficient capture; both time-limited and unconverged. Same-objective saved-state continuation is live. See [initial result](../STRUCTURED_BILINEAR_NATIVE_V2_RESULT.json) and [continuation protocol](../STRUCTURED_BILINEAR_CONTINUE_V1_PREREGISTRATION.md); no circuit identification claim.

- 11 September06:54: structured-stage diagonal balancing preserves complete maps (<=2.26e-15), but 5.55/5.66x smaller squared parameter norms do not imply a fitting improvement. Actual seed0 stationarity/max-gradient reductions1.62/1.02x miss2x bars; see [gradient audit](../STRUCTURED_BALANCED_GRADIENT_V1_AUDIT.json). Native weights and live continuation untouched.

- 11 September07:05: new complete input-reader dictionary MSP test is queued; full1152dimensions, sparse nativeL/Rcoordinates and nativeDown retained. This differs from the64-reader subspace. Synthetic sparse recovery and dense held-out null controls complete; no native result. See [protocol](../FULL_READER_DICTIONARY_MSP_V1_PREREGISTRATION.md).

- 11 September07:20: full-U native product energy is broad (top10%14.76%; effective4184.65/4608). Native normalized function-Gram participation4594.11, maxpaircos.4645. These are current-product metrics, not a bound on refactorization or shared intermediate reuse; [counterexample](../PRODUCT_GRAM_NONBOUND_V1_CONTROL.json). [Energy](../NATIVE_READER_METRIC_V1_AUDIT.json), [Gram](../NATIVE_PRODUCT_GRAM_V1_AUDIT.json).

- 11 September08:08: first native complete-reader MSPstart gives54.85%train/47.31%held-out top128reader capture, belowPCAheld-out47.54%; foldedcapture29.56%. Combinedconvergencecriterionmissed. [Weight audit](../NATIVE_READER_MSP_GENERALIZATION_V1_AUDIT.json) finds fourth-momenttrainingconcentration (medianper-atomparticipation1.045), not robustsharedfeatureidentification. [Explanation](2026-09-11/explanation_2026-09-11_0808.md); secondstartandqueuedrelaxationspending.

- 11 September08:22: bothcomplete-reader MSPstartsmissheldoutgain/convergence; [exactfunctioncomparison](../READER_DICTIONARY_FUNCTION_STABILITY_V1_AUDIT.json) cosine.34613fails.9despitesimilar29.56/29.58%foldedcapture. Notstablefeatureidentificationorjustagaugechange. OvercompleteL1inputreaderdiscoveryqueued; see[currentupdate](2026-09-11/explanation_2026-09-11_0808.md#update0822).

- 11 September08:38: firstordinary-oblique reader fitconverged, heldout43.7888%, folded29.9058%; [subsetencoderdiagnostic](../NATIVE_OBLIQUE_ENCODER_V1_AUDIT.json) shows smallerLassopenaltyhelps1.886ppbutmisses2ppbar. [Bias/radial audit](../NATIVE_BIAS_RADIAL_V1_AUDIT.json) doesnotfindcancellation: cosine+.9722,biasnorm.398%ofradial. Knowntrace/sphere splitispriorwork, notactualinputstatistics. See[methodindex](../WEIGHT_ONLY_METHODS_INDEX.md) forlimits.

11September09:09: Full-U coefficient updates on128products of frozen ordinary-oblique0 improve complete folded capture29.90585%to29.94420% atunchangedbasis/support/Down/price. Reused exact symmetric-product ALS through fixed sparse spans, including downstream writer cross terms. [First sweep](../FOLDED_SUPPORT_READER_V1_AUDIT.json), [V1timeout](../FOLDED_SUPPORT_CONTINUE_V1_RESULT.json), [V2joint local convergence](../FOLDED_SUPPORT_CONTINUE_V2_RESULT.json). V2support gradients9.34e-8/8.24e-10; independentCPdelta replay2.47e-18. This settles only those fixed128-product coefficients locally, notnewfeatures, globaloptimum, extraction, selectivity orbehavior. The broader dictionary remains unvalidated.

11September09:18: Completed[obliquecomparison](../OBLIQUE_READER_DICTIONARY_V1_RESULT.json) missesheldoutreader/fulltensorimprovementbars; ordinarybothconverged, Tylerbothunconverged. ExactconditionalDownrefits[receipt](../READER_CONDITIONAL_WRITER_V1_RESULT.json) raisebestordinaryfullcapture30.7649%, vsrefittedPCA29.1721%, all8linear solvesverified. Improvements<1ppmissregisteredgainbar. Convergedordinary0/937fullfunctionsafterDownfit havecosine.367805[stabilityaudit](../REFITTED_READER_FUNCTION_STABILITY_V1_AUDIT.json); exactnativecapture/projectionidentityreplays<=4.44e-16. Bettercoefficientcapturedoesnotestablishsharedstableunits oranyfourpropertycircuit.

11September09:40: OLScompletionofLassononzeros onfrozenordinary0dictionary improvesfullfoldedcapture29.9058%to33.2108%, unchanged128terms/reader andnativeDown. [Fullreceipt](../FULL_OBLIQUE_OLS_COMPLETION_V1_RESULT.json) allbarsheld, sparseexecution2.82e-15. 7025/9216readersneededcompletion; [syntheticpermutationcontrol](../LASSO_PADDING_PERMUTATION_V1_CONTROL.json) exposespriorarbitraryzero-padding. Historicaltestreader47.61%onlyslightlyabovePCA47.54%, notfreshselectionvalidation. Thisisencodinggain, notnewfeaturesorstablecircuits.

Secondordinaryseed937OLScompletionreplicates[fullgain](../FULL_OBLIQUE_OLS_COMPLETION_S937_V1_RESULT.json),29.8454%to33.1620%, butwholefunctioncos.423463misses.9; nativeDowninboth. Encodingimprovementisreplicated, stableidentificationstillunsupported.

11September10:22: [Corrected overcomplete sparse inference](../OVERCOMPLETE_OLS_REENCODE_V1_RESULT.json) reaches53.60/53.61%full coefficient capture, same learned2304feature dictionaries/nativeDown; complete-function cosine.698<.9, parent fits unconverged, not stable units. [Frozen FineWeb radial check](../FROZEN_RADIAL_FINEWEB_V1_RESULT.json) rejects predicted probability improvement. [Cached native-state decomposition](../FINEWEB_RADIAL_CANCELLATION_V1_AUDIT.json) shows strong radial/traceless cancellation (mean U-metric cosine-.966), distinct from the earlier bias/radial noncancellation audit. Exact trace split remains valid; uniform-sphere interpretation does not transfer automatically to model states. No new fit or circuit claim.

11September10:48: [One direct folded-objective step](../FOLDED_SPARSE_DICTIONARY_STEP_V1_RESULT.json) improves both frozen overcomplete programs53.60%to55.96/55.94%, unchanged fixed supports/nativeDown; full coefficient gradient cost about1second, all numerical/descent/gain/cost bars held. Reader reconstruction also improves. The sustained fit now permits an exactly solved output matrix, with fixed supports/pairings; no native convergence or behavioral circuit result yet. See[update](2026-09-11/explanation_2026-09-11_1021.md#update1048).

- 11 September11:26: in a frozen intermediate full-folded sparse program,16sampled shared-feature removals match the executable program<=1.01e-13 but require136–167partner/output directions for90%coefficient capture. [Receipt](../FROZEN_FEATURE_REMOVAL_V1_AUDIT.json). These internal coordinates are not simple identified units. [Coordinate counterexample](../FEATURE_REMOVAL_COORDINATE_V1_CONTROL.json) shows even an exactly separable polynomial can have broad removal spectra in a mixed basis; this is not absence of native structure or an available same-price repair.

- 11 September12:22: The first full-folded sparse fit captures64.685% of coefficient energy but is unconverged. In that fitted representation, exact output reallocation with fixed readers reduces summed component energy fourfold for .087855percentage point capture loss. This establishes avoidable cancellation in the fitted program, not native circuit simplification. [Budget receipt](../OUTPUT_COMPONENT_BUDGET_V1_AUDIT.json), [methods and limitations](2026-09-11/explanation_2026-09-11_1222.md).

- 11 September12:46: Cross-start scalar function candidate: separate-output canonicalcos.964261; associated nativefunctions agree.999933, and one realproduct captures81.016%ofthatnative scalar quadratic (only.108142%ofwholecoefficientenergy). Leading native product loadings3547/4182/3093 are descriptive, not causal attribution. [Weight receipt](../SHARED_NATIVE_FUNCTION_PRODUCTS_V1_AUDIT.json). [Cached FineWeb validation](../SHARED_FUNCTION_FINEWEB_V1_AUDIT.json) rejects one/fourproduct fidelity targets;16products give9.941%relativeMSE and.97321centeredcorrelation. Inputs are historical64rowpanel, notfresh/OOD; no selective/removal/composition property established. Small spectral tail strongly affects natural outputs ([diagnosis](../SHARED_FUNCTION_TAIL_ENERGY_V1_AUDIT.json)).

- Follow-up to12:46candidate: [outputreadout](../SHARED_FUNCTION_INTERFACE_V1_AUDIT.json) is he/his/him versus she/her, with negligible common-token shift. This substantially recovers the prior pronoun products documented above: [alias audit](../SHARED_FUNCTION_PRIOR_ALIAS_V1_AUDIT.json) oldtwoinputspan capture82.16/81.17%, writer-span62.20/61.95%. Do not count it as a newgendercircuit or erase earlierreflection/selectivity failures. [Opening-parenthesis association](../SHARED_FUNCTION_PUNCTUATION_V1_AUDIT.json) holds in all32eligible cached rows,median-2.775nativeRMS; this is an observational confound, nota causal token-edit result. Saved physicalread/writeinterface defines a scalarcoordinate, notvalidatedsemanticselectivity.

- 11 September13:10: Matched-price native2645product selection wins coefficientcapture68.63%versuslearneddictionaries65.00%, but frozen128positioncachedFineWebvalidation reversesranking: nativeCEadded+.355/KL.294, dictionaries+.012/+.036 andKL.021/.022. [Weight control](../MATCHED_PRICE_NATIVE_V1_AUDIT.json), [tail validation](../MATCHED_PRICE_FINEWEB_V1_AUDIT.json). This validates small-panel functional retention byweight-discoveredrepresentations; no fresh/OOD, selective unit, full-corpus or stablefeature claim.

### 11 September13:30 — Cancellation reduction does not align the dictionaries

Both joint penalized starts retain64.685%coefficientcapture and reduce componentenergy to.613native, but miss convergence and have wholefunctioncosine.744842. No stable-unit promotion or prior pronoun-test repair. [Final receipt](../PENALIZED_PROJECTED_FIT_V1_RESULT.json), [requested report](for_logan/research_update_2026-09-11_1327.md).

### 11 September — LL1 shared-parent proposal recovers prior square 150

Group matching across two native LL1 pilots fails broadly: only 4 of 64 matches have cosine >=0.8. Yet a selected shared-parent function reaches cosine 0.996389 in the other start's component span; one component from group 18 already gives 0.996387. Its energy is 99.6705% a squared reader, so this is repeated-square recovery rather than a rich DAG hierarchy. It matches atom 150 of the earlier converged 256-square fit at input cosine 0.999108 and function cosine 0.994917. Do not count a new circuit. [Alias](../LL1_SHARED_SQUARE_ALIAS_V1_AUDIT.json), [support](../LL1_SHARED_FUNCTION_SUPPORT_V1_AUDIT.json), [math and results](../LL1_SHARED_PARENT_GRAPH_V1_MATH.md).

The old unit reader's exact native writer, D[(Lu)*(Ru)], agrees with its old fitted writer at cosine 0.999944. A physical pair of 2,304 coefficients is now saved. Its 0.16999% share of native coefficient energy and exact projection identity do not prove behavioral extraction. Token writes before normalization include strong negative quotation/apostrophe loadings; no semantic label or selective effect is validated. Existing dossier entries were searched for quotation, apostrophe, and square-150 aliases without finding a direct entry. That bounded search does not establish novelty. [Interface receipt](../STABLE_SQUARE150_NATIVE_INTERFACE_V1_AUDIT.json).

### 11 September — Frozen square150 text screen is limited

On 128 fixed historical FineWeb positions, deleting the frozen native square raises the four selected quote-token scores at every position, with an average absolute effect 83.8 times that on comma/period/semicolon/colon controls. Numerical checks pass. Square-only prediction of the full input-reader deletion effect fails: relative quote-score error 18.65% versus a 10% bar, and full residual-output error 71.5%. Reader deletion includes mixed branches, so this does not reject the square as a separate subcomputation. No true quote targets occurred among those 128 positions. [Local screen](../STABLE_SQUARE150_READER_SCREEN_V1_AUDIT.json).

The follow-up finds 32 quote targets across 15 cached rows. Compared with nearest nonquote positions from the same rows, squared-reader activation is 2.53 times higher, failing the proposed low-activation gate. Native quote-family top1 is 37.5%, below the registered 50% bar. Deleting the square improves quote-target CE by 0.159 nats on average but hurts matched nonquote CE by 0.0084. These are observed local effects, not validated calibration, OOD prediction, or causal identification of an input cue. The interface was frozen and its hash checked; no factors were fitted to text. [Coverage and gating check](../STABLE_SQUARE150_QUOTE_TARGETS_V1_AUDIT.json).

###11 September16:51 — stable whole-group directions are not square aliases
Two same-start cross-optimizer canonical directions coincide with old LL1groups8/18, but their best individual prior-square cosines are0.73742(atom190)and0.90148(atom150), below0.95. The earlier square150 alias concerned a shared-square component, not the complete rank16group18. No new circuit identity. [Audit](../LL1_CANONICAL_MODE_ALIASES_V1.json), [math and scope](../THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-11_1651.md).

###11 September17:02 — fitted shared nodes retain mixed computations
All12spectral graph parents retain two effective consumers after joint fitting. None matches an individual old256-square atom at0.95whole-function cosine. The most temporally consistent parent6reads close to atom71(inputcos0.99695), but its removal function has16.05%mixed energy and only0.57674cosine to thatatom. No semantic identity or independent-start evidence. [All-parent audit](../SHARED_READER_POSTFIT_ALIASES_V1_SPECTRAL.json), [intervention accounting](../SHARED_READER_POSTFIT_INTERFACE_V1_SPECTRAL.json).
## 11 September 17:44 — Shared-reader scope and native remainder

For the frozen spectral graph's parent1, the complete native fixed-reader
projection has two leading branches covering37.02% of its coefficient energy;
16 cover52.47%, with a broad remainder. Its old two-writer span captures33.22%.
Named graph-node removal affects two declared consumers, while15 fitted groups
have at least1% own quadratic energy touching the same input direction. These
are different intervention scopes, not15 semantic circuits. The frozen native
behavioral screen retains named-node branch removal. [Derivation and receipts](../SHARED_FACTOR_OUTPUT_MIXTURES_V1_MATH.md).
Cross-start fitted node stability also misses; no new circuit identity is added.
## 11 September 18:10 — Frozen shared-parent branch screen

Parent1's two frozen spectral-graph branches pass native execution/capability,
but the support prediction fails: own removals improve mean CE by0.0420/0.05435
on24 prefix units. Both branches have negative own-target contributions on87.5%
of these examples; direct writes dominate RMS corrections. Paired own-versus-other
intervals include zero, so two distinct inhibitory behaviors are not established.
The separate128-prefix screen is queued with unchanged factors and new registered
suppression/specificity hypotheses. Earlier calibration and quote-suppression
findings remain relevant; no novel circuit identity or OOD/sufficiency claim.
[Primary result and sign accounting](../SHARED_NODE_CANONICAL_BRANCHES_V1_MATH.md#native-behavioral-screen-completed-at-1759-utc).

### 11 September 18:26 — Frozen shared-reader branches: separate suppression screen

The 128-prefix follow-up holds signed suppression and own-token specificity, with mean absolute nearby-control CE changes about 0.005. Prospective input-gating specificity misses for both branches. This supports distinct token-facing effects while leaving contextual task specialization, OOD and sufficiency unestablished. Original support-direction screen remains failed. [Primary explanation and receipts](../SHARED_NODE_CANONICAL_BRANCHES_V1_MATH.md).


### 11 September 19:12 — Corpus-shift suppression and its limits

The original two branches pass pooled four-domain suppression/specificity and per-domain mean suppression/control bars. The relative branch1preference differs in Wikipedia (point reversal, own interval crosseszero; difference from other domains excludeszero). Prospective input-gating specificity misses again. This supports token-facing suppression across the selected corpus domains, not domain-invariant tasks or an independent circuit. [Primary evidence](../SHARED_NODE_CANONICAL_BRANCHES_V1_MATH.md). The three input ports' conditional MLP16producers have an exact fold but no16-square simplification even after arbitrary output mixing within their fixedspan; see [producer derivation](../PARENT1_MLP16_PRODUCER_V1_MATH.md).

### Parent1 branch context alignment — 11 September 19:36

The two frozen branches have more than token-facing suppression: within-family/domain amplitude interchange increases CE on both existing panels; all-donor first-order alignment passes for both, with positive covered-cell means after exact-target conditioning. Their scalar products are poorly approximated by the weight-defined spherical constant. This does not identify what the common reader or private partners represent, and does not establish standalone extraction or fresh OOD confirmation. Failed broad family-gating tests remain. [Derivation and all receipts](../SHARED_NODE_CANONICAL_BRANCHES_V1_MATH.md).

The 19:42 port decomposition limits that interpretation: independent shared-reader swaps change the product mean and fail the positive-cost test. Exact accounting explains the negative effect, but its positive centered remainder is not a separately validated circuit. Preserve the complete bilinear product and its correlated inputs until port-level interventions establish otherwise. [Details](../SHARED_NODE_CANONICAL_BRANCHES_V1_MATH.md).

The private-matched donor follow-up reduces the product mean-shift confound but still misses the positive common-reader finite-loss bar on both panels; the independent-reader screen is closed at that scope. Coherent whole-product swaps remain positive. A separate Schur compiler rewrite preserves the frozen graph and parent interfaces with25fewer variable products; it is not new behavioral evidence. [Math review](../THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-11_1951.md).

### 11 September: frozen product temporal response and native local control

A frozen weight-derived product responds to present/past cues on all 32 tested pairs, but its two-branch bank does not transfer the is/was answer change. Native MLP17-output-only donor swaps oppose that change in all four prompt/direction cell means, even with the base normalization denominator held fixed for explanatory accounting. This conditional interface result does not establish global irrelevance or repair the failed branch screen. [Primary explanation and exact CPU receipt](../SHARED_NODE_CANONICAL_BRANCHES_V1_MATH.md).

The subsequent suffix-directed screen distinguishes this from agreement/count number: frozen bank swaps transfer 3.6–4.5% of the full-model margin change, with native capability and small-control bars held but 10% recovery missed. Native MLP17-output-only swaps transfer 25–33%; the bank covers only 12.8–17.9% of that local effect. The output contrast is largely a broad suffix category, not lexical-pair-specific structure. [Primary atlas, red-team and native results](../BRANCH_TOKEN_RELATIONS_V1_MATH.md).

The full-native suffix fold now supplies a larger 48-square change predictor: its physical three-readout-span swaps transfer 18–21% of the complete model's grammatical answer change and reproduce native-span effects within 4–6% error. Control swaps remain small. General-text absolute values and count-noun ordinary replacement fail; a strict invariant-error explanation also fails. This is conditional signed-effect prediction from frozen weights, not a sufficient replacement or OOD-identified circuit. [Same primary evidence](../BRANCH_TOKEN_RELATIONS_V1_MATH.md).

That fixed program now passes a lexical/construction holdout with noun-head agreement, plural attractors, relative clauses and single/several count cues: 12–17% native-gap recovery, 4–9% effect error, small adjective/unrelated swap controls. Ordinary replacement remains failed even after a weight-derived trace correction. The leading three-product approximation has about 15% physical-write error for held-out verbs but 53–55% for nouns; it is not a joint replacement. [Holdout and red-team receipts](../BRANCH_TOKEN_RELATIONS_V1_MATH.md).

Whole-component removal now attenuates original/held-out grammatical contrasts by 13–21%, while the sixteen reused unrelated controls pass the limited collateral bar. Leading/remainder writes compose exactly; strict cross-panel task-specificity and scalar CE-additivity bars fail. Exact accounting attributes most CE nonadditivity to loss curvature, not a large additional nonlinearity in the native tail. Broader grammatical collateral tests and ordinary replacement remain unresolved. [Removal, composition and accounting](../BRANCH_TOKEN_RELATIONS_V1_MATH.md).

Broader past/progressive controls now limit the suffix component's selectivity: whole removal mean absolute CE changes 0.07078/0.05948 nats, above 0.05, despite only 4.17/3.21% contrast attenuation. Exact native readout-span removal also misses preservation; direct writing dominates, with small normalization corrections. No clean number-circuit claim. [Primary controls and accounting](../BRANCH_TOKEN_RELATIONS_V1_MATH.md#neighboring-inflections-expose-the-limit-of-selective-removal).

An exact mean-null output split separates much of the suffix component's neighboring grammatical-choice effect from lexical-pair probability effects. It requires extra writer directions, uses unchanged input computations, misses intended verb-retention and full CE-preservation bars, and has only developmental-panel validation. [Primary math and receipts](../NATIVE_RELATION_OUTPUT_SPLIT_V1_MATH.md).

Fresh output-split validation on disjoint answer lexemes confirms small neighboring binary-choice CE changes (0.00197/0.00128 nats) and branch margin composition, while full CE preservation and verb retention miss again. The frozen private program predicts its exact native-span intended effects within 4–14% margin error, all signs correct; native input/background remain required. [Fresh evidence](../NATIVE_RELATION_OUTPUT_SPLIT_V1_MATH.md#fresh-lexical-and-construction-validation).

The48reader suffix program now has an exact attention17 output/value pullback. On the frozen fresh panel, attention-only component-mediated swaps account for0.4–8.5%of the full swap effect; incoming residual dominates. Separating scaledMLP16output fromremainingresidual gives large interactions, confirmed by exactnativecomponent effects:38%verbs/19%nouns. Partial mixed-input approximation can fail despite accurate full donor/base differences. These are component-specific mediated edits, not wholemodule ablations. [Requested report with formulas and receipts](for_logan/research_update_2026-09-11_2142.md#6-newest-result-tracing-the-input-upstream).
