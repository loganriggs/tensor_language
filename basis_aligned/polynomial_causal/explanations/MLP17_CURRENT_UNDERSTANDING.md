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
