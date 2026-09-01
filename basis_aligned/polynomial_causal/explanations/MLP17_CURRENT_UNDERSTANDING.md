# What we currently understand about MLP17

*Created 2026-09-01 from the late-layer duplicate-work audit.*

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
