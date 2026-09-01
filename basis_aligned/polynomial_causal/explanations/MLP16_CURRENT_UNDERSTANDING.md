# What we currently understand about MLP16

*Created 2026-09-01 from the late-layer duplicate-work audit.*

## Short version

MLP16 has an unusually concentrated function on natural text, but three different “low-rank” statements describe
different objects. An old whole-layer quadratic surrogate was reported as 13,832 numbers, but its literal price is
14,984 after counting the constant output vector; it measured +0.03073 CE on overlapping fit/evaluation rows. A
later activation-conditioned factorization found that one `Down` output direction recovers about 90% of the
module's loss benefit. The new 2,065,536-number three-mode Tucker core adds +0.047796 CE and passes 22/62 current
behavior checks. It confirms meaningful joint tensor structure, but does not improve the old size/damage result.

## Native object

`x in R^1152 -> Left x, Right x in R^4608 -> product in R^4608 -> Down -> y in R^1152`.

Native price: `2*(4608*1152) + 1152*4608 + 1152 = 15,926,400` stored numbers.

## Prior whole-layer quadratic replacement (§§9–10)

The old depth-followup experiment projected MLP16's output onto four directions. Each scalar coefficient was
represented by a rank-2 symmetric quadratic form, or two signed squared input projections. The whole substitute
therefore used eight squared projections plus four output directions:

`8*1152 input-direction numbers + 4*1152 output-direction numbers + 8 coefficients = 13,832`.

That old formula omitted the 1,152-number constant vector `mu_perp + bias` needed by the executable replacement.
The corrected literal price is therefore **14,984**. It cost 4.2% of the measured damage of deleting the layer;
the receipt's exact CE was `3.48647776` against `3.45574331`, or **+0.03073445**. The output PCs, whitening metric,
and CE evaluation were all derived from overlapping portions of `bilin18_eval_tokens.pt`, so this was a live-model
replacement but not a clean held-out generalization test. It also predates the current 62 behavior checks,
fresh-corpus suite, shifted-corpus suite, and exact standalone dependency audit.

Primary artifacts:

- `bilin18_depth_followup_results.json`
- `experiments/structure_mapping/bilin18_depth_followup.py`
- main ledger §§9–10

## Activation-conditioned Down-map result (§§713, 715)

With real 4,608-dimensional product activations as input, an activation-conditioned SVD of `Down` found:

- deleting MLP16 lost 0.881 nats on that evaluation;
- rank 1 recovered 90.2% of that benefit;
- the leading direction was associated with sentence-ending punctuation and common continuation writing.

This is strong evidence that the **output map on observed data** is nearly one-dimensional. It is not by itself a
one-direction implementation of the full MLP, because computing the coefficient can still require the native
`Left`, `Right`, and 4,608 products. The semantic naming also had a weak concentration null; the CE recovery is
the stronger claim.

Primary artifacts: `rspd_depth_rank_map_results.json`, `rspd_mlp16_rank1_results.json`, ledger §§713 and 715.

## Current invariant-mode and Tucker results (§§2481–2485)

The gauge-invariant context-weighted tensor spectra found concentrated late-layer modes:

- product energy retained: 0.849 at width 576 and 0.925 at width 2,304;
- output energy retained: 0.993 at width 512;
- context-measured input reduction was already known to be strong.

The `(input=512, product=576, output=512)` core computes

`R^1152 -> R^512 -> two R^576 maps -> R^576 product -> R^512 -> R^1152`

and costs 2,065,536 numbers. Its function R² was 0.81456 on held-out rows and 0.83063 on fresh rows. In the
physical model it added 0.047796 census CE, retained 22/62 behavior checks, added mean 0.05451 CE on shifted
WikiText and 0.05396 on untouched FineWeb. A same-price product-only control was catastrophic (+0.75230 CE), so
the input/product/output coupling is real. The core is nevertheless not accurate enough to adopt.

Primary artifacts: `mlp16_tucker_physical_calibration_results.json`, `ops/mlp16_tucker_physical_calibration.py`,
ledger §§2481–2485.

## Reconciliation and next legal question

The old corrected-14,984-number surrogate and new 2,065,536-number core have similar-scale aggregate CE damage on
different old/current evaluations. The correct next experiment is a **clean-split current-harness rebuild and
direct head-to-head comparison of the same mathematical surrogate family**, including all 62 behavior checks and
the fresh/shifted corpora. Until that is done:

- do not call the new Tucker result a compression frontier;
- do not tune Tucker ranks at layer 16;
- do not infer that the old surrogate will preserve current causal behaviors; and
- do not start another late-layer core merely because its tensor spectrum is concentrated.
