# What we currently understand about MLP16

*Created 2026-09-01 from the late-layer duplicate-work audit.*

## Short version

MLP16 has an unusually concentrated function on natural text, but three different “low-rank” statements describe
different objects. The old whole-layer quadratic surrogate was reported as 13,832 numbers; after correcting its
constant-inclusive mathematical factor price to 14,984 and rebuilding it on clean splits, it adds +0.038978 CE and passes 27/62 current
behavior checks. A later activation-conditioned factorization found that one `Down` output direction recovers about
90% of the module's loss benefit. The new 2,065,536-number three-mode Tucker core adds +0.047796 CE and passes
22/62. Its joint structure is real, but the clean old quadratic representation strictly dominates it.

## Native object

`x in R^1152 -> Left x, Right x in R^4608 -> product in R^4608 -> Down -> y in R^1152`.

Native price: `2*(4608*1152) + 1152*4608 + 1152 = 15,926,400` stored numbers.

## Prior whole-layer quadratic replacement (§§9–10)

The old depth-followup experiment projected MLP16's output onto four directions. Each scalar coefficient was
represented by a rank-2 symmetric quadratic form, or two signed squared input projections. The whole substitute
therefore used eight squared projections plus four output directions:

`8*1152 input-direction numbers + 4*1152 output-direction numbers + 8 coefficients = 13,832`.

That old formula omitted the 1,152-number constant vector `mu_perp + bias` needed by the replacement.
The corrected mathematical factor price is therefore **14,984**. A later literal-storage audit found that the
scored hook retained dense quadratic matrices (5,314,176 values); rung391 must reproduce the same function with
the actual eight-projection tensors before 14,984 becomes a physical executable price. Rung391 did so: its saved
object contains exactly 14,984 tensor values, no dense form, and reproduces the dense function and signed gate to
numerical precision. It cost 4.2% of the measured damage of deleting the layer;
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

## Reconciliation and completed adoption decision

Rung389 completed the clean-split current-harness rebuild. The R4/k2 quadratic function, whose target factored
price is 14,984 but whose scored hook was later found to store 5,314,176 dense values, achieved:

- local function R² .82368 heldout / .82178 fresh;
- census +.038978 and 27/62 behavior checks;
- shifted WikiText mean/p95/max .039663/.071978/.094419;
- untouched FineWeb mean/max .040634/.066339;
- damage-ray cosine .97722 and vector R² .81476;
- versus a matched random-output arm at R² −.0062, census +.148198, and 5/62.

It beats rung388's Tucker core on every common modern quality measure. Its 0.725%-of-Tucker price claim was
conditional on rung391 physically reproducing the function from eight projection vectors; rung391 passed and
restored that price claim.

Rungs392–393 then installed the physical factor program jointly with structural QK64 and split-B MLP0/4 p768.
The 495,847,230-scalar composite scored +.052854 and 17/62 checks, only 1.03185 times its additive CE prediction;
its 62-effect vector matched that prediction at cosine .999966. The final original-native attention-16 knockout
gate passed at signed cosine/error/norm .965712/.317149/1.147792, collateral Spearman .989592, and a16-owned
median ratio 1.135428. Thus the object is adopted as a lower-fidelity predictive, composable, and manipulable
tier, not as the high-fidelity frontier. Therefore:

- do not call the Tucker result a compression frontier;
- do not tune Tucker ranks at layer 16;
- do not repeat the L16 Tucker or quadratic rank/site sweep;
- quote the exact physical four-tensor object rather than the old intended/dense hook; and
- treat 17/62 as a deliberately lower-fidelity tier, not a replacement for the 43/50/62-check programs.

Primary new artifacts: `mlp16_rank2_quadratic_current_gate_results.json`,
`mixed64_mlp04_mlp16_factored_composition_results.json`,
`a16_transfer_mixed64_mlp04_mlp16_factored_results.json`, ledger §§2486–2492.
