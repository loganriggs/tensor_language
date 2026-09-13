# Generating the interaction background and separating numerical errors

13 September2026. The160-prefix input-generator test passed, but its additive output background still came from native child/remainder trajectories. This screen removes that input from the predictor on eight fixed prefixes, one per existing group.

Write the residual-plus-MLP10 map as

$$
g_{10}(z)=z+D_{10}[(L_{10}\operatorname{RMS}(z))\odot(R_{10}\operatorname{RMS}(z))]+b_{10}.
$$

The generated input changes are \(\widehat c,\widehat r\). From the pristine state \(z_0\), compute

$$
\widehat h_C=g_{10}(z_0+\widehat c),\qquad
\widehat h_R=g_{10}(z_0+\widehat r),\qquad
\widehat{\bar h}=\widehat h_C+\widehat h_R-g_{10}(z_0).
$$

All these states are generated from pristine context and the scalar edit fields. The original observed background is \(\bar h=h_C+h_R-h_N\). Let \(k\) be the observed-input direct product and \(\widehat k\) its generated counterpart. With native suffix/readout \(f\), distinguish

$$
E=f(\bar h+k)-f(\bar h),\qquad
\widehat E=f(\widehat{\bar h}+\widehat k)-f(\widehat{\bar h}),
$$

$$
\delta_0=f(\widehat{\bar h})-f(\bar h),\qquad
f(\widehat{\bar h}+\widehat k)-f(\bar h+k)=\delta_0+(\widehat E-E).
$$

This identity makes baseline drift visible. A good transported effect alone does not imply good absolute output preservation.

## Eight-prefix screen

Fixed rows0,24,48,72,96,112,128,144 were selected before execution. Background relative-state error is3.06e-7–3.72e-7. Regional transported target-effect errors are0.256%,0.169%,0.105%,0%; unrelated-token errors reach2.12%. FineWeb newline-CE effect errors are32.1%,0%,68.4%,31.0%, despite absolute errors no larger than2.33e-6nats. These are descriptive pilot results, not a passed full-panel background criterion. The screen took7.58seconds on two CPU threads.

## Executed precision discriminator

The same five injected states are rounded to FP32 **first**, then passed through both FP32 and FP64 suffix arithmetic. All RMS calls in the FP64 suffix explicitly retain the native FP32 epsilon; the original weights, first values, x0 and rounded rotary tables are unchanged in value. CastedLinear converts the same weight entries to the activation dtype. This does not remove input-generation or state-injection rounding. It changes the downstream arithmetic being studied, not the model's mathematical normalization convention.

Across the same eight prefixes, FP64 transported-effect discrepancies are:

| Panel | Target relative error | Other endpoint relative error |
|---|---:|---:|
| Four regional prefixes | 0.00010–0.00305% | 0.00372–0.0701% |
| Four FineWeb prefixes | 0.0113–0.9602% | 0.00785–0.2949% |

The68.4% FineWeb case becomes0.9602%; its remaining absolute CE effect error is3.04e-9. FineWeb absolute effect discrepancies are at most9.01e-9nats in this FP64 comparison. This is evidence that suffix arithmetic accounts for a substantial part of the FP32 discrepancies on these examples, rather than merely asserting a precision floor. It does not rescind FP32 failures, certify other prefixes, or prove every residual discrepancy is roundoff. The suffix discriminator took28.96seconds.

Background drift remains distinct: in FP64, total predicted-versus-observed output discrepancies reach roughly1.69e-6 on the other FineWeb endpoint, even though transported-effect errors are much smaller. Existing pristine context, full MLP10/attention10 weights and full downstream suffix are still supplied. This is a more complete conditional interaction program, not a simple autonomous model or a new semantic circuit count.

Next: confirm generated-background preservation over the full panel with registered total-output and effect criteria, then assess original child/remainder causal interaction rather than only the local direct-product term. Pending GPU experiments remain separate.

Receipts: `COMPOSED_MLP10_BACKGROUND_V1_SCREEN.json`, `COMPOSED_MLP10_BACKGROUND_PRECISION_V1_RESULT.json`; scripts `check_composed_mlp10_background_v1.py`, `check_composed_mlp10_background_precision_v1.py`.


## Full-panel confirmation with the shared generator — 13 September, 07:12

The background construction was already implemented above; this is its first full-panel promotion with the current shared response/projection generator. The managed GPU run covers all160 historical prefixes, keeping native branch states solely as references. The candidate obtains changed attention, joint normalization and additive post-MLP10 background from pristine inputs and the frozen scalar edit fields. It still uses the full attention10/MLP10 weights and native suffix.

The registered numerical and regional criteria pass. Maximum background-state relative error is4.79e-7; maximum product error is0.0553%. Regional target/control preservation and total-output errors are:

| Group | Target-effect error | Control-effect error | Maximum total output error |
|---|---:|---:|---:|
| Regional0 | 0.262% | 0.533% | 3.34e-6 |
| Regional1 | 0.305% | 0.551% | 4.77e-6 |
| Regional2 | 0.116% | 0.772% | 2.86e-6 |
| Regional3 | 0.233% | 0.392% | 5.72e-6 |

No regional target sign with reference magnitude>=1e-5 reverses. These outcomes confirm the generated-background version of the local direct-product intervention across the existing96-prefix regional panel. They do not establish sufficiency for the original full child/remainder causal interaction, nor fresh language OOD prediction.

The FineWeb criterion **fails**: its four newline-CE-effect relative errors are12.22%,3.72%,6.23%,1.74%, against an all-groups10% bar. Total-output absolute errors remain below7.63e-6, but that does not waive the effect-magnitude criterion. Across FineWeb groups there are five CE-effect sign reversals and two control-margin reversals; none of the recorded target reversals exceeds the material1e-5 threshold. Small signs are preserved as outcomes rather than filtered out of the receipt.

### Executed error-source discriminator

The five saved readouts permit a direct distinction between input-generation error under the native background and additional effect change from moving to the generated background. Their vector errors add, not their norms. The independent baseline-drift plus effect-error identity replays exactly for every group.

In the failing FineWeb0 group, generated-product error under the native background is10.07%; additional background transport contributes14.22%. Their error-vector cosine is-0.539, so partial cancellation leaves12.22% combined error. Thus neither a background-only explanation nor adding those percentages is correct. Maximum absolute CE-effect error is2.62e-6nats. The earlier eight-prefix FP64-suffix discriminator suggests an important numerical contribution, but is not a full-panel certificate and does not rescind this native-FP32 failure.

Execution took7.84GPU seconds after loading/binding, with160pristine-prefix calculations,320native branch references and800suffix readouts. This is validation cost, not the runtime of an independently extracted model. The completed receipt replaces the earlier pending full-panel step for this implementation; no rerun is needed solely because an older handoff says it is pending.

[Full-panel receipt](COMPOSED_BACKGROUND_SHARED_V2_RESULT.json) · [Registered criteria](COMPOSED_BACKGROUND_SHARED_V2_PREREGISTRATION.md) · [Executed error-source partition](COMPOSED_BACKGROUND_SHARED_V2_ERROR_PARTITION.json).
