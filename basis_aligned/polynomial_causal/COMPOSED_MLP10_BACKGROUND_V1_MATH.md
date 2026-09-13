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
