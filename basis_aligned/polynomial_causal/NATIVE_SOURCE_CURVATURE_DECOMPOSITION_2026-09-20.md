# Native source curvature decomposition and sparse cross-source path

## Exact folded object

For the five named pre11 source amplitudes, decompose each of four output Hessians into fifteen local terms: attention11–17, MLP11–17 and final RMS/softcap readout. Each term uses the actual source Jacobian before its operation and actual downstream adjoint after it. MLP terms come from the exact weight-contracted rational core; attention and readout terms are differentiated independently on local affine source inputs. Residual mixing is linear, changes transports, and supplies no additional local Hessian.

The native sum matches the independently generated full Hessian at maximum absolute1.25e-16 and relative1.25e-15. Full-gradient and local rational-output replays pass. This is an exact derivative identity at the declared baseline and source interface, not an exact quadratic representation of the finite normalized model. [Primary receipt](../bilinear_quotient/circuits/followups/native_source_curvature_sum_v1_result.json).

Producer price:24prefix,16fullforward,80full-source JVP,64reader reverse,112MLP-core compilations/224dense replay checks,112local attention and16local readout forward,512localgradient and2560Hessian-row reverse passes. The archived term tensors occupy2,614,985 bytes. This experiment includes controls; its runtime cannot be compared directly to a producer with different validation work. Native source/context transport remains required.

## Omission tests

Full quadratic prediction on prior unitB/nullfull/nullhalf settings remains8.94% number/2.82% modal error. MLP-only curvature fails48.42%/6.69%; adding readout curvature fails48.45%/6.71%. Attention-plus-readout alone fails27.43%/9.36%. Every predictor retains the full first-order reader. These are curvature omissions, not native module ablations.

Seven terms individually pass omission bars: MLP14, attention14, MLP15, MLP16, attention16, MLP17 and readout. Omitting all seven jointly fails: number7.22%, modal5.5568%, coefficient defect12.34%. Individual dispensability does not establish joint dispensability. [Per-term audit](SOURCE_CURVATURE_TERMS_CPU_AUDIT.json), [joint audit](SOURCE_CURVATURE_JOINT_OMISSION_CPU_AUDIT.json).

## Expanded finite coverage

Run all five single-source and ten source-pair settings against native outputs, reusing frozen derivative coefficients. Their symmetric outer-product design has rank15. The linear coefficients are independently fixed derivatives: this is not a claim that Boolean samples uniquely identify both unknown linear and diagonal-quadratic coefficients. No new fitting or fresh-text claim. Twenty-four prefix and256native suffix batches, with old known number corners replaying1.19e-5.

The full quadratic passes: max number4.7043%, modal1.3926%. The linear baseline fails52.42%/7.76%. Zeroing all cross entries between A=(embedding recurrence,early writes) and B=(middle writes,MLP8,MLP10) fails number27.02%, though modal prediction remains2.40%. Joint source contractions matter; independent group compression is not supported by this test. [Finite receipt](../bilinear_quotient/circuits/followups/five_source_full_span_v1_result.json).

## A narrower path candidate and its limits

A fixed opened-data path candidate retains all non-A/B curvature, but retains A/B cross entries only from attention11 and MLP11. It passes the fifteen-setting output bars: number6.8242%, modal1.9990%. Later first-order transports/readers remain fully native; later non-A/B curvature remains retained. This splits curvature contributions within native modules, rather than declaring whole modules dispensable. It removes six symmetric A/B pairs from each of thirteen later local terms, but the final summed Hessian still has its full coordinate shape. No end-to-end speed or storage saving is measured; an implementation must avoid those contractions before claiming producer savings. It does not sparsify the exact rational numerator or prove a fixed-polynomial model.

Interaction fidelity is weaker than total-output fidelity. Comparing -H_ij against the actual pair-minus-singles effect,281 number-interaction cells exceed1% of their pair-number-effect norm. Maximum own-interaction relative error is41.69%, while maximum interaction error normalized by pair-number effect is3.55%. Keep both denominators; do not call every interaction accurately identified. [Cross-source audit](SOURCE_CROSS_CURVATURE_CPU_AUDIT.json).

The early A/B pattern is now a candidate to freeze for prospective construction/vocabulary tests and selective manipulation checks. It has only opened-data evidence. Native capability failures, five-source strength/selectivity tradeoffs, source-generator dependencies, and doubled-amplitude failure all remain. Full goal unachieved.
