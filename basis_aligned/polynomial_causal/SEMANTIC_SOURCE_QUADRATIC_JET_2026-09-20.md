# Quadratic observable in semantic source amplitudes

The finite Boolean interaction graph is not evidence that the normalized model is a degree-two polynomial. On a Boolean grid, a_i squared equals a_i, and higher powers alias. This experiment independently asks whether the local quadratic Taylor observable predicts signed and strengthened interventions.

Let z(a)=h+sum_i a_i*d_i at the pre11 boundary, where i indexes middle writes4–7, MLP8 and MLP10 at one semantic noun site. Define m(a) as the native correct-answer logit margin after full blocks11–17, final RMS and softcap, with baseline residual reinjection and cached first values fixed. The damage approximation is -g^T*a - (1/2)*a^T*H*a, with g=gradient m(0), H=Hessian m(0). Native parameters are frozen; exact reverse-mode derivatives generate these quantities, not regression on intervention outcomes.

For m=g_suffix composed with F11, the Hessian contains J_F^T H_suffix J_F plus the sum over output coordinates of gradient_suffix[a] times Hessian F[a]. These are the downstream curvature and transported local curvature terms selected by the preceding boundary experiment. Normalization derivatives remain explicit in the computation; this is a local Taylor approximation, not a fixed full-model polynomial tensor.

Conditional scalar runtime price: three linear and six distinct symmetric quadratic coefficients per row/site context, three square and three cross products. The serialized audit stores all nine Hessian entries and repeats them across arms, so literal artifact size exceeds the minimal runtime price. Native source construction, full model weights, baseline context and derivative generation remain charged. It is not a standalone token circuit or a compression claim.

Opened96 rows, two sites, four amplitude arms: unit(1,1,1), negative(-1,-1,-1), mixed(1,-1,1), double(2,2,2). No fresh-text claim or repair of prior native capability failures. Cost24prefix and144suffix forward batches,16 gradient and48 Hessian-row reverse passes. Instrument: old unit effects replay1e-4 absolute, Hessian symmetry1e-4 relative, central directional gradient1% and Hessian15% at steps.1 and.2. Direction(.7,-.4,.5) is fixed before evaluation. Quadratic prediction bars10% every cell for unit and separately for all other arms; tangent baseline reported. If the derivative instrument fails, prediction results are not valid evidence against the method.

Runner: ../bilinear_quotient/ops/run_semantic_source_jet_v1.py. Result: ../bilinear_quotient/circuits/followups/semantic_source_jet_v1_result.json.

## Instrument correction and validated result

v1 is instrument-invalid: one h=.1 float32 second difference has16.48% error against the15% bar, while h=.2 has5.18%. No method failure is inferred from that receipt. A float64 functional reference retained the float32 RMS epsilon, original rounded rotary constants and original float32 source directions. Initial v2 mistakenly omitted the separately stored MLP Down_bias; native replay1.66e-3 failed the1e-4 gate. Preserve it as implementation-invalid. v2r1 includes Down_bias and asserts nongated MLPs. Its native unit-effect replay discrepancy is1.03e-5, maximum gradient finite-difference discrepancy0.0363%, maximum Hessian discrepancy0.00965%; instrument passes with unchanged bars.

| Conditional method | Values/context | Unit error | Negative error | Mixed error | Doubled error |
|---|---:|---:|---:|---:|---:|
| Linear derivative |3|52.73%|47.77%|36.92%|118.73%|
| Diagonal Hessian |6|39.33%|24.46%|70.89%|101.12%|
| Signed rank-one Hessian |7|8.41%|9.11%|7.32%|33.78%|
| Full symmetric Hessian |9|8.94%|9.10%|2.86%|37.16%|

Errors are maxima across context cells under each arm's own native effect norm. Rank-one is the largest-absolute-eigenvalue truncation, the conventional coefficient-Frobenius baseline, chosen without fitting intervention outcomes. The baseline comparison is posthoc on opened contexts. The registered unit-quadratic gate passes; the signed/strengthened conjunction fails because doubled edits exceed10%. Negative and mixed subarms pass separately, but do not turn the conjunction into a pass. Prior native capability failures remain.

## Conditional extraction and limits

[SOURCE_AMPLITUDE_QUADRATIC_V1.json](SOURCE_AMPLITUDE_QUADRATIC_V1.json) stores192 fixed text/site contexts with1728 coefficients,62,245 serialized bytes. [source_amplitude_runtime.py](source_amplitude_runtime.py) evaluates the scalar damage without torch, a checkpoint, or hidden native fallback. Independent fresh-process CPU replay over all saved arms differs by4.44e-16. This is exact replay of the local quadratic, not exact replay of native effects. [CPU audit](SOURCE_AMPLITUDE_QUADRATIC_CPU_AUDIT.json) includes all methods and cells; native coefficient production remains the full24prefix/144suffix forward plus64 reverse-pass program. This is an extracted conditional observable, not a new-context token circuit or model-wide compression.

A coefficient-only rank-one consistency audit is cautionary: median leading Hessian energy98.29%, minimum53.01%; leading input vectors have median absolute cosine.858 to a common reference, minimum.0617. Thus good per-context rank-one prediction does not identify one shared semantic input feature. No shared-feature or stable-identification claim follows. [Feature audit](SOURCE_AMPLITUDE_RANK1_FEATURE_AUDIT.json).

Next discrimination: a common low-width source feature dictionary versus context-dependent curvature directions, validated on coefficients and fresh effects separately. A local quadratic also needs an explicit amplitude-validity region; doubled-edit failure must survive into any exported interface description. The prior cross-layer chain-rule target remains: preserve both transported block11 curvature and downstream curvature.

## Later coverage correction

[Full-span audit](SHARED_SOURCE_QUADRATIC_DICTIONARY_2026-09-20.md) shows the four amplitude arms cover only2of6 quadratic directions. Existing six-setting native census validates fullquadratic4.72% but rejects per-context rank1 at15.74%. Earlier rank1 success is restricted to the listed arms, not general three-source geometry. Sharedrank2 alternatives also fail.
