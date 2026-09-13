# Generating the attention partner from an upstream residual change

13 September 2026. This extends the existing raw-projection attention algebra to all heads and query positions. It is a dependency-removal step for the MLP9–attention10–MLP10 interaction, not a new semantic head identification.

Let \(r_t\in\mathbb R^{1152}\) be the pristine affine residual input to attention10, and \(\delta_t\) a predicted change from MLP9. For any of the five attention projection matrices \(W\),

$$
p'_t=W(r_t+\delta_t)=p_t+W\delta_t,
\qquad \rho'_t=\|r_t+\delta_t\|^2/1152+\epsilon.
$$

Each projection has nine heads of width128. In a query or key head, canceling the common input normalization gives

$$
\widehat p'_t=
\frac{p'_t}{\sqrt{\|p'_t\|^2/128+\epsilon\rho'_t}}.
$$

The epsilon-times-input-norm term must stay: normalizing raw projected vectors with a bare epsilon changes the native function. Apply the model's actual half-split rotary transform, with BF16-rounded sine/cosine tables, to these normalized queries and keys. For head \(h\),

$$
\Gamma_{hts}=\mathbf1_{s\le t}
\frac{\langle \widehat q'_{1,ht},\widehat k'_{1,hs}\rangle}{128}
\frac{\langle \widehat q'_{2,ht},\widehat k'_{2,hs}\rangle}{128}.
$$

Both QK factors participate. There is no softmax. Values and the residual output are

$$
v'_{hs}=(1-\mu)\frac{p'_{V,hs}}{\sqrt{\rho'_s}}+\mu v^{\mathrm{first}}_{hs},
\qquad A(r+\delta)=W_O\operatorname{concat}_h\sum_s\Gamma_{hts}v'_{hs}.
$$

The first-layer values are unchanged under the downstream head9 intervention. Therefore the attention partner can be predicted as \(A(r+\delta)-A(r)\), given pristine inputs, the predicted residual change and actual attention10 weights. Coupled to the pending MLP9 fold, this could replace the measured child/remainder attention inputs. It does not remove the pristine upstream generator, nor does it approximate or split attention into semantic units.

## Executed control

`check_raw_attention_response_v1.py` compares the helper against the actual checkpoint's FP32 attention10 module on two synthetic17-token sequences, all positions and heads. Perturbation strengths are0,.001,.03,.3,-.3. Linear projection replacement versus direct FP64 projection agrees within2.75e-15. Native full output relative error is at most5.80e-7. Native change relative error is largest at the smallest nonzero strength:4.25e-4, or0.043%; zero strength gives exactly zero change. The larger-strength change errors are1.45e-5 or smaller.

This is a CPU implementation control, not corpus evidence or validation of the MLP9 input generator. Small response differences are more sensitive to FP32 subtraction than full outputs, so the future native test must measure both. The joint MLP10 denominator must also be rebuilt from the generated pre-MLP states if the goal is to remove supplied intervention trajectories; retaining an observed joint denominator would leave that dependency unresolved.

All five projection matrices and the output matrix remain charged:6×1152²=7,962,624 attention weight scalars, plus the mixing scalar. No storage reduction or measured runtime improvement is claimed. The implementation exposes the full source/query/head dependence for later selective path tests rather than assuming that a native head is the circuit.

Receipt: `RAW_ATTENTION_RESPONSE_V1_CONTROL.json`. The already queued `MLP9_TO_MLP10_RESIDUAL_FOLD_V1` does **not** use this new helper; its frozen scope and results remain distinct.


## Integrated MLP9 → attention10 → MLP10 input control

`composed_mlp10_inputs_v1.execute` now combines the fixed-writer MLP9 response with the full raw-projection attention generator. It returns both pre-MLP10 input changes and computes the joint RMS denominator from the pristine pre-MLP10 state plus these predicted changes. Neither observed child/remainder attention changes nor their observed joint denominator are inputs. The pristine states, bias-free baseline MLP9 output, first values, scalar edit fields and actual weights remain supplied.

The executed `check_composed_mlp10_inputs_v1.py` compares against separate actual-weight FP32 MLP9 and attention10 branch execution on two synthetic17-token contexts and signed edits at scales.003,.03,.3. Worst relative errors: child input2.78e-5, remainder2.39e-5, joint denominator4.23e-7, and the normalized MLP10 cross-product2.59e-5. Thus the largest product discrepancy is0.0026%. These are computational controls; the backgrounds are synthetic and no downstream behavioral measurement has been made. Receipt: `COMPOSED_MLP10_INPUTS_V1_CONTROL.json`.

This integrated helper is the candidate for the next native dependency-removal test after interpreting the already queued residual-only fold. It must preserve the full cross-product effect with its own generated denominator, and separately report the residual-plus-mixed approximation. A passed control cannot establish real-text effect preservation or replace the earlier registered test.


## Real-text CPU pilot, rows0 and96

The fixed first regional prefix (15tokens) and first FineWeb prefix (78tokens) were executed through native CPU prefix blocks, the actual head9 edit ordering, MLP9 and attention10. The integrated product errors are2.18e-5 and8.36e-5 relative, respectively. Each then receives three suffix11–17 readouts at the same additive background: no product, direct observed-input product, and generated product. Regional target effects are0.000372887 direct and0.000373840 generated (0.256% relative difference); the unrelated-token margin differs by3.10e-6, or1.84%. FineWeb newline CE effects are1.669e-6 and1.550e-6 (7.14% relative difference,1.19e-7 absolute); newline-comma margin error is9.54e-7 (6.25%). All four effect directions agree. This preserves the warning that tiny state discrepancies can be larger relative behavioral discrepancies.

The complete two-prefix CPU pilot took1.64seconds with two threads and no GPU. These are existing rows and frozen GPU-derived scalar edit fields, so this is not fresh OOD or CPU/GPU equivalence evidence. It does provide the first real-text computational and local behavioral check of the integrated generator. The160-prefix GPU registrations remain separate and unscored. Receipt: `COMPOSED_MLP10_REAL_TEXT_V1_CONTROL.json`; executor: `check_composed_mlp10_real_text_v1.py`.

The measured CPU latency suggests that a small direct CPU pilot is a useful scheduling option when the managed GPU lane has tens of minutes of queued work. No competing GPU process is needed. Broad real-text prediction and scope validation are still required before adopting the integrated path.


## 04:12 — Full existing-panel CPU confirmation

The registered160-prefix CPU experiment passesA/B in105.53seconds. Generated full-product local effect relative errors across the four regional groups are0.215%,0.201%,0.124%,0.095%; all96 target signs match. Unrelated-token margin errors are0.312–0.723%, with one sign reversal whose reference is2.38e-7. Maximum per-prefix errors: child input0.2044%, remainder0.0108%, joint denominator1.22e-7 relative, and product0.0489%.

FineWeb newline-CE errors are7.58%,2.22%,6.14%,2.41%, with maximum absolute errors0.894e-6–2.384e-6nats. There are two opposite CE signs, reference3.58e-7 and-1.61e-6, plus two predicted-zero/nonzero-reference cases and one reverse case. Newline-comma margin errors2.22–5.54% include four opposite signs. The complete unfiltered discrepancy audit is saved. These small absolute values constrain the practical discrepancy but do not prove a rounding-only explanation; no uniformFineWeb sign guarantee is claimed.

This extends the two-prefix pilot on the existing panel, with no fitting or newOOD. It predicts the direct-product effect conditional on the original additive postMLP10 background and native suffix. The generator itself receives no changed-state or joint-denominator oracle; the **evaluation background still depends on native child/remainder states**. Removing that remaining dependency is the next extraction step, rather than claiming a fully extracted circuit now. The pendingGPU registrations remain separate.

[CPU panel](COMPOSED_MLP10_CPU_PANEL_V1_RESULT.json) · [Signed audit](COMPOSED_MLP10_CPU_PANEL_V1_AUDIT.json) · [Preregistration](COMPOSED_MLP10_CPU_PANEL_V1_PREREGISTRATION.md).
