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
