# Task14 grouped MLP6–7 quadratic circuit equation

## Status and scope

This is an executable local tensor-program description of one causally supported path for subject–verb agreement. It predicts manipulations of the grouped MLP6–7 source at the subject position through MLP8 and attention-11 head-3 on the licensed Task14 HOLDOUT rows. It is not a claim that MLP6–7 is a unique grammatical-number variable: the registered same-number lexical control fails at this boundary.

The binding causal receipts are:

- `task14_head11_3_fresh_matched_subject_mlp8_mlp6_7_source_factorial_v1_result.json`, SHA `eff2b9e7…`: MLP6 and MLP7 must remain grouped; their child split is distributed, interacting, direction-dependent, and not lexically selective.
- `task14_mlp6_7_contextual_midpoint_tangent_readout_v1_result.json`, SHA `48c72ea0…`: the midpoint directional derivative predicts the complete finite grouped-source effect at the L11H3 head interface.
- `task14_mlp6_7_quadratic_gain_manipulation_v1_result.json`, SHA `5285b484…`: coefficients frozen at the endpoints predict unseen interpolation and extrapolation gains after native-tail installation.

## Read, compute, write, and use

| Stage | Operational tensor |
|---|---|
| Read | Subject-position pre-MLP8 residual `x0` and grouped MLP6–7 donor displacement `d=x1-x0`, with recipient `E/A/U/W` or donor-context `E/A/U/W` held explicitly. |
| Compute | Native MLP8 RMS normalization and bilinear MLP, native scalar propagation through blocks 9–11, then native L11H3 current-value normalization, value projection, recipient attention score, and frozen non-subject complement. |
| Write | A 1152-dimensional L11H3 pre-output-projection head vector at the final prediction position. |
| Use | Install that vector into the unchanged native downstream tail and measure the `" are"-minus-" is"` logit margin and cross-entropy. |

Define the exact context-preserving map

\[
f(x)=C_{\neg s}+p_s\,P_{11,3}\!\left(
  V_{11,3}\operatorname{RMSNorm}\!\left(
    H_{\mathrm{fixed}}+\Lambda_{9:11}
    \left[D_8\big((L_8 n)\odot(R_8 n)\big)+b_8\right]
  \right), u_{s,0}
\right),\qquad n=\operatorname{RMSNorm}(x).
\]

Here `C_{¬s}` is the frozen sum of all non-subject L11H3 source terms, `p_s` is the recipient subject attention weight, `P_11,3` denotes the native current-plus-cached value projection, `H_fixed` is the explicitly chosen non-MLP8 high-state background, and `Λ_9:11` is the product of the native residual propagation scalars. This notation names the actual executed weights and fixed interfaces; it does not substitute an activation reconstruction.

For a manipulation gain `t`, freeze

\[
b=J_f(x_0)d,\qquad
c=J_f(x_0+\tfrac12d)d-J_f(x_0)d,
\]

and execute the quadratic circuit law

\[
\boxed{\widehat f(t)=f(x_0)+t b+t^2 c.}
\]

No coefficient is fit to an evaluated gain. For a degree-two path this expression is exact; its residual measures the higher-order contribution from normalization and later nonlinear value formation.

## Prospective tests already passed

On the complete registered opposite-number lattice across both grammatical directions, both templates, and both `E/A/U/W` contexts:

- at the original finite endpoint `t=1`, midpoint head-delta cosine is at least `0.999927`, relative error at most `0.013932`, and installed task-margin recovery lies in `[0.959081, 1.046326]`;
- at unseen `t∈{-0.5,0.5,1.5}`, head-delta cosine is at least `0.998896`, relative error at most `0.047229`, and installed task-margin recovery lies in `[0.959933,1.035275]`;
- both extrapolation gains, `-0.5` and `1.5`, pass the same registered head and task gates;
- endpoint-local linearization fails, sometimes reversing the task-effect sign, so the quadratic term is operationally necessary; and
- switching between recipient and donor `E/A/U/W` backgrounds changes midpoint relative error by at most `0.011475`, below the registered contextual-change bar.

## Honest boundary and remaining work

The same-number lexical ratio is above the `0.25` selectivity bar in both the child factorial and the quadratic manipulation. The equation therefore predicts transport along a causally supported operational source direction; it does not establish that the direction encodes only number. The alternative lexical-donor replication is instrument-invalid and cannot repair this conclusion.

This card supplies a predictive and manipulable circuit equation, and it exposes a stable fixed-background interface. It does not yet establish a globally smaller program: `b` and `c` are contextual JVP vectors evaluated from native weights, and literal storage/compute savings have not been priced. The next program-level discriminator should test whether these coefficients can be shared or generated across independent text without losing causal task prediction. Rank or reconstruction alone would not answer that question.
