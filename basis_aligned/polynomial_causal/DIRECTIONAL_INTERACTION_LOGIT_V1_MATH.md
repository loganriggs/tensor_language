# Predicting a producer interaction through MLP8 and attention9

12 September 2026, 22:22 UTC. **A rank64 weight-derived response map predicts the signed final-logit interaction within 0.58–1.58% on 72 fresh longer contexts.** This extends the value-only result to the opposing routing change and the native downstream computation. It remains conditional on pristine native inputs and the model background.

## The equation and its interfaces

Let $z\in\mathbb R^{1152}$ be the pristine residual immediately before MLP8, $d\in\mathbb R^{1152}$ the fixed head8.2 writer, and $a$ its removed scalar amplitude at a token. The selective edit gives $z_-=z-ad$. Define squared RMS denominators

$$
r=\frac{\|z\|^2}{1152}+\epsilon,\qquad
r_-=\frac{\|z-ad\|^2}{1152}+\epsilon,
\qquad \epsilon=\operatorname{finfo}(\mathrm{FP32}).\mathrm{eps}.
$$

MLP8 has $L,R\in\mathbb R^{4608\times1152}$, $D\in\mathbb R^{1152\times4608}$ and output bias $b\in\mathbb R^{1152}$. Its bias-free normalized output is

$$
u_0=\frac{D[(Lz)\odot(Rz)]}{r}.
$$

The existing fixed-writer map is

$$
J_d=D\operatorname{diag}(Rd)L+D\operatorname{diag}(Ld)R
\in\mathbb R^{1152\times1152}.
$$

Expansion of both linear factors gives the exact finite edit

$$
D[(L(z-ad))\odot(R(z-ad))]
=D[(Lz)\odot(Rz)]-aJ_dz+\frac{a^2}{2}J_dd.
$$

Combining the residual edit and both normalizations therefore gives

$$
\Delta x_8=-ad+left(\frac r{r_-}-1\right)u_0
-\frac a{r_-}J_d\left(z-\frac a2d\right).
$$

The bias cancels. If $h_9$ is the pristine raw input to attention9 before RMS normalization, the model's block9 reentry coefficient $\lambda_{9,0}$ gives

$$
\widehat h_{9,-}=h_9+\lambda_{9,0}\widehat{\Delta x}_8.
$$

For compression, replace only $J_d$ by its leading64 SVD approximation $U_{64}\Sigma_{64}V_{64}^T$. The runtime stores $U_{64}\Sigma_{64}$ of shape1152×64 and $V_{64}^T$ of shape64×1152:147,456 matrix scalars. This count excludes the routing program and native interfaces. No data fits these matrices; the earlier short-prompt screen selected64 from a fixed rank ladder.

Then RMS-normalize the predicted raw state and run the frozen head9.8 scalar program, recomputing **both** joint QK routing and current-state value. The predictor receives pristine $z,u_0,h_9,a$, not the actual changed state or changed denominator. It nevertheless needs those pristine native ports; this is not a standalone token-to-logit program.

## Why both paths matter

For scalar $s=\Gamma v$, the exact symmetric decomposition is

$$
\Delta s=(\Delta\Gamma)\frac{v_-+v_0}{2}
+\frac{\Gamma_-+\Gamma_0}{2}\Delta v.
$$

The earlier selective-intervention cache showed routing opposing the value contribution in every cue family. Four MLP eigenmodes predict the value contribution well, but dropping routing causes10.74–56.69% total scalar-response error. The joint predictor preserves this cancellation.

Rank64 retains66.81% of the mixed-map coefficient energy. On the earlier short contexts its full residual-change error was20.9–34.1%, while scalar-response error was0.44–1.22%. This is readout-specific fidelity, not accurate prediction of every residual coordinate. Rank16 missed the routing criterion on nationality cues; rank64 was frozen before the longer-context test.

## Native test and results

The new panel prepends newsletter or transcription context to the three fixed cue families. It uses72 prompts, no score filtering, the same spelling endpoints, and longer positions. It is controlled context generalization, not corpus OOD.

Compare six complete native runs per prompt: pristine; remove8; remove8 and actual updated9 component; remove8 and predicted64 component; remove8 and frozen pristine9 component; remove8 and full-map predicted9 component. All edits use the frozen physical writers and all later native layers run normally. For final UK-minus-US token logit margin $m$, define

$$
I=m_{\mathrm{remove8,actual9}}-m_{\mathrm{remove8,frozen9}},
\qquad
\widehat I=m_{\mathrm{remove8,predicted9}}-m_{\mathrm{remove8,frozen9}}.
$$

This is a signed path interaction relative to a frozen mediator, not the ordinary four-arm additive ablation interaction. The full map checks the instrument independently of compression.

| Cue family | Full-map interaction error | Rank64 interaction error | Rank64 joint-removal effect error | Actual cue contrast removed |
|---|---:|---:|---:|---:|
| New cities | 0.00114% | 1.581% | 0.219% | 51.60% |
| Nationality | 0.00088% | 0.812% | 0.177% | 46.92% |
| Style rule | 0.00089% | 0.577% | 0.109% | 50.02% |

All registered A/B/C criteria pass. Coverage is descriptive here: the nationality result remains below50%; predictor fidelity does not erase that limitation. Native paired cue contrasts are positive for all36 pairs.432 body forwards took7.58seconds inside the managed job, including loading and setup counted by its timer; this is not full authoring time or a demonstrated runtime speedup.

A post-result diagnostic checks aggregation: predicted interaction signs agree on all72 contexts and all36 paired cue differences. Paired-interaction errors are1.013%,0.689%,0.500%. The largest individual absolute logit-margin error is0.00574. This uses the same data and is not independent confirmation.

## Critical interpretation and next useful work

**Critique:** pristine MLP output and raw attention9 state still borrow expensive native computation. **Counter-review:** no changed-state oracle is used, and the signed downstream interaction transfers to fresh positions with a full-map control. This is meaningful conditional prediction, but not autonomous extraction.

**Critique:** a small final error could hide a trivial interaction or cancellation across examples. **Executed check:** nonzero interaction norms are0.6246–0.6988 per family; all per-context and paired signs agree. The signed-interaction criterion is separate from the larger joint-removal effect.

**Critique:** rank64 might only work for one writer and the selected cue task. **Counter-review:** that is exactly the declared scope. A shared downstream consumer can make its response simpler than the whole residual map. Do not infer arbitrary-writer or whole-model fidelity.

Best next investment: inspect the equation's actually consumed QK/value coordinates before another rank sweep, and test whether they close a native port or generalize across a second intervention amplitude. Preserve the full map as an anchor. Do not implement generic guessed-graph tensor-sim search; the user's deferred note remains deferred.

[Native result](DIRECTIONAL_INTERACTION_LOGIT_V1_RESULT.json) · [Frozen preregistration](DIRECTIONAL_INTERACTION_LOGIT_V1_PREREGISTRATION.md) · [Aggregation diagnostic](DIRECTIONAL_INTERACTION_LOGIT_V1_DIAGNOSTIC.json) · [Earlier rank ladder](DIRECTIONAL_ROUTING_PREDICTOR_V1_RESULT.json) · [Value-only derivation](SCALAR_VALUE_GENERATOR_MLP8_V1_MATH.md).
