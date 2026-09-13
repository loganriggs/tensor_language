# Final-block mixed generation: attention input versus MLP interaction

13 September2026. Existing MLP17 dossiers were checked: normalized bilinear response, contextual gating, calibration and output-reader folding were already known. This test concerns the particular crossfirst child/remainder composition under the additive-input convention from the boundary experiment. It is not another whole-layer rank study.

## Exact split and native result

Let zN,zC,zR be the preMLP17 residual states on the original native/child/remainder trajectories and let hBar=hC+hR-hN be their additive postblock17 state. Let zBar=zC+zR-zN. Running block17 on the additive incoming block16 state produces zA before the MLP and hA after it. The affine residual re-entry contributes no new mixed term; zA-zBar is attention-generated mixed input, up to native rounding.

For g(z)=z+MLP17(RMS(z)), define

$$
G_{\mathrm{MLP}}=g(\bar z)-\bar h,
\qquad
G_{\mathrm{attn}}=g(z_A)-g(\bar z).
$$

These sum exactly to hA-hBar. The attention branch includes the direct attention write and its propagation through the MLP, not simply an attention output projection.

Physical final-state interventions compare hBar, hBar+GMLP, hBar+Gatt, and hA, retaining the actual final readout. All native/boundary anchors replay exactly and the state partition error iszero. 1280fullforwards plus160MLP17calls took13.56seconds.

| Regional group | MLP-only error versus full local effect | MLP aligned contribution | Attention aligned contribution | Effect-addition error |
|---|---:|---:|---:|---:|
| Old near | 53.4% | 48.7% | 51.3% | 0.189% |
| Near-message | 65.5% | 35.8% | 64.2% | 0.137% |
| Near-person | 83.9% | 20.1% | 79.9% | 0.182% |
| Distant | 70.8% | 29.3% | 70.7% | 0.124% |

A and C pass; B fails all four groups. The MLP's own finite mixed response is insufficient under this convention. Attention-generated mixed input and its MLP propagation are at least as important in the aligned regional outcome. Because these are projections of outcome vectors, the percentages are not independent semantic circuit fractions or variance explained.

The executed counter-review shows positive MLP/attention outcome cosines0.348–0.970. Thus the aggregate result is not dominated by opposite-sign cancellation between enormous branches. However, each branch agrees with the full effect sign on only15–21 of24prefixes depending on group/branch. Mean alignment is not a per-example explanation or a repaired universal sign rule. The close sum of physical effects remains valid.

FineWeb results differ: MLP aligned contributions are81.6–91.2%, attention8.7–18.8%. Separate-effect sum errors are1.64–8.07%, with absolute CE discrepancies1.67–4.29e-6nats. These are descriptive control results; no new threshold or universal task-specific split is claimed.

[Native result](CROSSFIRST_LAST_BLOCK_SPLIT_V1_RESULT.json), [protocol](CROSSFIRST_LAST_BLOCK_SPLIT_V1_PREREGISTRATION.md), [executed outcome audit](CROSSFIRST_LAST_ATTENTION_RESPONSE_V1_CONTROL.json).

## Following attention through the last bilinear layer

The result changes the next question: determine whether Gatt is mainly the raw attention write, its MLP cross response, or normalization's effect. Use the existing exact bilinear response algebra with a context-dependent direction rather than fit a new factorization.

Let z=zBar and v=zA-zBar. Define the bias-free bilinear function B(z)=D[(Lz) elementwise-multiplied by (Rz)], with L,R of shape4608x1152 and D of shape1152x4608. Let rho(z)=mean(z squared)+native epsilon and m0=B(z)/rho(z). Then

$$
g(z+v)-g(z)
=v+\frac{D[(Lz)\odot(Rv)+(Lv)\odot(Rz)]}{\rho(z+v)}
+\frac{D[(Lv)\odot(Rv)]}{\rho(z+v)}
+\left(\frac{\rho(z)}{\rho(z+v)}-1\right)m_0.
$$

The four terms are direct write, bilinear cross response, perturbation square, and normalization rescaling. Bias cancels. No denominator is frozen. Unlike the earlier fixed-head9writer compiler, v varies with the context, so a single fixed matrix Jv is not being claimed.

An executed actual-MLP17-weight FP64control on24random signed perturbations matches direct evaluation to3.12e-14relative error. This is algebraic implementation validation, not native-text term mediation. The native test must preserve the same additive background and report any FP32 cancellation floor. Full weights and native states remain charged.

[Control implementation](crossfirst_last_attention_response_v1.py), [receipt](CROSSFIRST_LAST_ATTENTION_RESPONSE_V1_CONTROL.json).

The next useful circuit question is which of these response terms predicts the attention branch's final effect, then which attention17 inputs/heads generate v. No new independent semantic circuit, OOD prediction or autonomous extraction has been established by this local split.
