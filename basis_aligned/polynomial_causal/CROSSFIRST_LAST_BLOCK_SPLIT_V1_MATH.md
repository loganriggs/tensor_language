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


## 02:23 — Native response terms: mostly direct regional write, with a useful correction

The native response-term experiment passes its registered A/B/C criteria. Original five trajectories and additive-background/native-attention readouts replay exactly. Fullformula local response errors are0.751%regional and0.692%FineWeb relative to the small native attention-propagated mixed response. The actual-weight FP64 algebra control remains3.12e-14; native FP32 subtraction and accumulation are a separate comparison.

| Regional group | Direct-only effect error | Direct+cross | Direct+cross+normalizer | Full formula |
|---|---:|---:|---:|---:|
| Old near | 3.53% | 3.85% | 0.135% | 0.124% |
| Near-message | 4.59% | 3.21% | 0.084% | 0.087% |
| Near-person | 9.45% | 1.77% | 0.098% | 0.099% |
| Distant | 7.07% | 1.67% | 0.064% | 0.060% |

The direct attention mixed write already predicts this branch's regional effect fairly closely. Retaining its bilinear cross and exact normalization correction makes it much closer. The perturbation-square term changes little at this tested strength; do not extrapolate its smallness to arbitrary edits. Direct-only was a recorded diagnostic arm, not a newly preregistered holdout sufficiency claim.

The executed per-prefix audit finds that direct-only, direct+cross+normalizer andfullformula all retain the native attention-branch effect sign on96/96regional prefixes. Direct+cross alone missesone sign. This supports a coherent local response rather than merely matching a groupmean. The native branch's own sign neednot always support the originalregional behavior; matching that sign doesnotrepair earlier task-direction failures.

**FineWeb precision limit:** fullformula effect-relative errors are33.4%,5.40%,8.46%,5.39% across thefour groups, despite maximumabsoluteCEerrors1.43–2.32e-6nats. Meanabsolute nativeattention effects areonly2.00–5.99e-6nats. Formula signagreement is11–14/16, with someexactzero nativeeffects. Thus allregisteredA/B/Cpassing doesnot license uniformly accurate control-effect prediction: A's localstate criterion is weaker than these tinybehavioral ratios, andB/C concernregionalgroups. Preserve bothmeasurements. The absolute/per-prefix audit was executed; stronger full-FP64 native mediation or larger-strength control prediction remains untested.

800nativeforwards,160extraMLPevaluations and1280finalreadouts took9.87seconds. Finalstate interventions needno additional transformer suffix, so terms were scored byreadout only. No datafit, autonomousprefixgenerator ornewOODevidence.

[Native response-term result](CROSSFIRST_LAST_ATTENTION_TERMS_V1_RESULT.json), [registered criteria](CROSSFIRST_LAST_ATTENTION_TERMS_V1_PREREGISTRATION.md), [per-prefix/absolute-error audit](CROSSFIRST_LAST_ATTENTION_TERMS_V1_AUDIT.json).

The next structural target is the generator of the attention17 mixed write: split it across actual head/value/query-key contributions while retaining both QK factors and normalization. Existinghead dossiers and the initial first/current-value mixture needchecking. The response formula supplies a conditional downstream interpreter for thatsplit; it doesnotbyitselfgenerate theattentionwrite.


## 02:27 — Known head17.2 supplies the regional direct mixed write

The fixedpriorhead17.2 hypothesis passes alongside replay and head-effect composition. Its directeffect differs fromfullattention17 by1.25–1.62%regional. Allheadsumstateerror0.128%regional/0.417%FineWeb andsumwrite-effecterror0.043–0.122%regional. Individualheadeffects sumwithin0.56–1.03%regional. Head17.2 was already known; this doesnot establish that the samepreviouslyextracted sourceblock carries the newmixedwrite. FineWeb head2errors37.8–73.5%limit a universalheadclaim. 800forwards+1920readouts took9.20seconds.

[Native head screen](CROSSFIRST_ATTENTION17_HEADS_V1_RESULT.json). The new[mathematical review](THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-13_0229.md) gives an executable jointQK/value mixed-product decomposition, preserving cross-port interactions that independent port analyses would miss. Its synthetic exactness and falsifier controls are completed; native port testing remains next.
