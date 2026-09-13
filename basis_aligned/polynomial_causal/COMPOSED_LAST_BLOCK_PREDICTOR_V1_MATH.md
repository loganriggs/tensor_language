# Composing the head interaction with the last bilinear layer

13 September2026,03:29UTC. This expands the target from a direct head17.2 mixedwrite to the full final block's conditional local interaction. It uses the same three native/child/remainder trajectories and previously scored120 regional prefixes. No new holdout or fitting.

## Computation and scope

Let z_N,z_C,z_R be preMLP17 states and h_N,h_C,h_R final states. Define

$$
\bar z=z_C+z_R-z_N,\qquad \bar h=h_C+h_R-h_N,
\qquad g(z)=z+\operatorname{MLP}_{17}(\operatorname{RMSNorm}(z)).
$$

The validated three-corner input interface reconstructs the head17.2 mixedwrite v_2 from its raw projections, residual norms, their cross-inner-product scalar and first-layer values. The candidate local mixedstate is

$$
\widehat G_{17}=g(\bar z+v_2)-\bar h
=\big[g(\bar z)-\bar h\big]+\big[g(\bar z+v_2)-g(\bar z)\big].
$$

The first bracket is the MLP local mixedresponse. The second propagates the selected attention mixedwrite through the MLP. For its tested final effect, evaluate f(g(zBar+v2))-f(hBar), where f is the native final readout. The target uses the actual last block run on additive block16 state. Otherheads are omitted fromv2; the whole native MLP17 is retained, including actual normalization andbias. This is a composed conditional circuit predictor, not another tiny MLP factorization.

## Native result

AllA/B/C criteria pass. NativeN/C/R andbackground readouts replayexactly. The run executed360fullforwards,360additionalMLPevaluations and480readouts in7.08seconds. No parent oradditive modeltrajectory wasexecuted bythecandidate; theirpreviouslyrecorded outcomes provide the scoringreference.

| Context group | Fullhead2+MLP error | Compacthead2+MLP error | MLP-only error |
|---|---:|---:|---:|
| Old near anchors | 1.05% | 2.13% | 53.4% |
| Reader reply | 1.51% | 4.03% | 58.5% |
| Grew up | 1.04% | 2.23% | 62.0% |
| Return home | 1.22% | 2.19% | 72.0% |
| Exact spelling | 1.09% | 2.25% | 74.6% |

Error is the norm of predictedminusreference effect across eachgroup, divided byreferenceeffectnorm. Retaining attention isnecessary atthisprecision; the MLP-only control doesnot explain the result. Meanabsolute fullpredictoreffects onthefournewergroups are0.000275–0.001131 scoremargin units.

The per-prefix audit prevents equating a small aggregate error with universally correct signs. Fullhead2+MLP hasone opposite sign andonezero prediction onthe96newergroupcases; compacthasthree opposite signs. The largest referenceeffect at a compact wrong sign is4.77e-6. On theold24anchors bothvariants haveonewrong sign at1.91e-6. These small signmisses remain recorded; signperfectness wasnot a registeredcriterion here.

## How much of the original interaction is now explained?

Using thefullcomposedprediction alone fororiginaltotalchild/remainder nonadditivity stillleaves61.2–69.5% relativeerror across thefournewergroups. Thecompactversion leaves62.2–68.7%. Its aligned fraction isabout32–39%. This improves on thedirecthead-only contribution but doesnot explain thewholeoriginalinteraction.

We used savednativeoutcomes toseparate the remaining question exactly. With Y_A thefinaloutcome afterrunningblock17 onadditiveblock16 input,

$$
Y_P-Y_C-Y_R+Y_N
=\underbrace{Y_P-Y_A}_{\text{effect of inherited input interaction}}
+\underbrace{Y_A-f(\bar h)}_{\text{local block17 effect}}
+\underbrace{f(\bar h)-Y_C-Y_R+Y_N}_{\text{final readout nonadditivity}}.
$$

The identity holds to1e-12 in the savedoutcome audit. On thefournewergroups, inheritedeffect alignedfractions are53.1–63.5%, localblock17 fractions31.4–38.8%, andfinalreadout fractions4.6–10.5%. Normfractions differ because vectors neednot beparallel. These are an explicit ordered conditional decomposition, not unique semantic responsibilities or fractions of variance explained.

This points the next scientific work upstream: determine which pre-final-block mixedstate computation is needed to predict the inherited term. Improving thealready1%-accurate localpredictor cannotremove that missing contribution. Fullnativeinput/state generation andtheopaque MLP17weights remain required; neither four-property completion nor broadselectivity follows fromthisscreen.

[Native result](COMPOSED_LAST_BLOCK_PREDICTOR_V1_RESULT.json) · [Registered protocol](COMPOSED_LAST_BLOCK_PREDICTOR_V1_PREREGISTRATION.md) · [Executed sign/scope/partition audit](COMPOSED_LAST_BLOCK_PREDICTOR_V1_AUDIT.json) · [Three-corner interface](ADDITIVE_HEAD_RAW_PORTS_V1_MATH.md).
