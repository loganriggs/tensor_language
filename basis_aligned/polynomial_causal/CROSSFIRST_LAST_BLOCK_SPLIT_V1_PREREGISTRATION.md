# Final-block local-generation split

13 September2026. Native160existingprefixes8arms,1280forwards plus160MLP17 calls; 180seconds. Frozen before scoring. Prior MLP17 dossier explicitly checked: normalized response/calibration and output folding already known, not another low-rank whole-layer screen.

N/C/P/R native trajectories define additive preMLP state zBar=zC+zR-zN and additive postblock state hBar=hC+hR-hN. Run block17 on additive block16state to get zA and hA. Let g(z)=z+MLP17(RMS(z)). Gmlp=g(zBar)-hBar, Gatt=hA-g(zBar). Gmlp+Gatt=hA-hBar. Affine residual re-entry cancels, so zA-zBar is the attention-generated mixed input (up to native rounding). Gatt includes both its direct residual write and its nonlinear propagation through MLP17.

Eight arms: native four, boundary16 reset (hA), boundary17 reset(hBar), hBar+Gmlp, hBar+Gatt. Same physical final readout. A: originalfour and bothboundary anchors replay<=1e-4 aggregate relative eachpanel; statepartition<=1e-4 relative. B: MLP-only outcome change versus hBar predicts full hA-versus-hBar change within20%relative in everyregional group. C: sum of separately measured Gmlp/Gatt effects predicts full within10%relative everyregionalgroup. Null: attention or final-readout interaction prevents MLP-local account. FineWeb descriptive with absolute errors; no extra threshold chosen after observing it.

No fitting or newdata. This isolates a conditional finite response, not a whole layer circuit or an additive attribution on the original joint-removal state. If B passes, use exact normalized bilinear mixed-term/normalizer split next; if not, trace the attention-dependent input and its propagated response. Prefix states, nativeweights andreadout remaincharged.
