# Frozen readout factors on historical FineWeb: improvement without preservation

13 September 2026. [Registered protocol](SHARED_LOCAL_FINEWEB_V1_PREREGISTRATION.md), [G64 result](SHARED_LOCAL_FINEWEB_V1_G64_RESULT.json). The larger-capacity fit remains live and will receive the same frozen check when its artifact is complete.

**The grouped fit improves on the matched global baseline, but fails the preservation criterion.** Replacing the entire unembedding is much worse than changing only its final bilinear contribution. This is a useful distinction between the local folded objective and a whole-readout replacement.

The 128 prediction positions come from 64 previously opened FineWeb sequences. No new model-body forwards, data fitting, factor selection or fresh/OOD evaluation occurred. Native FP32/FP64 logit replay is 4.65e-7 relative error; mean CE replay is 2.39e-7. The cached bilinear contribution agrees with its weight-derived calculation to 5.39e-7 after unembedding. These errors are far below the measured changes.

| Replacement | Program | Mean CE added | Mean absolute position CE change | Mean KL |
|---|---|---:|---:|---:|
| Bilinear output-reading route only | Grouped G64 | +0.08295 | 0.34210 | 0.11448 |
| Bilinear output-reading route only | Matched global rank78 | +0.11340 | 0.40371 | 0.15206 |
| Whole unembedding | Grouped G64 | +3.33631 | 3.63510 | 3.05028 |
| Whole unembedding | Matched global rank78 | +4.05452 | 4.35754 | 3.87660 |

CE added is damage relative to the native model; lower is better. The preservation bar requires both mean absolute per-position CE change and mean KL at most 0.05 for both grouped routes. It fails. Native instrument and grouped-versus-global KL comparison bars pass. A modest signed average does not mean individual predictions are preserved.

The quadratic-only arm adds \((\widehat U-U)b/\rho\) to native raw logits, where b is the final bias-free bilinear contribution and rho is the unchanged final residual RMS. The whole-readout arm uses \(\widehat Uh/\rho\) instead of \(Uh/\rho\). Every arm applies the separate token softcap \(30\tanh(\mathrm{raw}/30)\). These are interventions on output readers, so keeping the residual state and its norm fixed is appropriate.

The comparison itself tests a major alternative explanation for the failure: the coefficient fit did not preserve U's reading of the residual route. Restricting replacement to its actual folded objective reduces damage substantially, but still misses fidelity. The failed optimization/convergence bars and mismatch between coefficient and reachable-input metrics remain unresolved; this screen does not prove the representation cannot work.

The positive comparison also needs qualification. [Paired uncertainty audit](SHARED_LOCAL_FINEWEB_PAIRED_V1_G64_RESULT.json) resamples whole sequences, keeping their two positions together; it reports differences in KL for this fixed historical panel. This is descriptive uncertainty, not independent replication. The larger fit and further weight-only structural alternatives should be interpreted before selecting a native replacement.

For pricing, retaining native U on the residual route means the quadratic-only arm does not remove the original unembedding storage. The whole-U arm could remove it but currently causes large damage. Neither is adopted, and neither establishes selective removal or reusable semantic circuits.
