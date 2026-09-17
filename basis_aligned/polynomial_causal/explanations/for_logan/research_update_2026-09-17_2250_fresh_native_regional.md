# The coupled regional write transfers to fresh constructions

The city-conditioned head8.2 write, followed by its full MLP8 response and all later native computation, selectively changes UK/US spelling on fresh constructions and held-out city pairs. The new screen passes every registered gate: **18–29% attenuation**, **15–27% prediction error** against the fuller city intervention, and all sixteen matched random controls beaten in every family (**edit**, fresh). This remains a path with native-state inputs and an external suffix: simple baseline comparisons for this native application, full-suffix extraction, and the requested composition property are still incomplete or failed. The coupled local executor now passes native and isolated replay.

```mermaid
flowchart LR
 C[Supplied current8 and donor-city8 states] -->|fold: two state inputs| W[Head8.2 retained city write]
 G[Supplied raw post-attention8 state] -->|fold: one context input| M[Exact MLP8 response]
 W -->|fold: full normalization and cross terms| M
 W --> J[Coupled direct write and MLP8 response]
 M --> J
 J -->|edit: 18–29% attenuation, fresh| N[All later native computations]
 N --> S[UK/US spelling]
 classDef native fill:#eeeeee
 class C,G,N native
 linkStyle 0,1,2,3,4 stroke:#2471a3
 linkStyle 5,6 stroke:#238b45
```

The native-state boxes mark unresolved inputs. The diagram does not claim that the downstream computations are extracted or independently identified.

**Metrics.** Prediction error is relative L2 error between the retained write's effect and the fuller city intervention's effect. Attenuation is the mean proportional reduction of capable British-minus-American spelling margins. Random-null strength compares the target's root-mean-square logit change with the median of sixteen same-site, per-position norm-matched random writes. Collateral is an unrelated readout's root-mean-square change divided by the target's.

| Claim | Evidence | Evaluation | Key numbers | Status |
|---|---|---|---|---|
| Retained write predicts fuller city intervention | edit | fresh | family error15–27%; endpoint error22–24%; gate35% | passes |
| Midpoint write reduces regional differences | edit | fresh | 18–29%; all120capable endpoint pairs decrease | passes |
| Effect exceeds matched directions | edit | fresh | 5.5–12times null median;16/16beaten per family | passes |
| Four unrelated readouts move less | edit | fresh | maximum0.17; gate0.50 | passes |
| Restricted head9 route predicts native8 application | edit | opened prior panel | 89–94%error | fails, unchanged |
| Direct write or MLP8 response suffices alone | edit | opened prior panel | 33–65% /79–90%error; all-family gate35% | fails, unchanged |
| Direct and MLP8 effects add | edit | opened prior panel | up to14%joint error, interaction/smaller0.46 | fails, unchanged |

The panel uses museum labels, editorial requests, plain memos, multiline records, and reported notes, with two authored constructions per family. Edinburgh/Denver and Bristol/Atlanta were held out from the native8 scope screen; they are not globally new to the research program. The six spelling endpoints are unchanged. There are20independent construction/city-pair cells,40sequences, and240endpoint rows. No model scores were used to select these prefixes.

The fuller parent intervention donates both city-key factors, inherited value, and current value. The retained formula keeps recipient current value. Both are applied at attention8 and run through the native MLP8 and suffix. Thus this prediction result tests the retained formula against a broader intervention, not a token-only prediction of logits. Earlier constant/fitted-baseline wins for the restricted route cannot be reused here.

| Required property | Current state |
|---|---|
| Predicts OOD | Fresh native-application transfer now passes on new constructions and held-out city pairs. Same endpoints, native states and suffix; constant/fitted baseline comparison remains missing here. |
| Extracted | Earlier conditional head8/head9 package is standalone at three native inputs. The broader coupled head8/MLP8 package now passes native and isolated replay with three inputs and about17million floats. The later suffix remains external. |
| Selective | Fresh paired midpoint screen passes four unrelated controls and sixteen same-site random directions. Donor-free removal remains a different, untested claim. |
| Composes | Multiple partitions and the direct/MLP8 split failed. Keeping their coupled expression is necessary but does not satisfy the small-interaction criterion. |
| Simple, separately priced | Explicit algebra and literal parameter/state counts are available; no matched-effect simplicity advantage or full-model reduction is established. |

## Appendix

All instrument gates pass. The fresh screen used800forwards in10.103370101seconds. Full-precision results, distributions, row provenance, and gates:

- [Fresh result](../../TYPED_FACE_NATIVE8_FRESH_V1_RESULT.json)
- [Frozen protocol](../../TYPED_FACE_NATIVE8_FRESH_V1_PREREGISTRATION.md)
- [Rows](../../TYPED_FACE_NATIVE8_FRESH_V1_ROWS.json) and [CPU overlap audit](../../TYPED_FACE_NATIVE8_FRESH_V1_CPU_CONTROL.json)
- [Coupled prototype CPU result](../../TYPED_FACE_MLP8_COUPLED_V1_CPU_RESULT.json) and [implementation](../../typed_face_mlp8_coupled_v1.py)
- [Previous response and mediation report, including failures](research_update_2026-09-17_2240_native_regional_write.md)

The coupled prototype counts16,812,545floating scalars plus20token indices, and74,880supplied state scalars at sequence length32. Its three arrays are normalized current8, normalized donor-city8, and raw post-attention8 state. It returns the full change at the block9 input. Bias cancels from the MLP8 response; both self/cross products and RMS denominators remain. The CPU test uses synthetic states with learned weights; it cannot certify native behavior.

## Coupled extraction update

The [standalone package](../../extracted_circuits/typed_face_mlp8_coupled_v1/README.md) now contains all local code and weights. Native120forward replay and isolated40fixture replay pass every frozen precision gate. This certifies the local change entering block9, not the later spelling readout without its native suffix. Native and isolated checks use the now-opened fresh panel and do not add another independent OOD result.

Full precision: [native result](../../TYPED_FACE_MLP8_COUPLED_V1_RESULT.json), [isolated result](../../TYPED_FACE_MLP8_COUPLED_V1_STANDALONE_RESULT.json), [manifest](../../extracted_circuits/typed_face_mlp8_coupled_v1/manifest.json), and [protocol](../../TYPED_FACE_MLP8_COUPLED_V1_PREREGISTRATION.md). Native run120forwards,2.472572887seconds. The price remains16,812,545floating scalars plus20indices and three supplied native arrays. No compression gain is claimed.

The next prospective prediction test now has [frozen simple baselines](../../NATIVE8_PREDICTION_NULLS_V1_FROZEN.json): cue constants and an ordinary least-squares predictor using text length, city position, cue, and endpoint features. They were trained only on opened parent-effect measurements; no held-out result exists yet. The formula must beat each baseline by the registered margin on a later fresh panel.
