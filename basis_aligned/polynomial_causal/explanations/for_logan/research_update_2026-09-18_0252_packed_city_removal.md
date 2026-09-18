# Smaller prefix preserves fresh city-removal effects

The regional removal path now generates head8.2's city edit using eight attention7 heads and five MLP7 readers from one supplied residual6 state. Removing one head from the generator's stored computation saves 884,736 FP32 values and passes fresh prediction/selectivity tests. The executor still requires native blocks0–6 and the downstream model; independent composition and matched-effect simplicity remain unresolved.

```mermaid
flowchart LR
 T[Tokens] -. native blocks 0–6 remain external .-> R[Residual6 state]
 R -->|fold: eight retained attention7 heads| A[MLP7 input]
 A -->|fold: five readers, opened exact packing replay| F[Head8.2 Q1 K1 Q2 K2 V]
 F -->|edit: 1.9% prediction error, 20 fresh Pile documents| D[City-removal delta]
 D -->|edit: full native suffix, 114 capable pairs| S[Regional spelling and four control readers]
```

This removes zero-based head3 from the **generator**, not from the native model. The generated delta approximates complete city-source removal at attention8 destinations. Both routing factors, current and inherited values, and all five MLP7 readers remain. Output normalization remains the previously registered approximation. No fitted proxy or quantization was introduced.

Metrics: prediction error is relative L2 error between generated and native-removal effects on spelling-logit margins. Selectivity is each unrelated reader's RMS movement divided by target RMS movement. Positive attenuation means removal reduces the paired UK/US city contrast. Capability requires a native contrast of at least0.1; probe endpoints within a document are not independent samples.

| Claim | Evidence | Evaluation | Key numbers | Status |
|---|---|---|---|---|
| Predicts native-removal effect | edit | fresh, 20 Pile documents | 1.9% error; gate5% | passes |
| Prediction on untouched natural arm | edit | fresh | 2.7% error; substituted arm1.6% | passes aggregate gate; strata descriptive |
| Selective removal | edit | fresh | 114/114 capable pairs positive; mean attenuation35%; controls≤12% | passes |
| Beats matched random removals | edit | fresh | 16/16;31×median target effect | passes |
| Standalone extraction | fold | replay of fresh fixtures | 40/40; declared native input only | established at this boundary |
| Smaller physical factor storage | fold + pricing | opened replay and fresh table | 884,736 fewer FP32 values at equal vocabulary | established storage reduction |
| Matched-effect simplicity null | pricing | not evaluated | storage reduction does not substitute for the required null | not yet tested |
| Independent composition | edit | not tested for this subset | related full-city key/value split failed2.3× versus0.35gate | unresolved; do not borrow a pass |

The frozen standalone package stores **22,418,566 FP32 values** (89,674,264 bytes) and requires36,864 native floats for a32-token residual6 input. Its386-token table is weight-derived; unsupported tokens are rejected. Against a nine-head generator with the same vocabulary, the saving is3.8%. This does not establish a whole-model compression frontier or reduce the native input requirement.

The next registered test transfers the same operator and gates to FineWeb. That tests a real corpus shift while preserving the regional task and fixed endpoint probes. It will not establish pretraining-disjointness, arbitrary vocabulary coverage, or structural template invariance.

## Appendix: reproducibility and limits

- [Standalone package](../../extracted_circuits/city_attention7_drop3_v1/README.md), [manifest](../../extracted_circuits/city_attention7_drop3_v1/manifest.json).
- [Fresh registration](../../CITY_ATTENTION7_DROP3_FRESH_V1_PREREGISTRATION.md), [binding](../../CITY_ATTENTION7_DROP3_FRESH_V1_BINDING.json), [result](../../CITY_ATTENTION7_DROP3_FRESH_V1_RESULT.json), [isolated execution](../../CITY_ATTENTION7_DROP3_FRESH_V1_ISOLATED_RESULT.json), [row audit](../../CITY_ATTENTION7_DROP3_FRESH_V1_ROW_AUDIT.json).
- [Physical packing replay](../../CITY_ATTENTION7_DROP3_PACK_V1_RESULT.json), [retained composition failure](research_update_2026-09-18_0245_city_composition.md), [FineWeb registration](../../CITY_ATTENTION7_DROP3_FINEWEB_V1_PREREGISTRATION.md).
- Fresh panel: first20 eligible previously unused documents reached after4,504 streamed documents;40 sequences and240 fixed spelling probes, one original and one city-substituted arm. Zero document or exact-input overlap with frozen panels. The preposition filter can include place-named organizations; endpoints are probes, not observed next tokens.
- CPU two threads;800 sequence-equivalent forwards,360 batched block calls;45.09456613799557seconds. All five gates pass. Prediction error0.018811160692865342; controls maximum0.12382678701573184; null-median ratio30.89119228877812. Full per-document signed attenuations are retained in JSON, including non-capable contexts.
- Standalone execution copies only package files to a temporary directory, then replays40 fixtures in0.4624707158654928seconds; unsupported token rejection and zero edit outside destinations pass. All algebraic/replay closures pass; see receipts.
- Fresh-reference query replay uses the complete nine-head reference computation. Candidate queries are allowed to differ after head omission and their errors are reported. This distinction was registered before native capture.
- Four properties: scoped fresh prediction supported; declared-boundary extraction supported; selective removal supported; independent composition unresolved. Simplicity is separately priced, with its matched-effect null still missing.
