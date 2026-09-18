# City interchange requires a coupled operation

The regional path generates a city write for head8.2 from two supplied residual6 prefixes, then installs it into the native model. The complete city swap retains its fresh prediction and selective-interchange evidence, but its key and value exchanges fail the registered independence test on the now-opened panel. Extraction still requires native prefixes and the native downstream model; token-only prediction, independent composition and matched-effect simplicity remain unproved.

```mermaid
flowchart LR
  T[Tokens] -. native blocks 0–6 remain external .-> R[Two residual6 prefixes]
  R -->|fold: standalone prefix replay, 40 opened sequences| F[Attention7 + five MLP7 readers]
  F -->|fold: full key × key × value operator| C[Coupled head8.2 city swap]
  C -->|edit: fresh prediction error 1.4%, 20 documents| S[Native MLP8 and remaining layers]
  S --> Y[Regional spelling and four control readers]
  K[Separate key and value edits] -. falsified: interaction 2.3 × smaller effect, opened .-> Y
```

The decomposition fails at both places where interactions can arise. Omitting the head's cross term changes the final target effect by **42%**, against a 10% gate. Even with that term omitted, the joint installation of the two main changes differs from the sum of their separately measured effects by **1.1 times** the smaller single effect, against a 0.35 gate. Keeping the complete swap is therefore necessary under this test; successful algebraic expansion does not justify independently reusable pieces.

Metrics: each effect is the edited minus native spelling-logit margin, stacked over 240 probes from 20 documents and 40 paired city sequences. Interaction ratio is the L2 norm of joint effect minus the sum of single effects, divided by the smaller single-effect norm. Head-cross omission is the L2 difference between complete swap and summed-main installation divided by complete-swap effect norm. The six endpoints per paired context are repeated measurements, not independent documents. Paired attenuation measures reduction of the native UK/US city contrast; 117 of 120 pairs exceed the frozen capability floor.

| Claim | Evidence | Evaluation | Key numbers | Status |
|---|---|---|---|---|
| Complete swap predicts native-reference effect | edit | fresh at original test | 1.4% relative error | passes |
| Two-prefix standalone extraction | fold | replay | 50,688 supplied native floats; 23,300,998 stored FP32 values | established at this boundary |
| Complete swap selectively changes regional behavior | edit | fresh at original test | 117/117 capable pairs positive; controls ≤7.5% of target; 16 nulls beaten | passes for interchange; removal is separate |
| Keys and value independently compose | edit | opened | 2.3 × smaller effect; gate 0.35 | fails |
| Head-cross omission is small | edit | opened | 42%; gate 10% | fails |
| Downstream main-effect interaction is small | edit | opened | 1.1 × smaller effect; gate 0.35 | fails |
| Matched-effect simplicity | fold + pricing | not evaluated against required null | storage and native-state costs explicit | not yet tested |

Both single pieces are live: key-only target RMS is 14% of joint RMS and value-only is 96%, each above the 10% floor. Key-only attenuation is positive on 44% of capable pairs; value-only and joint are positive on all capable pairs. These opened descriptive signs do not promote the value-only arm as a fresh selective component.

The failure is not aggregate cancellation hiding a few exceptional contexts: joint interaction exceeds the 0.35 gate in all 20 documents (range 1.4–5.3). The head-cross omission exceeds 10% in 19 documents; downstream main interaction exceeds 0.35 in 16. A subsequent descriptive signed accounting shows partial cancellation between those two behavioral differences (cosine −0.68). Their aligned fractions of total interaction are 1.2 and −0.18. These are differences propagated through the full suffix, not localized independent MLP interventions.

All four behavioral properties remain tracked: fresh prediction and boundary extraction are supported, selective interchange is supported with the stated scope, and independent composition is falsified for this split. Keep the coupled operator and stop subdividing it without a new mechanistic reason. The next weight-folding handoff remains the registered attention7 head3 omission candidate: pack the actual reduced factors and test fresh prediction/selectivity before adopting its proposed savings.

## Appendix: receipts and execution

- [Frozen registration](../../CITY_INTERCHANGE_COMPOSITION_V1_PREREGISTRATION.md), [binding](../../CITY_INTERCHANGE_COMPOSITION_V1_BINDING.json), [runner](../../check_city_interchange_composition_v1.py).
- [Full result including document signs and four controls](../../CITY_INTERCHANGE_COMPOSITION_V1_RESULT.json), [saved scores](../../CITY_INTERCHANGE_COMPOSITION_V1_ARTIFACT.pt), [signed accounting](../../CITY_INTERCHANGE_COMPOSITION_V1_SIGNED_ACCOUNTING.json).
- [Prior fresh interchange and extracted-interface report](research_update_2026-09-18_0235_city_interchange.md), including the retained installed-prefix scope correction.
- Five arms: native; both city keys swapped; complete city value swapped; sum of the two head deltas; complete joint swap. Recipient queries/background are fixed; MLP8 and every later layer are recomputed natively. Both key factors and current/inherited values are retained.
- CPU, two threads, 200 sequence-equivalent forwards, 90 batched block calls; measured execution 13.348048896994442 seconds. Native and fulljoint scores reproduce the prior reference exactly at installed precision. Support and finite gates pass. All algebraic closures pass; see CPU preflight receipt.
- pred_a=true, pred_b=false, pred_c=false. Full precision interaction2.344823504760865; headcross0.4186297368244815; suffix1.0746264914225099.
- Invalid preflight attempt: binding construction initially used the wrong directory for a shared runtime; it failed before model access. Corrected dependency path and dry-run passed before the registered run. No outcomes or thresholds changed.
- No random-split specificity or fresh joint promotion was run because the basic composition screen failed. No new dataset, fitted coefficient, or selected split was introduced by the signed accounting.

Engineering handoff completed after the composition analysis: [packed head-omission replay](../../CITY_ATTENTION7_DROP3_PACK_V1_RESULT.json) passes on40opened fixtures with physically owned storage22,441,606FP32 values (884,736fewer). This is a city-removal candidate, not the interchange operator tested above. [Fresh confirmation is registered](../../CITY_ATTENTION7_DROP3_FRESH_V1_PREREGISTRATION.md); no fresh outcome has yet been evaluated.
