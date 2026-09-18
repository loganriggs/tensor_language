# Fresh V2 reuse of the coupled value path

The same frozen direct-plus-MLP8 value operator was replayed on a second outcome-blind FineWeb panel. The panel contains 20 new documents, 40 sequences, and 240 fixed probes, with no document overlap and no outcome-based selection. The coupled operator again behaves coherently globally: interaction `.15`, joint correction `.80` of the parent swap, controls at most `.19` of target, and 108/108 capable swaps remain positive.

| Claim | Evidence | Evaluation | Key result | Status |
|---|---|---|---|---|
| Fresh V2 panel is selected without outcomes | fit/selection receipt | 20 documents, 240 probes | 2,066 documents scanned; 0 overlap; no model calls | passes |
| Frozen folded value correction replays | fold | fresh V2 panel | max local error `2.2e-5`; support and subtraction checks pass | passes |
| Coupled direct-plus-MLP8 path suppresses parent transmission | edit | fresh V2 | joint/parent `.80`; controls ≤ `.19`; signed suppression `.79` | passes globally |
| Direct and MLP8 terms compose as independent pieces | edit | fresh V2 | interaction `.15`; no native-reversed subgroup occurred | not established |
| Fresh capable direction | edit | fresh V2 | `108/108` positive | passes for this panel |

The subgroup gate is deliberately not converted into a pass: the preregistration requires a native-reversed subgroup and an other-document subgroup to be reported separately, and V2 has zero reversed rows. This is informative reuse evidence for the coupled operator, not evidence that the direct and MLP8 terms are independently reusable in all contexts. The first fresh panel still contains the reversed subgroup and fails its preservation gate (interaction `.49`, one control `1.12×` target).

The four-property reading is therefore sharper. Prediction and extraction remain supported at the declared native boundary. Selective manipulation has fresh null-controlled support. Composition/reuse is a candidate for the coupled operator across two panels, while independent direct/MLP8 composition remains unresolved. Simplicity is still unestablished.

## Receipts

- [V2 registration](../../CITY_VALUE_PATH_FRESH_V2_PREREGISTRATION.md), [rows](../../CITY_VALUE_PATH_FRESH_V2_ROWS.json), [binding](../../CITY_VALUE_PATH_FRESH_V2_BINDING.json), and [row audit](../../CITY_VALUE_PATH_FRESH_V2_ROW_AUDIT.json).
- [V2 composition result](../../CITY_VALUE_PATH_FRESH_V2_RESULT.json), [preflight](../../CITY_VALUE_PATH_FRESH_V2_PREFLIGHT.json), and [artifact](../../CITY_VALUE_PATH_FRESH_V2_ARTIFACT.pt).
- [V1 composition result](../../CITY_VALUE_PATH_FRESH_V1_RESULT.json) remains the controlling failure for the reversed-subgroup preservation gate.
