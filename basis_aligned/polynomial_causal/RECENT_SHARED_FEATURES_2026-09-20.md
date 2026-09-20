# Recent-path shared linear features: negative baseline

The v629 comparison found no compressed candidate meeting the declared 10% calibration limit. The selected program is the dense v628 reference; the storage-saving prediction failed. This is not a new simple circuit.

## Comparison

Weights-only SVD constructs a common input basis for Left16 and Right16, optionally a low-rank Down16 output map. Independent input bases provide a matched-storage control. Every arm keeps biases, attention17, and normalization explicit. All stored tensors in the complete conditional executor count toward price.

Each panel uses eight documents of 64 prediction positions. The first panel selects rank/family; the second reports the frozen selection. Both panels have been used previously and are not fresh holdouts. MLP16 full-write scales are 0, 0.5, 1, and 1.5. Maximum errors include both scalar reconstruction and intervention-change prediction.

| Family | Rank | Fraction of dense storage | Max calibration error | Max validation error |
| --- | ---: | ---: | ---: | ---: |
| dense | 1152 | 1.000 | 0.0610 | 0.0569 |
| shared_input | 128 | 0.617 | 2.7485 | 3.0126 |
| shared_input | 256 | 0.671 | 2.7950 | 3.0255 |
| shared_input | 512 | 0.781 | 2.8136 | 3.0055 |
| shared_input | 768 | 0.890 | 2.8000 | 2.9441 |
| shared_input_output | 128 | 0.428 | 2.6653 | 2.9478 |
| shared_input_output | 256 | 0.513 | 2.7132 | 2.9700 |
| shared_input_output | 512 | 0.684 | 2.7743 | 2.9845 |
| shared_input_output | 768 | 0.854 | 2.8012 | 2.9583 |
| independent_matched_input_output | 128 | 0.428 | 2.6502 | 2.9378 |
| independent_matched_input_output | 256 | 0.513 | 2.6519 | 2.9160 |
| independent_matched_input_output | 512 | 0.683 | 2.6908 | 2.9184 |
| independent_matched_input_output | 768 | 0.854 | 2.7022 | 2.8863 |

## Interpretation and checks

The compressed candidates have roughly unit relative intervention-change error, while the dense reference remains at 2.05–3.12%. A CPU SVD audit confirms the common-basis matrix error decreases from 0.896 at rank128 to 0.437 at rank768 and 2.07e-6 at full rank. Better matrix fidelity therefore does not ensure preservation of this downstream component.

Two executor tests pass: native small-model replay and factored execution against explicitly reconstructed dense matrices, across intervention scales. Native dense-component replay is 8.59e-7. The managed job completed with 16 native forwards and 208 local replays, no fitting or model updates.

This only rejects the tested spectral matrix-factor families and ranks. It is not a joint tensor decomposition, a polynomial-quotient optimum, or a lower bound on sparse DAG complexity. The next discovery objective must account for the contracted observable and shared bilinear features, rather than assume individual matrix singular directions preserve the circuit.

The dense fallback still stores 24,235,715 tensor values (about 97 MB) and requires upstream h16/x0/v1 ports. The regenerated v629 package is a local fallback, not a claimed compressed deliverable; the committed v628 package remains the portable reference.

[Native receipt](../bilinear_quotient/circuits/followups/recent_shared_features_v629_result.json) · [SVD check](../bilinear_quotient/circuits/followups/recent_shared_features_v629_svd_check.json) · [HT/shared-DAG objective](HIERARCHICAL_TUCKER_SHARED_DAG_DIRECTION_2026-09-20.md)
