# The foldability tax — conventional softmax+GELU baseline

Positive gap = the conventional model wins = what the exact fold costs in prediction quality. Held cross-entropy in nats/token at T=512 on held rows [0:1500]; both families share the tokenizer, so nats are directly comparable (bits/byte = nats / (ln2 × 3.755)).

| depth | width | family CE | conventional ×4 CE | gap ×4 | conventional ×7 CE | gap ×7 | seeds (fam/×4/×7) | params (fam / ×4 / ×7) |
|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 5.1479 ± 0.0055 | — | **—** | — | **—** | 3/0/0 | 598080 / — / — |
| 1 | 128 | 4.8226 ± 0.0029 | — | **—** | — | **—** | 3/0/0 | 1343616 / — / — |
| 1 | 256 | 4.5591 ± 0.0027 | — | **—** | — | **—** | 3/0/0 | 3277056 / — / — |
| 2 | 64 | 5.0181 ± 0.0047 | — | **—** | — | **—** | 3/0/0 | 671872 / — / — |
| 2 | 128 | 4.6463 ± 0.0075 | — | **—** | — | **—** | 3/0/0 | 1638656 / — / — |
| 2 | 256 | 4.3254 ± 0.0013 | — | **—** | — | **—** | 3/0/0 | 4456960 / — / — |
| 3 | 64 | 4.9417 ± 0.0010 | — | **—** | — | **—** | 3/0/0 | 745664 / — / — |
| 3 | 128 | 4.5276 ± 0.0007 | — | **—** | — | **—** | 3/0/0 | 1933696 / — / — |
| 3 | 256 | 4.2110 ± 0.0065 | — | **—** | — | **—** | 3/0/0 | 5636864 / — / — |

## Induction — the interesting half

Synthetic order-sensitive score (shuffled − repeat), decided over MODEL seeds at t > 4.3.

| depth | width | family induction | family inducts? | conventional ×4 induction | conventional inducts? | family bag | conventional bag |
|---|---|---|---|---|---|---|---|
| 1 | 64 | -0.0115 ± 0.0025 (t —) | no | — ± 0.0000 (t —) | no | +0.0313 | — |
| 1 | 128 | -0.0264 ± 0.0019 (t —) | no | — ± 0.0000 (t —) | no | +0.0599 | — |
| 1 | 256 | -0.0354 ± 0.0015 (t —) | no | — ± 0.0000 (t —) | no | +0.0870 | — |
| 2 | 64 | -0.0140 ± 0.0022 (t —) | no | — ± 0.0000 (t —) | no | +0.0453 | — |
| 2 | 128 | -0.0034 ± 0.0099 (t —) | no | — ± 0.0000 (t —) | no | +0.0858 | — |
| 2 | 256 | +0.0938 ± 0.0086 (t —) | no | — ± 0.0000 (t —) | no | +0.1397 | — |
| 3 | 64 | +0.0035 ± 0.0041 (t —) | no | — ± 0.0000 (t —) | no | +0.0556 | — |
| 3 | 128 | +0.1085 ± 0.0133 (t —) | no | — ± 0.0000 (t —) | no | +0.0990 | — |
| 3 | 256 | +0.2207 ± 0.0605 (t —) | no | — ± 0.0000 (t —) | no | +0.1420 | — |

## Registered predictions, scored

| prediction | verdict |
|---|---|
| L1_conventional_wins_every_cell_by_0.05_to_0.20 | **None** |
| L2_A2_gap_grows_with_depth | **None** |
| A4_conventional_inducts_at_depth2_width128 | **None** |
| A5_depth1_null_in_both_families | **CONFIRMED** |
| L3 (matched parameters shrink the gap by a third) | **None** |
| A3 (matched parameters WIDEN the gap, because the family is the larger model) | **None** |

## Controls


Full record: `tf_baseline_std.json`; registered predictions: `tf_baseline_predictions.json`.
