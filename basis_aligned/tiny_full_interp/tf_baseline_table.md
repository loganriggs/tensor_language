# The foldability tax — conventional softmax+GELU baseline

Positive gap = the conventional model wins = what the exact fold costs in prediction quality. Held cross-entropy in nats/token at T=512 on held rows [0:1500]; both families share the tokenizer, so nats are directly comparable (bits/byte = nats / (ln2 × 3.755)).

| depth | width | family CE | conventional ×4 CE | gap ×4 | conventional ×7 CE | gap ×7 | seeds (fam/×4/×7) | params (fam / ×4 / ×7) |
|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 5.1479 ± 0.0055 | 5.1138 ± 0.0021 | **0.0341** | 5.0767 ± 0.0042 | **0.0711** | 3/2/2 | 598080 / 573504 / 598080 |
| 1 | 128 | 4.8226 ± 0.0029 | 4.7939 ± 0.0009 | **0.0287** | 4.7487 ± 0.0001 | **0.0738** | 3/2/2 | 1343616 / 1245312 / 1343616 |
| 1 | 256 | 4.5591 ± 0.0027 | 4.5199 ± 0.0012 | **0.0393** | 4.4765 ± 0.0024 | **0.0827** | 3/2/2 | 3277056 / 2883840 / 3277056 |
| 2 | 64 | 5.0181 ± 0.0047 | 5.0297 ± 0.0020 | **-0.0116** | 4.9751 ± 0.0122 | **0.0430** | 3/2/2 | 671872 / 622720 / 671872 |
| 2 | 128 | 4.6463 ± 0.0075 | 4.6026 ± 0.0152 | **0.0437** | 4.5525 ± 0.0131 | **0.0938** | 3/2/2 | 1638656 / 1442048 / 1638656 |
| 2 | 256 | 4.3254 ± 0.0013 | 4.2652 ± 0.0120 | **0.0601** | 4.2098 ± 0.0137 | **0.1156** | 3/2/2 | 4456960 / 3670528 / 4456960 |
| 3 | 64 | 4.9417 ± 0.0010 | 4.9267 ± 0.0102 | **0.0150** | 4.8800 ± 0.0012 | **0.0616** | 3/2/2 | 745664 / 671936 / 745664 |
| 3 | 128 | 4.5276 ± 0.0007 | 4.4741 ± 0.0055 | **0.0535** | 4.4259 ± 0.0004 | **0.1017** | 3/2/2 | 1933696 / 1638784 / 1933696 |
| 3 | 256 | 4.2110 ± 0.0065 | 4.1490 ± 0.0035 | **0.0621** | 4.0963 ± 0.0009 | **0.1148** | 3/2/2 | 5636864 / 4457216 / 5636864 |

## Induction — the interesting half

Synthetic order-sensitive score (shuffled − repeat), decided over MODEL seeds at t > 4.3.

| depth | width | family induction | family inducts? | conventional ×4 induction | conventional inducts? | family bag | conventional bag |
|---|---|---|---|---|---|---|---|
| 1 | 64 | -0.0115 ± 0.0025 (t -8.1) | no | -0.0219 ± 0.0044 (t -7.1) | no | +0.0313 | +0.0543 |
| 1 | 128 | -0.0264 ± 0.0019 (t -24.4) | no | -0.0349 ± 0.0014 (t -34.2) | no | +0.0599 | +0.0956 |
| 1 | 256 | -0.0354 ± 0.0015 (t -40.2) | no | -0.0447 ± 0.0008 (t -80.3) | no | +0.0870 | +0.1500 |
| 2 | 64 | -0.0140 ± 0.0022 (t -10.8) | no | -0.0153 ± 0.0009 (t -22.8) | no | +0.0453 | +0.0504 |
| 2 | 128 | -0.0034 ± 0.0099 (t -0.6) | no | +0.1356 ± 0.0750 (t 2.6) | no | +0.0858 | +0.1009 |
| 2 | 256 | +0.0938 ± 0.0086 (t 18.8) | YES | +0.3283 ± 0.0363 (t 12.8) | **YES** | +0.1397 | +0.1461 |
| 3 | 64 | +0.0035 ± 0.0041 (t 1.5) | no | +0.1365 ± 0.0477 (t 4.0) | no | +0.0556 | +0.0648 |
| 3 | 128 | +0.1085 ± 0.0133 (t 14.1) | YES | +0.5871 ± 0.0501 (t 16.6) | **YES** | +0.0990 | +0.1126 |
| 3 | 256 | +0.2207 ± 0.0605 (t 6.3) | YES | +0.8197 ± 0.0461 (t 25.1) | **YES** | +0.1420 | +0.1490 |

## Registered predictions, scored

| prediction | verdict |
|---|---|
| L1_conventional_wins_every_cell_by_0.05_to_0.20 | **REFUTED** |
| L2_A2_gap_grows_with_depth | **CONFIRMED** |
| A4_conventional_inducts_at_depth2_width128 | **REFUTED** |
| A5_depth1_null_in_both_families | **CONFIRMED** |
| L3 (matched parameters shrink the gap by a third) | **REFUTED** |
| A3 (matched parameters WIDEN the gap, because the family is the larger model) | **CONFIRMED** |

## Controls

- `model_and_harness`: pass = **True**
- `probe_shim`: pass = **True**

Full record: `tf_baseline_std.json`; registered predictions: `tf_baseline_predictions.json`.
