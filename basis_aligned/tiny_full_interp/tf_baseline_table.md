# The foldability tax — conventional softmax+GELU baseline

Positive gap = the conventional model wins = what the exact fold costs in prediction quality. Held cross-entropy in nats/token at T=512 on held rows [0:1500]; both families share the tokenizer, so nats are directly comparable (bits/byte = nats / (ln2 × 3.755)).

| depth | width | family CE | conventional ×4 CE | gap ×4 | conventional ×7 CE | gap ×7 | seeds (fam/×4/×7) | params (fam / ×4 / ×7) |
|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 5.1479 ± 0.0055 | 5.1139 ± 0.0015 | **0.0340** | 5.0783 ± 0.0040 | **0.0696** | 3/3/3 | 598080 / 573504 / 598080 |
| 1 | 128 | 4.8226 ± 0.0029 | 4.7943 ± 0.0009 | **0.0283** | 4.7491 ± 0.0007 | **0.0734** | 3/3/3 | 1343616 / 1245312 / 1343616 |
| 1 | 256 | 4.5591 ± 0.0027 | 4.5199 ± 0.0009 | **0.0392** | 4.4757 ± 0.0021 | **0.0834** | 3/3/3 | 3277056 / 2883840 / 3277056 |
| 2 | 64 | 5.0181 ± 0.0047 | 5.0286 ± 0.0023 | **-0.0105** | 4.9798 ± 0.0118 | **0.0384** | 3/3/3 | 671872 / 622720 / 671872 |
| 2 | 128 | 4.6463 ± 0.0075 | 4.6137 ± 0.0220 | **0.0326** | 4.5524 ± 0.0093 | **0.0939** | 3/3/3 | 1638656 / 1442048 / 1638656 |
| 2 | 256 | 4.3254 ± 0.0013 | 4.2642 ± 0.0087 | **0.0612** | 4.2082 ± 0.0100 | **0.1172** | 3/3/3 | 4456960 / 3670528 / 4456960 |
| 3 | 64 | 4.9417 ± 0.0010 | 4.9248 ± 0.0079 | **0.0169** | 4.8754 ± 0.0080 | **0.0662** | 3/3/3 | 745664 / 671936 / 745664 |
| 3 | 128 | 4.5276 ± 0.0007 | 4.4723 ± 0.0051 | **0.0553** | 4.4243 ± 0.0028 | **0.1033** | 3/3/3 | 1933696 / 1638784 / 1933696 |
| 3 | 256 | 4.2110 ± 0.0065 | 4.1536 ± 0.0084 | **0.0575** | 4.0951 ± 0.0021 | **0.1159** | 3/3/3 | 5636864 / 4457216 / 5636864 |

## Induction — the interesting half

Synthetic order-sensitive score (shuffled − repeat), decided over MODEL seeds at t > 4.3.

| depth | width | family induction | family inducts? | conventional ×4 induction | conventional inducts? | family bag | conventional bag |
|---|---|---|---|---|---|---|---|
| 1 | 64 | -0.0115 ± 0.0025 (t -8.1) | no | -0.0212 ± 0.0033 (t -11.1) | no | +0.0313 | +0.0545 |
| 1 | 128 | -0.0264 ± 0.0019 (t -24.4) | no | -0.0344 ± 0.0013 (t -47.4) | no | +0.0599 | +0.0963 |
| 1 | 256 | -0.0354 ± 0.0015 (t -40.2) | no | -0.0430 ± 0.0030 (t -24.4) | no | +0.0870 | +0.1464 |
| 2 | 64 | -0.0140 ± 0.0022 (t -10.8) | no | -0.0149 ± 0.0010 (t -25.0) | no | +0.0453 | +0.0506 |
| 2 | 128 | -0.0034 ± 0.0099 (t -0.6) | no | +0.1061 ± 0.0736 (t 2.5) | no | +0.0858 | +0.0967 |
| 2 | 256 | +0.0938 ± 0.0086 (t 18.8) | YES | +0.3277 ± 0.0257 (t 22.1) | **YES** | +0.1397 | +0.1435 |
| 3 | 64 | +0.0035 ± 0.0041 (t 1.5) | no | +0.1252 ± 0.0391 (t 5.5) | **YES** | +0.0556 | +0.0642 |
| 3 | 128 | +0.1085 ± 0.0133 (t 14.1) | YES | +0.6151 ± 0.0600 (t 17.7) | **YES** | +0.0990 | +0.1147 |
| 3 | 256 | +0.2207 ± 0.0605 (t 6.3) | YES | +0.7826 ± 0.0720 (t 18.8) | **YES** | +0.1420 | +0.1456 |

## Registered predictions, scored

| prediction | verdict |
|---|---|
| L1_conventional_wins_every_cell_by_0.05_to_0.20 | **REFUTED** |
| L2_A2_gap_grows_with_depth | **CONFIRMED** |
| A4_conventional_inducts_at_depth2_width128 | **REFUTED** |
| A5_depth1_null_in_both_families | **CONFIRMED** |
| L3 (matched parameters shrink the gap by a third) | **REFUTED** |
| A3 (matched parameters WIDEN the gap, because the family is the larger model) | **CONFIRMED** |

## Learning-rate fairness bound

| cell | 0.01 | 0.02 (primary) | 0.04 | best | gain over 0.02 |
|---|---|---|---|---|---|
| d1_w128_x4 | 4.7995 | 4.7943 | 4.7962 | 0.02 | 0.0000 |
| d2_w128_x4 | 4.6067 | 4.6137 | 4.6246 | 0.01 | 0.0070 |
| d2_w128_x7 | 4.5432 | 4.5524 | 4.5563 | 0.01 | 0.0091 |
| d3_w128_x4 | 4.4723 | 4.4723 | 4.4854 | 0.02 | 0.0000 |

## Query/key-norm control

| cell | with QK-norm | without | cost of removing |
|---|---|---|---|
| d2_w128_x4 | 4.613713333333333 | 4.439203333333333 | -0.17450999999999972 |

## Controls

- `model_and_harness`: pass = **True**
- `probe_shim`: pass = **True**

Full record: `tf_baseline_std.json`; registered predictions: `tf_baseline_predictions.json`.
