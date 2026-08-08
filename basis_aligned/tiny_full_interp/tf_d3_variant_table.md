# The six architectures at DEPTH 3, width 128

Registered predictions: `tf_d3_variant_predictions.json`, written before the first training step. Instruments: `tf_interp3.py` verbatim and `tf_depth_addendum.py` - the same code path as the depth-2 slice and the depth ladder.

SLOT GEOMETRY: 128 is not divisible by 2*depth = 6, so `slots` and `shrink` run with 8 slots of 16 rather than depth 2's 4 of 32; `bandwidth`/`predicate`/`codebook` scatter into 6 solved slots (stream 168). Controls in `GRID.md`.

| variant | seeds | n_slots x slot | params | held CE (T512) | induction ± sd | model-seed t | above own probe floor | A0 into layer 1 [zero, resample] | A1 into layer 2 [zero, resample] | A1 share of dominant MLP |
|---|---|---|---|---|---|---|---|---|---|---|
| vanilla | 3 | 1x128 | 1,933,696 | 4.5276 ± 0.0007 | +0.1085 ± 0.0133 | 14.08 | 3/3 | [4.388e-06, 2.201e-06] | [0.1617, 0.102] | 0.2326 |
| slots | 3 | 8x16 | 1,933,696 | 4.7433 ± 0.0084 | +0.0822 ± 0.0181 | 7.86 | 3/3 | [0.3266, 0.0849] | [0.06521, 0.03655] | 0.569 |
| bandwidth | 3 | 6x28 | 2,268,756 | 4.5307 ± 0.0134 | +0.2617 ± 0.0287 | 15.77 | 3/3 | [0.8009, 0.08703] | [0.2147, 0.09159] | 1.329 |
| predicate | 3 | 6x28 | 2,281,092 | 4.3146 ± 0.0008 | +2.7578 ± 0.0954 | 50.08 | 3/3 | [0.09611, 0.03922] | [0.08713, 0.02518] | 0.6664 |
| codebook | 3 | 6x28 | 2,268,756 | 4.6496 ± 0.0188 | +0.1491 ± 0.0138 | 18.66 | 3/3 | [0.1534, 0.08122] | [0.1183, 0.08443] | 1.575 |
| shrink | 3 | 8x16 | 1,958,272 | 4.7024 ± 0.0057 | +0.1150 ± 0.0215 | 9.25 | 3/3 | [0.07065, 0.06796] | [0.05545, 0.06572] | 0.9464 |

## Verdicts against the registered predictions

```
{
  "PD1_qualitative_advantage": {
    "vanilla_induction": {
      "mean": 0.10849020922625528,
      "sd": 0.013345512068083595,
      "per_seed": [
        0.09735919104682118,
        0.12328491210937571,
        0.10482652452256894
      ],
      "n": 3
    },
    "vanilla_seeds_above_own_probe_floor": 3,
    "vanilla_model_seed_test_positive": true,
    "call": "CONFIRMED - the plain model inducts at this cell, so \"the variants induct where the plain model cannot\" is no longer true at depth 3 width 128"
  },
  "PD2_magnitude_advantage": {
    "induction_ratio_to_vanilla": {
      "slots": 0.7574692997897902,
      "bandwidth": 2.412646325465168,
      "predicate": 25.419918410593354,
      "codebook": 1.3746441101495388,
      "shrink": 1.059801671234319
    },
    "excluding_predicate": {
      "slots": 0.7574692997897902,
      "bandwidth": 2.412646325465168,
      "codebook": 1.3746441101495388,
      "shrink": 1.059801671234319
    },
    "n_variants_above_2x": 2,
    "n_variants_below_0.5x": 0,
    "call": "(a) ACCELERANT - the advantage the architectures had at depth 2 is what depth supplies by itself at depth 3"
  },
  "PD3_variant_layer0_channel": {
    "A0_into_layer1_read_zero_kl": {
      "vanilla": 4.387783747006324e-06,
      "slots": 0.32658656189839047,
      "bandwidth": 0.8008530040582021,
      "predicate": 0.0961130540817976,
      "codebook": 0.1534243058413267,
      "shrink": 0.07065437982479732
    },
    "A0_write_norm_share_of_that_read": {
      "vanilla": 0.0014598530048474925,
      "slots": 0.5033288482679424,
      "bandwidth": 0.49668930663426786,
      "predicate": 0.41616623581355555,
      "codebook": 0.4722537225748842,
      "shrink": 0.37658537446009815
    },
    "n_variants_at_or_above_0.05_nats": 5,
    "call": "CONFIRMED",
    "review_caveat": "per tf_reviewer_round_4.json O2b this is a MAGNITUDE statement about how much the first attention block writes, not a claim that a channel is open or closed; the write norm share is printed beside it for exactly that reason"
  },
  "PD4_depth_supplied_route": {
    "A1_into_layer2_share_of_dominant_mlp": {
      "vanilla": 0.23255361554841583,
      "slots": 0.5690481364738195,
      "bandwidth": 1.3285996848981394,
      "predicate": 0.6663690865781563,
      "codebook": 1.5750579202574393,
      "shrink": 0.946353568478593
    },
    "all_six_between_0.10_and_0.70": false,
    "variant_mean_at_or_above_vanilla": "True"
  },
  "PD5_route_use": {
    "A1_out_of_layer2_read": {
      "vanilla": 0.857227960538934,
      "slots": 0.5990262050655262,
      "bandwidth": 0.918197229159644,
      "predicate": 0.014860359762864359,
      "codebook": 0.9096448679566252,
      "shrink": 0.9675571366273815
    },
    "A0_out_of_layer1_read": {
      "vanilla": 0.00017149081180437527,
      "slots": 0.7119724724180796,
      "bandwidth": 0.8012496451182555,
      "predicate": 0.015021843514048462,
      "codebook": 0.5668727375297505,
      "shrink": 0.2275064949933593
    },
    "A0_out_of_layer2_read": {
      "vanilla": -0.0013024800167664638,
      "slots": 0.7687286857690522,
      "bandwidth": 0.29881053256619067,
      "predicate": -0.018310579049593455,
      "codebook": 0.2356200429905089,
      "shrink": 0.20664342276003023
    },
    "note": "fractions whose baseline induction is at or below its own floor are meaningless ratios and must not be read"
  },
  "PD6_held_ce": {
    "vanilla": {
      "mean": 4.527606666666667,
      "sd": 0.0007393465583428476,
      "per_seed": [
        4.52845,
        4.52707,
        4.5273
      ],
      "n": 3
    },
    "slots": {
      "mean": 4.7432799999999995,
      "sd": 0.008422155306096033,
      "per_seed": [
        4.75181,
        4.73497,
        4.74306
      ],
      "n": 3
    },
    "bandwidth": {
      "mean": 4.53073,
      "sd": 0.013412855773473552,
      "per_seed": [
        4.51706,
        4.53126,
        4.54387
      ],
      "n": 3
    },
    "predicate": {
      "mean": 4.314649999999999,
      "sd": 0.0008072793816266223,
      "per_seed": [
        4.31402,
        4.31437,
        4.31556
      ],
      "n": 3
    },
    "codebook": {
      "mean": 4.6496466666666665,
      "sd": 0.018796867646853337,
      "per_seed": [
        4.63025,
        4.65091,
        4.66778
      ],
      "n": 3
    },
    "shrink": {
      "mean": 4.702356666666667,
      "sd": 0.005707541794269812,
      "per_seed": [
        4.70816,
        4.69675,
        4.70216
      ],
      "n": 3
    }
  },
  "PD7_depth_gain": {
    "vanilla": {
      "held_ce_depth2": 4.6463,
      "held_ce_depth3": 4.527606666666667,
      "gain": 0.11869333333333287
    },
    "slots": {
      "held_ce_depth2": 4.741406666666666,
      "held_ce_depth3": 4.7432799999999995,
      "gain": -0.0018733333333331714
    },
    "bandwidth": {
      "held_ce_depth2": 4.627883333333333,
      "held_ce_depth3": 4.53073,
      "gain": 0.09715333333333298
    },
    "predicate": {
      "held_ce_depth2": 4.38614,
      "held_ce_depth3": 4.314649999999999,
      "gain": 0.07149000000000072
    },
    "codebook": {
      "held_ce_depth2": 4.754213333333333,
      "held_ce_depth3": 4.6496466666666665,
      "gain": 0.10456666666666692
    },
    "shrink": {
      "held_ce_depth2": 4.724306666666667,
      "held_ce_depth3": 4.702356666666667,
      "gain": 0.021950000000000358
    }
  }
}
```
