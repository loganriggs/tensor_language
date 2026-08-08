# Slot-geometry controls for the depth-3 variant slice

Same instruments as the slice (`tf_interp3.py`, `tf_depth_addendum.py`). Induction decided over MODEL seeds; route KLs quoted [zero, resample] beside the write norm share.

| arm | seeds | n_slots x slot | params | held CE | induction ± sd | model-seed t | above own probe floor | A0 into layer-1 read [zero, resample] | A0 write norm share |
|---|---|---|---|---|---|---|---|---|---|
| **slots_d2_w128_n4** | 3 | 4x32 | 1,638,656 | 4.7414 ± 0.0056 | +0.0972 ± 0.0275 | 6.12 | 3/3 | [0.5513, 0.1241] | 0.5628 |
| **shrink_d2_w128_n4** | 3 | 4x32 | 1,650,944 | 4.7243 ± 0.0100 | +0.0860 ± 0.0303 | 4.92 | 3/3 | [0.2289, 0.1434] | 0.1589 |
| **vanilla_d2_w128** | 3 | 1x128 | 1,638,656 | 4.6463 ± 0.0075 | -0.0034 ± 0.0099 | -0.59 | 0/3 | [1.287e-05, 4.324e-06] | 0.00212 |
| **slots_d2_w128_n8** | 3 | 8x16 | 1,638,656 | 4.8904 ± 0.0075 | +0.0200 ± 0.0191 | 1.81 | 2/3 | [0.2234, 0.09172] | 0.5417 |
| **shrink_d2_w128_n8** | 2 | 8x16 | 1,659,136 | 4.8386 ± 0.0034 | +0.0295 ± 0.0034 | 12.32 | 2/2 | [0.0707, 0.08689] | 0.3992 |
| **vanilla_d3_w128** | 3 | 1x128 | 1,933,696 | 4.5276 ± 0.0007 | +0.1085 ± 0.0133 | 14.08 | 3/3 | [4.388e-06, 2.201e-06] | 0.00146 |
| **slots_d3_w128_n8** | 3 | 8x16 | 1,933,696 | 4.7433 ± 0.0084 | +0.0822 ± 0.0181 | 7.86 | 3/3 | [0.3266, 0.0849] | 0.5033 |
| **shrink_d3_w128_n8** | 3 | 8x16 | 1,958,272 | 4.7024 ± 0.0057 | +0.1150 ± 0.0215 | 9.25 | 3/3 | [0.07065, 0.06796] | 0.3766 |
| **vanilla_d3_w192** | 1 | 1x192 | 3,564,096 | 4.3324 ± 0.0000 | +0.1871 ± 0.0000 | — | 1/1 | [1.972e-05, 2.947e-06] | 0.003913 |
| **slots_d3_w192_n6** | 1 | 6x32 | 3,564,096 | 4.4219 ± 0.0000 | +0.3443 ± 0.0000 | — | 1/1 | [0.7243, 0.135] | 0.4765 |
| **shrink_d3_w192_n6** | MISSING | | | | | | | | |
| **slots_d3_w192_n8** | MISSING | | | | | | | | |

## Controls

```
{
  "a_depth2_geometry_slots": {
    "label": "CONTROL A - slots at depth 2 width 128: 8x16 vs the published 4x32. Same depth, same width, same parameters; only the slot geometry moves. If 8x16 costs induction HERE, the depth-3 slots deficit is not evidence about the architecture.",
    "a": "slots d2 w128_g8",
    "b": "slots d2 w128",
    "a_geometry": "8x16",
    "b_geometry": "4x32",
    "a_params": 1638656,
    "b_params": 1638656,
    "induction_a": {
      "mean": 0.019984605577256787,
      "sd": 0.019073665963222186,
      "per_seed": [
        -0.0004955079820426533,
        0.03724096086290167,
        0.02320836385091134
      ],
      "n": 3
    },
    "induction_b": {
      "mean": 0.09721570897985406,
      "sd": 0.02753146277211961,
      "per_seed": [
        0.11294615003797795,
        0.1132752948337135,
        0.06542568206787074
      ],
      "n": 3
    },
    "induction_delta": -0.07723110340259727,
    "induction_ratio": 0.20556971488422907,
    "induction_welch_t": -3.9939058714382063,
    "induction_welch_df": 3.560397423252762,
    "induction_separated_at_95": false,
    "held_ce_a": {
      "mean": 4.8903533333333336,
      "sd": 0.007487137859912471,
      "per_seed": [
        4.89867,
        4.88415,
        4.88824
      ],
      "n": 3
    },
    "held_ce_b": {
      "mean": 4.741406666666666,
      "sd": 0.0056014492172410625,
      "per_seed": [
        4.74182,
        4.73561,
        4.74679
      ],
      "n": 3
    },
    "held_ce_delta": 0.14894666666666723,
    "held_ce_welch_t": 27.5900550297426
  },
  "a_depth2_geometry_shrink": {
    "label": "CONTROL A - shrink at depth 2 width 128: 8x16 vs the published 4x32.",
    "a": "shrink d2 w128_g8",
    "b": "shrink d2 w128",
    "a_geometry": "8x16",
    "b_geometry": "4x32",
    "a_params": 1659136,
    "b_params": 1650944,
    "induction_a": {
      "mean": 0.029500410291883838,
      "sd": 0.0033856244965949677,
      "per_seed": [
        0.027106412251790245,
        0.03189440833197743
      ],
      "n": 2
    },
    "induction_b": {
      "mean": 0.08600764097990776,
      "sd": 0.03027607588887017,
      "per_seed": [
        0.05104906294080891,
        0.10323367648654483,
        0.10374018351236955
      ],
      "n": 3
    },
    "induction_delta": -0.05650723068802392,
    "induction_ratio": 0.3429975517963041,
    "induction_welch_t": -3.2027991551729302,
    "induction_welch_df": 2.074273151853633,
    "induction_separated_at_95": false,
    "held_ce_a": {
      "mean": 4.83862,
      "sd": 0.0034223968209427017,
      "per_seed": [
        4.84104,
        4.8362
      ],
      "n": 2
    },
    "held_ce_b": {
      "mean": 4.724306666666667,
      "sd": 0.009994520165237218,
      "per_seed": [
        4.73574,
        4.71995,
        4.71723
      ],
      "n": 3
    },
    "held_ce_delta": 0.1143133333333326,
    "held_ce_welch_t": 18.268933534438812
  },
  "b_width192_slots_vs_plain": {
    "label": "CONTROL B - depth 3 width 192, where 6 slots x 32 is EXACT: slots against the plain model at the same width. This is the depth-3 verdict re-run with the geometry the architecture wants.",
    "a": "slots d3 w192",
    "b": "vanilla d3 w192",
    "a_geometry": "6x32",
    "b_geometry": "1x192",
    "a_params": 3564096,
    "b_params": 3564096,
    "induction_a": {
      "mean": 0.3442906697591141,
      "sd": 0.0,
      "per_seed": [
        0.3442906697591141
      ],
      "n": 1
    },
    "induction_b": {
      "mean": 0.1871306313408745,
      "sd": 0.0,
      "per_seed": [
        0.1871306313408745
      ],
      "n": 1
    },
    "induction_delta": 0.1571600384182396,
    "induction_ratio": 1.8398413305834422,
    "induction_welch_t": null,
    "induction_welch_df": null,
    "induction_separated_at_95": false,
    "held_ce_a": {
      "mean": 4.42188,
      "sd": 0.0,
      "per_seed": [
        4.42188
      ],
      "n": 1
    },
    "held_ce_b": {
      "mean": 4.33237,
      "sd": 0.0,
      "per_seed": [
        4.33237
      ],
      "n": 1
    },
    "held_ce_delta": 0.08950999999999976,
    "held_ce_welch_t": null
  },
  "b_width192_shrink_vs_plain": {
    "label": "CONTROL B - shrink against plain at depth 3 width 192.",
    "status": "MISSING",
    "have_a": false,
    "have_b": true
  },
  "b2_width192_geometry_only": {
    "label": "CONTROL B2 - the geometry contrast at FIXED width 192: 8x24 (two dead slots) against 6x32. Isolates slot geometry from width.",
    "status": "MISSING",
    "have_a": false,
    "have_b": true
  },
  "w128_slots_vs_plain": {
    "label": "REFERENCE - the depth-3 width-128 comparison the verdict was read from (slots 8x16 vs plain).",
    "a": "slots d3 w128",
    "b": "vanilla d3 w128",
    "a_geometry": "8x16",
    "b_geometry": "1x128",
    "a_params": 1933696,
    "b_params": 1933696,
    "induction_a": {
      "mean": 0.08217800281665942,
      "sd": 0.018110121275048862,
      "per_seed": [
        0.062017907036675625,
        0.09706995222303619,
        0.08744614919026646
      ],
      "n": 3
    },
    "induction_b": {
      "mean": 0.10849020922625528,
      "sd": 0.013345512068083595,
      "per_seed": [
        0.09735919104682118,
        0.12328491210937571,
        0.10482652452256894
      ],
      "n": 3
    },
    "induction_delta": -0.026312206409595862,
    "induction_ratio": 0.7574692997897902,
    "induction_welch_t": -2.025856397208549,
    "induction_welch_df": 3.6774744306533766,
    "induction_separated_at_95": false,
    "held_ce_a": {
      "mean": 4.7432799999999995,
      "sd": 0.008422155306096033,
      "per_seed": [
        4.75181,
        4.73497,
        4.74306
      ],
      "n": 3
    },
    "held_ce_b": {
      "mean": 4.527606666666667,
      "sd": 0.0007393465583428476,
      "per_seed": [
        4.52845,
        4.52707,
        4.5273
      ],
      "n": 3
    },
    "held_ce_delta": 0.21567333333333227,
    "held_ce_welch_t": 44.184183239225504
  },
  "w128_shrink_vs_plain": {
    "label": "REFERENCE - shrink 8x16 vs plain at depth 3 width 128.",
    "a": "shrink d3 w128",
    "b": "vanilla d3 w128",
    "a_geometry": "8x16",
    "b_geometry": "1x128",
    "a_params": 1958272,
    "b_params": 1933696,
    "induction_a": {
      "mean": 0.11497810505054627,
      "sd": 0.021526452731252886,
      "per_seed": [
        0.09148985544840507,
        0.13376600477430572,
        0.119678454928928
      ],
      "n": 3
    },
    "induction_b": {
      "mean": 0.10849020922625528,
      "sd": 0.013345512068083595,
      "per_seed": [
        0.09735919104682118,
        0.12328491210937571,
        0.10482652452256894
      ],
      "n": 3
    },
    "induction_delta": 0.00648789582429099,
    "induction_ratio": 1.059801671234319,
    "induction_welch_t": 0.4436793938268036,
    "induction_welch_df": 3.3395165045948487,
    "induction_separated_at_95": false,
    "held_ce_a": {
      "mean": 4.702356666666667,
      "sd": 0.005707541794269812,
      "per_seed": [
        4.70816,
        4.69675,
        4.70216
      ],
      "n": 3
    },
    "held_ce_b": {
      "mean": 4.527606666666667,
      "sd": 0.0007393465583428476,
      "per_seed": [
        4.52845,
        4.52707,
        4.5273
      ],
      "n": 3
    },
    "held_ce_delta": 0.17474999999999952,
    "held_ce_welch_t": 52.59145240122904
  }
}
```

## Verdict inputs

```
{
  "geometry_costs_induction_at_depth2_slots": {
    "published_4x32": {
      "mean": 0.09721570897985406,
      "sd": 0.02753146277211961,
      "per_seed": [
        0.11294615003797795,
        0.1132752948337135,
        0.06542568206787074
      ],
      "n": 3
    },
    "same_cell_8x16": {
      "mean": 0.019984605577256787,
      "sd": 0.019073665963222186,
      "per_seed": [
        -0.0004955079820426533,
        0.03724096086290167,
        0.02320836385091134
      ],
      "n": 3
    },
    "ratio_8x16_over_4x32": 0.20556971488422907,
    "welch_t": -3.9939058714382063,
    "held_ce_cost_of_8x16": 0.14894666666666723
  },
  "geometry_costs_induction_at_depth2_shrink": {
    "published_4x32": {
      "mean": 0.08600764097990776,
      "sd": 0.03027607588887017,
      "per_seed": [
        0.05104906294080891,
        0.10323367648654483,
        0.10374018351236955
      ],
      "n": 3
    },
    "same_cell_8x16": {
      "mean": 0.029500410291883838,
      "sd": 0.0033856244965949677,
      "per_seed": [
        0.027106412251790245,
        0.03189440833197743
      ],
      "n": 2
    },
    "ratio_8x16_over_4x32": 0.3429975517963041,
    "welch_t": -3.2027991551729302,
    "held_ce_cost_of_8x16": 0.1143133333333326
  },
  "depth3_exact_geometry_slots_vs_plain": {
    "induction_ratio": 1.8398413305834422,
    "held_ce_delta": 0.08950999999999976,
    "clears_2x_bar": false,
    "below_0.5x_bar": false
  }
}
```
