# Slot-geometry controls for the depth-3 variant slice

Same instruments as the slice (`tf_interp3.py`, `tf_depth_addendum.py`). Induction decided over MODEL seeds; route KLs quoted [zero, resample] beside the write norm share.

| arm | seeds | n_slots x slot | params | held CE | induction ± sd | model-seed t | above own probe floor | A0 into layer-1 read [zero, resample] | A0 write norm share |
|---|---|---|---|---|---|---|---|---|---|
| **slots_d2_w128_n4** | 3 | 4x32 | 1,638,656 | 4.7414 ± 0.0056 | +0.0972 ± 0.0275 | 6.12 | 3/3 | [0.5513, 0.1241] | 0.5628 |
| **shrink_d2_w128_n4** | 3 | 4x32 | 1,650,944 | 4.7243 ± 0.0100 | +0.0860 ± 0.0303 | 4.92 | 3/3 | [0.2289, 0.1434] | 0.1589 |
| **vanilla_d2_w128** | 3 | 1x128 | 1,638,656 | 4.6463 ± 0.0075 | -0.0034 ± 0.0099 | -0.59 | 0/3 | [1.287e-05, 4.324e-06] | 0.00212 |
| **slots_d2_w128_n8** | 3 | 8x16 | 1,638,656 | 4.8904 ± 0.0075 | +0.0200 ± 0.0191 | 1.81 | 2/3 | [0.2234, 0.09172] | 0.5417 |
| **shrink_d2_w128_n8** | 3 | 8x16 | 1,659,136 | 4.8354 ± 0.0060 | +0.0291 ± 0.0025 | 20.18 | 3/3 | [0.07429, 0.0873] | 0.4016 |
| **vanilla_d3_w128** | 3 | 1x128 | 1,933,696 | 4.5276 ± 0.0007 | +0.1085 ± 0.0133 | 14.08 | 3/3 | [4.388e-06, 2.201e-06] | 0.00146 |
| **slots_d3_w128_n8** | 3 | 8x16 | 1,933,696 | 4.7433 ± 0.0084 | +0.0822 ± 0.0181 | 7.86 | 3/3 | [0.3266, 0.0849] | 0.5033 |
| **shrink_d3_w128_n8** | 3 | 8x16 | 1,958,272 | 4.7024 ± 0.0057 | +0.1150 ± 0.0215 | 9.25 | 3/3 | [0.07065, 0.06796] | 0.3766 |
| **vanilla_d3_w192** | 3 | 1x192 | 3,564,096 | 4.3286 ± 0.0033 | +0.1911 ± 0.0175 | 18.88 | 3/3 | [1.884e-05, 3.665e-06] | 0.003567 |
| **slots_d3_w192_n6** | 3 | 6x32 | 3,564,096 | 4.4311 ± 0.0086 | +0.3206 ± 0.0424 | 13.09 | 3/3 | [0.6305, 0.1259] | 0.2994 |
| **shrink_d3_w192_n6** | 3 | 6x32 | 3,607,104 | 4.4098 ± 0.0169 | +0.3557 ± 0.0966 | 6.38 | 3/3 | [0.2875, 0.09297] | 0.2208 |
| **slots_d3_w192_n8** | 3 | 8x24 | 3,564,096 | 4.4927 ± 0.0037 | +0.2050 ± 0.0149 | 23.85 | 3/3 | [0.4442, 0.1136] | 0.3323 |

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
      "mean": 0.029091241624620425,
      "sd": 0.002496694536939255,
      "per_seed": [
        0.027106412251790245,
        0.03189440833197743,
        0.028272904290093593
      ],
      "n": 3
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
    "induction_delta": -0.056916399355287334,
    "induction_ratio": 0.33824019927969456,
    "induction_welch_t": -3.2450902526151437,
    "induction_welch_df": 2.027200166317241,
    "induction_separated_at_95": false,
    "held_ce_a": {
      "mean": 4.835436666666666,
      "sd": 0.006021397955070928,
      "per_seed": [
        4.84104,
        4.8362,
        4.82907
      ],
      "n": 3
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
    "held_ce_delta": 0.11112999999999928,
    "held_ce_welch_t": 16.49631117936882
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
      "mean": 0.32057136253074353,
      "sd": 0.042404738942398656,
      "per_seed": [
        0.3442906697591141,
        0.2716144349839954,
        0.3458089828491211
      ],
      "n": 3
    },
    "induction_b": {
      "mean": 0.19111093591760708,
      "sd": 0.017535229279848782,
      "per_seed": [
        0.1871306313408745,
        0.17590800391303177,
        0.21029417249891508
      ],
      "n": 3
    },
    "induction_delta": 0.12946042661313645,
    "induction_ratio": 1.6774098300107232,
    "induction_welch_t": 4.886579467357243,
    "induction_welch_df": 2.6645644977762633,
    "induction_separated_at_95": true,
    "held_ce_a": {
      "mean": 4.431100000000001,
      "sd": 0.008643633495238195,
      "per_seed": [
        4.42188,
        4.43902,
        4.4324
      ],
      "n": 3
    },
    "held_ce_b": {
      "mean": 4.328596666666667,
      "sd": 0.0032946825846100203,
      "per_seed": [
        4.33237,
        4.32713,
        4.32629
      ],
      "n": 3
    },
    "held_ce_delta": 0.10250333333333383,
    "held_ce_welch_t": 19.193077507208137
  },
  "b_width192_shrink_vs_plain": {
    "label": "CONTROL B - shrink against plain at depth 3 width 192.",
    "a": "shrink d3 w192",
    "b": "vanilla d3 w192",
    "a_geometry": "6x32",
    "b_geometry": "1x192",
    "a_params": 3607104,
    "b_params": 3564096,
    "induction_a": {
      "mean": 0.35565042142514836,
      "sd": 0.09659202379528571,
      "per_seed": [
        0.2515564812554253,
        0.37300794389512826,
        0.44238683912489163
      ],
      "n": 3
    },
    "induction_b": {
      "mean": 0.19111093591760708,
      "sd": 0.017535229279848782,
      "per_seed": [
        0.1871306313408745,
        0.17590800391303177,
        0.21029417249891508
      ],
      "n": 3
    },
    "induction_delta": 0.16453948550754127,
    "induction_ratio": 1.860963213421123,
    "induction_welch_t": 2.903009732740205,
    "induction_welch_df": 2.131682758276093,
    "induction_separated_at_95": false,
    "held_ce_a": {
      "mean": 4.4097599999999995,
      "sd": 0.016925108566859966,
      "per_seed": [
        4.42833,
        4.40575,
        4.3952
      ],
      "n": 3
    },
    "held_ce_b": {
      "mean": 4.328596666666667,
      "sd": 0.0032946825846100203,
      "per_seed": [
        4.33237,
        4.32713,
        4.32629
      ],
      "n": 3
    },
    "held_ce_delta": 0.08116333333333259,
    "held_ce_welch_t": 8.152910071377258
  },
  "b2_width192_geometry_only": {
    "label": "CONTROL B2 - the geometry contrast at FIXED width 192: 8x24 (two dead slots) against 6x32. Isolates slot geometry from width.",
    "a": "slots d3 w192_g8",
    "b": "slots d3 w192",
    "a_geometry": "8x24",
    "b_geometry": "6x32",
    "a_params": 3564096,
    "b_params": 3564096,
    "induction_a": {
      "mean": 0.20502077032018584,
      "sd": 0.014889560312386108,
      "per_seed": [
        0.19141449398464658,
        0.22092615763346296,
        0.20272165934244804
      ],
      "n": 3
    },
    "induction_b": {
      "mean": 0.32057136253074353,
      "sd": 0.042404738942398656,
      "per_seed": [
        0.3442906697591141,
        0.2716144349839954,
        0.3458089828491211
      ],
      "n": 3
    },
    "induction_delta": -0.11555059221055769,
    "induction_ratio": 0.6395479892578486,
    "induction_welch_t": -4.453198691677109,
    "induction_welch_df": 2.4857837093490804,
    "induction_separated_at_95": true,
    "held_ce_a": {
      "mean": 4.492703333333334,
      "sd": 0.0036908309814094086,
      "per_seed": [
        4.48852,
        4.4955,
        4.49409
      ],
      "n": 3
    },
    "held_ce_b": {
      "mean": 4.431100000000001,
      "sd": 0.008643633495238195,
      "per_seed": [
        4.42188,
        4.43902,
        4.4324
      ],
      "n": 3
    },
    "held_ce_delta": 0.06160333333333323,
    "held_ce_welch_t": 11.35270451574639
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
      "mean": 0.029091241624620425,
      "sd": 0.002496694536939255,
      "per_seed": [
        0.027106412251790245,
        0.03189440833197743,
        0.028272904290093593
      ],
      "n": 3
    },
    "ratio_8x16_over_4x32": 0.33824019927969456,
    "welch_t": -3.2450902526151437,
    "held_ce_cost_of_8x16": 0.11112999999999928
  },
  "depth3_exact_geometry_slots_vs_plain": {
    "induction_ratio": 1.6774098300107232,
    "held_ce_delta": 0.10250333333333383,
    "clears_2x_bar": false,
    "below_0.5x_bar": false
  },
  "depth3_exact_geometry_shrink_vs_plain": {
    "induction_ratio": 1.860963213421123,
    "held_ce_delta": 0.08116333333333259,
    "clears_2x_bar": false,
    "below_0.5x_bar": false
  }
}
```
