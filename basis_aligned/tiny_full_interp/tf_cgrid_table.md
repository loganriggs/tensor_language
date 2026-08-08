# Compressibility across the grid

One scalar per cell, FINDING 12 §7b's own construction: for every
point on the description frontier, how many bits the SAME weights
need under naive uniform quantisation + entropy coding to reach the
same held CE -- the median of that ratio over the frontier.  The
denominator is interpolated inside ONE family, so a set-inclusion
artifact cannot make it look better or worse than it is.

`R` is the primary number: best description of any kind against the
STRONGER of the two naive scale groupings (per-tensor as well as
per-row scales).  The per-row-only denominator `R(per-row)` is the
literal FINDING-12 definition and is quoted beside it, because a
32-bit-per-row scale is 1.0 bits/weight of pure overhead at width 32
and 0.125 at width 256 -- a width trend in `R(per-row)` alone would
be nothing but that.  `R(structure)` restricts the numerator to
descriptions made out of an INTERPRETATION -- low rank, row
prototypes, subspace codebooks, exact anchor rows, and each of those
plus an honestly coded remainder -- with recodings excluded.

| depth | width | seed | params | emb share | held CE | **R** | R range | R (per-row denom) | R (KL not CE) | R (embedding only) | **R (structure)** | R (structure, per-row) | R (structure, embedding only) | naive per-row overhead share |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 32 | 0 | 280,608 | 93% | 5.4249 | **1.162** | 0.540–1.263 | 1.211 | 1.151 | 1.139 | **0.826** | 0.876 | 0.907 | 20% |
| 1 | 64 | 0 | 598,080 | 88% | 5.0384 | **1.132** | 0.485–1.181 | 1.148 | 1.093 | 1.093 | **0.834** | 0.866 | 0.884 | 12% |
| 1 | 128 | 0 | 1,343,616 | 78% | 4.7114 | **1.051** | 0.879–1.144 | 1.051 | 1.067 | 1.055 | **0.750** | 0.757 | 0.850 | 7% |
| 1 | 256 | 0 | 3,277,056 | 64% | 4.4320 | **1.049** | 0.814–1.114 | 1.050 | 1.048 | 1.036 | **0.763** | 0.769 | 0.873 | 4% |
| 2 | 32 | 0 | 299,072 | 88% | 5.3168 | **1.085** | 0.398–1.203 | 1.141 | 1.082 | 1.086 | **0.847** | 0.847 | 0.914 | 21% |
| 2 | 64 | 0 | 671,872 | 78% | 4.9135 | **1.035** | 0.438–1.140 | 1.109 | 1.076 | 1.041 | **0.837** | 0.860 | 0.912 | 12% |
| 2 | 128 | 0 | 1,638,656 | 64% | 4.5386 | **1.040** | 0.760–1.101 | 1.054 | 1.053 | 1.040 | **0.775** | 0.809 | 0.898 | 7% |
| 2 | 256 | 0 | 4,456,960 | 47% | 4.2090 | **0.987** | 0.741–1.057 | 1.039 | 0.977 | 1.022 | **0.768** | 0.829 | 0.922 | 4% |
| 3 | 64 | 0 | 745,664 | 70% | 4.8320 | **1.083** | 0.691–1.159 | 1.119 | 1.070 | 1.050 | **0.842** | 0.869 | 0.929 | 12% |
| 3 | 128 | 0 | 1,933,696 | 54% | 4.4262 | **1.030** | 0.721–1.104 | 1.066 | 1.047 | 1.044 | **0.781** | 0.802 | 0.919 | 7% |
| 3 | 256 | 0 | 5,636,864 | 37% | 4.1102 | **1.008** | 0.728–1.053 | 1.039 | 0.989 | 1.017 | **0.827** | 0.882 | 0.944 | 4% |
| 4 | 64 | 0 | 819,456 | 64% | 4.7742 | **1.067** | 0.630–1.104 | 1.090 | 1.071 | 1.046 | **0.869** | 0.870 | 0.939 | 12% |
| 4 | 128 | 0 | 2,228,736 | 47% | 4.3550 | **1.024** | 0.664–1.096 | 1.083 | 1.035 | 1.035 | **0.838** | 0.870 | 0.940 | 7% |

## Trend

```
{
  "n_cells": 13,
  "n_cells_seed0": 13,
  "trend_R_vs_params": {
    "slope_per_efold": -0.041825527077835496,
    "se": 0.008511224334686533,
    "intercept": 1.6446659815425322,
    "n": 13,
    "t": -4.914161045829832
  },
  "trend_R_vs_params_depth12": {
    "slope_per_efold": -0.0431599283678765,
    "se": 0.013662649098423924,
    "intercept": 1.6642644919610894,
    "n": 8,
    "t": -3.158972177134761
  },
  "trend_R_perrow_vs_params": {
    "slope_per_efold": -0.04796060389192985,
    "se": 0.007352441357120929,
    "intercept": 1.7651105814183954,
    "n": 13,
    "t": -6.523085538857025
  },
  "trend_R_perrow_vs_params_depth12": {
    "slope_per_efold": -0.05443089493188165,
    "se": 0.010572992429659162,
    "intercept": 1.8528620579704622,
    "n": 8,
    "t": -5.148106867001353
  },
  "trend_R_kl_vs_params": {
    "slope_per_efold": -0.041257160752910464,
    "se": 0.006018495121578987,
    "intercept": 1.63705567756438,
    "n": 13,
    "t": -6.855062589481075
  },
  "trend_R_kl_vs_params_depth12": {
    "slope_per_efold": -0.040832798019440404,
    "se": 0.00967246370886504,
    "intercept": 1.6327403003890788,
    "n": 8,
    "t": -4.221550914894225
  },
  "trend_R_struct_vs_params": {
    "slope_per_efold": -0.021284581623514938,
    "se": 0.010352175620682236,
    "intercept": 1.1106797785954483,
    "n": 13,
    "t": -2.0560491246874952
  },
  "trend_R_struct_vs_params_depth12": {
    "slope_per_efold": -0.03226400338113422,
    "se": 0.008457077811309489,
    "intercept": 1.2460905393044626,
    "n": 8,
    "t": -3.815029742068612
  },
  "trend_R_struct_perrow_vs_params": {
    "slope_per_efold": -0.014082870250313764,
    "se": 0.012513827646099594,
    "intercept": 1.0364952277393389,
    "n": 13,
    "t": -1.1253847063095217
  },
  "trend_R_struct_perrow_vs_params_depth12": {
    "slope_per_efold": -0.028872032294056757,
    "se": 0.013207842764111865,
    "intercept": 1.2258635268109999,
    "n": 8,
    "t": -2.1859763785580015
  },
  "trend_R_embonly_vs_params": {
    "slope_per_efold": -0.028975001816781445,
    "se": 0.005780322659147683,
    "intercept": 1.4606300850093268,
    "n": 13,
    "t": -5.012696267210428
  },
  "trend_R_embonly_vs_params_depth12": {
    "slope_per_efold": -0.03175456696335119,
    "se": 0.008290271459745669,
    "intercept": 1.5030445261374397,
    "n": 8,
    "t": -3.8303410349756346
  },
  "trend_R_embonly_struct_vs_params": {
    "slope_per_efold": 0.0036356423337322954,
    "se": 0.008642343412756626,
    "intercept": 0.8590291413757881,
    "n": 13,
    "t": 0.4206778370280757
  },
  "trend_R_embonly_struct_vs_params_depth12": {
    "slope_per_efold": -0.004811738068907487,
    "se": 0.00944800944180317,
    "intercept": 0.9616122814778653,
    "n": 8,
    "t": -0.509285908163652
  },
  "trend_ratio_vs_params": {
    "slope_per_efold": -0.03592295081757731,
    "se": 0.008049228961356177,
    "intercept": 1.5797842630233654,
    "n": 13,
    "t": -4.462905824898391
  },
  "trend_ratio_vs_params_depth12": {
    "slope_per_efold": -0.04098856313884757,
    "se": 0.01078282172772658,
    "intercept": 1.6474583068916522,
    "n": 8,
    "t": -3.8012835762137267
  },
  "trend_ratio_strong_vs_params": {
    "slope_per_efold": -0.030101512676761623,
    "se": 0.006169687009854808,
    "intercept": 1.466997541686904,
    "n": 13,
    "t": -4.878936748117147
  },
  "trend_ratio_strong_vs_params_depth12": {
    "slope_per_efold": -0.030626234097703462,
    "se": 0.010086066039227683,
    "intercept": 1.4747992634740597,
    "n": 8,
    "t": -3.0364895469243423
  },
  "trend_ratio_struct_vs_params": {
    "slope_per_efold": -0.005774324595816307,
    "se": 0.008723223658706905,
    "intercept": 0.9405742076043452,
    "n": 13,
    "t": -0.6619484747536862
  },
  "trend_ratio_struct_vs_params_depth12": {
    "slope_per_efold": -0.014947295469122915,
    "se": 0.007727953559975651,
    "intercept": 1.0528511292940992,
    "n": 8,
    "t": -1.934185467487464
  },
  "seed_spread": {
    "R": {
      "mean_sd_over_seeds": null,
      "n_cells_with_seeds": 0
    },
    "R_struct": {
      "mean_sd_over_seeds": null,
      "n_cells_with_seeds": 0
    },
    "ratio_strong": {
      "mean_sd_over_seeds": null,
      "n_cells_with_seeds": 0
    }
  },
  "verdict_call": "NOT GROWING, AND SIGNIFICANTLY NEGATIVE -- P5's falsifier (growth) is rejected with a wide margin; P5's letter (\"indistinguishable from zero\") is NOT met, because the slope is significantly below zero (t = -4.91) though small in magnitude; P5's scientific claim -- the negative is a property of the family, not of the smallest model -- is confirmed in the STRONGER direction",
  "verdict": {
    "primary_scalar": "held-CE frontier ratio, best description of any kind against the STRONGER naive quantiser, median over that cell's frontier points",
    "slope_per_efold_of_parameters": -0.041825527077835496,
    "se": 0.008511224334686533,
    "registered_P5": "FLAT: |slope| < 0.05 per e-fold",
    "call": "NOT GROWING, AND SIGNIFICANTLY NEGATIVE -- P5's falsifier (growth) is rejected with a wide margin; P5's letter (\"indistinguishable from zero\") is NOT met, because the slope is significantly below zero (t = -4.91) though small in magnitude; P5's scientific claim -- the negative is a property of the family, not of the smallest model -- is confirmed in the STRONGER direction",
    "t": -4.914161045829832,
    "range_over_cells": [
      0.9868283826750522,
      1.1621252455753504
    ]
  },
  "P7_structure_only_below_1": {
    "cells_below_1": 13,
    "cells_measured": 13,
    "values": {
      "d1_w128": 0.749910557979726,
      "d1_w256": 0.7629291438612966,
      "d1_w32": 0.8255201074250839,
      "d1_w64": 0.8343283298358175,
      "d2_w128": 0.7750906118623064,
      "d2_w256": 0.7679378348168968,
      "d2_w32": 0.8472448893988911,
      "d2_w64": 0.8371978411176237,
      "d3_w128": 0.7813388039287679,
      "d3_w256": 0.8269128612454297,
      "d3_w64": 0.8418118760236459,
      "d4_w128": 0.8381600409282594,
      "d4_w64": 0.8685657196180343
    },
    "call": "CONFIRMED"
  }
}
```
