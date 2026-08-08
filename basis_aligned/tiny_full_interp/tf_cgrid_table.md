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
| 1 | 32 | 1 | 280,608 | 93% | 5.4351 | **1.114** | 0.453–1.203 | 1.177 | 1.144 | 1.115 | **0.836** | 0.836 | 0.886 | 21% |
| 1 | 32 | 2 | 280,608 | 93% | 5.4254 | **1.105** | 0.419–1.197 | 1.171 | 1.100 | 1.107 | **0.811** | 0.826 | 0.889 | 21% |
| 1 | 64 | 0 | 598,080 | 88% | 5.0384 | **1.132** | 0.485–1.181 | 1.148 | 1.093 | 1.093 | **0.834** | 0.866 | 0.884 | 12% |
| 1 | 128 | 0 | 1,343,616 | 78% | 4.7114 | **1.051** | 0.879–1.144 | 1.051 | 1.067 | 1.055 | **0.750** | 0.757 | 0.850 | 7% |
| 1 | 256 | 0 | 3,277,056 | 64% | 4.4320 | **1.049** | 0.814–1.114 | 1.050 | 1.048 | 1.036 | **0.763** | 0.769 | 0.873 | 4% |
| 1 | 256 | 1 | 3,277,056 | 64% | 4.4307 | **0.993** | 0.873–1.051 | 1.060 | 0.960 | 1.040 | **0.701** | 0.718 | 0.865 | 4% |
| 1 | 256 | 2 | 3,277,056 | 64% | 4.4249 | **1.050** | 0.865–1.085 | 1.056 | 1.030 | 1.042 | **0.701** | 0.703 | 0.862 | 4% |
| 2 | 32 | 0 | 299,072 | 88% | 5.3168 | **1.085** | 0.398–1.203 | 1.141 | 1.082 | 1.086 | **0.847** | 0.847 | 0.914 | 21% |
| 2 | 32 | 1 | 299,072 | 88% | 5.3287 | **1.152** | 0.399–1.212 | 1.159 | 1.126 | 1.094 | **0.816** | 0.833 | 0.903 | 21% |
| 2 | 32 | 2 | 299,072 | 88% | 5.3356 | **1.130** | 0.414–1.172 | 1.154 | 1.128 | 1.095 | **0.784** | 0.820 | 0.907 | 20% |
| 2 | 64 | 0 | 671,872 | 78% | 4.9135 | **1.035** | 0.438–1.140 | 1.109 | 1.076 | 1.041 | **0.837** | 0.860 | 0.912 | 12% |
| 2 | 128 | 0 | 1,638,656 | 64% | 4.5386 | **1.040** | 0.760–1.101 | 1.054 | 1.053 | 1.040 | **0.775** | 0.809 | 0.898 | 7% |
| 2 | 256 | 0 | 4,456,960 | 47% | 4.2090 | **0.987** | 0.741–1.057 | 1.039 | 0.977 | 1.022 | **0.768** | 0.829 | 0.922 | 4% |
| 2 | 256 | 1 | 4,456,960 | 47% | 4.2185 | **0.993** | 0.751–1.087 | 1.042 | 0.973 | 1.022 | **0.778** | 0.860 | 0.920 | 4% |
| 2 | 256 | 2 | 4,456,960 | 47% | 4.2047 | **1.010** | 0.812–1.119 | 1.055 | 0.991 | 1.025 | **0.835** | 0.851 | 0.921 | 4% |
| 3 | 64 | 0 | 745,664 | 70% | 4.8320 | **1.083** | 0.691–1.159 | 1.119 | 1.070 | 1.050 | **0.842** | 0.869 | 0.929 | 12% |
| 3 | 64 | 1 | 745,664 | 70% | 4.8396 | **1.043** | 0.431–1.114 | 1.088 | 1.061 | 1.043 | **0.822** | 0.870 | 0.923 | 12% |
| 3 | 128 | 0 | 1,933,696 | 54% | 4.4262 | **1.030** | 0.721–1.104 | 1.066 | 1.047 | 1.044 | **0.781** | 0.802 | 0.919 | 7% |
| 3 | 256 | 0 | 5,636,864 | 37% | 4.1102 | **1.008** | 0.728–1.053 | 1.039 | 0.989 | 1.017 | **0.827** | 0.882 | 0.944 | 4% |
| 3 | 256 | 1 | 5,636,864 | 37% | 4.1072 | **0.964** | 0.733–1.039 | 1.028 | 0.963 | 1.016 | **0.803** | 0.862 | 0.940 | 4% |
| 4 | 64 | 0 | 819,456 | 64% | 4.7742 | **1.067** | 0.630–1.104 | 1.090 | 1.071 | 1.046 | **0.869** | 0.870 | 0.939 | 12% |
| 4 | 128 | 0 | 2,228,736 | 47% | 4.3550 | **1.024** | 0.664–1.096 | 1.083 | 1.035 | 1.035 | **0.838** | 0.870 | 0.940 | 7% |
| 4 | 256 | 0 | 6,816,768 | 31% | 4.0513 | **1.004** | 0.678–1.039 | 1.034 | 1.009 | 1.014 | **0.882** | 0.911 | 0.958 | 4% |

## Trend

```
{
  "n_cells": 24,
  "n_cells_seed0": 14,
  "trend_R_vs_params": {
    "slope_per_efold": -0.03986687019224658,
    "se": 0.007394965176898547,
    "intercept": 1.6181651060684554,
    "n": 14,
    "t": -5.391082883904366
  },
  "trend_R_vs_params_depth12": {
    "slope_per_efold": -0.0431599283678765,
    "se": 0.013662649098423924,
    "intercept": 1.6642644919610894,
    "n": 8,
    "t": -3.158972177134761
  },
  "trend_R_perrow_vs_params": {
    "slope_per_efold": -0.04523789865286604,
    "se": 0.006508614376825442,
    "intercept": 1.7282720350533174,
    "n": 14,
    "t": -6.950465342353052
  },
  "trend_R_perrow_vs_params_depth12": {
    "slope_per_efold": -0.05443089493188165,
    "se": 0.010572992429659162,
    "intercept": 1.8528620579704622,
    "n": 8,
    "t": -5.148106867001353
  },
  "trend_R_kl_vs_params": {
    "slope_per_efold": -0.03881440322453762,
    "se": 0.00535970216522521,
    "intercept": 1.6040048591443343,
    "n": 14,
    "t": -7.241895543445122
  },
  "trend_R_kl_vs_params_depth12": {
    "slope_per_efold": -0.040832798019440404,
    "se": 0.00967246370886504,
    "intercept": 1.6327403003890788,
    "n": 8,
    "t": -4.221550914894225
  },
  "trend_R_struct_vs_params": {
    "slope_per_efold": -0.009018794265034012,
    "se": 0.011416096443765598,
    "intercept": 0.9447221237503175,
    "n": 14,
    "t": -0.7900068389803445
  },
  "trend_R_struct_vs_params_depth12": {
    "slope_per_efold": -0.03226400338113422,
    "se": 0.008457077811309489,
    "intercept": 1.2460905393044626,
    "n": 8,
    "t": -3.815029742068612
  },
  "trend_R_struct_perrow_vs_params": {
    "slope_per_efold": -0.0030138991875920773,
    "se": 0.01253843508074168,
    "intercept": 0.8867306487987323,
    "n": 14,
    "t": -0.24037283506147067
  },
  "trend_R_struct_perrow_vs_params_depth12": {
    "slope_per_efold": -0.028872032294056757,
    "se": 0.013207842764111865,
    "intercept": 1.2258635268109999,
    "n": 8,
    "t": -2.1859763785580015
  },
  "trend_R_embonly_vs_params": {
    "slope_per_efold": -0.027872754668629908,
    "se": 0.005003342633139529,
    "intercept": 1.4457165416369573,
    "n": 14,
    "t": -5.570826687745775
  },
  "trend_R_embonly_vs_params_depth12": {
    "slope_per_efold": -0.03175456696335119,
    "se": 0.008290271459745669,
    "intercept": 1.5030445261374397,
    "n": 8,
    "t": -3.8303410349756346
  },
  "trend_R_embonly_struct_vs_params": {
    "slope_per_efold": 0.008478505750542667,
    "se": 0.007939778444246523,
    "intercept": 0.7935045866203676,
    "n": 14,
    "t": 1.0678516800033038
  },
  "trend_R_embonly_struct_vs_params_depth12": {
    "slope_per_efold": -0.004811738068907487,
    "se": 0.00944800944180317,
    "intercept": 0.9616122814778653,
    "n": 8,
    "t": -0.509285908163652
  },
  "trend_ratio_vs_params": {
    "slope_per_efold": -0.0317011022765993,
    "se": 0.007336463425629343,
    "intercept": 1.5226621181096431,
    "n": 14,
    "t": -4.3210332332406995
  },
  "trend_ratio_vs_params_depth12": {
    "slope_per_efold": -0.04098856313884757,
    "se": 0.01078282172772658,
    "intercept": 1.6474583068916522,
    "n": 8,
    "t": -3.8012835762137267
  },
  "trend_ratio_strong_vs_params": {
    "slope_per_efold": -0.028242984818241154,
    "se": 0.005406130756044246,
    "intercept": 1.4418514246175305,
    "n": 14,
    "t": -5.224251149801454
  },
  "trend_ratio_strong_vs_params_depth12": {
    "slope_per_efold": -0.030626234097703462,
    "se": 0.010086066039227683,
    "intercept": 1.4747992634740597,
    "n": 8,
    "t": -3.0364895469243423
  },
  "trend_ratio_struct_vs_params": {
    "slope_per_efold": 0.0021024199785256215,
    "se": 0.008789182775264507,
    "intercept": 0.834000856936558,
    "n": 14,
    "t": 0.23920539966952162
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
      "mean_sd_over_seeds": 0.028092046911194078,
      "n_cells_with_seeds": 6
    },
    "R_struct": {
      "mean_sd_over_seeds": 0.024475623406869013,
      "n_cells_with_seeds": 6
    },
    "ratio_strong": {
      "mean_sd_over_seeds": 0.015615628265508207,
      "n_cells_with_seeds": 6
    }
  },
  "verdict_call": "NOT GROWING, AND SIGNIFICANTLY NEGATIVE -- P5's falsifier (growth) is rejected with a wide margin; P5's letter (\"indistinguishable from zero\") is NOT met, because the slope is significantly below zero (t = -5.39) though small in magnitude; P5's scientific claim -- the negative is a property of the family, not of the smallest model -- is confirmed in the STRONGER direction",
  "verdict": {
    "primary_scalar": "held-CE frontier ratio, best description of any kind against the STRONGER naive quantiser, median over that cell's frontier points",
    "slope_per_efold_of_parameters": -0.03986687019224658,
    "se": 0.007394965176898547,
    "registered_P5": "FLAT: |slope| < 0.05 per e-fold",
    "call": "NOT GROWING, AND SIGNIFICANTLY NEGATIVE -- P5's falsifier (growth) is rejected with a wide margin; P5's letter (\"indistinguishable from zero\") is NOT met, because the slope is significantly below zero (t = -5.39) though small in magnitude; P5's scientific claim -- the negative is a property of the family, not of the smallest model -- is confirmed in the STRONGER direction",
    "t": -5.391082883904366,
    "range_over_cells": [
      0.9868283826750522,
      1.1621252455753504
    ]
  },
  "P7_structure_only_below_1": {
    "cells_below_1": 14,
    "cells_measured": 14,
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
      "d4_w256": 0.8824008691852117,
      "d4_w64": 0.8685657196180343
    },
    "call": "CONFIRMED"
  }
}
```
