# The depth ladder (vanilla, V=8192 trained BPE, three seeds a cell)

Every number through ONE code path: `tf_interp3.py`, the same revision that produced the six-architecture slice, gated against `tf_interp2` on a vanilla checkpoint. Depth-1/2 cells that predated that path were re-run through it.

| depth | width | params | held CE (T512) | bits/byte | ladder CE | induction ± sd (floor) | seeds above floor | natural swap | attention first / last | order ratio | interaction |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 32 | 280,608 | 5.5075 ± 0.0077 | 2.1162 | 5.4130 | -0.0058 ± 0.0018 (0.0091) | **0/3** | +0.0231 | 2.03 / 0.29 | 7.1x | 1.74 |
| 1 | 64 | 598,080 | 5.1479 ± 0.0055 | 1.9781 | 5.0477 | -0.0115 ± 0.0025 (0.0096) | **0/3** | +0.0407 | 3.47 / 0.47 | 7.4x | 3.00 |
| 1 | 128 | 1,343,616 | 4.8226 ± 0.0029 | 1.8531 | 4.7234 | -0.0264 ± 0.0019 (0.0093) | **0/3** | +0.0671 | 4.63 / 0.70 | 6.6x | 3.93 |
| 1 | 256 | 3,277,056 | 4.5591 ± 0.0027 | 1.7518 | 4.4592 | -0.0354 ± 0.0015 (0.0092) | **0/3** | +0.0853 | 4.26 / 0.94 | 4.5x | 3.32 |
| 2 | 32 | 299,072 | 5.4127 ± 0.0098 | 2.0798 | 5.3166 | -0.0077 ± 0.0015 (0.0079) | **0/3** | +0.0258 | 4.22 / 0.37 | 11.4x | 3.85 |
| 2 | 64 | 671,872 | 5.0181 ± 0.0047 | 1.9282 | 4.9124 | -0.0140 ± 0.0022 (0.0111) | **0/3** | +0.0552 | 7.67 / 0.62 | 12.4x | 7.05 |
| 2 | 128 | 1,638,656 | 4.6463 ± 0.0075 | 1.7853 | 4.5503 | -0.0034 ± 0.0099 (0.0103) | **0/3** | +0.1032 | 11.63 / 0.94 | 12.4x | 10.69 |
| 2 | 256 | 4,456,960 | 4.3254 ± 0.0013 | 1.6620 | 4.2453 | +0.0938 ± 0.0086 (0.0101) | **3/3** | +0.2407 | 14.87 / 1.23 | 12.1x | 13.64 |
| 3 | 64 | 745,664 | 4.9417 ± 0.0000 | 1.8989 | 4.8425 | +0.0077 ± 0.0000 (0.0109) | **0/1** | +0.0546 | 6.30 / 0.74 | 8.5x | 5.56 |
| 3 | 128 | 1,933,696 | 4.5285 ± 0.0000 | 1.7400 | 4.4500 | +0.0974 ± 0.0000 (0.0078) | **1/1** | +0.1747 | 8.53 / 1.09 | 7.8x | 7.43 |
| 3 | 256 | 5,636,864 | 4.2182 ± 0.0000 | 1.6208 | 4.1435 | +0.1642 ± 0.0000 (0.0156) | **1/1** | +0.2799 | 13.54 / 1.41 | 9.6x | 12.13 |
| 4 | 64 | 819,456 | 4.8817 ± 0.0000 | 1.8758 | 4.7843 | +0.0173 ± 0.0000 (0.0133) | **1/1** | +0.0553 | 6.54 / 0.83 | 7.9x | 5.71 |
| 4 | 128 | 2,228,736 | 4.4601 ± 0.0000 | 1.7138 | 4.3866 | +0.1264 ± 0.0000 (0.0112) | **1/1** | +0.1899 | 9.18 / 1.27 | 7.2x | 7.92 |
| 4 | 256 | 6,816,768 | 4.1436 ± 0.0000 | 1.5921 | 4.0835 | +0.3019 ± 0.0000 (0.0137) | **1/1** | +0.3179 | 13.33 / 1.51 | 8.8x | 11.82 |

## The composition budget, measured causally

Each upstream write deleted from layer l's Q/K/V read ONLY (residual untouched, everything downstream recomputed), KL from the true model in nats/token, [zero, resample]. Seed 0 shown; the per-seed record is in `tf_depth_ladder.json`.

| cell | layer | dominant source | its KL | largest attention-to-attention source | its KL | as a fraction of the dominant MLP |
|---|---|---|---|---|---|---|
| d2 w32 | 1 | M0 | 1.016 | A0 | 1.094e-06 | 1.08e-06 |
| d2 w64 | 1 | M0 | 0.882 | A0 | 2.731e-06 | 3.10e-06 |
| d2 w128 | 1 | M0 | 1.796 | A0 | 2.416e-05 | 1.34e-05 |
| d2 w256 | 1 | M0 | 1.623 | A0 | 2.294e-05 | 1.41e-05 |
| d3 w64 | 1 | M0 | 0.7934 | A0 | 9.016e-07 | 1.14e-06 |
| d3 w64 | 2 | M0 | 0.4829 | A1 | 0.08164 | 1.69e-01 |
| d3 w128 | 1 | M0 | 1.682 | A0 | 2.44e-06 | 1.45e-06 |
| d3 w128 | 2 | M0 | 0.7023 | A1 | 0.1798 | 2.56e-01 |
| d3 w256 | 1 | M0 | 2.123 | A0 | 3.375e-05 | 1.59e-05 |
| d3 w256 | 2 | M0 | 0.6918 | A1 | 0.2668 | 3.86e-01 |
| d4 w64 | 1 | M0 | 0.8399 | A0 | 9.813e-07 | 1.17e-06 |
| d4 w64 | 2 | M0 | 1.048 | A1 | 0.09372 | 8.94e-02 |
| d4 w64 | 3 | M0 | 0.1658 | A2 | 0.05247 | 3.17e-01 |
| d4 w128 | 1 | M0 | 0.9229 | A0 | 1.315e-06 | 1.42e-06 |
| d4 w128 | 2 | M0 | 2.503 | A1 | 0.214 | 8.55e-02 |
| d4 w128 | 3 | M0 | 0.3301 | A1 | 0.07256 | 2.20e-01 |
| d4 w256 | 1 | M0 | 1.966 | A0 | 1.194e-05 | 6.07e-06 |
| d4 w256 | 2 | M0 | 1.01 | A1 | 0.3481 | 3.44e-01 |
| d4 w256 | 3 | M0 | 0.2669 | A1 | 0.07457 | 2.79e-01 |

## Verdicts against the registered predictions

```
{
  "P1": {
    "registered": "depth LOWERS the width threshold by one octave. Concretely: at depth 3, width 128 shows induction above the planted-oracle 3-SE floor at >= 2 of 3 seeds, with a mean score in [+0.015, +0.070]; at depth 4, width 128 does so at 3 of 3 seeds. Width 64 stays below floor at BOTH depths 3 and 4 at >= 2 of 3 seeds.",
    "threshold_is_provisional_at_depth": [
      3,
      4
    ],
    "induction_width_threshold_by_depth": {
      "1": null,
      "2": 256,
      "3": 128,
      "4": 64
    },
    "call": "CONFIRMED IN ITS MAIN CLAUSE AND REFUTED IN ITS LAST: the threshold falls ONE OCTAVE PER LAYER (256 at depth 2, 128 at depth 3, 64 at depth 4). P1 registered the depth-3 octave correctly but also registered that width 64 would stay below floor at depths 3 AND 4; at depth 4 it does not."
  },
  "P2": {
    "max_attention_over_dominant_mlp_by_cell": {
      "d2_w32": 2.194670629456969e-06,
      "d2_w64": 3.0964872818523534e-06,
      "d2_w128": 1.3448340450490483e-05,
      "d2_w256": 1.9269480278870034e-05,
      "d3_w64": 0.16906093749146633,
      "d3_w128": 0.2559576222394559,
      "d3_w256": 0.38559191643193175,
      "d4_w64": 0.3165145800670584,
      "d4_w128": 0.21980894324833708,
      "d4_w256": 0.3444943141659726
    },
    "cells_above_1pc": {
      "d3_w64": 0.16906093749146633,
      "d3_w128": 0.2559576222394559,
      "d3_w256": 0.38559191643193175,
      "d4_w64": 0.3165145800670584,
      "d4_w128": 0.21980894324833708,
      "d4_w256": 0.3444943141659726
    },
    "named_exception_only": false,
    "call": "REFUTED at ['d3_w128', 'd3_w256', 'd3_w64', 'd4_w128', 'd4_w256', 'd4_w64']"
  }
}
```
