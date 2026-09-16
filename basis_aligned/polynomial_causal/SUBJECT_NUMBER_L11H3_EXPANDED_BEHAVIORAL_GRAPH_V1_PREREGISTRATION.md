# Subject-number L11H3 expanded behavioral graph V1

## Purpose

Open the largest remaining node in the passed five-edge behavioral graph,
`upstream_0_7`, without changing its intervention or downstream boundary.

## Frozen design

Replace `upstream_0_7` by three exact subports:

1. embedding recurrence;
2. attention/MLP writes from layers 0–3;
3. attention/MLP writes from layers 4–7.

Keep the passed `mlp_8` and `mlp_10` ports, yielding five ports total. MLP10
uses the parent's float64 closed gauge; the middle band is closed against the
parent `upstream_0_7` group and its correction is measured. Swapping all five
ports must reproduce the parent's three-port joint state at the pre-L11 subject
site.

Run all 32 corners through the native L11–L17 suffix and compute all 31 exact
behavioral Möbius terms in float64. Only the 15 main and pair terms are eligible
for selection. Select eight unscaled terms greedily on the frozen discovery
panel, fit no coefficients, and evaluate every prefix on the three frozen OOD
panels.

## Registered gates

- Instrument: finite; native corner-0 replay at most `1e-5`; expanded-to-parent
  aggregation error at most `1e-10`; middle- and MLP10-gauge correction relative
  L2 at most `1e-5`; absolute and relative Möbius closure at most `1e-10`; exact
  prices/checkpoint; native accuracy at least `0.75` in every panel.
- Six-term graph: relative L2 at most `0.10`, cosine at least `0.99`, and positive
  aligned recovery on every panel.
- Eight-term graph: the same gates on every panel.

Removal, equal-L2 selectivity, and rescue are inherited only if the all-five
corner exactly reproduces the parent joint intervention. Passing yields a
finer native-port graph, not a token-only extraction.
