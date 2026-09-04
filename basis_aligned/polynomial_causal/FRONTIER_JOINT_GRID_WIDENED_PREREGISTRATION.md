# Frontier: widen the joint grid — §2907's optimum sat on two edges. Preregistration

Registered 2026-09-04T12:38Z (the exact string `date -u` returned in its own tool call immediately before this write). Before the run.
Immutable; the rung's frozen-hash check refuses to execute if this file changes.

## SIGN CONVENTION

Frontier L2 is **CE ADDED ABOVE THE REAL MODEL, so LOWER IS BETTER** (§2135; §312: "+2.6735 beating +2.84/+2.93"). A cost is
`L2(arm) − L2(baseline)`, **POSITIVE = WORSE**, so a **negative cost is an improvement**. §2128/§2129/§2133/§2134 RETRACTED;
**§2125 STANDS** — this rescales already-fitted objects; it neither selects nor reorders.

## Why

§2907 swept a 3×3×4 joint grid over the tail, CP and motif scalars and found the optimum at **tail 0.30, CP 0.65, motif 1.25**, beating
the current setting by **+0.0314** (implied L2_F **+2.3058**) and improving **in sample** (−0.3551) as well as fresh (−0.3677).

**Nothing was adopted.** Tail 0.30 and CP 0.65 are both the **top** of their grids, `d_null_the_grid_is_too_narrow` fired, and the
preregistration's rule bound: *"an optimum on a grid edge means the true optimum lies outside and no optimum may be quoted."* §2907's
−0.3677 is therefore a **bound, not an optimum**.

The **direction** is established, and is why widening is the right move rather than stopping: **every block wants less shrinkage once
the others are shrinking too** — tail .30 against the solo .25, CP .65 against the solo .50. That is, across blocks, exactly what
§2899/§2901 found within the tail band, where each layer alone preferred .05 and all eight together preferred ~.20.

Grid moves up on both open edges: **tail {0.25, 0.30, 0.375, 0.45} × CP {0.50, 0.65, 0.80, 0.95} × motif {1.15, 1.25}** = **32 cells in
one pipeline run**. Motif is narrowed to its two best values from §2906/§2907 — it was never at an edge and spending cells on it would
crowd out the two coordinates that were.

## Predictions, each with its worked-example line

- **pred_a — REPRODUCTION GATE, verbatim from §2125 rung 30.** `|L2_F(baseline) − 2.6735| ≤ .05`. *Worked example:* every rung in this
  family reads +2.6735/+2.6736; past .05 and **nothing else here is readable.**
- **pred_b — the anchor cell reproduces §2907.** `|cost(0.30, 0.65, 1.25) − (−0.3677)| ≤ .01`. *Worked example:* ≈ **.000**; a miss ≥
  .03 (`b_null_the_anchor_fails`) puts the whole grid out of comparison with §2907 and nothing else is readable.
- **pred_c — the widened grid beats §2907's bound.** `−0.3677 − cost(best cell) ≥ +.01`. *Worked example:* if the trend continues the
  optimum sits near tail .375 / CP .80 and reads ≈ **−0.39**, a gain of ≈ **+.02**; if §2907's edge cell is in fact the optimum this
  reads ≤ 0 (`c_null_the_edge_was_the_optimum`) — which would be a clean close, and would mean §2907's number can be adopted after all
  once pred_d confirms interiority.
- **pred_d — the optimum is interior in all three coordinates.** *Worked example:* the same discipline as §2907 and §2905. **If it fails
  again, no optimum is quoted again** — and a third widening would need justifying rather than assuming, since a monotone march to the
  edge would suggest the scale parametrisation itself is wrong.
- **pred_e — the optimum improves in sample too.** `cost_fit(best cell) < 0`. *Worked example:* §2907's did (−0.3551); a fresh-only gain
  is the overfitting signature (§2895) and **must not be adopted**.

## Nulls

- `b_null_the_anchor_fails` (≥ .03).
- `c_null_the_edge_was_the_optimum` (≤ 0): §2907's cell stands as the optimum, pending interiority.
- `d_null_the_grid_is_too_narrow`: **no optimum quoted**, for the second time.

**Adoption rule, stated in advance:** a configuration beyond §2904's adopted T+C may be entered **only if pred_a, pred_b, pred_d and
pred_e all hold** — identical to §2907's rule, which is what kept its edge result unadopted.

## Price

**1 full frontier pipeline run, ≤ 500 GPU-seconds** (§2907 measured 214.3 s for 36 cells; 32 here), 0 backwards, 0 fitted parameters
beyond the pipeline's own. The parent `ops/frontier_fisher8.py` is **unmodified**. It is **not forward-instrumented**, so the receipt
reports `gpu_forwards: 0` with `forwards_instrumented: false` and `pipeline_runs: 1`. Receipt:
`frontier_joint_grid_widened_results.json`, read with `price` in the same command the ledger section is written from, in the canonical
`Price: N GPU forwards, X GPU-seconds` / `Results: <file>.json` form (§2853, §2858), under a filename no other section cites (§2876).
