# Frontier: settle the joint grid — all three axes at three points around the optimum. Preregistration

Registered 2026-09-04T12:59Z (the exact string `date -u` returned in its own tool call immediately before this write). Before the run.
Immutable; the rung's frozen-hash check refuses to execute if this file changes.

## SIGN CONVENTION

Frontier L2 is **CE ADDED ABOVE THE REAL MODEL, so LOWER IS BETTER** (§2135; §312: "+2.6735 beating +2.84/+2.93"). A cost is
`L2(arm) − L2(baseline)`, **POSITIVE = WORSE**, so a **negative cost is an improvement**. §2128/§2129/§2133/§2134 RETRACTED;
**§2125 STANDS** — this rescales already-fitted objects; it neither selects nor reorders.

## Why

The joint optimisation has converged in value and **failed interiority twice, both times for reasons about the grid rather than the
frontier**:

| rung | grid | best cell | cost | pred_d |
|---|---|---|---|---|
| §2907 | 3×3×4 | t .30 / c .65 / m 1.25 | −0.3677 | **FAILED** — tail and CP on the top edge |
| §2909 | 4×4×2 | t .30 / c .80 / m 1.25 | −0.3736 | **FAILED** — motif at the edge of a **2-point axis I narrowed myself**, which guarantees an edge optimum and could not have passed |

Value has essentially converged: **+0.0314** then **+0.0059**. What has never been shown **within a single rung** is that the optimum is
interior in all three coordinates — and that is the only thing standing between the measured **−0.3736 / +2.2999** and adoption under
the rule that has governed all of these rungs.

This grid is built so the predicate can pass or fail **on the frontier rather than on my grid design**:
**tail {0.25, 0.30, 0.35} × CP {0.65, 0.80, 0.95} × motif {1.15, 1.25, 1.35}** — three points on every axis, centred on §2909's
optimum, **27 cells in one pipeline run**. If the optimum is where §2907 and §2909 put it, it lands in the middle of all three.

## Predictions, each with its worked-example line

- **pred_a — REPRODUCTION GATE, verbatim from §2125 rung 30.** `|L2_F(baseline) − 2.6735| ≤ .05`. *Worked example:* every rung in this
  family reads +2.6735/+2.6736; past .05 and **nothing else here is readable.**
- **pred_b — the anchor cell reproduces §2909.** `|cost(0.30, 0.80, 1.25) − (−0.3736)| ≤ .01`. *Worked example:* ≈ **.000**; a miss ≥
  .03 (`b_null_the_anchor_fails`) puts the grid out of comparison and nothing else is readable.
- **pred_c — the settled grid does not lose ground.** `−0.3736 − cost(best cell) ≥ 0`. *Worked example:* the two previous steps gained
  +0.0314 and +0.0059, so this one should gain **≈ 0 to +0.005** — convergence, not improvement. **The bar is deliberately 0, not .01:**
  this rung exists to establish interiority, and demanding a further gain would make a converged optimisation look like a failure.
- **pred_d — the optimum is interior in all three coordinates.** *Worked example:* if the optimum is (.30, .80, 1.25) it sits in the
  middle of every axis and this passes. **If it fails a third time, the scale parametrisation itself is suspect** — a monotone march to
  the edge across three independently-centred grids would say the knob is not the right coordinate — and that, not a fourth widening,
  is what the next rung should address.
- **pred_e — the optimum improves in sample too.** `cost_fit(best cell) < 0`. *Worked example:* §2907 and §2909 both did (−0.3551,
  −0.3444); a fresh-only gain is the overfitting signature (§2895) and **must not be adopted**.

## Nulls

- `b_null_the_anchor_fails` (≥ .03).
- `c_null_the_current_setting_is_already_jointly_optimal` (≤ 0) — expected and fine; convergence is the point.
- `d_null_the_grid_is_too_narrow` — **third failure would indict the parametrisation, not the grid.**

**Adoption rule, stated in advance:** the configuration may be entered as a result **only if pred_a, pred_b, pred_d and pred_e all
hold** — identical to §2907's and §2909's rule, which is what kept both unadopted. **pred_c is deliberately not in the list**: it
measures whether there is anything further to gain, not whether an adoption would be sound.

## Price

**1 full frontier pipeline run, ≤ 500 GPU-seconds** (§2907 214.3 s for 36 cells, §2909 201.7 s for 32; 27 here), 0 backwards, 0 fitted
parameters beyond the pipeline's own. The parent `ops/frontier_fisher8.py` is **unmodified**. It is **not forward-instrumented**, so the
receipt reports `gpu_forwards: 0` with `forwards_instrumented: false` and `pipeline_runs: 1`. Receipt:
`frontier_joint_grid_settled_results.json`, read with `price` in the same command the ledger section is written from, in the canonical
`Price: N GPU forwards, X GPU-seconds` / `Results: <file>.json` form (§2853, §2858), under a filename no other section cites (§2876).
