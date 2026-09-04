# Frontier: jointly optimise the three scalars on a 3×3×4 grid — preregistration

Registered 2026-09-04T12:29Z (the exact string `date -u` returned in its own tool call immediately before this write). Before the run.
Immutable; the rung's frozen-hash check refuses to execute if this file changes.

## SIGN CONVENTION

Frontier L2 is **CE ADDED ABOVE THE REAL MODEL, so LOWER IS BETTER** (§2135; §312: "+2.6735 beating +2.84/+2.93"). A cost is
`L2(arm) − L2(baseline)`, **POSITIVE = WORSE**, so a **negative cost is an improvement**. §2128/§2129/§2133/§2134 RETRACTED;
**§2125 STANDS** — this rescales already-fitted objects; it neither selects nor reorders, so it is none of §2118/§2125/§2131.

## Why

Three corrections exist, each measured **alone or in pairs, never on a joint grid**:

| | § | operation | status |
|---|---|---|---|
| TAIL | §2896 | `LW` × 0.25 | adopted; §2903 closed per-layer tuning — one scalar is the whole effect |
| CP | §2902 | `Dk` × 0.50 | adopted |
| MOTIF | §2905/§2906 | `ALPHA` × 1.25 | bracketed, but **91% redundant** with T+C (increment +0.0149) |
| **T+C** | §2904 | both | **adopted, −0.3213 fresh / −0.2899 in sample → frontier +2.3522** |

**The blocks are subadditive, so the individual optima need not be the joint optimum.** §2906 found the motif block's entire standalone
value (−0.1604) almost entirely absorbed by T+C; §2904 measured pairwise interactions of +0.0149, +0.0602 and +0.1234. When corrections
overlap, each one's best *solo* setting typically **overshoots in company** — which is exactly what §2899/§2901 found *within* the tail
band (each layer alone prefers .05; all eight together prefer ~.20). **Nothing has tested that across blocks.**

Grid: tail {0.20, 0.25, 0.30} × CP {0.35, 0.50, 0.65} × motif {1.00, 1.15, 1.25, 1.40} = **36 cells in one pipeline run**, built with
`ops/frontier_evalarms.factorial_arms` — the 12:07Z ops action, whose measurement showed 40-arm rungs cost **5.8 GPU-s per arm** against
**14.8** for 8-arm rungs and need far fewer authoring cycles. **This is its first use.**

The cell **(0.25, 0.50, 1.25)** is the current best-measured configuration, so pred_b re-anchors the whole grid to §2906.

## Predictions, each with its worked-example line

- **pred_a — REPRODUCTION GATE, verbatim from §2125 rung 30.** `|L2_F(baseline) − 2.6735| ≤ .05`. *Worked example:* every rung in this
  family reads +2.6735/+2.6736; past .05 and **nothing else here is readable.**
- **pred_b — the current cell reproduces §2906.** `|cost(0.25, 0.50, 1.25) − (−0.3363)| ≤ .01`. *Worked example:* ≈ **.000**; a miss
  ≥ .03 (`b_null_the_anchor_fails`) puts the whole grid out of comparison and nothing else is readable.
- **pred_c — the joint optimum beats the current setting.** `cost(current) − cost(best cell) ≥ +.01` nats. *Worked example:* if the
  solo settings overshoot in company, the joint optimum sits at gentler scales — say (0.30, 0.65, 1.15) — and reads ≈ **−0.36**, a gain
  of ≈ **+.02**; if the current setting is already jointly optimal this reads ≤ 0
  (`c_null_the_current_setting_is_already_jointly_optimal`), which would be a clean close: **three independently-chosen scalars happen to
  be jointly optimal**, and no further scale tuning is warranted anywhere.
- **pred_d — the joint optimum is interior in all three coordinates.** *Worked example:* an optimum on a grid edge means the true
  optimum lies outside and **no optimum may be quoted** — `d_null_the_grid_is_too_narrow` — the same discipline §2905 needed when 1.25
  sat at its grid edge and §2906 had to discharge it by combining grids.
- **pred_e — the optimum improves in sample too.** `cost_fit(best cell) < 0`. *Worked example:* the tail and CP corrections both improve
  on both windows (objective mismatch); the front tables did not (overfitting, §2895). A joint optimum that only improves fresh L2 would
  be the overfitting signature and **must not be adopted**, which is why this is a predicate and not a footnote.

## Nulls

- `b_null_the_anchor_fails` (≥ .03).
- `c_null_the_current_setting_is_already_jointly_optimal` (≤ 0) — a genuine close, not a failure.
- `d_null_the_grid_is_too_narrow` — no optimum quoted.

**Adoption rule, stated in advance:** a configuration beyond §2904's adopted T+C may be entered **only if pred_a, pred_b, pred_d and
pred_e all hold** — baseline reproduces, the grid re-anchors, the optimum is interior, **and it improves in sample**. Note pred_c is
deliberately **not** in that list: it decides whether there is anything to adopt, while the other four decide whether an adoption would
be sound.

## Price

**1 full frontier pipeline run, ≤ 500 GPU-seconds** (37 arms at ~90 s fit + ~3.5 s per arm ≈ 220 s expected), 0 backwards, 0 fitted
parameters beyond the pipeline's own. The parent `ops/frontier_fisher8.py` is **unmodified**. It is **not forward-instrumented**, so the
receipt reports `gpu_forwards: 0` with `forwards_instrumented: false` and `pipeline_runs: 1`, and the ledger's `Price:` line says so —
the count is absent, not zero. Receipt: `frontier_joint_three_scalar_results.json`, read with `price` in the same command the ledger
section is written from, in the canonical `Price: N GPU forwards, X GPU-seconds` / `Results: <file>.json` form (§2853, §2858), under a
filename no other section cites (§2876).
