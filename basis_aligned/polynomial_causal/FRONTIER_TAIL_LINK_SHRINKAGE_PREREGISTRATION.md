# Frontier: is the ridge solution the end-to-end optimum? A one-parameter test — preregistration

Registered 2026-09-04T10:26Z (the exact string `date -u` returned in its own tool call immediately before this write). Before the run.
Immutable; the rung's frozen-hash check refuses to execute if this file changes.

## SIGN CONVENTION

Frontier L2 is **CE ADDED ABOVE THE REAL MODEL, so LOWER IS BETTER** (§2135; §312: "+2.6735 beating +2.84/+2.93"). A cost is
`L2(arm) − L2(baseline)`, **POSITIVE = WORSE**. §2128/§2129/§2133/§2134 RETRACTED; **§2125 STANDS** — norm-2304 at 2.6735.
Both `L2_F` (120 fresh documents) and `L2_C` (**the window the maps are fitted on**, `CA,CB = 300,512`) are reported per arm, because
§2890 showed the in-sample curve is the decisive one.

## Why

§2890 established, **from the fitting window itself**, that the frontier's components are fitted to a *local* objective and scored
*end-to-end*, and that the two disagree. The tail dictionaries' link maps come from ridge regression minimising per-layer reconstruction
of that layer's attention output; the frontier is scored by cross-entropy at the output. A rank-1 truncation of the ridge solution beats
it by **−0.0062 nats in sample** and −0.0294 out of sample, and the fit-window and fresh-window rank curves agree at **Pearson .962**
with the same argmin and argmax. Overfitting is refuted; objective mismatch is what remains.

**Rank is a clumsy knob for that claim** — it changes the model class as well as the magnitude, which is why §2884/§2887 could not read
it and both registered it unreportable. **Scale is the clean knob.** Multiply every fitted `LW[k]` by a scalar `s` and sweep
`s ∈ {0, .25, .5, .75, .9, 1, 1.1, 1.25}`. Nothing about the model class changes; only the magnitude along the ridge solution's own
direction. If the ridge solution overshoots for the end-to-end objective, **some s < 1 must beat s = 1**.

**§2881 supplies a free anchor**: it measured `LW := {}` at **+0.1740 nats**, which is exactly this sweep's `s = 0`. pred_c requires the
sweep to land on it, making that arm a cross-rung reproduction rather than an unanchored curve.

Nine arms differ only in what `evalM` sees, so they share **one fitted stack** (`ops/frontier_evalarms.py`, validated at a baseline
deviation of exactly **0.0** in §2888 and §2889): **one pipeline run instead of nine**. The original maps are snapshotted after the
refit loop and restored before each arm.

## Predictions, each with its worked-example line

- **pred_a — REPRODUCTION GATE, verbatim from §2125 rung 30.** `|L2_F(baseline) − 2.6735| ≤ .05`. *Worked example:* every rung in this
  family reads +2.6735/+2.6736; past .05 and **nothing else here is readable.**
- **pred_b — some shrunk map beats the ridge fit.** at least one `s < 1` with `cost < 0`. *Worked example:* if the ridge solution
  overshoots, the curve dips below zero somewhere around **s ≈ .75–.9**, by ≈ **−.01 to −.03** (§2890 measured −.0294 for the rank-1
  model class); if the ridge fit is optimal along this direction the curve rises monotonically from s = 1 in both directions and
  nothing goes below zero — `b_null_the_ridge_fit_is_optimal_along_this_direction`, which **bounds §2890's account** to model-class
  changes rather than magnitude.
- **pred_c — scale zero reproduces §2881.** `|cost(s=0) − 0.1740| ≤ .02`. *Worked example:* `s = 0` is exactly `LW := {}`, so ≈
  **.000**; a deviation ≥ **.05** means this rung and §2881 disagree (`c_null_this_rung_disagrees_with_S2881`) and neither is usable.
  This is the clause that anchors the whole curve to an independently measured endpoint.
- **pred_d — the arms are connected.** `|cost(s=0)| ≥ .005`. §2879's rule as a measured predicate: a disconnected manipulation reads
  exactly **.0000**.
- **pred_e — the curve is unimodal.** turns in the cost sequence ≤ **1**. *Worked example:* a smooth trade-off between under- and
  over-shooting gives one turn; an erratic curve would suggest the scaling is interacting with the class switch rather than acting as a
  simple magnitude knob.

## Nulls

- `b_null_the_ridge_fit_is_optimal_along_this_direction` (nothing below 1 improves): the mismatch does **not** show along magnitude, so
  §2890's rank-1 effect is about the *model class* (a fixed direction with a scalar gain) and not about the fit being too large. That
  is a real bound on the account and it redirects the follow-up from re-scaling to re-parameterising.
- `c_null_this_rung_disagrees_with_S2881` (anchor deviation ≥ .05).

## Price

**1 full frontier pipeline run, ≤ 400 GPU-seconds** (§2888 104.4 s for four arms; §2891's sweep 166.3 s for ten), 0 backwards, 0 fitted
parameters beyond the pipeline's own. The parent `ops/frontier_fisher8.py` is **unmodified**. It is **not forward-instrumented**, so the
receipt reports `gpu_forwards: 0` with `forwards_instrumented: false` and `pipeline_runs: 1`, and the ledger's `Price:` line says so —
the count is absent, not zero. Receipt: `frontier_tail_link_shrinkage_results.json`, read with `price` in the same command the ledger
section is written from, in the canonical `Price: N GPU forwards, X GPU-seconds` / `Results: <file>.json` form (§2853, §2858), under a
filename no other section cites (§2876).
