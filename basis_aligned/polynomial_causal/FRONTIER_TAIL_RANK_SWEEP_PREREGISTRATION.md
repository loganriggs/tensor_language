# Frontier: a nine-point rank sweep of the tail link maps — preregistration

Registered 2026-09-04T10:18Z (the exact string `date -u` returned in its own tool call immediately before this write). Before the run.
Immutable; the rung's frozen-hash check refuses to execute if this file changes.

## SIGN CONVENTION

Frontier L2 is **CE ADDED ABOVE THE REAL MODEL, so LOWER IS BETTER** (§2135; §312: "+2.6735 beating +2.84/+2.93"). A cost is
`L2_F(arm) − L2_F(baseline)`, **POSITIVE = WORSE**. §2128/§2129/§2133/§2134 RETRACTED; **§2125 STANDS** — norm-2304 at 2.6735.

## Why, and what this rung is NOT for

§2884 and §2887 both measured a **non-monotone** rank curve on the tail dictionaries' within-class link maps, and **both were registered
unreportable because of it**. §2884 blamed the sequential refit; §2887 froze the fits and the non-monotonicity survived, **refuting that
explanation**. Measured so far, against a full-rank baseline of 0:

| rank | 1 | 8 | 64 |
|---|---|---|---|
| cost | **−0.0294** | **+0.1489** | **+0.0674** |

A maximum at rank 8 is what no account of a pure SVD truncation predicts. Three explanations survive and a three-point curve cannot
separate them:

1. **Regularisation** — the maps overfit their 512-document fitting window and truncation removes variance the frontier does not want.
   Predicts a smooth single-minimum curve.
2. **A degenerate rank-1 map** that happens to suit the class table. §2881 measured that removing `LW` outright costs **+0.1740**, so
   rank 1 is 0.20 nats away from "effectively removed" and this account has work to do.
3. **An implementation fault** in the truncation, benign at rank 1.

**Nine points fix the shape.** This rung exists to explain the curve, and **it does not report a low-rank number as a usable result** —
§2884 and §2887 both forbid that until the shape is explained, and this document does not lift that.

Ranks 1, 2, 4, 8, 16, 32, 64, 128, 256. Every arm differs only in what `evalM` sees, so they share **one fitted stack** under the
fit-once/eval-many pattern (`ops/frontier_evalarms.py`, validated in §2888 at a baseline deviation of exactly **0.0**): **one pipeline
run instead of ten**. The original link maps are **snapshotted after the refit loop and restored before each arm's truncation**, so the
arms are independent perturbations rather than a compounding sequence — the failure mode that would otherwise reintroduce §2884's
problem in a new form.

## Predictions, each with its worked-example line

- **pred_a — REPRODUCTION GATE, verbatim from §2125 rung 30.** `|L2_F(baseline) − 2.6735| ≤ .05`. *Worked example:* every rung in this
  family reads +2.6735/+2.6736; past .05 and **nothing else here is readable.**
- **pred_b — the curve has at most one turn.** counting sign changes in successive differences of `cost(rank)` across the nine points,
  **turns ≤ 1**. *Worked example:* the regularisation account gives a smooth U — one turn, ≈ **1**; a pure truncation with no
  overfitting gives a monotone decay to 0 — **0**; an implementation fault or numerical noise gives an erratic curve, **3+**, which is
  `b_null_the_curve_is_erratic` and points at account 3.
- **pred_c — the three known ranks reproduce §2887.** `|cost(r) − §2887's cost(r)| ≤ .02` for r ∈ {1, 8, 64}. *Worked example:* same
  truncation, same frozen stack, so ≈ **.00**; a deviation ≥ **.05** means this sweep and §2887 disagree
  (`c_null_this_rung_disagrees_with_S2887`) and neither can be used. This is the clause that makes the sweep a *check* on §2887 rather
  than a replacement for it.
- **pred_d — the arms are connected.** `|cost(rank 1)| ≥ .005`. §2879's rule as a measured predicate: a disconnected manipulation reads
  exactly **.0000**.
- **pred_e — the shape is recorded.** the full nine-point curve, the argmax rank and the argmin rank are written to the receipt
  regardless of outcome, so a later rung can reason about the shape without re-running.

## Nulls

- `b_null_the_curve_is_erratic` (turns ≥ 3): the truncation is not behaving like a truncation — evidence for account 3, an
  implementation fault, and the low-rank line stays closed.
- `c_null_this_rung_disagrees_with_S2887` (any deviation ≥ .05).

## Price

**1 full frontier pipeline run, ≤ 400 GPU-seconds** (§2888 measured 104.4 s for a four-arm fit-once run; nine extra evaluations plus 72
SVDs of 1152×1152 add little), 0 backwards, 0 fitted parameters beyond the pipeline's own. The parent is **not forward-instrumented**,
so the receipt reports `gpu_forwards: 0` with `forwards_instrumented: false` and `pipeline_runs: 1`, and the ledger's `Price:` line says
so — the count is absent, not zero. Receipt: `frontier_tail_rank_sweep_results.json`, read with `price` in the same command the ledger
section is written from, in the canonical `Price: N GPU forwards, X GPU-seconds` / `Results: <file>.json` form (§2853, §2858), under a
filename no other section cites (§2876).
