# Frontier: does per-layer tuning compose? The constructive test of §2898's additivity — preregistration

Registered 2026-09-04T10:59Z (the exact string `date -u` returned in its own tool call immediately before this write). Before the run.
Immutable; the rung's frozen-hash check refuses to execute if this file changes.

## SIGN CONVENTION

Frontier L2 is **CE ADDED ABOVE THE REAL MODEL, so LOWER IS BETTER** (§2135; §312: "+2.6735 beating +2.84/+2.93"). A cost is
`L2(arm) − L2(baseline)`, **POSITIVE = WORSE**, so a **negative cost is an improvement**. §2128/§2129/§2133/§2134 RETRACTED;
**§2125 STANDS** — the published frontier remains norm-2304 at 2.6735.

## Why

§2896 **adopted** a global scale of 0.25 on the tail-refit link maps: **+2.6736 → +2.4448**. §2898 then measured the tail band as **the
only nearly-additive block in this construction** — eight per-layer gains summing to **−0.2512** against a global **−0.2288**, a gap of
**0.0224** (under 10%) — while every other block carries interactions an order of magnitude larger: the motif band **subadditive**
(§2889/§2892, .7441 → .3988), the front MLP stage **superadditive and one-way directional** (§2880 +3.2104; §2897 4.9479 vs 0.0215).

**Additivity is a claim with a constructive consequence.** If the layers really are independent knobs, tuning each separately and
composing should beat the single global scale by roughly the sum of the individual improvements. This rung tests that **by building
it**, which is a sharper test than any correlation:

| run | arms |
|---|---|
| 1 — per-layer grid | every tail layer at every scale in {.05, .10, .20, .35, .60}, one fitted stack (41 arms) |
| 2 — composition | the per-layer argmin composed into **one** arm, against the global 0.20 and the baseline |

## Predictions, each with its worked-example line

- **pred_a — REPRODUCTION GATE, verbatim from §2125 rung 30.** `|L2_F(baseline) − 2.6735| ≤ .05`. *Worked example:* every rung in this
  family reads +2.6735/+2.6736; past .05 and **nothing else here is readable.**
- **pred_b — the composed per-layer scale beats the global.** `cost(global) − cost(composed) ≥ +.01` nats. *Worked example:* §2898's
  per-layer optima at 0.25 already summed to .0224 more than the global; letting each layer pick its own scale from a wider grid should
  add a little more, ≈ **+.01 to +.04**. If it reads ≤ 0 (`b_null_per_layer_tuning_buys_nothing`) then the global scale is already
  optimal per layer and **§2896's single number is the whole effect** — a clean, useful negative.
- **pred_c — the composed gain is predicted by additivity.** `|cost(composed) − Σ(per-layer best gains)| ≤ .05`. *Worked example:* this
  is §2898's additivity as a **forward prediction of a number not yet measured**, not a retrospective fit: perfect independence gives
  ≈ **.00**, and §2898's measured 10% discrepancy suggests ≈ **.02**. A miss ≥ **.10** (`c_null_additivity_fails`) would overturn
  §2898's headline and say the tail band is no more additive than the rest of the construction.
- **pred_d — the global arm reproduces §2898.** `|cost(global .20) − (−0.2291)| ≤ .01`. *Worked example:* ≈ **.000**; this is the
  **sixth** reproduction of the adopted effect and keeps the comparison honest across runs.
- **pred_e — several layers prefer a different scale.** at least **3** of 8 layers have an argmin ≠ 0.20. *Worked example:* if the band
  is heterogeneous, most layers pick something else, ≈ **5–7**; if nearly all pick 0.20 the band is homogeneous and pred_b will
  necessarily be small — the two clauses are registered together so a flat result is interpretable rather than merely disappointing.

## Nulls

- `b_null_per_layer_tuning_buys_nothing` (≤ 0): §2896's global 0.25 is the whole effect.
- `c_null_additivity_fails` (≥ .10): §2898's central claim is overturned.

**Adoption rule, stated in advance:** any improvement beyond §2896's adopted −0.2287 may be entered as a result **only if pred_a,
pred_c and pred_d all hold** — that is, only when the baseline reproduces, the global arm re-anchors, and the composed number is
predicted by the additivity it is supposed to demonstrate.

## Price

**2 full frontier pipeline runs, ≤ 500 GPU-seconds** (this family measures 104–166 s per multi-arm fit-once run; 41 arms in run 1),
0 backwards, 0 fitted parameters beyond the pipeline's own. The parent `ops/frontier_fisher8.py` is **unmodified**. It is **not
forward-instrumented**, so the receipt reports `gpu_forwards: 0` with `forwards_instrumented: false` and `pipeline_runs: 2`, and the
ledger's `Price:` line says so — the count is absent, not zero. Receipt:
`frontier_tail_per_layer_optimum_results.json`, read with `price` in the same command the ledger section is written from, in the
canonical `Price: N GPU forwards, X GPU-seconds` / `Results: <file>.json` form (§2853, §2858), under a filename no other section cites
(§2876).
