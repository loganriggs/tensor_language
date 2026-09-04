# Frontier: optimise the tail scales jointly, not one at a time — preregistration

Registered 2026-09-04T11:31Z (the exact string `date -u` returned in its own tool call immediately before this write). Before the run.
Immutable; the rung's frozen-hash check refuses to execute if this file changes.

## SIGN CONVENTION

Frontier L2 is **CE ADDED ABOVE THE REAL MODEL, so LOWER IS BETTER** (§2135; §312: "+2.6735 beating +2.84/+2.93"). A cost is
`L2(arm) − L2(baseline)`, **POSITIVE = WORSE**, so a **negative cost is an improvement**. §2128/§2129/§2133/§2134 RETRACTED;
**§2125 STANDS** — norm-2304 at 2.6735.

## Why

§2899 found that per-layer tuning does not compose. §2901 **refuted** my cascade explanation for it and located the real one: **the
coupling is in the optimal amount, not the sign.** Every layer's shrinkage helps — the best prefix length is 8, the whole band — but
uniform **.05** gives **−0.2048** while uniform **.20** gives **−0.2290**. Each layer alone prefers .05; all eight together prefer ~.20.
§2901 closed with the consequence: **any per-layer tuning must optimise jointly.**

This rung takes the first joint step in the only form that is cheap: a **factorial probe of a global scale crossed with a multiplier on
one layer**. `a10L` is the target because §2898 and §2899 both give it the largest single-layer gain (−0.0698 at .25, −0.0847 at .05).
Eleven arms, one fitted stack, **one pipeline run**.

- If some (global, multiplier) pair **beats** the pure global, a proper coordinate descent is warranted.
- If the pure global **wins**, §2896's single adopted number is the whole story for this block and no further tuning is justified — a
  clean, cheap close, and the outcome I would bet on given §2901.

## Predictions, each with its worked-example line

- **pred_a — REPRODUCTION GATE, verbatim from §2125 rung 30.** `|L2_F(baseline) − 2.6735| ≤ .05`. *Worked example:* every rung in this
  family reads +2.6735/+2.6736; past .05 and **nothing else here is readable.**
- **pred_b — the global arm reproduces §2898.** `|cost(global .20) − (−0.2291)| ≤ .01`. *Worked example:* ≈ **.000**; the **eighth**
  reproduction of the adopted effect and the tie that makes this grid comparable to every earlier curve.
- **pred_c — a joint setting beats the pure global.** `cost(pure global .20) − cost(best grid arm) ≥ +.005` nats. *Worked example:* if
  a10L wants relatively more shrinkage than its neighbours, the (g=.20, m=.5) cell reads ≈ **−.235** against −.229, a gain of ≈
  **+.006**; if the pure global is already optimal the best cell **is** a multiplier of 1.0 and this reads ≤ 0 —
  `c_null_the_pure_global_is_optimal`, which **closes tail-scale tuning** at §2896's adopted number.
- **pred_d — the best arm uses a nontrivial multiplier.** the argmin cell has `m ≠ 1.0`. *Worked example:* registered separately from
  pred_c because a tiny win at m = 1.0 would be a grid artefact rather than evidence for joint tuning; both must hold for the joint
  direction to be worth pursuing.
- **pred_e — the arms are connected.** `|cost(global .20)| ≥ .005`. §2879's rule as a measured predicate.

## Nulls

- `b_null_the_anchor_fails` (≥ .03).
- `c_null_the_pure_global_is_optimal` (≤ 0): **tail-scale tuning is closed** — §2896's global scale is the whole effect for this block,
  and further per-layer work is not warranted. Registered as a genuine, useful outcome rather than a failure.

**Adoption rule, stated in advance:** any improvement beyond §2896's adopted −0.2287 may be entered as a result **only if pred_a,
pred_b and pred_d all hold** — baseline reproduces, the global arm re-anchors, and the winning cell is genuinely joint rather than the
pure global under another name.

## Price

**1 full frontier pipeline run, ≤ 400 GPU-seconds** (this family measures 104–166 s per multi-arm fit-once run; eleven arms here),
0 backwards, 0 fitted parameters beyond the pipeline's own. The parent `ops/frontier_fisher8.py` is **unmodified**. It is **not
forward-instrumented**, so the receipt reports `gpu_forwards: 0` with `forwards_instrumented: false` and `pipeline_runs: 1`, and the
ledger's `Price:` line says so — the count is absent, not zero. Receipt: `frontier_tail_joint_scale_results.json`, read with `price` in
the same command the ledger section is written from, in the canonical `Price: N GPU forwards, X GPU-seconds` / `Results: <file>.json`
form (§2853, §2858), under a filename no other section cites (§2876).
