# Frontier: resolve the anchor — is the .061 gap the refit-time / eval-time distinction? Preregistration

Registered 2026-09-04T10:38Z (the exact string `date -u` returned in its own tool call immediately before this write). Before the run.
Immutable; the rung's frozen-hash check refuses to execute if this file changes.

## SIGN CONVENTION

Frontier L2 is **CE ADDED ABOVE THE REAL MODEL, so LOWER IS BETTER** (§2135; §312: "+2.6735 beating +2.84/+2.93"). A cost is
`L2(arm) − L2(baseline)`, **POSITIVE = WORSE**, so a **negative cost is an improvement**. §2128/§2129/§2133/§2134 RETRACTED;
**§2125 STANDS** — norm-2304 at 2.6735.

## Why

§2893 and §2894 both measured a large apparent improvement — **−0.2288 nats at scale 0.25**, reproducing across two independent runs to
**0.0003**, present on the fitting window as well as fresh — and **neither adopted it**, because in both the `LW := {}` anchor missed.

§2894 narrowed the cause to a single number. The anchor reads **+0.1130** when `LW := {}` is applied to a **frozen** stack, against
§2881's **+0.1740** when it is applied **inside the sequential refit loop**, where the downstream fits adapt to it. Gap **0.0610**.

That is the same refit-time/eval-time distinction §2884 raised and §2887 tested for the **rank** knob — where freezing did *not* change
the answer. Here it apparently does, which is why it must be measured rather than argued. This rung measures **both applications in one
script against one baseline**:

| run | arms |
|---|---|
| 1 — frozen stack | baseline; `LW := {}` applied **after** all eight refits; `s = 0.25` |
| 2 — refit-time | `LW := {}` applied **inside** the refit loop — exactly §2881's operation |

## Predictions, each with its worked-example line

- **pred_a — REPRODUCTION GATE, verbatim from §2125 rung 30.** `|L2_F(baseline) − 2.6735| ≤ .05`. *Worked example:* every rung in this
  family reads +2.6735/+2.6736; past .05 and **nothing else here is readable.**
- **pred_b — the refit-time arm reproduces §2881.** `|cost(refit-time) − 0.1740| ≤ .02`. *Worked example:* it is §2881's exact
  operation, so ≈ **.000**. If it misses by ≥ .05 (`b_null_the_refit_arm_does_not_reproduce_S2881`) then **something other than this
  distinction is wrong**, and the whole scale line stays closed — the clause that can refute my §2894 explanation.
- **pred_c — the frozen arm reproduces §2894.** `|cost(frozen) − 0.1130| ≤ .02`. *Worked example:* same operation as §2894's
  `lw_empty`, so ≈ **.000**; a miss means the frozen path is itself unstable.
- **pred_d — the two applications genuinely differ.** `|cost(refit) − cost(frozen)| ≥ .04`. *Worked example:* §2894 implies ≈ **.061**;
  if instead they agree to within .01 (`d_null_the_distinction_is_not_the_cause`), the refit/eval distinction is **not** what separated
  §2894 from §2881 and the .061 remains unexplained.
- **pred_e — the s = 0.25 arm reproduces the curve.** `|cost(s=.25) − (−0.2288)| ≤ .01`. *Worked example:* two runs already agree to
  .0003, so ≈ **.000**. This keeps the improvement tied to the same measured object across a third run.

## Nulls

- `b_null_the_refit_arm_does_not_reproduce_S2881` (≥ .05): my §2894 diagnosis is wrong; the scale line stays closed.
- `d_null_the_distinction_is_not_the_cause` (≤ .01).

**Adoption rule, stated in advance so it is not decided after seeing the number:** the **−0.2288** improvement may be entered as a
result **only if pred_a, pred_b, pred_c and pred_e all hold** — that is, only when the refit-time arm lands on §2881, the frozen arm
lands on §2894, and the curve reproduces a third time. The receipt records this gate explicitly as
`adoption_gate_all_of_a_b_c_e`. §2893's and §2894's numbers stay unadopted regardless.

## Price

**2 full frontier pipeline runs, ≤ 500 GPU-seconds** (this family measures 104–166 s per multi-arm fit-once run), 0 backwards, 0 fitted
parameters beyond the pipeline's own. The parent `ops/frontier_fisher8.py` is **unmodified**. It is **not forward-instrumented**, so the
receipt reports `gpu_forwards: 0` with `forwards_instrumented: false` and `pipeline_runs: 2`, and the ledger's `Price:` line says so —
the count is absent, not zero. Receipt: `frontier_tail_anchor_resolution_results.json`, read with `price` in the same command the ledger
section is written from, in the canonical `Price: N GPU forwards, X GPU-seconds` / `Results: <file>.json` form (§2853, §2858), under a
filename no other section cites (§2876).
