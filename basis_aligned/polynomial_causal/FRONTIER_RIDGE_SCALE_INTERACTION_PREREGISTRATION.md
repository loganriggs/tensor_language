# Frontier: is the shrinkage intrinsic, or compensating for the ridge penalty? Penalty × scale. Preregistration

Registered 2026-09-04T13:24Z (substituted by `date -u` at write time, so the stamp cannot drift from the value read). Before the run.
Immutable; the rung's frozen-hash check refuses to execute if this file changes.

## SIGN CONVENTION

Frontier L2 is **CE ADDED ABOVE THE REAL MODEL, so LOWER IS BETTER** (§2135; §312: "+2.6735 beating +2.84/+2.93"). A cost is
`L2(arm) − L2(baseline)`, **POSITIVE = WORSE**, so a **negative cost is an improvement**. §2128/§2129/§2133/§2134 RETRACTED;
**§2125 STANDS** — this re-solves an already-specified estimator and rescales it; it neither selects nor reorders.

## Why

§2917 established that **increasing** the ridge penalty is monotonically worse, and **declared its own gap out loud**: the grid was
one-sided, so `λ < 1` — *less* ridge than the fitted penalty — was never tested, and the rung could rule out larger penalties without
naming an optimum. This closes that gap and asks a sharper question with the same cached machinery.

If the adopted `×0.25` exists because the **ridge bias** makes the solution wrong, then how much shrinkage is needed should **depend on
the penalty**. If instead the excess belongs to the **local objective itself** — least squares against a local target, whatever the
regulariser — the optimal scale should be the **same at every penalty**.

| | reading |
|---|---|
| the optimum **moves** with the penalty | the shrinkage is entangled with the ridge; `×0.25` is not a clean object and its value is an accident of `1e-2·n` |
| the optimum is **penalty-independent** | the excess belongs to **what** was fitted, not **how** — and the scalar is a property of the local objective, which is what §2890/§2902/§2905 actually claim |

Three penalties × five scales, one fitted stack, **no refit** — each cell is a re-solve from cached normal equations.

## Predictions, each with its worked-example line

- **pred_a — REPRODUCTION GATE.** `|L2_F(baseline) − 2.6735| ≤ .05`. *Worked example:* +2.6735; past .05, **nothing else is readable**.
- **pred_b — the identity cell is a physical no-op.** `|cost(λ×1, scale 1.0)| ≤ .005`. *Worked example:* **0.0000** — the fitted penalty
  with no scaling is the baseline stack rebuilt from its own normal equations. **This is the control §2879 taught me to register**: it
  proves the re-solve path reconstructs what the pipeline actually fitted, so every other cell means what it says. A failure here
  invalidates the whole rung, including §2917's λ arms, which used the same path.
- **pred_c — the fitted-penalty quarter-scale cell reproduces §2896.** `|cost(λ×1, scale 0.25) − (−0.2287)| ≤ .01`. *Worked example:*
  ≈ −0.2287, a seventh reproduction. A miss ≥ .03 makes the λ rows incomparable to the adopted result.
- **pred_d — the optimal scale is the same at every penalty.** `argmin_scale cost(λ, ·)` identical for λ ∈ {0.25, 1, 4}. *Worked
  example:* intrinsic ⇒ **0.25 at all three**, holds; ridge-entangled ⇒ the optimum **rises with λ** (more penalty already shrinks the
  solution, so less extra shrinkage is wanted — e.g. 0.15 / 0.25 / 0.40) and fails. Three grid points compared for equality: no ratio,
  no operand that can change sign.
- **pred_e — the scale grid brackets every optimum.** Each row's best scale interior to {0.15, 0.25, 0.40, 0.60, 1.00}. *Worked
  example:* **registered because this is my recurring failure — §2907, §2909 and §2917 all failed interiority on grids I chose too
  narrow.** The grid extends to 0.15 so that 0.25 is interior. A non-interior optimum makes that row's value a **bound**, and pred_d's
  equality test is then between bounds, which I will say explicitly rather than report as a clean equality.

## Nulls

- `b_null_the_machinery_is_not_a_noop` (|identity| > .02) — invalidates this rung **and retroactively questions §2917's λ arms**.
- `c_null_the_anchor_fails` (≥ .03).
- **`d_null_the_shrinkage_is_compensating_for_the_ridge_penalty`** — the interesting null.
- `e_null_the_scale_grid_is_too_narrow`.

**What I will do with each outcome, stated in advance.** pred_d holds ⇒ the tail scalar is recorded as a property of the **local
objective**, which strengthens §2890/§2905's reading and makes the same question worth asking of every locally-fitted block. pred_d
fails ⇒ the scalar is entangled with the regulariser, `×0.25` is an accident of `1e-2·n`, and the correct object is the (penalty, scale)
pair — I will say so plainly even though §2896, §2904 and §2912 all rest on that scalar, and **an independent physical control comes
before any withdrawal**, per the standing rule. **Nothing is adopted from this rung**; §2912's configuration is untouched by it.

## Price

**1 full frontier pipeline run + 16 arms × 3 windows of forward evaluation, ≤ 600 GPU-seconds** (§2917's 7-arm run took 126.6 s; each
cell adds 32 re-solves of a 1152×1152 system, sub-second on GPU), 0 backwards, **0 fitted parameters** — every cell re-solves an
estimator whose form is fixed. The parent `ops/frontier_fisher8.py` is **unmodified**. Not forward-instrumented, so the receipt reports
`gpu_forwards: 0` with `forwards_instrumented: false` and `pipeline_runs: 1`. Receipt:
`frontier_ridge_scale_interaction_results.json`, read with `price` in the same command the ledger section is written from, in the
canonical `Price:` / `Results:` form (§2853, §2858), under a filename no other section cites (§2876).
