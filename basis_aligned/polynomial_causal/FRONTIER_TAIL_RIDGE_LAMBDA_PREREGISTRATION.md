# Frontier: is the magic 0.25 just under-regularisation? A ridge-λ sweep on the tail link maps. Preregistration

Registered 2026-09-04T13:13Z (the exact string `date -u` returned in its own tool call immediately before this write). Before the run.
Immutable; the rung's frozen-hash check refuses to execute if this file changes.

## SIGN CONVENTION

Frontier L2 is **CE ADDED ABOVE THE REAL MODEL, so LOWER IS BETTER** (§2135; §312: "+2.6735 beating +2.84/+2.93"). A cost is
`L2(arm) − L2(baseline)`, **POSITIVE = WORSE**, so a **negative cost is an improvement**. §2128/§2129/§2133/§2134 RETRACTED;
**§2125 STANDS** — this re-solves an already-specified estimator; it neither selects nor reorders.

## Why

§2914 settled that the shrinkage is real: 98.6% of the gain transports to documents that chose nothing, selection bias +0.0044 nats.
So the question stops being *is it an artefact* and becomes **what is it**.

The tail link maps are ridge solutions, `LW[k] = (Xkᵀ Xk + 10⁻²·n·I)⁻¹ Xkᵀ Yk` — a penalty chosen once, for **local reconstruction**,
and never revisited. §2890/§2902/§2905 all say the same thing: components fitted to a local criterion come out **systematically too
large** for the end-to-end score. **A ridge penalty is precisely the knob that makes a least-squares solution smaller.** The obvious
hypothesis has never been tested: the adopted `×0.25` is a crude stand-in for a λ that was simply too small.

This sweeps `λ → mult·λ` for `mult ∈ {4, 16, 64, 256, 1024}`, re-solving the **same estimator on the same data with only the penalty
changed**, and compares against the adopted uniform `×0.25` measured in the same run. The normal equations `(XkᵀXk, XkᵀYk, n)` are
captured during the fit and kept on CPU (~340 MB over 32 layer×class pairs), so each λ is a **re-solve, not a refit** — seven arms for
one pipeline run. Inherited from §2914's machinery, **every arm is also evaluated on the held-out window**, so each λ's transport is
reported at no extra pipeline cost.

**Both outcomes are worth having, which is why this is the rung to run:**

- **ridge REACHES uniform (pred_d holds)** ⇒ the correction gets a principled reading — *the local ridge is under-regularised for the
  end-to-end objective* — and the magic scalar becomes a λ with a standard statistical meaning.
- **ridge CANNOT reach it (pred_d fails)** ⇒ uniform shrinkage is a **genuinely different operator**. Ridge shrinks along data-aligned
  directions by eigenvalue; uniform scaling shrinks every direction equally. Failing pred_d would say the end-to-end excess is
  **isotropic**, not concentrated in the low-eigenvalue directions ridge attacks — a real structural fact about where the error lives.

## Predictions, each with its worked-example line

- **pred_a — REPRODUCTION GATE.** `|L2_F(baseline) − 2.6735| ≤ .05`. *Worked example:* +2.6735/+2.6736; past .05, **nothing else is
  readable**.
- **pred_b — uniform scaling reproduces §2896.** `|cost(scale25) − (−0.2287)| ≤ .01`. *Worked example:* §2896's adopted number has
  reproduced five times at ≈ −0.2287; this arm is the in-run control that the λ arms are compared against, so a miss ≥ .03
  (`b_null_the_anchor_fails`) makes pred_d unreadable.
- **pred_c — more ridge improves the frontier at all.** `min_λ cost ≤ −0.05`. *Worked example:* if under-regularisation is any part of
  the story, a large λ recovers a visible slice of the 0.2287 (≈ −0.10 to −0.23). If ridge is simply the wrong knob, every λ arm sits
  near 0.000 or **positive** (more penalty, worse fit, worse score) and this fails.
- **pred_d — ridge reaches what uniform shrinkage achieves.** `min_λ cost ≤ −0.2287 + 0.02 = −0.2087`. *Worked example:* the
  under-regularisation reading predicts ≈ −0.23 or better and this holds; the isotropic-excess reading predicts the λ path tops out well
  short (≈ −0.10) and this fails. **This is the predicate the rung exists for, and I have no strong prior on which way it goes.**
- **pred_e — the best λ multiplier is interior to the grid.** *Worked example:* an optimum at ×4 or ×1024 means the grid did not bracket
  it. **This is my recurring failure mode — §2907 and §2909 both failed interiority on grids I chose too narrow — so it is registered as
  a predicate here rather than discovered afterwards.** A non-interior optimum makes pred_d's number a bound, not a value, and I will
  report it as such.

## Nulls

- `b_null_the_anchor_fails` (≥ .03).
- `c_null_more_ridge_does_nothing` (`min_λ cost > −0.05`).
- `d_null_ridge_cannot_reach_uniform_shrinkage` — **the interesting null**: the excess is isotropic, not eigenvalue-aligned.
- `e_null_the_lambda_grid_is_too_narrow`.

**What I will do with each outcome, stated in advance.** pred_d holds and pred_e holds ⇒ the tail correction is restated as a λ, and the
next rung asks whether the CP correction (§2902) has the same reading. pred_d fails with pred_c holding ⇒ record that ridge captures
*part* of the excess but not all, quantify the remainder, and **do not** adopt λ over the reproduced uniform scalar. pred_c fails ⇒ the
under-regularisation hypothesis is dead and the isotropy of the excess becomes the finding. **No outcome licenses tuning λ on the
held-out window** — the held-out numbers are reported, never optimised against.

## Price

**1 full frontier pipeline run + 7 arms × 2 windows of forward evaluation, ≤ 500 GPU-seconds** (§2914's 2-arm × 2-window run took
101.5 s; the extra cost here is 5 more arms plus 32 re-solves of a 1152×1152 system per λ, which is sub-second on GPU), 0 backwards,
**0 fitted parameters beyond the pipeline's own** — the λ arms re-solve an estimator whose form is fixed. The parent
`ops/frontier_fisher8.py` is **unmodified**. Not forward-instrumented, so the receipt reports `gpu_forwards: 0` with
`forwards_instrumented: false` and `pipeline_runs: 1`. Receipt: `frontier_tail_ridge_lambda_results.json`, read with `price` in the
same command the ledger section is written from, in the canonical `Price:` / `Results:` form (§2853, §2858), under a filename no other
section cites (§2876).
