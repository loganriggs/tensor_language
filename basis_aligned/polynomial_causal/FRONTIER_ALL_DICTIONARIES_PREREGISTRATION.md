# Frontier: collapse every attention dictionary, and measure the instrument's own resolution — preregistration

Registered 2026-09-04T09:11Z (`date -u` read in the same tool call that composed this header). Before the run. Immutable; the rung's
frozen-hash check refuses to execute if this file changes.

## SIGN CONVENTION — first, because it is the rule this family most depends on

Frontier L2 is **CE ADDED ABOVE THE REAL MODEL, so LOWER IS BETTER** (§2135; §312: "+2.6735 beating +2.84/+2.93"). A collapse **cost**
is `L2_F(collapsed) − L2_F(baseline)`, **POSITIVE = WORSE**. §2128/§2129/§2133/§2134 RETRACTED for reading higher L2 as better;
**§2125 STANDS** — the frontier is norm-2304 at 2.6735.

## Why, and the thing that must be measured before anything else

§2874 collapsed a5's fitted `attnd` dictionary to a single constant and measured **−.0001 nats** — and the a6 control collapsed for
**exactly the same −.0001**, firing the registered null `c_null_collapsing_anything_is_free`. Row spreads were .0416 (a5) and .0422
(a6): indistinguishable. Two consequences, and this rung is built around both.

**First: −.0001 is at the reproduction noise floor.** §2874's own baseline differs from the published 2.6735 by +.0001. A single
layer's collapse therefore cannot be resolved, and §2874 said so rather than claiming "free". **Nothing in this family is readable
until the floor is measured**, so **arm 2 is the baseline run again, identical configuration** — its difference from arm 1 *is* the
instrument's resolution. That measurement is owed to §2874 and to every collapse rung after it.

**Second: no layer is special, so the question changes.** If the fit imposes near-constancy on every motif dictionary, the interesting
quantity is **how much of the frontier is a table of constants**. Arm 3 collapses **all sixteen** attention dictionaries, a2–a17. Each
carries ten class rows plus four 1152×1152 link maps for the LINK classes [2,7,8,9]; each collapse removes **5,318,784** parameters,
and sixteen remove **85,100,544**.

The collapse replaces each layer's fitted `CV` with ten copies of `Y.mean(0)` and empties `LW`. Derived from `ops/frontier_fisher8.py`
(§2125 rung 30), which is **unmodified**; the derived file retargets the parent's single `OUT` so no path can clobber §2125's cited
receipt. The hook loops `for k in LW` rather than `for k in LINK` — provably a no-op for uncollapsed layers, since `fit_attnd` builds
`LW` with exactly the `LINK` keys three lines above (disclosed in §2874's rung and carried here).

## Predictions, each with its worked-example line

- **pred_a — REPRODUCTION GATE, carried over verbatim from §2125 rung 30.** `|L2_F(baseline) − 2.6735| ≤ .05`. *Worked example:*
  §2874 measured +2.6736; anything past .05 means the derivation perturbed the construction and **nothing else here is readable.**
- **pred_b — the instrument resolves better than one milli-nat.** `|L2_F(baseline repeat) − L2_F(baseline)| ≤ .001`. *Worked example:*
  a deterministic pipeline on identical inputs reads **.0000**; if it reads **.01**, then §2874's −.0001, this rung's arm 3, and every
  future collapse number are all inside the noise and the whole family needs a different instrument. This clause can invalidate my own
  §2874 and is registered for that purpose.
- **pred_c — all sixteen dictionaries collapse cheaply.** `cost_all ≤ +.05` nats. *Worked example:* if each layer is genuinely free,
  sixteen are too, ≈ **.00–.02**; if each costs a real .01 that the single-layer test could not resolve, sixteen cost ≈ **+.16** and
  this fails — which is the informative failure, because it would mean the per-layer effect is real but individually unmeasurable.
- **pred_d — the joint cost is resolvable above the floor.** `|cost_all| ≥ 2 × floor`. *Worked example:* sixteen simultaneous
  collapses should be sixteen times easier to see than one; if this reads FALSE the joint collapse is **genuinely free to the limit of
  this instrument**, which is a stronger and cleaner result than a small positive cost. Registered so that FALSE is interpretable
  rather than disappointing.
- **pred_e — the parameter saving is stated exactly**: 5,318,784 per layer, 85,100,544 for all sixteen.

## Nulls

- `b_null_the_instrument_cannot_resolve_this_family` (floor ≥ .01): the measurement family is unusable and §2874's caveat becomes a
  retraction.
- `c_null_the_dictionaries_are_jointly_load_bearing` (`cost_all ≥ +.20`): the constants story fails at full scale, and the frontier's
  attention dictionaries genuinely need their class structure.
- `d_null_the_joint_cost_is_below_the_floor` (`|cost_all| < 2 × floor`): recorded as the outcome that makes the simplification
  **maximally** clean, not as a failure.

## Price

**3 full frontier pipeline runs, ≤ 800 GPU-seconds** (§2874 measured 283 s for three arms), 0 backwards, 0 fitted parameters beyond the
pipeline's own. The parent is **not forward-instrumented**, so the receipt reports `gpu_forwards: 0` with `forwards_instrumented:
false` and `pipeline_runs: 3` beside it, and the ledger's `Price:` line says so — the count is absent, not zero. Receipt:
`frontier_all_dictionaries_collapse_results.json`, read with `price` in the same command the ledger section is written from, in the
canonical `Price: N GPU forwards, X GPU-seconds` / `Results: <file>.json` form (§2853, §2858).
