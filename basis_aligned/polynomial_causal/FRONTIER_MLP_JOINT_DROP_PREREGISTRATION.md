# Frontier: is the MLP stage one job done twice, or two jobs? Preregistration

Registered 2026-09-04T09:28Z (`date -u` read in the same tool call that composed this header). Before the run. Immutable; the rung's
frozen-hash check refuses to execute if this file changes.

## SIGN CONVENTION

Frontier L2 is **CE ADDED ABOVE THE REAL MODEL, so LOWER IS BETTER** (§2135; §312: "+2.6735 beating +2.84/+2.93"). A drop **cost** is
`L2_F(dropped) − L2_F(baseline)`, **POSITIVE = WORSE**. A **subadditivity gap** is `(cost_table + cost_residual) − cost_both`,
**POSITIVE = the two halves are redundant with each other**. §2128/§2129/§2133/§2134 RETRACTED; **§2125 STANDS** — the frontier is
norm-2304 at 2.6735.

## Why

§2877 dropped each half of the frontier's MLP stage separately and found **both load-bearing and nearly equally so**: removing the
token lookup costs **+0.6814 nats**, removing the rank-64 quadratic residual **+0.7536**, with an asymmetry of only .0722. It flagged
the quantity it did not measure and declined to infer it:

- if the halves are **independent**, the joint cost is the sum, **1.4350**;
- if they are **largely redundant** — each able to stand in for the other — the joint cost lands near the worse single drop, **0.7536**.

Those are the same measurements under two very different accounts of what the MLP stage *is*, and the difference decides whether it is
**two jobs** or **one job done twice**. Under §2876's measured resolution — the pipeline reproduces L2_F exactly at four decimals —
the ~0.68-nat gap between the two accounts is enormous relative to the instrument.

Two arms of §312's published norm-selection pipeline: BASELINE, and `tb := 0` **and** `A := 0` together, applied at **both** `tableres`
construction sites (§2877's pred_e established that patching only one silently mixes arms). §2877's single-drop costs are read from its
receipt **under a frozen hash** rather than retyped, so the additivity arithmetic cannot drift from the numbers it is compared against.
Derived from `ops/frontier_fisher8.py` (§2125 rung 30), which is **unmodified**; the derived file retargets the parent's single `OUT`.

## Predictions, each with its worked-example line

- **pred_a — REPRODUCTION GATE, carried over verbatim from §2125 rung 30.** `|L2_F(baseline) − 2.6735| ≤ .05`. *Worked example:*
  §2874–§2877 all measured +2.6735/+2.6736 on this derivation; past .05 and **nothing else here is readable.**
- **pred_b — the halves are not additive.** `|1.4350 − cost_both| ≥ .20` nats. *Worked example:* perfect independence gives
  **1.4350** and this reads .0000 → FALSE; heavy redundancy gives ≈ **0.78** and this reads ≈ **.65** → TRUE. A FALSE here means the
  two halves genuinely do separate work that simply adds, which is `b_null_the_halves_are_additive`.
- **pred_c — the halves are redundant with each other.** `gap = 1.4350 − cost_both ≥ +.20`. *Worked example:* if each half can partly
  substitute for the other, the joint damage is much less than the sum, gap ≈ **+.6**; if dropping both is *worse* than the sum
  (superadditive — the pair is load-bearing in a way neither is alone) the gap goes **negative**, which pred_b would still catch and
  which is reported with its sign rather than as an absolute.
- **pred_d — dropping both is worse than dropping either.** `cost_both − max(0.6814, 0.7536) ≥ +.10`. *Worked example:* a sanity
  bound — removing more structure should not help, so ≈ **+.1 to +.7**; if this reads ≈ **.00** the two halves are **fully**
  redundant (`c_null_the_halves_are_fully_redundant`), i.e. whichever survives carries the stage alone. If it reads **negative** the
  arms are inconsistent and the rung is void.
- **pred_e — the baseline reproduces §2877 exactly.** `|L2_F(baseline) − 2.6735| ≤ .001` against §2877's own recorded baseline.
  *Worked example:* same derivation, same pipeline, and §2876 measured the resolution as **0.0**, so ≈ **.0000**. This is the
  cross-rung instrument check that licenses comparing this run's joint cost against §2877's single costs at all.

## Nulls

- `b_null_the_halves_are_additive` (|gap| ≤ .10): the MLP stage is **two independent jobs**, and no part of it is redundant — the
  cleanest negative result for any further simplification of the stage.
- `c_null_the_halves_are_fully_redundant` (`cost_both − max(single) ≤ .05`): the stage is **one job done twice**, and one of the two
  halves could in principle be dropped once the other is refit to compensate — which would be a genuine simplification lead and is
  registered so it is recognised rather than read as pred_d merely failing.

## Price

**2 full frontier pipeline runs, ≤ 600 GPU-seconds** (§2874–§2877 measured 279–283 s for three arms, so two should land near 190 s),
0 backwards, 0 fitted parameters beyond the pipeline's own. The parent is **not forward-instrumented**, so the receipt reports
`gpu_forwards: 0` with `forwards_instrumented: false` and `pipeline_runs: 2` beside it, and the ledger's `Price:` line says so — the
count is absent, not zero. Receipt: `frontier_mlp_joint_drop_results.json`, read with `price` in the same command the ledger section is
written from, in the canonical `Price: N GPU forwards, X GPU-seconds` / `Results: <file>.json` form (§2853, §2858).
