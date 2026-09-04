# Frontier: which half of the MLP stage carries it — token table or quadratic residual? Preregistration

Registered 2026-09-04T09:14Z (`date -u` read in the same tool call that composed this header). Before the run. Immutable; the rung's
frozen-hash check refuses to execute if this file changes.

## SIGN CONVENTION

Frontier L2 is **CE ADDED ABOVE THE REAL MODEL, so LOWER IS BETTER** (§2135; §312: "+2.6735 beating +2.84/+2.93"). A drop **cost** is
`L2_F(dropped) − L2_F(baseline)`, **POSITIVE = WORSE**. §2128/§2129/§2133/§2134 RETRACTED; **§2125 STANDS** — the frontier is
norm-2304 at 2.6735.

## Why

§2874 asked the analogous question of the **attention** stage and got a null: every `attnd` dictionary collapses to one constant for
−.0001 nats, a5 and its a6 control alike (row spreads .0416 vs .0422), so nothing there was specific to any layer and the whole stage
is already effectively a table of constants.

The **MLP** stage is built differently, and its structure decomposes in a way the attention dictionaries do not: both `fit_tableres`
and `fit_res` (its matched-context twin) return a **token-indexed table** `tb[ids]` plus a **low-rank quadratic residual** `A` in a
64-dimensional projected subspace. Those are two qualitatively different objects — a lookup and a polynomial correction — and no rung
has asked which one carries the frontier.

Three runs of §312's published norm-selection pipeline, splitting the stage at **both** construction sites (there are two `tableres`
fitters; patching only one would silently mix the arms, and pred_e records that both are patched):

| arm | change |
|---|---|
| BASELINE | none |
| residual dropped | `A := 0` — keep the lookup, drop the polynomial correction |
| table dropped | `tb := 0` — keep the polynomial correction, drop the lookup |

Both touch **fitted values only, never control flow**, the discipline §2874 used. Derived from `ops/frontier_fisher8.py` (§2125 rung
30), which is **unmodified**; the derived file retargets the parent's single `OUT` so no path can clobber §2125's cited receipt.

## Predictions, each with its worked-example line

- **pred_a — REPRODUCTION GATE, carried over verbatim from §2125 rung 30.** `|L2_F(baseline) − 2.6735| ≤ .05`. *Worked example:*
  §2874 measured +2.6736 on this derivation; past .05 and **nothing else here is readable.**
- **pred_b — the two halves are not equally important.** `|cost(table dropped) − cost(residual dropped)| ≥ .10` nats. *Worked
  example:* if the lookup does the work and the correction is decoration, ≈ **.5+**; if the two are interchangeable the asymmetry is
  ≈ **.00** and the decomposition carries no information, which is registered as `b_null_the_halves_are_interchangeable`.
- **pred_c — the token table carries the stage.** `cost(table dropped) ≥ +.20` nats. *Worked example:* a lookup that supplies most of
  each MLP's output costs a lot to remove, ≈ **+.5 to +2**; if removing it costs ≈ **+.02**, the table is nearly redundant given the
  residual, and the direction of pred_c is simply wrong — recorded as FAILED, not reinterpreted.
- **pred_d — the residual is cheap to drop.** `cost(residual dropped) ≤ +.10` nats. *Worked example:* if the quadratic correction is a
  small refinement, ≈ **+.01–.05** and the frontier's MLP stage reduces to token lookups — a large structural simplification; if it is
  load-bearing, ≈ **+.3** and `d_null_the_residual_is_load_bearing` fires.
- **pred_e — both fitters are patched**, recorded so that a future reader can see the arms were not silently mixed.

## Nulls

- `b_null_the_halves_are_interchangeable` (asymmetry ≤ .02).
- `d_null_the_residual_is_load_bearing` (`cost(residual dropped) ≥ +.30`): the polynomial correction is doing real work, the MLP stage
  does **not** reduce to lookups, and the "simpler tensor program" claim does not extend from the attention stage to the MLP stage.
  This is the outcome that would most limit the frontier arc and it is registered to be recognised.

## Price

**3 full frontier pipeline runs, ≤ 800 GPU-seconds** (§2874 measured 283 s for three arms), 0 backwards, 0 fitted parameters beyond the
pipeline's own. The parent is **not forward-instrumented**, so the receipt reports `gpu_forwards: 0` with `forwards_instrumented:
false` and `pipeline_runs: 3` beside it, and the ledger's `Price:` line says so — the count is absent, not zero. Receipt:
`frontier_mlp_table_vs_residual_results.json`, read with `price` in the same command the ledger section is written from, in the
canonical `Price: N GPU forwards, X GPU-seconds` / `Results: <file>.json` form (§2853, §2858).
