# Frontier: is 0.25 optimal, and is the mismatch uniform across the tail band? — preregistration

Registered 2026-09-04T10:45Z (the exact string `date -u` returned in its own tool call immediately before this write). Before the run.
Immutable; the rung's frozen-hash check refuses to execute if this file changes.

## SIGN CONVENTION

Frontier L2 is **CE ADDED ABOVE THE REAL MODEL, so LOWER IS BETTER** (§2135; §312: "+2.6735 beating +2.84/+2.93"). A cost is
`L2(arm) − L2(baseline)`, **POSITIVE = WORSE**, so a **negative cost is an improvement**. §2128/§2129/§2133/§2134 RETRACTED;
**§2125 STANDS** — the published frontier remains norm-2304 at 2.6735.

## Why

§2896 **adopted** the campaign's first frontier improvement: scaling every tail-refit link map `LW[k]` by **0.25** takes the frontier
from **+2.6736 to +2.4448**, an improvement of **0.2287 nats**. Both anchors resolved (refit-time **+0.1740** at deviation **0.0000**,
frozen **+0.1131** at **0.0001**) and the curve reproduced across four runs at a maximum deviation of **0.0003**.

Two questions follow immediately, and both are eval-only, so both fit in **one pipeline run**:

1. **Is 0.25 optimal?** §2893's grid was coarse — 0/.25/.5/.75/.9/1 — and .25 was simply its lowest interior point. A finer grid
   (.10/.15/.20/.25/.30/.40) locates the optimum and says how much more is available.
2. **Is the mismatch uniform across the band?** Scaling **one layer at a time** at 0.25 gives a per-layer profile. Concentrated gain
   means the eventual end-to-end refit should be targeted; spread gain means the mismatch is a property of the *fitting procedure*
   rather than of particular layers.

## Predictions, each with its worked-example line

- **pred_a — REPRODUCTION GATE, verbatim from §2125 rung 30.** `|L2_F(baseline) − 2.6735| ≤ .05`. *Worked example:* every rung in this
  family reads +2.6735/+2.6736; past .05 and **nothing else here is readable.**
- **pred_b — the `s = .25` arm reproduces §2896.** `|cost(.25) − (−0.2287)| ≤ .01`. *Worked example:* four runs already agree to
  .0003, so ≈ **.000**. This makes the rung a **fifth reproduction of the adopted number** as well as an extension of it; a miss ≥ .03
  (`b_null_the_anchor_fails`) would put §2896's adoption itself in question, which is why the bar is tight.
- **pred_c — the optimum is interior.** the best grid point has cost ≤ **−0.22** and is **not** an endpoint of the grid. *Worked
  example:* if the true optimum is near .25 the curve turns inside the grid, ≈ **−0.23 to −0.26**; if the best point is .10 or .40 the
  optimum lies outside and the grid must be extended before any "optimal scale" is claimed.
- **pred_d — no single layer matches the global scale.** `cost(best single layer) − cost(global .25) ≥ +.05`. *Worked example:* eight
  layers each contributing part of the gain gives singles around **−.02 to −.06** against the global **−.229**, a gap of ≈ **.17**; if
  one layer alone reproduces the global gain, the effect is that layer's and not the band's.
- **pred_e — the per-layer gains are not additive.** `|Σ singles − global| ≥ .05`. *Worked example:* §2889 and §2892 found the motif
  layers strongly **sub**additive, and §2880/§2888 found superadditivity elsewhere, so additivity would be the surprise here;
  `e_null_the_layers_are_additive` (≤ .02) would mean the tail band behaves as eight independent knobs, which would make a targeted
  end-to-end refit much easier than the rest of this construction has been.

## Nulls

- `b_null_the_anchor_fails` (≥ .03): §2896's adopted number does not reproduce a fifth time — the adoption would need revisiting, and
  this clause is registered so that possibility is checked rather than assumed away.
- `e_null_the_layers_are_additive` (≤ .02).

## Price

**1 full frontier pipeline run, ≤ 400 GPU-seconds** (this family measures 104–166 s per multi-arm fit-once run; fifteen arms here),
0 backwards, 0 fitted parameters beyond the pipeline's own. The parent `ops/frontier_fisher8.py` is **unmodified**. It is **not
forward-instrumented**, so the receipt reports `gpu_forwards: 0` with `forwards_instrumented: false` and `pipeline_runs: 1`, and the
ledger's `Price:` line says so — the count is absent, not zero. Receipt: `frontier_tail_scale_profile_results.json`, read with `price`
in the same command the ledger section is written from, in the canonical `Price: N GPU forwards, X GPU-seconds` /
`Results: <file>.json` form (§2853, §2858), under a filename no other section cites (§2876).
