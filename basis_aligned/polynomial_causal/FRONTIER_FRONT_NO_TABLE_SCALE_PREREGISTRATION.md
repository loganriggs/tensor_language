# Frontier: is the local/end-to-end mismatch procedural, or specific to the table+residual pair? — preregistration

Registered 2026-09-04T11:01Z (the exact string `date -u` returned in its own tool call immediately before this write). Before the run.
Immutable; the rung's frozen-hash check refuses to execute if this file changes.

## SIGN CONVENTION

Frontier L2 is **CE ADDED ABOVE THE REAL MODEL, so LOWER IS BETTER** (§2135; §312: "+2.6735 beating +2.84/+2.93"). A cost is
`L2(arm) − L2(baseline)`, **POSITIVE = WORSE**; a **gain** here is `cost(tb dropped) − cost(tb dropped and scaled)`, **POSITIVE = the
scaling helps**. §2128/§2129/§2133/§2134 RETRACTED; **§2125 STANDS** — norm-2304 at 2.6735.

## Why

§2897 established that the front MLP stage's compensation is **one-way**: given a refit, the rank-64 quadratic residual `A` absorbs
**4.9479** of the 5.629 nats that removing the token table costs on a frozen stack, while the token table absorbs only **0.0215** of the
residual's. Its practical reading was *"drop the table, not the residual"* — the opposite of what parameter counts suggest.

That sets up a sharp question about §2890's central claim. §2895 found that scaling the residual improves the frontier **while the table
is present**. **Does it still improve once the table is gone and `A` has been refitted to cover for it?**

- **If YES**, the local/end-to-end objective mismatch is a property of **the ridge fitting procedure itself** — it reappears in whatever
  configuration is fitted — and the end-to-end refitting move §2890 argued for is general.
- **If NO**, the mismatch was specific to the **table+residual pair**, an artefact of two co-fitted blocks dividing one job, and
  §2890's account is much narrower than it looks. That would be a real limit on my own top-ranked mathematical move.

| run | arms |
|---|---|
| 1 — reference | the published frontier, unmodified |
| 2 — no table | `tb := 0` applied at **refit** time (so `A` refits to cover it), then `A` scaled at .25/.5/.75/.9 — eval-only on that stack |

## Predictions, each with its worked-example line

- **pred_a — REPRODUCTION GATE, verbatim from §2125 rung 30.** `|L2_F(baseline) − 2.6735| ≤ .05`. *Worked example:* every rung in this
  family reads +2.6735/+2.6736; past .05 and **nothing else here is readable.**
- **pred_b — the table drop reproduces §2897.** `|cost(tb dropped) − 0.6811| ≤ .02`. *Worked example:* §2897 measured this to a
  ten-thousandth against §2877, so ≈ **.000**; a miss ≥ .05 (`b_null_the_anchor_fails`) means the refit path is not reproducible here
  and nothing else is readable.
- **pred_c — scaling still helps after the table is gone.** best gain ≥ **+.02** nats. *Worked example:* if the mismatch is procedural,
  the refitted `A` is again over-large for the end-to-end objective and scaling recovers ≈ **+.05 to +.15**; if it reads ≤ 0
  (`c_null_the_mismatch_was_pair_specific`) then §2890's account is confined to co-fitted pairs — **the outcome that most limits my own
  top-ranked move, and it is registered for that reason.**
- **pred_d — the gain is smaller than with the table present.** best gain < **0.1648** (§2895's with-table gain). *Worked example:* with
  the table gone `A` has been refitted to do more of the work, so there should be less slack to remove, ≈ **+.05**; if the gain is
  *larger* the mismatch grows when a block is asked to do more, which would be a different and more interesting story than either branch
  above.
- **pred_e — the arms are connected.** `|cost(tb dropped)| ≥ .005`. §2879's rule as a measured predicate.

## Nulls

- `b_null_the_anchor_fails` (≥ .05).
- `c_null_the_mismatch_was_pair_specific` (best gain ≤ 0): §2890's account does not generalise beyond co-fitted pairs.

**Adoption note:** this rung is diagnostic and adopts nothing. §2896's tail-map improvement stands on its own; §2895's front-table
number remains unadopted.

## Price

**2 full frontier pipeline runs, ≤ 500 GPU-seconds** (this family measures 104–166 s per multi-arm fit-once run; run 2 refits, which is
the expensive part), 0 backwards, 0 fitted parameters beyond the pipeline's own. The parent `ops/frontier_fisher8.py` is **unmodified**.
It is **not forward-instrumented**, so the receipt reports `gpu_forwards: 0` with `forwards_instrumented: false` and `pipeline_runs: 2`,
and the ledger's `Price:` line says so — the count is absent, not zero. Receipt:
`frontier_front_no_table_scale_results.json`, read with `price` in the same command the ledger section is written from, in the canonical
`Price: N GPU forwards, X GPU-seconds` / `Results: <file>.json` form (§2853, §2858), under a filename no other section cites (§2876).
