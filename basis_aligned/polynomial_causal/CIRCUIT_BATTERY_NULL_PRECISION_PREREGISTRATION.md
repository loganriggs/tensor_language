# Circuit battery — null calibration precision replication preregistration

Registered 2026-09-04T08:29Z (`date -u` read in the same tool call that composed this header, per §2858's rule). Before the run.
Immutable; the rung's frozen-hash check refuses to execute if this file changes.

## Why, and what is deliberately NOT changed

§2862 computed the v6 selectivity metric with all 36 components standing in as the writer, at PER_CELL=16, under §2820's admissibility
gate (`|d_A1| ≥ .10`). **pred_a failed by .014**: attn8's median percentile among live components was **.214** against a bar of ≤ .20.
Everything else held — null spread (p90−p10) **1.457**, a median of **6** live components clearing the now-retired .25 bar, component
ranking reproducing at Spearman **.596**.

A knife-edge miss at that margin is not a result until it is shown to be stable. This rung is the **same design at PER_CELL=24**
(§2852's value, 1.5× the rows), **everything else identical and every bar carried over verbatim** from
`CIRCUIT_BATTERY_SELECTIVITY_NULL_PREREGISTRATION.md`. Re-registering a bar after seeing .214 would make the replication meaningless.

Both outcomes are worth the price and neither is the one I am hoping for: if the percentile moves below .20 the original was a sampling
artefact and the ledger records the reversal; if it holds or rises, **attn8 is genuinely not the outlier I predicted**, which is the
more interesting result and the one that further limits my own §2861 replacement metric.

## Predictions

Identical to the null-calibration preregistration, verbatim:

- **pred_a** — median percentile of attn8 among live components ≤ **.20** (percentile = fraction scoring LOWER; LOWER = MORE
  SELECTIVE, so small = attn8 is among the most selective). *Worked example:* a specific writer sits in the bottom fifth, ≈ **.05–.15**;
  a generic late component ≈ **.5**. §2862 measured **.214**.
- **pred_b** — median (p90 − p10) across live components ≥ **.30**. *Worked example:* a metric with range ≈ **1.4**; one returning the
  same value for everything ≈ **.00**, which is the failure mode §2860 found in the OLD metric.
- **pred_c** — median count of live components scoring ≤ .25 ≥ **4**. *Worked example:* ≈ **6–10** means the inherited bar marks a
  sixth of the model rather than a special writer; ≈ **1–2** would mean it transfers after all.
- **pred_d** — median Spearman ρ(component ordering, SELECT vs TEST) ≥ **.50** over the COMPONENT axis — the invariant index shared by
  both splits, never a sample-indexed loading (§2647/§2649). *Worked example:* ρ ≈ **.7** if per-component selectivity is real; ≈ **.0**
  if noise, in which case pred_a's percentile is not a stable fact either.
- **pred_e** — at least one component excluded by the `|d_A1| ≥ .10` gate. *Worked example:* early layers are inert for these
  behaviours (§2821), so several should be gated; if none is, the gate is not binding and pred_a's pool is not what this describes.

## Nulls

Carried over verbatim: `a_null_writer_is_typical` (percentile ≥ .50), `b_null_every_component_scores_alike` (spread ≤ .10),
`c_null_inherited_bar_is_fine` (≤ 1 component under .25), `d_null_component_ranking_is_noise` (ρ ≤ .20).

## Price

≤ 12,000 GPU forwards, 0 backwards, 0 fitted parameters, ≤ 240 GPU-seconds — 1.5× §2862's 6,408 forwards / 73.8 s. Receipt:
`circuit_battery_null_precision_replication_results.json`, read with `price` in the same command the ledger section is written from, in
the canonical `Price: N GPU forwards, X GPU-seconds` / `Results: <file>.json` form the guard can parse (§2853, §2858).
