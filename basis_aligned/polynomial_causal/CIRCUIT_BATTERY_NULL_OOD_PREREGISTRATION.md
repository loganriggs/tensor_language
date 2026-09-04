# Circuit battery — OOD handle for the writer percentile preregistration

Registered 2026-09-04T08:36Z (`date -u` read in the same tool call that composed this header, per §2858's rule). Before the run.
Immutable; the rung's frozen-hash check refuses to execute if this file changes.

## Why, and what is deliberately NOT changed

§2862 measured attn8's median percentile among live components at **.214** (bar ≤ .20 — pred_a FALSE by .014). §2866 replicated the
same design at 1.5× the rows and measured **.1875** — pred_a TRUE. **The two straddle the bar and the entire difference is .027.**
Declaring attn8 an outlier on the strength of that crossing is exactly the sort of bar-driven claim this lane keeps catching, so §2866
recorded the flip and declined to upgrade the claim.

Another rerun of the same population would not settle it. This rung changes the **population** instead: the same 36-component sweep
scored on the **OOD** split, built from held-out vocabulary pools and disjoint from both SELECT and TEST, with TEST retained only as
the comparison axis for the ranking clause. Everything else is identical and **every bar is carried over verbatim** from
`CIRCUIT_BATTERY_SELECTIVITY_NULL_PREREGISTRATION.md`.

If the percentile lands near .19–.21 again on genuinely different situations, the reading is stable across populations. If it moves
substantially, then "attn8's percentile" is not a single number, and **neither §2862's FALSE nor §2866's TRUE should be carried
forward** — which is the outcome that would most limit my own §2861–§2866 arc.

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
