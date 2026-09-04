# Circuit battery — selectivity null calibration preregistration

Registered 2026-09-04T08:19Z (`date -u` read immediately before composing this header, per §2858's rule). Before the run. Immutable;
the rung's frozen-hash check refuses to execute if this file changes.

## Why

§2860 established that the old selectivity metric was pinned near 1 by a control that preserves the causal variable, and §2861
re-specified it as `|d_C| / max(d_A1, .5)` over the copy control. All five of §2861's predictions held on the seven attn8 behaviours:
the two metrics differ by a median **.574**; P behaves as a positive control at **.056**; the writer opposes the copy answer on **7 of
7**; the score generalises to held-out TEST with a median gap of **.091**; the behaviour ordering is stable at Spearman **.714**.

**But the resulting numbers — .27 to .63 — have no meaning yet.** The campaign's .25 "selective" bar was calibrated for the OLD metric,
and carrying it across to a differently-constructed quantity is a category error; §2861 says so and deliberately does not commit it.
A ratio needs a null before it can be read, and the right null is not a bar borrowed from elsewhere: it is **what this same metric
reads for a component that is not the writer**. This rung computes it with every one of the 36 components standing in as the writer,
so attn8 is read against a distribution measured under identical conditions.

**Admissibility gate, carried from §2820:** a stand-in component must DO something before its ratio is eligible — `|d_A1| ≥ .10`.
§2820's error was crowning an inert head "perfectly selective" because a ratio of two near-zero numbers is meaningless; without this
gate the null distribution would be dominated by exactly that artefact, and would make attn8 look ordinary for a spurious reason.

## Predictions, each with its worked-example line

- **pred_a — the identified writer is an outlier.** median over behaviours of attn8's percentile among live components ≤ **.20**
  (percentile = fraction of live components scoring LOWER, and LOWER = MORE SELECTIVE, so a small percentile means attn8 is among the
  most selective). *Worked example:* if attn8's write is specific to the causal variable it sits in the bottom fifth, ≈ **.05–.15**;
  if the metric is measuring something generic about late components, ≈ **.5**.
- **pred_b — the null distribution is not degenerate.** median (p90 − p10) of live components' selectivity ≥ **.30**. *Worked
  example:* a metric with range separates components, spread ≈ **.5**; a metric that returns the same value for everything —
  the failure mode §2860 found in the OLD metric, which read ≈1 for all — has spread ≈ **.00**, and pred_a would then be unreadable
  even if it passed.
- **pred_c — the inherited bar is not the right bar.** median number of live components scoring ≤ .25 (the OLD metric's bar) ≥ **4**.
  *Worked example:* if a quarter of all components clear a bar meant to mark a special writer, the bar is not measuring what its name
  says, ≈ **6–10**; if only attn8 and a neighbour clear it, ≈ **1–2** and the bar transfers after all.
- **pred_d — the component ranking is stable across held-out splits.** median Spearman ρ(selectivity over components, SELECT vs TEST)
  ≥ **.50**. *Worked example:* if per-component selectivity is a real property, the ranking survives disjoint situations, ρ ≈ **.7**;
  if it is noise, ρ ≈ **.0** and pred_a's percentile is not a stable fact either. (Registered on the COMPONENT axis — the invariant
  index shared between the two splits — not on a sample-indexed loading; that is the trap recorded against §2647/§2649.)
- **pred_e — inert components are gated out.** at least one component is excluded by the `|d_A1| ≥ .10` gate. *Worked example:* early
  layers are inert for these behaviours (§2821 found nothing live past mlp11 in the reader direction), so several should be gated; if
  none is, the gate is not binding and pred_a's denominator pool is not what this document describes.

## Nulls

- `a_null_writer_is_typical` (percentile ≥ .50): **attn8 is unremarkable on the re-specified metric** — the re-specification would then
  have produced a number with range but no ability to identify the writer, and §2860's critique would stand while §2861's replacement
  would not be usable. This is the outcome that most cleanly limits my own correction and it is registered for that purpose.
- `b_null_every_component_scores_alike` (spread ≤ .10), `c_null_inherited_bar_is_fine` (≤ 1 component under .25),
  `d_null_component_ranking_is_noise` (ρ ≤ .20).

## Price

≤ 5,000 GPU forwards, 0 backwards, 0 fitted parameters, ≤ 120 GPU-seconds. PER_CELL=16 (below §2852's 24) because the component sweep
multiplies the row count by 36; the reduction is registered here rather than chosen after seeing the result. Receipt:
`circuit_battery_selectivity_null_calibration_results.json`, read with `price` in the same command the ledger section is written from,
in the canonical `Price: N GPU forwards, X GPU-seconds` / `Results: <file>.json` form the guard can parse (§2853, §2858).
