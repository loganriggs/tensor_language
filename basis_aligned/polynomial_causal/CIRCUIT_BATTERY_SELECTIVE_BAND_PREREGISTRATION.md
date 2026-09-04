# Circuit battery — selective band overlap preregistration

Registered 2026-09-04T08:44Z (`date -u` read in the same tool call that composed this header, per §2858's rule). Before the run.
Immutable; the rung's frozen-hash check refuses to execute if this file changes.

## Why

§2870 established a dissociation the rest of this arc had been circling: the per-**component** selectivity ranking survives a population
change (ρ **.763** between OOD and TEST, §2868), while the per-**behaviour** ordering collapses (ρ **.253**). And §2862, §2864, §2867
and §2869 have shown **four times**, across two populations and three sample sizes, that the **argmin** of that component ranking never
reproduces — **0 of 7, every time**.

A stable ranking with an unstable minimum is what **many components sharing a similar true value** looks like. Every section since
§2864 has been converging on that account without testing it. This rung tests it on the object the evidence says is robust: not the
single most selective component, but the **set**.

All 36 components scored on all four populations (FIT, SELECT, TEST, OOD — the last built from held-out vocabulary pools), with the
top-k selective sets (k=8) compared by Jaccard overlap between populations, against the overlap a **random k-subset of the same live
pool** would give. Matching the random comparison to the same live pool matters: pool sizes differ between splits, and an unmatched
random baseline would make the overlap look better than it is. Admissibility gate is §2820's (`|d_A1| ≥ .10`).

## Predictions, each with its worked-example line

- **pred_a — the top set beats a random subset.** median Jaccard(SELECT, TEST) − median random-subset Jaccard ≥ **.25**. *Worked
  example:* k=8 drawn from a live pool of ~20 gives a random Jaccard of ≈ **.25**; a real set reproducing 6 of 8 members gives ≈ **.60**,
  a margin of ≈ **.35**. If the set is arbitrary the margin is ≈ **.00** and the ρ .763 ranking is carried entirely by the inert tail.
- **pred_b — the top set survives the population change.** median Jaccard(TEST, OOD) ≥ **.40**. *Worked example:* the component ranking
  transports at ρ .763, so its top set should largely transport too, ≈ **.5–.7**; if the set is pool-specific it falls to the random
  level ≈ **.25**, and "the selective band" would be a within-pool artefact like the behaviour ordering in §2870.
- **pred_c — the band is contiguous in depth.** median fraction of the OOD top-k lying within a single 8-layer window ≥ **.75**.
  *Worked example:* a genuine band in the 8–15 region puts 7 or 8 of 8 in one window, ≈ **.90–1.00**; components scattered across all 18
  layers give ≈ **.45**. This is the clause that decides whether "band" is the right word or merely a convenient one.
- **pred_d — the argmin is unstable where the set is not.** the number of behaviours whose argmin is the SAME component across all four
  populations ≤ **1**. *Worked example:* §2862/§2864/§2867/§2869 each measured 0 of 7 on pairs; across four populations at once it
  should be 0–1. If it is 4+, the argmin is stable after all and four prior sections need revisiting.
- **pred_e — inert components are gated out**, as in §2862.

## Nulls

- `a_null_top_set_is_random` (margin ≤ .05) — **there is no selective set**, the transporting ranking is an artefact of the inert tail,
  and the account §2867–§2870 have been converging on is retired. This is the clause that can most cleanly kill the band hypothesis and
  it is registered for that purpose.
- `b_null_set_does_not_survive_population` (≤ .20), `c_null_band_is_scattered` (≤ .50), `d_null_argmin_is_stable` (≥ 4).

## Price

≤ 16,000 GPU forwards, 0 backwards, 0 fitted parameters, ≤ 300 GPU-seconds — four populations at PER_CELL=24. Receipt:
`circuit_battery_selective_band_overlap_results.json`, read with `price` in the same command the ledger section is written from, in the
canonical `Price: N GPU forwards, X GPU-seconds` / `Results: <file>.json` form the guard can parse (§2853, §2858).
