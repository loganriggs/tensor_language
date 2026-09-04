# Circuit battery — v6 bank sweep preregistration

Registered 2026-09-04T08:22Z (`date -u` read immediately before composing this header, per §2858's rule). Before the run. Immutable;
the rung's frozen-hash check refuses to execute if this file changes.

## Why

§2861 validated the re-specified selectivity metric — `|d_C| / max(d_A1, .5)` over the copy control, with P as a **positive** control —
on the seven attn8 successor behaviours: metrics differ by a median .574, positive control .056, writer opposes the copy answer 7 of 7,
held-out gap .091, Spearman .714. **Those seven share one writer and one task family.** Nothing yet distinguishes "the metric works"
from "attn8's successor circuit happens to score this way", and that distinction decides whether the campaign can use the metric at all.

This rung applies it to every behaviour the battery identified a writer for, **at that behaviour's own writer**, on the **FULL** arm
§2840 published. FULL rather than §2852's calibrated ladder: the calibrated arms exist for only eight behaviours, so using them would
silently restrict the sweep to those, and FULL requires no fitting. Splits SELECT / TEST / OOD; the bank is not mutated.

## Predictions, each with its worked-example line

- **pred_a — the two controls measure different things.** median |old-style ratio − new selectivity| ≥ **.30** on SELECT.
  *Worked example:* old-style takes `max(|d_P|,|d_C|)/d_A1` ≈ 1 by construction (§2860); if the copy control reads ≈ .45 the difference
  is ≈ **.55**; if the two agree, ≈ **.00** and the re-specification is empty outside attn8.
- **pred_b — P behaves as a positive control bank-wide.** median `|d_P_donor − d_A1|/d_A1` ≤ **.25**. *Worked example:* a
  variable-carrying writer damages the variable-preserving prompt just as much, ≈ **.05–.2** (§2861 measured .056 on seven); a writer
  not carrying its behaviour's variable reads ≈ **1.0** — and if that happens for many behaviours, the battery's writer identification,
  not the metric, is what is wrong.
- **pred_c — the writer opposes the copy answer.** `d_C < 0` on ≥ **80%** of behaviours. *Worked example:* a writer that marks "which
  item was last" competes with verbatim copying, so removal *helps* the copy answer: negative, ≈ 90–100% of behaviours. A writer
  indifferent to copying reads ≈ 0 with random sign, ≈ **50%**. (Registered as a FRACTION, not the "6 of 7" count §2861 used, because
  the task count changes here — a count carried across a differently-sized sample is the kind of clause this lane keeps catching.)
- **pred_d — selectivity generalises.** median |selectivity(SELECT) − selectivity(TEST)| ≤ **.15**. *Worked example:* a real property
  transfers to disjoint held-out situations, ≈ **.09** (§2861's value on seven); a number driven by particular rows, ≈ **.4+**.
- **pred_e — the behaviour ordering is stable.** Spearman ρ(selectivity SELECT, TEST) ≥ **.60** over the BEHAVIOUR axis — the invariant
  index shared by the two splits, never a sample-indexed loading (§2647/§2649). *Worked example:* ρ ≈ **.8** if per-behaviour
  selectivity is a fact; ρ ≈ **.0** if it is noise, in which case §2861's .714 on seven was a small-sample accident.

## Nulls

- `a_null_metrics_agree` (≤ .10) — the re-specification is empty outside attn8.
- `b_null_preserving_control_is_not_positive` (≥ .50) — **P is not a positive control bank-wide**, which would confine §2860's account
  to attn8's circuit rather than the bank's design. Registered so my own correction can be bounded.
- `c_null_writer_does_not_oppose_copy` (≤ 50%), `d_null_does_not_generalise` (≥ .40), `e_null_ordering_is_noise` (ρ ≤ .20).

## Price

≤ 3,000 GPU forwards, 0 backwards, 0 fitted parameters, ≤ 90 GPU-seconds. Receipt:
`circuit_battery_v6_bank_sweep_results.json`, read with `price` in the same command the ledger section is written from, in the
canonical `Price: N GPU forwards, X GPU-seconds` / `Results: <file>.json` form the guard can parse (§2853, §2858).
