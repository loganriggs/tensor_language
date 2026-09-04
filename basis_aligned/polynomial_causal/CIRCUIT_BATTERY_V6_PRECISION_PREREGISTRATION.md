# Circuit battery — v6 precision replication preregistration

Registered 2026-09-04T08:29Z (`date -u` read in the same tool call that composed this header, per §2858's rule). Before the run.
Immutable; the rung's frozen-hash check refuses to execute if this file changes.

## Why, and what is deliberately NOT changed

§2863 ran the v6 metric over 14 behaviours at their own writers, PER_CELL=24, and two clauses missed narrowly:

| clause | measured | bar |
|---|---|---|
| pred_c — the writer opposes the copy answer | **10 of 14 = 71.4%** | ≥ 80% |
| pred_e — the behaviour ordering is stable (Spearman SELECT↔TEST) | **.556** | ≥ .60 |

Both are close enough to their bars that the honest question is whether they are sampling noise. This rung is the **same design at
PER_CELL=64** — 2.7× the rows per cell — with **everything else identical and every bar carried over verbatim** from
`CIRCUIT_BATTERY_V6_BANK_SWEEP_PREREGISTRATION.md`. Re-registering a bar for a replication would make the replication meaningless, and
choosing a new bar after seeing 71.4% and .556 is precisely the predicate-fitting this lane keeps catching. **A miss that survives 2.7×
the data is a fact about the model; one that flips was a sample-size limit.** The ledger will say which, either way.

pred_a, pred_b and pred_d passed in §2863 (.781, .072, .117) and are re-run unchanged as instrument checks: if any of them now fails,
the replication itself is suspect and pred_c/pred_e cannot be read from it.

## Predictions

Identical to the bank-sweep preregistration, verbatim, and repeated here so this document stands alone:

- **pred_a** — median |old-style ratio − new selectivity| ≥ **.30** on SELECT. *Worked example:* old-style ≈ 1 by construction
  (§2860), new ≈ .45 → **≈ .55**; if they agree, ≈ **.00**.
- **pred_b** — median `|d_P_donor − d_A1|/d_A1` ≤ **.25**. *Worked example:* a variable-carrying writer gives ≈ **.05–.2** (§2863
  measured .072); one not carrying its variable ≈ **1.0**.
- **pred_c** — `d_C < 0` on ≥ **80%** of behaviours. *Worked example:* a writer marking "which item was last" competes with verbatim
  copying, so removal helps the copy answer on ≈ 90–100%; indifference gives ≈ **50%**. §2863 measured 71.4% and this rung asks whether
  that is where it stays.
- **pred_d** — median |selectivity(SELECT) − selectivity(TEST)| ≤ **.15**. *Worked example:* a real property transfers, ≈ **.09–.12**;
  row-driven noise ≈ **.4+**.
- **pred_e** — Spearman ρ(selectivity SELECT, TEST) ≥ **.60** over the BEHAVIOUR axis — the invariant index shared by both splits,
  never a sample-indexed loading (§2647/§2649). *Worked example:* ρ ≈ **.8** if per-behaviour selectivity is a fact; §2863's **.556**
  sits between that and the ρ ≤ .20 null, which is exactly why it needs more data.

## Nulls

Carried over verbatim: `a_null_metrics_agree` (≤ .10), `b_null_preserving_control_is_not_positive` (≥ .50),
`c_null_writer_does_not_oppose_copy` (≤ 50%), `d_null_does_not_generalise` (≥ .40), `e_null_ordering_is_noise` (ρ ≤ .20).

## Price

≤ 8,000 GPU forwards, 0 backwards, 0 fitted parameters, ≤ 180 GPU-seconds. Receipt:
`circuit_battery_v6_precision_replication_results.json`, read with `price` in the same command the ledger section is written from, in
the canonical `Price: N GPU forwards, X GPU-seconds` / `Results: <file>.json` form the guard can parse (§2853, §2858).
