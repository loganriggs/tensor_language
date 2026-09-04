# CIRCUIT BATTERY — ATTENTION 5 SURROGATE PRICE (preregistration)

Registered 2026-09-04 05:25Z (box clock, read immediately before this line was written). Claude, LANE 1 CUDA.
Rung `circuit_battery_attn5_surrogate_price`. Script: `ops/circuit_battery_attn5_surrogate_price.py`.
Input receipts: `circuit_battery_attn5_class_gate_price_results.json` (§2830, sha 870f1c4065ddfea418a545db5536f93776cd445ff8e1b97cbb9f33f88b592e56)
and `circuit_battery_attn5_head_class_split_results.json` (§2831, sha 42c1d8adcd9c7d241d32b1383f9a8ca17a06d0d10c95ad99000c32b072bbd7fc).
IMMUTABLE: any change gets a new document, not an edit.

## Why this rung, and what it is careful NOT to be

Three results now converge on a testable design rule. §2830: attention 5 costs 2.200 nats of document CE when ablated, 3rd of 36, and
is 20.4× more expensive per unit of its own write norm than the median component. §2831: its class gate lives in heads {5, 7}, which
carry .542 of its class damage, and its class and margin head maps are identical. §2825/§2826: in this model ENERGY-ranked structure is
not causal structure — an in-sample rank-4 subspace holds .700 of a removal effect's energy and delivers .139 of its damage, while a
rank-1 causally-defined axis holding .0021 of the energy delivers .199.

The prediction that follows is that a surrogate chosen CAUSALLY (keep the two heads that gate the class) costs less document CE than one
chosen by ENERGY (project the write onto its own top-k singular directions), even when the energy surrogate is given far more freedom
than the head surrogate.

**Two things this rung is deliberately not.** (i) The energy-basis arm is a NEGATIVE CONTROL and never a proposed interface:
metric-constructed bases and spans are CLOSED (the §2118 lineage), and nothing here reopens them or proposes installing them.
(ii) Nothing here installs into the §312 frontier. Every number below is a LOCAL surrogate measurement on documents:
d_ce = CE_arm − CE_NATIVE in nats, POSITIVE = the arm HURTS. It is NOT the frontier's L2, which is CE ADDED ABOVE THE REAL MODEL by an
installed approximation and where LOWER IS BETTER (frontier norm-2304 at 2.6735, §2135); no value in this rung may be quoted as an L2.

Fixed before the run: layer 5, kept heads {5, 7} (from §2831, not re-chosen here), ranks {8, 32, 128}, 32 natural documents from the
frozen row cache, chunk 8. The energy basis and the mean write are fitted on those same documents and are DECLARED fitted parameters
(Σ k × 1152 = 197,  632 for the three ranks); the head surrogate has none.

## Arms

- `ZERO` — attention 5's write set to zero at every position (the §2830 reference; expected ≈ 2.20 nats).
- `HEADS_57` — the write recomputed as `c_proj` of the concatenated head outputs with all heads but 5 and 7 zeroed. Zero fitted parameters.
- `RANK_k` for k ∈ {8, 32, 128} — the write projected onto the top-k right singular directions of its own document distribution.
- `MEAN` — the write replaced everywhere by its mean over positions and documents (a rank-0 constant).

## Predictions

```
BARS  = {beat_energy_nats: .20, shallow: .50, mean_gap: .30, head_energy: .40, ce_tol: .01}
NULLS = {beat_energy_le: 0.0, shallow_le: .20, mean_gap_le: 0.0, head_energy_ge: .70}
```

**pred_a_causal_surrogate_beats_energy_surrogate** — `d_ce(RANK_128) − d_ce(HEADS_57)` ≥ .20 nats. *Worked example:* if causal choice
beats energy choice, HEADS_57 costs ~.4–.9 nats while RANK_128 costs ~1.2–2.0, giving .5–1.3. If energy captures what matters, RANK_128
is nearly free and this goes negative — which would be a clean refutation of the §2825/§2826 lesson at the component level and would say
attention 5's write IS its high-variance directions. A DIFFERENCE of two damages in the same units, both non-negative by construction
(a surrogate that HELPS would make one negative and is reportable as such). Null: ≤ 0.

**pred_b_energy_rank_curve_is_shallow** — `d_ce(RANK_128) / max(d_ce(ZERO), 1e-9)` ≥ .50, i.e. keeping the top 128 of 1152 energy
directions still costs at least half of what deleting the component entirely costs. *Worked example:* §2824's rank sweep on a reader's
removal effect was flat from rank 1 to rank 8; if the same holds for the write, rank 128 recovers little and this reads .5–.9. If the
write is genuinely low-rank in the way its energy suggests, .05–.2. Both operands are damages; the denominator is the largest of them
and cannot be near zero (§2830 measured 2.20 nats). Null: ≤ .20.

**pred_c_mean_write_is_not_enough** — `d_ce(MEAN) − d_ce(HEADS_57)` ≥ .30 nats. *Worked example:* the control that keeps pred_a from
being satisfied by an arm that merely writes *something*: a constant write should cost nearly as much as zeroing (~1.8–2.4), so the gap
over HEADS_57 is ~1.0–1.6. If a constant is as good as two real heads, the component's value is a bias and every claim about its heads
is wrong. Difference of two damages. Null: ≤ 0.

**pred_d_two_heads_are_a_small_share_of_energy** — the squared-norm fraction of attention 5's write retained by heads {5, 7} ≤ .40.
*Worked example:* two of nine heads hold ~.22 of the write's energy if heads are equal; §2831's class-gate pair holding ≤ .40 makes the
CE comparison in pred_a a genuine energy-versus-causality contrast rather than "the big heads win". If it reads ≥ .70 the head
surrogate is simply the big part of the write and pred_a's interpretation collapses to a size effect. A fraction of non-negative
energies. Null: ≥ .70.

**pred_e_instrument_reproduces_native_ce** — |native CE from this rung's forward − the model module's own CE| ≤ .01 nats.
*Worked example:* the same computation; ~1e-4. A larger gap and nothing else here can be read.

## Stated null

Energy beats causality at the component level: RANK_128 is cheaper than HEADS_57, the rank curve is steep (≤ .20 of the full ablation
cost by rank 128), a constant write is as good as the two heads, and those heads are most of the write's energy anyway. That would
overturn the reading of §2825/§2826 for writes (as opposed to reader-removal effects), and would be recorded as such rather than
explained away.

## Price

7 arm settings (native, native-with-collection, zero, heads, mean, 3 ranks) × 32 documents of 256 tokens in chunks of 8.
Literal budget: ≤ 300 GPU document-forwards, 0 backwards, **197,632 declared fitted parameters** (the energy bases and the mean write,
all fitted on the same documents they are scored on — an in-sample generosity toward the arm this rung predicts will LOSE).

## What this does NOT claim

The energy bases are fitted in-sample, which favours them; a held-out energy basis would only do worse, so pred_a is conservative in
the direction it is registered. Whole-write surrogates only — no per-position or per-context adaptivity. Document CE on one frozen
32-document natural cache; no code or OOD corpus. Nothing installs into the §312 frontier and no number here is an L2. Does not satisfy
Codex's four-phase integration contract; updates no circuit record.
