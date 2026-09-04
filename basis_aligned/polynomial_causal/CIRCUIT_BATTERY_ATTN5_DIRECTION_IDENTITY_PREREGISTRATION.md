# CIRCUIT BATTERY — ATTENTION 5 DIRECTION IDENTITY (preregistration)

Registered 2026-09-04 05:41Z (box clock, read immediately before this line was written). Claude, LANE 1 CUDA.
Rung `circuit_battery_attn5_direction_identity`. Script: `ops/circuit_battery_attn5_direction_identity.py`.
Input receipt: `circuit_battery_write_rank_census_results.json` (§2834, sha 269689cc0586ef591c8395c338d4c8b526244f47c90540d26aa3db272bcbca41).
IMMUTABLE: any change gets a new document, not an edit.

## Object

§2834: attention 5 has the most one-dimensional write in bilin18 — top-direction energy .981, effective rank 1.0, 1st of 36. §2833: that
direction is universal (|cos| 1.000 across disjoint natural sets, .997 against code). §2832: a constant write costs .119 nats against a
2.200-nat ablation, so most of the component's value survives without any context-dependence at all.

Taken together those say attention 5 may be COMPILABLE to two objects: one fixed vector `u` estimated off-line, and one scalar per
position `α(pos) = ⟨write(pos), u⟩`. This rung measures that reduction directly and asks what the two objects are — whether `u` is
simply the mean-write direction (making the component a bias with a varying gain), and how much `α` varies at all. `u` and the constant
gain are fitted on 24 documents and everything is scored on 24 DISJOINT ones.

Sign convention: d_ce = CE_arm − CE_NATIVE in nats, POSITIVE = the arm HURTS. **Not the §312 frontier's L2 (CE added above the real
model by an installed approximation, LOWER IS BETTER, frontier norm-2304 at 2.6735, §2135); nothing installs; metric-constructed
bases/spans remain CLOSED (§2118 lineage) and this reconstruction is a DIAGNOSTIC of compilability, not a proposed interface.**

## Arms

- `ZERO` — attention 5's write set to zero (reference; §2833 measured 2.203 nats on held-out natural).
- `RANK1` — the write replaced by `⟨write, u⟩ · u`, i.e. one fixed direction with a per-position gain. **1,152 fitted parameters.**
- `CONST` — the write replaced by `ᾱ · u` with `ᾱ` the mean gain over the fit documents. **1,153 fitted parameters, no context at all.**

## Predictions

```
BARS  = {rank1_nats: .05, scalar_value: .03, cos_mean: .90, gain_cv: .50, ce_tol: .01}
NULLS = {rank1_ge: .50, scalar_value_le: 0.0, cos_mean_le: .50, gain_cv_ge: 1.50}
```

**pred_a_rank_one_reconstruction_is_cheap** — `d_ce(RANK1)` ≤ .05 nats on held-out documents. *Worked example:* §2834 measured .981 of
this write's energy in one direction, and §2833's held-out rank-32 cost .049; a rank-ONE reconstruction should therefore land .01–.05 if
the missing 1.9% of energy is not load-bearing, and .3–2.2 if it is (approaching the 2.203 of deletion). This is the clause that decides
whether "compilable to one vector and one scalar" is a real statement. Null: ≥ .50.

**pred_b_the_scalar_carries_value** — `d_ce(CONST) − d_ce(RANK1)` ≥ .03 nats. *Worked example:* §2832 measured a constant write at .119
nats in-sample; if the per-position gain matters, RANK1 comes in well below CONST and the difference is .05–.10. If it reads ≤ 0, the
gain is decoration and attention 5 reduces to a pure BIAS — a stronger and simpler result than the one this rung is set up to find, and
one I would rather discover than assume. A DIFFERENCE of two damages in the same units, not a ratio. Null: ≤ 0.

**pred_c_the_direction_is_the_mean_write** — |cos(u, mean write)| ≥ .90. *Worked example:* if the write is a bias-like vector whose
magnitude varies, the top singular direction and the mean direction coincide at .95–1.00; if the dominant direction is a
context-carrying axis orthogonal to a large constant offset, .0–.4. Two random directions in R^1152 sit at |cos| ≈ .03. Absolute cosine
in [0, 1]. Null: ≤ .50.

**pred_d_the_gain_is_stable** — the coefficient of variation of `α` over held-out positions, `std(α)/|mean(α)|`, ≤ .50.
*Worked example:* a bias-like write with mild modulation reads .1–.5; a genuinely context-driven gain (sign changes, order-of-magnitude
swings) reads ≥ 1.0. Note this pred and pred_b are complementary rather than redundant: pred_b asks whether the gain is WORTH anything
in CE, pred_d asks whether it MOVES much at all, and a small but consistently-placed modulation can be worth a lot while varying little.
The denominator is |mean(α)| and is floored; if the mean gain were near zero the CV would be unstable and that is reported rather than
hidden. Null: ≥ 1.50.

**pred_e_instrument_reproduces_native_ce_matched** — |manual forward CE − the model module's own CE on the SAME chunk| ≤ .01 nats,
in the matched-sample form that §2832's version of this check got wrong.

## Stated null

The rank-1 reconstruction fails (≥ .50 nats), the gain is worthless, the direction is not the mean write, and the gain swings wildly.
That would mean attention 5's near-rank-1 ENERGY (§2834) does not correspond to a near-rank-1 FUNCTION — which would be the same
energy-is-not-causality lesson as §2825/§2826, arriving one level up, and would be recorded as such.

## Price

Basis and gain fitted on 24 documents; four scored passes (native, zero, rank1, const) plus one collection pass on 24 disjoint
documents, plus one matched instrument chunk. Literal budget: ≤ 200 GPU document-forwards, 0 backwards, **1,153 declared fitted
parameters** (one direction plus one scalar — the smallest fitted object in this campaign). < 2 GPU-minutes.

## What this does NOT claim

One corpus (natural), one layer, and the held-out set is a different slice of the same frozen cache — §2833 already did the
cross-corpus version of the direction test. A cheap reconstruction on document CE is not a claim about behaviour on the task bank, and
not a claim about the §312 frontier, where the quantity is L2 and lower is better. Nothing installs. Does not satisfy Codex's four-phase
integration contract; updates no circuit record.
