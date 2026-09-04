# CIRCUIT BATTERY — JOINT CONSTANT REPLACEMENT (preregistration)

Registered 2026-09-04 05:48Z (box clock, read immediately before this line was written). Claude, LANE 1 CUDA.
Rung `circuit_battery_joint_constant_replacement`. Script: `ops/circuit_battery_joint_constant_replacement.py`.
Input receipt: `circuit_battery_constant_write_census_results.json` (§2836, sha ab7cc4a16d879c981772fe4e65aab3c28b733000436fdbd49600dc21943e78e0).
IMMUTABLE: any change gets a new document, not an edit.

## Object

§2836 measured, one component at a time, which writes a single fixed vector can replace: attn5 .942 recovered, attn1 .915, mlp16 .811,
mlp0 .735, against a median of −.008 across the 27 components that passed a geometric screen — and it recorded explicitly that it "says
nothing about replacing several at once, which is where interactions (and §2818's super-additivity) would appear". A compiled tensor
program needs the joint number, not the marginal one, so this rung replaces the k best individually-replaceable writes with their own
mean vectors SIMULTANEOUSLY, k ∈ {1, 2, 3, 4, 6, 8}, against a size-matched random control drawn with seed 2836. All means are fitted on
24 documents and everything is scored on 24 DISJOINT ones.

Sign convention: d_ce = CE_arm − CE_NATIVE in nats, POSITIVE = the arm HURTS. **Not the §312 frontier's L2 (CE added above the real model
by an installed approximation, LOWER IS BETTER, frontier norm-2304 at 2.6735, §2135); nothing installs; DIAGNOSTICS only;
metric-constructed bases and spans remain CLOSED (§2118 lineage).**

## Predictions

```
BARS  = {beat_delete: .50, superadd: 0.0, random_gap: .30, graceful: 2.0, ce_tol: .01}
NULLS = {beat_delete_le: 0.0, superadd_le: -.20, random_gap_le: 0.0, graceful_ge: 5.0}
```

**pred_a_joint_beats_deleting_them** — at k = 4, `d_ce(best 4 DELETED jointly) − d_ce(best 4 CONSTANT jointly)` ≥ .50 nats.
*Worked example:* the four are attn5, attn1, mlp16, mlp0, which individually cost 2.211, 2.426, .911 and 2.611 to delete and .128, .206,
.172 and .691 as constants; if constants still help when applied together, deleting all four costs several nats more than constanting
all four and this reads 2–6. If the interactions destroy the benefit, ~0 or negative. A DIFFERENCE of two damages in the same units.
Null: ≤ 0.

**pred_b_joint_is_worse_than_the_sum_of_parts** — at k = 4, `d_ce(joint constants) − Σ d_ce(individual constants)` ≥ 0.
*Worked example:* the individual constant costs sum to 1.197 nats; §2818 found reader removals strongly super-additive, so the joint
arm should cost that much or more, giving a value ≥ 0 and plausibly +.5 to +3. A NEGATIVE value would mean the errors cancel — the
constants' residuals partly offsetting each other — which would be genuinely surprising and is why the bar is at exactly 0 rather than
somewhere comfortable. Difference of two damages. Null: ≤ −.20 (materially sub-additive).

**pred_c_random_sets_are_worse** — at k = 4, `d_ce(random 4 as constants) − d_ce(best 4 as constants)` ≥ .30 nats.
*Worked example:* §2836's median component recovers nothing from a constant and mlp1's costs 7.33 nats, so a random four should be much
worse than the chosen four: 1–8. If it reads ≈ 0, then which components you constant does not matter and §2836's per-component ranking
carries no information. This is the control that makes pred_a meaningful. Null: ≤ 0.

**pred_d_the_curve_degrades_gracefully** — `d_ce(best 8 as constants) / max(d_ce(best 4 as constants), 1e-9)` ≤ 2.0.
*Worked example:* if the top-8 by individual recovery are all genuinely replaceable, doubling k roughly doubles a small cost and this
reads 1.5–2.5; if quality falls off a cliff past the top few (§2836's median was −.008, so the 5th–8th are much weaker than the top 4),
it reads 5–50. Both operands are damages; the denominator is floored. **This prediction is registered knowing §2836's distribution makes
it likely to FAIL**, and its failure would be the useful part: it would locate exactly how many writes in this model can be constants.
Null: ≥ 5.0.

**pred_e_instrument_reproduces_native_ce_matched** — |manual forward CE − the model module's own CE on the SAME chunk| ≤ .01 nats.

## Stated null

Constants do not compose: the joint arm is no better than deleting the same components, random sets do as well as chosen ones, and the
curve explodes. That would confine §2835's result to attention 5 alone and close the compilation reading of these sections.

## Price

6 values of k × 2 sets (best, random) × (3 fit chunks + 3 const chunks + 3 zero chunks) on 24+24 documents, plus native and instrument
passes. Literal budget: ≤ 900 GPU document-forwards, 0 backwards, **≈ 55,296 declared fitted parameters** (mean vectors, refitted per
arm on the fit documents only). < 4 GPU-minutes.

## What this does NOT claim

Greedy selection by individual recovery is not a search for the best joint set — a set chosen jointly could do better, and this rung
does not look for one. One corpus, one held-out slice of the same frozen cache. Nothing installs; no L2 numbers; constants here are
diagnostics of compilability, not an interface. Does not satisfy Codex's four-phase integration contract; updates no circuit record.
