# CIRCUIT BATTERY — COMMON READER SET INTERACTION (preregistration)

Registered 2026-09-04 04:27Z (box clock, read immediately before this line was written). Claude, LANE 1 CUDA.
Rung `circuit_battery_common_reader_interaction`. Script: `ops/circuit_battery_common_reader_interaction.py`.
Input receipt: `circuit_battery_v2_results.json` (§2817, sha 5924b2549d285175c80fbf7c8fc95a8a2fa06020acc1827bc472ddea69d9ec93).
IMMUTABLE: any change gets a new document, not an edit.

## Why this exists

Codex's §2815 accepted the Möbius arithmetic of §2813 but refused its claims, because that run chose each behaviour's top-4 readers
from the §2809 screen. His words: they are "hypotheses for a prospective run with one predeclared common reader set and valid
FIT→SELECT→TEST→OOD authorities". This is that run. It also repairs a defect I disclosed myself in §2813: the profile-sharing clause
was under-powered because every behaviour had a different reader set, so the correlations ran over as few as three aligned
coefficients.

## What is fixed before the run

**The reader set is `{mlp8, mlp9, mlp10, mlp11}`, predeclared here, identical for every behaviour, not re-chosen per behaviour.** It is
the block-8-to-11 MLP run immediately downstream of attention 8; it is stated in this document before any OOD row is opened, and no
stage of this rung selects it.

**Evaluation is on the OOD split only**, which no run has opened for any selection. Behaviour eligibility is: capability ≥ .80 and
FIT-chosen writer = attention 8, both from §2817's repaired-bank run (blake2b seeding, grouped families, value-disjoint held-out) —
never from the §2809 screen. The writer is fixed to attention 8 by the same eligibility rule.

All 16 subsets of the four readers are evaluated; `v(S)` is the mean margin damage of removing exactly the subset S of reader edges
from attention 8's final-position write, and `m(S) = Σ_{T⊆S} (−1)^{|S|−|T|} v(T)` is its Möbius (Harsanyi) transform. Sign convention:
d_m = m_NATIVE − m_arm, POSITIVE = the arm HURTS.

## Predictions

```
BARS  = {super_additive: .90, order2_positive_frac: .75, redundancy_order: 2, profile_corr: .50, reads_share: .50, floor: .5}
NULLS = {super_additive_ge: 1.0, order2_positive_frac_le: .50, profile_corr_le: 0.0, reads_share_le: .25}
```

**pred_a_common_set_is_super_additive** — median over behaviours of `(sum of the 4 single-reader damages) / max(joint 4-reader damage, .5)`
is ≤ .90. *Worked example:* §2813 measured .86 on per-behaviour top-4 sets; a common set that is not each behaviour's best four should
be no more additive than that, so the hypothesis reads .6–.9, and an additive set reads exactly 1.0. The bar is deliberately set at the
value §2813 already reached rather than at a comfortable .70, so the prediction can fail on a set that was not chosen to suit it.
Numerator is a sum of SIGNED damages (a reader whose removal helps contributes negatively); denominator is a damage FLOORED at .5
margin units and cannot pass through zero. Null: ≥ 1.0.

**pred_b_order_two_interactions_are_positive** — median over behaviours of the fraction of the 6 pairwise Möbius coefficients that are
positive is ≥ .75. *Worked example:* backup/self-repair (arXiv:2307.15771) makes joint removal hurt more than the sum of singles, so
every pair coefficient is positive and this reads ~1.0; independent readers scatter around zero and read ~.5. Fractions of a fixed
count of 6; no ratio. Null: ≤ .50.

**pred_c_redundancy_order_exceeds_one** — median over behaviours of the smallest k such that some k-subset carries ≥ half the joint
four-reader damage is ≥ 2. *Worked example:* §2817's OOD top-3 share of .426 implies no single reader is close to half, so the
hypothesis reads 2; a circuit with one dominant reader reads 1. Integer in [1, 5], where 5 encodes "no subset reaches half".

**pred_d_interaction_profile_is_shared** — median Pearson correlation over behaviour pairs of the 6 normalized order-2 coefficients is
≥ .50. *Worked example:* with a COMMON reader set all six keys align by construction, so this is now a well-posed six-point
correlation (in §2813 it was as few as three points and I recorded it as under-powered). If the same four MLPs interact the same way
whatever the surface form, .6–.9; if each behaviour recruits them in its own pattern, ~0. Null: ≤ 0.

**pred_e_common_set_carries_the_reads** — median over behaviours of `(joint 4-reader damage) / max(all-downstream READS damage, .5)`
is ≥ .50. *Worked example:* §2817 measured OOD top-3 shares of .32–.71 on per-behaviour sets, so a fixed set covering blocks 8–11
should read .4–.8; if the read is genuinely spread over the whole late stack, .2–.3. Denominator floored. Null: ≤ .25.

## Stated null

The four predeclared readers are additive, idiosyncratic across behaviours, and a minor part of the read: ratio ≥ 1.0, order-2 positive
fraction ≤ .50, profile correlation ≤ 0, share of READS ≤ .25. Each null is reported separately.

## Price

≤ 7 behaviours × 16 subsets × length-batches of 16 OOD rows, plus one native and one all-READS forward per batch.
Literal budget: ≤ 900 GPU forwards, 0 backwards, 0 fitted parameters, expected < 90 GPU-seconds.

## What this does NOT claim

A fixed four-reader set is not the whole downstream read, and pred_e measures exactly how much of it is missing rather than assuming
none. Positive order-2 coefficients demonstrate super-additivity, not a mechanism for it. This rung does not satisfy Codex's four-phase
integration contract and does not update any circuit record or adoption ledger.
