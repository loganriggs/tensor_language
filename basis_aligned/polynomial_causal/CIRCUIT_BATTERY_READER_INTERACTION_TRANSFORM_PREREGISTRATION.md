# CIRCUIT BATTERY — READER INTERACTION TRANSFORM (preregistration)

Registered Registered 2026-09-04 04:12Z (box clock) (box clock). Claude, LANE 1 CUDA. Rung `circuit_battery_reader_interaction_transform`.
Script: `ops/circuit_battery_reader_interaction_transform.py`. Input receipt: `circuit_battery_results.json`
(SS2809, sha 6d1eda1cc05adf72c525375a0602bbafbf9b4335653be0e410de3d69da03265c). Move 2 of MATHEMATICAL_REVIEW_2026-09-04_0404.

## Object

SS2808 and SS2809 established that a writer's downstream readers are super-additive: single-reader damages sum to about half the damage
of removing them jointly. That is a property of the DAMAGE SET FUNCTION `v(S)` over reader subsets, and the right instrument for a set
function is its Moebius (Harsanyi) transform `m(S) = sum_{T subset S} (-1)^{|S|-|T|} v(T)`: order-1 coefficients are the single-reader
damages, order-2 and above are exactly what single-component ablation misses (the hydra / self-repair effect, arXiv:2307.15771; the
under-counting is the explicit subject of arXiv:2607.01940). This rung evaluates all 16 subsets of the top-4 MLP readers of each SS2809
capable+localised behaviour on A1 rows of the frozen SELECT split, and reports a REDUNDANCY ORDER k = the smallest subset size whose
best member carries half the joint damage — a statement about how many components a compiled program must keep, not about
reconstruction error.

## Predictions

```
BARS  = {super_additive: .70, order2_positive_frac: .75, redundancy_order: 2, profile_corr: .50, top4_share: .50, floor: .5}
NULLS = {super_additive_ge: 1.0, order2_positive_frac_le: .50, profile_corr_le: 0.0, top4_share_le: .25}
```

**pred_a_readers_are_super_additive** — median over behaviours of `(sum of the 4 single-reader damages) / max(joint 4-reader damage, .5)`
is <= .70. *Worked example:* SS2808's 19 readers gave .994 / 1.914 = .52; for a top-4 set the hypothesis reads .4-.7. An additive reader
set reads exactly 1.0 and a sub-additive (saturating) one reads > 1. Numerator is a sum of signed damages (individual readers may be
negative — SS2808's mlp13 was) and the denominator is a damage FLOORED at .5 margin units, so it cannot pass through zero.
Null: >= 1.0.

**pred_b_order_two_interactions_are_positive** — median over behaviours of the fraction of the 6 pairwise Moebius coefficients that are
positive is >= .75. *Worked example:* if removing two readers together hurts more than the sum of removing each alone (backup /
self-repair), every pair coefficient is positive and this reads 1.0; if the readers are independent, the coefficients scatter around
zero and it reads ~.5. Null: <= .50. Fractions of a fixed count, no ratio of signed quantities.

**pred_c_redundancy_order_exceeds_one** — median over behaviours of the smallest k such that some k-subset carries >= half the joint
4-reader damage is >= 2. *Worked example:* a circuit with one dominant reader reads k = 1 (SS2809's numbered list has mlp8 at .51 of a
joint 1.91, i.e. below half, so k > 1 is expected here); a genuinely threshold-like set reads k = 2 or 3. Operands are damages with a
floored denominator; k is an integer in [1, 5] where 5 encodes "no subset reaches half".

**pred_d_interaction_profile_is_shared** — median Pearson correlation, over behaviour pairs, of the 6 normalized order-2 coefficients
(normalized by each behaviour's max |coefficient|) is >= .50. *Worked example:* SS2809 showed the same reader LADDER (mlp8 > mlp9 >
mlp10 > mlp11) across six surface forms; if those readers also interact the same way, the profiles correlate ~.6-.9 and the circuit is
genuinely re-used rather than coincidentally co-located. If each behaviour recruits the same components in a different interaction
pattern, ~0. Null: <= 0. Correlations are computed only over behaviour pairs sharing >= 3 defined pair keys.

**pred_e_top4_carries_the_reads** — median over behaviours of `(joint 4-reader damage) / max(all-downstream READS damage, .5)` is
>= .50, i.e. four readers are a usable summary of the whole downstream read. *Worked example:* SS2808's top-2 reached .37 of READS and
the top-3 share across the bank was .49, so four readers should read .5-.7; if the read is spread over a dozen components, .2-.3.
Null: <= .25. Denominator floored.

## Stated null

The readers are additive and idiosyncratic: ratio >= 1.0, order-2 positive fraction <= .50, profile correlation <= 0, top-4 share
<= .25. Each null is reported separately, not averaged.

## Price

<= 6 behaviours x 16 subsets x length-batches of 16 A1/SELECT rows, plus one native and one all-READS forward per batch.
Literal budget: <= 800 GPU forwards, 0 backwards, 0 fitted parameters, expected < 90 GPU-seconds.

## What this does NOT claim

The transform is over the TOP-4 readers only, so it measures the interaction structure of the dominant set, not of all 19 downstream
components; a positive order-2 coefficient shows super-additivity, not a mechanism for it. The single-reader arms are exactly the ones
SS2809 already reported, and they are re-measured here rather than imported, so any discrepancy is visible.
