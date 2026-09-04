# CIRCUIT BATTERY — ABSOLUTE-RANKED CONSTANT SET (preregistration)

Registered 2026-09-04 05:52Z (box clock, read immediately before this line was written). Claude, LANE 1 CUDA.
Rung `circuit_battery_absolute_ranked_constant_set`. Script: `ops/circuit_battery_absolute_ranked_constant_set.py`.
Input receipt: `circuit_battery_constant_write_census_results.json` (§2836, sha ab7cc4a16d879c981772fe4e65aab3c28b733000436fdbd49600dc21943e78e0).
IMMUTABLE: any change gets a new document, not an edit.

## Why this exists: it is the correction, run as an experiment

§2837 found that constants compose — four writes as fixed vectors cost 1.386 nats where deleting them costs 4.432 — but that the set I
chose was **twice as expensive as a random one** (.681). The cause was that §2836's ranking statistic, `recovered = 1 − const/zero`, is
a RATIO: it promoted attn0 on a small denominator, and adding attn0 took the joint cost from .377 to 1.257. That was the fifth time in
one night that a normalised quantity standing in for an absolute one misled me (§2820, §2821, §2825, §2826, §2837).

The fix is one line — **rank by the absolute nats a constant costs, `const_damage`, not by the recovery fraction** — and this rung tests
whether the fix works rather than assuming it. All three sets (absolute-ranked, §2837's ratio-ranked, and a fresh random control at seed
2837) are scored side by side at k ∈ {1,2,3,4,6,8}, means fitted on 24 documents, everything scored on 24 DISJOINT ones. The two
comparison values are taken from §2837's published receipt and are not re-measured: ratio-ranked k=4 = 1.386037 nats, random k=4 =
.681264 nats.

Sign convention: d_ce = CE_arm − CE_NATIVE in nats, POSITIVE = the arm HURTS. **Not §312 L2 (CE added above the real model by an
installed approximation, LOWER IS BETTER, frontier norm-2304 at 2.6735, §2135); nothing installs; DIAGNOSTICS only; metric-constructed
bases/spans remain CLOSED (§2118 lineage).**

## Predictions

```
BARS  = {beat_ratio: .30, beat_random: 0.0, n_cheap: 6, cheap_nats: 1.50, superadd: 0.0, ce_tol: .01}
NULLS = {beat_ratio_le: 0.0, beat_random_ge: .30, n_cheap_le: 3, superadd_le: -.20}
```

**pred_a_absolute_beats_ratio_ranking** — `1.386037 − d_ce(absolute-ranked k=4)` ≥ .30 nats. *Worked example:* the absolute ranking
selects the four writes whose individual constants cost fewest nats, which by construction excludes attn0 (whose constant cost is large
in absolute terms); if that was the whole problem, the absolute set lands .3–.8 and this reads .6–1.1. If it reads ≤ 0 then the ratio
was not the culprit and the real cause is an interaction the marginal numbers cannot see either way — which would be worth knowing before
any further greedy selection anywhere in this campaign. A DIFFERENCE of two damages in the same units. Null: ≤ 0.

**pred_b_absolute_beats_random** — `.681264 − d_ce(absolute-ranked k=4)` ≥ 0. *Worked example:* this is the real test. §2837's random
set beat my chosen set; a principled ranking has to at least match random before it can be called principled. Expected .0–.5 if the fix
works, negative if the marginal-cost ranking is ALSO worse than chance — in which case the honest conclusion is that no per-component
statistic predicts joint replaceability and only joint search does, and I will write that rather than try a third ranking.
Null: absolute is worse than random by ≥ .30.

**pred_c_more_writes_can_be_constants** — the largest k in {1,2,3,4,6,8} whose absolute-ranked joint constant costs ≤ 1.50 nats is ≥ 6.
*Worked example:* §2837 located six writes at 1.43 nats using the bad ranking; a better ranking should reach six comfortably and
possibly eight. If it reads ≤ 3, fewer writes are constant-able than §2837 suggested and that section's "six" was luck. Integer in
{0, 1, 2, 3, 4, 6, 8}. Null: ≤ 3.

**pred_d_composition_stays_super_additive** — `d_ce(absolute-ranked k=4) − Σ d_ce(individual constants of those four)` ≥ 0.
*Worked example:* §2837 measured +.646 for its set, and §2818 found the same qualitative structure for reader removals; a better-chosen
set should still compound rather than cancel, so ≥ 0 and plausibly +.2 to +.6. A negative value would mean the chosen constants' errors
partly offset one another, which would be a genuinely new fact about this model. Null: ≤ −.20.

**pred_e_instrument_reproduces_native_ce_matched** — |manual forward CE − the model module's own CE on the SAME chunk| ≤ .01 nats.

## Stated null

The absolute ranking is no better than the ratio ranking and still loses to random, few writes can be constants, and the composition is
sub-additive. That would say per-component measurements of any kind do not predict joint replaceability in this model, and the only
honest route is joint search — which this rung does not attempt and which I would register separately rather than improvise.

## Price

6 values of k × 3 sets × (3 fit chunks + 3 const chunks) plus a zero arm per k on 24+24 documents, plus native and instrument passes.
Literal budget: ≤ 800 GPU document-forwards, 0 backwards, **≈ 82,944 declared fitted parameters** (mean vectors, refitted per arm on the
fit documents only). < 4 GPU-minutes.

## What this does NOT claim

Ranking by marginal absolute cost is still a greedy heuristic, not a joint search; a better set may exist and this rung will not find it.
One corpus, one held-out slice of the same frozen cache. Nothing installs and no L2 numbers. Does not satisfy Codex's four-phase
integration contract; updates no circuit record.
