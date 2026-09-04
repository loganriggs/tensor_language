# CIRCUIT BATTERY — SAVING-RANKED CONSTANT SET (preregistration)

Registered 2026-09-04 05:56Z (box clock, read immediately before this line was written). Claude, LANE 1 CUDA.
Rung `circuit_battery_saving_ranked_constant_set`. Script: `ops/circuit_battery_saving_ranked_constant_set.py`.
Input receipt: `circuit_battery_constant_write_census_results.json` (§2836, sha ab7cc4a16d879c981772fe4e65aab3c28b733000436fdbd49600dc21943e78e0).
IMMUTABLE: any change gets a new document, not an edit.

## The rule this rung tests

Two rankings for choosing which writes to replace with constants have now failed, in opposite directions:

- **§2837** ranked by the ratio `recovered = 1 − const/zero`. It promotes components with tiny DELETION costs (attn0), and its joint
  set at k = 4 cost **1.386 nats** — twice as expensive as a random set (.681).
- **§2838** ranked by the absolute `const` cost. It promotes components with tiny EVERYTHING (attn15, attn17, attn12, …), and its joint
  set at k = 4 cost .070 nats while costing only **.060 to delete outright** — at k = 8 the constants were actually WORSE than deletion
  (.391 vs .295). All five of that rung's predictions passed and the result was vacuous.

Both are single-endpoint statistics for a two-sided question. The quantity a compiled program wants is the **SAVING**: the nats avoided
by writing a constant instead of writing nothing, `zero_damage − const_damage`. On §2836's per-component measurements the ordering is
attn1 2.220, attn5 2.083, mlp0 1.920, mlp16 .739, and every component §2838 selected saves under .01. This rung ranks by that difference
and measures the joint arms, k ∈ {1,2,3,4,6,8}, means fitted on 24 documents, everything scored on 24 DISJOINT ones, with §2837's
ratio-ranked value (1.386) and §2838's absolute-ranked value (.070) carried in from their receipts as references.

**The predictions below are written so that a degenerate set CANNOT pass them** — that is the whole point of pred_c, and it is the check
§2838 lacked. Sign convention: d_ce = CE_arm − CE_NATIVE in nats, POSITIVE = the arm HURTS. **Not §312 L2 (CE added above the real model
by an installed approximation, LOWER IS BETTER, frontier norm-2304 at 2.6735, §2135); nothing installs; DIAGNOSTICS only;
metric-constructed bases/spans CLOSED (§2118 lineage).**

## Predictions

```
BARS  = {saving_k4: 4.0, beat_both: 0.0, delete_cost: 3.0, superadd: 0.0, ce_tol: .01}
NULLS = {saving_k4_le: 1.0, beat_both_le: -.30, delete_cost_le: 1.0, superadd_le: -.20}
```

**pred_a_saving_set_is_worth_keeping** — at k = 4, `d_ce(these four DELETED) − d_ce(these four as CONSTANTS)` ≥ **4.0 nats**.
*Worked example:* the four highest individual savings are attn1 2.220, attn5 2.083, mlp0 1.920, mlp16 .739, summing to 6.96; joint
super-additivity will erode that, so 4.0–6.0 if the marginal savings survive composition and ≤ 1.0 if they do not. This is the quantity
that matters and it is registered as an ABSOLUTE number of nats, not a fraction of anything. Null: ≤ 1.0.

**pred_b_saving_beats_both_earlier_rankings** — `1.386037 − d_ce(saving-ranked k = 4 CONSTANT)` ≥ 0, i.e. at least as cheap as §2837's
ratio-ranked set. *Worked example:* the saving-ranked set contains the model's four most expensive-to-delete components, so its constant
arm will not be as cheap in absolute nats as §2838's degenerate set (.070) and it need not be — what it must not do is cost MORE than
the ratio-ranked set that this whole lineage set out to improve on. Expected .3–1.2. Null: worse than §2837 by ≥ .30.

**pred_c_the_set_is_not_degenerate** — `d_ce(these four DELETED)` ≥ **3.0 nats**. *Worked example:* this is the clause §2838 did not
have. §2838's selected four cost .060 nats to delete, so its passing predictions meant nothing; the saving-ranked four should cost
6–9 nats to delete because they include mlp0, attn1 and attn5, three of the four most expensive components in the model (§2830). If
this reads below 3.0 the set is unimportant and NOTHING else in this rung may be read as a compilation result, whatever the other
predicates say. Null: ≤ 1.0.

**pred_d_composition_stays_super_additive** — `d_ce(joint constants) − Σ d_ce(individual constants)` ≥ 0. *Worked example:* §2837
measured +.646 and §2838 +.019; expected +.2 to +1.5 here since the components are larger. A negative value would mean these particular
constants' errors partly cancel. Null: ≤ −.20.

**pred_e_instrument_reproduces_native_ce_matched** — |manual forward CE − the model module's own CE on the SAME chunk| ≤ .01 nats.

## Stated null

The savings do not survive composition (≤ 1.0 nat at k = 4), the set is no better than §2837's, or it turns out degenerate after all.
Then no per-component statistic predicts joint constant-replaceability in this model and the honest route is joint search, which I would
register separately rather than improvise — the mistake this lineage has already made twice.

## Price

6 values of k × 3 sets (saving-ranked, §2837's ratio order, random at seed 2838) × (3 fit + 3 const chunks), plus a delete arm per k,
on 24 + 24 documents. Literal budget: ≤ 800 GPU document-forwards, 0 backwards, **≈ 82,944 declared fitted parameters**. < 4 GPU-minutes.

## What this does NOT claim

Ranking by marginal saving is still greedy, not a joint search. One corpus, one held-out slice of the same frozen cache. Nothing
installs; no L2 numbers. Does not satisfy Codex's four-phase integration contract; updates no circuit record.
