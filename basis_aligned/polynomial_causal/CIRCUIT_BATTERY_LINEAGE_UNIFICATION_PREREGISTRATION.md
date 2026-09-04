# CIRCUIT BATTERY — LINEAGE UNIFICATION (preregistration)

Registered 2026-09-04 06:44Z (box clock, read immediately before this line was written). Claude, LANE 1 CUDA.
Rung `circuit_battery_lineage_unification`. Script: `ops/circuit_battery_lineage_unification.py`.
Input receipts: `circuit_battery_roundness_decision_ladder_results.json` (§2847, sha 7d6f806ddb5258518f9893cebbd2aa8b8d35668f6e1a34e97e7a26cbd0585fe7)
and `circuit_battery_v2_bank21_results.json` (§2840, sha 7c1db82061bb6cac010fa7d2141cfc8e2faa4330759bdfe5aed299bab9a94d50).
IMMUTABLE: any change gets a new document, not an edit.

## Object

Two lineages met at §2847 and nobody has tested whether they are one mechanism.

- **Successor lineage** (§2808, §2818, §2819, §2821): attention 8 writes a context-blind function of the last visible item;
  mlp8 > mlp9 > mlp10 > mlp11 read it as a 2-of-4 redundant threshold with specificity rising in depth, and nothing past
  mlp11 is causally live.
- **Roundness lineage** (§2841–§2847): the same attention 8, in the same heads {3, 7}, along one 128-dimensional direction,
  carries whether the last item is ROUND; and the DECIDING readers are mlp8, mlp9 (+mlp1), not the writer.

So attention 8 carries **two** features and the same MLP stack performs **two** computations (+1 and +step). One function of
two inputs is a much smaller compiled program than two functions sharing hardware, and this rung tests which it is.

**Two measurements on the same reader set** `{mlp8, mlp9, mlp10, mlp11, mlp1}`, fixed before the run:

1. **PROFILE** — each reader's contribution to the roundness decision (§2842's logit-difference recovery, on §2842's
   held-out percent pairs) and to the successor margin (removal damage on the BANK's frozen OOD rows of
   `numbered_list.index_successor`), then the rank correlation between the two profiles.
2. **CHANNEL** — the discriminating arm: project §2844's roundness direction OUT of head 3's slice and run the SUCCESSOR
   task. Roundness is irrelevant to a numbered list, so damage there means the two computations share a channel; no damage
   means attention 8 writes two separable features into one stream and the MLP stack reads them independently.

Sign convention: successor damage d_m = m_NATIVE − m_arm in margin units, POSITIVE = the arm HURTS the successor answer;
roundness contribution is a logit-difference recovery, HIGHER = the reader carries more of the switch. **No CE and no §312
L2 — the frontier's L2 is CE ADDED ABOVE THE REAL MODEL where LOWER IS BETTER (§2135, norm-2304 at 2.6735), and nothing
here installs or may be quoted as one.**

## Predictions

```
BARS  = {rho: .50, separable: .10, random: .05, repro: .30, floor: .5}
NULLS = {rho_le: 0.0, separable_ge: .40, random_ge: .30}
```

**pred_a_reader_profiles_correlate** — Spearman between the five readers' roundness recovery and their successor damage
≥ .50. *Worked example:* §2819 ranked the successor readers mlp8 > mlp9 > mlp10 > mlp11 and §2847 put the roundness
decision in mlp8, mlp9 (+mlp1); if the same readers serve both in the same order, ρ lands .7–1.0 over five points. If the
two computations recruit the stack differently, ρ ≈ 0 or negative. Five paired values — thin, and reported as such rather
than dressed up; it is the CHANNEL arm below that carries the weight. Null: ≤ 0.

**pred_b_roundness_direction_is_separable** — `d_m(project out the roundness direction) / max(d_m(remove all five
readers), .5)` ≤ .10. *Worked example:* a numbered list's answer does not depend on whether the label is a multiple of ten,
so if attention 8 writes identity and roundness into separable directions, removing the roundness one costs ~0–.05 of what
removing the whole reader set costs. If the two features share a channel, .3–.8. Both operands are damages in the same
margin units with a floored denominator that cannot pass through zero. **This is the discriminating prediction.**
Null: ≥ .40.

**pred_c_the_top_reader_is_shared** — the reader with the highest roundness recovery is the reader with the highest
successor damage. *Worked example:* §2819 and §2847 both point at mlp8, so this should be TRUE; a mismatch would say the
stack's leading reader differs by computation even if the sets overlap. Boolean over five readers.

**pred_d_random_direction_is_inert** — |`d_m(project out a random direction of the same norm)`| / max(full, .5) ≤ .05.
*Worked example:* removing an arbitrary direction from a 128-dimensional head slice should barely move a successor
margin, ~.00–.02. This is the control that makes pred_b's number readable: without it, a small separability fraction could
just mean that removing any one direction is harmless. Null: ≥ .30.

**pred_e_full_removal_reproduces_the_battery** — |`d_m(remove attention 8)` − §2840's FULL value for this task| /
max(that value, .5) ≤ .30. *Worked example:* §2840 measured attention 8's whole-write removal on these same frozen OOD
rows; this rung re-measures it with different code, so agreement within 30% says the two instruments are the same
measurement and the profiles above can be compared to the earlier lineage. A large gap and nothing here is comparable to
§2818/§2819. Instrument check.

## Stated null

The profiles do not correlate (ρ ≤ 0) and removing the roundness direction damages the successor task (≥ .40) — i.e. the
two computations share a channel and attention 8's write is not feature-separable. That is a perfectly good outcome and
arguably the more interesting one: it would say the MLP stack computes a single function whose behaviour depends on a
roundness-modulated input, rather than two functions selected by a flag.

## Price

≈12 held-out percent pairs × (2 native + 5 reader patches) + 16 successor OOD rows × (1 native + 5 reader removals + 4
arms), batched by token length. Literal budget: ≤ 400 GPU forwards, 0 backwards, **128 declared fitted parameters** (the
roundness direction, fitted on the percent pairs' fit half only). < 60 GPU-seconds. Uses `ops/fastload.py`.

## What this does NOT claim

Five readers, one successor task, one roundness format (percent). Projecting a direction out of one head's slice is not the
same as ablating a feature everywhere it might be represented, so a clean pred_b bounds separability AT THAT SITE and not
in general. Nothing installs; no L2. Does not satisfy Codex's four-phase integration contract; updates no circuit record.
