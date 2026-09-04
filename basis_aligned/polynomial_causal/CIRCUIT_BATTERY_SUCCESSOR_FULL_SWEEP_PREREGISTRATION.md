# CIRCUIT BATTERY — SUCCESSOR FULL SWEEP (preregistration)

Registered 2026-09-04 06:48Z (box clock, read immediately before this line was written). Claude, LANE 1 CUDA.
Rung `circuit_battery_successor_full_sweep`. Script: `ops/circuit_battery_successor_full_sweep.py`.
Input receipts: `circuit_battery_lineage_unification_results.json` (§2848, sha f9a467b6ea48fc6204c29c36eb712aad3ec98e4724b1b87ad6e1d300fef6e669)
and `circuit_battery_v2_bank21_results.json` (§2840, sha 7c1db82061bb6cac010fa7d2141cfc8e2faa4330759bdfe5aed299bab9a94d50).
IMMUTABLE: any change gets a new document, not an edit.

## Why this rung exists, and what it says about the last forty sections

§2848 measured mlp1's removal on the numbered-list successor at **3.504** margin units — seven times mlp8's .503, and MORE
than removing attention 8's write entirely (2.646). **mlp1 is upstream of layer 8 and cannot read attention 8's write**, so
§2818, §2819 and §2821 could never have found it: those sections swept READERS of that write by construction. The successor
circuit as this campaign has described it is missing a term larger than the writer it was built around, and mlp1 entered the
picture only because §2847's roundness ladder happened to name it.

But whole-component ablation of an early MLP damages everything, so raw damage cannot separate "part of this circuit" from
"the model needs it to function". This rung therefore sweeps **all 36 components** on the bank's frozen OOD rows and scores
each one the way the battery scores a writer — A1 damage together with the answer-preserving family P and the copy control
C — with the admissibility gate §2821 forced after §2820 crowned an inert head. Two tasks (`numbered_list.index_successor`,
`paren_list.index_successor`) so that the ranking can be checked for task stability rather than read off one.

Sign convention: damage d_m = m_NATIVE − m_arm, POSITIVE = the arm HURTS that family's own answer; selectivity ratio =
max(|d_P|, |d_C|) / max(d_A1, .5), LOWER IS MORE SELECTIVE. **No CE and no §312 L2 — the frontier's L2 is CE ADDED ABOVE THE
REAL MODEL where LOWER IS BETTER (§2135, norm-2304 at 2.6735); nothing here installs or may be quoted as one.**

## Predictions

```
BARS  = {mlp1_tol: .30, not_selective: .75, selective: .25, admit: .25, rho: .50, repro: .30, floor: .5}
NULLS = {mlp1_selective_le: .25, upstream_selective_none: 0, rho_le: 0.0}
§2848 reference: mlp1 damage 3.504
```

**pred_a_mlp1_damage_replicates** — |mlp1's A1 damage − 3.504| / 3.504 ≤ .30, on the same task and split.
*Worked example:* §2848 measured it with different code on the same frozen OOD rows, so agreement inside 30% says the two
instruments agree and the rest of this rung is comparable to it. A large gap and nothing here can be read against §2848.

**pred_b_mlp1_is_not_selective** — mlp1's selectivity ratio ≥ .75. *Worked example:* this is registered as the DEFLATING
prediction against my own §2848 excitement. An early MLP that every downstream computation depends on should damage the
answer-preserving family P and the copy control C about as much as the target, giving .8–1.1. If instead it reads ≤ .25,
mlp1 is doing something specific to numbered-list succession before attention 8 runs, which would be a genuinely new
circuit component and the more interesting outcome. Both operands are damages in the same units with a floored denominator.
Null: ≤ .25.

**pred_c_some_upstream_component_is_selective** — at least one component at layer < 8 is ADMISSIBLE (A1 damage ≥ .25 ×
attention 8's) and has selectivity ratio ≤ .25. *Worked example:* if the successor computation genuinely begins before
layer 8, some early component should be both live and task-specific; §2829 already found attention 5 gating the answer
class, but that is a type gate and should NOT be selective for one numeric task. So this prediction expects 0–2 and its
failure would say the pre-layer-8 contribution is entirely generic. Count over the 18 components below layer 8.

**pred_d_the_ranking_is_task_stable** — Spearman between the 36 components' A1 damage on `numbered_list.index_successor`
and on `paren_list.index_successor` ≥ .50. *Worked example:* the two tasks are the same computation in two surfaces
(§2840: both capable, both attn8-written), so a real component ranking should transport at .7–.95; if it reads ≈ 0 the
ranking is surface noise and no single-task sweep in this campaign means anything. Null: ≤ 0.

**pred_e_attn8_reproduces_the_battery** — |attention 8's A1 damage − §2840's FULL value for this task| / that value ≤ .30.
*Worked example:* NOTE the arms differ by construction — §2840 removes the writer's FINAL-POSITION write from every reader
edge, while this rung zeroes the component at every position — so exact agreement is not expected and the bar is loose.
It is a scale check, not an identity check, and it is registered as such rather than as the tight instrument bound §2848
was able to use.

## Stated null

mlp1 is selective after all (≤ .25), no upstream component is both live and selective, and the ranking does not transport
across the two surfaces. The first of those would overturn this rung's framing in the most interesting direction and is
registered so that it can.

## Price

2 tasks × 3 families × 24 rows × 36 component ablations plus natives, batched by token length.
Literal budget: ≤ 6,000 GPU forwards, 0 backwards, **0 fitted parameters**, < 3 GPU-minutes. Uses `ops/fastload.py`.

## What this does NOT claim

Whole-component ablation at every position — a blunt instrument that says a component matters, not what it computes, and
one that cannot distinguish a circuit member from a general dependency except through the P/C controls used here. Two
tasks, one split, one bank. Nothing installs; no L2. Does not satisfy Codex's four-phase integration contract; updates no
circuit record.
