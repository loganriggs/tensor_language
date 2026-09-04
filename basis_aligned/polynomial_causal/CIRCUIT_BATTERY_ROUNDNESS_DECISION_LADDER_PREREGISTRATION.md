# CIRCUIT BATTERY — ROUNDNESS DECISION LADDER (preregistration)

Registered 2026-09-04 06:38Z (box clock, read immediately before this line was written). Claude, LANE 1 CUDA.
Rung `circuit_battery_roundness_decision_ladder`. Script: `ops/circuit_battery_roundness_decision_ladder.py`.
Input receipts: `circuit_battery_roundness_two_head_edit_results.json` (§2846, sha 7c96fe4a4108f3b4a55bb51d2d547f738642c1d733e9857ec50aa3b8b68f3dd5)
and `circuit_battery_roundness_localisation_results.json` (§2842, sha 331454aac1ce218d9194255e19c81c53eca38d99cc6c2b685ff2d9e0ac12788c),
the latter supplying each format's top non-attn8 components. IMMUTABLE.

## Object

§2843 and §2844 put the roundness FEATURE in heads {3, 7} of attention 8, along one 128-dimensional direction. §2846 then showed that
the strongest intervention confined to that pair — swapping both slices outright — flips only **.208** of held-out non-round prompts to
step continuation, and that an additive edit matches but cannot exceed it. **The feature is in the pair; the decision is not.**

§2842 measured logit-difference RECOVERY for all 36 components but never a flip rate, so nothing in this lineage says how much of the
model must move before the behaviour actually changes. This rung measures that as a curve: the flip rate under progressively larger
donor patches, from the pair to the entire model.

**Ladder, fixed before the run:** `pair` (heads {3, 7} of attention 8) → `component` (all of attention 8) → `component_plus_readers`
(attention 8 plus that format's top three non-attn8 components from §2842's published order) → `everything` (all 36 components). The
donor is always the ROUND twin and the edited prompt always the NON-ROUND one.

Sign convention: flip rate is the fraction of held-out non-round prompts whose argmax over the numeric vocabulary becomes the STEP
answer; HIGHER MEANS THE INTERVENTION DECIDED THE BEHAVIOUR. **Interchange patching at run time — no weight change, no CE, no §312 L2,
nothing installs.**

## Predictions

```
BARS  = {component: .40, with_readers: .70, everything: .95, pair_tol: .10}
NULLS = {component_le: .25, with_readers_le: .40, everything_le: .70}
§2846 reference: pair flip .208
```

**pred_a_the_component_beats_the_pair** — median flip rate for `component` ≥ .40. *Worked example:* §2843 put the pair at .925 of
attention 8's positive HEAD recovery, so if the component's remaining seven heads were irrelevant the component would flip ≈ .208 and
this fails; if the decision needs the whole component, .4–.7. Either outcome localises the decision one step further, which is why the
bar sits between them. A rate over held-out prompts. Null: ≤ .25.

**pred_b_adding_readers_beats_the_component** — median flip rate for `component_plus_readers` ≥ .70. *Worked example:* §2842's map had
a broad, format-dependent tail (top-3 components only .255 of positive recovery), so if the decision is distributed across the readers
that consume attention 8's write, adding three of them should carry most of the remaining distance: .7–.95. If it stalls near the
component's own value, the decision is not in those readers either and the next rung has to sweep further. Null: ≤ .40.

**pred_c_everything_reproduces_the_donor** — median flip rate for `everything` ≥ .95. *Worked example:* patching all 36 components makes
the base run compute the donor's function, so it should flip essentially every prompt; §2842 measured .978 logit recovery for the same
arm. A value well below .95 would mean the patch set is incomplete (embeddings, final norm) and no rung on this ladder is readable. This
is the instrument check.

**pred_d_the_ladder_is_monotone** — flip rate is non-decreasing along pair ≤ component ≤ component+readers ≤ everything, in BOTH
formats. *Worked example:* each rung strictly contains the previous one, so a decrease means an interference effect large enough to
undo a superset patch, which would qualify every reading here and is worth catching explicitly rather than assuming away.

**pred_e_the_pair_replicates** — |median `pair` flip − .208| ≤ .10, against §2846's published value. *Worked example:* the same
intervention measured in a different script on the same pairs should reproduce; a gap means the two rungs' patch semantics differ and
§2846's conclusion cannot be carried into this one.

## Stated null

The component is no better than the pair (≤ .25), adding readers adds nothing (≤ .40), or the full patch fails to reproduce the donor
(≤ .70). The first two together would say the decision is not localisable by component patching at all on these prompts, which — given
§2822–§2824 found the analogous negative for the reader side — would be a coherent and reportable outcome.

## Price

2 formats × ≈24 pairs × 4 ladder rungs × 3 forwards each, batched by token length.
Literal budget: ≤ 700 GPU forwards, 0 backwards, **0 fitted parameters**, < 90 GPU-seconds. Uses `ops/fastload.py`.

## What this does NOT claim

Whole-component granularity above the first rung; no head or direction decomposition of anything but the pair. The readers are taken
from §2842's published order for each format, so `component_plus_readers` is a fixed set, not a search. One step size, one digit range,
two formats, §2842's purpose-built pairs rather than the bank's frozen splits. Does not satisfy Codex's four-phase integration contract;
updates no circuit record.
