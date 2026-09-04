# CIRCUIT BATTERY — ROUNDNESS STEERING (preregistration)

Registered 2026-09-04 06:28Z (box clock, read immediately before this line was written). Claude, LANE 1 CUDA.
Rung `circuit_battery_roundness_steering`. Script: `ops/circuit_battery_roundness_steering.py`.
Input receipt: `circuit_battery_roundness_direction_results.json` (§2844, sha 232e023b58e6b429d936143d1b139c6c417d8efe77fa7ef1716eb4c9264d3205).
IMMUTABLE: any change gets a new document, not an edit.

## Object: localisation is not manipulation

§2841 established that roundness switches this model between two behaviours with no overlap — multiples of ten continued BY THE STEP
(1.000, 6/6), everything else by PLUS ONE (.313, and step .000 on 0/48). §2842, §2843 and §2844 localised the switch to attention 8,
then to heads {3, 7}, then to a single 128-dimensional direction inside head 3 carrying **.874** of that head's effect on held-out pairs
and transporting across surfaces at |cos| **.974**.

Every one of those is an observation. The campaign's goal names three things — predictive, **manipulable**, editable — and nothing in
this lineage has yet tested the middle one. This rung does: **inject the vector into head 3's output on a NON-ROUND prompt the model
would continue by plus-one, and ask whether it switches to step continuation**; then inject its negative into a ROUND prompt and ask
whether it switches the other way. The direction and its single scale coefficient are fitted on one half of the pairs; every arm is
scored on the held-out half.

Arms: `NATIVE`; `ADD` (add α·u to head 3's slice at every position); `ADD_RANDOM` (a seeded random unit direction in the same
128-dimensional space, same α); `ADD_NEG` (−u, injected into round prompts); `SWAP` (head 3's whole slice taken from the round twin —
§2843's intervention, and the natural upper bound on what any edit inside that head can do).

Sign convention: **flip rate** is the fraction of held-out prompts whose argmax over the numeric candidate vocabulary becomes the STEP
answer (forward edit) or the PLUS-ONE answer (reverse edit); HIGHER MEANS THE EDIT WORKED. **Logit gain** is the change in
`logit(target) − logit(the other answer)`. **No CE and no §312 L2; nothing installs into the frontier**, and this rung edits an
activation at run time, not a weight.

## Predictions

```
BARS  = {flip: .50, random_flip: .05, rec_frac: .50, reverse_flip: .30}
NULLS = {flip_le: .10, random_flip_ge: .30, rec_frac_le: .15, reverse_flip_le: .05}
```

**pred_a_injection_flips_non_round_prompts** — median over the two formats of the flip rate under `ADD` ≥ .50 on held-out prompts.
*Worked example:* §2844's direction carries .874 of head 3's slice effect and §2843 measured that slice at .45/.31 of the whole switch,
so a well-scaled injection should move a good majority of prompts across the argmax boundary: .5–.9 if the vector is a handle, .0–.15 if
it merely correlates with the switch without being able to drive it. A rate over held-out prompts, not a ratio. Null: ≤ .10.

**pred_b_random_direction_does_not** — median flip rate under `ADD_RANDOM` at the same α ≤ .05. *Worked example:* §2844 measured a
random direction's recovery at −.0005, so a random injection of the same magnitude should essentially never flip a prompt: .00–.02.
This is the control that separates "this vector is a handle" from "perturbing the head at this magnitude breaks it into the other
behaviour". Null: ≥ .30 — if a random direction flips prompts too, the edit is damage, not steering, and pred_a means nothing.

**pred_c_injection_recovers_the_full_swap** — median over formats of `logit_gain(ADD) / logit_gain(SWAP)` ≥ .50.
*Worked example:* `SWAP` replaces head 3's entire 128-dimensional output with the round twin's and is the ceiling for any edit confined
to that head; if one direction is most of what the head contributes, the ratio lands .5–.9 (§2844 measured .874 for the analogous
quantity under interchange). The denominator is a measured, non-degenerate gain (§2843 reported the slice arm live in both formats), and
it is reported alongside so a small denominator would be visible rather than inflating the ratio. Null: ≤ .15.

**pred_d_the_reverse_edit_works** — median flip rate under `ADD_NEG` on ROUND prompts ≥ .30. *Worked example:* a genuine handle should
be bidirectional — subtracting the roundness vector from a round prompt should push it toward plus-one. The bar is lower than pred_a's
because the round regime is the model's confident one (§2841: 1.000 accuracy, 6/6) and pushing a confident behaviour off is harder than
nudging an uncertain one. If the edit is one-directional only, that is worth knowing and is what a FALSE here would say. Null: ≤ .05.

**pred_e_full_swap_bounds_the_injection** — `logit_gain(SWAP) ≥ logit_gain(ADD)` in BOTH formats.
*Worked example:* the swap is a strictly larger intervention on the same slice, so it should dominate; if the injection beats it, α has
been fitted to a magnitude the natural difference never reaches and pred_a is measuring an out-of-distribution push rather than the
feature. This is the sanity bound, and per the standing rule it will not be retired in the entry that benefits from ignoring it.

## Stated null

The vector is an observation and not a handle: injection flips ≤ .10 of prompts, or a random direction flips as many, or the injection
recovers ≤ .15 of the swap. Then §2844's direction predicts the switch without being able to drive it, which is a meaningful negative for
the "manipulable" half of this campaign's goal and would be recorded as such.

## Price

2 formats × (≈12 fit pairs + ≈12 held-out pairs) × (2 native + 4 edited) forwards, batched by token length.
Literal budget: ≤ 400 GPU forwards, 0 backwards, **258 declared fitted parameters** (one 128-dimensional direction plus one scale per
format). < 60 GPU-seconds. This rung also adopts `ops/fastload.py` (§ops log 06:24Z, 5.5× faster load, bit-identical over 218 tensors).

## What this does NOT claim

An activation edit at run time, not a weight edit and not an installed approximation — nothing here touches the §312 frontier. One head
of one component, one direction, one scale, one step size and one digit range (inherited from §2841). Two formats. The pairs are
§2842's construction, not the bank's frozen splits, so no number here may be quoted as a bank capability. A successful flip shows the
vector drives the switch on these prompts; it does not show the model has no other route to the same behaviour. Does not satisfy Codex's
four-phase integration contract; updates no circuit record.
