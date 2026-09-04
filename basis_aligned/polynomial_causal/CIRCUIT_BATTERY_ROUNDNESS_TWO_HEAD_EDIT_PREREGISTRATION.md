# CIRCUIT BATTERY — TWO-HEAD ROUNDNESS EDIT (preregistration)

Registered 2026-09-04 06:35Z (box clock, read immediately before this line was written). Claude, LANE 1 CUDA.
Rung `circuit_battery_roundness_two_head_edit`. Script: `ops/circuit_battery_roundness_two_head_edit.py`.
Input receipt: `circuit_battery_roundness_steering_results.json` (§2845, sha 7aa2134d911b3ba2fd1144e11af93e88647522258e603611b5f4d2423ce846db).
IMMUTABLE: any change gets a new document, not an edit.

## Two things §2845 left behind

§2845 injected §2844's roundness direction into head 3 on held-out non-round prompts. As a push on the logits it worked fully —
+.90 (percent) and +.64 (bare) against the full-slice swap's +1.05 and +.51 — and a random direction at the same magnitude flipped
nothing. But it changed the argmax on only **.125** of prompts, and the reason was that the ceiling was equally low: swapping head 3's
entire output flips .083 and .000. **Head 3 is where the feature lives and head 3 alone cannot decide the behaviour**, which fits §2843
(head 3 is .45/.31 of the switch; the pair {3, 7} is .925).

It also **failed a registered sanity bound**: on the bare format the injection (+.64) beat the swap (+.51) it was supposed to be bounded
by, which the §2845 document had said in advance would mean α was fitted to a magnitude the natural difference never reaches, making
that arm an out-of-distribution push rather than the feature.

This rung addresses both: edit **both heads of the pair** with their own round-ward directions and per-head scales, and carry arms that
test the scaling directly.

**Arms** (all on held-out non-round prompts, directions and scales fitted on the other half):
`ADD_BOTH` (α·u₃ into head 3 and α·u₇ into head 7, each head's own fitted α); `ADD_ONE` (head 3 only, this rung's own re-measurement of
§2845); `ADD_HALF` (both heads at 0.5α); `ADD_RANDOM_PAIR` (seeded random unit directions in both heads at the same α);
`SWAP_PAIR` (both slices taken from the round twin — the ceiling for any edit confined to the pair).

Sign convention: flip rate is the fraction of held-out prompts whose argmax over the numeric vocabulary becomes the STEP answer, HIGHER
MEANS THE EDIT WORKED; gain is the change in `logit(step) − logit(plus-one)`. **Activation edit at run time — not a weight, no CE, no
§312 L2, nothing installs.**

## Predictions

```
BARS  = {flip: .30, gain_over_one_head: .30, random_flip: .05, alpha_mono: 0.0}
NULLS = {flip_le: .10, gain_over_one_head_le: 0.0, random_flip_ge: .30}
SS2845 references: one-head flip .125, one-head gain .7704
```

**pred_a_two_heads_flip_more_than_one** — median flip rate under `ADD_BOTH` ≥ .30. *Worked example:* §2843 put head 3 at .45/.31 of the
switch and the pair at .925, so editing both should roughly double the push and carry a real fraction of prompts across the boundary:
.3–.6 if the missing mass was simply head 7, and ≈ .125 (unchanged from §2845) if the rest of the switch lives outside the pair —
which §2842's diffuse tail (top-3 components only .255 of positive recovery) makes a live possibility. A rate over held-out prompts.
Null: ≤ .10.

**pred_b_two_head_gain_exceeds_one_head** — median of `gain(ADD_BOTH) − gain(ADD_ONE)` ≥ .30 nats of logit difference.
*Worked example:* if head 7 carries an independent share, the two-head gain lands near the sum of the parts and the difference is
.3–.9; if head 7's contribution is redundant with head 3's, ~0. Both arms measured in this rung on the same rows, so this is a paired
difference and does not depend on §2845's absolute values. Null: ≤ 0.

**pred_c_the_sanity_bound_now_holds** — `gain(SWAP_PAIR) ≥ gain(ADD_BOTH)` in BOTH formats. *Worked example:* the swap is a strictly
larger intervention on the same two slices, so it must dominate; §2845's version of this bound FAILED on the bare format and that
failure is the reason this rung exists. If it fails again, the per-head α is still over-scaled and the flip rates in this rung are not
interpretable either — which I will report as such rather than tune α until the bound passes.

**pred_d_random_pair_is_inert** — median flip rate under `ADD_RANDOM_PAIR` ≤ .05. *Worked example:* §2845 measured 0 for a single random
direction; two random directions at the same magnitudes should also flip essentially nothing. Without this, pred_a could be satisfied by
perturbation damage. Null: ≥ .30.

**pred_e_the_edit_is_monotone_in_alpha** — `gain(ADD_BOTH) > gain(ADD_HALF)` in BOTH formats. *Worked example:* a genuine feature push
should increase with magnitude over this range; a non-monotone response would mean α sits past the point where the edit still behaves
like the feature, which is the §2845 failure's diagnosis and would qualify everything else here.

## Stated null

Two heads do no better than one (flip ≤ .10, gain difference ≤ 0), or the random pair flips as much. That would say the roundness switch
cannot be driven from the {3, 7} pair at all and the deciding mass is elsewhere in attention 8 or downstream — a clean negative for the
"editable" half of the campaign's goal at this granularity.

## Price

2 formats × (≈12 fit pairs + ≈12 held-out pairs) × (2 native + 5 arms) forwards, batched by token length.
Literal budget: ≤ 500 GPU forwards, 0 backwards, **516 declared fitted parameters** (two 128-dimensional directions plus two scales per
format). < 60 GPU-seconds. Uses `ops/fastload.py` (9.0× end-to-end, forward-identical).

## What this does NOT claim

Two heads of one component; the rest of attention 8 and everything downstream are untouched, so a low flip rate bounds what the PAIR can
do and not what the model's roundness computation is. Activation edit at run time, not a weight edit, and nothing touches the §312
frontier. One step size, one digit range, two formats, §2842's purpose-built pairs rather than the bank's frozen splits. Does not satisfy
Codex's four-phase integration contract; updates no circuit record.
