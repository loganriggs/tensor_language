# CIRCUIT BATTERY — ROUNDNESS LOCALISATION (preregistration)

Registered 2026-09-04 06:12Z (box clock, read immediately before this line was written). Claude, LANE 1 CUDA.
Rung `circuit_battery_roundness_localisation`. Script: `ops/circuit_battery_roundness_localisation.py`.
Input receipt: `circuit_battery_roundness_capability_results.json` (§2841, sha f099b983dcb1fa35c112d6c1dd3565f6024bb6ea5836b72e4bb050c537ae6923).
IMMUTABLE: any change gets a new document, not an edit.

## Object

§2841 established that roundness switches this model between two behaviours: with the step fixed at 10, a percentage run whose values
are multiples of ten is continued BY ITS STEP (1.000, 6/6) and never by plus-one (.000), while a non-round run is continued by plus-one
(.313) and never by the step (.000, 0/48). That gives a minimal pair with identical token length and one digit changed per term:

    "10% 20% 30%"  ->  " 40"   (step)          "11% 21% 31%"  ->  " 32"   (plus one)

which is exactly the shape the battery's interchange-localisation stage consumes. This rung asks WHICH of the 36 components carries the
switch, patching each component's whole output from the non-round run into the round run and scoring normalized logit-difference
recovery between the two answers. Pairs are built for two formats (percent, bare), six round starts (10…60) and four offsets (1…4),
with every pair required to have equal token length, single-token distinct answers, and the joint-tokenization prefix property.

Sign convention: `ld = logit(plus-one answer) − logit(step answer)`; `REC = (ld_patch − ld_base) / max(ld_donor − ld_base, 1e-3)`,
0 for no effect and 1 for a full switch, HIGHER = the component carries more of the switch. The denominator is floored and is positive
by construction (the donor prompt prefers the plus-one answer). **No CE and no §312 L2 here; nothing installs; this is localisation of a
behavioural switch, not a claim about the frontier.**

## Predictions

```
BARS  = {top_rec: .50, top3_share: .60, attn8_rank: 3, format_agree: 1, all_rec: .90}
NULLS = {top_rec_le: .20, top3_share_le: .30, attn8_top: 1, all_rec_le: .50}
```

**pred_a_the_switch_is_localised** — the median over the two formats of the best single component's REC ≥ .50.
*Worked example:* §2817's battery found single components recovering .58–.92 of an interchange when the causal variable is carried by
one write; if roundness is likewise carried, the best component reads .5–.9. If the switch is computed from many places, every REC sits
at .1–.3. Both operands are logit differences in the same units with a floored denominator. Null: ≤ .20.

**pred_b_the_switch_is_not_the_item_writer** — attention 8 ranks WORSE than 3rd in both formats.
*Worked example:* attention 8 writes the last salient item (§2808, §2820) and is the FIT-chosen writer for 8 of 9 capable bank
behaviours (§2840), so the null hypothesis of this campaign is "attention 8 again". Roundness is a property OF the item rather than the
item's identity, so the prediction is that a different component decides it — and if attention 8 leads anyway, that is the more
interesting outcome and would say the write already encodes the roundness of what it carries. Integer rank in [1, 36]. Null: attention 8
ranks 1st in either format.

**pred_c_the_switch_is_concentrated** — median over formats of the top-3 components' share of the total POSITIVE recovery ≥ .60.
*Worked example:* a switch carried by a couple of components reads .6–.9; one computed diffusely reads .2–.3. The denominator sums only
positive recoveries so that components which push the other way cannot inflate the share. Null: ≤ .30.

**pred_d_the_leader_is_format_invariant** — the same component leads in both the percent and the bare format.
*Worked example:* roundness is a property of the number, not of the "%" surface, so a genuine roundness detector leads both; a
surface-specific artefact leads only one. Boolean over two formats — deliberately weak evidence on its own, which is why it is
registered alongside pred_c rather than as a headline.

**pred_e_patching_everything_recovers_the_donor** — patching ALL 36 components simultaneously gives median REC ≥ .90.
*Worked example:* patching every component should reconstruct the donor's computation almost exactly, so ~1.0; a value well below .9
would mean the patch set is incomplete (embeddings, the final norm) and no single-component number in this rung could be read as a share
of anything. This is the instrument check.

## Stated null

The switch is not localised (best REC ≤ .20), it is diffuse (top-3 ≤ .30), attention 8 leads it after all, or the instrument fails to
reconstruct the donor. Any of those is reported as measured; the first two together would say roundness is computed distributively and
this campaign's component-level machinery cannot address it.

## Price

2 formats × up to 24 pairs × (2 native + 36 component patches + 1 all-patch) forwards, batched by token length.
Literal budget: ≤ 400 GPU forwards, 0 backwards, **0 fitted parameters**, < 90 GPU-seconds.

## What this does NOT claim

Whole-component patching only — no head, position or subspace decomposition of the switch, and no selectivity control (there is no
answer-preserving family here, so nothing in this rung is a selectivity claim). One step size (10) and one digit range, as in §2841.
The pairs are this rung's own construction, not the bank's frozen splits, so no number here may be quoted as a bank capability or
localisation. Does not satisfy Codex's four-phase integration contract; updates no circuit record.
