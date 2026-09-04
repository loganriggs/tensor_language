# CIRCUIT BATTERY — READER DEPTH GRADIENT (preregistration)

Registered 2026-09-04 04:38Z (box clock, read immediately before this line was written). Claude, LANE 1 CUDA.
Rung `circuit_battery_reader_depth_gradient`. Script: `ops/circuit_battery_reader_depth_gradient.py`.
Input receipt: `circuit_battery_v2_results.json` (§2817, sha 5924b2549d285175c80fbf7c8fc95a8a2fa06020acc1827bc472ddea69d9ec93).
IMMUTABLE: any change gets a new document, not an edit.

## Object and the fix this document carries

§2819 found, on four readers, that task specificity rises with depth: mlp8 is as generic as the writer (ratio 1.00–1.12) while mlp11 is
the most specific on 6 of 7 behaviours (.14–.59). This rung extends it to ALL ten MLPs that can read attention 8's write (mlp8…mlp17)
and asks whether specificity keeps rising past mlp11 or saturates.

It also carries the correction forced by §2820, where my selectivity ratio crowned an INERT attention head (target damage ±.001 margin
units) as "perfectly selective" because the ratio has no minimum-damage requirement. **From this document on, every selectivity ratio
has a registered ADMISSIBILITY GATE: a reader is admissible only if its own A1 damage is at least .10 × the whole READS damage.** The
gate is not merely applied — pred_d MEASURES the failure mode it prevents, so that the §2820 error becomes a recorded property of the
metric rather than an anecdote.

Fixed before the run: readers mlp8…mlp17, writer attn8, split OOD (never opened for selection), families A1/P/C from the same group,
admissibility fraction .10. Nothing is selected in this rung. Selectivity ratio = max(|d_P|, |d_C|) / max(d_A1, .5), LOWER IS MORE
SPECIFIC; d_m = m_NATIVE − m_arm, POSITIVE = the arm HURTS that family's own answer.

## Predictions

```
BARS  = {rho: -.50, deep_tasks: 5, saturate: -.10, inert_tasks: 4, coverage: .80, floor: .5}
NULLS = {rho_ge: 0.0, deep_tasks_le: 2, saturate_ge: .20, inert_tasks_le: 1, coverage_le: .40}
ADMIT = .10
```

**pred_a_specificity_rises_with_depth** — median over behaviours of the Spearman correlation between an ADMISSIBLE reader's layer index
and its selectivity ratio is ≤ −.50. *Worked example:* §2819's four readers were monotone on every successor behaviour (1.05 → .23 for
month, .64 → .14 for the numeric run), so on the admissible subset the hypothesis reads −.7 to −1.0; if depth is unrelated to
specificity, ~0. Spearman over ≥ 3 admissible readers; behaviours with fewer are excluded from the median and counted. Null: ≥ 0.

**pred_b_the_specific_reader_is_deep** — on at least 5 of the ≤7 behaviours the most specific ADMISSIBLE reader sits at layer ≥ 10.
*Worked example:* §2819's winner was mlp11 on 6 of 7; if the gradient is real this reads 5–7, and if specificity were arbitrary the
winner would be uniform over 8–17 and this would read 2–3. Count over behaviours. Null: ≤ 2.

**pred_c_the_gradient_saturates_after_eleven** — median over behaviours of `min(ratio over admissible readers at layers 12–17) −
min(ratio over admissible readers at layers 8–11)` is ≥ −.10, i.e. going deeper than mlp11 buys at most .10 of extra specificity.
*Worked example:* if the gradient saturates, the two minima are close and this reads −.1 to +.3; if specificity keeps rising with depth,
the deep minimum is much lower and this reads −.3 to −.8, failing the prediction — which would be the more interesting outcome and is
why it is registered in this direction. A DIFFERENCE of two floored ratios, not a ratio of ratios. Behaviours with no admissible reader
in either band are excluded and counted. Null: ≤ −.20.

**pred_d_ungated_ratios_crown_inert_readers** — on at least 4 of the ≤7 behaviours there is at least one NON-admissible reader whose
raw ratio is ≤ .25, i.e. the ungated metric would have declared an inert reader maximally selective. *Worked example:* §2820 showed
exactly this for attention heads on every behaviour; the late MLPs mlp15–mlp17 have near-zero damage on these tasks, so the hypothesis
reads 5–7. If it reads 0–1, the gate was unnecessary and I should say so. Count over behaviours; the arms counted are by construction
those the gate excludes. Null: ≤ 1.

**pred_e_admissible_readers_carry_the_read** — median over behaviours of `(sum of admissible readers' A1 damages) / max(READS damage, .5)`
is ≥ .80. *Worked example:* §2818's four readers alone carried .669 of READS, and the gate admits every reader above a tenth of it, so
the hypothesis reads .8–1.1 (values above 1 are possible and expected, because single-reader damages are super-additive — §2818
measured singles/joint at .759, so a SUM of singles can exceed the joint). Sum of signed damages over a floored denominator. Null: ≤ .40.

## Stated null

Depth is unrelated to specificity (ρ ≥ 0), the most specific reader is shallow, specificity keeps rising past mlp11, the admissibility
gate is unnecessary, and the admissible set is a minority of the read. Each null is reported separately.

## Price

≤ 7 behaviours × 3 families × (10 readers + 1 READS arm) × 2 forwards per length-batch of 16 OOD rows.
Literal budget: ≤ 1,800 GPU forwards, 0 backwards, 0 fitted parameters, expected < 3 GPU-minutes.

## What this does NOT claim

MLP readers only (attention reads were ≈ .10 of READS in §2808 and are not re-measured here); single-reader arms only, so the
super-additivity of §2818 is not re-derived; no within-block decomposition. Does not satisfy Codex's four-phase integration contract;
updates no circuit record.
