# CIRCUIT BATTERY — READER-SIDE SELECTIVITY (preregistration)

Registered 2026-09-04 04:29Z (box clock, read immediately before this line was written). Claude, LANE 1 CUDA.
Rung `circuit_battery_reader_selectivity`. Script: `ops/circuit_battery_reader_selectivity.py`.
Input receipt: `circuit_battery_v2_results.json` (§2817, sha 5924b2549d285175c80fbf7c8fc95a8a2fa06020acc1827bc472ddea69d9ec93).
IMMUTABLE: any change gets a new document, not an edit.

## The question

§2817's most robust result is negative and survived the repair that was most likely to overturn it: attention 8 is the FIT-chosen
writer for 7 of 8 capable behaviours and **not one is writer-selective** — removing its write damages the answer-preserving family P
and the copy control C by .55–1.03 of what it does to the target, with all three families now drawn from the SAME generated situation.
So the task-specific part of these behaviours is not in the write. The only remaining place inside this circuit is the READ. This rung
tests that directly, and it is designed so that the interesting outcome is the one that can fail: if no reader is selective either,
then these behaviours do not contain a selective circuit at all at this granularity, and that is a finding worth having explicitly.

## What is fixed before the run

Reader set `{mlp8, mlp9, mlp10, mlp11}` (predeclared, identical for every behaviour), writer `attn8`, evaluation split `OOD` (never
opened for any selection), behaviours = §2817's capable set with FIT-chosen writer attn8, families A1 (target) / P (answer-preserving)
/ C (copy control) drawn from the same group. Nothing in this rung selects anything. Sign convention: d_m = m_NATIVE − m_arm, POSITIVE
= the arm HURTS that family's own answer; a NEGATIVE damage on a control means removing the edge HELPS that control's answer.
Selectivity ratio of an arm = `max(|d_P|, |d_C|) / max(d_A1, .5)` — absolute values on the control side so that a control the arm HELPS
counts as non-selective damage, and a denominator floored at .5 margin units so it cannot pass through zero.

## Predictions

```
BARS  = {sel_ratio: .25, sel_tasks: 4, beat_writer: .25, shared_mode: 4, copy_help_tasks: 4, repro_tol: .15, floor: .5}
NULLS = {sel_tasks_le: 0, beat_writer_le: 0.0, shared_mode_le: 2, copy_help_tasks_le: 1}
```

**pred_a_some_reader_is_selective** — on at least 4 of the ≤7 behaviours, SOME single reader in the common set has selectivity ratio
≤ .25. *Worked example:* if the +1 computation is reader-local, mlp8's removal should barely move a prompt whose answer does not
depend on the incremented value, reading .0–.2; if the readers are as generic as the writer, every ratio sits near the writer's
.55–1.03 and this reads 0 of 7. Null: 0 behaviours (no reader is selective anywhere).

**pred_b_readers_beat_the_writer_on_selectivity** — median over behaviours of `(writer's selectivity ratio) − (best reader's ratio)`
is ≥ .25. *Worked example:* the writer reads ~.9 (§2817); if the best reader reads ~.4 the difference is ~.5; if reading is as generic
as writing the difference is ~0. This is registered as a DIFFERENCE of two ratios, not a ratio of ratios, because both quantities have
floored but independently varying denominators. Null: ≤ 0 (readers are no better than the writer).

**pred_c_the_selective_reader_is_shared** — the same reader is the most selective one on at least 4 of the behaviours.
*Worked example:* §2817 showed mlp8 dominant in damage on every capable successor behaviour; if selectivity tracks that, mlp8 (or one
fixed reader) wins 5–7 times; if selectivity is idiosyncratic, the mode is 2–3. Operand is a count over ≤7 behaviours. Null: ≤ 2.

**pred_d_readers_push_away_from_copying** — on at least 4 behaviours the joint removal of the four readers has NEGATIVE damage on the
copy control C, i.e. removing the read HELPS the copy answer. *Worked example:* §2808 measured exactly this on the numbered list's
repeated-index control (READS removal −.58 CE, i.e. it helped copying) and attributed the writer's non-selectivity to it; on OOD with
paired situations the margin damage should be negative on most behaviours. If the readers are neutral about copying, the damages are
small positives and this reads 0–1. Null: ≤ 1.

**pred_e_ood_target_damage_reproduces_the_battery** — instrument consistency: the four common readers are a strict SUBSET of §2817's
FULL arm, so their joint OOD damage over §2817's OOD FULL damage must lie in [.30, 1.05] on every behaviour.
*Worked example:* §2817's OOD top-3 shares were .32–.71 of READS, so the top-4 over FULL should read .4–.9; a value above 1.05 would
mean a subset hurts more than the whole (a bookkeeping bug), and below .30 would mean these four readers are not the ones §2817 found.
Both operands are damages in the same units with a floored denominator.

## Stated null

The read is as generic as the write: no reader anywhere reaches ratio .25, readers gain nothing over the writer, the most selective
reader is idiosyncratic across behaviours, and the readers are neutral about copying. If that null holds it is the headline, and it
says these behaviours have no selective circuit at writer-or-single-reader granularity.

## Price

≤ 7 behaviours × 3 families × (4 single readers + joint + writer-full) × 2 forwards per length-batch of 16 OOD rows.
Literal budget: ≤ 900 GPU forwards, 0 backwards, 0 fitted parameters, expected < 90 GPU-seconds.

## What this does NOT claim

Single readers and one joint arm only; a selective SUBSET other than the full four is not searched (searching one would be selection,
and this rung selects nothing). Does not satisfy Codex's four-phase integration contract; updates no circuit record.
