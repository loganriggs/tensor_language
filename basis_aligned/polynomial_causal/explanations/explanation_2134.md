# Plain-English update — 2026-08-31 21:34Z (day close)

(Yardstick: damage = extra prediction error above the real model; LOWER IS BETTER. A "certificate" = a
per-behavior pass: the compressed model's damage at that behavior's positions must be under half the
damage of knocking out that behavior's key component in the real model.)

## The day in one line
Error halved, then halved again, then the first behavior certificates in program history appeared — and
every step was a preregistered, reproduced claim.

## The three discoveries that did it
1. **The lookup tables were the elephant** (morning): the oldest part of the compiled model carried most of
   the damage; replaced by pruned copies of the model's own neurons.
2. **One grammar for all attention** (evening): every replaced attention head now uses the same rule — keep
   the model's real attention computation but shrink its "where to look" maps by plain SVD. A rank theorem
   (pattern = product of two score matrices, so ranks multiply) predicted the critical size, ~11 of 128,
   before we measured the knee at 10-12.
3. **Certificates finally move** (night): with everything else exact and only the attention maps compressed,
   2 then 7 of 62 behaviors passed for the first time ever — and the pass count tracks the map size.
   The median behavior sits at 1.0x its threshold (needs 0.5x). Repeated-token prediction is now BETTER
   than the real model.

## The honest ledger
- Registered trade-off curve: 57M values -> 1.31 damage / 88M -> 0.64 / 162M -> 0.149 (2 passes) /
  186M -> 0.085 (7 passes).
- Shrinking the MLPs gets MORE expensive the better everything else is; the attention maps are the one
  cheaply-compressible part at certificate grade.
- Running now: three-quarter-size maps (do passes reach a majority?) and a mixed allocation (big maps only
  where the retrieval work lives).
