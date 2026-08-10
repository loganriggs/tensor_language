# F8 redesign (tail feature, no 4th class) — registered prediction (commit time = registration time)

Logan's redesign 2026-08-10: instead of the handoff's 4th class (Human = hands+dog-ears),
add ONE extra feature (tail) to the 3-class toy, used as an AND for Dog and an AND for Cat:

- features [furry, happy, whiskers, tail]
- Dog = (furry AND happy) OR (happy AND tail)      — happy is Dog's core feature
- Cat = (furry AND whiskers) OR (whiskers AND tail) — whiskers is Cat's core feature
- Catfish = (whiskers AND happy), unchanged

Five training keys: Dog(f,h), Dog(h,t), Cat(f,w), Cat(w,t), Catfish. Edit: zero the
(happy,tail)->Dog interaction entry while preserving (furry,happy)->Dog; Cat and Catfish
rows are untouched by construction (edit lives in Dog's D-row). This is a harder sharing
test than the old design: tail is shared between the REMOVED Dog path and a KEPT Cat path,
and happy is shared between the removed path and both kept Dog/Catfish structure.

Predictions (before running):

1. Overcomplete (H=12), minimal-norm D-row edit: surgical on all 5 seeds — Dog(h,t) key
   stops being classified Dog; the other four keys keep their labels, margin changes small
   (|delta| < ~15% of pre-edit margin). Confidence 0.75.
2. Undercomplete (H=3 or 4), same edit: collateral appears; the most-damaged non-target
   keys are Cat(w,t) (shares tail) and/or Dog(f,h) (shares happy via Dog's own row);
   at least 1 of 5 seeds has a non-target key actually flip label. Confidence 0.6.
3. Greedy whole-unit ablation (both regimes): non-surgical — at least one non-target key
   flips in most seeds, because units carrying (happy,tail) also carry tail/happy mass
   used by kept paths. Confidence 0.7.

## Addendum (registered before the follow-up measurement)

Outcome of prediction 1: SURGICAL half confirmed (no other key flips, preserves exact) but
Dog(h,t) did NOT flip on any seed — the diagonal entries (happy,happy)/(tail,tail) co-carry
the path (replicates the Part-1 surprise from the old design). Follow-up edit: FULL tail
removal — least-norm Dog D-row update zeroing all four tail-involving entries of Dog's slice
((furry,tail),(happy,tail),(whiskers,tail),(tail,tail)) while preserving (furry,happy).
Only exactly solvable when H >= 5, so overcomplete only; in undercomplete H=3 the edit is
not even expressible (5 constraints > 3 free parameters) — itself a reportable fact.

Prediction: in overcomplete H=12, full tail removal flips Dog(h,t) away from Dog in >= 4/5
seeds while all four other keys keep their labels. Confidence 0.65.

## Addendum 2 (registered before the follow-up measurement)

Outcome of the addendum-1 prediction: FAILED (0/5 flips). Even with ALL tail-involving
entries of Dog's slice zeroed, {happy,tail} is still classified Dog on every seed: the
(happy,happy) diagonal — Dog's core-feature linear term, shared with the kept (furry,happy)
path — alone outscores Cat and Catfish on that input. Weight-basis path surgery cannot
remove this fact without touching shared structure.

Follow-up (F8b): FUNCTIONAL key-frame edit — least-norm update to Dog's D-row setting
f_Dog(z_{h,t}) to (max other-class logit on z_{h,t} - 1) while exactly preserving
f_Dog(z_{f,h}). Constraints are linear in the D-row via the stored-key representation
a_z = (Lz)*(Rz). Expressible in BOTH regimes (2 constraints).

Predictions: (1) overcomplete H=12 — Dog(h,t) flips away from Dog on 5/5 seeds, no other
key flips, and the (happy,tail) TENSOR ENTRY stays visibly nonzero (behavior removed
without zeroing the mechanistic entry). Confidence 0.8. (2) undercomplete H=3 — target
flips on 5/5 (exact constraint) but collateral is larger; at least 1 seed has a
non-target key flip. Confidence 0.55.
