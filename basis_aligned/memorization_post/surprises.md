# Surprises — anything contradicting the handoff's pre-registered predictions

(Each part appends its own section.)

## Part 1

1. Unpredicted positive own-feature diagonals in 1b. The pre-registered prediction (positive
   pair off-diagonal + negative absent-feature structure) is confirmed, but SGD additionally puts
   POSITIVE diagonal (linear) weight on the class's own two features (means +0.7 to +2.2). These
   are the least reliable entries — sign flips across seeds in 7 of the 9 own-feature cells at
   weight decay 1e-3 — so the trained solution is "AND-pair plus not-absent-feature plus a noisy
   linear boost on the present features", not the clean two-term logic.

2. The trained Dog slice is not close to rank 1 asymmetrically. The hand-coded slice recovers
   rank-1 L/R/D exactly, but the trained H=8 slice has asymmetric singular values
   (6.13, 2.45, 1.83) — solidly rank 3. Pull-out is still functionally exact (the fold is exact
   regardless), but "CP-decompose the slice and get the rank-1 fact back" only holds for the
   clean construction, not for what SGD writes.

3. Zeroing the (furry,dog-ears) interaction does not behaviorally remove Dog's second path.
   With the entry exactly zeroed (and (furry,happy) exactly preserved), the Dog(furry,dog-ears)
   key still classifies as Dog in 9/10 runs across both regimes; its margin drops by ~11-12 nats
   but stays positive. The second path's classification is co-carried by diagonal linear terms
   (positive furry/dog-ears diagonals in Dog's slice and negative diagonals in competitors'
   slices), consistent with the diagonal-as-linear channel from 1b being load-bearing. "The
   path" in the draft should be identified with interaction + diagonal structure, not the single
   off-diagonal cell.

4. Whole-unit surgery fails even in the overcomplete regime. The handoff's framing suggests
   overcomplete models should permit surgical path removal; that holds for the minimal-norm
   tensor-entry edit but NOT for unit-level ablation: H=12 networks spread the (furry,dog-ears)
   path over 5-8 units that also carry other functions, so ablating them flips 40-80% of Dog keys
   and halves Human's key entry. Overcompleteness did not buy dedicated units (training had no
   sparsity pressure); surgery must be done in folded-tensor coordinates, not unit coordinates.

5. Minor: undercomplete H=2 (3 classes) trained to 100% on every seed — the handoff's
   contingency ("if H=2 fails to train") was not needed; and the 3-4-point dataset was not
   degenerate under softmax CE (the all-zeros row pins the loss but has zero gradient, and all
   runs converge to consistent structure), so no label noise or 4th class was added in 1b/1c.
