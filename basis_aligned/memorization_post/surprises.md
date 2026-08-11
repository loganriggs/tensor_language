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

3. Zeroing the removed path's interaction entry does not behaviorally remove the path — and in
   the tail redesign (2026-08-10), even zeroing EVERY tail-involving entry of Dog's slice does
   not. In the [furry,happy,whiskers,tail] setup, with all four tail terms exactly zeroed and
   (furry,happy) preserved, the Dog(happy,tail) key still classifies as Dog on 5/5 seeds
   (registered 0.65-confidence prediction failed): the (happy,happy) diagonal — Dog's core-
   feature linear term, shared with the kept path — alone outscores the competitors on that
   input. "The path" is not a set of tensor entries you can cut; shared linear structure
   co-carries it.

3b. What DOES remove the fact is the functional key-frame edit (the Part-2 KKT family brought
   back to the toy): constrain f_Dog on the stored key itself (set below competitors, preserve
   the kept key's logit exactly). Overcomplete: 5/5 target flips, zero collateral, kept-key
   margin delta exactly 0 — while the (happy,tail) tensor entry stays visibly NONZERO. The
   same edit in the undercomplete regime flips Cat keys to Dog on 3/5 seeds: collateral is a
   property of the regime (stored-key overlap), not of the edit — exactly the Gram story from
   F5/F6. Bonus asymmetry: full tail removal is not even EXPRESSIBLE in the undercomplete
   model (5 constraints > 3 free D-row parameters).

4. Whole-unit surgery fails even in the overcomplete regime. The handoff's framing suggests
   overcomplete models should permit surgical path removal; that holds for the minimal-norm
   D-row edits but NOT for unit-level ablation: H=12 networks spread the (happy,tail) path
   over 4-9 units that also carry other functions, so ablating them flips 40-60% of Dog keys
   and degrades the preserved entries ((furry,happy)->Dog +3.37 -> +0.66 mean). Overcompleteness
   did not buy dedicated units (training had no sparsity pressure); surgery must be done in
   folded-tensor or stored-key coordinates, not unit coordinates. (Numbers updated 2026-08-10
   for the tail redesign; the old Human-class design showed the same phenomenon.)

5. Minor: undercomplete H=2 (3 classes) trained to 100% on every seed — the handoff's
   contingency ("if H=2 fails to train") was not needed; and the 3-4-point dataset was not
   degenerate under softmax CE (the all-zeros row pins the loss but has zero gradient, and all
   runs converge to consistent structure), so no label noise or 4th class was added in 1b/1c.

## Part 2 surprises (contradicting the handoff's expectations)

1. D approx -I does NOT emerge (F9 panel iii premise refuted). Sparsity-regularized SGD
   (L1 1e-3) at H=40 gives per-unit dominance 0.39-0.49 (one-hot would be 1.0) and the
   dominant D entries are negative only 40-55% of the time, with sign flips across seeds.
   Hidden units are shared across classes; nothing prefers the -I convention.
2. The F9 acceptance criterion passes but is weaker than it looks: SGD-vs-construction
   similarity (0.09-0.15) is numerically the SAME as construction-vs-construction under
   ALS re-initialization (0.07-0.13). The honest statement is "same wide solution family,
   clearly distinct from both nulls", not "SGD finds the construction". SGD seeds are 2.5x
   closer to each other (0.32) than to the construction.
3. Blind extraction of facts as rank-1-ish components fails COMPLETELY (F10): recovery
   0-1% at every Gram-overlap bin, for every model, and at every capacity tested up to
   H=400 (10x overparameterized). Recovery-vs-overlap only exists for an informed
   key-frame attribution (dictionary = C^{-1}-weighted keys), and even there overlap
   explains little variance (corr -0.14). The folded slice is a compressed joint code of
   its keys, not a sum of recoverable per-fact components.
4. The naive-removal baseline is worse than expected in kind, not just degree: subtracting
   the raw-key rank-1 components does not even achieve forgetting (residual deviation from
   uniform 23-68 logits, from cross-talk among the 10 edits) while flipping 55% of retained
   facts. Decomposition of the advantage: re-tensioning buys exact forgetting; the
   C^{-1}-weighted direction buys the ~9x collateral reduction.
5. Off-fact-set margins are not small (F12): for SGD and the ALS construction ~20% of all
   2^20 - 100 inputs have a larger top1-top2 margin than the least-confident stored fact,
   and the analytic Gram bound, though valid everywhere, is 35-55x loose. Only the
   minimum-C-norm KKT interpolant is well behaved off-distribution (1.4%). The post's
   "behavior guarantee on all 2^n inputs" must be stated as the bound itself.

## Part 3 surprises

1. Single-block capacity is 4x the monomial count. A quadratic in 20 booleans has ~210
   free monomials, yet one bilinear block memorizes 800 random 10-class facts (and 400,
   and 200) at every width tried; the ceiling sits between 800 and 1200. At 1200 facts it
   saturates at 54% with H = 40 and H = 210 giving identical accuracy — the failure is
   expressivity of the quadratic function class, not parameters. Margin classification is
   much cheaper than interpolation.

2. At capacity-stressed sizing nothing is stored "in a layer". 62-66% of the 1200 facts
   are classified correctly by NEITHER single-layer evaluation, and every attribution bin
   except "neither" sits at the 10% chance floor. The degree-2 additive surrogate keeps
   18% of facts; ~98% of logit magnitude on stored keys lives in the composed degree-3/4
   cross terms. The handoff's question "disjoint per layer or composed?" has an extreme
   answer: composed, almost totally.

3. The "negation for every fact" hunch is not supported at the logit level (P3 refuted):
   cos(cross-term, layer-1 write) distributions are broad and centered near zero (medians
   +0.07 to +0.13), so the composed term is a nearly orthogonal channel, not an
   interference canceller. P4 also refuted: zeroing block 1 hurts slightly more than
   zeroing block 2 (block 2 computes on the block-1-enriched stream).

4. Editability collapses in the composed regime. The identical retain-aware KKT machinery
   that unlearned 10 of 100 facts with 2/450 collateral flips in Part 2 costs ~45% of the
   store (516-536/1190 flips) when unlearning 10 of 1200 facts in the 2-layer model's
   40-dim last-block frame — and that is the collateral-OPTIMAL closed-form edit (the
   20-dim readout frame is 1.3-1.5x worse). Injection is asymmetrically cheap: 10/10 new
   facts land exactly with 18-32% collateral.

5. There are no per-fact components at all — and the naive metric lies in the OTHER
   direction. The F10-style "component classifies its own key" recovery metric returns
   ~100% here, which is a degeneracy (30x-overcomplete dictionary makes each component a
   whitened copy of the full model's logits), not recovery; caught before claiming by
   asking what the null would produce. The deletion test gives the true answer: removing
   any fact's component breaks nothing (0/1200 own-fact breaks, 0.0% median collateral) —
   each component is weightless. Facts in the composed regime are functionally present
   but have no addressable weight-space location.

6. The capacity plateau is exact: 1022-1026 facts fit from a 4000 pool at every 1-block
   width from 40 to 300 — flat THROUGH the full-quadratic-span boundary (H=210), pinning
   the failure to the degree-2 function class itself. Two blocks blow past the pool at
   H=80 (true capacity > 4000).
