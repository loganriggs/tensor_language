# F11d — the multi-frame margin-LP editor applied to the ONE-layer case (registered
# before measurement; commit time = registration time)

Context: in the 2-layer model (F13d), a margin LP proved no single-frame edit can unlearn
10 of 1200 facts cleanly (~550 forced breaks), while alternating exact single-frame LPs
(D2 -> R2 -> L2) reach zero collateral in 6 rounds. Logan asked to bring the same frames
story back to the one-layer Part-2 setting and update its explanation/figures.

One-layer frames: logits = D((Lz)*(Rz)) is exactly linear in D, in R (L, D fixed), and in
L (R, D fixed) — the same three-frame structure as the 2-layer last block.

P15 (working point, 100 facts, H*=40, same 10 edit facts as F11): the margin LP in the
    D-frame ALONE is feasible on all 5 SGD seeds — exact-uniform forgetting with all 90
    retained facts at margin >= 0.5, zero collateral, ONE round, no alternation. The
    frame is only 2.5x loaded (100 keys in 40 dims). This upgrades F11's KKT result
    (2/450 flips) to exactly zero. Confidence 0.7.

P16 (overload arm, one layer, 350 facts, H=40, seed 0, trained without the L1 penalty —
    stated deviation, capacity arm): the D-frame LP is INFEASIBLE (the frame is 8.75x
    loaded), and alternating D -> R -> L margin LPs reach zero retained flips within 6
    rounds — demonstrating that the ladder is about FRAME LOAD, not depth: one layer
    behaves exactly like the 2-layer model once its frame is overloaded. Confidence 0.5
    (the alternation half is the uncertain part; infeasibility half ~0.7).

## Addendum (registered before the escalation measurement)

P16 outcome: first half FAILED — at 350 facts the D-frame LP is still feasible (0 flips,
one round). Escalation: N in {600, 900} (one layer, H=40, l1=0, seed 0), stopping where
memorization drops below 100%.

P16b: the one-layer model remains ONE-ROUND editable (D-frame margin LP feasible, zero
collateral) at every N it can still memorize 100% — the editing wall coincides with the
capacity wall, so the multi-frame ladder NEVER appears in a single layer; it is specific
to composed (cross-layer) storage. Confidence 0.5.
