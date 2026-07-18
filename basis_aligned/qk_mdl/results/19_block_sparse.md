# The selection tensor is block-sparse: a 3%-density rulebook

TN-native MDL structure (Logan directive): coarsen the exact layer-0 pattern tensor by
embedding classes (256) and ask how many (class_q, class_k) blocks per head the model
behaviorally needs. Keep the top-B blocks by data-weighted pattern energy, ZERO all
others:

| kept blocks/head | density | ΔCE |
|---|---|---|
| 32,768 | 50% | +0.0000 |
| 8,192 | 12.5% | +0.0006 |
| **2,048** | **3.1%** | **+0.0004** |
| 512 | 0.8% | +0.0468 |

**BS-1:** the layer-0 selection function is describable as a per-head rulebook of ~2k
allowed class interactions (16 bits each ≈ 32k bits/head of structure on top of the
factor tables) — 97% of the class-pair space is behaviorally inert and can be hard-
zeroed. This composes (unlike EH-5's free-edge cut) because the kept blocks carry
essentially all pattern mass; the zeroed tail does not sum coherently.

Relation to the monosemanticity rounds (results/18): these same blocks are NOT
individually output-monosemantic — the rulebook is meaningful as a *selection
structure* (which kinds of tokens attend to which), not as a set of output-aligned
features. The two claims are compatible and jointly sharpen what layer-0 attention is:
a class-interaction router whose consequences only become output-aligned higher up.

Next: name the top blocks per head with class exemplars (the human-readable rulebook),
and check density curves for other layers via the cond-mean factor tables.

Files: `../bs_pattern.py/.json`.
