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

## BS-2: sparsity is universal, and the rulebook is readable

Depth ladder (same keep-top-B mask on LIVE patterns): at 3.1% density, L1 +0.006,
L5 +0.008, L12 +0.001, L16 +0.001 — every layer of the model runs on a ~2k-block
rulebook. At 0.8% the uppers stay cheap (+0.005) while L5 resists (+0.250: its two
contextual heads need the tail). Whole-model selection structure: 18×9 rulebooks ×
2048 blocks × 16 bits ≈ **0.66 MB for all attention routing in the 546M model**.

The named rulebook ([cards/rulebook_L0.md](cards/rulebook_L0.md)) reads as SAME-KIND
MATCHING plus structure anchors: pronouns attend pronouns, quote-punctuation attends
quote-punctuation, code-identifiers attend code-identifiers, spatial prepositions
attend spatial prepositions — with the junk-token classes claiming high energy (the
CP-1 pathology) but sitting harmlessly inside the kept 3%.

Files: `../bs_pattern.py/.json`.


## SR-1/SR-2: generality on sqrd12 — flavor yes, composition no

Same construction on sqrd12 (classes from its own embedding; row-normalized patterns
masked-then-renormalized): single layers hold (3.1%: L3 +0.027, L8 +0.008) and the
blocks read identically (brackets↔brackets, prepositions→sentence-enders). But the
ALL-layer composition is much worse: **+0.569** at 3.1% (vs bilin18 +0.190), +1.82 at
0.8%. Suspected mechanism (denominator coupling) — TESTED AND REFUTED
(sqrd12_coupling.json): masking with the ORIGINAL row sums is slightly WORSE (+0.687
vs +0.569), so renormalization helps rather than couples. Leading alternative, open:
redundancy — sqrd12 has 6 single-branch heads vs bilin18's 9×2 head-branches, and its
per-head patterns spread energy wider, so block loss has fewer places to hide
(consistent with its no-free-head-ablations profile, results/05). The compressibility ranking between the
two models is decomposition-family-specific even within the rulebook family
(cf. the windowed-D inversion, results/11 §8). Files: ../sqrd12_rulebook.py/.json.

## RD-1: same-class matching is bottom-heavy; the top goes cross-class

Diagonal-block (cq=ck) energy fraction by layer: L0 0.31 · L1 **0.44** · L5 0.19 ·
L12 0.21 · L16 0.10 · L17 0.21. Same-kind matching dominates early selection and
fades with depth; upper-layer rulebooks are mostly cross-class. Cleanest named
cross-class rule (L17): pronouns → temporal conjunctions
([' everything',' anyone',' they'] attends [' when',' after',' while']).
Per-layer named rulebooks: cards/rulebook_L{0,1,5,16}.md. Caveat: raw-energy
rankings are junk-class-polluted outside L17 (frequency filtering needed before
quoting those names). Files: ../rulebook_depth.py/.json.