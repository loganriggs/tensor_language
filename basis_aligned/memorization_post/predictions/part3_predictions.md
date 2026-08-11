# Part 3 (200 facts, 2 bilinear blocks + residual stream) — registered predictions
(commit time = registration time; committed before the sizing sweep and all measurements)

Setup per handoff: n=20-bit keys, 200 facts, 10 classes; residual stream x -> x + B1(x)
-> ... + B2(...), linear readout W; sized (by sweep) so one layer alone fails but two
suffice. Attribution of each fact by single-layer evaluation (zero the other block);
cross terms = full logits minus (W z + W B1(z) + W B2(z)) on fact keys (degree-3/4
interference); ablation survivor counts; F13.

Metric positive controls (gate, run before any claim): train the 2-layer model with one
block frozen at zero — the attribution metric must assign >= 90% of memorized facts to
the live layer. If this fails the metric is repaired before any verdict counts.

Predictions:

P1. Storage is NOT cleanly disjoint per layer. Fewer than 30% of facts land in the clean
    single-layer bins (correct under exactly one single-layer evaluation); the majority
    are "both" or "neither" (joint code). Basis: Part 2's blind-extraction null — facts
    are a joint code even within ONE layer. Confidence 0.6.

P2. The composed degree-3/4 cross terms are load-bearing, not noise: evaluating the
    additive degree-2 surrogate (W z + W B1(z) + W B2(z)) loses >= 25% of the 200 facts
    (vs ~0 lost by the full model). Confidence 0.55.

P3. Directional "negation" hunch (the draft's): cross-term logit contributions
    anti-correlate with layer-1's direct contribution on stored keys — median over facts
    of cos(cross_vec, W B1(z) vec) < 0, i.e. the composition partially CANCELS the
    early-layer write rather than adding to it. Confidence 0.5 (genuinely unsure; a
    positive-median outcome = layers amplify, also interpretable).

P4. Ablation asymmetry: zeroing block 2 kills more facts than zeroing block 1 (block 2
    sees the enriched stream and the readout sits after it), i.e. survivors(no block 2)
    < survivors(no block 1). Confidence 0.55.

Honest-outcome clause per handoff: "verified: it is/isn't cross-layer" both acceptable.

## Addendum (registered before the capacity curve and 2-layer edit measurements)

P9 (capacity curve, F13b): training on a fixed pool of 4000 facts, the 1-block curve of
facts-fit vs width is FLAT beyond H=40 (variation < 10% from H=40 to H=300, all near the
~650 seen at N=1200 overload), because H=210 already spans all quadratics; the 2-block
curve rises with width through H=210, exceeding 3000 facts. Confidence 0.5.

2-layer edits at N*=1200, H*=40 (all closed-form; logits are LINEAR in the last block's
output map D2 via G = W D2, logits = W x1 + G h2, so the Part-2 KKT machinery transfers
with the last-layer hidden representations h2(z) in R^40 as the stored-key frame):

P5 (removal): the retain-aware joint KKT edit (C^-1-weighted, C = sum h2 h2^T over the
   1200) unlearning 10 random facts achieves EXACT forgetting by construction, but flips
   >= 5% of the 1190 retained facts — 1200 facts crowd a 40-dim frame (vs Part 2's
   2/450 flips with 100 facts in the same-size frame). Confidence 0.6.

P6 (frame comparison): the same edit done in the readout frame (Delta W, keys x2 in R^20)
   produces >= 2x the retained flips of the h2-frame edit. Confidence 0.7.

P7 (injection): 10 NEW facts injected by the same least-norm machinery land exactly
   (10/10 correct after edit, by construction) with retained-fact collateral of the same
   order as removal. Confidence 0.55.

P8 (pull-out): informed h2-frame attribution (dictionary = C^-1-weighted h2 keys, the
   2-layer analog of F10's informed extraction) recovers < 10% of the 1200 facts —
   far below Part 2's 44-51%, because the frame is 30x overloaded. Confidence 0.65.
   (Class-level pull-out — keep one row of W plus both blocks — is exact by construction
   and will be reported as such; there is no generalizable behavior to pull out of random
   facts: that question belongs to Part 4's structured variant.)
