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
