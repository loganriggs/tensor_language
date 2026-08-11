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

## Addendum 2 (registered before the "better removal" measurements — Logan's question:
## is the KKT collateral fundamental, or just the wrong objective?)

Clarification of framing: the h2-frame KKT edit achieves EXACT forgetting; what it loses
is retention. Its L2 (C-metric) objective is a proxy that spreads disturbance evenly and
knows nothing about margins. Methods that use more knowledge:

P10 (LS readout refit): jointly refit the full linear readout [W, G] over [x1, h2]
    (60-dim frame) by weighted least squares to targets {retained: current logits,
    removed: uniform, weight 100}. Beats the KKT edit's retained flips by >= 30% but
    still flips > 20% of retained facts. Confidence 0.5.

P11 (margin LP — the decisive test): a hinge LP over Delta-G (400 vars) with EXACT
    removal equalities and per-retained-fact margin constraints (>= 0.5) decides whether
    ANY last-layer-output edit can remove the 10 facts with zero collateral. Prediction:
    INFEASIBLE (minimum total hinge violation > 0) — the h2 code is a class-clustered
    40-dim representation, and separating a fact from its class-mates inside it is
    exactly what "no addressable location" forbids; the min-violation solution still
    flips >= 10% of retained facts. Confidence 0.6.

P12 (cross-layer repair): one exact repair round in the R2 frame (Delta-R2, 800 dof —
    logits are exactly linear in R2 with everything else fixed, and h2 is invariant to
    the D2 edit) after the best last-layer edit, holding the removal equalities, cuts
    retained flips by >= half vs the plain KKT edit. Confidence 0.45 (genuinely unsure;
    moot if P11 turns out feasible).

P13 (oracle existence baseline): retraining from scratch on the 1190 retained facts
    gives 100% retention and puts the 10 removed facts at chance (0-3 of 10 correct) —
    the edit we want EXISTS in parameter space; the question is only whether closed-form
    reaches it. Confidence 0.85.

## Addendum 3 (registered before the alternating-frames measurement)

Context from Addendum-2 results: the margin LP is INFEASIBLE in the last-layer frame on
all 5 seeds (certified: NO Delta-D2 edit removes the 10 facts with zero collateral;
~550-620 facts must violate) — but ONE exact R2-frame repair round after the LP edit
halves retained flips to 233-274 (~20%), below the certified last-layer floor.

P14: alternating exact single-frame hinge LPs (G -> R2 -> L2 -> repeat, all three frames
exactly linear with the others held fixed, removal equalities re-imposed every round)
drive retained flips below 10% of 1190 within 6 rounds (seed 0). Confidence 0.4
(may plateau around 15-20% — each frame is only 400-800 dof against ~550 required
repairs; the composed storage may not be reachable by single-frame moves at all).

## Addendum 4 (registered before the masking-diagnosis battery — response to the other
## session's critique of the LP editor as "certified masking")

Their critique, accepted as the test: the LP certificate covers logits on constrained
inputs only; block 1 is not even an exact frame (logits are QUADRATIC in block-1 weights
— x1 enters the last block twice via (L2 x1)*(R2 x1); conceded), so nothing the
alternation does touches early computation, and the edit may be shallow (resurrectable,
fast to relearn). Their registered-style prediction is adopted as P17. One toy-specific
nuance to be measured rather than assumed: in THIS capacity-stressed toy, block-1-alone
readout was already ~chance BEFORE editing (composed storage; F13), so the LM failure
mode "lens finds the answer mid-stack" has no direct analog here — the masking question
must be settled by relearn speed and perturbation recovery instead.

Battery (seed 0, N=1200, H*=40; LP-alternation edit regenerated deterministically and
saved; oracle = retrained from scratch on the 1190 retained):

P17 (relearn speed, THEIR prediction): fine-tuning the LP-edited model on all 1200 facts
    (the 10 removed restored to their original labels) reaches 10/10 on the removed
    facts in >= 3x fewer optimizer steps than the oracle model learning the same 10 as
    genuinely new facts (same protocol, 3 finetune seeds each). Confidence 0.6.

P18 (perturbation resurrection): adding Gaussian weight noise to the LP-edited model, at
    the largest noise level where <= 5% of retained facts break, at least 20 percentage
    points more of the 10 removed facts revert to their ORIGINAL labels than in the
    noise-matched oracle control (where reversion to the never-stored label is ~chance).
    Confidence 0.5.

P19 (edit cost, their unmeasured-norm point): the alternation's total weight change
    (sum of per-round ||Delta||_F over D2/R2/L2, relative) is >= 5x the single
    C-weighted KKT edit's ||Delta G||_F — echoing their 10-20x retension-cost
    observation. Confidence 0.6.

P20 (lens, analytic — no uncertainty claimed): W and block 1 are untouched by the
    alternation, so the block-1-only readout W(z + B1(z)) is IDENTICAL pre/post edit;
    reported with counts of removed facts decodable from it (expected ~chance, per F13).
