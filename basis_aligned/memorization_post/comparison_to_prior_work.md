# Comparison: this session's Parts 1-3 vs github.com/loganriggs/memorization

Read 2026-08-11 (README, reports/results.md, p2_transfer_report.md, p3_report.md,
t3_retension_report.md, t5_lm_report.md, research_log.md headlines). The prior repo is a
toy-to-real ladder (token toy -> dense -> MNIST -> multilayer transformers -> 6-layer LM
-> TOFU at 410M-1.4B) organized around "facts are stored as margins". This session is a
deep vertical on the small end (1-2 bilinear layers) with pre-registered predictions and
closed-form editing. The overlap is large and almost entirely CORROBORATING; each side
has results the other lacks.

## Where the two lines independently agree (mutual replication)

1. No per-fact components / dense superposition. Prior: trained bilinear stores each
   fact as a "slightly-above-interference bump in a dense random-looking quadratic form"
   (effective rank ~20 of 36, ~7-8 effective neurons per fact, zero zero-activations).
   Session: blind eigen-extraction recovers 0-1% of facts (F10); in 2 layers the
   component-DELETION test shows every per-fact component is weightless (0/1200 own-fact
   breaks) and every attribution bin sits at the 10% chance floor (F13/F13c). Same
   conclusion by different instruments (his: participation/spectra; ours: causal
   deletion + chance-baselined attribution).

2. Margins, not reconstruction, are the storage criterion. Prior: CP/ALS hand-coding
   that RECONSTRUCTS the fact tensor gets capacity ~m (1 fact per component), ~200x
   below trained — "the compression lives in the gap between reconstruct-the-table and
   win-every-argmax". Session: F13b pins the same gap at the function-class level — a
   single quadratic layer holds ~1024 facts against only 210 monomials (~5x the
   interpolation count), flat from H=40 to H=300.

3. D ~ -I is a sparsity-conditional solution FAMILY, not the solution. Prior: pruned
   sparse models converge to silence-code variants sharing family statistics but not
   weights (Hungarian-aligned cosine 0.28-0.66). Session: F9/F9b — dominance rises with
   L1 (0.78-0.88 at 1e-3 -> 0.87-0.95 at 1e-1 on live units), sign is pure gauge,
   ALS-vs-ALS self-similarity 0.07-0.13. Both say: the target of interpretation is a
   solution family.

4. Editing: explicit repair beats single edits, and zero collateral is reachable.
   Prior (t3): two-stage delete-then-retension (proximal rank-1 delete, then hinge-
   repair of bystander margins from post-delete weights) drives collateral to 0 in
   every regime tested, including where the best SINGLE edit floors at ~23 broken facts
   (dense toy, n=768 crowded) and at 2-layer depth (33/17/5 collateral -> 0). Session:
   margin-constrained LPs — single-frame edit certified infeasible at high load, but
   alternating exact frames reaches 0 collateral (F13d, F11d). THE SAME DISCOVERY,
   arrived at independently: his via optimization (greedy hinge repair), ours via exact
   convex solves. His "oracle floor ~23 was a floor for single edits, not for editing
   per se" = our "certified impossible in one frame, exactly solvable in six".

5. Crowding is what kills surgical editing. Prior (p3): saturation primary, input
   correlation secondary; below capacity zero-collateral succeeds even at MNIST-level
   correlation, at ~80% load it fails even uncorrelated. Session: the load ratio
   (facts / frame dim) sets the regime — 2.5x trivial, 8.8x one LP round, 15x two
   rounds, 22.5x weight-frame stall, 30x (2-layer) certified per-frame infeasible.

6. Exact ledgers through linear paths. Prior (t4): closed-form margin change for
   last-layer edits in the no-norm multilinear transformer, predicted collateral
   matching reality fact-for-fact. Session: the same structural fact (logits linear in
   D2/W/R2/L2 with the rest fixed) is what powers the KKT edits (prediction r =
   1.000000) and the frame LPs.

7. Where facts live vs load. Prior (p2): freezing EITHER width-matched block costs
   nothing below capacity (storage is opportunistic), 40-50% at saturation. Session
   (Part 3, capacity-stressed by construction): 62-66% of facts are classified by
   NEITHER single layer; ablating either block is catastrophic (~16-20% survive). No
   conflict — the two sit at opposite ends of one law: opportunistic/redundant below
   capacity, composed/joint under stress. (The session's "either alone (redundant)"
   bin, ~2% at stress, is presumably his no-cost-freezing regime's dominant bin.)

## New in this session (not in the prior repo)

1. Feasibility CERTIFICATES for editing. The prior repo's floors are empirical (search
   over candidate edits); the margin LP turns "we could not find a clean edit" into
   "none exists in this frame" (infeasibility with quantified minimum violation, all 5
   seeds). Plus the exact feasibility boundary: D-frame feasible at 350 facts,
   infeasible at 600, weight-frame ladder stalls at 900.

2. The frame taxonomy and its two punchlines. (a) The one-vs-two-layer INVERSION:
   near capacity, one layer is weight-frame UNEDITABLE (alternation stalls at ~403
   broken) while the 2-layer model at higher load converges to 0 — composition
   enriches the last block's frames. (b) The folded tensor as one layer's free maximal
   frame (one LP, zero collateral at every memorizable load) vs its ~d^4 unaffordable
   2-layer analog. Unification: editability = the affordable linear frame.

3. Composed-storage decomposition. ~98% of logit magnitude on stored keys lives in the
   degree-3/4 cross terms; the degree-2 additive surrogate keeps 18% of facts; the
   cross term is a nearly orthogonal third channel (cancellation/"negation for every
   fact" hunch refuted, cosines centered ~0). The prior repo edits and audits at depth
   but does not decompose composed storage into additive vs cross terms.

4. Path-vs-entries (Part 1 F8/F8b): even zeroing EVERY tail-involving tensor entry of
   Dog's slice fails to remove the path (shared diagonal carries it); the functional
   key-frame edit removes it with the entry left nonzero. Sharpens the prior repo's
   margin thesis into: the fact is a margin RELATION, not a set of weight entries.

5. Pre-registered Gram forecasting. Stored-key Gram off-diagonals predict measured
   collateral at r = 0.943 (60 predictions committed to git before measurement);
   KKT collateral prediction exact (r = 1.000000). Prior t3c found weights-side
   subspace ranking does NOT beat the raw input-overlap baseline (AUC 0.56-0.71 vs
   0.63-0.75) — the session's representation-side Gram (hidden-rep cosines) is the
   stronger pre-edit forecaster in the regimes tested, consistent with his
   "breaking = interference x slack" diagnosis (the Gram sees interference where it
   acts; the LP adds the slack).

6. All-inputs behavior audit (F12): exhaustive 2^20 evaluation against the analytic
   Gram-overlap bound — bound holds everywhere, 35-55x loose; ~20% of off-fact inputs
   exceed the least-confident stored fact's margin for SGD/ALS (1.4% for the KKT
   interpolant). The prior repo's margin framing has no all-inputs guarantee audit.

7. Injection into a trained composed model: 10/10 new facts land exactly (closed form)
   with 18-32% collateral — the removal/injection asymmetry. (Prior insertion work is
   certified insertion in toys and the gameability result.)

## In the prior repo, absent from this session (candidates to cite or import)

1. The real-data ladder: MNIST/SVHN triage economics (structure bought first at
   saturation, 65-82% clean kept vs 16-34% random), the 6-layer LM audit (quantization
   fragility AUC 0.958, normalized margins rho 0.846), the 500M bilinear GPT 36-scalar
   single-point-of-failure audit, TOFU at 410M-1.4B. The session is all synthetic.
2. Capacity vs architecture: bilinear beats param-matched ReLU MLP (+13-70%), SwiGLU ~
   bilinear, ~2.4-3.1 bits/param, d^1.65-1.7 scaling; random bilinear features are
   WORSE than random ReLU features (the advantage is in learned L,R).
3. Two solution phases (degenerate tie-manifolds vs max-margin tension webs) and
   criterion gameability (785 facts at d16 as +-1e-9 ties) — directly relevant to
   F13b: our capacity curve uses bare argmax; a margin-floored variant would be the
   robust version (the editing LPs already use eps = 0.5, but the capacity sweep does
   not).
4. The silence-code mechanism detail (signed cancellations in L; per-neuron 1-D
   signed-graph embedding; anti-Rayleigh + reweighting + hinge-repair construction at
   0.70x trained capacity, challenge-legal).
5. The noise-dial (per-example input noise as a training-time differential-forgetting
   knob with smooth dose-response).
6. Retension cost accounting: repair trades weight proximity for function fidelity
   (10-20x larger total weight change). The session never measured edit norms for the
   alternating LPs — worth adding if the post compares editing costs.

## Loose ends the comparison surfaces

- Run the session's attribution bins on a SUB-capacity 2-layer model (e.g. 300 facts,
  H=40) to close the loop with p2's opportunistic-storage law: prediction, the
  "either alone (redundant)" bin dominates and "needs both (composed)" ~ 0.
- Re-run F13b with a margin floor (his gameability point) — plateau should persist.
- Measure ||Delta W|| for the alternating-frame LP edits (his 10-20x retension cost
  number is the natural comparison).
- His t3c two-stage neighbor pipeline (weights-side shortlist -> forward-margin
  confirmation) vs our Gram forecast on the same model would settle which pre-edit
  forecaster is stronger where.

## Update 2026-08-11: the sister session's critique and the battery outcome

The prior repo's session reviewed this comparison and returned a critique: the LP editor
is "certified masking" (exact on constrained logits only; block 1 is not even an exact
frame — logits are quadratic in block-1 weights; conceded), the retain set must be
enumerable (conceded; their KL-to-base anchor has no LP analog), and the missing
experiment is their diagnosis battery on LP-edited models. Battery run here (registered
P17-P21; details in results.md):

- Their relearn-speed prediction REFUTED (25 vs 20 steps — no masking signature; weak
  test power) and no perturbation resurrection (reversion ~chance at all noise levels).
- Their cost concern CONFIRMED exactly (alternation = 10.0x the KKT edit's weight
  change) and superseded by the sharper finding: LP vertex solutions collapse retained
  margins to the constraint floor (median 24 -> 0.5), making the edited model 50-100x
  more noise-fragile than a from-scratch retrain. "Certified masking" in this toy is
  better named "certified brittleness". eps = 10 improves fragility ~6x at 2x cost but
  stays vertex-pinned; phase-2 max-min-margin is the un-run method both sides now point
  to.
- Synthesis adopted: margin-LP for certified stage-1 deletion where frames exist;
  gradient repair + anchoring for robustness, depth, and non-enumerable retain sets.
