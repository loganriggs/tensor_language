# What we know about MLP 0 and MLP 1 (the measured black box)

Compiled 2026-07-29. These two interfaces carry 58% of the completeness-ledger floor mass
(MLP 0: 3.63 nats, 29% understood; MLP 1: 2.15 nats, 0% generated). Everything below is a
verified finding with a script behind it; "→ attack" marks the planned explicit-program
substitution for that function.

## MLP 0 (block-0 bilinear MLP)

1. **Exact object.** Folds exactly to a symmetric third-order tensor (CP rank 4608, fold gate
   ~1e-7). Neuron-level DENSE: flat usage, pruning half costs +0.030, downstream readers touch
   all neurons. Channel effective rank median 68/128 — dense in weight-rank too.
2. **Worth.** Whole-block importance +2.50 nats; single-interface floor 3.63 nats.
3. **Realized interface is thin.** What layer-1 QK actually reads from it: effective rank ~10
   per channel, token-identity share 0.56 (the "manifold collapse" result). The block is dense
   but its *consumed output* is thin and mostly token-keyed.
4. **Pricing.** Token tables + oracle rank-16 adapters recover 78% of its interface gap
   (rank-64 → 99.6%, ~2.4 Mbit). Honest (non-oracle) linear generator from PCA-64: 29%; the
   remainder is genuinely nonlinear (matches the R² 0.45–0.64 ceiling).
5. **Authorship.** The context remainder of layer-1 QK factors (21–41% of factor norm) is
   authored by MLP 0, not layer-0 attention.
6. **Negative results.** Dilutes the *linear* previous-token readout (probe rises when MLP 0 is
   ablated). Not category-selective (the category-engine strong claim is falsified; damage is
   general lexical).
   → **attack:** explicit program `out ≈ TokenTable[token] + low-rank contextual adapter`,
   token table fit by least squares, adapter rank swept 4–16; verify by predicate-style
   substitution (natural dCE + task battery), then scalar/gain finetune.

## MLP 1 (block-1 bilinear MLP)

1. **Memory key-enrichment.** Largest in-place key-enrichment writer in the memory pipeline
   (0.50, front-loaded band 0.81); the enriched state is context-BOUND (cross-context
   transplant fails 0.04 vs 1.00 own-restore).
2. **Induction match service (two-branch-specific).** Ablating MLP 1 collapses the
   induction-match rate 85% and inverts the task; content-independent (holds on shuffled
   sequences: +7.27 → −0.88). Softmax models compute the same match without it. The
   two-branch product (q1·k1)(q2·k2) needs MLP 1 so both branches co-fire at the copy source.
3. **High-rank multiplexer.** Output top-8 principal components hold only 24.7% of variance;
   different tasks read nearly disjoint 16–32-dimensional slices (importance-direction overlap
   Jaccard 0.14–0.23). Partial low-rank replacement is WORSE than none for induction
   (rank 1–4 negative).
4. **The one shared direction (PC 0) is a content-versus-structure axis** (mean projection:
   punctuation −914, subword fragments +323, other +137).
5. **Task loads.** Dominant knockout in every battery task: inverts induction; +8.54 subword;
   +3.14 punctuation.
   → **attack:** decompose by *consumer*: explicit program
   `out ≈ TokenTable[token] + PrevTable[prev token] + (content/structure scalar)·pc0 +
   per-consumer slice adapters`, fit consumer slices against the circuit heads' QK reads
   (the induction-service slice first, since its function is now precisely named), verify by
   substitution on the induction predicate stack, finetune scalars.

## Where this is written up
- RESULTS_l0_mdl.md §32/§32b (atlas + generality + mechanism), paper_atlas_bilin18.md
  (§4 category falsification, §7 ledger, §7b properties, §7c induction predicate).
- Memory pipeline (MLP 1 enrichment): qk_two_layer_story.md §8; RESULTS §11g–11j.
- MLP 0 anatomy: MLP ARC results (block-0 sections), tick 201-205 entries in LOG.md.
