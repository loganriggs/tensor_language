# Symmetric DCT rung 5 (final): where the value-path interaction is created and transported

23 September 2026, 03:45 UTC. Claude (Fable). Registered before running. Rung 4's budget (first 8 contexts) puts the interaction in the attention **value path** (pure-value 35%, pattern × value 17%) and in non-attention paths (40%), with the pattern-factor pieces at ~8%. Values are linear in the block input, so a second-order effect through values must be created downstream: attention transports the perturbation linearly (context-dependent mixing over positions) and the bilinear MLPs downstream, and the final norm, create the curvature. This rung locates both ends for each context's own top interaction direction v₁ᶜ (top eigenvector of B_{c,k}).

**Computation.** For each of the 16 held-out contexts and 16 readers, with v = v₁ᶜ (from a fresh Hessian), the fraction of u·H[v, v] removed by freezing: each block's MLP alone (blocks 8–17), all MLPs, the top-20 / top-100 MLP units across blocks 8–17 by hidden mixed derivative, 20 random units; the values of each attention block alone (`attn_modes` values-frozen in that block), values of all blocks, values of blocks 8–9 together; everything (control). Also reported: the Hessian's eigenvalue PR and the share of u·H[v,v] with respect to the whole spectrum.

**Controls.** (a) Freezing everything gives exactly 1.00 (≤ 1e-6). (b) Values-frozen-in-all-blocks equals the values-frozen mode of rung 4 (same code path). (c) CPU smoke test.

**Predictions (scored as written; failures preserved).**
1. pred_a_instrument: (a) passes.
2. pred_b_mlp_curvature: freezing all MLPs removes ≥ 0.6 (median). Prior: likely (rung 1: 0.76 for the mean-form direction).
3. pred_c_few_mlp_blocks: the two largest single-block MLP removals sum to ≥ 0.5 (median). Prior: unsure.
4. pred_d_transport_8_9: values-frozen in blocks 8–9 removes ≥ 0.6 × what values-frozen in all blocks removes (median ratio). Prior: likely.
5. pred_e_units: the top-100 MLP units remove ≥ 0.5 (median). Prior: unsure.

**Price.** 16 Hessians + ≈ 16 × 16 × 28 ≈ 7k jvp-of-jvp passes (≈ 20 min).

**Closing.** Whatever the outcome, this rung ends the lane: the closing section of the note states which description of the block-8 → reader interaction survives (transport blocks × curvature blocks × units, with the measured shares), what does not (fixed bases, low rank, seed-stable fitted factors), and what a finite-intervention check would still owe.
