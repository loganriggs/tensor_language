# Circuits lane, rung C3: the four properties on creator-unit circuits

24 September 2026, 02:15 UTC. Claude (Fable). Registered before running (launch after the C1c per-block breakdown lands; it supplies the creator sets).

**Circuit definition.** For output direction u: creators(u) = the 30 units of blocks 8–12 that appear most often in the top-5-per-block of C1c's closed-form energy attribution across the 8 contexts; heads(u) = C1a's top-3 value-transport heads. Shared core = creator units present in ≥ 50% of the 32 directions' creator sets; specific(u) = creators(u) minus the shared core. Readers(u, c) = C1a's top-100 hidden-mixed-derivative units for that context (for comparison).

**Measurement.** Finite interaction D_u(spec) = f_u(2αv_u; spec) − 2f_u(αv_u; spec) + f_u(0), α = 1% of the mean block-8 residual norm, v_u the power-iteration probe, all patches to true clean values; fraction removed = 1 − D_u(spec)/D_u(∅). 8 held-out FineWeb contexts (as before) and 8 AdvBench prompts; 32 directions (16 readers, 16 token directions).

- **Removal:** creators(u); creators(u) + heads(u); readers(u, c); 30 random units of blocks 8–12; the shared core alone; specific(u) alone.
- **Selectivity:** for every ordered pair (A, B), the fraction of D_B removed by patching creators(A) (and by specific(A)); the selectivity ratio for A = frac_A(specific(A)) / mean_{B ≠ A} frac_B(specific(A)).
- **Composition:** for every pair, frac_A(creators(A) ∪ creators(B)) against frac_A(A) + frac_A(B); the deviation |union − sum| (cancellation or overlap makes it non-zero; overlapping units counted once).
- **Generalisation:** the removal fractions on the 8 AdvBench prompts with the FineWeb-derived creator sets.
- **Costs:** |creators|, |shared core|, heads; literal multiply–adds for the creator forms (2 reads × 1,152 per unit) and for the transport (the three heads' value maps, 2 × 128 × 1,152 each).

**Controls.** (a) Zero-mask patching reproduces the unpatched forward exactly; patching all heads' values and all units to clean removes ≥ 0.99 of D at 1% (the finite analogue of freezing everything). (b) CPU smoke test.

**Predictions (scored as written; failures preserved).**
1. pred_a_instrument: (a) passes.
2. pred_b_removal: median fraction removed by creators(u) ≥ 0.3, and by creators + heads ≥ 0.5, on FineWeb. Prior: unsure (the creators carry ~half of the closed-form energy; the finite response has a 30% remainder).
3. pred_c_selective: median selectivity ratio ≥ 2 (a direction's specific creators affect it at least twice as much as they affect other directions). Prior: unsure.
4. pred_d_composable: median |union − sum| ≤ 0.1. Prior: unsure.
5. pred_e_generalises: median AdvBench removal by creators(u) ≥ 0.7 × the FineWeb value. Prior: unsure.

**Price.** 16 contexts × 32 directions × ~70 patched evaluations × 2 forwards ≈ 72k forwards through 10 blocks ≈ 40 min.

**Closing.** This rung ends the lane's ladder: the closing section states what survives as a circuit (creator units × transport heads × reader units, with the weight-space decomposition of C1c and the finite checks), which of the four properties hold at what level, and the literal costs.
