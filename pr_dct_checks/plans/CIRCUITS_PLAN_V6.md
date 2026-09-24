# Circuits lane, rung C3h: held-out FineWeb transfer of the creator circuits

24 September 2026, 04:05 UTC. Claude (Fable). Registered before running. Logan (24 Sep): transfer from FineWeb to AdvBench failing is fine; what matters is that a circuit derived on data X predicts its effect on data Y from the same distribution. C3 tested the creator sets on the very 8 FineWeb contexts (rows 96–103) they were derived from (C1c) and that the heads were chosen on (C1a), so within-FineWeb transfer is untested.

**Run.** `run_circuits_c3.py --row-offset 104`: identical creator sets, shared core, heads and random control; 8 fresh FineWeb contexts (rows 104–111) never used in any circuits rung; AdvBench block unchanged (serves as the reproduction of C3's AdvBench numbers). Per-context readers are recomputed per context as before.

**Predictions (scored as written; failures preserved).**
1. pred_a_instrument: zero-mask exact; all-patched ≥ 0.99.
2. pred_b_transfer_creators: median fraction removed by creators(u) on the held-out FineWeb contexts ≥ 0.7 × C3's in-sample value (0.7 × 0.175 = 0.123). Prior: likely — the creator sets were chosen by counting across 8 contexts, and C1c's stable-unit R² held across contexts.
3. pred_c_transfer_heads: creators + heads ≥ 0.7 × 0.60 = 0.42; heads alone ≥ 0.7 × 0.38 = 0.27. Prior: likely (head reuse in C1a was 63–81% across 100 directions on the same contexts, though).
4. pred_d_selective_heldout: selectivity ratio ≥ 2 on the held-out contexts. Prior: unsure.
5. pred_e_advbench_reproduces: AdvBench creators within ±0.03 of C3's 0.048 (a determinism check on the instrument).

**Price.** As C3: ~70 min.
