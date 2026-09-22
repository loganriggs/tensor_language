# A shared 96-atom bank versus output-local atoms at 3,040 documents

22 September 2026, 18:55 UTC. Claude (Fable). Registered before any fitting. Follows BIDEGREE_DATA_SCALING_PLAN_V1.md: the output-local 96-atom correction reaches 34.3 / 34.1% fresh small-output value error at 3,040 documents (pred_b passed) but its six-atom spans per output are not identified across starts (every output has a canonical correlation below 0.31; pred_e failed) while the correction functions agree at 0.76–0.97.

**Question.** At matched coefficient count and the same data, does one bank of 96 quartic atoms shared by all 16 outputs (per-output coefficients) match or beat the output-local correction, and is the shared bank's span identified across starts where the local spans were not?

**Data, teacher, loop.** Exactly the 3,040-document fitting set of the previous rung (608 cache documents + FIT_CORPUS_TOKENS_V1), the same 64 held-out documents for the snapshot, exact teacher F4, unweighted objective (mean over 16 outputs of the relative squared error of the correction to residual = F4(x) − parent(x)), Adam lr 0.05, 400 updates, evaluation of the held-out objective every 10 updates, seeds 25001 and 25002. Fresh panel, matched pairs and the sensitivity-weighted calibration error scored exactly as before and never fitted.

**Arms.** SHARED: 96 atoms, each a product of four unit reads of x, used by every output with its own profiled coefficient (96 × 4 × 1,152 + 16 × 96 = 443,904 coefficients). LOCAL: the previous receipt's output-local fits at 3,040 documents (6 atoms per output, 442,464 coefficients; not re-run).

**Audit.** Cross-start cosine of the correction functions on the fresh panel per output; canonical correlations between the two seeds' 96-atom spans on the fresh panel (plain inner product); for comparison the local arm's per-output six-atom CCs from the previous receipt.

**Predictions (scored as written; failures preserved).**
1. pred_a_integrity: replays ≤ 1e-4 (S + C vs stored rows on calibration and fresh; vs x on both fitting panels), teacher ≤ 1e-4, exports reloaded reproduce every fresh error ≤ 1e-8, planted controls pass.
2. pred_b_shared_not_worse: for both seeds, the shared arm's fresh small-output value RMS ≤ 1.05 × the local arm's (same seed) and response RMS ≤ 1.05 × the local arm's. Prior: unsure.
3. pred_c_shared_better: for both seeds, shared value RMS ≤ 0.95 × local and response RMS ≤ 0.95 × local. Prior: unsure.
4. pred_d_cross_start_stability: the shared arm's cross-start cosine on the fresh panel has mean ≥ 0.9 and minimum ≥ 0.8 over outputs 4–15. Prior: unsure.
5. pred_e_bank_identified: at least 48 of the 96 canonical correlations between the two seeds' shared-bank spans are ≥ 0.9. Prior: unsure.

**Price.** Forwards: 432 (the same four captures; bar 460); model backwards 0; two fits of 443,904 parameters for 400 updates on 194,560 states. Literal cost per atom unchanged (4 × 1,152 read multiply-adds, 3 products); per output 96 coefficient multiplies and adds instead of 6.

**Not licensed by this rung.** No feature names, no semantic labels, no intervention. A passing pred_b and pred_e licenses a frozen-bank native-removal rung; a failing pred_b says output-locality is real at this capacity.
