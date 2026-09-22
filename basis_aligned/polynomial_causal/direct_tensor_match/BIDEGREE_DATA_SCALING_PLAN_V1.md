# Data scaling of the untyped 96-atom correction, with a cross-start span audit

22 September 2026, 18:20 UTC. Claude (Fable). Registered before any fitting. Follows BIDEGREE_DATA_ABLATION_PLAN_V1.md: seven times the fitting data took the 96-atom correction from ~50% to 39.8–39.9% fresh small-output value error (Codex's hybrid: 46.6%), the bidegree-typed arm was slightly worse than untyped at both data sizes (closed), the fits still overfit (fit 0.07 vs held-out 0.46 at step 400) and cross-start cosines rose to 0.71–0.98 (mean 0.87) without reaching identification.

**Question.** Does the untyped 96-atom correction keep improving, and does it become identified across starts, as the fitting data grows another 2× and 5×?

**Fitting data.** The exact teacher F4(x) (replays the stored calibration targets to ≤ 1e-4, checked in-run). Documents: the 608 fitting documents of the previous rung (`fineweb_n480_skip80` + `fineweb_n192_skip11000`, first 64 positions) plus a new frozen corpus `FIT_CORPUS_TOKENS_V1.pt` (2,432 FineWeb documents streamed and frozen by `prepare_fit_corpus.py` with the fresh panel's exclusion rules: every local cache prefix, the fresh panel's tokens and every enumerated fresh document excluded; receipt `FIT_CORPUS_ROWS_V1.json`). Ladder: 608 documents (the previous receipt, not re-run), 1,216 documents (608 + the first 608 of the corpus), 3,040 documents (608 + all 2,432). Held-out for the snapshot: the same 64 documents as before (never fitted). Objective: unweighted relative squared error of the correction to residual = F4(x) − parent(x), mean over the 16 outputs. The sensitivity-weighted error on the 6,144 calibration states is a held-out secondary metric; the fresh 16,384-state panel and the 2,494 matched pairs are scored exactly as before and never enter fitting or selection.

**Arm.** Untyped only: per output 6 atoms of four unit reads of x (96 atoms, 442,464 coefficients). Coefficients profiled in closed form (ridge 1e-8); reads by Adam lr 0.05, 400 updates, seeds 25001 and 25002; snapshot at the best held-out objective (evaluated every 10 updates).

**Audit.** For each data size and output, the canonical correlations between the two seeds' six-atom spans on the fresh panel (plain inner product; orthonormal bases by QR; singular values of Q₁ᵀQ₂), and the cross-start cosine of the correction functions as before.

**Predictions (scored as written; failures preserved).**
1. pred_a_integrity: S + C reproduces the stored rows (calibration, fresh) and x (both fitting panels) to ≤ 1e-4; the teacher reproduces the stored calibration targets to ≤ 1e-4; exports reloaded from disk reproduce every reported fresh error to ≤ 1e-8; the planted controls pass.
2. pred_b_more_data_helps: at 3,040 documents, both seeds' fresh small-output value RMS ≤ 0.36 and response RMS ≤ 0.39 (608 documents: 0.398/0.399 and 0.427/0.434). Prior: unsure.
3. pred_c_monotone: for both seeds, value RMS at 3,040 ≤ at 1,216 ≤ at 608 (the previous receipt's numbers). Prior: likely.
4. pred_d_cross_start_stability: at 3,040 documents, the cross-start cosine on the fresh panel has mean ≥ 0.9 and minimum ≥ 0.8 over outputs 4–15. Prior: unsure.
5. pred_e_spans_identified: at 3,040 documents, at least 6 of the 12 small outputs have all six canonical correlations ≥ 0.9 between the two seeds' atom spans (Codex's hybrid dictionaries: 0 of 12 at the 0.9 bar). Prior: unsure.

**Price.** Forwards: 304 (corpus) + 84 (previous fitting panel) + 12 (calibration) + 32 (fresh) = 432, bar 460; model backwards 0; four fits (two sizes × two seeds) of 442,464 parameters for 400 updates each, the largest on 194,560 states. Literal program cost per atom unchanged.

**Not licensed by this rung.** No feature names, no semantic labels, no intervention. A passing pred_e licenses a frozen-dictionary rung (the identified atoms, fixed, scored under native removal on the fresh panel); a failing pred_e with a passing pred_b says the class keeps improving without becoming identified, and the next lever is sharing reads across outputs.
