# late_mlp_branch_width_probe — preregistration (Registered 2026-09-04 00:20Z (box clock))

Lane 1 CUDA (Claude). Follows §2773 (the late width consumer is the MLP READ: blocks 8–17's MLP reads through the bus at 768 cost
.125, their attention reads .015; at 896 .066 vs .008). Goes below the MLP block: each late MLP is Down[(Left x) ⊙ (Right x)];
arms truncate the input of ONE branch only (Left reads x through U_8 at k, Right reads the exact x; and vice versa), all other
reads untouched; and the per-block MLP-only read cost at 768 (block l's MLP alone reads through U_8 at 768).

Sign convention (§2135): CE numbers are CE ADDED ABOVE THE REAL MODEL on held-out docs 0–63 (FRESH split; fits docs 96–191) —
LOWER IS BETTER. Priors: LATE_MLP_768 = .1249, LATE_MLP_896 = .0662, BUS_768 = .1636 (§2773).

Arms: SPLIT8_1024 (instrument), LATE_MLP_768 (repro), LEFT_768, RIGHT_768, LEFT_896, RIGHT_896, ONE_MLP_<l>_768 for l = 8..17.

Frozen: this file, §2773 results (late_width_by_kind_probe_results.json), checkpoint, fit_natural.pt.

- pred_a_instrument: baseline 3.0322401 within 1e-4; SPLIT8_1024 within .015 of .0374; LATE_MLP_768 within .015 of .1249.
- pred_b_one_branch_is_cheaper_than_both: max(LEFT_768, RIGHT_768) ≤ 0.70 × LATE_MLP_768. Null: max ≥ 0.95 × LATE_MLP_768 (one branch
  alone already carries the whole cost).
- pred_c_branches_symmetric: LEFT_768 / RIGHT_768 ∈ [0.5, 2.0]. Null: ratio ≤ 0.25 or ≥ 4.0 (one branch owns the width).
- pred_d_branch_costs_additive: (LEFT_768 + RIGHT_768) / LATE_MLP_768 ∈ [0.8, 1.3] (the product's truncation error is first-order
  in each branch's error; no strong interaction). Null: ≤ 0.6 or ≥ 1.6.
- pred_e_mlp_only_compounds: Σ_l ONE_MLP_l_768 / LATE_MLP_768 ≤ 0.70 (the §2771 compounding is carried by the MLP kind). Null: ≥ 0.95.

Price: 1 fit pass (96 docs) + baseline + 16 arms × 64 docs = 1184 GPU document-forwards (≈ 40 s). Descriptive; nothing installs
into the §312 frontier; bases are data covariances scored by CE only (§2118 stays closed).
