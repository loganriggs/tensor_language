# Where the late shared core comes from: massive-subspace provenance map — preregistration

Registered 2026-09-03 20:48Z (box clock), before the script exists. Lane 1 (CUDA; one 96-doc collection pass plus a 64-doc
baseline check). SIGN CONVENTION (§2135): the only CE number is the baseline instrument (docs 0–63, 3.0322401). All other numbers
are FRACTIONS in [0,1] — subspace overlaps ov(U_j, V_k) = ‖U_jᵀV_k‖²_F/j (chance k/1152) and energy fractions tr(MᵀCM)/tr(C) —
HIGHER = more of the object lies in the core. Descriptive; nothing installs into the §312 frontier.

## Question
§2713: the dictionary shared by the late MLP writes (§2710) is the final residual stream's dominant geometry (XPCA eff rank 19).
Reusability then means: some early site ESTABLISHES a low-dimensional stream subspace M that later sites keep writing into.
When is M established, how much of the stream's energy it carries by depth, and which of the 36 sites write into it?

## Objects (docs 96–191)
M_16 = CORE_TW_16 (§2710/§2713 pooled late-MLP write PCA, trace-weighted); X_16 = final-stream PCA top-16 (reported alongside).
Per block l = 0…17: RESID_l = residual stream after block l's MLP write (+ bias); centred covariance C_l and uncentred second
moment S_l; ov_l = ov(M_16, RESIDPCA_l_128); e_l = tr(M_16ᵀ S_l M_16)/tr(S_l) (uncentred energy fraction of the stream in M);
ec_l the centred version. Per site s (36 writes): f_s = tr(M_16ᵀ C_s M_16)/tr(C_s), the centred energy fraction of the write in M.

## Predictions (bars fixed now)
- **pred_a_instrument**: baseline within 1e-4 of 3.0322401; CORE_TW eff rank within .5 of 10.004; ov(M_16, X_128) ≥ .70 (§2713
  measured .718 — the objects must be the same as before).
- **pred_b_established_by_block_3**: ov_3 ≥ .70 (chance .11). Null: ≤ .30.
- **pred_c_carries_the_stream**: e_17 ≥ .50 and Spearman(l, e_l) ≥ .80 over l = 0…17. Null: e_17 ≤ .20.
- **pred_d_writers_early_and_late**: f_s ≥ .50 for mlp16 and mlp17, AND f_s ≥ .50 for at least one site in blocks 0–3. Null: no
  site in blocks 0–3 has f_s ≥ .20.
- **pred_e_middle_avoids_it**: median f_s over the 20 sites of blocks 4–13 ≤ .15. Null: ≥ .35.

## Price
96 + 64 GPU document-forwards, 54 accumulators, ~60 eigendecompositions ≈ 15 s. Output massive_subspace_provenance_map_probe_results.json.
Frozen: this file, §2713 results (20ff21d0…), checkpoint, fit_natural.pt.
