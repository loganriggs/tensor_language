# Source/context geometry census of the selected quartic branch

22 September 2026, 16:40 UTC. Registered before execution. First rung of the continuation named in the final Codex compression report (section 8): build features from propagated source/context geometry rather than another output-local dictionary on the normalized MLP16 input. This rung fits nothing. It measures, exactly, whether that geometry offers leverage.

**Exact split.** The pre-normalization MLP16 input at a position decomposes into propagated block writes: x_mid = c_E x0 + sum_{l<=15} c_l (a_l + m_l) + a_16 with c_l = prod_{j=l+1}^{16} lambda_{j,0} and c_E the embedding-reinjection recurrence through block 16. Group the sources as STATE = c_E x0 + sum_l c_l m_l (the position's own embedding and MLP writes, including the constant Down biases) and CONTEXT = sum_l c_l a_l + a_16 (attention writes, which read other positions). With S = STATE / rms(x_mid) and C = CONTEXT / rms(x_mid), the target's input is x = S + C exactly.

**Exact bidegree pieces.** With m(x) = lambda_{17,0} D16[(L16 x) o (R16 x)] and B17(u, v) = D17[(L17 u) o (R17 v)], the selected quartic F4(x) = B17(m, m) projected on the 16 fixed readers splits into five pieces by the number of S factors: P40 = B17(m20, m20), P31 = B17(m20, m11) + B17(m11, m20), P22 = B17(m20, m02) + B17(m02, m20) + B17(m11, m11), P13 = B17(m11, m02) + B17(m02, m11), P04 = B17(m02, m02), where m20 = m(S), m02 = m(C), m11 = m(S + C) - m(S) - m(C). The fixed CP512 parent splits the same way (each atom prod_s a_s^T x = prod_s (a_s^T S + a_s^T C), expanded by the number of S factors), so the residual the local dictionaries failed to repair splits into five residual pieces R_i as well. All identities are algebraic; they are checked numerically against the stored calibration targets and the parent prediction.

**Panel.** The 6,144 calibration states (96 FineWeb documents x 64 positions, the same tokens and order as SENSITIVE_ROOT_CALIBRATION_V2 / NATIVE_QUARTIC_COVARIANCE_V1 / QUARTIC_ADDITIONAL_STATES_V1). No fitting, no evaluation-panel access, no intervention.

**Reported.** Per output g (all 16, with outputs 4-15 the registered set): the 5x5 Gram matrix of the pieces and of the residual pieces, the diagonal shares s_i = |P_i|^2 / sum_j |P_j|^2 (sum to 1 by construction; cross terms reported separately because the pieces are not orthogonal), the same for residual pieces; per-source energies of the 34 propagated sources inside S and C; the covariance participation ratio and top-64 spectrum of S and of C; replay errors. Literal costs of forming S and C: zero new arithmetic (the writes exist in the native forward), one extra 1152-vector of storage per state.

**Frozen predictions (scored as written; failures preserved).**

1. pred_a_instrument: STATE + CONTEXT reconstructs x_mid to relative error <= 1e-4 on all 6,144 states; the five pieces sum to the stored native targets to relative error <= 1e-4; the five parent pieces sum to the parent prediction to relative error <= 1e-8.
2. pred_b_state_piece_dominates: the mean over outputs 4-15 of the pure source-state share s_40 is >= 0.5. Prior: unsure.
3. pred_c_context_low_dimensional: the participation ratio of the covariance of C is <= 0.25 x that of S. Prior: unsure.
4. pred_d_mixed_pieces_live: the mean over outputs 4-15 of s_31 + s_22 + s_13 is >= 0.2. Prior: unsure.
5. pred_e_residual_in_context_pieces: the mean over outputs 4-15 of the residual share carried by pieces with at least one context factor (r_31 + r_22 + r_13 + r_04) is >= 0.6. Prior: likely; this is the hypothesis that would explain why x-only local atoms fail to repair the residual.

**Price.** 96 documents in 12 forwards of 8 through block 16 plus attention 16; one 4-document replay forward for the x16 capture check; 0 backwards; 0 fits; 0 parameter updates. Bar: 16 forwards.

**Not licensed by this rung.** No feature names, no circuit claim, no dictionary. A passing pred_e licenses the next registered rung: a matched-capacity correction whose atoms are bidegree-typed (products of reads of S and reads of C) targeting the residual pieces, scored on the same panels and gates as the hybrid experiment.
