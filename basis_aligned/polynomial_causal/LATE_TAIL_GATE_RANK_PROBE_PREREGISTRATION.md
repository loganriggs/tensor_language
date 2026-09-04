# Preregistration — late_tail_gate_rank_probe (Claude, LANE 1 CUDA) — Registered 2026-09-04 02:19Z (box clock)

Sign convention (§2135): every CE number is CE ADDED ABOVE THE REAL MODEL on held-out docs 0–63 (FRESH split; fits on docs 96–191) — LOWER IS BETTER.

## Question
Each late MLP reads the tail through a core-gated LINEAR read (§2791): the cross term J(c) t = W_D[(L c) ∘ (R t) + (R c) ∘ (L t)], exactly the MLP's Jacobian at the core applied to the tail, worth ~83% of the 768-cost (§2780). It is linear in the gate c, so J(c) = Σ_i c_i J_i over the 768 core coordinates. How many INDEPENDENT gate modes does the model use? A low-rank gate (≲ 64 modes) would make the tail read programmable as a few scalar gates — the only candidate in the late-tail lineage (§2775–§2798) that could reduce parameters. The pattern of this model (§2673 MLP operator families 438–749 of 1152; §2779 tail read not low-rank; §2798 readout channel eff rank 261) predicts a high-rank gate.

## Instrument (exact, weight-side; CE arms on the FRESH split)
Per late block l ∈ 8..17: G_ij = tr(J_iᵀ J_j M_t), computed without materialising J_i via Hadamard-trace identities (G = Lkᵀ(Gd∘Rt Mt Rtᵀ)Lk + Rkᵀ(Gd∘Lt Mt Ltᵀ)Rk + cross terms; Gd = W_Dᵀ W_D), with M_t the tail second moment of the normalised MLP input; whitened by the core second moment M_c: Gw = M_c^{1/2} G M_c^{1/2}; eigenvalues = gate-mode energies under the independence factorisation E‖J(c)t‖² ≈ tr(M_c G). Reported per block: effective rank and rank-90 of Gw (and of unweighted G), energy captured by the top 16/64/128/256 modes, the core input's own effective rank, and the independence factor tr(M_c G) / measured E‖J(c)t‖² (descriptive). Instrument checks: for blocks 8 and 17, six random (i, j) pairs with J_i materialised, relative error of G_ij ≤ 1e-6 (double); GATE_EXACT (the split MLP(c) + cross + MLP(t) with the cross computed directly) adds ≤ 1e-3 CE; baseline and SPLIT8_1024 repro. Arms: GATE_0 (cross term removed in all ten late MLPs), GATE_64 / 128 / 256 (gate input c projected on the top-k modes, all ten blocks). recovered(k) = 1 − GATE_k / GATE_0. Price ≈ 96 + 64 × 7 ≈ 545 GPU document-forwards + ten 4608² double Grams (~30 s).

## Predictions (scored exactly as written)
- pred_a_instrument: |baseline − 3.0322401| ≤ 1e-4; |SPLIT8_1024 − .0374| ≤ .015; |GATE_EXACT| ≤ 1e-3; direct-check relative error ≤ 1e-6.
- pred_b_gate_is_high_rank: median over blocks of eff_rank(Gw) ≥ 307 (= 0.4 × 768). Null: ≤ 128.
- pred_c_gate_rank90_large: median rank-90 of Gw ≥ 256. Null: ≤ 96.
- pred_d_top64_gate_modes_carry_a_minority: median energy captured by the top 64 modes ≤ 0.5. Null: ≥ 0.8.
- pred_e_gate_narrower_than_its_core_input: median eff_rank(Gw) / eff_rank(M_c) ≤ 0.8. Null: ≥ 1.0.
- pred_f_gate128_recovers_at_most_half_of_gate0: recovered(128) ≤ 0.5. Null: ≥ 0.8.

Bars: {"ce_tol": 1e-4, "repro_tol": 0.015, "exact_tol": 1e-3, "gram_tol": 1e-6, "b_eff": 307, "c_r90": 256, "d_cap": 0.5, "e_ratio": 0.8, "f_rec": 0.5}. Null bars: {"b_eff": 128, "c_r90": 96, "d_cap": 0.8, "e_ratio": 1.0, "f_rec": 0.8}.

## What each outcome means
b, c, d, f TRUE: the gate is high-rank like everything else — the tail read cannot be programmed as a few scalar gates, and the late-tail lineage closes with no parameter reduction anywhere. b/d/f FALSE with nulls met: a compressible gate (≤ 128 modes recover ≥ 80% of the cross term's CE) — the first real parameter-reducing item in the lineage; would then be priced honestly (PR2 accounting) before any frontier claim. e: whether the gate uses fewer directions than the core carries (a selective gate) or all of them. Nothing installs into the §312 frontier here; the strict explained fraction is unchanged by any outcome.
