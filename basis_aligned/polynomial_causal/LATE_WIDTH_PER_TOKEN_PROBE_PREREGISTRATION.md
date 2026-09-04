# late_width_per_token_probe — preregistration (Registered 2026-09-04 00:23Z (box clock))

Lane 1 CUDA (Claude). Follows §2773/§2774 (the late MLPs read the 768-complement of the bus through both bilinear branches; .125 at
768). Is that width cost a TAIL-DICTIONARY phenomenon — concentrated on rare targets or a few positions — or spread over ordinary
tokens? Arms: the same patches as before, but the per-token CE (16,384 held-out target tokens) is kept and the CE ADDED is
decomposed by (i) the per-token loss increase's concentration, (ii) target frequency in the fit set (docs 96–191), (iii) position.

Sign convention (§2135): CE numbers are CE ADDED ABOVE THE REAL MODEL on held-out docs 0–63 — LOWER IS BETTER. Priors: LATE_MLP_768
= .1249 (§2773), SPLIT8_1024 = .0374 (§2769).

Arms (per-token): BASE, SPLIT8_1024, LATE_MLP_768, BUS_768. "Share" of a token subset S in an arm = Σ_{t∈S} Δ_t / Σ_t Δ_t, with Δ_t
the per-token CE increase (signed). "Token share" = |S| / 16,384. RARE = targets with fit-set count 0; FREQ = fit-set count ≥ 100.

Frozen: this file, §2774 results (late_mlp_branch_width_probe_results.json), checkpoint, fit_natural.pt.

- pred_a_instrument: baseline 3.0322401 within 1e-4; mean of per-token BASE CE equals the baseline within 1e-4; SPLIT8_1024 within
  .015 of .0374; LATE_MLP_768 within .015 of .1249.
- pred_b_width_cost_is_concentrated: in LATE_MLP_768 the top 10% of tokens by Δ_t carry ≥ 0.60 of the total CE added. Null: ≤ 0.35
  (spread like a uniform perturbation).
- pred_c_rare_targets_overweighted: RARE's share of LATE_MLP_768 ≥ 1.5 × RARE's token share. Null: ≤ 1.0 × (rare targets carry no
  more than their share — not a dictionary effect).
- pred_d_not_a_position_effect: positions 0–15's share of LATE_MLP_768 ≤ 2 × their token share (16/256). Null: ≥ 4 ×.
- pred_e_same_tokens_as_whole_program: Spearman(Δ_t of LATE_MLP_768, Δ_t of SPLIT8_1024) over all 16,384 tokens ≥ 0.30. Null: ≤ 0.10.

Price: 1 fit pass (96 docs) + 4 arms × 64 docs = 352 GPU document-forwards (≈ 15 s). Descriptive; nothing installs into the §312
frontier; bases are data covariances scored by CE only (§2118 stays closed).
