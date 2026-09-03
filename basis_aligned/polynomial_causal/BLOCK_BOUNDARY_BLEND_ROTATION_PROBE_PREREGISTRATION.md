# block_boundary_blend_rotation_probe — preregistration (Registered 2026-09-03 23:26Z (box clock))

Lane 1 CUDA (Claude). Follows §2757 (observed, unregistered: the read frame turns 1.4–1.9× more across a block boundary,
mlp_l → attn_{l+1}, than inside a block). Each block begins with x ← λ₀x + λ₁x₀ (the blend with the normalised token embedding
x₀). Question: is the early frame rotation produced by that blend, or by mlp_l's write? Decompose the mlp_l → attn_{l+1} step into
(write) mlp_l's input frame → the frame of the PRE-blend residual x + mw_l, and (blend) the pre-blend frame → attn_{l+1}'s input
frame; and test causally whether attention reading through the pre-blend frame is what the blend's rotation costs.

Sign convention (§2135): CE numbers are CE ADDED ABOVE THE REAL MODEL on held-out docs 0–63 (FRESH split; fits docs 96–191) —
LOWER IS BETTER; gaps "arm − EARLY22_OWN_768" = extra damage.

Construction: §2753's own top-768 input cores (rms-normed inputs, fit docs 96–191) plus, from the same pass, the top-768 core
U_pre(l) of the rms-normed pre-blend residual after block l's MLP (l = 0..16). Principal angles as in §2757 (≤ 384 of 768 free);
n30 = number of angles > 30°. For l = 0..7: write(l) = n30(U_mlp_l, U_pre(l)); blend(l) = n30(U_pre(l), U_attn_{l+1});
total(l) = n30(U_mlp_l, U_attn_{l+1}) (§2757's block-boundary numbers). CE arms at k = 768 over the 22 early sites:
EARLY22_OWN_768 (all own); EARLY_ATTN_PREBLEND_768 (attn_1..attn_10 read through U_pre(l−1), all other early sites own);
EARLY_ATTN_PREV_768 (attn_1..attn_10 read through mlp_{l−1}'s own core, others own — §2755's carry-over restricted to attention).

Arms: EARLY22_OWN_768, EARLY_ATTN_PREBLEND_768, EARLY_ATTN_PREV_768.

Frozen: this file, §2757 results (frame_principal_angle_spectrum_probe_results.json), checkpoint, fit_natural.pt.

- pred_a_instrument: baseline 3.0322401 within 1e-4; EARLY22_OWN_768 within .02 of .057.
- pred_b_blend_step_dominates: median over l = 0..7 of blend(l) / write(l) ≥ 1.5. Null: ≤ 1.0.
- pred_c_blend_accounts_for_most: median over l = 0..7 of blend(l) / total(l) ≥ 0.6. Null: ≤ 0.3.
- pred_d_preblend_frame_costs: EARLY_ATTN_PREBLEND_768 − EARLY22_OWN_768 ≥ .020. Null: ≤ .005.
- pred_e_write_step_is_the_smaller_cost: (PREBLEND − OWN) ≤ 0.6 × (PREV − OWN). Null: ≥ 0.9 × (PREV − OWN).

Price: 1 fit pass (96 docs) + baseline + 3 arms × 64 docs = 352 GPU document-forwards (≈ 15 s) + CPU SVDs. Descriptive; nothing
installs into the §312 frontier.
