# early_residual_energy_budget_probe — preregistration (Registered Registered 2026-09-03 23:41Z (box clock))

Lane 1 CUDA (Claude). Follows §2759 (the blend x ← λ₀x + λ₁x₀ is geometrically inert from block 2 on although λ₁ = 8 and λ₀ is as
small as .064; inferred, not measured: the residual's energy dwarfs 8·x₀ by block 2) and §2760 (block 1's λ₀ = .0127 nearly
restarts the residual from 8·x₀, yet attn_1's frame is 234 of 384 angles from the embedding frame). This rung MEASURES the energy
budget of the early residual stream so those inferences stop being inferences: per block, the mean squared norm per token of the
pre-blend residual, of the post-blend residual, of the attention write and of the MLP write, on fit docs 96–191.

Sign convention (§2135): the one CE number (instrument) is CE ADDED ABOVE THE REAL MODEL on held-out docs 0–63 — LOWER IS BETTER.
Everything else is an energy (mean ‖·‖² per token, in units where ‖x₀‖² = D = 1152 exactly because x₀ is rms-normed).

Construction: one collect pass over fit docs 96–191 (256 tokens each) recording E‖pre(l)‖² (residual after block l's MLP write,
before block l+1's blend), E‖post(l)‖² (after block l's blend = attn_l's input), E‖aw_l‖², E‖mw_l‖² (raw writes, before any
patch). Blend term ratio r(l) = λ₁(l)²·D / (λ₀(l)²·E‖pre(l−1)‖²) for l = 1..7 (embedding term energy over carried-residual term
energy, cross term ignored; the measured E‖post(l)‖² is reported alongside). Instrument arm EARLY22_OWN_768 as §2753.

Arms: EARLY22_OWN_768 (instrument only).

Frozen: this file, §2759 results (block_boundary_blend_rotation_probe_results.json), checkpoint, fit_natural.pt.

- pred_a_instrument: baseline 3.0322401 within 1e-4; EARLY22_OWN_768 within .02 of .057.
- pred_b_block1_blend_is_embedding_dominated: r(1) ≥ 1 (block 1's input energy comes mostly from 8·x₀, not from .0127·pre(0)).
  Null: r(1) ≤ 0.25.
- pred_c_blend_is_minor_from_block_2: median over l = 2..7 of r(l) ≤ 0.25. Null: ≥ 1.
- pred_d_mlp_writes_carry_more_energy: median over l = 0..7 of E‖mw_l‖² / E‖aw_l‖² ≥ 2. Null: ≤ 1.
- pred_e_mlp0_is_the_largest_early_write: E‖mw_0‖² ≥ 3 × the largest of the other 15 early writes (aw_0..7, mw_1..7). Null: ≤ 1.2 ×.

Price: 1 stats pass + 1 energy pass (96 docs each) + baseline + 1 arm × 64 docs = 320 GPU document-forwards (≈ 20 s). Descriptive;
nothing installs into the §312 frontier.
