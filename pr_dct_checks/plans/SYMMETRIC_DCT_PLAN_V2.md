# Symmetric DCT rung 2: per-context rank, cross-context consistency, and per-head completeness

23 September 2026, 01:50 UTC. Claude (Fable). Registered before running. Rung 1 (SYMMETRIC_DCT_PLAN_V1) found the *mean* interaction forms B̄_u high-rank (eigenvalue PR 92–408 of 1152) and their top-8 eigen-directions capturing only 3–4% of the per-context interaction energy on held-out text, so a context-independent low-rank description in the block-8 residual basis does not exist for these readers. This rung asks whether the forms are low-rank *per context*, how much of the interaction energy is shared across contexts at all, and which attention heads carry it.

**Same model, span, readers, contexts** as rung 1 (bilin18, θ at block 8, output = final normalised residual at the last 3 positions, the 16 root readers; 16 held-out FineWeb contexts). B̄_u is loaded from rung 1's archive (`results/symmetric_dct_v1_forms.pt`).

**Computation (no fitting).** For each held-out context c and reader k: the full Hessian B_{c,k}; its eigenvalue PR; the Frobenius fraction in its own top-8 eigenvectors; the cosine between B_{c,k} and B̄_k and between pairs of contexts (Frobenius inner products); the consistency ratio ‖B̄_k‖²_F / mean_c ‖B_{c,k}‖²_F. Per-head completeness (new: per-head freezing of the squared attention in `circuit_checks/heads.py`, exact replica of the module's forward) for two directions per (c, k): v̄₁ = top eigenvector of B̄_k and v₁ᶜ = top eigenvector of B_{c,k}: fraction of vᵀB v removed by freezing each of the 90 heads (blocks 8–17 × 9) alone, the top-3 heads together (ranked by their single removals), and each block's attention.

**Controls.** (a) Per-head replica parity against the module forward on the real model ≤ 1e-5 relative; freezing all 9 heads of a block equals freezing the block's attention (≤ 1e-4 relative); Hessian symmetry ≤ 1e-3. (b) CPU smoke test on a tiny random squared-attention model. (c) The consistency ratio is bounded above by 1 and equals 1 when all contexts share the form (checked on planted identical forms in the smoke test).

**Predictions (scored as written; failures preserved).**
1. pred_a_instrument: controls (a) pass.
2. pred_b_low_rank_per_context: median over (c, k) of the per-context eigenvalue PR ≤ 32. Prior: unsure.
3. pred_c_shared_energy: median over readers of the consistency ratio ≥ 0.25. Prior: unlikely (rung 1).
4. pred_d_few_heads: for the per-context top eigenvector v₁ᶜ, freezing the top-3 heads removes ≥ 0.5 of the interaction (median over (c, k)). Prior: unsure.
5. pred_e_same_head: for each reader, the single most-removing head for v₁ᶜ is the same head in ≥ 50% of the 16 contexts (median over readers). Prior: unsure.

**Price.** 16 Hessians (≈ 5 min) + 16 × 16 × 2 × (90 + 10 + 2) ≈ 52k jvp-of-jvp passes (≈ 40 min).

**Not licensed.** No head names as circuits; rung 3 (polynomialised attention) decides whether the head-carried interaction is describable in weight space.
