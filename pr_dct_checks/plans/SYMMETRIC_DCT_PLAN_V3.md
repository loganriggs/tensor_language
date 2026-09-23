# Symmetric DCT rung 3: is each context's interaction form a combination of fixed weight-space head forms?

23 September 2026, 03:20 UTC. Claude (Fable). Registered before running. Rungs 1–2 found the interaction forms B_{c,k} (block-8 residual → reader k, per context c) high-rank both averaged (eigen-PR 92–408) and per context (median 113), with 25–33% of each context's energy shared across contexts, and carried by attention in blocks 8–9 (91%; per-head block of rung 2 still running). This rung asks whether the context dependence is only in *coefficients* on a fixed, weight-defined dictionary — the "sparse in an alternative basis" reading Logan asked for.

**Dictionary.** For a squared-attention head (block L, head h) the pattern is p_ij = (q_i·k_j)(q₂ᵢ·k₂ⱼ)/D² with q_i·k_j = (W_q n_i)ᵀ R(j−i) (W_k n_j) after QK-norm and rotary (convention verified numerically: R(i)ᵀR(j) = R(j−i)). A perturbation θ added to every position enters both factors, so the part of ∂²p_ij/∂θ² that comes from one pattern factor is a fixed matrix times context-dependent scalars: F_{L,h,δ}^{qk} = sym(W_qᵀ R(−δ) W_k) and F_{L,h,δ}^{q₂k₂} = sym(W_q₂ᵀ R(−δ) W_k₂) for offset δ = i − j ∈ [0, 31]. The dictionary is all 90 heads × 2 pattern types × 32 offsets = 5,760 forms (QK-norm Jacobians, the block-8 pass-through to block 9, the pattern × pattern and pattern × value cross terms, and downstream curvature are *not* in it; those are what a low captured fraction would point to).

**Computation.** 16 held-out FineWeb contexts (same as rungs 1–2), the 16 readers: B_{c,k} recomputed (16 Hessians). Inner products ⟨F, B⟩ and the dictionary Gram are computed exactly through 128 × 128 head-space matrices (never materialising the forms): ⟨sym(AᵀRC), B⟩ = ⟨R, A B Cᵀ⟩, and ⟨sym(A_aᵀR_aC_a), sym(A_bᵀR_bC_b)⟩ from the four 128 × 128 cross-Grams of the weights. Ridge least squares (λ = 1e-6 · mean diag) gives α_{c,k}; captured fraction R² = αᵀb / ‖B‖²_F. Sub-dictionaries by index subset: blocks 8–9 only (1,152 forms); offset δ = 0 only (180 forms); top-k forms by |α|·‖F‖ re-solved (k = 10, 50, 200). Stability: Pearson correlation of α across contexts per reader.

**Controls.** (a) For 6 random forms, the head-space inner-product shortcut matches the explicit Frobenius inner product with a materialised form to ≤ 1e-4 relative; the same for 6 random Gram entries. (b) A planted B = Σ of 3 dictionary forms with known coefficients is recovered with R² ≥ 0.999 and the coefficients to ≤ 1e-3. (c) Noise baseline: R² of 8 random symmetric matrices with the norm of a typical B_{c,k} (expected ≈ 5,760 / 663,552 ≈ 0.009). (d) CPU smoke test on a tiny random squared-attention model.

**Predictions (scored as written; failures preserved).**
1. pred_a_instrument: (a), (b) pass and the noise baseline is ≤ 0.02.
2. pred_b_captured: median over (c, k) of R² with the full dictionary ≥ 0.5. Prior: unsure — the cross terms are excluded.
3. pred_c_blocks_8_9: R² with blocks 8–9 only ≥ 0.8 × the full-dictionary R² (median of the ratio). Prior: likely.
4. pred_d_offsets_matter: R² with the offset-resolved dictionary ≥ 1.5 × R² with δ = 0 only (median ratio). Prior: unsure.
5. pred_e_sparse: the top-50 forms re-solved reach ≥ 0.8 × the full R² (median ratio). Prior: unsure.

**Price.** 16 Hessians (≈ 5 min); Gram of 5,760 forms via 32k head-type pairs × 1,024 offset pairs of 128 × 128 contractions; 256 right-hand sides. ≤ 30 min.

**Not licensed.** A high R² licenses a structured dictionary rung (adding the pattern × pattern and pattern × value cross forms with clean-forward coefficients, which would make the expansion exact for the attention blocks); a low R² with blocks 8–9 carrying the completeness says the interaction lives in the cross terms, i.e. in context-dependent rank-1 pieces, and the fixed-basis reading fails.
