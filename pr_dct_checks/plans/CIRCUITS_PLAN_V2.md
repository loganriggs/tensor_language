# Circuits lane, rung C1b: the weight-space dictionary — transport heads × unit forms

23 September 2026, 07:20 UTC. Claude (Fable). Registered before running (launch after C1a lands; it supplies the head and unit lists).

**The claim to test (Logan, 23 Sep: "is there any way to make this not context-bound? we have a tensor network with weights").** Rung 4 of the symmetric lane located the interaction in the attention value path feeding bilinear-MLP curvature. Values are linear in a position's residual, so head h moves a block-8 perturbation θ into position i as s_i^h·M_h θ with M_h = C_h W_v^h fixed (the head's value-then-output map, rank ≤ 128) and s_i^h the head's attention mass at i (context-dependent scalar); the direct residual path is the identity transport. A bilinear unit (a, b) downstream reading the transported perturbation contributes (a·Δx)(b·Δx), so the exact form is a sum of **fixed** rank-2 forms sym(M_hᵀ a bᵀ M_{h′}) with context-dependent scalar coefficients (products of attention masses and the unit's downstream weight). If this dictionary fits the exact per-context forms, the circuit is a weight-space object and the context only sets coefficients.

**Dictionary.** Transports T ∈ {I} ∪ {M_h : h in the top-6 heads by mean value-share in C1a}. Units: the union of C1a's stable units over all 100 directions, capped at the 600 most reused. Forms: for every unordered transport pair (T, T′) and every unit, F = sym((Tᵀa)(Tᵀ... precisely sym(x yᵀ) with x = Tᵀ a_unit, y = T′ᵀ b_unit, plus the swapped assignment x = T′ᵀ a, y = Tᵀ b when T ≠ T′. At most 28 transport pairs × 600 units ≈ 17k rank-1 symmetric forms; inner products ⟨sym(xyᵀ), B⟩ = xᵀ B y and the Gram ½[(x·x′)(y·y′) + (x·y′)(y·x′)] are cheap; ridge least squares as in rung 3.

**Data.** 8 held-out FineWeb contexts × 32 directions (the 16 readers and the 16 most frequent token directions): exact Hessians B_{c,u} (256 × 17 s ≈ 1.2 h). Sub-dictionaries: identity transport only (the direct path); head transports only; single best head + identity.

**Controls.** (a) Inner-product and Gram shortcuts vs materialised forms ≤ 1e-10 on 6 random forms; a planted combination of 3 forms recovered with R² ≥ 0.999. (b) Noise baseline: R² of 8 random symmetric matrices of the same norm (expected ≈ 17k / 664k ≈ 0.025). (c) Physics control on the real model: a single unit's own curvature through the identity transport (θ added directly to that block's input; the Hessian of a linear readout of that unit's hidden activation) must be captured at ≥ 0.9 by its own two forms. (d) CPU smoke test.

**Predictions (scored as written; failures preserved).**
1. pred_a_instrument: (a)–(c) pass.
2. pred_b_captured: median R² over (c, u) ≥ 0.5 (rung 3's pattern-form dictionary: 0.026). Prior: unsure.
3. pred_c_transport_matters: R² with head transports ≥ 1.5 × R² with the identity transport alone (median ratio). Prior: likely (values carry 81%).
4. pred_d_coefficients_track_attention: for the top-3 heads, the Pearson correlation across contexts between the fitted coefficient mass on that head's forms and the head's summed attention mass at the last 3 positions ≥ 0.5 (median over directions). Prior: unsure.
5. pred_e_tokens: median R² for token directions ≥ 0.8 × that for readers. Prior: unsure.

**Price.** 256 Hessians ≈ 1.2 h; dictionary fits seconds each.
