# Symmetric DCT rung 4: an exact mechanism budget of the interaction forms

23 September 2026, 03:30 UTC. Claude (Fable). Registered before running. Rung 3 showed that the fixed pattern-factor curvatures capture only 2.6% of each context's interaction form although they capture a single factor's own curvature at 70–95%. This rung measures, exactly and without derivation, where the second-order interaction is created.

**Method.** Inside every squared-attention head of the span (blocks 8–17) the perturbation reaches the output through four live pieces: the first pattern factor a = q·k, the second b = q₂·k₂, the values v, and (outside attention) the source MLP and the downstream blocks ("rest"). Holding a piece at its clean value for derivatives at θ = 0 is a `.detach()` (`attn_modes` in `FreezeSpec`; verified on the tiny model: freezing pattern + values equals freezing the block's attention output, and freezing a + b equals freezing the pattern, both exactly). The Hessian of u·Δ is, at second order, a sum of terms each involving one or two live pieces, so inclusion–exclusion over six Hessians per (context, reader) is exact:

- pure(a) = H[v,b frozen] − H[all attention frozen]; pure(b) likewise; pure(v) = H[pattern frozen] − H[all frozen];
- cross(a, b) = H[v frozen] − H[v, a frozen] − H[v, b frozen] + H[all frozen] — the product of the two factors' gradients, 2·sym(∇a ∇bᵀ) per position pair, pushed downstream;
- cross(pattern, v) = H[full] − H[pattern frozen] − H[v frozen] + H[all frozen];
- rest = H[all attention frozen] (source MLP, MLP paths, downstream curvature); the six pieces sum to H[full] identically.

Per (c, k): the signed share ⟨piece, H⟩/‖H‖²_F of each piece (sums to 1), its Frobenius energy ‖piece‖²/‖H‖², its eigenvalue participation ratio, and the rung-3 fixed dictionary's R² on pure(a) + pure(b) and on cross(a, b). 16 held-out FineWeb contexts, 16 readers; the same span and readers as rungs 1–3.

**Controls.** (a) On the real model, H[all attention frozen by modes] equals the Hessian with `attn_layers = all` (≤ 1e-6 relative) and H[a, b frozen] equals H[pattern frozen] (≤ 1e-6). (b) Symmetry of every Hessian ≤ 1e-3. (c) CPU smoke test on the tiny model.

**Predictions (scored as written; failures preserved).**
1. pred_a_instrument: (a), (b) pass.
2. pred_b_pattern_path: median signed share of pure(a) + pure(b) + cross(a, b) ≥ 0.5. Prior: unsure.
3. pred_c_cross_dominates: the median signed share of cross(a, b) exceeds that of each of pure(a), pure(b), pure(v), cross(pattern, v). Prior: unsure.
4. pred_d_pure_is_fixed_form: the rung-3 dictionary's R² on pure(a) + pure(b) ≥ 0.6 (median). Prior: likely.
5. pred_e_cross_low_rank: eigenvalue PR of cross(a, b) ≤ 32 (median). Prior: unsure.

**Price.** 6 Hessians × 16 contexts (≈ 30 min) + dictionary Gram (16 s) + fits.

**What follows.** If cross(a, b) dominates and is low-rank, the interaction has a structured, context-dependent description (per head, the gradient of one pattern factor times the gradient of the other at few position pairs) and the lane ends with that characterisation plus a finite-intervention check; if the budget is spread across pieces with no dominant low-rank part, the lane ends with a plain statement that the block-8 → reader interaction has no compact description in any of the tested bases.
