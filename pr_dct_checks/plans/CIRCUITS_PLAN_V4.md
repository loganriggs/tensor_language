# Circuits lane, rung C1c: the zero-free-parameter decomposition — unit forms pulled back through the exact linearised transport

24 September 2026, 02:10 UTC. Claude (Fable). Registered before running. C1b showed the unit set is right but a fixed matrix per head is not the transport. This rung uses the exact transport and fits nothing.

**Construction.** For context c, block L, position i (the last 3), the normalised MLP input n_{L,i}(θ) is a differentiable function of the block-8 perturbation θ through everything before block L (attention values and patterns, intermediate MLPs, norms). For a unit (a, b) at block L, g_a = ∇_θ (a·n_{L,i}) and g_b = ∇_θ (b·n_{L,i}) at θ = 0 are its read directions pulled back to block 8 by the exact linearised transport (one vjp each). Its hidden activation h = (a·n)(b·n) has second derivative 2·sym(g_a g_bᵀ) plus terms with the transport's own curvature (excluded by design). Its downstream weight for direction u is w_{u,L,i} = ∂(u·out)/∂h_{L,i} at the clean point (one vjp per (u, L, i) gives it for all 4,608 units of the block at once). The predicted form is

  B̂_{c,u} = Σ_{(L,h,i) ∈ S} w_{u,L,h,i} · 2·sym(g_{a,h} g_{b,h}ᵀ),

with S = C1a's 314 stable units × 3 positions and **no fitted coefficients**. Reported per (c, u): R²_closed = 1 − ‖B − B̂‖²/‖B‖² against the exact Hessian; the same with all 46,080 units of blocks 8–17 (S = everything; the exact "unit curvature through linearised transport" part of the form, which isolates what is missing: transport curvature and the final norm); a least-squares refit of the 942 stable-unit coefficients (upper bound for that set); and the per-unit energy attribution ⟨w·2sym(g gᵀ), B⟩/‖B‖² summed over positions (signed), which is the circuit's weight on each unit.

**Data.** 8 held-out FineWeb contexts × 32 directions (16 readers, 16 token directions), exact Hessians as in C1b.

**Controls.** (a) On the smoke model and on the real model, for a unit-and-position readout f(θ) = h_{L,i}(θ) (a scalar), the closed form 2·sym(g_a g_bᵀ) + (a·n)·H_b + (b·n)·H_a equals its exact Hessian (≤ 1e-4 relative; H_a = Hessian of a·n), and 2·sym(g_a g_bᵀ) alone captures ≥ 0.8 of it for the source-block units (where the transport is the identity plus block-8 attention). (b) With S = all units and the prediction *including* the transport-curvature terms for the source block only, R² must not decrease. (c) Random unit sets of size 314 with exact weights give the noise-level R² (reported).

**Predictions (scored as written; failures preserved).**
1. pred_a_instrument: (a) passes.
2. pred_b_all_units: median R²_closed with all 46,080 units ≥ 0.6 — the interaction is mostly unit curvature acting on the linearly transported perturbation. Prior: likely (rung 5: all MLPs 0.98 by removal, but removal counts transport paths too).
3. pred_c_stable_set: median R²_closed with the 314 stable units ≥ 0.4. Prior: unsure.
4. pred_d_refit_bound: the least-squares refit on the stable set exceeds the closed form by ≤ 0.15 (the exact weights are nearly the best coefficients). Prior: unsure.
5. pred_e_core_units: the 39 units stable for ≥ 50% of C1a's directions carry ≥ 0.5 of the stable set's attributed energy (median over (c, u)). Prior: unsure.

**Price.** Per context: 2 × 46,080 × 3 pull-back vjps for the all-unit prediction is too many — instead the all-unit prediction is computed block-wise through the Jacobian: for block L and position i, J_{L,i} = ∂n_{L,i}/∂θ (1152 × 1152, by 1152 vjps... also too many). Use the identity B̂ = Σ_L Σ_i J_{L,i}ᵀ [Σ_h w_h 2 sym(a_h b_hᵀ)] J_{L,i} = Σ_{L,i} J_{L,i}ᵀ M_{u,L,i} J_{L,i} with M_{u,L,i} = Lᵀ diag(w) R + Rᵀ diag(w) L (a fixed 1152 × 1152 unit-weighted form of the block's MLP), and J_{L,i}ᵀ X J_{L,i} computed by 2 × 1152 vjp/jvp passes per (L, i): 30 (L, i) pairs × 2,304 passes ≈ 70k passes per context ≈ 15 min. 8 contexts ≈ 2 h. The stable-set prediction is cheap (1,884 vjps per context).

**What follows.** If pred_b passes, the circuit description is: fixed MLP forms M_{u,L,i} (weights × the direction's downstream read) pulled back through the context's linearised transport J_{L,i}, and C3 tests the four properties on the stable-unit version of it. If pred_b fails, the remaining share is transport curvature (pattern and intermediate-MLP second order) and the description needs the second-order transport, which we would state and stop.
