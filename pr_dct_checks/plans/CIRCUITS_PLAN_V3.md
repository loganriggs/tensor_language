# Circuits lane, rung C2: the finite-scale check

23 September 2026, 07:25 UTC. Claude (Fable). Registered before running (launch after C1a lands).

**Question.** Do the derivative-level shares (which heads' values transport, which units create curvature) hold when the perturbation is finite, at scales the model operates at?

**Method.** For each of 8 held-out FineWeb contexts and 32 directions (16 readers, 16 token directions), with v = C1a's probe direction (recomputed by power iteration) and ρ = the mean residual norm at block 8: the finite symmetric interaction along v, D(α) = f(2αv) − 2f(αv) + f(0) with f = u·(final normalised residual, last-3 mean), at α = 0.003ρ, 0.01ρ, 0.03ρ. Baseline: D(α)/(2α²) → u·H[v, v] as α → 0 (reported as the ratio). Patched: the same with (i) the values of C1a's top-3 heads for that direction held at their clean values (true clean tensors, not detach), (ii) C1a's top-100 units held at clean, (iii) both, (iv) 3 random heads' values, (v) 100 random units. Fraction removed = 1 − D_patched/D. Compared with the derivative-level fractions from C1a for the same (context, direction).

**Controls.** (a) At α = 0.003ρ, D(α)/(2α²) within 10% of u·H[v, v] for ≥ 80% of pairs (the quadratic regime is reached). (b) Clean patching with an all-zero mask reproduces the unpatched forward exactly (0). (c) Patching all heads' values and all units to clean at finite α reproduces the linear pass-through: D = the final-norm-only term (reported). (d) CPU smoke test.

**Predictions (scored as written; failures preserved).**
1. pred_a_instrument: (a), (b) pass.
2. pred_b_heads_finite: at α = 0.01ρ, the median fraction removed by the top-3 heads' values is ≥ 0.7 × the derivative-level fraction and ≥ 0.3 in absolute terms. Prior: unsure.
3. pred_c_units_finite: at α = 0.01ρ, the median fraction removed by the top-100 units ≥ 0.7 × the derivative-level fraction. Prior: unsure.
4. pred_d_random_null: random heads and random units remove ≤ 0.05 (median absolute) at every scale. Prior: likely.
5. pred_e_scale_stability: the median fractions at 0.003ρ and 0.03ρ are within 0.15 of those at 0.01ρ. Prior: unsure (SwiGLU reversed sign at 5%).

**Price.** 256 pairs × 3 scales × 8 forward evaluations (each 4 forwards) ≈ 25k forwards through 10 blocks ≈ 30 min.
