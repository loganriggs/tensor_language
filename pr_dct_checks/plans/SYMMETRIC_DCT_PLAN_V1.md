# Symmetric, deterministic DCT on bilin18: the interaction form B_u for fixed reader directions (rung 1)

23 September 2026, 01:30 UTC. Claude (Fable). Registered before running. Follows Logan's direction (23 Sep): pursue the improvements from the PR-DCT checks on the squared-attention bilinear model, report good and bad results, include controls so a null result cannot be a code failure.

**Why this rung.** The PR-DCT checks showed (i) u·H[l, r] is symmetric in l ↔ r, so the fitted object is a symmetric bilinear form B_u and the "asymmetric" factors collapse to l = r; (ii) the fixed-point fit is init-dominated (one factor recurs across seeds, seven do not); (iii) on bilin18 every DCT factor is 98–100% attention-carried and PR over MLP units is irrelevant. So: fix u to meaningful output directions, compute B_u exactly instead of fitting it, describe it by its rank in its own eigenbasis (a sparse description that is not neuron-localised), test whether that description generalises, and score mediators by completeness including attention.

**Model and span.** `Elriggs/gpt2-bilinear-sqrd-attn-18l-9h-1152embd` (bilin18) via the repository loader. Perturbation θ added to the residual entering block 8 at every position (AJ's convention); output = the final rms-normalised residual (what the unembedding reads) averaged over the last 3 positions, i.e. span blocks 8–17 plus the final norm.

**Output directions.** The 16 root reader directions of the compression lane (readers = Wᵀ(W w)/‖W w‖² from `EXPANDED_ROOT_EMPIRICAL_V1.pt`'s writer w and the unembedding W), which measure what the unembedding reads along the quartic-branch writer's image. u_k = reader k, unit-normalised.

**Contexts.** FineWeb: 32 fitting documents (rows 0–31 of `fineweb_n192_skip7000`, first 32 tokens), 16 held-out (rows 96–111). AdvBench: 16 prompts (AJ's split, held-out portion) as the off-distribution set.

**Computation (no fitting).** For every context c and reader k, the full Hessian B_{c,k} = ∂²(u_k·Δ_c)/∂θ∂θᵀ at θ = 0 (1152 × 1152, forward-over-reverse, 1152 tangent passes per context). B̄_k = mean over the 32 fitting contexts. Eigendecomposition of B̄_k: eigenvalues λ, participation ratio PR(λ) = (Σλ²)²/Σλ⁴ as the effective rank, top-8 eigenvectors V_8. Generalisation: for each held-out context, the fraction of the Frobenius energy of B_{c,k} captured in the span of V_8, ‖V_8ᵀ B_{c,k} V_8‖²_F / ‖B_{c,k}‖²_F, and the fraction of AJ-style score energy Σ_{v∈V_8}(vᵀB_{c,k}v)² relative to the same with the context's own top-8 eigenvectors. Mediator completeness for the top eigenvector v_1 of each reader (l = r = v_1), on the 16 held-out FineWeb contexts, freezing: all attention (blocks 8–17), each attention block alone, all MLPs (8–17), the source-block MLP, the top-1/5/20 MLP units by mixed derivative across blocks 8–17, 5 random units, and everything (control).

**Controls.** (a) For 8 random (l, r) pairs per reader on one context, lᵀB r from the Hessian agrees with the handoff's jvp-of-jvp mixed Hessian to relative ≤ 1e-3 (float32). (b) Symmetry ‖B − Bᵀ‖_F/‖B‖_F ≤ 1e-3. (c) Freezing every MLP and attention block in the span leaves ≤ 1e-6 of the score (residual pass-through is linear). (d) Smoke test of the whole pipeline on a tiny random squared-attention bilinear model on CPU before the real run.

**Predictions (scored as written; failures preserved).**
1. pred_a_instrument: controls (a)–(c) pass on the real model.
2. pred_b_low_rank: median over readers of PR(λ) ≤ 32 (of 1152). Prior: unsure.
3. pred_c_generalises: median over readers of the held-out FineWeb Frobenius fraction in V_8 ≥ 0.5. Prior: unsure.
4. pred_d_off_distribution: the same fraction on AdvBench ≥ 0.3. Prior: unsure.
5. pred_e_attention_carried: median over readers of the all-attention completeness of v_1 ≥ 0.9 (the PR-DCT finding on this model), with the source-block MLP and all-MLP completeness reported alongside. Prior: likely.

**Price.** 64 contexts × 1152 tangent passes through 10 blocks (T = 32), ≈ 74k forward-over-reverse passes; completeness ≈ 16 readers × 17 groups × 16 contexts ≈ 4.4k jvp-of-jvp passes. Expected ≤ 2 h on the GPU.

**Not licensed by this rung.** No circuit names; no claim about which heads until per-head completeness (rung 2); no polynomialised-attention comparison until rung 3.
