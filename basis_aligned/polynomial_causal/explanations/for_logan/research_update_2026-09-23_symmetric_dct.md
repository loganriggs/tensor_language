# Symmetric, deterministic DCT on bilin18: the interaction forms are attention-carried, high-rank and context-specific (23 September 2026, 02:50 UTC)

Claude (Fable). Lane started on Logan's direction (23 Sep, ~01:10 UTC): pursue the improvements from the PR-DCT checks on the squared-attention bilinear model until they are exhausted, report good and bad, and carry controls so a null cannot be failed code. Project `pr_dct_checks/` (plans in `plans/`, receipts in `results/`). Each rung is preregistered; predictions are scored as written.

## The idea being tested

The PR-DCT checks (previous note) showed that the DCT score u·H[l, r] is symmetric in l ↔ r, so the object a DCT fits is a symmetric bilinear form B_u in the source residual, and that the fixed-point fit is init-dominated. The proposal: fix u to meaningful output directions, compute B_u exactly instead of fitting it, describe it by its rank in its own eigenbasis (a sparse description that need not be neuron-localised), test whether that description generalises, and score mediators by completeness with attention included.

**Setup.** bilin18 (`Elriggs/gpt2-bilinear-sqrd-attn-18l-9h-1152embd`). Perturbation θ added to the residual entering block 8 at every position; output = the final rms-normalised residual (what the unembedding reads) averaged over the last 3 positions, so the span is blocks 8–17 plus the final norm. u_k = the 16 root reader directions from the compression lane. B_{c,k} = the full 1152 × 1152 Hessian of u_k·Δ_c at θ = 0, per context (forward-over-reverse, 1152 tangent passes each); B̄_k = its mean over 32 FineWeb fitting contexts; held-out: 16 FineWeb contexts and 16 AdvBench prompts.

## Rung 1 — exact forms for the 16 readers (2 of 5 preregistered predictions passed)

**Controls (pass).** Hessian entries against the handoff's jvp-of-jvp mixed Hessian for random pairs: relative error 1.5e-6 median, 1.3e-4 max; symmetry 1.9e-6; with every MLP and attention block frozen and the final norm removed the score is exactly 0 (the residual pass-through is linear); the final norm alone contributes −0.2%.

| prediction | bar | result |
|---|---|---|
| b: B̄_k low-rank (median eigenvalue participation ratio) | ≤ 32 | **✗ 92–408** (median 270) |
| c: top-8 eigen-directions of B̄_k capture held-out FineWeb Frobenius energy | ≥ 0.5 | **✗ 0.039** (per reader 0.014–0.081) |
| d: the same on AdvBench | ≥ 0.3 | **✗ 0.001** |
| e: top eigenvector's interaction is attention-carried (all-attention completeness) | ≥ 0.9 | ✓ 0.91 (0.61–0.96) |

What the numbers say:

1. **There is no context-independent low-rank interaction form in the block-8 residual basis for these readers.** The mean form is spread over hundreds of directions, and its leading directions carry ~4% of any single context's interaction energy on the same distribution and ~0.1% off it. The leading direction is real but weak: its sign is consistent in 94–100% of held-out contexts, and the fixed top-8 directions score 27% of what each context's own top-8 directions score.
2. **Whatever interaction there is runs through attention in blocks 8 and 9.** Freezing all attention removes 91% (median); block 8's attention alone 32%, block 9's 33%, blocks 10–17 ≤ 9% each. The source-block MLP carries 8%. The top single MLP unit carries 2%, the top 20 units 31% (random 5: 0.00); freezing everything gives exactly 1.00.
3. The interaction energies are tiny in absolute terms (|u·H[v₁, v₁]| ~ 2–8 × 10⁻⁶ for unit-norm directions against a residual of norm ~10⁴), i.e. the quadratic response to a unit perturbation at block 8 is far below anything the model does at scale — the same caveat as for all DCT quantities.

So the "sparse in an alternative basis" hope fails at the level of the averaged form: the interaction geometry at block 8 is set per context by the attention patterns. Rung 2 (running) asks the two questions that remain: are the forms low-rank *per context*, and which heads carry them.

**Receipts.** `results/symmetric_dct_v1.json` (+ `_forms.pt`: B̄_k, eigenvalues, top-64 eigenvectors), plan `plans/SYMMETRIC_DCT_PLAN_V1.md`, runner `scripts/run_symmetric_dct_v1.py`; 74 min on the GPU.

## Rung 2 (partial, 03:10 UTC) — per-context rank and cross-context consistency

The per-head block is still running (14 minutes per context); the rank block is in. Per context the forms are still high-rank (eigenvalue PR median 113, range 7–264; a context's own top-8 eigen-directions capture 20% of its energy), and **25–33% of each context's interaction energy is shared across contexts** (consistency ratio ‖B̄‖²/mean‖B_c‖², all 16 readers in that band; mean pairwise cosine between contexts' forms 0.21–0.28). So there is a shared component, but it is spread over many directions — which is why rung 1's top-8 truncation of the mean form saw only 4% of it. First two contexts of the per-head block: freezing the top-3 heads removes 56–70% of the per-context top direction.

## Rung 3 — a fixed weight-space head-form dictionary does not describe the forms (1 of 5)

**The test.** For a squared-attention head the pattern is (q_i·k_j)(q₂ᵢ·k₂ⱼ), and the second derivative of one factor with respect to a perturbation added at every position is, up to the QK-norm Jacobians, a *fixed* matrix: sym(W_qᵀ R(−δ) W_k) for relative offset δ (rotary convention verified numerically). Dictionary: all 90 heads × 2 pattern factors × 32 offsets = 5,760 fixed forms; each context's exact B_{c,k} is least-squares fitted to it (inner products and the Gram computed exactly in head space, never materialising the forms). The "sparse in an alternative basis" hypothesis in its cleanest form: context enters only through coefficients.

**Controls.** Inner-product and Gram shortcuts exact to 1e-15 against materialised forms; a planted combination of three forms is recovered with R² = 1.000000 (its coefficients are not identifiable because adjacent offsets at low rotary frequencies are near-collinear — reported, not gated, amended before the run); random symmetric matrices of the same norm get R² = 0.004. Post-hoc physics control on the real model: a single head's own pattern factor with random pair weights, whose Hessian should lie in that head's 32 forms by construction, is captured at **95–99.9% without QK-norm and 69–95% with it** (heads 8.0 and 8.3, both factors) — so the dictionary is the right object and the QK-norm Jacobian costs at most ~30%.

| prediction | bar | result |
|---|---|---|
| b: captured fraction R², full dictionary | ≥ 0.5 | **✗ 0.026** median (0.013–0.124; noise 0.004) |
| c: blocks 8–9 forms alone, share of the full R² | ≥ 0.8 | ✗ 0.68 |
| d: offset-resolved vs offset-free (δ = 0 only) | ≥ 1.5× | ✗ 1.49× (0.026 vs 0.018) |
| e: top-50 forms re-solved, share of full R² | ≥ 0.8 | ✗ 0.25 |

**What it means.** The interaction forms are *not* combinations of the heads' pattern-factor curvatures, even with free context-dependent coefficients: those pieces account for ~2–3% of the energy, barely above noise, although the same pieces capture a single factor's curvature almost perfectly. So the interaction created by attention lives in the terms the fixed dictionary cannot contain: the **pattern × pattern cross term** (the product of the two factors' *gradients*, 2·sym(∇a_ij ∇b_ijᵀ), a context-dependent rank-1 piece per position pair), the **pattern × value** cross term, the value path, and downstream curvature. The next rung measures that budget exactly by freezing mechanisms inside each head (one factor, both factors, the pattern, the values) and computing the Hessian under each, which needs no derivation.

**Receipts.** `results/symmetric_dct_v3.json`, plan `plans/SYMMETRIC_DCT_PLAN_V3.md`, runner `scripts/run_symmetric_dct_v3.py` (5.5 min on the GPU after a memory fix).

## Rung 4 — the exact mechanism budget: attention transports, the curvature is downstream (1 of 5 as written, but decisive)

**Method.** Inside every squared-attention head the perturbation reaches the output through the first pattern factor a = q·k, the second b = q₂·k₂, the values v, and, outside attention, the source MLP and all downstream blocks. Holding a piece at its clean value for derivatives at θ = 0 is a `.detach()`; second-order terms each involve one or two pieces, so inclusion–exclusion over six Hessians per (context, reader) splits H exactly into pure(a), pure(b), cross(a, b), pure(v), cross(pattern, v) and rest. Controls: freezing pattern + values by this mechanism equals freezing the blocks' attention outputs (0.00e+00), freezing a + b equals freezing the pattern (0.00e+00), symmetry 2e-6.

| piece (median over 16 contexts × 16 readers) | signed share ⟨piece, H⟩/‖H‖² | energy ‖piece‖²/‖H‖² | eigenvalue PR |
|---|---|---|---|
| pure(a) — first pattern factor | 0.044 | 0.040 | 96 |
| pure(b) — second pattern factor | 0.044 | 0.031 | 85 |
| cross(a, b) — product of the factors' gradients | 0.007 | 0.015 | 39 |
| **pure(v) — values** | **0.342** | 0.281 | 78 |
| **cross(pattern, v)** | **0.186** | 0.186 | 21 |
| **rest — no attention piece live (source MLP, MLP paths, downstream curvature)** | **0.350** | 0.296 | 299 |

Per reader the three large shares are stable (pure(v) 0.28–0.37, cross(pattern, v) 0.16–0.21, rest 0.31–0.40). Predictions b (pattern path ≥ 0.5: 0.10), c (cross(a, b) dominant), d (fixed dictionary R² on the pure-factor pieces ≥ 0.6: 0.13, because a pure-factor piece summed over all heads still includes downstream curvature acting on the first-order pattern change) and e (cross(a, b) low-rank: PR 39) all fail as written.

**What it means.** The attention patterns are not where the second-order interaction is made. Values are linear in a block's input, so a piece that involves only values has its curvature *after* the attention: the perturbation is transported linearly across positions by the clean attention (the value path), and the bilinear MLPs downstream (and the final norm) turn the transported first-order change into a second-order one; the same holds for the 35% that never touches an attention piece. That is the structure this architecture makes natural — attention moves, bilinear MLPs multiply — and it explains every earlier negative: the interaction form in the block-8 basis is a fixed MLP form pulled back through a context-dependent linear transport, so it is high-rank, context-specific, and not in any fixed dictionary of pattern forms. Rung 5 (running) locates the two ends: which blocks' MLPs carry the curvature and which blocks' values carry the transport, for each context's top interaction direction.

**Receipts.** `results/symmetric_dct_v4.json`, plan `plans/SYMMETRIC_DCT_PLAN_V4.md`, runner `scripts/run_symmetric_dct_v4.py`; 25 min.
