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

## Rung 5 — where the curvature is made and where the transport happens (5 of 5)

For each context's own top interaction direction v₁ᶜ (top eigenvector of B_{c,k}; 16 contexts × 16 readers), the fraction of u·H[v, v] removed by block-level freezes. Control: with the final norm removed, freezing everything gives exactly 1 (0.00e+00).

| freeze | median fraction removed |
|---|---|
| all MLPs (blocks 8–17) | **0.98** (per reader 0.90–1.01) |
| MLP of block 8 alone / 9 / 10 / 11 / 12 / 13 / 14 / 15 / 16 / 17 | 0.39 / 0.28 / 0.24 / 0.23 / 0.18 / 0.15 / 0.12 / 0.15 / 0.25 / **0.42** |
| two largest single MLP blocks, summed | 0.92 |
| top-20 / top-100 MLP units across blocks 8–17 (by hidden mixed derivative) / 20 random units | 0.27 / 0.53 / 0.00 |
| values of all attention blocks | **0.81** |
| values of block 9 alone / block 8 / 10 / 11 / 12 / 15 / 17 (13, 14, 16 ≤ 0.01) | **0.43** / 0.15 / 0.09 / 0.12 / 0.05 / 0.05 / 0.06 |
| values of blocks 8–9 together (share of all-values) | 0.58 (0.79) |

The most-removing single MLP block is block 17 for 127 of 256 (context, reader) pairs and block 8 for 87 (block 10: 16, block 16: 13, block 9: 12). All five preregistered predictions pass (all-MLPs ≥ 0.6; two blocks ≥ 0.5; blocks 8–9 values ≥ 0.6 of all values; top-100 units ≥ 0.5).

## Closing: what the block-8 → reader interaction is, and what it is not

**What survives.** The second-order interaction between a perturbation of the block-8 residual and the 16 reader directions is, per context: the perturbation is carried **linearly** across positions by attention values — block 9's values above all (0.43), then block 8's (0.15), blocks 10–11 (0.1 each); and the **curvature is made by the bilinear MLPs**: essentially all of it (freezing all MLPs removes 98%), split between the source block's own MLP (block 8, 0.39, the direct path) and the last MLP (block 17, 0.42, the one that writes the readers' directions), with the intermediate MLPs contributing 0.12–0.28 each in an overlapping way. About a hundred MLP units carry half of a given direction's interaction; twenty carry a quarter; twenty random units carry nothing. Rung 4's budget says the same in mechanism terms: the pattern factors' own curvature is ~10% of the total; the value path plus its cross term 53%; paths that never touch an attention piece 35%.

So a circuit description of this interaction that is faithful to the model reads: *reader k ← MLP17 units and MLP8 units (bilinear, weight-fixed forms in their own input space) ← the block-9 and block-8 attention values transporting the block-8 perturbation into those units' inputs (context-dependent linear maps).* It is not localised (a hundred units, several blocks), and it does generalise in the sense that matters — the shares above are medians over 16 held-out FineWeb contexts with tight per-reader spreads, and the same two blocks lead in every reader.

**What fails, plainly.**
- A **fixed basis** for the interaction in the block-8 residual: the mean form is high-rank (PR 92–408), its top-8 directions capture 4% of a held-out context's energy (0.1% on AdvBench), 25–33% of the energy is shared across contexts at all, and a 5,760-form dictionary of the heads' pattern curvatures captures 2.6% (noise 0.4%) although it captures a single factor's curvature at 70–95%. This is now understood: the form is an MLP bilinear form pulled back through a context-dependent transport, so no basis fixed in the block-8 coordinates can hold it.
- **Low rank** per context: eigenvalue PR median 113 (7–264); the top eigen-direction carries 5% of a context's form.
- **Seed-stable fitted factors** (previous note): the DCT and PR-DCT fits find init-dependent single neurons; AJ's branch variant finds single source-MLP neurons by Jacobian ranking, which is the block-8-MLP half of the picture above with the transport half removed by construction.

**What is still owed.** Everything here is second-order at θ = 0 with perturbation energies of 10⁻⁶ against a residual of norm 10⁴. The finite-intervention check at ≤ 1% of the residual norm (planned rung 4 of the original ladder) was not run; the SwiGLU sign reversals in the PR-DCT checks show that derivative-level shares can fail to predict finite effects, so the transport × curvature description should be treated as a derivative-level attribution until patched at finite scale. Rung 2's per-head shares (below) name the transporting heads: 9.8, 9.7 and 8.2.

**Receipts.** `results/symmetric_dct_v5.json`, plan `plans/SYMMETRIC_DCT_PLAN_V5.md`, runner `scripts/run_symmetric_dct_v5.py`; 50 min. All five rungs: `pr_dct_checks/plans/SYMMETRIC_DCT_PLAN_V1–5.md`, `scripts/run_symmetric_dct_v1–5.py`, `results/symmetric_dct_v1–5.json`.

## Rung 2 (complete, 06:20 UTC) — which heads carry the transport (4 of 5)

Per-head freezing (an exact replica of the squared-attention forward with detach-freezing of single heads; parity 0.00e+00, and freezing all nine heads of a block equals freezing the block, 0.00e+00), for each context's own top direction v₁ᶜ and for the mean form's v̄₁, over the 90 heads of blocks 8–17.

| quantity (16 contexts × 16 readers) | median |
|---|---|
| single best head, fraction of u·H[v₁ᶜ, v₁ᶜ] removed | 0.46 |
| top-3 heads together | **0.70** (mean-form direction: 0.77) |
| same best head across contexts, per reader | 0.53 (0.31–0.69) |
| block-level, single attention block: 8 / 9 / 10 / 11 / 12 / 15 / 17 | 0.21 / **0.47** / 0.11 / 0.13 / 0.05 / 0.05 / 0.07 |

The best head is **9.8** for 87 of the 256 (context, reader) pairs, **9.7** for 72 and **8.2** for 50 (11.6: 22, 9.6: 11); in the top-3 sets, 9.8 appears 158 times, 8.2 141, 9.7 127, 11.6 53. These are the same heads whose fixed pattern forms the rung-3 fit reached for first (8.2 and 9.8), so the lanes agree on *who*, and rung 4 says on *what*: their values, not their patterns. Predictions: a ✓, b ✗ (per-context eigenvalue PR 113 ≥ 32), c ✓ (shared energy 0.28 ≥ 0.25), d ✓ (top-3 heads ≥ 0.5), e ✓ (same head in ≥ 50% of contexts). Receipt `results/symmetric_dct_v2.json` (3.75 h; the per-head loop is latency-bound).

**The description, completed.** Reader k's second-order response to a block-8 perturbation = *heads 9.8, 9.7 and 8.2 (and to a lesser degree 11.6) transporting the perturbation through their values into the inputs of the bilinear MLPs of blocks 8 and 17 (and, less, 9–12 and 16), whose fixed weight forms create the curvature.* Three heads carry 70% of the transport for a typical (context, reader); the same head leads in about half the contexts per reader; a hundred MLP units carry half the curvature. That is the circuit this model has for these readers at this order, and it is neither localised to a neuron nor expressible in a fixed basis of the block-8 residual — it is a small set of value paths feeding a large but weight-fixed bilinear map. The ladder is exhausted; the finite-scale intervention check remains the one thing owed.
