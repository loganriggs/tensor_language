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
