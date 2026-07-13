# Handoff: Tensor-Similarity-Regularized Bilinear Transcoders

Context for a coding session with GPU. Goal: prototype a sparse bilinear transcoder
whose faithfulness to the original layer is enforced in *weight space* via the
closed-form Gaussian tensor inner product from the tensor similarity paper
(arXiv:2605.15183, code at https://github.com/tdooms/tensor-similarity), rather
than only via on-distribution MSE.

## Core idea

- Original bilinear layer: y = D((L x̃) ⊙ (R x̃)), with x̃ = (1, x) ∈ R^{d+1},
  L, R ∈ R^{r×(d+1)}, D ∈ R^{K×r}. Weight tensor A_{kij} = Σ_h D_{kh} L_{hi} R_{hj}.
- Transcoder: same functional form with overcomplete rank r' ≫ r:
  ŷ = D'((L' x̃) ⊙ (R' x̃)), tensor Â. Same CP decomposition class, so all
  tensor-sim machinery applies directly.
- Division of labor: a data-dependent **sparsity term** (TopK/BatchTopK on hidden
  activations h = (L'x̃) ⊙ (R'x̃)) selects *which* gauge/parametrization we land in;
  a data-free **weight-space fidelity term** guarantees the global polynomial is
  preserved (including off-distribution mechanisms, e.g. backdoors).

## Fidelity loss (closed form, n = 2)

Gaussian metric via Isserlis: E[x̃_i x̃_j x̃_k x̃_l] = δ_ij δ_kl + δ_ik δ_jl + δ_il δ_jk.
Per output slice k:

    ⟨A_k | Λ | Â_k⟩ = tr(A_k) · tr(Â_k) + 2 ⟨A_k^sym, Â_k^sym⟩

with tr(A_k) = Σ_h D_{kh} (l_h · r_h). The symmetric inner product is computed via
the E∥ + E× four-matrix-product identity (paper Appendix A.2, Eq. 12–14) with
G = I: needs L'L^T, R'R^T, L'R^T, R'L^T, each r'×r, cost O(r r' d). Fully
differentiable, zero sampling variance, computable every step.

**Train on Λ-weighted relative distance, not cosine:**

    L_fid = ‖A − Â‖²_Λ / ‖A‖²_Λ
          = (⟨A|Λ|A⟩ − 2⟨A|Λ|Â⟩ + ⟨Â|Λ|Â⟩) / ⟨A|Λ|A⟩

Cosine is scale-invariant (right for *measuring* equivalence, wrong for a drop-in
replacement). Report cosine sim as an eval metric.

**Data-matched metric (do this, residual streams are not N(0, I)):** Isserlis holds
for N(0, Σ); each pairing contributes Σ_{ab} instead of δ_{ab}. Estimate empirical
covariance Σ of the lifted inputs once, then plug into the Gram recursion
(G^(0) = Σ instead of I). Still closed form, still data-free per step.

## Total loss

    L = MSE(ŷ_topk, y)  +  λ · L_fid  (+ optional aux losses per standard SAE practice)

where ŷ_topk uses the TopK-masked hidden activations, and L_fid is computed on the
**dense** transcoder tensor Â (no mask — the mask is not part of the multilinear model).

## Two sparsity regimes to compare (comparison is itself a finding)

1. **TopK as training regularizer only.** Deploy/analyze the dense transcoder;
   sparsity is a property of activations on-distribution. Check at end of training
   that h_topk ≈ h on-distribution (mask ≈ no-op). Tensor sim well-defined for the
   dense model.
2. **Structural weight sparsity.** Constraints on L', R', D' themselves (block
   structure à la BSF; L1/L0 on factor rows; fixed sparse support). Stays strictly
   inside the tensor network class; sparsity survives OOD. Mechanistic rather than
   distributional claim.

Hypothesis: (1) wins the sparsity/fidelity Pareto, (2) wins guarantees; the gap
measures how distribution-specific the natural sparse decomposition is.

## Experiments, in order

### E1. Synthetic recovery (do first — settles feasibility)
Generate a bilinear layer with known sparse ground-truth CP structure, apply a
random rotation/gauge transformation to hide it. Train transcoders with
(a) MSE+TopK only, (b) L_fid only, (c) both. Success = both terms together recover
ground-truth factors (up to permutation/scaling); neither alone does.
Also answers: can sim = 1 coexist with sparsity at achievable overcompleteness r'?
(Sparse-decomposition rank may exceed symmetric rank of A^sym.)

### E2. SVHN backdoor (flagship)
Reuse the paper's backdoor setup (Section 3.3: diamond trigger → class 9, 10%
poison rate; model/training in Appendix B, Table 1). Train two transcoders on the
backdoored model **using clean data only**:
- T_mse: MSE + TopK
- T_sim: MSE + TopK + λ·L_fid
Evals: (a) attack success rate with transcoder spliced into the model;
(b) digit-9 tensor slice similarity of transcoder vs. original.
Prediction: T_mse silently drops the backdoor (never fires on clean data), T_sim
preserves it. This is the transcoder version of the paper's Figure 1 story.

### E3. Pareto frontier
Sweep λ and K (TopK). Plot L0 vs. tensor sim vs. MSE. Include MSE-only transcoders
as baseline curve and plot their *measured* tensor sim (expected: low even at
good MSE).

### E4. Feature interpretability
Each feature h is a quadratic form (l'_h · x̃)(r'_h · x̃) — itself a tiny symmetric
tensor. Eigendecompose per Pearce et al. 2025. Note the lifted coordinate means
the dictionary contains linear features (quadratic form dominated by the
1-coordinate) and genuinely quadratic features in one family. Optional: regularize
toward l'_h ≈ r'_h ("squared readout") or toward linear structure, connecting to
the linear-approximation idea for intermediate-similarity dimensions.

## Implementation notes

- Reference implementation of Gram recursion / E∥+E× uses quimb, but for n=2 it's
  four matmuls + two Hadamards + one matmul; just write it in raw PyTorch.
- Precompute ⟨A|Λ|A⟩ once (constant).
- For the Σ-metric: lift inputs before estimating covariance (the 1-coordinate
  gives E[x̃] cross-terms, i.e. the mean, for free).
- Numerical hygiene: normalize A by ‖A‖_Λ at load; watch for L_fid gradient scale
  vs. MSE at large r' (λ warmup if needed).
- Sanity check: L_fid(A, A) = 0 and cosine sim(A, gauge-transformed A) = 1 after
  symmetrization (insert U, U^{-1} on the hidden index; permute + rescale rows of
  L', R' with compensating D').

## Open questions

- Does the TopK-trained dense transcoder actually stay sparse without the mask (E1/E3)?
- Best λ schedule: constant vs. anneal MSE→sim vs. constrained optimization
  (sim ≥ threshold as constraint)?
- Structural constraint that provably keeps the transcoder in the tree-decomposition
  class while giving per-feature sparsity — block CP seems safe; verify BatchTopK
  variants don't leak into the analyzed object.
