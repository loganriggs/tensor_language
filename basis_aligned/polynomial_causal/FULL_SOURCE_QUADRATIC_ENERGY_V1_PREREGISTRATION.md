# Full centered-unembedding source-coherence energy

Extend the four-form result to the entire centered U→MLP17→O17→mixed-value
coefficient tensor, with no output-rank truncation. Same nine heads and full
2304-dimensional shared source tuple; raw native value mix, including its sign.
This is still a formal coefficient norm, not natural-text weighting.

Use K=D^T(U_centered^T U_centered)D in native4608-product coordinates. Contract
the separately symmetric and double-antisymmetric head/source tensor norms
exactly. The trace of each cross-head source matrix squared supplies its partial
transpose inner product. Dense all-output toy and common/independent source
permutation controls pass. No explicit50304×20736×20736 tensor is constructed.

Native W_h and the same independently signed/permuted head source coordinates
as the four-form screen (CPU seed33417, consuming the common permutation first).
Independent permutations preserve each W_hW_h^T and therefore total lifted
energy/head-diagonal energy; the symmetry split may change. No outcome-guided
control selection or extra seeds in this run.

- A: dense combined-output controls held; finite/nonnegative component energies,
  symmetric+antisymmetric closure, and independent-permutation total and diagonal
  invariance, normalized errors<=1e-10.
- B: A plus native antisymmetric fraction>=0.10. This removes the four-form
  sampling limitation, not the distinction between one and multiple sources.
- C: A plus absolute native-minus-permuted antisymmetric fraction difference
  >=0.05. A failed C does not support large special learned alignment in this norm.

FP64 managed GPU,900-second alarm, zero body forwards, output only compact JSON.
Roughly6GB of Gram intermediates; all native weights and unexplained routing,
value source states, residual terms and normalization remain charged. If B holds
and C fails, treat the large one-source vanishing component as generic geometry
until its live multi-source computation is identified. Do not turn a coefficient
kernel into a legal whole-attention deletion.
