# Equal-token varimax comparison

Follow the completed raw-varimax experiment without changing its rank128
centered tensor projection or introducing data. The raw solution is far from
stationary for the equal-row criterion (relative gradient0.613). This makes
the weighting assumption worth testing; it does not predict better structure.

Normalize each token's128-dimensional loading row to unit Euclidean norm before
calling the unchanged controlled varimax optimizer. Tokens now have equal total
loading energy in the rotation objective. Apply the returned rotation to the
original unnormalized loadings and inverse-rotate the original quadratic forms,
so the actual projected tensor is unchanged. All50304weight rows; no token
frequency, labels, natural states or nonnegativity. Same identity initialization,
240-second cap and original convergence rules. Raw run took62.24seconds.

Reuse frozen OUTPUT_VARIMAX_V1_CHECKPOINT.pt factors, source hash and instrument
receipt. Save only rotation/CG state with exact factor-source reference; the
original factor checkpoint stays required. No extra dense loading serialization.
The same opaque native input readers, U and remainder remain charged.

- A: source bindings and original instrument held; unit-row norm, tensor rotation
  replay, raw loading energy, frame orthogonality and monotonicity errors<=1e-10.
- B: A plus unchanged relative-gradient<=1e-5 and five-accepted-step plateau<=1e-7.
- C: A plus median per-token participation at least25%lower than the initial
  spectral frame and mean per-token top4 energy fraction>=0.50. Report aggregate
  raw-energy top4 fraction as well. The equal-token50%bar differs explicitly
  from raw-energy weighting in V1; do not rewrite V1's failed bar.

Also report identical raw/equal-row metrics for both learned rotations, without
selecting a winner on incompatible metrics. No independent-restart stability
claim. If sparsity stays insufficient, retain orthogonal-function and fixed
subspace limitations, then prioritize coupled input/output structure rather
than endless weighting variants. Four-property circuit evidence remains absent.
